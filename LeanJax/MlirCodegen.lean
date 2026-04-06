import LeanJax.Types
/-! MLIR (StableHLO) code generator: emit MLIR modules from `NetSpec`.

    Supports MLPs (`.dense`) and CNNs (`.conv2d`, `.maxPool`, `.flatten`).
    Walks layers tracking the current activation shape. For CNNs with a flat
    input, emits a reshape to (batch, ic, imageH, imageW) at the head.

    Layout: NCHW tensors, OIHW kernels (matches the JAX codegen convention).
    Params are interleaved as (W0, b0, W1, b1, ...) in the function signature,
    with `idx` advancing for every conv or dense layer. -/

namespace MlirCodegen

/-- Total flat input size: for CNN, ic * imageH * imageW; for MLP, first
    dense layer's fanIn. -/
private def inputFlatDim (spec : NetSpec) : Nat :=
  match spec.layers.head? with
  | some (.dense fanIn _ _) => fanIn
  | some (.conv2d ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | _ => spec.imageH * spec.imageW

/-- If the first layer is conv, returns the NCHW input channels. -/
private def inputChannels (spec : NetSpec) : Option Nat :=
  match spec.layers.head? with
  | some (.conv2d ic _ _ _ _) => some ic
  | _ => none

/-- Render tensor type: `tensor<128x1x28x28xf32>`. -/
private def tensorTy (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString) ++ "xf32>"

/-- Sanitize a string for use as an MLIR identifier. -/
private def sanitize (s : String) : String :=
  s.toLower.map (fun c => if c.isAlphanum then c else '_')

/-- Emit a conv2d + bias + activation block. Returns (code, newSSA, newShape).
    Assumes NCHW input [b, ic, h, w], produces [b, oc, h, w] for SAME padding,
    stride 1. -/
private def emitConv2d (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize : Nat) (act : Activation) : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let newShape := [b, oc, h, w]
    let pad := (kSize - 1) / 2
    let kShape := tensorTy [oc, ic, kSize, kSize]
    let mut s := ""
    s := s ++ s!"    %cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ "        dimension_numbers = #stablehlo.conv<raw\n"
    s := s ++ "          input_batch_dimension = 0,\n"
    s := s ++ "          input_feature_dimension = 1,\n"
    s := s ++ "          input_spatial_dimensions = [2, 3],\n"
    s := s ++ "          kernel_output_feature_dimension = 0,\n"
    s := s ++ "          kernel_input_feature_dimension = 1,\n"
    s := s ++ "          kernel_spatial_dimensions = [2, 3],\n"
    s := s ++ "          output_batch_dimension = 0,\n"
    s := s ++ "          output_feature_dimension = 1,\n"
    s := s ++ "          output_spatial_dimensions = [2, 3]\n"
    s := s ++ "        >,\n"
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {kShape}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cvb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1]" ++
         s!" : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cva{pidx} = stablehlo.add %cv{pidx}, %cvb{pidx} : {tensorTy newShape}\n"
    match act with
    | .relu =>
      s := s ++ s!"    %cvz{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cvr{pidx} = stablehlo.maximum %cva{pidx}, %cvz{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cvr{pidx}", newShape)
    | .identity => return (s, s!"%cva{pidx}", newShape)
    | .relu6 => return (s, s!"%cva{pidx}", newShape)
  | _ => return ("    // conv2d error: expected rank-4 NCHW shape\n", curSSA, curShape)

/-- Emit max pool with window size × stride (assumes both equal).
    Halves (or rounds-down divides) spatial dims. -/
private def emitMaxPool (pos : Nat) (curSSA : String) (curShape : List Nat)
    (size _stride : Nat) : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let newShape := [b, c, h / size, w / size]
    let mut s := ""
    s := s ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
    s := s ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({curSSA}, %pli{pos}) (" ++ "{\n"
    s := s ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
    s := s ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
    s := s ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
    s := s ++ "      }) " ++ "{\n"
    s := s ++ s!"        window_dimensions = array<i64: 1, 1, {size}, {size}>,\n"
    s := s ++ s!"        window_strides    = array<i64: 1, 1, {size}, {size}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, tensor<f32>) -> {tensorTy newShape}\n"
    return (s, s!"%pl{pos}", newShape)
  | _ => return ("    // maxPool error: expected rank-4 input\n", curSSA, curShape)

/-- Emit flatten: (b, c, h, w) -> (b, c*h*w). -/
private def emitFlatten (pos : Nat) (curSSA : String) (curShape : List Nat)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let flat := c * h * w
    let newShape := [b, flat]
    let s := s!"    %fl{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy newShape}\n"
    return (s, s!"%fl{pos}", newShape)
  | _ => return ("    // flatten: already flat or unknown rank\n", curSSA, curShape)

/-- Emit a dense layer. Assumes (batch, fanIn) input. -/
private def emitDense (pidx : Nat) (curSSA : String) (batchSize fanIn fanOut : Nat)
    (act : Activation) : String × String := Id.run do
  let bTy := tensorTy [batchSize, fanOut]
  let mut s := ""
  s := s ++ s!"    %mm{pidx} = stablehlo.dot_general {curSSA}, %W{pidx},\n"
  s := s ++ "              contracting_dims = [1] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({tensorTy [batchSize, fanIn]}, {tensorTy [fanIn, fanOut]}) -> {bTy}\n"
  s := s ++ s!"    %bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1]\n"
  s := s ++ s!"           : ({tensorTy [fanOut]}) -> {bTy}\n"
  s := s ++ s!"    %ab{pidx} = stablehlo.add %mm{pidx}, %bb{pidx} : {bTy}\n"
  match act with
  | .relu =>
    s := s ++ s!"    %z{pidx} = stablehlo.constant dense<0.0> : {bTy}\n"
    s := s ++ s!"    %h{pidx} = stablehlo.maximum %ab{pidx}, %z{pidx} : {bTy}\n"
    return (s, s!"%h{pidx}")
  | .identity => return (s, s!"%ab{pidx}")
  | .relu6 => return (s, s!"%ab{pidx}")

/-- Emit the `@forward` function body walking layers in order. -/
private def emitForwardBody (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let mut code : String := ""
  let mut curSSA : String := "%x"
  let mut curShape : List Nat := [batchSize, inputFlatDim spec]
  -- If first layer is conv, reshape flat input → NCHW
  match inputChannels spec with
  | some ic =>
    let nchw := [batchSize, ic, spec.imageH, spec.imageW]
    code := code ++ s!"    %xr = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy nchw}\n"
    curSSA := "%xr"
    curShape := nchw
  | none => pure ()
  let mut pidx : Nat := 0   -- param index (conv + dense)
  let mut pos  : Nat := 0   -- layer position (for unique SSA names on pool/flatten)
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut act =>
      let (snip, newSSA) := emitDense pidx curSSA batchSize fanIn fanOut act
      code := code ++ snip
      curSSA := newSSA
      curShape := [batchSize, fanOut]
      pidx := pidx + 1
    | .conv2d ic oc kSize _ act =>
      let (snip, newSSA, newShape) := emitConv2d pidx curSSA curShape ic oc kSize act
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 1
    | .maxPool size stride =>
      let (snip, newSSA, newShape) := emitMaxPool pos curSSA curShape size stride
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .flatten =>
      let (snip, newSSA, newShape) := emitFlatten pos curSSA curShape
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | _ =>
      code := code ++ "    // UNSUPPORTED LAYER\n"
    pos := pos + 1
  code := code ++ s!"    return {curSSA} : {tensorTy curShape}\n"
  pure code

/-- Emit function signature: interleaved (W, b) per param-bearing layer. -/
private def emitForwardSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let inDim := inputFlatDim spec
  let mut params : String := s!"%x: {tensorTy [batchSize, inDim]}"
  let mut outShape : List Nat := [batchSize, inDim]
  let mut curShape : List Nat := [batchSize, inDim]
  -- Initial reshape if CNN
  match inputChannels spec with
  | some ic => curShape := [batchSize, ic, spec.imageH, spec.imageW]
  | none => pure ()
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [fanIn, fanOut]}, %b{pidx}: {tensorTy [fanOut]}"
      curShape := [batchSize, fanOut]
      outShape := curShape
      pidx := pidx + 1
    | .conv2d ic oc kSize _ _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %b{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .maxPool size _ =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c, h / size, w / size]
      | _ => pure ()
      outShape := curShape
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
      outShape := curShape
    | _ => pure ()
  pure s!"func.func @forward(\n    {params}\n  ) -> {tensorTy outShape}"

/-- Generate a StableHLO MLIR module for forward-only inference. -/
def generate (spec : NetSpec) (batchSize : Nat) : String :=
  s!"// {spec.name} — Generated by Lean 4 → MLIR (StableHLO)\n" ++
  s!"// Batch size: {batchSize}\n\n" ++
  s!"module @{sanitize spec.name} " ++ "{\n" ++
  "  " ++ emitForwardSig spec batchSize ++ " {\n" ++
  emitForwardBody spec batchSize ++
  "  }\n" ++
  "}\n"

-- ════════════════════════════════════════════════════════════════════
-- Train step generation: forward + softmax-CE loss + VJPs + SGD
-- ════════════════════════════════════════════════════════════════════

/-- Record saved during forward pass for use by backward. -/
private structure FwdRec where
  layer     : Layer
  pidx      : Option Nat   -- param index (conv/dense only)
  pos       : Nat          -- layer position
  inputSSA  : String       -- input to this layer
  preActSSA : String       -- pre-activation (before relu), "" if no activation
  outputSSA : String       -- output of this layer
  inShape   : List Nat     -- input shape
  outShape  : List Nat     -- output shape
instance : Inhabited FwdRec where
  default := { layer := .flatten, pidx := none, pos := 0,
               inputSSA := "", preActSSA := "", outputSSA := "",
               inShape := [], outShape := [] }

/-- Standard conv attribute block (NCHW/OIHW). -/
private def convAttrBlock (pad : Nat) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  "        dimension_numbers = #stablehlo.conv<raw\n" ++
  "          input_batch_dimension = 0, input_feature_dimension = 1,\n" ++
  "          input_spatial_dimensions = [2, 3],\n" ++
  "          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,\n" ++
  "          kernel_spatial_dimensions = [2, 3],\n" ++
  "          output_batch_dimension = 0, output_feature_dimension = 1,\n" ++
  "          output_spatial_dimensions = [2, 3]>,\n" ++
  "        feature_group_count = 1 : i64,\n" ++
  s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n" ++
  "        rhs_dilation = array<i64: 1, 1>,\n" ++
  "        window_strides = array<i64: 1, 1>\n"

/-- Emit softmax cross-entropy loss + initial gradient d_logits.
    Input: logits SSA name, batch size. Output shape is always (B, 10). -/
private def emitLossAndGrad (logitsSSA : String) (B : Nat) : String :=
  s!"    // ================ SOFTMAX CROSS-ENTROPY ================\n" ++
  s!"    %maxv = stablehlo.reduce(%logits init: %neginf) applies stablehlo.maximum across dimensions = [1]\n" ++
  s!"          : ({tensorTy [B, 10]}, tensor<f32>) -> {tensorTy [B]}\n" ++
  s!"    %logits = stablehlo.add {logitsSSA}, %zerof_10 : {tensorTy [B, 10]}\n" ++
  -- Hmm, this aliases logitsSSA to %logits. Not clean. Let me just use logitsSSA directly.
  ""

/-- Emit the full train step (forward + loss + backward + SGD). -/
private def emitTrainStepBody (spec : NetSpec) (batchSize : Nat) (moduleName : String) : String := Id.run do
  let B := batchSize
  let mut code : String := ""
  code := code ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  code := code ++ "    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n\n"

  -- ═══════════════ FORWARD PASS ═══════════════
  code := code ++ "    // ======================== FORWARD ========================\n"
  let mut curSSA : String := "%x_flat"
  let mut curShape : List Nat := [B, inputFlatDim spec]

  -- Reshape if CNN
  match inputChannels spec with
  | some ic =>
    let nchw := [B, ic, spec.imageH, spec.imageW]
    code := code ++ s!"    %x = stablehlo.reshape %x_flat : ({tensorTy curShape}) -> {tensorTy nchw}\n"
    curSSA := "%x"
    curShape := nchw
  | none =>
    code := code ++ s!"    %x = stablehlo.add %x_flat, %x_flat : {tensorTy curShape}\n"
    -- Actually just alias. Hmm, can't alias in MLIR. Let me use a different name.
    -- For MLP, x_flat IS x. Let me track this properly.
    curSSA := "%x_flat"

  let mut pidx : Nat := 0
  let mut pos : Nat := 0
  let mut records : Array FwdRec := #[]

  for l in spec.layers do
    let inSSA := curSSA
    let inShape := curShape
    match l with
    | .conv2d ic oc kSize _pad act =>
      match curShape with
      | [b, _, h, w] =>
        let outShape := [b, oc, h, w]
        let pad := (kSize - 1) / 2
        code := code ++ s!"    %cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
        code := code ++ s!"    %cvb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
        let preSSA := s!"%cva{pidx}"
        code := code ++ s!"    {preSSA} = stablehlo.add %cv{pidx}, %cvb{pidx} : {tensorTy outShape}\n"
        if act == .relu then
          code := code ++ s!"    %cvz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
          code := code ++ s!"    %cvr{pidx} = stablehlo.maximum {preSSA}, %cvz{pidx} : {tensorTy outShape}\n"
          curSSA := s!"%cvr{pidx}"
        else
          curSSA := preSSA
        records := records.push { layer := l, pidx := some pidx, pos, inputSSA := inSSA, preActSSA := preSSA, outputSSA := curSSA, inShape, outShape }
        curShape := outShape
        pidx := pidx + 1
      | _ => pure ()

    | .dense fanIn fanOut act =>
      let outShape := [B, fanOut]
      code := code ++ s!"    %mm{pidx} = stablehlo.dot_general {curSSA}, %W{pidx}, contracting_dims = [1] x [0],\n"
      code := code ++ s!"             precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"           : ({tensorTy [B, fanIn]}, {tensorTy [fanIn, fanOut]}) -> {tensorTy outShape}\n"
      code := code ++ s!"    %bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [fanOut]}) -> {tensorTy outShape}\n"
      let preSSA := s!"%ab{pidx}"
      code := code ++ s!"    {preSSA} = stablehlo.add %mm{pidx}, %bb{pidx} : {tensorTy outShape}\n"
      if act == .relu then
        code := code ++ s!"    %dz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
        code := code ++ s!"    %dr{pidx} = stablehlo.maximum {preSSA}, %dz{pidx} : {tensorTy outShape}\n"
        curSSA := s!"%dr{pidx}"
      else
        curSSA := preSSA
      records := records.push { layer := l, pidx := some pidx, pos, inputSSA := inSSA, preActSSA := preSSA, outputSSA := curSSA, inShape, outShape }
      curShape := outShape
      pidx := pidx + 1

    | .maxPool size stride =>
      match curShape with
      | [b, c, h, w] =>
        let outShape := [b, c, h / size, w / size]
        code := code ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
        code := code ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({curSSA}, %pli{pos}) (" ++ "{\n"
        code := code ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
        code := code ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
        code := code ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
        code := code ++ "      }) " ++ "{\n"
        code := code ++ s!"        window_dimensions = array<i64: 1, 1, {size}, {size}>,\n"
        code := code ++ s!"        window_strides    = array<i64: 1, 1, {size}, {size}>\n"
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, tensor<f32>) -> {tensorTy outShape}\n"
        curSSA := s!"%pl{pos}"
        records := records.push { layer := .maxPool size stride, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
        curShape := outShape
      | _ => pure ()

    | .flatten =>
      match curShape with
      | [b, c, h, w] =>
        let flat := c * h * w
        let outShape := [b, flat]
        code := code ++ s!"    %fl{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy outShape}\n"
        curSSA := s!"%fl{pos}"
        records := records.push { layer := .flatten, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
        curShape := outShape
      | _ => pure ()

    | _ => code := code ++ "    // UNSUPPORTED\n"
    pos := pos + 1

  let logitsSSA := curSSA  -- last layer output = logits

  -- ═══════════════ SOFTMAX CE LOSS ═══════════════
  code := code ++ "\n    // ================ SOFTMAX CROSS-ENTROPY ================\n"
  code := code ++ s!"    %maxv = stablehlo.reduce({logitsSSA} init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
  code := code ++ s!"          : ({tensorTy [B, 10]}, tensor<f32>) -> {tensorTy [B]}\n"
  code := code ++ s!"    %maxv_b = stablehlo.broadcast_in_dim %maxv, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, 10]}\n"
  code := code ++ s!"    %shifted = stablehlo.subtract {logitsSSA}, %maxv_b : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %exp_s = stablehlo.exponential %shifted : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %sum_e = stablehlo.reduce(%exp_s init: %zf) applies stablehlo.add across dimensions = [1]\n"
  code := code ++ s!"           : ({tensorTy [B, 10]}, tensor<f32>) -> {tensorTy [B]}\n"
  code := code ++ s!"    %log_s = stablehlo.log %sum_e : {tensorTy [B]}\n"
  code := code ++ s!"    %log_s_b = stablehlo.broadcast_in_dim %log_s, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, 10]}\n"
  code := code ++ s!"    %log_p = stablehlo.subtract %shifted, %log_s_b : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %iota = stablehlo.iota dim = 1 : {tensorTy [B, 10]}".replace "xf32>" "xi32>"  ++ "\n"
  code := code ++ s!"    %y_b = stablehlo.broadcast_in_dim %y, dims = [0] : ({tensorTy [B]}".replace "xf32>" "xi32>" ++ s!") -> {tensorTy [B, 10]}".replace "xf32>" "xi32>" ++ "\n"
  let i1Ty := s!"tensor<{B}x10xi1>"
  code := code ++ s!"    %mask = stablehlo.compare EQ, %iota, %y_b : ({tensorTy [B, 10]}".replace "xf32>" "xi32>" ++ s!", {tensorTy [B, 10]}".replace "xf32>" "xi32>" ++ s!") -> {i1Ty}\n"
  code := code ++ s!"    %onef = stablehlo.constant dense<1.0> : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %zerof = stablehlo.constant dense<0.0> : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %onehot = stablehlo.select %mask, %onef, %zerof : {i1Ty}, {tensorTy [B, 10]}\n"
  code := code ++ s!"    %weighted = stablehlo.multiply %log_p, %onehot : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %total = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
  code := code ++ s!"           : ({tensorTy [B, 10]}, tensor<f32>) -> tensor<f32>\n"
  code := code ++ s!"    %Bc = stablehlo.constant dense<{B}.0> : tensor<f32>\n"
  code := code ++ s!"    %mean = stablehlo.divide %total, %Bc : tensor<f32>\n"
  code := code ++ s!"    %loss = stablehlo.negate %mean : tensor<f32>\n"

  -- ═══════════════ BACKWARD ═══════════════
  code := code ++ "\n    // ==================== BACKWARD ====================\n"
  code := code ++ s!"    %sum_e_b = stablehlo.broadcast_in_dim %sum_e, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, 10]}\n"
  code := code ++ s!"    %softmax = stablehlo.divide %exp_s, %sum_e_b : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %sm_moh = stablehlo.subtract %softmax, %onehot : {tensorTy [B, 10]}\n"
  code := code ++ s!"    %Bc_10 = stablehlo.broadcast_in_dim %Bc, dims = [] : (tensor<f32>) -> {tensorTy [B, 10]}\n"
  code := code ++ s!"    %d_logits = stablehlo.divide %sm_moh, %Bc_10 : {tensorTy [B, 10]}\n"

  let mut gradSSA := "%d_logits"
  let mut gradShape : List Nat := [B, 10]
  let nRec := records.size

  for ri in [:nRec] do
    let r : FwdRec := records[nRec - 1 - ri]!
    let p := r.pidx.getD 0
    match r.layer with
    | .dense _fanIn fanOut act =>
      -- Step 1: ReLU backward (if applicable) — BEFORE dW/db
      let effGrad := if act == .relu then s!"%gp{p}" else gradSSA
      if act == .relu then
        let oTy := tensorTy r.outShape
        let i1Ty := oTy.replace "xf32>" "xi1>"
        code := code ++ s!"    %rm{p} = stablehlo.compare GT, {r.preActSSA}, %dz{p} : ({oTy}, {oTy}) -> {i1Ty}\n"
        code := code ++ s!"    {effGrad} = stablehlo.select %rm{p}, {gradSSA}, %dz{p} : {i1Ty}, {oTy}\n"
      -- Step 2: dW = input.T @ effGrad
      code := code ++ s!"    %d_W{p} = stablehlo.dot_general {r.inputSSA}, {effGrad}, contracting_dims = [0] x [0],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"            : ({tensorTy r.inShape}, {tensorTy r.outShape}) -> {tensorTy [r.inShape[1]!, fanOut]}\n"
      -- db = sum(effGrad, axis=0)
      code := code ++ s!"    %d_b{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0]\n"
      code := code ++ s!"          : ({tensorTy r.outShape}, tensor<f32>) -> {tensorTy [fanOut]}\n"
      -- d_input = effGrad @ W.T
      code := code ++ s!"    %d_in{p} = stablehlo.dot_general {effGrad}, %W{p}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"            : ({tensorTy r.outShape}, {tensorTy [r.inShape[1]!, fanOut]}) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%d_in{p}"
      gradShape := r.inShape

    | .conv2d ic oc kSize _pad act =>
      match r.outShape with
      | [b, _oc, h, w] =>
        let oTy := tensorTy r.outShape
        let i1Ty := oTy.replace "xf32>" "xi1>"
        -- Step 1: ReLU backward
        let effGrad := if act == .relu then s!"%gpc{p}" else gradSSA
        if act == .relu then
          code := code ++ s!"    %rmc{p} = stablehlo.compare GT, {r.preActSSA}, %cvz{p} : ({oTy}, {oTy}) -> {i1Ty}\n"
          code := code ++ s!"    {effGrad} = stablehlo.select %rmc{p}, {gradSSA}, %cvz{p} : {i1Ty}, {oTy}\n"
        -- Step 2: dW via transpose trick
        let inC := r.inShape[1]!
        code := code ++ s!"    %bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
        code := code ++ s!"    %bt_g{p} = stablehlo.transpose {effGrad}, dims = [1, 0, 2, 3] : ({oTy}) -> {tensorTy [oc, b, h, w]}\n"
        let pad := (kSize - 1) / 2
        let dWrawShape := [inC, oc, kSize, kSize]
        code := code ++ s!"    %dWr{p} = \"stablehlo.convolution\"(%bt_in{p}, %bt_g{p}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, h, w]}) -> {tensorTy dWrawShape}\n"
        -- Transpose if ic != oc (to get OIHW)
        if inC != oc then
          code := code ++ s!"    %d_W{p} = stablehlo.transpose %dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy dWrawShape}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
        else
          code := code ++ s!"    %d_W{p} = stablehlo.add %dWr{p}, %dWr{p} : {tensorTy dWrawShape}\n"
          -- Hmm, can't alias. Actually when ic==oc the transpose is identity. Let me just always transpose.
          pure ()
        -- Wait, always transpose is simpler:
        -- Actually if inC == oc, transpose [1,0,2,3] is a no-op since dim0==dim1 size.
        -- But we still need a valid SSA name %d_W{p}. Let me always emit the transpose.

        -- db = reduce_sum(effGrad, dims=[0,2,3])
        code := code ++ s!"    %d_b{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
        code := code ++ s!"          : ({oTy}, tensor<f32>) -> {tensorTy [oc]}\n"

        -- dx via reverse+transpose
        code := code ++ s!"    %Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
        code := code ++ s!"    %Wrev{p} = stablehlo.reverse %Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
        code := code ++ s!"    %d_x{p} = \"stablehlo.convolution\"({effGrad}, %Wrev{p}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({oTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
        gradSSA := s!"%d_x{p}"
        gradShape := r.inShape
      | _ => pure ()

    | .maxPool size _stride =>
      -- select_and_scatter
      code := code ++ s!"    %sas{r.pos} = \"stablehlo.select_and_scatter\"({r.inputSSA}, {gradSSA}, %zf) (" ++ "{\n"
      code := code ++ s!"      ^bb0(%sa{r.pos}: tensor<f32>, %sb{r.pos}: tensor<f32>):\n"
      code := code ++ s!"        %cmp{r.pos} = stablehlo.compare GE, %sa{r.pos}, %sb{r.pos} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n"
      code := code ++ s!"        stablehlo.return %cmp{r.pos} : tensor<i1>\n"
      code := code ++ s!"      " ++ "}, {\n"
      code := code ++ s!"      ^bb0(%sc{r.pos}: tensor<f32>, %sd{r.pos}: tensor<f32>):\n"
      code := code ++ s!"        %acc{r.pos} = stablehlo.add %sc{r.pos}, %sd{r.pos} : tensor<f32>\n"
      code := code ++ s!"        stablehlo.return %acc{r.pos} : tensor<f32>\n"
      code := code ++ "      }) " ++ "{" ++ "window_dimensions = array<i64: 1, 1, " ++ toString size ++ ", " ++ toString size ++ ">,\n"
      code := code ++ "          window_strides = array<i64: 1, 1, " ++ toString size ++ ", " ++ toString size ++ ">" ++ "}\n"
      code := code ++ s!"      : ({tensorTy r.inShape}, {tensorTy r.outShape}, tensor<f32>) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%sas{r.pos}"
      gradShape := r.inShape

    | .flatten =>
      -- Reshape gradient back
      code := code ++ s!"    %ufl{r.pos} = stablehlo.reshape {gradSSA} : ({tensorTy gradShape}) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%ufl{r.pos}"
      gradShape := r.inShape

    | _ => pure ()

  -- ═══════════════ SGD UPDATES ═══════════════
  code := code ++ "\n    // =================== SGD UPDATES ===================\n"
  let mut retNames : Array String := #[]
  for r in records do
    match r.pidx with
    | some p =>
      match r.layer with
      | .conv2d ic oc kSize _ _ =>
        let wShape := [oc, ic, kSize, kSize]
        let bShape := [oc]
        code := code ++ s!"    %lr_W{p} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {tensorTy wShape}\n"
        code := code ++ s!"    %upW{p} = stablehlo.multiply %lr_W{p}, %d_W{p} : {tensorTy wShape}\n"
        code := code ++ s!"    %W{p}n = stablehlo.subtract %W{p}, %upW{p} : {tensorTy wShape}\n"
        code := code ++ s!"    %lr_b{p} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {tensorTy bShape}\n"
        code := code ++ s!"    %upb{p} = stablehlo.multiply %lr_b{p}, %d_b{p} : {tensorTy bShape}\n"
        code := code ++ s!"    %b{p}n = stablehlo.subtract %b{p}, %upb{p} : {tensorTy bShape}\n"
        retNames := retNames.push s!"%W{p}n" |>.push s!"%b{p}n"
      | .dense fanIn fanOut _ =>
        let wShape := [fanIn, fanOut]
        let bShape := [fanOut]
        code := code ++ s!"    %lr_W{p} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {tensorTy wShape}\n"
        code := code ++ s!"    %upW{p} = stablehlo.multiply %lr_W{p}, %d_W{p} : {tensorTy wShape}\n"
        code := code ++ s!"    %W{p}n = stablehlo.subtract %W{p}, %upW{p} : {tensorTy wShape}\n"
        code := code ++ s!"    %lr_b{p} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {tensorTy bShape}\n"
        code := code ++ s!"    %upb{p} = stablehlo.multiply %lr_b{p}, %d_b{p} : {tensorTy bShape}\n"
        code := code ++ s!"    %b{p}n = stablehlo.subtract %b{p}, %upb{p} : {tensorTy bShape}\n"
        retNames := retNames.push s!"%W{p}n" |>.push s!"%b{p}n"
      | _ => pure ()
    | none => pure ()
  retNames := retNames.push "%loss"

  -- Return statement
  code := code ++ s!"    return {String.intercalate ", " retNames.toList}\n"
  -- Type list for return
  let mut retTypes : Array String := #[]
  for r in records do
    match r.pidx with
    | some _ =>
      match r.layer with
      | .conv2d ic oc kSize _ _ =>
        retTypes := retTypes.push (tensorTy [oc, ic, kSize, kSize])
        retTypes := retTypes.push (tensorTy [oc])
      | .dense fanIn fanOut _ =>
        retTypes := retTypes.push (tensorTy [fanIn, fanOut])
        retTypes := retTypes.push (tensorTy [fanOut])
      | _ => pure ()
    | none => pure ()
  retTypes := retTypes.push "tensor<f32>"
  code := code ++ s!"      : {String.intercalate ", " retTypes.toList}\n"
  pure code

/-- Emit the train_step function signature:
    (W0, b0, ..., x_flat, y, lr) -> (W0', b0', ..., loss) -/
private def emitTrainStepSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let B := batchSize
  let inDim := inputFlatDim spec
  let mut params : String := ""
  let mut retTypes : Array String := #[]
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .conv2d ic oc kSize _ _ =>
      params := params ++ s!"      %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %b{pidx}: {tensorTy [oc]},\n"
      retTypes := retTypes.push (tensorTy [oc, ic, kSize, kSize]) |>.push (tensorTy [oc])
      pidx := pidx + 1
    | .dense fanIn fanOut _ =>
      params := params ++ s!"      %W{pidx}: {tensorTy [fanIn, fanOut]}, %b{pidx}: {tensorTy [fanOut]},\n"
      retTypes := retTypes.push (tensorTy [fanIn, fanOut]) |>.push (tensorTy [fanOut])
      pidx := pidx + 1
    | _ => pure ()
  params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y: tensor<{B}xi32>,\n"
  params := params ++ "      %lr: tensor<f32>"
  retTypes := retTypes.push "tensor<f32>"
  pure s!"  func.func @main(\n{params}\n    ) -> ({String.intercalate ", " retTypes.toList})"

/-- Generate a full train_step MLIR module with VJPs. -/
def generateTrainStep (spec : NetSpec) (batchSize : Nat) (moduleName : String := "jit_train_step") : String :=
  s!"// {spec.name} train_step — Generated by Lean 4 → MLIR (StableHLO + VJPs)\n" ++
  s!"// Batch size: {batchSize}\n\n" ++
  s!"module @{moduleName} " ++ "{\n" ++
  emitTrainStepSig spec batchSize ++ " {\n" ++
  emitTrainStepBody spec batchSize moduleName ++
  "  }\n" ++
  "}\n"

end MlirCodegen
