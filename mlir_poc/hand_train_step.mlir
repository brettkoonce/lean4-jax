// Hand-written MNIST MLP train_step — forward + softmax CE loss + manual VJPs
// + SGD update. Drop-in replacement for jit_train_step.main (JAX-exported).
//
// Inputs (order matches JAX export):
//   W0: 784x512, b0: 512, W1: 512x512, b1: 512, W2: 512x10, b2: 10,
//   x: 128x784, y: 128 (int32 labels), lr: scalar f32
// Outputs: W0', b0', W1', b1', W2', b2', loss (scalar f32)

module @jit_train_step {
  func.func @main(
      %W0: tensor<784x512xf32>, %b0: tensor<512xf32>,
      %W1: tensor<512x512xf32>, %b1: tensor<512xf32>,
      %W2: tensor<512x10xf32>,  %b2: tensor<10xf32>,
      %x:  tensor<128x784xf32>, %y:  tensor<128xi32>,
      %lr: tensor<f32>
    ) -> (tensor<784x512xf32>, tensor<512xf32>,
          tensor<512x512xf32>, tensor<512xf32>,
          tensor<512x10xf32>,  tensor<10xf32>,
          tensor<f32>) {

    %zf = stablehlo.constant dense<0.0> : tensor<f32>
    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>

    // ======================= FORWARD =======================
    // h0pre = x @ W0 + b0 ; h0 = relu(h0pre)
    %mm0 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x784xf32>, tensor<784x512xf32>) -> tensor<128x512xf32>
    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1]
         : (tensor<512xf32>) -> tensor<128x512xf32>
    %h0pre = stablehlo.add %mm0, %b0b : tensor<128x512xf32>
    %z512 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %h0 = stablehlo.maximum %h0pre, %z512 : tensor<128x512xf32>

    // h1pre = h0 @ W1 + b1 ; h1 = relu(h1pre)
    %mm1 = stablehlo.dot_general %h0, %W1, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1]
         : (tensor<512xf32>) -> tensor<128x512xf32>
    %h1pre = stablehlo.add %mm1, %b1b : tensor<128x512xf32>
    %h1 = stablehlo.maximum %h1pre, %z512 : tensor<128x512xf32>

    // logits = h1 @ W2 + b2
    %mm2 = stablehlo.dot_general %h1, %W2, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1]
         : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %mm2, %b2b : tensor<128x10xf32>

    // ================= SOFTMAX CROSS-ENTROPY =================
    // m = reduce_max(logits, axis=1) : (128,)
    %maxv = stablehlo.reduce(%logits init: %neginf) applies stablehlo.maximum across dimensions = [1]
          : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %maxv_b = stablehlo.broadcast_in_dim %maxv, dims = [0]
            : (tensor<128xf32>) -> tensor<128x10xf32>
    %shifted = stablehlo.subtract %logits, %maxv_b : tensor<128x10xf32>
    %exp_s = stablehlo.exponential %shifted : tensor<128x10xf32>
    %sum_e = stablehlo.reduce(%exp_s init: %zf) applies stablehlo.add across dimensions = [1]
           : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %log_s = stablehlo.log %sum_e : tensor<128xf32>
    %log_s_b = stablehlo.broadcast_in_dim %log_s, dims = [0]
             : (tensor<128xf32>) -> tensor<128x10xf32>
    %log_p = stablehlo.subtract %shifted, %log_s_b : tensor<128x10xf32>

    // one_hot(y): iota(axis=1) == broadcast(y) → select(mask, 1.0, 0.0)
    %iota = stablehlo.iota dim = 1 : tensor<128x10xi32>
    %y_b = stablehlo.broadcast_in_dim %y, dims = [0]
         : (tensor<128xi32>) -> tensor<128x10xi32>
    %mask = stablehlo.compare EQ, %iota, %y_b : (tensor<128x10xi32>, tensor<128x10xi32>) -> tensor<128x10xi1>
    %onef = stablehlo.constant dense<1.0> : tensor<128x10xf32>
    %zerof = stablehlo.constant dense<0.0> : tensor<128x10xf32>
    %onehot = stablehlo.select %mask, %onef, %zerof : tensor<128x10xi1>, tensor<128x10xf32>

    // loss = -mean( sum(log_p * onehot, axis=1) ) = -sum(log_p * onehot) / B
    %weighted = stablehlo.multiply %log_p, %onehot : tensor<128x10xf32>
    %total = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [0, 1]
           : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %Bc = stablehlo.constant dense<128.0> : tensor<f32>
    %mean = stablehlo.divide %total, %Bc : tensor<f32>
    %loss = stablehlo.negate %mean : tensor<f32>

    // ======================= BACKWARD =======================
    // softmax = exp_shifted / sum_exp[:, None]
    %sum_e_b = stablehlo.broadcast_in_dim %sum_e, dims = [0]
             : (tensor<128xf32>) -> tensor<128x10xf32>
    %softmax = stablehlo.divide %exp_s, %sum_e_b : tensor<128x10xf32>
    // d_logits = (softmax - onehot) / B
    %sm_moh = stablehlo.subtract %softmax, %onehot : tensor<128x10xf32>
    %Bc_10 = stablehlo.broadcast_in_dim %Bc, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %d_logits = stablehlo.divide %sm_moh, %Bc_10 : tensor<128x10xf32>

    // d_W2 = h1.T @ d_logits  (contract batch dim of both)
    %d_W2 = stablehlo.dot_general %h1, %d_logits, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    // d_b2 = sum(d_logits, axis=0)
    %d_b2 = stablehlo.reduce(%d_logits init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    // d_h1 = d_logits @ W2.T  (contract d_logits dim 1 with W2 dim 1)
    %d_h1 = stablehlo.dot_general %d_logits, %W2, contracting_dims = [1] x [1],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>

    // ReLU backward: d_h1pre = select(h1pre > 0, d_h1, 0)
    %m1 = stablehlo.compare GT, %h1pre, %z512 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %d_h1pre = stablehlo.select %m1, %d_h1, %z512 : tensor<128x512xi1>, tensor<128x512xf32>

    // d_W1 = h0.T @ d_h1pre
    %d_W1 = stablehlo.dot_general %h0, %d_h1pre, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %d_b1 = stablehlo.reduce(%d_h1pre init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    // d_h0 = d_h1pre @ W1.T
    %d_h0 = stablehlo.dot_general %d_h1pre, %W1, contracting_dims = [1] x [1],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>

    // ReLU backward
    %m0 = stablehlo.compare GT, %h0pre, %z512 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %d_h0pre = stablehlo.select %m0, %d_h0, %z512 : tensor<128x512xi1>, tensor<128x512xf32>

    // d_W0 = x.T @ d_h0pre
    %d_W0 = stablehlo.dot_general %x, %d_h0pre, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x784xf32>, tensor<128x512xf32>) -> tensor<784x512xf32>
    %d_b0 = stablehlo.reduce(%d_h0pre init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>

    // ======================= SGD UPDATE =======================
    %lr_W0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %upW0 = stablehlo.multiply %lr_W0, %d_W0 : tensor<784x512xf32>
    %W0n = stablehlo.subtract %W0, %upW0 : tensor<784x512xf32>

    %lr_b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %upb0 = stablehlo.multiply %lr_b, %d_b0 : tensor<512xf32>
    %b0n = stablehlo.subtract %b0, %upb0 : tensor<512xf32>

    %lr_W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %upW1 = stablehlo.multiply %lr_W1, %d_W1 : tensor<512x512xf32>
    %W1n = stablehlo.subtract %W1, %upW1 : tensor<512x512xf32>

    %upb1 = stablehlo.multiply %lr_b, %d_b1 : tensor<512xf32>
    %b1n = stablehlo.subtract %b1, %upb1 : tensor<512xf32>

    %lr_W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %upW2 = stablehlo.multiply %lr_W2, %d_W2 : tensor<512x10xf32>
    %W2n = stablehlo.subtract %W2, %upW2 : tensor<512x10xf32>

    %lr_b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %upb2 = stablehlo.multiply %lr_b2, %d_b2 : tensor<10xf32>
    %b2n = stablehlo.subtract %b2, %upb2 : tensor<10xf32>

    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n, %loss
      : tensor<784x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>,
        tensor<512x10xf32>, tensor<10xf32>, tensor<f32>
  }
}
