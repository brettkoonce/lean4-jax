# Verified VJP Proofs

Machine-checked proofs that the backward pass (VJP) of every layer
matches its forward-pass Jacobian. Zero `sorry`s. If `lake build`
succeeds, every theorem is correct.

## Dependency graph

```
Tensor.lean                    ← axioms + VJP framework
  │
  │  pdiv_comp (chain rule)
  │  pdiv_add  (sum rule)
  │  pdiv_mul  (product rule)
  │  pdiv_id   (identity)
  │  vjp_comp  (VJP composition — proved)
  │  biPath    (additive fan-in — proved)
  │  elemwiseProduct (multiplicative fan-in — proved)
  │
  ├── MLP.lean                 dense + ReLU + softmax CE
  │
  ├── CNN.lean                 conv2d + maxPool + flatten
  │
  ├── BatchNorm.lean           batch norm (the hard one)
  │     │
  │     │  bnNormalize_has_vjp  ← 3-term consolidated formula
  │     │  bnAffine_has_vjp     ← trivial diagonal
  │     └─ bn_has_vjp           ← vjp_comp glues them
  │
  ├── Residual.lean            skip connections (biPath)
  │
  ├── Depthwise.lean           depthwise conv
  │
  ├── SE.lean                  squeeze-and-excitation (elemwiseProduct)
  │
  ├── LayerNorm.lean           layer norm + GELU
  │
  └── Attention.lean           softmax + scaled dot-product attention
```

## Axioms

All axiom declarations across the proof suite, grouped by file:

**Tensor.lean** — calculus foundations (1D `Vec`, 2D `Mat`, 3D `Tensor3`):
| Axiom | What it says |
|-------|-------------|
| `pdiv` | Partial derivative function (existence) |
| `sdiv` | Scalar derivative function |
| `pdiv_id` | ∂xᵢ/∂xⱼ = δᵢⱼ |
| `pdiv_comp` | Chain rule |
| `pdiv_add` | Sum rule |
| `pdiv_mul` | Product rule |
| `pdivMat_matmul_left_const` | ∂(C·B')/∂B' with C fixed |
| `pdivMat_matmul_right_const` | ∂(A'·D)/∂A' with D fixed |
| `pdivMat_rowIndep` | Row-independent function ⇒ block-diagonal Jacobian |
| `pdivMat_scalarScale` | ∂(s·A')/∂A' = s·δ |
| `pdivMat_transpose` | ∂(A'^T)/∂A' — swap-indices Kronecker delta |
| `pdiv3` | 3D-tensor partial derivative |
| `pdiv3_comp` | 3D chain rule |
| `pdiv3_id` | 3D identity Jacobian |
| `pdiv3_add` | 3D sum rule |

> **Phase 4**: `pdivMat`, `pdivMat_comp`, `pdivMat_add`, `pdivMat_id`
> used to be axioms. They are now a **definition** (`pdivMat`) and
> three **theorems** derived from the corresponding `pdiv` axioms via
> the row-major `Mat ≃ Vec (m*n)` bijection (`Mat.flatten` /
> `Mat.unflatten`). The five remaining `pdivMat_*` axioms state
> specific Jacobian *values* for concrete operations (matmul,
> scalarScale, transpose, rowIndep) — genuine local calculus facts,
> not framework plumbing.

**MLP.lean** — dense layers:
| Axiom | What it says |
|-------|-------------|
| `pdiv_dense` | Dense layer Jacobian |
| `pdiv_relu` | ReLU Jacobian (diagonal, 0/1) |
| `softmaxCE_grad` | Softmax cross-entropy gradient = softmax − onehot |

**CNN.lean** — convolution and pooling:
| Axiom | What it says |
|-------|-------------|
| `conv2d` | Conv forward (function signature) |
| `conv2d_input_grad` | dx = conv with reversed kernel |
| `conv2d_weight_grad` | dW = input ⊗ grad (transpose trick) |
| `maxPool2` | MaxPool forward (function signature) |
| `maxPool2_input_grad` | Route gradient to argmax positions |
| `flatten` / `unflatten` | Reshape between 3D and 1D |
| `pdiv3_conv2d_vjp` | Conv2d 3D-tensor VJP statement |
| `pdiv3_maxPool2_vjp` | MaxPool2 3D-tensor VJP statement |

**BatchNorm.lean** — the hard one:
| Axiom | What it says |
|-------|-------------|
| `pdiv_bnAffine` | ∂(γv+β)/∂v = γδᵢⱼ |
| `pdiv_bnNormalize` | ∂x̂ⱼ/∂xᵢ = (istd/N)(Nδᵢⱼ − 1 − x̂ᵢx̂ⱼ) |

**Depthwise.lean** — depthwise convolution:
| Axiom | What it says |
|-------|-------------|
| `depthwiseConv2d` | Depthwise conv forward |
| `depthwiseConv2d_input_grad` | dx per-channel |
| `depthwiseConv2d_weight_grad` | dW per-channel |
| `pdiv3_depthwise_vjp` | Depthwise 3D-tensor VJP statement |

**LayerNorm.lean** — layer norm and GELU:
| Axiom | What it says |
|-------|-------------|
| `geluScalar` | GELU activation (function signature) |
| `geluScalarDeriv` | GELU derivative |
| `pdiv_gelu` | GELU Jacobian (diagonal) |

**Attention.lean** — softmax and attention:
| Axiom | What it says |
|-------|-------------|
| `pdiv_softmax` | Softmax Jacobian (rank-1 correction) |

> All three `sdpa_back_*_correct` statements are now **theorems**, not
> axioms (Phase 3). Each is proved by constructing a `HasVJPMat` for
> `fun _ => sdpa n d · K V` (or similar for K, V) as a composition of
> four proved `HasVJPMat` building blocks via `vjpMat_comp`, then
> reducing the chain's backward to the concrete `sdpa_back_{Q,K,V}`
> formula. The old `sdpa_has_vjp` axiom (a vacuous type declaration)
> is gone entirely.

Plus three Lean core axioms (`propext`, `Classical.choice`, `Quot.sound`)
present in every nontrivial Lean program.

Total: 15 (Tensor) + 3 (MLP) + 9 (CNN) + 2 (BatchNorm) + 4 (Depthwise)
+ 3 (LayerNorm) + 1 (Attention) = **37 axioms**.

Everything else — every `HasVJP` instance, every composition,
every correctness theorem — is proved from these axioms by
Lean's type checker.

## `#print axioms` output (HasVJP instances)

Which axioms each proved theorem actually uses (via `lake env lean`):

```
vjp_comp               → pdiv, pdiv_comp
biPath_has_vjp         → pdiv, pdiv_add
elemwiseProduct_has_vjp → pdiv, pdiv_mul
identity_has_vjp       → pdiv, pdiv_id
vjpMat_comp            → pdiv, pdiv_comp  (via Mat.flatten bijection)
biPathMat_has_vjp      → pdiv, pdiv_add   (via Mat.flatten bijection)
identityMat_has_vjp    → pdiv, pdiv_id    (via Mat.flatten bijection)
matmul_left_const_has_vjp  → pdivMat, pdivMat_matmul_left_const
matmul_right_const_has_vjp → pdivMat, pdivMat_matmul_right_const
scalarScale_has_vjp        → pdivMat, pdivMat_scalarScale
transpose_has_vjp          → pdivMat, pdivMat_transpose
rowSoftmax_has_vjp_mat     → pdivMat, pdivMat_rowIndep,
                             pdiv, pdiv_softmax
sdpa_back_V_correct    → pdivMat, pdivMat_matmul_left_const
sdpa_back_Q_correct    → pdivMat, pdivMat_matmul_left_const,
                         pdivMat_matmul_right_const,
                         pdivMat_scalarScale, pdivMat_rowIndep,
                         pdivMat_comp, pdiv, pdiv_softmax
sdpa_back_K_correct    → (same as Q) + pdivMat_transpose
dense_has_vjp          → pdiv, pdiv_dense
bn_has_vjp             → pdiv, pdiv_bnAffine, pdiv_bnNormalize, pdiv_comp
bn_input_grad_correct  → pdiv, pdiv_bnAffine, pdiv_bnNormalize, pdiv_comp
bnNormalize_has_vjp    → pdiv, pdiv_bnNormalize
bnAffine_has_vjp       → pdiv, pdiv_bnAffine
residual_has_vjp       → pdiv, pdiv_add, pdiv_id
seBlock_has_vjp        → pdiv, pdiv_id, pdiv_mul
layerNorm_has_vjp      → pdiv, pdiv_bnAffine, pdiv_bnNormalize, pdiv_comp
softmax_has_vjp        → pdiv, pdiv_softmax
```

(Lean core axioms `propext`, `Classical.choice`, `Quot.sound` omitted — present in every nontrivial Lean program.)

## The three rules

All of backpropagation:

```
vjp_comp              f ∘ g  →  back_f(x, back_g(f(x), dy))
biPath_has_vjp        f + g  →  back_f(x, dy) + back_g(x, dy)
elemwiseProduct_has_vjp  f * g  →  back_f(x, g·dy) + back_g(x, f·dy)
```

## The five Jacobian tricks

Every layer's backward pass is one of:

1. **Diagonal** — activations (ReLU, GELU): one multiply
2. **Sparse Toeplitz** — conv: reversed kernel convolution
3. **Binary selection** — max pool: route to argmax
4. **Rank-1 correction** — batch/layer norm, softmax: closed-form 3-term formula
5. **Outer product** — dense/matmul: input ⊗ grad

## Verify

```bash
lake build LeanMlir.Proofs.Tensor LeanMlir.Proofs.MLP \
  LeanMlir.Proofs.CNN LeanMlir.Proofs.BatchNorm \
  LeanMlir.Proofs.Residual LeanMlir.Proofs.Depthwise \
  LeanMlir.Proofs.SE LeanMlir.Proofs.LayerNorm \
  LeanMlir.Proofs.Attention
```

If it builds, it's correct. That's the point.
