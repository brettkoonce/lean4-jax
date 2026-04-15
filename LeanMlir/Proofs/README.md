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

Six facts from real analysis, stated in `Tensor.lean`:

| Axiom | What it says |
|-------|-------------|
| `pdiv_id` | ∂xᵢ/∂xⱼ = δᵢⱼ |
| `pdiv_comp` | Chain rule: ∂(g∘f)/∂x = Σ (∂f/∂x)(∂g/∂f) |
| `pdiv_add` | Sum rule: ∂(f+g)/∂x = ∂f/∂x + ∂g/∂x |
| `pdiv_mul` | Product rule: ∂(f·g)/∂x = f'g + fg' |
| `pdiv_bnAffine` | ∂(γv+β)/∂v = γδᵢⱼ |
| `pdiv_bnNormalize` | ∂x̂ⱼ/∂xᵢ = (istd/N)(Nδᵢⱼ − 1 − x̂ᵢx̂ⱼ) |

Everything else — every `HasVJP` instance, every composition,
every correctness theorem — is proved from these six axioms by
Lean's type checker.

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
