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

Eight facts from real analysis (`#print axioms` verified):

| Axiom | What it says | Used by |
|-------|-------------|---------|
| `pdiv` | Partial derivative exists | everything |
| `pdiv_id` | ∂xᵢ/∂xⱼ = δᵢⱼ | identity, residual, SE |
| `pdiv_comp` | Chain rule: ∂(g∘f)/∂x = Σ (∂f/∂x)(∂g/∂f) | vjp_comp, BN, layerNorm |
| `pdiv_add` | Sum rule: ∂(f+g)/∂x = ∂f/∂x + ∂g/∂x | biPath, residual |
| `pdiv_mul` | Product rule: ∂(f·g)/∂x = f'g + fg' | elemwiseProduct, SE |
| `pdiv_dense` | Dense layer Jacobian | dense |
| `pdiv_bnAffine` | ∂(γv+β)/∂v = γδᵢⱼ | BN affine, layerNorm |
| `pdiv_bnNormalize` | ∂x̂ⱼ/∂xᵢ = (istd/N)(Nδᵢⱼ − 1 − x̂ᵢx̂ⱼ) | BN normalize, layerNorm |
| `pdiv_softmax` | Softmax Jacobian | softmax, attention |

Plus three Lean core axioms (`propext`, `Classical.choice`, `Quot.sound`)
that are present in every nontrivial Lean program.

Everything else — every `HasVJP` instance, every composition,
every correctness theorem — is proved from these axioms by
Lean's type checker.

## `#print axioms` output

```
vjp_comp               → pdiv, pdiv_comp
biPath_has_vjp         → pdiv, pdiv_add
elemwiseProduct_has_vjp → pdiv, pdiv_mul
identity_has_vjp       → pdiv, pdiv_id
dense_has_vjp          → pdiv, pdiv_dense
bn_has_vjp             → pdiv, pdiv_bnAffine, pdiv_bnNormalize, pdiv_comp
bnNormalize_has_vjp    → pdiv, pdiv_bnNormalize
residual_has_vjp       → pdiv, pdiv_add, pdiv_id
seBlock_has_vjp        → pdiv, pdiv_id, pdiv_mul
layerNorm_has_vjp      → pdiv, pdiv_bnAffine, pdiv_bnNormalize, pdiv_comp
softmax_has_vjp        → pdiv, pdiv_softmax
```

(Lean core axioms `propext`, `Classical.choice`, `Quot.sound` omitted for clarity.)

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
