---
usemathjax: true
---

# Lean4-MLIR: Verified VJP Proofs for Deep Learning

Machine-checked proofs that every layer's backward pass (vector-Jacobian product)
matches its forward-pass Jacobian, assembled from primitives all the way up to
the full Vision Transformer backbone. Zero `sorry`s, 30 axioms.

**Companion artifact for the book *Verified Deep Learning with Lean 4*.**
Full source: [github.com/brettkoonce/lean4-jax](https://github.com/brettkoonce/lean4-jax).

## Navigate the proofs

* [Blueprint (interactive web version)]({{ site.url }}/blueprint/)
* [Dependency graph (clickable DAG)]({{ site.url }}/blueprint/dep_graph_document.html)
* [Blueprint (PDF)]({{ site.url }}/blueprint.pdf)
* [Lean API docs (doc-gen4)]({{ site.url }}/docs/)

## The big idea

Every modern architecture's backward pass decomposes into **three structural rules**
(chain, fan-in, product) plus **five closed-form Jacobian tricks**
(diagonal, sparse toeplitz, binary selection, rank-1 correction, outer product).
The proof suite makes this claim concrete: every layer VJP is a `HasVJP` instance,
every composition is `vjp_comp`, every residual is `biPath_has_vjp`, and the whole
ViT backbone lands as a single `HasVJPMat` theorem derived from one axiom
(`mhsa_has_vjp_mat`).

## Results from the companion code

Trained from scratch on Imagenette, 224×224, Adam + cosine LR:

| Model | Val accuracy |
|---|---|
| ResNet-34 | 90.29% |
| ResNet-50 | 89.40% |
| EfficientNetV2-S | 88.50% |
| MobileNetV4-Medium | 84.58% |
| ViT-Tiny | 71.70% |

All generated from ~15-line Lean `NetSpec` declarations, compiled through
StableHLO MLIR, and trained end-to-end via IREE on AMD gfx1100 / NVIDIA CUDA.
