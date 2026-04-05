"""Export a JAX train_step (MLP forward + softmax CE loss + SGD) to StableHLO.
Used to bootstrap Phase 2 — give us a known-correct training MLIR module we can
run via IREE, until hand-written VJPs in Lean replace it."""
import jax
import jax.numpy as jnp
from jax import export
import numpy as np

BATCH = 128

def forward(W0, b0, W1, b1, W2, b2, x):
    h = jax.nn.relu(x @ W0 + b0)
    h = jax.nn.relu(h @ W1 + b1)
    return h @ W2 + b2

def loss_fn(W0, b0, W1, b1, W2, b2, x, y):
    logits = forward(W0, b0, W1, b1, W2, b2, x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    # labels are int32 class indices
    return -jnp.mean(logp[jnp.arange(x.shape[0]), y])

def train_step(W0, b0, W1, b1, W2, b2, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0,1,2,3,4,5))(
        W0, b0, W1, b1, W2, b2, x, y)
    dW0, db0, dW1, db1, dW2, db2 = grads
    W0 = W0 - lr * dW0
    b0 = b0 - lr * db0
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W0, b0, W1, b1, W2, b2, loss

# Shape specs for export (static shapes)
spec_W0 = jax.ShapeDtypeStruct((784, 512), jnp.float32)
spec_b0 = jax.ShapeDtypeStruct((512,),      jnp.float32)
spec_W1 = jax.ShapeDtypeStruct((512, 512),  jnp.float32)
spec_b1 = jax.ShapeDtypeStruct((512,),      jnp.float32)
spec_W2 = jax.ShapeDtypeStruct((512, 10),   jnp.float32)
spec_b2 = jax.ShapeDtypeStruct((10,),       jnp.float32)
spec_x  = jax.ShapeDtypeStruct((BATCH, 784), jnp.float32)
spec_y  = jax.ShapeDtypeStruct((BATCH,),     jnp.int32)
spec_lr = jax.ShapeDtypeStruct((),           jnp.float32)

exported = export.export(jax.jit(train_step))(
    spec_W0, spec_b0, spec_W1, spec_b1, spec_W2, spec_b2, spec_x, spec_y, spec_lr)

mlir = exported.mlir_module()
open("mlir_poc/train_step.mlir", "w").write(mlir)
print(f"Wrote train_step.mlir ({len(mlir)} chars)")
print(f"  function name: {exported.fun_name}")
print(f"  input signatures: {[str(x) for x in exported.in_avals]}")
print(f"  output signatures: {[str(x) for x in exported.out_avals]}")
