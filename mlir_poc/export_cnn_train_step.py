"""Export a JAX CNN train_step to StableHLO.
Matches mnistCnn: 2 conv (1→32, 32→32) + max pool + flatten + 3 dense."""
import jax, jax.numpy as jnp
from jax import export
import numpy as np

BATCH = 128

def forward(W0, b0, W1, b1, W2, b2, W3, b3, W4, b4, x):
    # x flat (batch, 784) → NHWC (batch, 28, 28, 1); kernels HWIO
    h = x.reshape(-1, 28, 28, 1)
    h = jax.nn.relu(jax.lax.conv_general_dilated(h, W0, (1,1), 'SAME',
          dimension_numbers=('NHWC','HWIO','NHWC')) + b0.reshape(1,1,1,-1))
    h = jax.nn.relu(jax.lax.conv_general_dilated(h, W1, (1,1), 'SAME',
          dimension_numbers=('NHWC','HWIO','NHWC')) + b1.reshape(1,1,1,-1))
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max,
          (1,2,2,1), (1,2,2,1), 'VALID')
    h = h.reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ W2 + b2)
    h = jax.nn.relu(h @ W3 + b3)
    return h @ W4 + b4

def loss_fn(W0, b0, W1, b1, W2, b2, W3, b3, W4, b4, x, y):
    logits = forward(W0, b0, W1, b1, W2, b2, W3, b3, W4, b4, x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(logp[jnp.arange(x.shape[0]), y])

def train_step(W0, b0, W1, b1, W2, b2, W3, b3, W4, b4, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0,1,2,3,4,5,6,7,8,9))(
        W0, b0, W1, b1, W2, b2, W3, b3, W4, b4, x, y)
    params = (W0, b0, W1, b1, W2, b2, W3, b3, W4, b4)
    new_params = tuple(p - lr * g for p, g in zip(params, grads))
    return (*new_params, loss)

specs = (
    jax.ShapeDtypeStruct((3, 3, 1, 32),   jnp.float32),  # W0 HWIO
    jax.ShapeDtypeStruct((32,),           jnp.float32),  # b0
    jax.ShapeDtypeStruct((3, 3, 32, 32),  jnp.float32),  # W1 HWIO
    jax.ShapeDtypeStruct((32,),           jnp.float32),  # b1
    jax.ShapeDtypeStruct((6272, 512),     jnp.float32),  # W2
    jax.ShapeDtypeStruct((512,),          jnp.float32),  # b2
    jax.ShapeDtypeStruct((512, 512),      jnp.float32),  # W3
    jax.ShapeDtypeStruct((512,),          jnp.float32),  # b3
    jax.ShapeDtypeStruct((512, 10),       jnp.float32),  # W4
    jax.ShapeDtypeStruct((10,),           jnp.float32),  # b4
    jax.ShapeDtypeStruct((BATCH, 784),    jnp.float32),  # x (flat input)
    jax.ShapeDtypeStruct((BATCH,),        jnp.int32),    # y
    jax.ShapeDtypeStruct((),              jnp.float32),  # lr
)
exported = export.export(jax.jit(train_step))(*specs)
mlir = exported.mlir_module()
open("mlir_poc/cnn_train_step.mlir", "w").write(mlir)
print(f"Wrote cnn_train_step.mlir ({len(mlir)} chars)")
print(f"  function: {exported.fun_name}")
