"""Diff the hand-written mnist_cnn.mlir against a JAX reference."""
import numpy as np, subprocess, os
import jax, jax.numpy as jnp

rng = np.random.default_rng(7)
B = 4  # batch for smoke test

x   = rng.standard_normal((B, 1, 28, 28), dtype=np.float32)
W0  = rng.standard_normal((32, 1, 3, 3),  dtype=np.float32) * 0.1
b0  = rng.standard_normal((32,),           dtype=np.float32) * 0.1
W1  = rng.standard_normal((32, 32, 3, 3), dtype=np.float32) * 0.05
b1  = rng.standard_normal((32,),           dtype=np.float32) * 0.1
W2  = rng.standard_normal((6272, 512),    dtype=np.float32) * 0.02
b2  = rng.standard_normal((512,),          dtype=np.float32) * 0.05
W3  = rng.standard_normal((512, 512),     dtype=np.float32) * 0.05
b3  = rng.standard_normal((512,),          dtype=np.float32) * 0.05
W4  = rng.standard_normal((512, 10),      dtype=np.float32) * 0.05
b4  = rng.standard_normal((10,),           dtype=np.float32) * 0.05

def forward_jax(x, W0, b0, W1, b1, W2, b2, W3, b3, W4, b4):
    # conv 1
    h = jax.lax.conv_general_dilated(x, W0, (1,1), 'SAME',
          dimension_numbers=('NCHW', 'OIHW', 'NCHW')) + b0.reshape(1,-1,1,1)
    h = jax.nn.relu(h)
    # conv 2
    h = jax.lax.conv_general_dilated(h, W1, (1,1), 'SAME',
          dimension_numbers=('NCHW', 'OIHW', 'NCHW')) + b1.reshape(1,-1,1,1)
    h = jax.nn.relu(h)
    # max pool 2x2 stride 2
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max,
          (1,1,2,2), (1,1,2,2), 'VALID')
    # flatten
    h = h.reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ W2 + b2)
    h = jax.nn.relu(h @ W3 + b3)
    return h @ W4 + b4

ref = np.asarray(forward_jax(x, W0, b0, W1, b1, W2, b2, W3, b3, W4, b4))
print(f"JAX ref: shape={ref.shape}, row0[:5]={ref[0,:5]}")

# Save all inputs as .npy for iree-run-module
os.makedirs("mlir_poc/cnn_inputs", exist_ok=True)
for name, arr in [("x",x),("W0",W0),("b0",b0),("W1",W1),("b1",b1),
                  ("W2",W2),("b2",b2),("W3",W3),("b3",b3),("W4",W4),("b4",b4)]:
    np.save(f"mlir_poc/cnn_inputs/{name}.npy", arr)

# Run IREE on CUDA
cmd = [
    "/home/skoonce/lean/klawd_max_power/iree-build/tools/iree-run-module",
    "--module=/tmp/mnist_cnn_cuda.vmfb", "--device=cuda", "--function=forward",
] + [f"--input=@mlir_poc/cnn_inputs/{n}.npy" for n in
     ["x","W0","b0","W1","b1","W2","b2","W3","b3","W4","b4"]] + [
    "--output=@/tmp/cnn_out.npy",
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("IREE failed:", r.stderr[:2000]); raise SystemExit(1)
iree_out = np.load("/tmp/cnn_out.npy")
print(f"IREE out: shape={iree_out.shape}, row0[:5]={iree_out[0,:5]}")
diff = np.abs(ref - iree_out)
print(f"max|ref - iree| = {diff.max():.3e}")
print(f"mean|ref - iree| = {diff.mean():.3e}")
assert diff.max() < 1e-3, f"Outputs diverge: {diff.max()}"
print("PASS — hand-written CNN MLIR matches JAX reference")
