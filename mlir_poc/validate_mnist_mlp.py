"""Validate MNIST MLP: run numpy reference vs IREE GPU output."""
import numpy as np
import subprocess
import os

rng = np.random.default_rng(42)

# Random inputs + params
x  = rng.standard_normal((128, 784), dtype=np.float32)
W1 = rng.standard_normal((784, 512), dtype=np.float32) * 0.05
b1 = np.zeros((512,), dtype=np.float32)
W2 = rng.standard_normal((512, 512), dtype=np.float32) * 0.05
b2 = np.zeros((512,), dtype=np.float32)
W3 = rng.standard_normal((512, 10),  dtype=np.float32) * 0.05
b3 = np.zeros((10,), dtype=np.float32)

# Numpy reference forward pass
def relu(a): return np.maximum(a, 0.0)
h1 = relu(x  @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
logits_np = h2 @ W3 + b3

# Save as .npy for iree-run-module
os.makedirs("inputs", exist_ok=True)
for name, arr in [("x",x),("W1",W1),("b1",b1),("W2",W2),("b2",b2),("W3",W3),("b3",b3)]:
    np.save(f"inputs/{name}.npy", arr)

# Save expected output so we can diff
np.save("inputs/logits_np.npy", logits_np)

print(f"numpy logits[0,:5] = {logits_np[0,:5]}")
print(f"numpy logits mean={logits_np.mean():.6f} std={logits_np.std():.6f}")

# Invoke iree-run-module with --output=@ to capture to file
cmd = [
    "../.venv/bin/iree-run-module",
    "--module=mnist_mlp_cuda.vmfb",
    "--device=cuda",
    "--function=forward",
    "--input=@inputs/x.npy",
    "--input=@inputs/W1.npy",
    "--input=@inputs/b1.npy",
    "--input=@inputs/W2.npy",
    "--input=@inputs/b2.npy",
    "--input=@inputs/W3.npy",
    "--input=@inputs/b3.npy",
    "--output=@inputs/logits_iree.npy",
]
r = subprocess.run(cmd, capture_output=True, text=True)
print("iree stdout:", r.stdout[:500])
if r.returncode != 0:
    print("iree stderr:", r.stderr[:2000])
    raise SystemExit(1)

logits_iree = np.load("inputs/logits_iree.npy")
print(f"iree  logits[0,:5] = {logits_iree[0,:5]}")
diff = np.abs(logits_np - logits_iree)
print(f"max |np - iree| = {diff.max():.3e}")
print(f"mean |np - iree| = {diff.mean():.3e}")
assert diff.max() < 1e-3, f"Outputs diverge: max diff {diff.max()}"
print("PASS — IREE GPU matches numpy reference")
