"""End-to-end validation: train MLP in JAX, run test-set inference through
Lean-generated IREE module, compare accuracies.

Layout matches Lean MlirCodegen.generate mnistMlp 128:
  @forward(x:128x784, W0:784x512, b0:512, W1:512x512, b1:512, W2:512x10, b2:10)
  -> 128x10
"""
import os, struct, subprocess, time
import numpy as np
import jax, jax.numpy as jnp
from jax import random, value_and_grad, jit

# ---------- data ----------
def load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, r, c = struct.unpack(">4I", f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r*c).astype(np.float32) / 255.0

def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">2I", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)

DATA = "data"
Xtr = load_idx_images(f"{DATA}/train-images-idx3-ubyte")
Ytr = load_idx_labels(f"{DATA}/train-labels-idx1-ubyte")
Xte = load_idx_images(f"{DATA}/t10k-images-idx3-ubyte")
Yte = load_idx_labels(f"{DATA}/t10k-labels-idx1-ubyte")
print(f"train: {Xtr.shape}, test: {Xte.shape}")

# ---------- JAX training ----------
# layer dims: 784 -> 512 -> 512 -> 10
key = random.PRNGKey(42)
def init_dense(key, fin, fout):
    k1, k2 = random.split(key)
    # He init (gain sqrt(2) for ReLU)
    W = random.normal(k1, (fin, fout)) * jnp.sqrt(2.0/fin)
    b = jnp.zeros((fout,))
    return W, b

k0, k1, k2 = random.split(key, 3)
W0, b0 = init_dense(k0, 784, 512)
W1, b1 = init_dense(k1, 512, 512)
W2, b2 = init_dense(k2, 512,  10)
params = (W0, b0, W1, b1, W2, b2)

def forward(params, x):
    W0,b0,W1,b1,W2,b2 = params
    h = jax.nn.relu(x @ W0 + b0)
    h = jax.nn.relu(h @ W1 + b1)
    return h @ W2 + b2

def loss_fn(params, x, y):
    logits = forward(params, x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(logp[jnp.arange(x.shape[0]), y])

@jit
def train_step(params, x, y, lr):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    new_params = tuple(p - lr*g for p,g in zip(params, grads))
    return new_params, loss

BS = 128
EPOCHS = 12
LR = 0.1
n = Xtr.shape[0]
print(f"Training: bs={BS}, epochs={EPOCHS}, lr={LR}")
t0 = time.time()
for ep in range(EPOCHS):
    perm = np.random.default_rng(ep).permutation(n)
    losses = []
    for i in range(0, n - BS + 1, BS):
        idx = perm[i:i+BS]
        params, loss = train_step(params, Xtr[idx], Ytr[idx], LR)
        losses.append(float(loss))
    # quick test accuracy per epoch
    logits = forward(params, Xte)
    acc = float((jnp.argmax(logits, -1) == Yte).mean())
    print(f"  epoch {ep+1:2d}: loss={np.mean(losses):.4f}  test_acc={acc*100:.2f}%")
print(f"train time: {time.time()-t0:.1f}s")

W0,b0,W1,b1,W2,b2 = [np.asarray(p) for p in params]
jax_logits_full = np.asarray(forward(params, Xte))
jax_preds = jax_logits_full.argmax(-1)
jax_acc = (jax_preds == Yte).mean()
print(f"JAX final test accuracy: {jax_acc*100:.4f}%")

# ---------- save weights + inputs for IREE ----------
os.makedirs("mlir_poc/inputs_trained", exist_ok=True)
for name, arr in [("W0",W0),("b0",b0),("W1",W1),("b1",b1),("W2",W2),("b2",b2)]:
    np.save(f"mlir_poc/inputs_trained/{name}.npy", arr.astype(np.float32))

# ---------- run IREE forward on test set, batch 128 ----------
n_test = Xte.shape[0]
n_batches = n_test // BS  # drop partial batch (10000/128 = 78.125 -> 78 full batches = 9984)
print(f"\nIREE forward: {n_batches} batches x {BS} = {n_batches*BS} test samples")

iree_preds = np.zeros(n_batches*BS, dtype=np.int64)
t0 = time.time()
for bi in range(n_batches):
    x_batch = Xte[bi*BS:(bi+1)*BS].astype(np.float32)
    np.save("/tmp/iree_x.npy", x_batch)
    cmd = [
        ".venv/bin/iree-run-module",
        "--module=.lake/build/mnist_mlp.vmfb",
        "--device=cuda",
        "--function=forward",
        "--input=@/tmp/iree_x.npy",
        "--input=@mlir_poc/inputs_trained/W0.npy",
        "--input=@mlir_poc/inputs_trained/b0.npy",
        "--input=@mlir_poc/inputs_trained/W1.npy",
        "--input=@mlir_poc/inputs_trained/b1.npy",
        "--input=@mlir_poc/inputs_trained/W2.npy",
        "--input=@mlir_poc/inputs_trained/b2.npy",
        "--output=@/tmp/iree_logits.npy",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"batch {bi} failed:", r.stderr.decode()[:500]); raise SystemExit(1)
    logits_i = np.load("/tmp/iree_logits.npy")
    iree_preds[bi*BS:(bi+1)*BS] = logits_i.argmax(-1)
print(f"IREE inference time: {time.time()-t0:.1f}s ({n_batches} subprocess invocations)")

iree_acc = (iree_preds == Yte[:n_batches*BS]).mean()
jax_trunc_acc = (jax_preds[:n_batches*BS] == Yte[:n_batches*BS]).mean()

print(f"\n=== Results (first {n_batches*BS} test samples) ===")
print(f"JAX  accuracy:  {jax_trunc_acc*100:.4f}%")
print(f"IREE accuracy:  {iree_acc*100:.4f}%")
print(f"prediction diff: {(iree_preds != jax_preds[:n_batches*BS]).sum()} / {n_batches*BS}")
assert abs(jax_trunc_acc - iree_acc) < 0.005, "Accuracy diverges too much"
print("PASS")
