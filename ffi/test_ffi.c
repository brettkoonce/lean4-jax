// Smoke test for iree_ffi.c — load mnist_mlp.vmfb, run forward, print logits.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct iree_ffi_session_t iree_ffi_session_t;
iree_ffi_session_t* iree_ffi_session_create(const char* vmfb_path);
void iree_ffi_session_release(iree_ffi_session_t* sess);
int iree_ffi_invoke_f32(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int n_inputs,
    const int32_t* input_ranks,
    const int64_t* input_dims_flat,
    const float* const* input_data,
    int n_outputs,
    const int64_t* output_totals,
    float* const* output_data);

// Read raw float32 bytes from a .npy file (skipping the header).
// NPY layout: "\x93NUMPY" + 2 bytes version + 2 bytes header len + header str + data
static float* read_npy_f32(const char* path, long expected_floats) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "open %s failed\n", path); return NULL; }
  unsigned char magic[10];
  if (fread(magic, 1, 10, f) != 10) { fclose(f); return NULL; }
  if (memcmp(magic, "\x93NUMPY", 6) != 0) {
    fprintf(stderr, "bad npy magic in %s\n", path); fclose(f); return NULL;
  }
  // major/minor at magic[6..7], header_len at magic[8..9] (little endian, v1) or 4 bytes (v2)
  int header_len;
  if (magic[6] == 1) {
    header_len = magic[8] | (magic[9] << 8);
  } else {
    unsigned char hl4[4]; hl4[0] = magic[8]; hl4[1] = magic[9];
    if (fread(hl4+2, 1, 2, f) != 2) { fclose(f); return NULL; }
    header_len = hl4[0] | (hl4[1] << 8) | (hl4[2] << 16) | (hl4[3] << 24);
  }
  fseek(f, header_len, SEEK_CUR);  // skip header string
  float* data = malloc(expected_floats * sizeof(float));
  size_t got = fread(data, sizeof(float), expected_floats, f);
  fclose(f);
  if ((long)got != expected_floats) {
    fprintf(stderr, "read %ld of %ld floats from %s\n", (long)got, expected_floats, path);
    free(data); return NULL;
  }
  return data;
}

int main() {
  const char* vmfb = ".lake/build/mnist_mlp.vmfb";
  fprintf(stderr, "Creating session with %s...\n", vmfb);
  iree_ffi_session_t* sess = iree_ffi_session_create(vmfb);
  if (!sess) { fprintf(stderr, "session_create failed\n"); return 1; }
  fprintf(stderr, "Session ready.\n");

  // Read the test inputs we already generated in mlir_poc/inputs/
  float* x  = read_npy_f32("mlir_poc/inputs/x.npy",   128*784);
  float* W0 = read_npy_f32("mlir_poc/inputs/W1.npy",  784*512);  // note: our test files use W1/W2/W3
  float* b0 = read_npy_f32("mlir_poc/inputs/b1.npy",     512);
  float* W1 = read_npy_f32("mlir_poc/inputs/W2.npy",  512*512);
  float* b1 = read_npy_f32("mlir_poc/inputs/b2.npy",     512);
  float* W2 = read_npy_f32("mlir_poc/inputs/W3.npy",  512*10);
  float* b2 = read_npy_f32("mlir_poc/inputs/b3.npy",      10);
  if (!x || !W0 || !b0 || !W1 || !b1 || !W2 || !b2) {
    fprintf(stderr, "failed to read inputs\n"); return 2;
  }

  float* logits = malloc(128*10*sizeof(float));

  // Input descriptions: 7 tensors (x, W0, b0, W1, b1, W2, b2)
  int32_t ranks[7]       = {2,    2,     1,    2,     1,    2,     1};
  int64_t dims[] = {
    128, 784,   // x
    784, 512,   // W0
    512,        // b0
    512, 512,   // W1
    512,        // b1
    512, 10,    // W2
    10,         // b2
  };
  const float* inputs[7] = {x, W0, b0, W1, b1, W2, b2};
  int64_t out_totals[1]  = {128*10};
  float* outputs[1]      = {logits};

  fprintf(stderr, "Invoking forward...\n");
  int rc = iree_ffi_invoke_f32(sess, "mnist_mlp.forward",
                               7, ranks, dims, inputs,
                               1, out_totals, outputs);
  if (rc != 0) { fprintf(stderr, "invoke failed rc=%d\n", rc); return rc; }
  fprintf(stderr, "Invoke done.\n");

  // Print first 10 logits of first sample
  printf("logits[0,:] =");
  for (int i = 0; i < 10; i++) printf(" %.6f", logits[i]);
  printf("\n");

  // Benchmark: how fast can we go through 78 batches (matching earlier comparison)?
  #include <time.h>
  struct timespec t0, t1;
  int N = 78;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; i++) {
    rc = iree_ffi_invoke_f32(sess, "mnist_mlp.forward",
                             7, ranks, dims, inputs,
                             1, out_totals, outputs);
    if (rc != 0) { fprintf(stderr, "loop invoke %d failed rc=%d\n", i, rc); return rc; }
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  printf("FFI: %d forward calls in %.3fs = %.2f ms/call\n",
         N, elapsed, elapsed * 1000.0 / N);

  free(x); free(W0); free(b0); free(W1); free(b1); free(W2); free(b2); free(logits);
  iree_ffi_session_release(sess);
  return 0;
}
