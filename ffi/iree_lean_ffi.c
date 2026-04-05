// Lean FFI shim for the IREE runtime wrapper.
// Converts between Lean's FloatArray (Float64) and IREE's expected float32.
//
// Exports:
//   lean_iree_session_create(path: String, world) : IO IreeSession
//   lean_iree_mlp_forward(sess, x, W0, b0, W1, b1, W2, b2, batch, world) : IO FloatArray

#include <lean/lean.h>
#include <stdlib.h>
#include <string.h>
#include "iree_ffi.h"

// ---- External class for IreeSession ----
static lean_external_class* g_iree_session_class = NULL;

static void iree_session_finalize(void* p) {
  iree_ffi_session_release((iree_ffi_session_t*)p);
}
static void iree_session_foreach(void* p, b_lean_obj_arg f) { (void)p; (void)f; }

static void ensure_iree_session_class(void) {
  if (!g_iree_session_class) {
    g_iree_session_class = lean_register_external_class(
        iree_session_finalize, iree_session_foreach);
  }
}

// ---- Session create ----
LEAN_EXPORT lean_obj_res lean_iree_session_create(
    b_lean_obj_arg path_obj, lean_obj_arg world) {
  (void)world;
  ensure_iree_session_class();
  const char* path = lean_string_cstr(path_obj);
  iree_ffi_session_t* sess = iree_ffi_session_create(path);
  if (!sess) {
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string("iree_ffi_session_create failed (see stderr)")));
  }
  return lean_io_result_mk_ok(
      lean_alloc_external(g_iree_session_class, sess));
}

// ---- Helpers: Float64 FloatArray → float32 staging buffer ----
static float* fa_to_f32(b_lean_obj_arg fa, size_t n) {
  double const* src = lean_float_array_cptr(fa);
  float* dst = (float*)malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
  return dst;
}

// ---- MLP forward ----
// Inputs (Lean-side): Float64 FloatArrays with the expected sizes.
// Hard-coded for the MNIST MLP shape: 784 -> 512 -> 512 -> 10, static batch.
LEAN_EXPORT lean_obj_res lean_iree_mlp_forward(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg x,
    b_lean_obj_arg W0, b_lean_obj_arg b0,
    b_lean_obj_arg W1, b_lean_obj_arg b1,
    b_lean_obj_arg W2, b_lean_obj_arg b2,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);

  // Float64 → Float32 staging buffers.
  float* x_f  = fa_to_f32(x,  batch * 784);
  float* W0_f = fa_to_f32(W0, 784 * 512);
  float* b0_f = fa_to_f32(b0, 512);
  float* W1_f = fa_to_f32(W1, 512 * 512);
  float* b1_f = fa_to_f32(b1, 512);
  float* W2_f = fa_to_f32(W2, 512 * 10);
  float* b2_f = fa_to_f32(b2, 10);

  size_t logits_n = batch * 10;
  float* logits_f = (float*)malloc(logits_n * sizeof(float));

  int32_t ranks[7] = {2, 2, 1, 2, 1, 2, 1};
  int64_t dims[11] = {
      (int64_t)batch, 784,   // x
      784, 512,              // W0
      512,                   // b0
      512, 512,              // W1
      512,                   // b1
      512, 10,               // W2
      10,                    // b2
  };
  const float* inputs[7] = {x_f, W0_f, b0_f, W1_f, b1_f, W2_f, b2_f};
  int64_t out_totals[1] = {(int64_t)logits_n};
  float* outputs[1] = {logits_f};

  int rc = iree_ffi_invoke_f32(sess, "mnist_mlp.forward",
                               7, ranks, dims, inputs,
                               1, out_totals, outputs);

  free(x_f); free(W0_f); free(b0_f); free(W1_f);
  free(b1_f); free(W2_f); free(b2_f);

  if (rc != 0) {
    free(logits_f);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string("iree_ffi_invoke_f32 failed (see stderr)")));
  }

  // Copy float32 back into a Float64 FloatArray.
  lean_object* result = lean_alloc_sarray(sizeof(double), logits_n, logits_n);
  double* rp = lean_float_array_cptr(result);
  for (size_t i = 0; i < logits_n; i++) rp[i] = (double)logits_f[i];
  free(logits_f);

  return lean_io_result_mk_ok(result);
}

// ---- MLP train step ----
// Inputs (Lean-side):
//   sess_obj  — IreeSession
//   params_fa — FloatArray f64, length 669706 (W0|b0|W1|b1|W2|b2 packed)
//   x_fa      — FloatArray f64, length batch*784
//   y_ba      — ByteArray, batch*4 bytes (packed int32 LE labels)
//   lr        — Float (double → f32 at boundary)
//   batch     — USize
// Output: FloatArray f64, length 669707 (new params + loss at index 669706)
LEAN_EXPORT lean_obj_res lean_iree_mlp_train_step(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg params_fa,
    b_lean_obj_arg x_fa,
    b_lean_obj_arg y_ba,
    double lr,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);

  const size_t N_W0 = 784*512, N_b0 = 512;
  const size_t N_W1 = 512*512, N_b1 = 512;
  const size_t N_W2 = 512*10,  N_b2 = 10;
  const size_t N_P  = N_W0+N_b0+N_W1+N_b1+N_W2+N_b2;  // 669706

  // Stage params: f64 → f32
  double const* p_src = lean_float_array_cptr(params_fa);
  float* p_f = (float*)malloc(N_P * sizeof(float));
  for (size_t i = 0; i < N_P; i++) p_f[i] = (float)p_src[i];
  float* W0_p = p_f;
  float* b0_p = W0_p + N_W0;
  float* W1_p = b0_p + N_b0;
  float* b1_p = W1_p + N_W1;
  float* W2_p = b1_p + N_b1;
  float* b2_p = W2_p + N_W2;

  // Stage x: f64 → f32
  size_t N_x = batch * 784;
  double const* x_src = lean_float_array_cptr(x_fa);
  float* x_f = (float*)malloc(N_x * sizeof(float));
  for (size_t i = 0; i < N_x; i++) x_f[i] = (float)x_src[i];

  // y labels: ByteArray → int32*
  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_ba);

  // Output buffers
  float* W0o = (float*)malloc(N_W0 * sizeof(float));
  float* b0o = (float*)malloc(N_b0 * sizeof(float));
  float* W1o = (float*)malloc(N_W1 * sizeof(float));
  float* b1o = (float*)malloc(N_b1 * sizeof(float));
  float* W2o = (float*)malloc(N_W2 * sizeof(float));
  float* b2o = (float*)malloc(N_b2 * sizeof(float));
  float loss_f = 0.0f;

  int rc = iree_ffi_train_step_mlp(
      sess, "jit_train_step.main", (int)batch,
      W0_p, b0_p, W1_p, b1_p, W2_p, b2_p,
      x_f, y_ptr, (float)lr,
      W0o, b0o, W1o, b1o, W2o, b2o, &loss_f);

  free(p_f); free(x_f);
  if (rc != 0) {
    free(W0o); free(b0o); free(W1o); free(b1o); free(W2o); free(b2o);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("iree_ffi_train_step_mlp failed")));
  }

  // Pack new params + loss into a single FloatArray (f64).
  size_t N_out = N_P + 1;
  lean_object* result = lean_alloc_sarray(sizeof(double), N_out, N_out);
  double* rp = lean_float_array_cptr(result);
  const float* outs[6] = {W0o, b0o, W1o, b1o, W2o, b2o};
  const size_t sizes[6] = {N_W0, N_b0, N_W1, N_b1, N_W2, N_b2};
  size_t off = 0;
  for (int k = 0; k < 6; k++) {
    for (size_t i = 0; i < sizes[k]; i++) rp[off+i] = (double)outs[k][i];
    off += sizes[k];
  }
  rp[N_P] = (double)loss_f;

  free(W0o); free(b0o); free(W1o); free(b1o); free(W2o); free(b2o);
  return lean_io_result_mk_ok(result);
}
