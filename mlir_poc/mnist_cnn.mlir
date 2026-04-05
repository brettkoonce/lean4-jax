// MNIST CNN smoke test: 2 conv layers + max pool + flatten + 3 dense.
// Matches MainCnn.lean architecture:
//   conv2d  1 32 3 .same .relu      (4, 1, 28, 28) -> (4, 32, 28, 28)
//   conv2d 32 32 3 .same .relu                    -> (4, 32, 28, 28)
//   maxPool 2 2                                   -> (4, 32, 14, 14)
//   flatten                                       -> (4, 6272)
//   dense 6272 512 .relu                          -> (4, 512)
//   dense  512 512 .relu                          -> (4, 512)
//   dense  512  10 .identity                      -> (4, 10)
//
// Layout convention (matches existing JAX codegen): NCHW inputs, OIHW kernels.
// Batch = 4 for the smoke test.

module @mnist_cnn {
  func.func @forward(
      %x:  tensor<4x1x28x28xf32>,
      %W0: tensor<32x1x3x3xf32>,  %b0: tensor<32xf32>,
      %W1: tensor<32x32x3x3xf32>, %b1: tensor<32xf32>,
      %W2: tensor<6272x512xf32>,  %b2: tensor<512xf32>,
      %W3: tensor<512x512xf32>,   %b3: tensor<512xf32>,
      %W4: tensor<512x10xf32>,    %b4: tensor<10xf32>
    ) -> tensor<4x10xf32> {

    // ---- conv 1: 1 -> 32, 3x3, SAME, stride 1 ----
    %c0 = "stablehlo.convolution"(%x, %W0) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0,
          kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0,
          output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]
        >,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<4x1x28x28xf32>, tensor<32x1x3x3xf32>) -> tensor<4x32x28x28xf32>
    // broadcast bias (32,) -> (4, 32, 28, 28) on channel dim 1
    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1]
         : (tensor<32xf32>) -> tensor<4x32x28x28xf32>
    %c0a = stablehlo.add %c0, %b0b : tensor<4x32x28x28xf32>
    %z0  = stablehlo.constant dense<0.0> : tensor<4x32x28x28xf32>
    %h0  = stablehlo.maximum %c0a, %z0 : tensor<4x32x28x28xf32>

    // ---- conv 2: 32 -> 32, 3x3, SAME, stride 1 ----
    %c1 = "stablehlo.convolution"(%h0, %W1) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0,
          input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0,
          kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0,
          output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]
        >,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<4x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<4x32x28x28xf32>
    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1]
         : (tensor<32xf32>) -> tensor<4x32x28x28xf32>
    %c1a = stablehlo.add %c1, %b1b : tensor<4x32x28x28xf32>
    %z1  = stablehlo.constant dense<0.0> : tensor<4x32x28x28xf32>
    %h1  = stablehlo.maximum %c1a, %z1 : tensor<4x32x28x28xf32>

    // ---- max pool 2x2 stride 2: (4,32,28,28) -> (4,32,14,14) ----
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %pool = "stablehlo.reduce_window"(%h1, %neg_inf) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %m = stablehlo.maximum %a, %b : tensor<f32>
        "stablehlo.return"(%m) : (tensor<f32>) -> ()
      }) {
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides    = array<i64: 1, 1, 2, 2>
      } : (tensor<4x32x28x28xf32>, tensor<f32>) -> tensor<4x32x14x14xf32>

    // ---- flatten: (4, 32, 14, 14) -> (4, 6272) ----
    %flat = stablehlo.reshape %pool
          : (tensor<4x32x14x14xf32>) -> tensor<4x6272xf32>

    // ---- dense 6272 -> 512 + relu ----
    %d0 = stablehlo.dot_general %flat, %W2,
            contracting_dims = [1] x [0],
            precision = [DEFAULT, DEFAULT]
          : (tensor<4x6272xf32>, tensor<6272x512xf32>) -> tensor<4x512xf32>
    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1]
         : (tensor<512xf32>) -> tensor<4x512xf32>
    %d0a = stablehlo.add %d0, %b2b : tensor<4x512xf32>
    %z2  = stablehlo.constant dense<0.0> : tensor<4x512xf32>
    %h2  = stablehlo.maximum %d0a, %z2 : tensor<4x512xf32>

    // ---- dense 512 -> 512 + relu ----
    %d1 = stablehlo.dot_general %h2, %W3,
            contracting_dims = [1] x [0],
            precision = [DEFAULT, DEFAULT]
          : (tensor<4x512xf32>, tensor<512x512xf32>) -> tensor<4x512xf32>
    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1]
         : (tensor<512xf32>) -> tensor<4x512xf32>
    %d1a = stablehlo.add %d1, %b3b : tensor<4x512xf32>
    %z3  = stablehlo.constant dense<0.0> : tensor<4x512xf32>
    %h3  = stablehlo.maximum %d1a, %z3 : tensor<4x512xf32>

    // ---- dense 512 -> 10 + identity ----
    %d2 = stablehlo.dot_general %h3, %W4,
            contracting_dims = [1] x [0],
            precision = [DEFAULT, DEFAULT]
          : (tensor<4x512xf32>, tensor<512x10xf32>) -> tensor<4x10xf32>
    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1]
         : (tensor<10xf32>) -> tensor<4x10xf32>
    %out = stablehlo.add %d2, %b4b : tensor<4x10xf32>

    return %out : tensor<4x10xf32>
  }
}
