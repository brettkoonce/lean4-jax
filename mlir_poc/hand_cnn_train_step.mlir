// Hand-written MNIST CNN train_step: forward + softmax-CE + manual VJPs + SGD.
// Architecture: conv(1→32) relu conv(32→32) relu maxpool flatten
//               dense(6272→512) relu dense(512→512) relu dense(512→10)
// All NCHW / OIHW. Batch 128.
//
// Conv backward dW uses the transpose-conv trick to avoid non-standard
// dim_numbers that IREE can't compile (see iree-org/iree#21955).
// Pool backward uses stablehlo.select_and_scatter.

module @jit_cnn_train_step {
  func.func @main(
      %W0: tensor<32x1x3x3xf32>,    %b0: tensor<32xf32>,
      %W1: tensor<32x32x3x3xf32>,   %b1: tensor<32xf32>,
      %W2: tensor<6272x512xf32>,     %b2: tensor<512xf32>,
      %W3: tensor<512x512xf32>,      %b3: tensor<512xf32>,
      %W4: tensor<512x10xf32>,       %b4: tensor<10xf32>,
      %x_flat: tensor<128x784xf32>,  %y:  tensor<128xi32>,
      %lr: tensor<f32>
    ) -> (tensor<32x1x3x3xf32>, tensor<32xf32>,
          tensor<32x32x3x3xf32>, tensor<32xf32>,
          tensor<6272x512xf32>, tensor<512xf32>,
          tensor<512x512xf32>, tensor<512xf32>,
          tensor<512x10xf32>, tensor<10xf32>,
          tensor<f32>) {

    %zf = stablehlo.constant dense<0.0> : tensor<f32>
    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>

    // ======================== FORWARD ========================
    // Reshape flat input → NCHW
    %x = stablehlo.reshape %x_flat : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>

    // Conv 1: 1→32, 3×3 SAME + bias + ReLU
    %cv0 = "stablehlo.convolution"(%x, %W0) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x1x28x28xf32>, tensor<32x1x3x3xf32>) -> tensor<128x32x28x28xf32>
    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %h0pre = stablehlo.add %cv0, %b0b : tensor<128x32x28x28xf32>
    %z28 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %h0 = stablehlo.maximum %h0pre, %z28 : tensor<128x32x28x28xf32>

    // Conv 2: 32→32, 3×3 SAME + bias + ReLU
    %cv1 = "stablehlo.convolution"(%h0, %W1) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %h1pre = stablehlo.add %cv1, %b1b : tensor<128x32x28x28xf32>
    %h1 = stablehlo.maximum %h1pre, %z28 : tensor<128x32x28x28xf32>

    // MaxPool 2×2 stride 2: (128,32,28,28) → (128,32,14,14)
    %pool = "stablehlo.reduce_window"(%h1, %neginf) ({
      ^bb0(%a0: tensor<f32>, %b0_: tensor<f32>):
        %m0 = stablehlo.maximum %a0, %b0_ : tensor<f32>
        "stablehlo.return"(%m0) : (tensor<f32>) -> ()
      }) {window_dimensions = array<i64: 1, 1, 2, 2>,
          window_strides = array<i64: 1, 1, 2, 2>}
      : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<128x32x14x14xf32>

    // Flatten: (128,32,14,14) → (128,6272)
    %flat = stablehlo.reshape %pool : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>

    // Dense 6272→512 + ReLU
    %mm2 = stablehlo.dot_general %flat, %W2, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x6272xf32>, tensor<6272x512xf32>) -> tensor<128x512xf32>
    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %d0pre = stablehlo.add %mm2, %b2b : tensor<128x512xf32>
    %z512 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %d0 = stablehlo.maximum %d0pre, %z512 : tensor<128x512xf32>

    // Dense 512→512 + ReLU
    %mm3 = stablehlo.dot_general %d0, %W3, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %d1pre = stablehlo.add %mm3, %b3b : tensor<128x512xf32>
    %d1 = stablehlo.maximum %d1pre, %z512 : tensor<128x512xf32>

    // Dense 512→10
    %mm4 = stablehlo.dot_general %d1, %W4, contracting_dims = [1] x [0],
             precision = [DEFAULT, DEFAULT]
           : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %logits = stablehlo.add %mm4, %b4b : tensor<128x10xf32>

    // ================ SOFTMAX CROSS-ENTROPY ================
    %maxv = stablehlo.reduce(%logits init: %neginf) applies stablehlo.maximum across dimensions = [1]
          : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %maxv_b = stablehlo.broadcast_in_dim %maxv, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %shifted = stablehlo.subtract %logits, %maxv_b : tensor<128x10xf32>
    %exp_s = stablehlo.exponential %shifted : tensor<128x10xf32>
    %sum_e = stablehlo.reduce(%exp_s init: %zf) applies stablehlo.add across dimensions = [1]
           : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %log_s = stablehlo.log %sum_e : tensor<128xf32>
    %log_s_b = stablehlo.broadcast_in_dim %log_s, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %log_p = stablehlo.subtract %shifted, %log_s_b : tensor<128x10xf32>

    %iota = stablehlo.iota dim = 1 : tensor<128x10xi32>
    %y_b = stablehlo.broadcast_in_dim %y, dims = [0] : (tensor<128xi32>) -> tensor<128x10xi32>
    %mask = stablehlo.compare EQ, %iota, %y_b : (tensor<128x10xi32>, tensor<128x10xi32>) -> tensor<128x10xi1>
    %onef = stablehlo.constant dense<1.0> : tensor<128x10xf32>
    %zerof = stablehlo.constant dense<0.0> : tensor<128x10xf32>
    %onehot = stablehlo.select %mask, %onef, %zerof : tensor<128x10xi1>, tensor<128x10xf32>

    %weighted = stablehlo.multiply %log_p, %onehot : tensor<128x10xf32>
    %total = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [0, 1]
           : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %Bc = stablehlo.constant dense<128.0> : tensor<f32>
    %mean = stablehlo.divide %total, %Bc : tensor<f32>
    %loss = stablehlo.negate %mean : tensor<f32>

    // ==================== BACKWARD ====================
    // d_logits = (softmax - onehot) / B
    %sum_e_b = stablehlo.broadcast_in_dim %sum_e, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %softmax = stablehlo.divide %exp_s, %sum_e_b : tensor<128x10xf32>
    %sm_moh = stablehlo.subtract %softmax, %onehot : tensor<128x10xf32>
    %Bc_10 = stablehlo.broadcast_in_dim %Bc, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
    %d_logits = stablehlo.divide %sm_moh, %Bc_10 : tensor<128x10xf32>

    // Dense 512→10 backward
    %d_W4 = stablehlo.dot_general %d1, %d_logits, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %d_b4 = stablehlo.reduce(%d_logits init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %d_d1 = stablehlo.dot_general %d_logits, %W4, contracting_dims = [1] x [1],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>

    // ReLU backward for d1
    %m_d1 = stablehlo.compare GT, %d1pre, %z512 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %d_d1pre = stablehlo.select %m_d1, %d_d1, %z512 : tensor<128x512xi1>, tensor<128x512xf32>

    // Dense 512→512 backward
    %d_W3 = stablehlo.dot_general %d0, %d_d1pre, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %d_b3 = stablehlo.reduce(%d_d1pre init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %d_d0 = stablehlo.dot_general %d_d1pre, %W3, contracting_dims = [1] x [1],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>

    // ReLU backward for d0
    %m_d0 = stablehlo.compare GT, %d0pre, %z512 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %d_d0pre = stablehlo.select %m_d0, %d_d0, %z512 : tensor<128x512xi1>, tensor<128x512xf32>

    // Dense 6272→512 backward
    %d_W2 = stablehlo.dot_general %flat, %d_d0pre, contracting_dims = [0] x [0],
              precision = [DEFAULT, DEFAULT]
            : (tensor<128x6272xf32>, tensor<128x512xf32>) -> tensor<6272x512xf32>
    %d_b2 = stablehlo.reduce(%d_d0pre init: %zf) applies stablehlo.add across dimensions = [0]
          : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %d_flat = stablehlo.dot_general %d_d0pre, %W2, contracting_dims = [1] x [1],
                precision = [DEFAULT, DEFAULT]
              : (tensor<128x512xf32>, tensor<6272x512xf32>) -> tensor<128x6272xf32>

    // Unflatten: (128, 6272) → (128, 32, 14, 14)
    %d_pool = stablehlo.reshape %d_flat : (tensor<128x6272xf32>) -> tensor<128x32x14x14xf32>

    // MaxPool backward via select_and_scatter
    %d_h1 = "stablehlo.select_and_scatter"(%h1, %d_pool, %zf) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %cmp = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %acc = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %acc : tensor<f32>
      }) {window_dimensions = array<i64: 1, 1, 2, 2>,
          window_strides = array<i64: 1, 1, 2, 2>}
      : (tensor<128x32x28x28xf32>, tensor<128x32x14x14xf32>, tensor<f32>) -> tensor<128x32x28x28xf32>

    // ReLU backward for h1 (conv 2 output)
    %m_h1 = stablehlo.compare GT, %h1pre, %z28 : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xi1>
    %d_h1pre = stablehlo.select %m_h1, %d_h1, %z28 : tensor<128x32x28x28xi1>, tensor<128x32x28x28xf32>

    // Conv 2 backward dW1: transpose trick
    //   x_t = transpose(h0, [1,0,2,3])  →  (32, 128, 28, 28)
    //   dy_t = transpose(d_h1pre, [1,0,2,3])  →  (32, 128, 28, 28)
    //   dW1_t = conv(x_t, dy_t, pad=1)  →  (32, 32, 3, 3)  standard dims
    %h0_t = stablehlo.transpose %h0, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %dh1p_t = stablehlo.transpose %d_h1pre, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %d_W1 = "stablehlo.convolution"(%h0_t, %dh1p_t) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x32x3x3xf32>

    // Conv 2 backward db1: sum over batch + spatial
    %d_b1 = stablehlo.reduce(%d_h1pre init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]
          : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>

    // Conv 2 backward dx (d_h0): reverse kernel + standard conv
    %W1_t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %W1_rev = stablehlo.reverse %W1_t, dims = [2, 3] : tensor<32x32x3x3xf32>
    %d_h0 = "stablehlo.convolution"(%d_h1pre, %W1_rev) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>

    // ReLU backward for h0 (conv 1 output)
    %m_h0 = stablehlo.compare GT, %h0pre, %z28 : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x32x28x28xi1>
    %d_h0pre = stablehlo.select %m_h0, %d_h0, %z28 : tensor<128x32x28x28xi1>, tensor<128x32x28x28xf32>

    // Conv 1 backward dW0: transpose trick
    //   x_t = transpose(x, [1,0,2,3])  →  (1, 128, 28, 28)
    //   dy_t = transpose(d_h0pre, [1,0,2,3])  →  (32, 128, 28, 28)
    //   dW0_raw = conv(x_t, dy_t, pad=1)  →  (1, 32, 3, 3)
    //   dW0 = transpose(dW0_raw, [1,0,2,3])  →  (32, 1, 3, 3) = OIHW
    %x_t = stablehlo.transpose %x, dims = [1, 0, 2, 3] : (tensor<128x1x28x28xf32>) -> tensor<1x128x28x28xf32>
    %dh0p_t = stablehlo.transpose %d_h0pre, dims = [1, 0, 2, 3] : (tensor<128x32x28x28xf32>) -> tensor<32x128x28x28xf32>
    %d_W0_raw = "stablehlo.convolution"(%x_t, %dh0p_t) {
        batch_group_count = 1 : i64,
        dimension_numbers = #stablehlo.conv<raw
          input_batch_dimension = 0, input_feature_dimension = 1,
          input_spatial_dimensions = [2, 3],
          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,
          kernel_spatial_dimensions = [2, 3],
          output_batch_dimension = 0, output_feature_dimension = 1,
          output_spatial_dimensions = [2, 3]>,
        feature_group_count = 1 : i64,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        rhs_dilation = array<i64: 1, 1>,
        window_strides = array<i64: 1, 1>
      } : (tensor<1x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<1x32x3x3xf32>
    %d_W0 = stablehlo.transpose %d_W0_raw, dims = [1, 0, 2, 3]
           : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>

    %d_b0 = stablehlo.reduce(%d_h0pre init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]
          : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<32xf32>

    // =================== SGD UPDATES ===================
    // W0 (32,1,3,3)
    %lr_W0 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x1x3x3xf32>
    %upW0 = stablehlo.multiply %lr_W0, %d_W0 : tensor<32x1x3x3xf32>
    %W0n = stablehlo.subtract %W0, %upW0 : tensor<32x1x3x3xf32>
    // b0 (32)
    %lr_b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %upb0 = stablehlo.multiply %lr_b, %d_b0 : tensor<32xf32>
    %b0n = stablehlo.subtract %b0, %upb0 : tensor<32xf32>
    // W1 (32,32,3,3)
    %lr_W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %upW1 = stablehlo.multiply %lr_W1, %d_W1 : tensor<32x32x3x3xf32>
    %W1n = stablehlo.subtract %W1, %upW1 : tensor<32x32x3x3xf32>
    // b1 (32)
    %upb1 = stablehlo.multiply %lr_b, %d_b1 : tensor<32xf32>
    %b1n = stablehlo.subtract %b1, %upb1 : tensor<32xf32>
    // W2 (6272,512)
    %lr_W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<6272x512xf32>
    %upW2 = stablehlo.multiply %lr_W2, %d_W2 : tensor<6272x512xf32>
    %W2n = stablehlo.subtract %W2, %upW2 : tensor<6272x512xf32>
    // b2 (512)
    %lr_b512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %upb2 = stablehlo.multiply %lr_b512, %d_b2 : tensor<512xf32>
    %b2n = stablehlo.subtract %b2, %upb2 : tensor<512xf32>
    // W3 (512,512)
    %lr_W3 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %upW3 = stablehlo.multiply %lr_W3, %d_W3 : tensor<512x512xf32>
    %W3n = stablehlo.subtract %W3, %upW3 : tensor<512x512xf32>
    // b3 (512)
    %upb3 = stablehlo.multiply %lr_b512, %d_b3 : tensor<512xf32>
    %b3n = stablehlo.subtract %b3, %upb3 : tensor<512xf32>
    // W4 (512,10)
    %lr_W4 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %upW4 = stablehlo.multiply %lr_W4, %d_W4 : tensor<512x10xf32>
    %W4n = stablehlo.subtract %W4, %upW4 : tensor<512x10xf32>
    // b4 (10)
    %lr_b10 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %upb4 = stablehlo.multiply %lr_b10, %d_b4 : tensor<10xf32>
    %b4n = stablehlo.subtract %b4, %upb4 : tensor<10xf32>

    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %loss
      : tensor<32x1x3x3xf32>, tensor<32xf32>,
        tensor<32x32x3x3xf32>, tensor<32xf32>,
        tensor<6272x512xf32>, tensor<512xf32>,
        tensor<512x512xf32>, tensor<512xf32>,
        tensor<512x10xf32>, tensor<10xf32>,
        tensor<f32>
  }
}
