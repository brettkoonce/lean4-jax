// MNIST MLP: 784 -> 512 -> 512 -> 10 (matches `mnistMlp` in MainMlp.lean)
// Static batch size 128. Params passed as args (option b from the plan).
// Layers: dense+relu, dense+relu, dense+identity.

module @mnist_mlp {
  func.func @forward(
    %x:  tensor<128x784xf32>,
    %W1: tensor<784x512xf32>, %b1: tensor<512xf32>,
    %W2: tensor<512x512xf32>, %b2: tensor<512xf32>,
    %W3: tensor<512x10xf32>,  %b3: tensor<10xf32>
  ) -> tensor<128x10xf32> {

    // ---- layer 1: dense 784 -> 512 + relu ----
    %h1 = stablehlo.dot_general %x, %W1,
            contracting_dims = [1] x [0],
            precision = [DEFAULT, DEFAULT]
          : (tensor<128x784xf32>, tensor<784x512xf32>) -> tensor<128x512xf32>
    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1]
         : (tensor<512xf32>) -> tensor<128x512xf32>
    %h1_add = stablehlo.add %h1, %b1b : tensor<128x512xf32>
    %zero1 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %h1_relu = stablehlo.maximum %h1_add, %zero1 : tensor<128x512xf32>

    // ---- layer 2: dense 512 -> 512 + relu ----
    %h2 = stablehlo.dot_general %h1_relu, %W2,
            contracting_dims = [1] x [0],
            precision = [DEFAULT, DEFAULT]
          : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1]
         : (tensor<512xf32>) -> tensor<128x512xf32>
    %h2_add = stablehlo.add %h2, %b2b : tensor<128x512xf32>
    %zero2 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %h2_relu = stablehlo.maximum %h2_add, %zero2 : tensor<128x512xf32>

    // ---- layer 3: dense 512 -> 10 + identity ----
    %logits = stablehlo.dot_general %h2_relu, %W3,
                contracting_dims = [1] x [0],
                precision = [DEFAULT, DEFAULT]
              : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1]
         : (tensor<10xf32>) -> tensor<128x10xf32>
    %out = stablehlo.add %logits, %b3b : tensor<128x10xf32>

    return %out : tensor<128x10xf32>
  }
}
