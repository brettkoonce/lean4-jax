// Tiny MLP smoke test: single dense layer 4 -> 3 + ReLU.
// y = relu(x @ W + b) where x: (1,4), W: (4,3), b: (3,)
//
// Validates StableHLO ops we'll need for MNIST MLP:
//   stablehlo.dot_general, stablehlo.broadcast_in_dim,
//   stablehlo.add, stablehlo.maximum, stablehlo.constant

module @tiny_mlp {
  func.func @forward(%x: tensor<1x4xf32>, %W: tensor<4x3xf32>, %b: tensor<3xf32>) -> tensor<1x3xf32> {
    // matmul: x @ W  -> (1,3)
    %matmul = stablehlo.dot_general %x, %W,
                contracting_dims = [1] x [0],
                precision = [DEFAULT, DEFAULT]
              : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>

    // broadcast bias (3,) -> (1,3) along dim 1
    %b_bcast = stablehlo.broadcast_in_dim %b, dims = [1]
             : (tensor<3xf32>) -> tensor<1x3xf32>

    // add
    %added = stablehlo.add %matmul, %b_bcast : tensor<1x3xf32>

    // relu via max(x, 0)
    %zero = stablehlo.constant dense<0.0> : tensor<1x3xf32>
    %out = stablehlo.maximum %added, %zero : tensor<1x3xf32>

    return %out : tensor<1x3xf32>
  }
}
