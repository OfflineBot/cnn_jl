

struct SGD <: Optim
    # Learning Rate
    lr::Float32
end


function update!(layer::DenseLayer, optim::SGD)
    layer.weight = layer.weight .- (optim.lr .* layer.grad_weight)
    layer.bias = layer.bias .- (optim.lr .* layer.grad_bias)

    w_shape = size(layer.weight)
    b_shape = size(layer.bias)

    layer.grad_weight = zeros(Float32, w_shape)
    layer.grad_bias = zeros(Float32, b_shape)
end


function update!(layer::Conv2d, optim::SGD)
    layer.kernel = layer.kernel .- (optim.lr .* layer.grad_kernel)
    layer.bias = layer.bias .- (optim.lr .* layer.grad_bias)

    k_shape = size(layer.kernel)
    b_shape = size(layer.bias)

    layer.grad_kernel = zeros(k_shape)
    layer.grad_bias = zeros(b_shape)
end

