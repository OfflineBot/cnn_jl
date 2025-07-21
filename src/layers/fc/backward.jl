
function backward!(layer::DenseLayer, delta::Matrix{Float32})::Matrix{Float32}

    delta .*= layer.activation.derivative(layer.z)

    layer.grad_weight .+= layer.input' * delta
    layer.grad_bias .+= sum(delta, dims=1)

    new_delta = delta * layer.weight'
    return new_delta 
end

