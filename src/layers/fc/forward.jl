
function forward!(layer::DenseLayer, input::Matrix{Float32})::Matrix{Float32}
    layer.input = input

    z = input * layer.weight .+ layer.bias
    layer.z = z

    a = layer.activation.activation(z)
    layer.a = a

    a
end

