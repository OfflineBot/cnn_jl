

import Distributions

mutable struct DenseLayer 
    # Shape: input, output
    weight::Matrix{Float32}
    # Shape: 1, output
    bias::Matrix{Float32}

    input::Union{Matrix{Float32}, Nothing}
    z::Union{Matrix{Float32}, Nothing}
    a::Union{Matrix{Float32}, Nothing}

    grad_weight::Matrix{Float32}
    grad_bias::Matrix{Float32}

    activation::Activation

    optimizer::Optim


    # Function takes on Float32 as input and returns Float32
    function DenseLayer(weight::Matrix{Float32}, bias::Matrix{Float32}, activation::Activation, optimizer::Optim)::DenseLayer
        w_shape = size(weight)
        b_shape = size(bias)
        grad_weight = zeros(Float32, w_shape)
        grad_bias = zeros(Float32, b_shape)
        new(
            weight, 
            bias, 
            nothing, 
            nothing,
            nothing,
            grad_weight, 
            grad_bias,
            activation, 
            optimizer
        )
    end

end


# Function takes on Float32 as input and returns Float32
function DenseLayer(input_size::Int, output_size::Int, activation::Activation, optimizer::Optim)::DenseLayer 
    distr = Distributions.Uniform(-1.f0, 1.f0)
    weight::Matrix{Float32} = rand(distr, input_size, output_size)
    bias::Matrix{Float32} = rand(distr, 1, output_size)
    DenseLayer(weight, bias, activation, optimizer)
end

