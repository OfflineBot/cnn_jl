
import Distributions

mutable struct Conv2d
    # Shape: out_channel, in_channel, height, width
    kernel::Array{Float32, 4}
    # Shape: out_channel
    bias::Vector{Float32}

    # Shape: batch_size, in_channel, height, width
    input::Union{Array{Float32, 4}, Nothing}
    z::Union{Array{Float32, 4}, Nothing}
    a::Union{Array{Float32, 4}, Nothing}

    grad_kernel::Array{Float32, 4}
    grad_bias::Vector{Float32}

    padding::Padding
    stride::Int

    activation::Activation
    optim::Optim


    function Conv2d(kernel::Array{Float32, 4}, bias::Vector{Float32}, padding::Int, stride::Int, activation::Activation, optim::Optim)::Conv2d

        k_shape = size(kernel)
        b_shape = size(bias)
        grad_kernel = zeros(k_shape)
        grad_bias = zeros(b_shape)
        padding_struct = Padding(padding)

        return new(
            kernel,         # kernel
            bias,           # bias
            nothing,        # input
            nothing,        # z
            nothing,        # a
            grad_kernel,    # grad_kernel
            grad_bias,      # grad_bias
            padding_struct, # padding
            stride,         # stride
            activation,     # activation function
            optim           # optimizer
        )

    end
end

function Conv2d(in_channel::Int, out_channel::Int, height::Int, width::Int, padding::Int, stride::Int, activation::Activation, optim::Optim)
    distr = Distributions.Uniform(-1.f0, 1.f0)
    kernel = rand(distr, in_channel, out_channel, height, width)
    bias = rand(distr, out_channel)
    return Conv2d(kernel, bias, padding, stride, activation, optim)
end

