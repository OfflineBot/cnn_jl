

struct Activation 
    activation::Function
    derivative::Function
end

include("./relu.jl")
include("./sigmoid.jl")
include("./softmax.jl")
include("./tanh.jl")

