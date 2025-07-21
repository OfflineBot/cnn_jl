
_sigmoid(x::AbstractArray{Float32})::AbstractArray{Float32} =
    1f0 ./ (1f0 .+ exp.(-x))

_deriv_sigmoid(x::AbstractArray{Float32})::AbstractArray{Float32} = 
    begin
        sig = _sigmoid(x)
        sig .* (1f0 .- sig)
    end

sigmoid = Activation(_sigmoid, _deriv_sigmoid)

