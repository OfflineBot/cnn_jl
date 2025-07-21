

_relu_func(x::AbstractArray{Float32})::AbstractArray{Float32} = 
    ifelse.(x .<= 0.f0, 0.f0, x)

_deriv_relu_func(x::AbstractArray{Float32})::AbstractArray{Float32} = 
    ifelse.(x .<= 0.f0, 0.f0, 1.f0)


relu = Activation(_relu_func, _deriv_relu_func)

