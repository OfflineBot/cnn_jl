

mse(pred::Matrix{Float32}, truth::Matrix{Float32})::Float32 =
    sum((pred .- truth).^2) / length(pred)

mse_backward(pred::Matrix{Float32}, truth::Matrix{Float32})::Matrix{Float32} =
    pred .- truth

