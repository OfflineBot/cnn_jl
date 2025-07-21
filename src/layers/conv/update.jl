
function update!(layer::Conv2d, optim::Optim)
    update!(layer, layer.optim)
end

