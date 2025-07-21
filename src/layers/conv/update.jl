
function update!(layer::Conv2d)
    update!(layer, layer.optim)
end

