

function update!(layer::DenseLayer)
    update!(layer, layer.optimizer)
end

