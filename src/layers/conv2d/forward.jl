
"""
Auto padding enabled by default
"""
function forward!(layer::Conv2d, input::Array{Float32, 4})::Array{Float32, 4}
    out_channel_size, in_channel_size, kernel_height, kernel_width = size(layer.kernel)
    batch_size, _, input_height, input_width = size(input)

    # stride check / auto padding handling
    output_height_pre_stride = (input_height + 2 * layer.padding.padding - kernel_height)
    output_width_pre_stride = (input_width + 2 * layer.padding.padding - kernel_width)

    output_height_stride_error = output_height_pre_stride % layer.stride
    output_width_stride_error = output_width_pre_stride % layer.stride

    if (output_height_stride_error != 0)
        error = floor(Int, output_height_stride_error / 2)
        layer.padding.top = error
        layer.padding.bottom = output_height_stride_error - error
    end
    if(output_width_stride_error != 0) 
        error = floor(Int, output_width_stride_error / 2)
        layer.padding.left = error
        layer.padding.right = output_width_stride_error - error
    end

    output_height = div(output_height_pre_stride, layer.stride)
    output_width = div(output_width_pre_stride, layer.stride)
    # end stride check

    @assert output_height >= 1 && output_width >= 1 "Output dimensions must be bigger than 1!"

    conv = zeros(batch_size, out_channel_size, output_height, output_width)
    for batch in 1:batch_size
        for out_c in 1:out_channel_size
            for in_c in 1:in_channel_size
                conv[batch, out_c, :, :] .+= 
                    convolution(
                        input[batch, in_c, :, :], 
                        layer.kernel[out_c, in_c, :, :], 
                        layer.padding, 
                        layer.stride,
                        (output_height, output_width)
                    )
            end
        end
    end
    layer.z = conv
    layer.a = layer.activation.activation(conv)
    conv
end


function convolution(input::Matrix{Float32}, kernel::Matrix{Float32}, padding::Padding, stride::Int, output_shape::Tuple{Int, Int})::Matrix{Float32}
    padded_input = apply_padding(input, padding)
    output_height, output_width = output_shape
    kernel_height, kernel_width = size(kernel)

    output = zeros(Float32, output_height, output_width)

    for i in 1:output_height
        for j in 1:output_width
            row_start = 1 + (i - 1) * stride
            col_start = 1 + (j - 1) * stride
            input_slice = padded_input[row_start:row_start+kernel_height-1, col_start:col_start+kernel_width-1]
            output[i, j] = sum(input_slice .* kernel)
        end
    end

    output
end


function apply_padding(input::Matrix{Float32}, padding::Padding)::Matrix{Float32}
    input_height, input_width = size(input)
    output_height = input_height + (2 * padding.padding) + padding.top + padding.bottom
    output_width = input_width + (2 * padding.padding) + padding.left + padding.right

    output = zeros(Float32, output_height, output_width)

    row_start = padding.padding + padding.top + 1
    row_end = padding.padding + padding.top + input_height
    col_start = padding.padding + padding.left + 1
    col_end = padding.padding + padding.left + input_width
    output[row_start:row_end, col_start:col_end] .= input
    output
end

