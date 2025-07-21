
mutable struct Adam{TW, TB} <: Optim
    m_weight::TW    # first moment for weights
    v_weight::TW    # second moment for weights
    m_bias::TB      # first moment for bias
    v_bias::TB      # second moment for bias
    t::Int
    alpha::Float32
    beta1::Float32
    beta2::Float32
    etha::Float32

    function Adam(weight::TW, bias::TB, alpha=0.001f0, beta1=0.9f0, beta2=0.999f0, etha=1e-8) where {TW, TB}
        m_weigth = zeros(eltype(weight), size(weight))
        v_weigth = zeros(eltype(weight), size(weight))
        m_bias = zeros(eltype(bias), size(bias))
        v_bias = zeros(eltype(bias), size(bias))
        new(
            m_weigth,
            v_weigth,
            m_bias,
            v_bias,
            0,
            alpha,
            beta1,
            beta2,
            etha
        )
    end

end


