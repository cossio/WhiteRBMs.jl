"""
    shift_fields(layer, offset)

Adds `offset` to the `layer` fields.
"""
function shift_fields end

shift_fields(layer::Binary, offset::AbstractArray) = Binary(layer.θ + offset)
shift_fields(layer::Spin, offset::AbstractArray) = Spin(layer.θ + offset)
shift_fields(layer::Potts, offset::AbstractArray) = Potts(layer.θ + offset)
shift_fields(layer::Gaussian, offset::AbstractArray) = Gaussian(layer.θ + offset, layer.γ)
shift_fields(layer::ReLU, offset::AbstractArray) = ReLU(layer.θ + offset, layer.γ)

function shift_fields(layer::dReLU, offset::AbstractArray)
    return dReLU(layer.θp + offset, layer.θn + offset, layer.γp, layer.γn)
end

function shift_fields(layer::pReLU, offset::AbstractArray)
    return pReLU(layer.θ + offset, layer.γ, layer.Δ, layer.η)
end

function shift_fields(layer::xReLU, offset::AbstractArray)
    return xReLU(layer.θ + offset, layer.γ, layer.Δ, layer.ξ)
end
