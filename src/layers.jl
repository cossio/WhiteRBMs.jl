"""
    shift_fields(layer, offset)

Adds `offset` to the `layer` fields.
"""
function shift_fields end

shift_fields(layer::Binary, offset::AbstractArray) = Binary(; θ = layer.θ + offset)
shift_fields(layer::Spin, offset::AbstractArray) = Spin(; θ = layer.θ + offset)
shift_fields(layer::Potts, offset::AbstractArray) = Potts(; θ = layer.θ + offset)
shift_fields(layer::Gaussian, offset::AbstractArray) = Gaussian(; θ = layer.θ + offset, γ = layer.γ)
shift_fields(layer::ReLU, offset::AbstractArray) = ReLU(; θ = layer.θ + offset, γ = layer.γ)
shift_fields(layer::dReLU, offset::AbstractArray) = dReLU(
    ; θp = layer.θp + offset, θn = layer.θn + offset, γp = layer.γp, γn = layer.γn
)
shift_fields(layer::pReLU, offset::AbstractArray) = pReLU(
    ; θ = layer.θ + offset, γ = layer.γ, Δ = layer.Δ, η = layer.η
)
shift_fields(layer::xReLU, offset::AbstractArray) = xReLU(
    ; θ = layer.θ + offset, γ = layer.γ, Δ = layer.Δ, ξ = layer.ξ
)

# In-place versions

"""
    shift_fields!(layer, offset)

In-place version of `shift_fields(layer, offset)`.
"""
function shift_fields! end

function shift_fields!(
    layer::Union{Binary, Spin, Potts, Gaussian, ReLU, pReLU, xReLU}, offset::AbstractArray
)
    layer.θ .+= offset
    return layer
end

function shift_fields!(layer::dReLU, offset::AbstractArray)
    layer.θp .+= offset
    layer.θn .+= offset
    return layer
end
