function whiten_visible_from_data!(rbm::WhiteRBM, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    affine_v = visible_affine_from_data(rbm, data; wts, ϵ)
    return whiten_visible!(rbm, affine_v)
end

function whiten_hidden_from_inputs!(white_rbm::WhiteRBM, inputs::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    affine_h = hidden_affine_from_inputs(white_rbm, inputs; wts, damping, ϵ)
    return whiten_hidden!(white_rbm, affine_h)
end

function whiten_hidden_from_v!(white_rbm::WhiteRBM, v::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    if !(white_rbm.affine_h isa Identity) # only compute inputs if necessary
        inputs = inputs_h_from_v(rbm, v)
        whiten_hidden_from_inputs!(rbm, inputs; damping, wts, ϵ)
    end
    return white_rbm
end

function visible_affine_from_data(rbm::AffineRBM{<:CenterAffine,<:AbstractAffine}, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    μ = batchmean(rbm.visible, data; wts)
    return oftype(rbm.affine_v, CenterAffine(vec(μ)))
end

function visible_affine_from_data(rbm::AffineRBM{<:StdizeAffine,<:AbstractAffine}, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    u = vec(batchmean(rbm.visible, data; wts))
    C = Diagonal(vec(batchvar(rbm.visible, data; wts, mean=u)))
    affine_v = whitening_transform(u, Symmetric(C + ϵ * I))
    return oftype(rbm.affine_v, affine_v)
end

function visible_affine_from_data(rbm::AffineRBM{<:Affine,<:AbstractAffine}, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    u = vec(batchmean(rbm.visible, data; wts))
    C = batchcov(rbm.visible, data; wts, mean=μ)
    C_flat = reshape(C, length(rbm.visible), length(rbm.visible))
    affine_v = whitening_transform(u, Symmetric(C_flat + ϵ * I))
    return oftype(rbm.affine_v, affine_v)
end

function hidden_affine_from_inputs(
    rbm::AffineRBM{<:AbstractAffine,CenterAffine}, inputs::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    h_ave = mean_from_inputs(layer, inputs)
    u = batchmean(rbm.hidden, h_ave; wts)
    return oftype(rbm.affine_h, damping * CenterAffine(vec(u)) + (1 - damping) * rbm.affine_h)
end

function hidden_affine_from_inputs(
    rbm::AffineRBM{<:AbstractAffine,StdizeAffine}, inputs::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    μ, ν = hidden_stats(rbm.hidden, inputs; wts)
    affine_h_new = whitening_transform(vec(μ), Diagonal(vec(ν .+ ϵ)))
    affine_h = damping * affine_h_new + (1 - damping) * rbm.affine_h
    return oftype(rbm.affine_h, affine_h)
end

function hidden_stats(layer::AbstractLayer, inputs::AbstractArray; wts = nothing)
    h_ave = mean_from_inputs(layer, inputs)
    h_var = var_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts)
    ν_ext = batchvar(layer, h_ave; wts, mean = μ)
    ν = ν_int + ν_ext # law of total variance
    return (μ = μ, ν = ν)
end
