abstract type AbstractTransform end
struct Whiten <: AbstractTransform end # whiten by an affine transform
struct Stdize <: AbstractTransform end # standardize by scaling and centering
struct Center <: AbstractTransform end # center by subtracting the mean
struct Identity <: AbstractTransform end # no transform

function whiten_visible_from_data(
    white_rbm::WhiteRBM, data::AbstractArray, transform::AbstractTransform = Whiten();
    wts = nothing, ϵ::Real = 0
)
    affine_v = visible_affine_from_data(white_rbm, data, transform; wts, ϵ)
    return whiten_visible(white_rbm, affine_v)
end

function whiten_visible_from_data!(
    white_rbm::WhiteRBM, data::AbstractArray, transform::AbstractTransform = Whiten();
    wts = nothing, ϵ::Real = 0
)
    affine_v = visible_affine_from_data(white_rbm, data, transform; wts, ϵ)
    return whiten_visible!(white_rbm, affine_v)
end

function whiten_hidden_from_inputs(
    white_rbm::WhiteRBM, inputs::AbstractArray, transform::AbstractTransform = Stdize();
    wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    affine_h = hidden_affine_from_inputs(white_rbm, inputs, transform; wts, damping, ϵ)
    return whiten_hidden(white_rbm, affine_h)
end

function whiten_hidden_from_inputs!(
    white_rbm::WhiteRBM, inputs::AbstractArray, transform::AbstractTransform = Stdize();
    wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    affine_h = hidden_affine_from_inputs(white_rbm, inputs, transform; wts, damping, ϵ)
    return whiten_hidden!(white_rbm, affine_h)
end

function safe_whiten_visible_from_data(
    white_rbm::WhiteRBM, data::AbstractArray, transform::AbstractTransform = Whiten();
    wts = nothing, ϵ::Real = 0
)
    affine_v = visible_affine_from_data(white_rbm, data, transform; wts, ϵ)
    return safe_whiten_visible(white_rbm, affine_v)
end

function safe_whiten_hidden_from_inputs(
    white_rbm::WhiteRBM, inputs::AbstractArray, transform::AbstractTransform = Stdize();
    wts = nothing, damping::Real=0, ϵ::Real=0
)
    affine_h = hidden_affine_from_inputs(white_rbm, inputs, transform; wts, damping, ϵ)
    return safe_whiten_hidden(white_rbm, affine_h)
end

"""
    visible_affine_from_data(white_rbm, data; wts = nothing, ϵ = 0)

Returns the affine transformation that whitens the visible data.
"""
function visible_affine_from_data(
    white_rbm::WhiteRBM, data::AbstractArray, ::Whiten = Whiten();
    wts = nothing, ϵ::Real = 0
)
    μ = batchmean(white_rbm.visible, data; wts)
    C = batchcov(white_rbm.visible, data; wts, mean=μ)
    C_flat = reshape(C, length(white_rbm.visible), length(white_rbm.visible))
    return whitening_transform(vec(μ), Symmetric(C_flat + ϵ * I))
end

function visible_affine_from_data(
    white_rbm::WhiteRBM, data::AbstractArray, ::Stdize; wts = nothing, ϵ::Real = 0
)
    μ = batchmean(white_rbm.visible, data; wts)
    ν = batchvar(white_rbm.visible, data; wts, mean=μ)
    return whitening_transform(vec(μ), Diagonal(vec(ν .+ ϵ)))
end

function visible_affine_from_data(
    white_rbm::WhiteRBM, data::AbstractArray, ::Center; wts = nothing, ϵ::Real = 0
)
    μ = batchmean(white_rbm.visible, data; wts)
    return whitening_transform(vec(μ))
end

function visible_affine_from_data(
    white_rbm::WhiteRBM, data::AbstractArray, ::Identity; wts = nothing, ϵ::Real = 0
)
    μ = batchmean(white_rbm.visible, data; wts)
    return whitening_transform(vec(zero(μ)))
end

"""
    hidden_affine_from_inputs(white_rbm, data; wts = nothing, ϵ = 0)

Returns the affine transformation that standardizes the hidden inputs.
Note that only the variances are scaled to unity.
The correlation matrix is not diagonalized.
"""
function hidden_affine_from_inputs(
    white_rbm::WhiteRBM, inputs::AbstractArray, ::Stdize = Stdize();
    wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    μ, ν = hidden_stats(white_rbm.hidden, inputs; wts)
    affine_h_new = whitening_transform(vec(μ), Diagonal(vec(ν .+ ϵ)))
    return damping * affine_h_new + (1 - damping) * white_rbm.affine_h
end

function hidden_affine_from_inputs(
    white_rbm::WhiteRBM, inputs::AbstractArray, ::Center;
    wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    h_ave = mean_from_inputs(white_rbm.hidden, inputs)
    μ = batchmean(white_rbm.hidden, h_ave; wts)
    affine_h_new = whitening_transform(vec(μ))
    return damping * affine_h_new + (1 - damping) * white_rbm.affine_h
end

function hidden_affine_from_inputs(
    white_rbm::WhiteRBM, inputs::AbstractArray, ::Identity;
    wts = nothing, damping::Real = 0, ϵ::Real = 0
)
    h_ave = mean_from_inputs(white_rbm.hidden, inputs)
    μ = batchmean(white_rbm.hidden, h_ave; wts)
    return whitening_transform(vec(zero(μ)))
end

function hidden_stats(layer, inputs::AbstractArray; wts = nothing)
    h_ave = mean_from_inputs(layer, inputs)
    h_var = var_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts)
    ν_ext = batchvar(layer, h_ave; wts, mean = μ)
    ν = ν_int + ν_ext # law of total variance
    return (μ = μ, ν = ν)
end
