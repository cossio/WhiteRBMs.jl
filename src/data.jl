function whiten_visible_from_data!(
    rbm::WhiteRBM, data::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
)
    affine_v = visible_affine_from_data(rbm, data; wts, ϵ)
    return whiten_visible!(rbm, affine_v)
end

function whiten_hidden_from_inputs!(
    white_rbm::WhiteRBM, inputs::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(white_rbm.hidden, inputs),
    damping::Real = 0, ϵ::Real = 0
)
    affine_h = hidden_affine_from_inputs(white_rbm, inputs; wts, damping, ϵ)
    return whiten_hidden!(white_rbm, affine_h)
end

function whiten_hidden_from_v!(
    white_rbm::WhiteRBM, v::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(white_rbm.visible, v),
    damping::Real = 0, ϵ::Real = 0
)
    inputs = inputs_h_from_v(white_rbm, v)
    return whiten_hidden_from_inputs!(white_rbm, inputs; damping, wts, ϵ)
end

function visible_affine_from_data(
    rbm::AffineRBM{<:CenterAffine,<:AbstractAffine}, data::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
)
    μ = batchmean(rbm.visible, data; wts)
    return CenterAffine(vec(μ))
end

function visible_affine_from_data(
    rbm::AffineRBM{<:StdizeAffine,<:AbstractAffine}, data::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
)
    μ = batchmean(rbm.visible, data; wts)
    ν = batchvar(rbm.visible, data; wts, mean = μ)
    return StdizeAffine(Diagonal(1 ./ sqrt.(vec(ν) .+ ϵ)), vec(μ))
end

function visible_affine_from_data(
    rbm::AffineRBM{<:Affine,<:AbstractAffine}, data::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, data), ϵ::Real = 0
)
    μ = batchmean(rbm.visible, data; wts)
    C = batchcov(rbm.visible, data; wts, mean = μ)
    C_flat = reshape(C, length(rbm.visible), length(rbm.visible))
    return whitening_transform(vec(μ), Symmetric(C_flat + ϵ * I))
end

function hidden_affine_from_inputs(
    rbm::AffineRBM{<:AbstractAffine,<:CenterAffine}, inputs::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs),
    damping::Real = 0, ϵ::Real = 0
)
    μ = total_mean_from_inputs(rbm.hidden, inputs; wts)
    u = (1 - damping) * rbm.affine_h.u + damping * vec(μ)
    return CenterAffine(u)
end

function hidden_affine_from_inputs(
    rbm::AffineRBM{<:AbstractAffine,<:Affine}, inputs::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs),
    damping::Real = 0, ϵ::Real = 0
)
    h_ave = mean_from_inputs(rbm.hidden, inputs)
    h_var = var_from_inputs(rbm.hidden, inputs)
    μ = batchmean(rbm.hidden, h_ave; wts)
    # law of total covariance: <cov(h|v)> (diagonal) + cov(<h|v>)
    C_int = Diagonal(vec(batchmean(rbm.hidden, h_var; wts)))
    C_ext = reshape(batchcov(rbm.hidden, h_ave; wts, mean = μ), length(rbm.hidden), length(rbm.hidden))
    C_new = C_int + C_ext + ϵ * I
    # mix moments with the current affine transform (A_old whitens C_old, so C_old = A⁻¹A⁻ᵀ)
    A_old = rbm.affine_h.A
    C_old = inv(A_old) * inv(A_old)'
    u = (1 - damping) * rbm.affine_h.u + damping * vec(μ)
    C = (1 - damping) * C_old + damping * C_new
    return whitening_transform(u, Symmetric(Matrix(C)))
end

function hidden_affine_from_inputs(
    rbm::AffineRBM{<:AbstractAffine,<:StdizeAffine}, inputs::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, inputs),
    damping::Real = 0, ϵ::Real = 0
)
    μ, ν = total_meanvar_from_inputs(rbm.hidden, inputs; wts)
    u = (1 - damping) * rbm.affine_h.u + damping * vec(μ)
    # mix variances (A = Diagonal(1 ./ scale)), as RBMs.jl does for StandardizedRBM
    old_var = 1 ./ diag(rbm.affine_h.A) .^ 2
    new_var = (1 - damping) * old_var + damping * (vec(ν) .+ ϵ)
    return StdizeAffine(Diagonal(1 ./ sqrt.(new_var)), u)
end
