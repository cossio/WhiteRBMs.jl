struct WhiteRBM{V,H,W,Av,Ah}
    visible::V
    hidden::H
    w::W
    affine_v::Av
    affine_h::Ah
    """
        WhiteRBM(rbm, affine_v, affine_h)

    Creates a whitened RBM with the given transforms. Note that the resulting model is
    not equivalent to the original `rbm`. To obtain an equivalent model (where energies
    differ by a constant), see [`whiten`](@ref).
    """
    function WhiteRBM(rbm::RBM{V,H,W}, affine_v::AbstractAffine, affine_h::AbstractAffine) where {V,H,W}
        @assert length(rbm.visible) == length(affine_v.u)
        @assert length(rbm.hidden) == length(affine_h.u)
        return new{V, H, W, typeof(affine_v), typeof(affine_h)}(rbm.visible, rbm.hidden, rbm.w, affine_v, affine_h)
    end
end

function WhiteRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray, affine_v::AbstractAffine, affine_h::AbstractAffine)
    rbm = RBM(visible, hidden, w)
    return WhiteRBM(rbm, affine_v, affine_h)
end

"""
    WhiteRBM(rbm)

Creates a whitened RBM with identity transforms (equivalent to `rbm`).
"""
function WhiteRBM(rbm::RBM)
    T = eltype(rbm.w)
    N, M = length(rbm.visible), length(rbm.hidden)
    affine_v = Affine(Matrix{T}(I, N, N), zeros(T, N))
    affine_h = Affine(Matrix{T}(I, M, M), zeros(T, M))
    return WhiteRBM(rbm, affine_v, affine_h)
end

# Having the affine transform type args first is convenient in some cases ...
const AffineRBM{Av,Ah,V,H,W} = WhiteRBM{V,H,W,Av,Ah}

"""
    RBM(white_rbm::WhiteRBM)

Returns an (unwhitened) `RBM` which neglects the affine transforms of `white_rbm`.
The resulting model is *not* equivalent to the original `white_rbm`.
To construct an equivalent model, use the function `blacken(white_rbm)` instead (see [`blacken`](@ref)).
"""
RestrictedBoltzmannMachines.RBM(rbm::WhiteRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

whiten_v(rbm::WhiteRBM, v::AbstractArray) = reshape(rbm.affine_v * flatten(rbm.visible, v), size(v))
whiten_h(rbm::WhiteRBM, h::AbstractArray) = reshape(rbm.affine_h * flatten(rbm.hidden, h), size(h))

function RestrictedBoltzmannMachines.interaction_energy(rbm::WhiteRBM, v::AbstractArray, h::AbstractArray)
    white_v = whiten_v(rbm, v)
    white_h = whiten_h(rbm, h)
    return interaction_energy(RBM(rbm), white_v, white_h)
end

function RestrictedBoltzmannMachines.free_energy(rbm::WhiteRBM, v::AbstractArray)
    inputs = inputs_h_from_v(rbm, v)
    E = energy(rbm.visible, v)
    Γ = cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = reshape(rbm.affine_h.u, size(rbm.hidden))), inputs)
    return E - ΔE - Γ
end

function RestrictedBoltzmannMachines.inputs_h_from_v(rbm::WhiteRBM, v::AbstractArray)
    white_v = whiten_v(rbm, v)
    inputs = inputs_h_from_v(RBM(rbm), white_v)
    return reshape(rbm.affine_h.A' * flatten(rbm.hidden, inputs), size(inputs))
end

function RestrictedBoltzmannMachines.inputs_v_from_h(rbm::WhiteRBM, h::AbstractArray)
    white_h = whiten_h(rbm, h)
    inputs = inputs_v_from_h(RBM(rbm), white_h)
    return reshape(rbm.affine_v.A' * flatten(rbm.visible, inputs), size(inputs))
end

function RestrictedBoltzmannMachines.mirror(rbm::WhiteRBM)
    function p(i)
        if i ≤ ndims(rbm.visible)
            return i + ndims(rbm.hidden)
        else
            return i - ndims(rbm.visible)
        end
    end
    perm = ntuple(p, ndims(rbm.w))
    w = permutedims(rbm.w, perm)
    mirrored = RBM(rbm.hidden, rbm.visible, w)
    return WhiteRBM(mirrored, rbm.affine_h, rbm.affine_v)
end

function RestrictedBoltzmannMachines.∂free_energy(
    rbm::WhiteRBM, v::AbstractArray;
    wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, v),
    moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    h_moments = moments_from_inputs(rbm.hidden, inputs)
    ∂v = ∂energy_from_moments(rbm.visible, moments)
    ∂h = ∂energy_from_moments(rbm.hidden, batchmean_moments(rbm.hidden, h_moments; wts))
    h = mean_from_moments(rbm.hidden, h_moments)
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

function RestrictedBoltzmannMachines.∂interaction_energy(
    rbm::WhiteRBM, v::AbstractArray, h::AbstractArray; kwargs...
)
    white_v = whiten_v(rbm, v)
    white_h = whiten_h(rbm, h)
    ∂w = ∂interaction_energy(RBM(rbm), white_v, white_h; kwargs...)
    return ∂w
end

function RestrictedBoltzmannMachines.log_pseudolikelihood(rbm::WhiteRBM, v::AbstractArray)
    return log_pseudolikelihood(blacken(rbm), v)
end
