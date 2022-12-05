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
    function WhiteRBM(rbm::RBM{V,H,W}, affine_v::Affine, affine_h::Affine) where {V,H,W}
        @assert length(rbm.visible) == length(affine_v.u)
        @assert length(rbm.hidden) == length(affine_h.u)
        return new{V, H, W, typeof(affine_v), typeof(affine_h)}(
            rbm.visible, rbm.hidden, rbm.w, affine_v, affine_h
        )
    end
end

function WhiteRBM(visible, hidden, w, affine_v, affine_h)
    rbm = RBM(visible, hidden, w)
    return WhiteRBM(rbm, affine_v, affine_h)
end

const AffineRBM{Av,Ah,V,H,W} = WhiteRBM{V,H,W,Av,Ah}

"""
    WhiteRBM(rbm)

Creates a WhiteRBM with identity transforms.
"""
function WhiteRBM(rbm::RBM)
    T = eltype(rbm.w)
    N = length(rbm.visible)
    M = length(rbm.hidden)
    affine_v = Affine(Diagonal(ones(T, N)), zeros(T, N))
    affine_h = Affine(Diagonal(ones(T, M)), zeros(T, M))
    return WhiteRBM(rbm, affine_v, affine_h)
end

"""
    RBM(white_rbm::WhiteRBM)

Returns an (unwhitened) `RBM` which neglects the affine transforms of `white_rbm`.
The resulting model is *not* equivalent to the original `white_rbm`.
To construct an equivalent model, use the function
`blacken(white_rbm)` instead (see [`blacken`](@ref)).
"""
function RestrictedBoltzmannMachines.RBM(white_rbm::WhiteRBM)
    return RBM(white_rbm.visible, white_rbm.hidden, white_rbm.w)
end

function whiten_v(white_rbm::WhiteRBM, v::AbstractArray)
    return reshape(white_rbm.affine_v * flatten(white_rbm.visible, v), size(v))
end

function whiten_h(white_rbm::WhiteRBM, h::AbstractArray)
    return reshape(white_rbm.affine_h * flatten(white_rbm.hidden, h), size(h))
end

function RestrictedBoltzmannMachines.energy(white_rbm::WhiteRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(white_rbm.visible, v)
    Eh = energy(white_rbm.hidden, h)
    Ew = interaction_energy(white_rbm, v, h)
    return Ev .+ Eh .+ Ew
end

function RestrictedBoltzmannMachines.interaction_energy(white_rbm::WhiteRBM, v::AbstractArray, h::AbstractArray)
    white_v = whiten_v(white_rbm, v)
    white_h = whiten_h(white_rbm, h)
    return interaction_energy(RBM(white_rbm), white_v, white_h)
end

function RestrictedBoltzmannMachines.free_energy(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = inputs_h_from_v(white_rbm, v)
    E_vis = energy(white_rbm.visible, v)
    F_hid = free_energy(white_rbm.hidden, inputs)
    b = reshape(white_rbm.affine_h.u, size(white_rbm.hidden))
    ΔE = energy(Binary(b), inputs)
    return E_vis - ΔE + F_hid
end

function RestrictedBoltzmannMachines.inputs_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    white_v = whiten_v(white_rbm, v)
    inputs = inputs_h_from_v(RBM(white_rbm), white_v)
    I_flat = flatten(white_rbm.hidden, inputs)
    return reshape(white_rbm.affine_h.A' * I_flat, size(inputs))
end

function RestrictedBoltzmannMachines.inputs_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    white_h = whiten_h(white_rbm, h)
    inputs = inputs_v_from_h(RBM(white_rbm), white_h)
    I_flat = flatten(white_rbm.visible, inputs)
    return reshape(white_rbm.affine_v.A' * I_flat, size(inputs))
end

function RestrictedBoltzmannMachines.sample_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = inputs_h_from_v(white_rbm, v)
    return sample_from_inputs(white_rbm.hidden, inputs)
end

function RestrictedBoltzmannMachines.sample_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    inputs = inputs_v_from_h(white_rbm, h)
    return sample_from_inputs(white_rbm.visible, inputs)
end

function RestrictedBoltzmannMachines.sample_v_from_v(white_rbm::WhiteRBM, v::AbstractArray; steps::Int = 1)
    @assert size(white_rbm.visible) == size(v)[1:ndims(white_rbm.visible)]
    for _ in 1:steps
        v = oftype(v, sample_v_from_v_once(white_rbm, v))
    end
    return v
end

function RestrictedBoltzmannMachines.sample_h_from_h(white_rbm::WhiteRBM, h::AbstractArray; steps::Int = 1)
    @assert size(white_rbm.hidden) == size(h)[1:ndims(white_rbm.hidden)]
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(white_rbm, h))
    end
    return h
end

function RestrictedBoltzmannMachines.sample_v_from_v_once(white_rbm::WhiteRBM, v::AbstractArray)
    h = sample_h_from_v(white_rbm, v)
    v = sample_v_from_h(white_rbm, h)
    return v
end

function RestrictedBoltzmannMachines.sample_h_from_h_once(white_rbm::WhiteRBM, h::AbstractArray)
    v = sample_v_from_h(white_rbm, h)
    h = sample_h_from_v(white_rbm, v)
    return h
end

function RestrictedBoltzmannMachines.mean_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = inputs_h_from_v(white_rbm, v)
    return mean_from_inputs(white_rbm.hidden, inputs)
end

function RestrictedBoltzmannMachines.mean_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    inputs = inputs_v_from_h(white_rbm, h)
    return mean_from_inputs(white_rbm.visible, inputs)
end

function RestrictedBoltzmannMachines.mode_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    inputs = inputs_v_from_h(white_rbm, h)
    return mode_from_inputs(white_rbm.visible, inputs)
end

function RestrictedBoltzmannMachines.mode_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = inputs_h_from_v(white_rbm, v)
    return mode_from_inputs(white_rbm.hidden, inputs)
end

function RestrictedBoltzmannMachines.reconstruction_error(white_rbm::WhiteRBM, v::AbstractArray; steps::Int = 1)
    @assert size(white_rbm.visible) == size(v)[1:ndims(white_rbm.visible)]
    v1 = sample_v_from_v(white_rbm, v; steps)
    ϵ = Statistics.mean(abs.(v .- v1); dims = 1:ndims(white_rbm.visible))
    if ndims(v) == ndims(white_rbm.visible)
        return only(ϵ)
    else
        return reshape(ϵ, size(v)[end])
    end
end

function RestrictedBoltzmannMachines.mirror(white_rbm::WhiteRBM)
    function p(i)
        if i ≤ ndims(white_rbm.visible)
            return i + ndims(white_rbm.hidden)
        else
            return i - ndims(white_rbm.visible)
        end
    end
    perm = ntuple(p, ndims(white_rbm.w))
    w = permutedims(white_rbm.w, perm)
    rbm = RBM(white_rbm.hidden, white_rbm.visible, w)
    return WhiteRBM(rbm, white_rbm.affine_h, white_rbm.affine_v)
end

function RestrictedBoltzmannMachines.∂free_energy(
    white_rbm::WhiteRBM, v::AbstractArray; wts = nothing,
    stats = RBMs.suffstats(white_rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(white_rbm, v)
    h = mean_from_inputs(white_rbm.hidden, inputs)
    ∂v = ∂energy(white_rbm.visible, stats)
    ∂h = ∂free_energy(white_rbm.hidden, inputs; wts)
    ∂w = ∂interaction_energy(white_rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function RestrictedBoltzmannMachines.∂interaction_energy(
    white_rbm::WhiteRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    white_v = whiten_v(white_rbm, v)
    white_h = whiten_h(white_rbm, h)
    ∂w = ∂interaction_energy(RBM(white_rbm), white_v, white_h; wts)
    return ∂w
end

function RestrictedBoltzmannMachines.log_pseudolikelihood(rbm::WhiteRBM, v::AbstractArray)
    return log_pseudolikelihood(blacken(rbm), v)
end
