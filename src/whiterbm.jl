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
        @assert length(visible(rbm)) == length(affine_v.u)
        @assert length(hidden(rbm)) == length(affine_h.u)
        return new{V, H, W, typeof(affine_v), typeof(affine_h)}(
            visible(rbm), hidden(rbm), weights(rbm), affine_v, affine_h
        )
    end
end

function WhiteRBM(visible, hidden, w, affine_v, affine_h)
    rbm = RBM(visible, hidden, w)
    return WhiteRBM(rbm, affine_v, affine_h)
end

const AffineRBM{Av,Ah,V,H,W} = WhiteRBM{V,H,W,Av,Ah}

RBMs.visible(white_rbm::WhiteRBM) = white_rbm.visible
RBMs.hidden(white_rbm::WhiteRBM) = white_rbm.hidden
RBMs.weights(white_rbm::WhiteRBM) = white_rbm.w

"""
    WhiteRBM(rbm)

Creates a WhiteRBM with identity transforms.
"""
function WhiteRBM(rbm::RBM)
    T = eltype(weights(rbm))
    N = length(visible(rbm))
    M = length(hidden(rbm))
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
function RBMs.RBM(white_rbm::WhiteRBM)
    return RBM(visible(white_rbm), hidden(white_rbm), weights(white_rbm))
end

function whiten_v(white_rbm::WhiteRBM, v::AbstractArray)
    return reshape(white_rbm.affine_v * flatten(visible(white_rbm), v), size(v))
end

function whiten_h(white_rbm::WhiteRBM, h::AbstractArray)
    return reshape(white_rbm.affine_h * flatten(hidden(white_rbm), h), size(h))
end

function RBMs.energy(white_rbm::WhiteRBM, v::AbstractArray, h::AbstractArray)
    Ev = RBMs.energy(visible(white_rbm), v)
    Eh = RBMs.energy(hidden(white_rbm), h)
    Ew = RBMs.interaction_energy(white_rbm, v, h)
    return Ev .+ Eh .+ Ew
end

function RBMs.interaction_energy(white_rbm::WhiteRBM, v::AbstractArray, h::AbstractArray)
    white_v = whiten_v(white_rbm, v)
    white_h = whiten_h(white_rbm, h)
    return RBMs.interaction_energy(RBM(white_rbm), white_v, white_h)
end

function RBMs.free_energy(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(white_rbm, v)
    E_vis = RBMs.energy(visible(white_rbm), v)
    F_hid = RBMs.free_energy(hidden(white_rbm), inputs)
    b = reshape(white_rbm.affine_h.u, size(hidden(white_rbm)))
    ΔE = RBMs.energy(RBMs.Binary(b), inputs)
    return E_vis - ΔE + F_hid
end

function RBMs.inputs_v_to_h(white_rbm::WhiteRBM, v::AbstractArray)
    white_v = whiten_v(white_rbm, v)
    inputs = inputs_v_to_h(RBM(white_rbm), white_v)
    I_flat = flatten(hidden(white_rbm), inputs)
    return reshape(white_rbm.affine_h.A' * I_flat, size(inputs))
end

function RBMs.inputs_h_to_v(white_rbm::WhiteRBM, h::AbstractArray)
    white_h = whiten_h(white_rbm, h)
    inputs = inputs_h_to_v(RBM(white_rbm), white_h)
    I_flat = flatten(visible(white_rbm), inputs)
    return reshape(white_rbm.affine_v.A' * I_flat, size(inputs))
end

function RBMs.sample_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = inputs_v_to_h(white_rbm, v)
    return RBMs.sample_from_inputs(hidden(white_rbm), inputs)
end

function RBMs.sample_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    inputs = inputs_h_to_v(white_rbm, h)
    return RBMs.sample_from_inputs(visible(white_rbm), inputs)
end

function RBMs.sample_v_from_v(white_rbm::WhiteRBM, v::AbstractArray; steps::Int = 1)
    @assert size(visible(white_rbm)) == size(v)[1:ndims(visible(white_rbm))]
    for _ in 1:steps
        v = oftype(v, RBMs.sample_v_from_v_once(white_rbm, v))
    end
    return v
end

function RBMs.sample_h_from_h(white_rbm::WhiteRBM, h::AbstractArray; steps::Int = 1)
    @assert size(hidden(white_rbm)) == size(h)[1:ndims(hidden(white_rbm))]
    for _ in 1:steps
        h = oftype(h, RBMs.sample_h_from_h_once(white_rbm, h))
    end
    return h
end

function RBMs.sample_v_from_v_once(white_rbm::WhiteRBM, v::AbstractArray)
    h = RBMs.sample_h_from_v(white_rbm, v)
    v = RBMs.sample_v_from_h(white_rbm, h)
    return v
end

function RBMs.sample_h_from_h_once(white_rbm::WhiteRBM, h::AbstractArray)
    v = RBMs.sample_v_from_h(white_rbm, h)
    h = RBMs.sample_h_from_v(white_rbm, v)
    return h
end

function RBMs.mean_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(white_rbm, v)
    return RBMs.mean_from_inputs(hidden(white_rbm), inputs)
end

function RBMs.mean_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    inputs = RBMs.inputs_h_to_v(white_rbm, h)
    return RBMs.mean_from_inputs(visible(white_rbm), inputs)
end

function RBMs.mode_v_from_h(white_rbm::WhiteRBM, h::AbstractArray)
    inputs = RBMs.inputs_h_to_v(white_rbm, h)
    return RBMs.mode_from_inputs(visible(white_rbm), inputs)
end

function RBMs.mode_h_from_v(white_rbm::WhiteRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(white_rbm, v)
    return RBMs.mode_from_inputs(hidden(white_rbm), inputs)
end

function RBMs.reconstruction_error(white_rbm::WhiteRBM, v::AbstractArray; steps::Int = 1)
    @assert size(visible(white_rbm)) == size(v)[1:ndims(visible(white_rbm))]
    v1 = RBMs.sample_v_from_v(white_rbm, v; steps)
    ϵ = Statistics.mean(abs.(v .- v1); dims = 1:ndims(visible(white_rbm)))
    if ndims(v) == ndims(visible(white_rbm))
        return only(ϵ)
    else
        return reshape(ϵ, size(v)[end])
    end
end

function RBMs.mirror(white_rbm::WhiteRBM)
    function p(i)
        if i ≤ ndims(visible(white_rbm))
            return i + ndims(hidden(white_rbm))
        else
            return i - ndims(visible(white_rbm))
        end
    end
    perm = ntuple(p, ndims(weights(white_rbm)))
    w = permutedims(weights(white_rbm), perm)
    rbm = RBM(hidden(white_rbm), visible(white_rbm), w)
    return WhiteRBM(rbm, white_rbm.affine_h, white_rbm.affine_v)
end

function RBMs.∂free_energy(
    white_rbm::WhiteRBM, v::AbstractArray; wts = nothing,
    stats = RBMs.suffstats(visible(white_rbm), v; wts)
)
    inputs = RBMs.inputs_v_to_h(white_rbm, v)
    h = RBMs.mean_from_inputs(hidden(white_rbm), inputs)
    ∂v = RBMs.∂energy(visible(white_rbm), stats)
    ∂h = RBMs.∂free_energy(hidden(white_rbm), inputs; wts)
    ∂w = RBMs.∂interaction_energy(white_rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function RBMs.∂interaction_energy(
    white_rbm::WhiteRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    white_v = whiten_v(white_rbm, v)
    white_h = whiten_h(white_rbm, h)
    ∂w = RBMs.∂interaction_energy(RBM(white_rbm), white_v, white_h; wts)
    return ∂w
end

function RBMs.log_pseudolikelihood(rbm::WhiteRBM, v::AbstractArray)
    return RBMs.log_pseudolikelihood(blacken(rbm), v)
end
