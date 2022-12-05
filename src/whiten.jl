@doc raw"""
    blacken(white_rbm::WhiteRBM)

Constructs a plain `RBM` equivalent to the given `white_rbm`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - \tilde{E}(v,h) = \mathbf{a}^\top \mathbb{A}^\top \tilde{\mathbb{W}} \mathbb{B}\mathbf{b}
```

where ``\tilde{E}(v,h)`` is the energy assigned by `white_rbm` and ``E(v,h)`` is the energy
assigned by the `RBM` constructed by this method.

This is the inverse operation of [`whiten`](@ref), always returning an ordinary `RBM`.

To construct an `RBM` that simply neglects the transformations,
call `RBM(white_rbm)` instead.
"""
blacken(white_rbm::WhiteRBM) = RBM(whiten(white_rbm))
blacken(rbm::RBM) = rbm

@doc raw"""
    whiten(rbm::RBM, affine_v, affine_h)

Constructs a `WhiteRBM` equivalent to the given `rbm`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - \tilde{E}(v,h) = \mathbf{a}^\top \mathbb{A}^\top \tilde{\mathbb{W}} \mathbb{B}\mathbf{b}
```

where ``E(v,h)`` is the energy assigned by the original `rbm`, and
``\tilde{E}(v,h)`` is the energy assigned by the returned `WhiteRBM`.

This is the inverse operation of [`blacken`](@ref), always returning a `WhiteRBM`.

To construct a `WhiteRBM` that simply includes these transforms,
call `WhiteRBM(rbm, affine_v, affine_h)` instead.
"""
whiten(rbm::RBM, affine_v::Affine, affine_h::Affine) = whiten(whiten(rbm), affine_v, affine_h)
whiten(rbm::RBM) = WhiteRBM(rbm)

function whiten(white_rbm::AffineRBM, affine_v::Affine, affine_h::Affine)
    @assert length(white_rbm.visible) == length(affine_v.u)
    @assert length(white_rbm.hidden) == length(affine_h.u)

    w1 = reshape(white_rbm.w, length(white_rbm.visible), length(white_rbm.hidden))
    w2 = affine_v.A' \ white_rbm.affine_v.A' * w1 * white_rbm.affine_h.A / affine_h.A
    Δg = inputs_v_from_h(white_rbm, reshape(affine_h.u, size(white_rbm.hidden)))
    Δθ = inputs_h_from_v(white_rbm, reshape(affine_v.u, size(white_rbm.visible)))

    vis = shift_fields(white_rbm.visible, reshape(Δg, size(white_rbm.visible)))
    hid = shift_fields(white_rbm.hidden, reshape(Δθ, size(white_rbm.hidden)))
    rbm = RBM(vis, hid, reshape(w2, size(white_rbm.w)))

    return WhiteRBM(rbm, affine_v, affine_h)
end

function whiten(white_rbm::WhiteRBM)
    affine_v = one(white_rbm.affine_v)
    affine_h = one(white_rbm.affine_h)
    return whiten(white_rbm, affine_v, affine_h)
end

function whiten_visible(white_rbm::WhiteRBM, affine_v::Affine)
    @assert length(white_rbm.visible) == length(affine_v.u)

    w1 = reshape(white_rbm.w, length(white_rbm.visible), length(white_rbm.hidden))
    w2 = affine_v.A' \ white_rbm.affine_v.A' * w1
    Δθ = inputs_h_from_v(white_rbm, reshape(affine_v.u, size(white_rbm.visible)))

    hid = shift_fields(white_rbm.hidden, Δθ)
    rbm = RBM(white_rbm.visible, hid, reshape(w2, size(white_rbm.w)))

    return WhiteRBM(rbm, affine_v, white_rbm.affine_h)
end

function whiten_hidden(white_rbm::WhiteRBM, affine_h::Affine)
    @assert length(white_rbm.hidden) == length(affine_h.u)

    w1 = reshape(white_rbm.w, length(white_rbm.visible), length(white_rbm.hidden))
    w2 = w1 * white_rbm.affine_h.A / affine_h.A
    Δg = inputs_v_from_h(white_rbm, reshape(affine_h.u, size(white_rbm.hidden)))

    vis = shift_fields(white_rbm.visible, Δg)
    rbm = RBM(vis, white_rbm.hidden, reshape(w2, size(white_rbm.w)))

    return WhiteRBM(rbm, white_rbm.affine_v, affine_h)
end

whiten_visible(rbm::RBM, affine_v::Affine) = whiten_visible(whiten(rbm), affine_v)
whiten_hidden(rbm::RBM, affine_h::Affine) = whiten_hidden(whiten(rbm), affine_h)

# In-place versions

function whiten!(rbm::AffineRBM, affine_v::Affine, affine_h::Affine)
    whiten_visible!(rbm, affine_v)
    whiten_hidden!(rbm, affine_h)
end

whiten!(rbm::WhiteRBM) = whiten!(rbm, one(rbm.affine_v), one(rbm.affine_h))

function whiten_visible!(rbm::WhiteRBM, affine_v::Affine)
    @assert length(rbm.visible) == length(affine_v.u)
    Δθ = inputs_h_from_v(rbm, reshape(affine_v.u, size(rbm.visible)))
    shift_fields!(rbm.hidden, Δθ)
    w = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    copyto!(rbm.w, affine_v.A' \ rbm.affine_v.A' * w)
    copy!(rbm.affine_v, affine_v)
    return rbm
end

function whiten_hidden!(rbm::WhiteRBM, affine_h::Affine)
    @assert length(rbm.hidden) == length(affine_h.u)
    Δg = inputs_v_from_h(rbm, reshape(affine_h.u, size(rbm.hidden)))
    shift_fields!(rbm.visible, Δg)
    w = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    copyto!(rbm.w, w * rbm.affine_h.A / affine_h.A)
    copy!(rbm.affine_h, affine_h)
    return rbm
end

"""
    energy_shift(rbm, affine_v, affine_h)

Computes the constant energy shift if the affine transformations were updated as given.
"""
function energy_shift(rbm::WhiteRBM, affine_v::Affine, affine_h::Affine)
    @assert length(rbm.visible) == length(affine_v.u)
    @assert length(rbm.hidden) == length(affine_h.u)
    E1 = interaction_energy(RBM(rbm), rbm.affine_v.A * rbm.affine_v.u, rbm.affine_h.A * rbm.affine_h.u)
    E2 = interaction_energy(RBM(rbm), rbm.affine_v.A * affine_v.u, rbm.affine_h.A * affine_h.u)
    return E2 - E1
end

function energy_shift_visible(rbm::WhiteRBM, affine_v::Affine)
    Δa = affine_v.u - rbm.affine_v.u
    return interaction_energy(RBM(rbm), rbm.affine_v.A * Δa, rbm.affine_h.A * rbm.affine_h.u)
end

function energy_shift_hidden(rbm::WhiteRBM, affine_h::Affine)
    Δb = affine_h.u - rbm.affine_h.u
    return interaction_energy(RBM(rbm), rbm.affine_v.A * rbm.affine_v.u, rbm.affine_h.A * Δb)
end

energy_shift(rbm::RBM, affine_v::Affine, affine_h::Affine) = energy_shift(whiten(rbm), affine_v, affine_h)
energy_shift_visible(rbm::RBM, affine_v::Affine) = energy_shift_visible(whiten(rbm), affine_v)
energy_shift_hidden(rbm::RBM, affine_h::Affine) = energy_shift_hidden(whiten(rbm), affine_h)

# Safe versions (preserve argument type)

"""
    safe_whiten(white_rbm, affine_v, affine_h)

Like `whiten(white_rbm, affine_v, affine_h)`, but ensures that the returned `WhiteRBM`
is of the same type as `white_rbm` (in particular, preserving the types of the
affine transformations).
"""
function safe_whiten(white_rbm::AffineRBM{Av,Ah}, affine_v::Av, affine_h::Ah) where {Av,Ah}
    return oftype(white_rbm, whiten(white_rbm, affine_v, affine_h))
end

safe_whiten(white_rbm::WhiteRBM) = oftype(white_rbm, whiten(white_rbm))

function safe_whiten_visible(white_rbm::AffineRBM{Av,Ah}, affine_v::Av) where {Av,Ah}
    return oftype(white_rbm, whiten_visible(white_rbm, affine_v))
end

function safe_whiten_hidden(white_rbm::AffineRBM{Av,Ah}, affine_h::Ah) where {Av,Ah}
    return oftype(white_rbm, whiten_hidden(white_rbm, affine_h))
end

# for RBM arguments fallback to normal methods
safe_whiten(rbm::RBM, affine_v::Affine, affine_h::Affine) = whiten(rbm, affine_v, affine_h)
safe_whiten(rbm::RBM) = whiten(rbm)
safe_whiten_visible(rbm::RBM, affine::Affine) = whiten_visible(rbm, affine)
safe_whiten_hidden(rbm::RBM, affine::Affine) = whiten_hidden(rbm, affine)
