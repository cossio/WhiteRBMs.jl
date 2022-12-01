import Zygote

using Test: @test, @testset, @inferred
using Random: bitrand
using Statistics: mean
using LinearAlgebra: I
using RestrictedBoltzmannMachines: RBM, BinaryRBM
using RestrictedBoltzmannMachines: visible, hidden, weights
using RestrictedBoltzmannMachines: energy, interaction_energy, free_energy, ∂free_energy
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using WhiteRBMs: BinaryWhiteRBM, WhiteRBM, Affine
using WhiteRBMs: whiten, blacken, whiten_visible, whiten_hidden
using WhiteRBMs: whiten!, whiten_visible!, whiten_hidden!
using WhiteRBMs: energy_shift, energy_shift_visible, energy_shift_hidden

@testset "whiten / blacken" begin
    rbm = @inferred BinaryRBM(randn(3), randn(2), randn(3,2))
    affine_v = @inferred Affine(randn(3,3), randn(3))
    affine_h = @inferred Affine(randn(2,2), randn(2))
    white_rbm = @inferred whiten(rbm, affine_v, affine_h)
    @test white_rbm.affine_v.u ≈ affine_v.u
    @test white_rbm.affine_h.u ≈ affine_h.u
    @test white_rbm.affine_v.A ≈ affine_v.A
    @test white_rbm.affine_h.A ≈ affine_h.A
    @test affine_v.A' * weights(white_rbm) * affine_h.A ≈ weights(rbm)
    @test visible(rbm).θ ≈ visible(white_rbm).θ - weights(rbm) * affine_h.u
    @test hidden(rbm).θ ≈ hidden(white_rbm).θ - weights(rbm)' * affine_v.u
    @test blacken(rbm) == rbm

    brbm = @inferred blacken(white_rbm)
    wrbm = @inferred whiten(white_rbm)
    @test visible(brbm).θ ≈ visible(wrbm).θ ≈ visible(rbm).θ
    @test hidden(brbm).θ ≈ hidden(wrbm).θ ≈ hidden(rbm).θ
    @test weights(brbm) ≈ weights(wrbm) ≈ weights(rbm)

    @test iszero(wrbm.affine_v.u)
    @test iszero(wrbm.affine_h.u)
    @test wrbm.affine_v.A ≈ I
    @test wrbm.affine_h.A ≈ I

    # whiten_visible
    affine_v_new = @inferred Affine(randn(3,3), randn(3))
    white_rbm_new = @inferred whiten_visible(white_rbm, affine_v_new)
    @test white_rbm_new.affine_v.u == affine_v_new.u
    @test white_rbm_new.affine_v.A == affine_v_new.A
    @test white_rbm_new.affine_h.u == affine_h.u
    @test white_rbm_new.affine_h.A == affine_h.A
    white_rbm_expected = whiten(white_rbm, affine_v_new, affine_h)
    @test visible(white_rbm_new).θ ≈ visible(white_rbm_expected).θ
    @test hidden(white_rbm_new).θ ≈ hidden(white_rbm_expected).θ
    @test weights(white_rbm_new) ≈ weights(white_rbm_expected)

    # whiten_hidden
    affine_h_new = @inferred Affine(randn(2,2), randn(2))
    white_rbm_new = @inferred whiten_hidden(white_rbm, affine_h_new)
    @test white_rbm_new.affine_v.u == affine_v.u
    @test white_rbm_new.affine_v.A == affine_v.A
    @test white_rbm_new.affine_h.u == affine_h_new.u
    @test white_rbm_new.affine_h.A == affine_h_new.A
    white_rbm_expected = whiten(white_rbm, affine_v, affine_h_new)
    @test visible(white_rbm_new).θ ≈ visible(white_rbm_expected).θ
    @test hidden(white_rbm_new).θ ≈ hidden(white_rbm_expected).θ
    @test weights(white_rbm_new) ≈ weights(white_rbm_expected)

    wrbm1 = whiten_visible(whiten_hidden(rbm, affine_h), affine_v)
    wrbm2 = whiten_hidden(whiten_visible(rbm, affine_v), affine_h)
    for wrbm in (wrbm1, wrbm2)
        @test wrbm.affine_v.u == affine_v.u
        @test wrbm.affine_v.A == affine_v.A
        @test wrbm.affine_h.u == affine_h.u
        @test wrbm.affine_h.A == affine_h.A
        @test visible(wrbm).θ ≈ visible(white_rbm).θ
        @test hidden(wrbm).θ ≈ hidden(white_rbm).θ
        @test weights(wrbm) ≈ weights(white_rbm)
    end
end

@testset "energy invariance" begin
    affine_v = @inferred Affine(randn(3,3), randn(3))
    affine_h = @inferred Affine(randn(2,2), randn(2))
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    white_rbm = @inferred whiten(rbm, affine_v, affine_h)
    v = bitrand(size(visible(rbm))..., 100)
    h = bitrand(size(hidden(rbm))..., 100)

    @test energy(visible(rbm), v) ≈ energy(visible(white_rbm), v) + v' * weights(rbm) * affine_h.u
    @test energy(hidden(rbm), h) ≈ energy(hidden(white_rbm), h) + h' * weights(rbm)' * affine_v.u

    @test inputs_v_to_h(rbm, v .- affine_v.u) ≈ @inferred inputs_v_to_h(white_rbm, v)
    @test inputs_h_to_v(rbm, h .- affine_h.u) ≈ @inferred inputs_h_to_v(white_rbm, h)

    ΔE = interaction_energy(rbm, affine_v.u, affine_h.u)::Real
    @test @inferred(energy_shift(rbm, affine_v, affine_h)) ≈ ΔE
    @test energy(rbm, v, h) ≈ energy(white_rbm, v, h) .- ΔE
    @test free_energy(rbm, v) ≈ free_energy(white_rbm, v) .- ΔE

    @test energy_shift_visible(rbm, affine_v) ≈ energy_shift(rbm, affine_v, one(affine_h))
    @test energy_shift_hidden(rbm, affine_h)  ≈ energy_shift(rbm, one(affine_v), affine_h)
end

@testset "whiten!" begin
    rbm = @inferred BinaryWhiteRBM(
        randn(3), randn(2), randn(3,2),
        Affine(randn(3,3), randn(3)),
        Affine(randn(2,2), randn(2))
    )
    @test iszero(energy_shift(rbm, rbm.affine_v, rbm.affine_h))

    v = bitrand(size(visible(rbm))..., 100)
    h = bitrand(size(hidden(rbm))..., 100)
    E = energy(rbm, v, h)
    F = free_energy(rbm, v)
    affine_v = @inferred Affine(randn(3,3), randn(3))
    affine_h = @inferred Affine(randn(2,2), randn(2))

    @test energy_shift_visible(rbm, affine_v) ≈ energy_shift(rbm, affine_v, rbm.affine_h)
    @test energy_shift_hidden(rbm, affine_h) ≈ energy_shift(rbm, rbm.affine_v, affine_h)

    ΔE = energy_shift(rbm, affine_v, affine_h)
    @test energy(whiten(rbm, affine_v, affine_h), v, h) ≈ E .+ ΔE

    whiten!(rbm, affine_v, affine_h)
    @test rbm.affine_v.u ≈ affine_v.u
    @test rbm.affine_h.u ≈ affine_h.u
    @test rbm.affine_v.A ≈ affine_v.A
    @test rbm.affine_h.A ≈ affine_h.A
    @test energy(rbm, v, h) ≈ E .+ ΔE
    @test free_energy(rbm, v) ≈ F .+ ΔE
end

@testset "free energy" begin
    affine_v = Affine(randn(3,3), randn(3))
    affine_h = Affine(randn(2,2), randn(2))
    white_rbm = BinaryWhiteRBM(randn(3), randn(2), randn(3,2), affine_v, affine_h)
    v = bitrand(size(visible(white_rbm))...)
    F = -log(sum(exp(-energy(white_rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test free_energy(white_rbm, v) ≈ F
end

@testset "∂free energy" begin
    affine_v = Affine(randn(3,3), randn(3))
    affine_h = Affine(randn(2,2), randn(2))
    white_rbm = BinaryWhiteRBM(randn(3), randn(2), randn(3,2), affine_v, affine_h)
    v = bitrand(size(visible(white_rbm))...)
    gs = Zygote.gradient(white_rbm) do white_rbm
        mean(free_energy(white_rbm, v))
    end
    ∂ = ∂free_energy(white_rbm, v)
    @test ∂.visible.θ ≈ only(gs).visible.θ
    @test ∂.hidden.θ ≈ only(gs).hidden.θ
    @test ∂.w ≈ only(gs).w
end
