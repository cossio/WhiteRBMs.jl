import Zygote

using Test: @test, @testset, @inferred
using Random: bitrand
using Statistics: mean
using LinearAlgebra: I
using RestrictedBoltzmannMachines: RBM, BinaryRBM,
    energy, interaction_energy, free_energy, ∂free_energy,
    inputs_h_from_v, inputs_v_from_h
using WhiteRBMs: BinaryWhiteRBM, WhiteRBM, Affine,
    whiten, blacken, whiten_visible, whiten_hidden,
    whiten!, whiten_visible!, whiten_hidden!,
    energy_shift, energy_shift_visible, energy_shift_hidden

@testset "whiten / blacken" begin
    rbm = @inferred BinaryRBM(randn(3), randn(2), randn(3,2))
    affine_v = @inferred Affine(randn(3,3), randn(3))
    affine_h = @inferred Affine(randn(2,2), randn(2))
    white_rbm = @inferred whiten(rbm, affine_v, affine_h)
    @test white_rbm.affine_v.u ≈ affine_v.u
    @test white_rbm.affine_h.u ≈ affine_h.u
    @test white_rbm.affine_v.A ≈ affine_v.A
    @test white_rbm.affine_h.A ≈ affine_h.A
    @test affine_v.A' * white_rbm.w * affine_h.A ≈ rbm.w
    @test rbm.visible.θ ≈ white_rbm.visible.θ - rbm.w * affine_h.u
    @test rbm.hidden.θ ≈ white_rbm.hidden.θ - rbm.w' * affine_v.u
    @test blacken(rbm) == rbm

    brbm = @inferred blacken(white_rbm)
    wrbm = @inferred whiten(white_rbm)
    @test brbm.visible.θ ≈ wrbm.visible.θ ≈ rbm.visible.θ
    @test brbm.hidden.θ ≈ wrbm.hidden.θ ≈ rbm.hidden.θ
    @test brbm.w ≈ wrbm.w ≈ rbm.w

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
    @test white_rbm_new.visible.θ ≈ white_rbm_expected.visible.θ
    @test white_rbm_new.hidden.θ ≈ white_rbm_expected.hidden.θ
    @test white_rbm_new.w ≈ white_rbm_expected.w

    # whiten_hidden
    affine_h_new = @inferred Affine(randn(2,2), randn(2))
    white_rbm_new = @inferred whiten_hidden(white_rbm, affine_h_new)
    @test white_rbm_new.affine_v.u == affine_v.u
    @test white_rbm_new.affine_v.A == affine_v.A
    @test white_rbm_new.affine_h.u == affine_h_new.u
    @test white_rbm_new.affine_h.A == affine_h_new.A
    white_rbm_expected = whiten(white_rbm, affine_v, affine_h_new)
    @test white_rbm_new.visible.θ ≈ white_rbm_expected.visible.θ
    @test white_rbm_new.hidden.θ ≈ white_rbm_expected.hidden.θ
    @test white_rbm_new.w ≈ white_rbm_expected.w

    wrbm1 = whiten_visible(whiten_hidden(rbm, affine_h), affine_v)
    wrbm2 = whiten_hidden(whiten_visible(rbm, affine_v), affine_h)
    for wrbm in (wrbm1, wrbm2)
        @test wrbm.affine_v.u == affine_v.u
        @test wrbm.affine_v.A == affine_v.A
        @test wrbm.affine_h.u == affine_h.u
        @test wrbm.affine_h.A == affine_h.A
        @test wrbm.visible.θ ≈ white_rbm.visible.θ
        @test wrbm.hidden.θ ≈ white_rbm.hidden.θ
        @test wrbm.w ≈ white_rbm.w
    end
end

@testset "energy invariance" begin
    affine_v = @inferred Affine(randn(3,3), randn(3))
    affine_h = @inferred Affine(randn(2,2), randn(2))
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    white_rbm = @inferred whiten(rbm, affine_v, affine_h)
    v = bitrand(size(rbm.visible)..., 100)
    h = bitrand(size(rbm.hidden)..., 100)

    @test energy(rbm.visible, v) ≈ energy(white_rbm.visible, v) + v' * rbm.w * affine_h.u
    @test energy(rbm.hidden, h) ≈ energy(white_rbm.hidden, h) + h' * rbm.w' * affine_v.u

    @test inputs_h_from_v(rbm, v .- affine_v.u) ≈ @inferred inputs_h_from_v(white_rbm, v)
    @test inputs_v_from_h(rbm, h .- affine_h.u) ≈ @inferred inputs_v_from_h(white_rbm, h)

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

    v = bitrand(size(rbm.visible)..., 100)
    h = bitrand(size(rbm.hidden)..., 100)
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
    v = bitrand(size(white_rbm.visible)...)
    F = -log(sum(exp(-energy(white_rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test free_energy(white_rbm, v) ≈ F
end

@testset "∂free energy" begin
    affine_v = Affine(randn(3,3), randn(3))
    affine_h = Affine(randn(2,2), randn(2))
    white_rbm = BinaryWhiteRBM(randn(3), randn(2), randn(3,2), affine_v, affine_h)
    v = bitrand(size(white_rbm.visible)...)
    gs = Zygote.gradient(white_rbm) do white_rbm
        mean(free_energy(white_rbm, v))
    end
    ∂ = ∂free_energy(white_rbm, v)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end
