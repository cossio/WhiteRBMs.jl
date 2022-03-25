using Test: @test, @testset, @inferred
using LinearAlgebra: I, norm, Symmetric, Diagonal, LowerTriangular
using WhitenedRBMs: Affine, whitening_transform, whitening_transform!

@testset "affine" begin
    x = randn(5, 7)
    t = Affine(randn(5,5), randn(5))
    s = Affine(randn(5,5), randn(5))

    @test Affine(5).A ≈ I
    @test iszero(Affine(5).u)
    @test Affine(5) * x ≈ x

    @test t * x ≈ t.A * (x .- t.u)
    @test t \ (t * x) ≈ t * (t \ x) ≈ x

    @test (s * t) * x ≈ s * (t * x)
    @test (s \ t) * x ≈ s \ (t * x)
    @test (t \ t) * x ≈ x

    @test (s + t) * x ≈ s * x + t * x
    @test (s - t) * x ≈ s * x - t * x

    @test (2 * t) * x ≈ 2 * (t * x)
    @test (t * 2) * x ≈ t * (2 * x)
    @test (-t) * x ≈ -(t * x)

    @test inv(t) * x ≈ t \ x
    @test inv(t) * (t * x) ≈ (inv(t) * t) * x ≈ x

    @test (inv(t) * t).A ≈ I
    @test norm((inv(t) * t).u) < 1e-10
end

@testset "whitening_transform" begin
    μ = randn(5)
    C = randn(5,5)
    C = C * C'
    affine = whitening_transform(μ, C)
    @test norm(affine * μ) < 1e-10
    @test affine.A * C * affine.A' ≈ I
    @test affine.A' * affine.A ≈ inv(C)

    affine = whitening_transform(μ)
    @test affine.A ≈ I
    @test affine.u ≈ μ
    @test norm(affine * μ) < 1e-10

    affine = whitening_transform(C)
    @test affine.A * C * affine.A' ≈ I
    @test iszero(affine.u)

    affine = whitening_transform(C) * whitening_transform(μ)
    @test norm(affine * μ) < 1e-10
    @test affine.A * C * affine.A' ≈ I
end


@testset "whitening_transform!" begin
    μ = randn(5)
    C = randn(5,5)
    C = C * C'
    affine = whitening_transform!(Affine(LowerTriangular(randn(5,5)), randn(5)), μ, C)
    @test norm(affine * μ) < 1e-10
    @test affine.A * C * affine.A' ≈ I
    @test affine.A' * affine.A ≈ inv(C)

    affine = whitening_transform!(Affine(Diagonal(randn(5)), randn(5)), μ)
    @test affine.A ≈ I
    @test affine.u ≈ μ
    @test norm(affine * μ) < 1e-10

    affine = whitening_transform!(Affine(LowerTriangular(randn(5,5)), randn(5)), C)
    @test affine.A * C * affine.A' ≈ I
    @test iszero(affine.u)

    C = Diagonal(rand(5))
    affine = whitening_transform!(Affine(Diagonal(randn(5)), randn(5)), μ, C)
    @test norm(affine * μ) < 1e-10
    @test affine.A * C * affine.A' ≈ I
    @test affine.A' * affine.A ≈ inv(C)
end
