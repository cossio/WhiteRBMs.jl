import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: energy
using Test: @test, @testset, @inferred
using WhiteRBMs: shift_fields, shift_fields!

function energy_shift(offset::AbstractArray, x::AbstractArray)
    @assert size(offset) == size(x)[1:ndims(offset)]
    if ndims(offset) == ndims(x)
        return -sum(offset .* x)
    elseif ndims(offset) < ndims(x)
        ΔE = -sum(offset .* x; dims=1:ndims(offset))
        return reshape(ΔE, size(x)[(ndims(offset) + 1):end])
    end
end

@testset "shift_fields" begin
    N = (3, 4)
    layers = (
        RBMs.Binary(randn(N...)),
        RBMs.Spin(randn(N...)),
        RBMs.Potts(randn(N...)),
        RBMs.Gaussian(randn(N...), rand(N...)),
        RBMs.ReLU(randn(N...), rand(N...)),
        RBMs.dReLU(randn(N...), randn(N...), rand(N...), rand(N...)),
        RBMs.pReLU(randn(N...), rand(N...), randn(N...), rand(N...) .- 0.5),
        RBMs.xReLU(randn(N...), rand(N...), randn(N...), randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = RBMs.sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        layer_shifted = @inferred shift_fields(layer, offset)
        @test energy(layer_shifted, x) ≈ energy(layer, x) + energy_shift(offset, x)
    end
end


@testset "shift_fields!" begin
    N = (3, 4)
    layers = (
        RBMs.Binary(randn(N...)),
        RBMs.Spin(randn(N...)),
        RBMs.Potts(randn(N...)),
        RBMs.Gaussian(randn(N...), rand(N...)),
        RBMs.ReLU(randn(N...), rand(N...)),
        RBMs.dReLU(randn(N...), randn(N...), rand(N...), rand(N...)),
        RBMs.pReLU(randn(N...), rand(N...), randn(N...), rand(N...) .- 0.5),
        RBMs.xReLU(randn(N...), rand(N...), randn(N...), randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = RBMs.sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        E = energy(layer, x)
        @inferred shift_fields!(layer, offset)
        @test energy(layer, x) ≈ E + energy_shift(offset, x)
    end
end
