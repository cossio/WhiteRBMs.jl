import MKL
using SafeTestsets: @safetestset

@time @safetestset "layers" begin include("layers.jl") end
@time @safetestset "affine" begin include("affine.jl") end
@time @safetestset "binary_white_rbm" begin include("binary_white_rbm.jl") end
