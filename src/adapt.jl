# Teach Adapt.jl to recurse through our structs, so that `adapt(CuArray, rbm)`, `cu(rbm)`,
# `adapt(Array, rbm)`, and RestrictedBoltzmannMachines' `gpu` / `cpu` (defined in its CUDA
# extension) work out of the box.
Adapt.@adapt_structure Affine
Adapt.@adapt_structure CenterAffine
Adapt.@adapt_structure StdizeAffine
Adapt.@adapt_structure WhiteRBM
