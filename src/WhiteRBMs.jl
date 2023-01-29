module WhiteRBMs

import Random
import Statistics
import LinearAlgebra
import RestrictedBoltzmannMachines as RBMs
import RestrictedBoltzmannMachines

using Optimisers: AbstractRule, setup, update!, Adam
using FillArrays: Falses, Zeros
using LinearAlgebra: Diagonal, cholesky, diagm, Symmetric, I
using RestrictedBoltzmannMachines: RBM, AbstractLayer, BinaryRBM,
    moments_from_samples, infinite_minibatches, ∂RBM, cgf, grad2ave, ∂cgfs, wmean,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU, ∂energy_from_moments,
    flatten, energy, free_energy,
    inputs_h_from_v, inputs_v_from_h, sample_from_inputs,
    sample_v_from_v_once, sample_h_from_h_once, sample_h_from_v, sample_v_from_h, sample_v_from_v,
    ∂energy, ∂free_energy, ∂interaction_energy,
    mean_from_inputs, var_from_inputs, mode_from_inputs,
    interaction_energy, log_pseudolikelihood,
    batchmean, batchvar, batchcov

include("affine.jl")
include("whiterbm.jl")
include("whiten.jl")
include("data.jl")
include("binary_white_rbm.jl")
include("layers.jl")
include("train/pcd.jl")

end # module
