module WhiteRBMs

import Random
import Statistics
import LinearAlgebra
import Adapt
import RestrictedBoltzmannMachines

using Optimisers: AbstractRule, setup, update!, Adam
using FillArrays: Falses
using LinearAlgebra: Diagonal, cholesky, diag, Symmetric, I
using Statistics: mean
using RestrictedBoltzmannMachines: RBM, AbstractLayer, BinaryRBM, Binary,
    moments_from_samples, moments_from_inputs, mean_from_moments, batchmean_moments,
    infinite_minibatches, ∂RBM, ∂energy_from_moments, cgf,
    flatten, energy, free_energy,
    inputs_h_from_v, inputs_v_from_h, sample_from_inputs, sample_v_from_v,
    ∂free_energy, ∂interaction_energy, ∂regularize!,
    mean_from_inputs, var_from_inputs,
    total_mean_from_inputs, total_meanvar_from_inputs,
    interaction_energy, log_pseudolikelihood,
    batchmean, batchvar, batchcov,
    uniform_wts, validate_wts, zerosum!,
    shift_fields, shift_fields!,
    cpu, gpu

include("affine.jl")
include("whiterbm.jl")
include("whiten.jl")
include("data.jl")
include("binary_white_rbm.jl")
include("train/pcd.jl")
include("adapt.jl")

end # module
