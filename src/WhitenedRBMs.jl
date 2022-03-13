module WhitenedRBMs

import Random
import Statistics
import LinearAlgebra
import ValueHistories
import Flux
import RestrictedBoltzmannMachines as RBMs

using LinearAlgebra: Diagonal, cholesky, diagm, Symmetric
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: RBM, AbstractLayer
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: visible, hidden, weights, flatten
using RestrictedBoltzmannMachines: inputs_v_to_h, inputs_h_to_v
using RestrictedBoltzmannMachines: batchmean, batchvar

include("affine.jl")
include("whiterbm.jl")
include("whiten.jl")
include("data.jl")
include("binary_white_rbm.jl")
include("layers.jl")
include("train/pcd.jl")

end # module
