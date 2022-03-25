#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

#=
Some references

<https://www.jmlr.org/beta/papers/v17/14-237.html>,

<http://www.cs.toronto.edu/~tang/papers/RbmZM.pdf>,

<https://doi.org/10.1007/978-3-642-35289-8_3>

=#

import CairoMakie
import Makie
import MLDatasets
import Flux
import RestrictedBoltzmannMachines as RBMs
import WhitenedRBMs as WhiteRBMs
using Statistics: mean, var, std
using Random: bitrand
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: BinaryRBM
using WhitenedRBMs: whiten, blacken, whiten_visible_from_data
nothing #hide

#=
Useful function to plot MNIST digits.
=#

"""
    imggrid(A)

Given a four dimensional tensor `A` of size `(width, height, ncols, nrows)`
containing `width x height` images in a grid of `nrows x ncols`, this returns
a matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap
to display all images.
"""
function imggrid(A::AbstractArray{<:Any,4})
    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))
end

#=
Now load the MNIST dataset.
=#

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
digit = 0 # the digit we work with
train_x = Array{Float}(train_x[:, :, train_y .== digit] .> 0.5)
train_y = train_y[train_y .== digit]
println(length(train_y), " train samples")

#=
Initialize and train a normal RBM
=#

rbm = BinaryRBM(Float, (28,28), 200)
RBMs.initialize!(rbm, train_x)
rbm = whiten_visible_from_data(whiten(rbm), train_x, WhiteRBMs.Identity(); ϵ=1f-3)
rbm.hidden.θ .= 0
nothing #hide

# Train init

batchsize = 256
optim = Flux.ADAM()
vm = bitrand(28, 28, batchsize) # fantasy chains
history = MVHistory()
push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))
nothing #hide

# Pseudolikelihood before training

mean(@time RBMs.log_pseudolikelihood(rbm, train_x))

# Train

@time for epoch in 1:100 # track pseudolikelihood every 5 epochs
    WhiteRBMs.pcd!(
        rbm, train_x; epochs=5, vm, history, batchsize, optim, ϵv=1f-3,
        transform_v=WhiteRBMs.Identity(), transform_h=WhiteRBMs.Identity()
    )
    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))
end
nothing #hide

# Convert to equivalent RBM (without affine transforms)

rbm = WhiteRBMs.blacken(rbm)
nothing #hide

# Plot log-pseudolikelihood of train data during learning.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="epochs", ylabel="pseudolikelihood")
Makie.lines!(ax, get(history, :lpl)...)
fig

# Seconds per epoch.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
Makie.lines!(ax, get(history, :Δt)...)
fig

# Log-pseudolikelihood vs. computation time instead of epoch count.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="seconds", ylabel="pseudolikelihood")
Makie.lines!(ax, cumsum([0; get(history, :Δt)[2]])[1:5:end], get(history, :lpl)[2])
fig

# Now we do the Gibbs sampling to generate RBM digits.

nrows, ncols = 10, 15
nsteps = 5000
fantasy_F = zeros(nrows*ncols, nsteps)
fantasy_x = bitrand(28,28,nrows*ncols)
fantasy_F[:,1] .= RBMs.free_energy(rbm, fantasy_x)
@time for t in 2:nsteps
    fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x)
    fantasy_F[:,t] .= RBMs.free_energy(rbm, fantasy_x)
end
nothing #hide

# Check equilibration of sampling

fig = Makie.Figure(resolution=(400,300))
ax = Makie.Axis(fig[1,1], xlabel="sampling time", ylabel="free energy")
fantasy_F_μ = vec(mean(fantasy_F; dims=1))
fantasy_F_σ = vec(std(fantasy_F; dims=1))
Makie.band!(ax, 1:nsteps, fantasy_F_μ - fantasy_F_σ/2, fantasy_F_μ + fantasy_F_σ/2)
Makie.lines!(ax, 1:nsteps, fantasy_F_μ)
fig

# Plot the resulting samples.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig
