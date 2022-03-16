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
using Statistics: mean, var
using Random: bitrand
using RestrictedBoltzmannMachines: BinaryRBM
using ValueHistories: MVHistory
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
tests_x, tests_y = MLDatasets.MNIST.testdata()
digit = 8
train_x = Array{Float}(train_x[:, :, train_y .== digit] .> 0.5)
tests_x = Array{Float}(tests_x[:, :, tests_y .== digit] .> 0.5)
train_y = train_y[train_y .== digit]
tests_y = tests_y[tests_y .== digit]
train_nsamples = length(train_y)
tests_nsamples = length(tests_y)
(train_nsamples, tests_nsamples)

#=
Initialize and train a whitened RBM
=#

rbm = BinaryRBM(Float, (28,28), 200)
RBMs.initialize!(rbm, train_x)
rbm_w = WhiteRBMs.whiten_visible_from_data(WhiteRBMs.whiten(rbm), train_x; ϵ=1f-3)
batchsize = 256
optim = Flux.ADAM()
vm = bitrand(28, 28, batchsize) # fantasy chains
history_w = MVHistory()
push!(history_w, :lpl, mean(RBMs.log_pseudolikelihood(rbm_w, train_x)))
nothing #hide

# Train

@time for epoch in 1:100 # track pseudolikelihood every 5 epochs
    RBMs.pcd!(rbm_w, train_x; epochs=5, vm, history=history_w, batchsize, optim, ϵv=1f-3, whiten_h=false)
    push!(history_w, :lpl, mean(RBMs.log_pseudolikelihood(rbm_w, train_x)))
end
nothing #hide

# Convert to equivalent RBM (without affine transforms)

rbm = WhiteRBMs.blacken(rbm_w)
nothing #hide

#=
For comparison, we also train a normal (not whitened) RBM.
=#

rbm_u = BinaryRBM(Float, (28,28), 200)
RBMs.initialize!(rbm_u, train_x)
vm = bitrand(28, 28, batchsize)
history_u = MVHistory()
push!(history_u, :lpl, mean(RBMs.log_pseudolikelihood(rbm_u, train_x)))
@time for epoch in 1:100 # track pseudolikelihood every 5 epochs
    RBMs.pcd!(rbm_u, train_x; epochs=5, vm, history=history_u, batchsize, optim)
    push!(history_u, :lpl, mean(RBMs.log_pseudolikelihood(rbm_u, train_x)))
end
nothing #hide

# Plot log-pseudolikelihood of train data during learning.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="epochs", ylabel="pseudolikelihood")
Makie.lines!(ax, get(history_u, :lpl)..., label="normal")
Makie.lines!(ax, get(history_w, :lpl)..., label="whitened")
Makie.axislegend(ax, position=:rb)
fig

# Seconds per epoch.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
Makie.lines!(ax, get(history_u, :Δt)..., label="normal")
Makie.lines!(ax, get(history_w, :Δt)..., label="whitened")
Makie.axislegend(ax, position=:rt)
fig

# Log-pseudolikelihood vs. computation time instead of epoch count.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="seconds", ylabel="pseudolikelihood")
Makie.lines!(ax, cumsum([0; get(history_u, :Δt)[2]])[1:5:end], get(history_u, :lpl)[2], label="normal")
Makie.lines!(ax, cumsum([0; get(history_w, :Δt)[2]])[1:5:end], get(history_w, :lpl)[2], label="whitened")
Makie.axislegend(ax, position=:rb)
fig

# Now we do the Gibbs sampling to generate RBM digits.

nrows, ncols = 10, 15
@time fantasy_x_w = RBMs.sample_v_from_v(rbm_w, bitrand(28,28,nrows*ncols); steps=10000)
@time fantasy_x_u = RBMs.sample_v_from_v(rbm_u, bitrand(28,28,nrows*ncols); steps=10000)
nothing #hide

# Plot the resulting samples.

# Normal RBM.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x_u, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig

# Whitened RBM.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x_w, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig
