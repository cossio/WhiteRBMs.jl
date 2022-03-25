function pcd!(
    white_rbm::AffineRBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm::AbstractArray = initial_fantasy_v(white_rbm, data, batchsize),
    damping::Real = 1//100, ϵv::Real = 0, ϵh::Real = 0,
    transform_v::AbstractTransform = Whiten(),
    transform_h::AbstractTransform = Stdize()
)
    @assert size(data) == (size(visible(white_rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    @assert 0 ≤ damping ≤ 1

    whiten_visible_from_data!(white_rbm, data, transform_v; wts, ϵ = ϵv)
    stats = RBMs.suffstats(visible(white_rbm), data; wts)

    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm .= RBMs.sample_v_from_v(white_rbm, vm; steps)
            # update hidden affine transform
            if !(transform_h isa Identity)
                inputs = RBMs.inputs_v_to_h(white_rbm, vd)
                whiten_hidden_from_inputs!(white_rbm, inputs, transform_h; damping, wts=wd, ϵ=ϵh)
            end
            # compute contrastive divergence gradient
            ∂ = RBMs.∂contrastive_divergence(white_rbm, vd, vm; wd, stats)
            # compute parameter step according to optimization algorithm
            Δ = RBMs.update!(∂, white_rbm, optim)
            # update parameters using gradient
            RBMs.update!(white_rbm, Δ)
            # store gradient and update step norms
            push!(history, :∂, RBMs.gradnorms(∂))
            push!(history, :Δ, RBMs.gradnorms(Δ))
        end

        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end

    return white_rbm, history
end

function RBMs.∂contrastive_divergence(
    rbm::WhiteRBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    stats = RBMs.suffstats(visible(rbm), vd; wts = wd),
)
    ∂d = RBMs.∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = RBMs.∂free_energy(rbm, vm; wts = wm)
    ∂ = RBMs.subtract_gradients(∂d, ∂m)
    return ∂
end

function RBMs.update!(white_rbm::WhiteRBM, ∂::NamedTuple)
    RBMs.update!(RBMs.RBM(white_rbm), ∂)
    return white_rbm
end

RBMs.update!(∂::NamedTuple, rbm::WhiteRBM, optim) = RBMs.update!(∂, RBM(rbm), optim)

function initial_fantasy_v(rbm, data::AbstractArray, batchsize::Int)
    inputs = falses(size(data)[1:(end - 1)]..., batchsize)
    vm = RBMs.transfer_sample(visible(rbm), inputs)
    return oftype(data, vm)
end
