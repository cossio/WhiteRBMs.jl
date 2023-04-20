function RestrictedBoltzmannMachines.pcd!(
    rbm::AffineRBM,
    data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1, # number of gradient updates
    optim::AbstractRule = Adam(),
    wts = nothing,
    steps::Int = 1,
    moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer,
    vm::AbstractArray = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
    damping::Real = 1//100,
    ϵv::Real = 0, ϵh::Real = 0, # "pseudocount" for estimating variances of v and h
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps), # initialize optimiser state
    callback = Returns(nothing), # called for every batch
    zerosum::Bool = true
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    @assert 0 ≤ damping ≤ 1

    zerosum && zerosum!(rbm)

    whiten_visible_from_data!(rbm, data; wts, ϵ = ϵv)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # update hidden affine transform
        whiten_hidden_from_v!(rbm, vd; damping, wts=wd, ϵ=ϵh)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # correct weighted minibatch bias
        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        zerosum && zerosum!(rbm)

        callback(; rbm, optim, iter, vm, vd, wd)
    end

    return rbm
end

RestrictedBoltzmannMachines.∂regularize!(∂::∂RBM, rbm::WhiteRBM; kwargs...) = ∂regularize!(∂, RBM(rbm); kwargs...)
