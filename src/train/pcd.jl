function RestrictedBoltzmannMachines.pcd!(
    rbm::WhiteRBM,
    data::AbstractArray;
    batchsize::Int = 1,
    shuffle::Bool = true,
    iters::Int = 1, # number of gradient updates
    wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data), # data weights
    steps::Int = 1, # MC steps to update fantasy chains
    optim::AbstractRule = Adam(),
    moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers

    # "pseudocount" for estimating variances of v and h, and damping of the affine updates
    damping::Real = 1 // 100, ϵv::Real = 0, ϵh::Real = 0,

    # init fantasy chains
    vm::AbstractArray = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., min(batchsize, size(data)[end]))),

    callback = Returns(nothing), # called for every batch

    # parameters to optimize
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps), # initialize optimiser state
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert 0 ≤ damping ≤ 1
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    size(data, ndims(data)) > 0 ||
        throw(ArgumentError("data must contain at least one sample"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    validate_wts(wts)
    wts_mean = mean(wts)
    batchsize = min(batchsize, length(wts))

    whiten_visible_from_data!(rbm, data; wts, ϵ = ϵv)
    zerosum && zerosum!(RBM(rbm))

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # positive phase
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)

        # negative phase: update persistent fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂m = ∂free_energy(rbm, vm)

        # weighted minibatch bias correction, in the gradient eltype
        batch_weight = convert(float(real(eltype(∂d.w))), mean(wd) / wts_mean)
        ∂ = (∂d - ∂m) * batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # update hidden affine transform
        whiten_hidden_from_v!(rbm, vd; wts = wd, damping, ϵ = ϵh)
        zerosum && zerosum!(RBM(rbm))

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm)
    end
    return state, ps
end

RestrictedBoltzmannMachines.∂regularize!(∂::∂RBM, rbm::WhiteRBM; kwargs...) = ∂regularize!(∂, RBM(rbm); kwargs...)
