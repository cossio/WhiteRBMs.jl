CudaRBMs.gpu(rbm::WhiteRBM) = CenteredRBM(
    gpu(rbm.visible), gpu(rbm.hidden), gpu(rbm.w), gpu(rbm.affine_v), gpu(rbm.affine_h)
)

CudaRBMs.cpu(rbm::WhiteRBM) = CenteredRBM(
    cpu(rbm.visible), cpu(rbm.hidden), cpu(rbm.w), cpu(rbm.affine_v), cpu(rbm.affine_h)
)

CudaRBMs.gpu(a::Affine) = Affine(gpu(a.A), gpu(a.u))
CudaRBMs.gpu(a::CenterAffine) = CenterAffine(gpu(a.u))
CudaRBMs.gpu(a::StdizeAffine) = StdizeAffine(gpu(a.A), gpu(a.u))
