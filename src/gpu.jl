CudaRBMs.gpu(rbm::WhiteRBM) = WhiteRBM(
    gpu(rbm.visible), gpu(rbm.hidden), gpu(rbm.w), gpu(rbm.affine_v), gpu(rbm.affine_h)
)

CudaRBMs.cpu(rbm::WhiteRBM) = WhiteRBM(
    cpu(rbm.visible), cpu(rbm.hidden), cpu(rbm.w), cpu(rbm.affine_v), cpu(rbm.affine_h)
)

CudaRBMs.gpu(a::Affine) = Affine(gpu(a.A), gpu(a.u))
CudaRBMs.gpu(a::CenterAffine) = CenterAffine(gpu(a.u))
CudaRBMs.gpu(a::StdizeAffine) = StdizeAffine(gpu(a.A), gpu(a.u))

CudaRBMs.cpu(a::Affine) = Affine(cpu(a.A), cpu(a.u))
CudaRBMs.cpu(a::CenterAffine) = CenterAffine(cpu(a.u))
CudaRBMs.cpu(a::StdizeAffine) = StdizeAffine(cpu(a.A), cpu(a.u))
