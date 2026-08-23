# WhiteRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/WhiteRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/WhiteRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/WhiteRBMs.jl/branch/master/graph/badge.svg?token=90I3AJIZIG)](https://codecov.io/gh/cossio/WhiteRBMs.jl)

Train and sample whitened Restricted Boltzmann machines in Julia.

## What this package does

This package builds on top of [RestrictedBoltzmannMachines.jl](https://github.com/cossio/RestrictedBoltzmannMachines.jl), extending its `standardize` functionality.

Training RBMs by gradient descent is known to work better when the units are centered and rescaled to have zero mean and unit variance (see e.g. [Montavon & Müller 2012](https://doi.org/10.1007/978-3-642-35289-8_33), or [Melchior et al 2016](https://jmlr.org/papers/v17/14-237.html)). This is what `standardize` in RestrictedBoltzmannMachines.jl does: it applies a *per-unit* (diagonal) affine transformation to the units.

WhiteRBMs.jl goes one step further: besides standardizing the units, it also **decorrelates** them (*whitening*). Instead of a diagonal rescaling, the visible and hidden units are passed through general affine transformations

```math
\tilde{\mathbf{v}} = \mathbb{A}_v (\mathbf{v} - \mathbf{u}_v), \qquad
\tilde{\mathbf{h}} = \mathbb{A}_h (\mathbf{h} - \mathbf{u}_h)
```

where the matrices $\mathbb{A}_v, \mathbb{A}_h$ can be full (non-diagonal) matrices, chosen so that the transformed units have zero mean and *identity covariance* (e.g. taking $\mathbb{A}$ as the inverse Cholesky factor of the covariance matrix of the units). The whitened RBM assigns the energy

```math
E(\mathbf{v}, \mathbf{h}) = E_v(\mathbf{v}) + E_h(\mathbf{h}) - \tilde{\mathbf{v}}^\top \tilde{\mathbb{W}} \tilde{\mathbf{h}}
```

where only the interaction term sees the whitened units.

The central type is `WhiteRBM`, which wraps an `RBM` together with the two affine transforms. Three kinds of transforms are supported, recovering the simpler tricks as special cases:

* `CenterAffine`: centering only, $\mathbf{v} \mapsto \mathbf{v} - \mathbf{u}$ (the "centering trick");
* `StdizeAffine`: diagonal $\mathbb{A}$, equivalent to `standardize` in RestrictedBoltzmannMachines.jl;
* `Affine`: full matrix $\mathbb{A}$, standardizing *and* decorrelating the units.

The functions `whiten` and `blacken` convert between a plain `RBM` and a `WhiteRBM` (in both directions) while preserving the encoded distribution: the energies assigned by the two models differ only by a constant, so whitening is a *reparameterization* that changes the gradient geometry of training, not the model class. This package also provides a `pcd!` method for `WhiteRBM` that trains by persistent contrastive divergence while estimating the affine transforms from data statistics (updating the hidden transform during training, with damping) and reparameterizing the model on the fly.

## Installation

This package is not registered. Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/WhiteRBMs.jl")
```

This package does not export any symbols.

## Related

* https://github.com/cossio/RestrictedBoltzmannMachines.jl.