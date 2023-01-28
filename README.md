# WhiteRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/WhiteRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/WhiteRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/WhiteRBMs.jl/branch/master/graph/badge.svg?token=90I3AJIZIG)](https://codecov.io/gh/cossio/WhiteRBMs.jl)

Train and sample whitned [Restricted Boltzmann machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia.

## Installation

This package is not registered.
Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/WhiteRBMs.jl")
```

This package does not export any symbols.

## Related

Builds upon the [RestrictedBoltzmannMachines](https://github.com/cossio/RestrictedBoltzmannMachines.jl) Julia package, which defines the `RBM` and layer types.