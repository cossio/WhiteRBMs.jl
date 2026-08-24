# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- This CHANGELOG file.
- Require RestrictedBoltzmannMachines v7, dropping support for older versions.
- Use `RBMs.pcd!`. `pcd!(white_rbm, data)` now mirrors the upstream trainer (weighted data, regularization and zerosum keywords) and returns `(state, ps)`.
- `∂free_energy(white_rbm, v)` returns a `RBMs.∂RBM` gradient.
- Drop the CudaRBMs dependency: RestrictedBoltzmannMachines now provides `gpu`/`cpu` via its CUDA extension. WhiteRBMs types support Adapt.jl, so `gpu`, `cpu`, `cu` and `adapt` recurse through them.
- `shift_fields` and `shift_fields!` are now provided by RestrictedBoltzmannMachines (re-exported here unchanged).

## v0.1.0
