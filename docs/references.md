# Upstream software and method references

This project uses the software and methods listed below. These scholarly references record
technical provenance; license and redistribution notices are in
[Third-party notices](../THIRD_PARTY_NOTICES.md). This document does not define or request a
scholarly citation for this repository.

## Software

- **tensor4all-rs**, revision
  [`880032daea2a84d7ddfe84a2594a6d160631afc2`](https://github.com/tensor4all/tensor4all-rs/tree/880032daea2a84d7ddfe84a2594a6d160631afc2).
  The project uses `tensor4all-core` for indexed tensors, contractions, and singular-value
  decompositions, and `tensor4all-hdf5` for ITensor-compatible tensor serialization. The
  [tensor4all provenance and citation policy](https://github.com/tensor4all/tensor4all-rs/blob/880032daea2a84d7ddfe84a2594a6d160631afc2/docs/PROVENANCE_AND_CITATION_POLICY.md)
  distinguishes these components from its DMRG, TDVP, and tensor cross-interpolation components;
  this project does not use those algorithms.
- **tenferro-rs**, revision
  [`f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266`](https://github.com/tensor4all/tenferro-rs/tree/f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266).
  Its CPU tensor, contraction, and linear-algebra crates are resolved transitively as the
  tensor4all backend.
- **ITensor / ITensors.jl** supplies design and index/file-format provenance for the tensor4all
  components used here. M. Fishman, S. R. White, and E. M. Stoudenmire, “The ITensor Software
  Library for Tensor Network Calculations,” *SciPost Physics Codebases* **4** (2022),
  [doi:10.21468/SciPostPhysCodeb.4](https://doi.org/10.21468/SciPostPhysCodeb.4).
- **HDF5** checkpoint I/O uses the Rust wrapper
  [`hdf5-metno` 0.12.6](https://crates.io/crates/hdf5-metno/0.12.6), built with its `static`
  feature. The bundled native library is separately versioned as **HDF5 2.1.0**. The HDF Group,
  *Hierarchical Data Format, version 5*, [HDF5 project](https://www.hdfgroup.org/solutions/hdf5/).

## Numerical methods

The implemented public scope is a two-site infinite-MPS purification evolved in imaginary time
with first- or second-order iTEBD. The relevant primary method references are:

- G. Vidal, “Classical Simulation of Infinite-Size Quantum Lattice Systems in One Spatial
  Dimension,” *Physical Review Letters* **98**, 070201 (2007),
  [doi:10.1103/PhysRevLett.98.070201](https://doi.org/10.1103/PhysRevLett.98.070201).
- R. Orús and G. Vidal, “Infinite Time-Evolving Block Decimation Algorithm Beyond Unitary
  Evolution,” *Physical Review B* **78**, 155117 (2008),
  [doi:10.1103/PhysRevB.78.155117](https://doi.org/10.1103/PhysRevB.78.155117).
- F. Verstraete, J. J. García-Ripoll, and J. I. Cirac, “Matrix Product Density Operators:
  Simulation of Finite-Temperature and Dissipative Systems,” *Physical Review Letters* **93**,
  207204 (2004),
  [doi:10.1103/PhysRevLett.93.207204](https://doi.org/10.1103/PhysRevLett.93.207204).
