# Provenance

Thermal iMPS Purification was extracted from Git commit
`55ed6ee4c1940dbfd2be13374ce3caf45169f515`. The extraction copied an audited manifest of the
two-site purification/iTEBD library, solver, configurations, tests, and examples, then removed the
unrelated application surfaces. Package/import names were changed to
`thermal-imps-purification` / `thermal_imps_purification`; retained numerical kernels preserve
their source content and conventions.

The Rust lockfile is part of the reproducibility record. In particular, `tensor4all` is pinned to
`880032daea2a84d7ddfe84a2594a6d160631afc2` and the resolved `tenferro` source to
`f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266`. Other dependency versions, checksums, features, and
sources are fixed by `Cargo.lock`. `serde_json/float_roundtrip` is enabled for exact checkpoint
metadata decoding. `hdf5-metno` uses its static feature and builds the vendored HDF5 C source.

Reproduction starts from this repository and its lockfile with Rust 1.96.1, a C toolchain, and
CMake 3.26 or newer. No sibling source checkout, `refs/` tree, local dependency patch, or symlink
is required. See the README for locked build and executable smoke commands. Repository hosting,
licensing, release tags, and registry publication are separate decisions and are not asserted by
this document.
