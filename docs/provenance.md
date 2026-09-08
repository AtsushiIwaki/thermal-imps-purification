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

## Extraction inventory and scope

[source-manifest.json](source-manifest.json) records source paths and original SHA-256 values
at the extraction commit, final destination paths and SHA-256 values, transformation reasons,
new files, and excluded source rows. It deliberately does not hash itself; its fingerprint is
retained in the private extraction delivery inventory. Hashes describe file bytes, independently
of subsequent documentation-only Git commits. The manifest is the historical extraction snapshot,
not a continuously updated checksum catalog for later repository changes.

The retained scope is two-site thermal purification/iTEBD, its real and complex backends,
checkpoint/restart and reduced-density-matrix APIs, solver, examples, and validation. Uniform
TDVP and one-site uMPS persistence APIs, executables, and dedicated tests were excluded. Shared
iTEBD helpers have neutral module names. Private project memory, research history, local caches,
and source execution records were not copied. Original source attribution is preserved; no
license grant, copyright owner, repository remote, release, or publication is invented.

## Later curated guidance

The development guidance, templates, and knowledge pages added after extraction are selected
adaptations of project-maintenance material from the same source project. They were rewritten to
refer only to files, tests, and evidence available in this standalone checkout. This later
documentation addition does not alter the original extraction manifest or imply that private
memory and execution records were part of the extraction. The adaptation preserves source
attribution while making no new license grant or publication claim.

The [validation record](validation.md) distinguishes extraction compatibility, scientific
qualification, and an independent cold-cache local macOS build. The prepared CI workflow has
not run on GitHub, and Linux execution has not been verified. This is a verified local candidate,
not a claim of a completed public release.
