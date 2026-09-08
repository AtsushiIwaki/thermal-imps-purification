# Third-party notices

The root MIT License covers project-authored code, documentation, test harnesses, and knowledge
material. Copied upstream material retains its upstream terms. Scholarly sources are listed
separately in [docs/references.md](docs/references.md).

| Material | Pinned source and terms | Included text |
| --- | --- | --- |
| `tensor4all` | `tensor4all-rs` revision `880032daea2a84d7ddfe84a2594a6d160631afc2`; MIT | [`tensor4all-MIT.txt`](licenses/tensor4all-MIT.txt), root `LICENSE-MIT` |
| `tensor4all-treetn` | same revision; MIT **AND** Apache-2.0 | tensor4all MIT above **and** [`tensor4all-treetn-APACHE-2.0.txt`](licenses/tensor4all-treetn-APACHE-2.0.txt), `crates/tensor4all-treetn/LICENSE-APACHE` |
| `tenferro` | `tenferro-rs` revision `f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266`; MIT **OR** Apache-2.0 | [`tenferro-MIT.txt`](licenses/tenferro-MIT.txt) and [`tenferro-APACHE-2.0.txt`](licenses/tenferro-APACHE-2.0.txt), root files |
| `hdf5-metno` wrappers | resolved `hdf5-metno` 0.12.6 and family; MIT **OR** Apache-2.0 | [`hdf5-metno-MIT.txt`](licenses/hdf5-metno-MIT.txt) and [`hdf5-metno-APACHE-2.0.txt`](licenses/hdf5-metno-APACHE-2.0.txt), upstream root at source revision `15683e2dda465c50304b6192acb01a645212548e` |
| bundled HDF5 | `hdf5-metno-src` 0.10.2, revision `574c6283e191aae0f78ff45d145dd16483b64dae`; HDF5 notice/BSD-style terms | [`HDF5.txt`](licenses/HDF5.txt), resolved `ext/hdf5/LICENSE` |
| `priority-queue` | crates.io 2.7.0; LGPL-3.0-or-later **OR** MPL-2.0, selecting MPL-2.0 here | [`priority-queue-MPL-2.0.txt`](licenses/priority-queue-MPL-2.0.txt), resolved `MPL-2.0.txt` |

The pinned `tensor4all-treetn` README states that portions of `src/tdvp/plan.rs` were derived from
ITensorNetworks.jl, copyright 2021 The Simons Foundation, Inc. See the pinned
[`tensor4all-treetn` README](https://github.com/tensor4all/tensor4all-rs/blob/880032daea2a84d7ddfe84a2594a6d160631afc2/crates/tensor4all-treetn/README.md)
and [source file](https://github.com/tensor4all/tensor4all-rs/blob/880032daea2a84d7ddfe84a2594a6d160631afc2/crates/tensor4all-treetn/src/tdvp/plan.rs).
This preserves upstream provenance for a resolved component; it does not claim that this repository
uses or exposes a public TDVP algorithm.

`Cargo.lock` is authoritative for the full resolved Rust graph. This selected list highlights
notices relevant to the checked-in source and bundled native HDF5; it is not a complete inventory
of every transitive crate. The repository does not vendor Rust dependency source trees.

The static HDF5 feature compiles bundled C source. Redistributors of resulting binaries must keep
applicable HDF5 notices and assess the exact artifacts shipped, including linked libraries,
runtime components, optional features, and packaging. These source notices provide useful
attribution but do not constitute legal clearance of any particular binary distribution.

Selecting MPL-2.0 for `priority-queue` requires more than carrying the license text when Covered
Software is distributed in executable form. A redistributor must meet the MPL's source-code
availability and notice conditions, including informing recipients how to obtain the corresponding
Source Code Form and providing modifications covered by the MPL. The exact unmodified component
source is available from the [`priority-queue` 2.7.0 crates.io release](https://crates.io/crates/priority-queue/2.7.0);
artifact-specific modifications, if any, must be handled by the redistributor. See
[MPL 2.0 section 3](https://www.mozilla.org/en-US/MPL/2.0/#distribution-responsibilities).
