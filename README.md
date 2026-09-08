# Thermal iMPS Purification

Finite-temperature calculations for infinite one-dimensional quantum spin chains using
two-site infinite matrix-product-state (iMPS) purification and imaginary-time iTEBD. The
program starts from the purified infinite-temperature state, cools it to a target inverse
temperature, and records energy density, specific heat, free-energy density, and a local
observable. We use units with `k_B = 1`, so `beta = 1/T`.

The implementation uses
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs) for indexed tensors,
contractions, singular-value decompositions, and HDF5 tensor serialization.
[tenferro-rs](https://github.com/tensor4all/tenferro-rs) provides the numerical tensor and
linear-algebra backend used through tensor4all. See [references](docs/references.md) for pinned revisions,
method references, and the exact component scope.

## Requirements

- Rust 1.96.1
- a C compiler and build tool
- CMake 3.26 or newer
- Python 3 for plotting

The lockfile pins Rust dependencies. Vendored static HDF5 means that a system HDF5,
BLAS/LAPACK, Fortran, MPI, or zlib installation is not required. On Ubuntu 24.04, install
`build-essential cmake`; on macOS, install Xcode Command Line Tools and a current CMake.

## Run the TFIM quickstart

From the repository root:

```bash
cargo run --release --locked --bin solve -- configs/quickstart.toml
```

This evolves the transverse-field Ising model (TFIM) to `beta_max = 0.2` and writes
`results/quickstart.json`. Output creation is exclusive: move that file or change
`output.path` before running the command again.

Create a plotting environment and render energy density, specific heat, and `beta * f`:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-plot.txt
python3 scripts/plot_run.py results/quickstart.json figures/quickstart.png
```

The quickstart includes exact TFIM reference columns. Its short run checks the interface and
produces a plot; it is not a convergence study.

## Configure a calculation

The main controls in [`configs/quickstart.toml`](configs/quickstart.toml) are:

```toml
[model]
type = "tfim"
j = 1.0
g = 0.7

[evolution]
dtau = 0.05
beta_max = 0.2
record_every_beta = 0.1
trotter_order = 2

[truncation]
epsilon = 1e-12
max_bond = 32

[run]
canonicalize_every = 1

[output]
path = "results/quickstart.json"
include_exact = true
```

Each step advances `beta` by `2 * dtau`. Second-order Strang evolution is the default.
`epsilon` controls relative discarded weight, `max_bond` caps the bond dimension, and
`record_every_beta` sets the observation interval.

Before using results scientifically, reduce `dtau` and `epsilon`, increase `max_bond`, check
canonicalization and model-specific invariants, and compare with independent references or
limiting cases. The [numerical conventions](docs/numerical-conventions.md),
[validation evidence](docs/validation.md), and [known limitations](docs/limitations.md) define
the qualified scope of the checked-in results.

## Models and outputs

Supported nearest-neighbor model types include:

- `tfim`: transverse-field Ising, with optional exact reference columns
- `xy`: anisotropic XY, with optional exact reference columns
- `bilinear_biquadratic`: spin-1 bilinear-biquadratic
- `aklt` and `aklt_projector`: two documented AKLT normalizations
- `heisenberg`: spin-1 Heisenberg preset
- `matrix`: user-supplied Hermitian model and local observable

Set `include_exact = false` for every model except `tfim` and `xy`.

The AKLT normalization relation is documented in the
[AKLT normalization guide](knowledge/topics/aklt-normalization.md). Do not interchange their configurations
or checkpoints by renaming the model tag.

Result JSON contains run metadata and records with `beta`, energy density `u`, specific heat
`c`, free-energy density `f`, dimensionless free-energy density `beta_f`, the configured local observable in the compatibility field
`magnetization`, the largest bond dimension reached in that step as `max_bond`, and optional
`exact` values. Fresh runs begin with an infinite-temperature (`beta = 0`) record:
`f` is `null` because it diverges, while `beta_f = -ln(local_dim)` is finite.

See [CLI usage](docs/usage.md) for model schemas, matrix input, smoke runs, outputs, checkpoints,
restarts, and figure reproduction. See the [library API guide](docs/library-api.md) for direct
real/complex evolution, specific heat, checkpoint I/O, and interval reduced density matrices.

## Development, provenance, and license

[CONTRIBUTING.md](CONTRIBUTING.md) describes development and verification. The
[provenance](docs/provenance.md), [references](docs/references.md), and
[third-party notices](THIRD_PARTY_NOTICES.md) record extraction history, upstream software,
methods, licenses, and redistribution obligations.

Project-authored code and documentation are licensed under the [MIT License](LICENSE), copyright
2026 Atsushi Iwaki. Copied upstream material remains under its upstream terms. No scholarly
citation of this repository is requested; the MIT notice obligations still apply.
