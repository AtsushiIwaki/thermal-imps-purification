# CLI usage

Run commands from the repository root and use `--release` for real sweeps. `solve` accepts
TOML and JSON with the same validated schema; paths are resolved relative to the working
directory.

## Build, run, and smoke test

```bash
cargo build --release --locked
cargo run --release --locked --bin solve -- configs/quickstart.toml
```

The quickstart writes `results/quickstart.json`. Output creation is exclusive, so move existing
results or select a fresh `output.path` before each repeat.

The smoke driver runs fresh real and genuinely complex phase-rotated TFIM inputs. Its output
directory must not exist:

```bash
cargo build --release --locked --bin solve
python3 scripts/smoke.py target/release/solve results/smoke-001
```

Use another directory when repeating it. These short cases test finite output, not convergence.

## Configuration

```toml
[model]
type = "tfim"
j = 1.0
g = 0.7

[evolution]
dtau = 0.01
trotter_order = 2
beta_max = 8.0
record_every_beta = 0.2

[truncation]
epsilon = 1e-12
max_bond = 64

[run]
canonicalize_every = 1

[output]
path = "results/tfim.json"
include_exact = true
```

Both Trotter orders advance beta by `2 * dtau` per step. Omitting `trotter_order` selects
second-order Strang evolution; set it to `1` for historical first-order runs.

| Model | Parameters | Meaning | Exact columns |
| --- | --- | --- | --- |
| `tfim` | `j`, `g` | transverse-field Ising | yes |
| `xy` | `gamma`, `h` | anisotropic XY in a field | yes |
| `bilinear_biquadratic` | `j1`, `j2` | spin-1 bilinear-biquadratic | no |
| `aklt` | none | historical AKLT normalization | no |
| `aklt_projector` | none | coefficient-one spin-2 projector | no |
| `heisenberg` | none | spin-1 Heisenberg preset | no |
| `matrix` | matrices, observable | custom Hermitian model | no |

`include_exact = true` is accepted only for TFIM and XY. The precise AKLT normalization
relation is in [numerical conventions](numerical-conventions.md).

## Matrix models

A matrix model has `version = 1`, positive `local_dim = d`, and
`basis_order = "first_site_fastest"`. It supplies Hermitian `two_site_h` and `site_energy`
matrices of shape `d^2 x d^2`, plus a Hermitian `observable.matrix` of shape `d x d` and a
nonblank `observable.name`.

Matrices use row arrays. Each has required `real` and optional same-shaped `imag` data;
omitting `imag` means exact zeros. A physical tuple `(s1,s2)` maps to `s1 + d*s2`.
Malformed, nonfinite, unsupported, or non-Hermitian inputs are rejected at scaled Frobenius
tolerance `1e-12`; inputs are never silently symmetrized.

Real storage is chosen only when every imaginary entry of both Hamiltonian matrices is exactly
zero. A complex observable alone does not select complex storage. Runnable encodings are in
[`configs/phase_tfim.toml`](../configs/phase_tfim.toml) and
[`configs/phase_tfim.json`](../configs/phase_tfim.json).

## Result JSON

Outputs contain `metadata` and `records`. Metadata preserves the model, settings, local
dimension, and Git revision when available. Each record has:

- `beta`: inverse temperature, with `k_B = 1`
- `u`, `c`, and `f`: energy, specific heat, and free energy per site
- `magnetization`: the configured local observable; the name remains for compatibility
- `max_bond`: largest bond dimension reached
- `exact`: exact reference fields when requested, otherwise `null`

For matrix models, the observable name and matrix remain in `metadata.model`.

## Automatic checkpoints and restart

Enable initial, periodic, and final HDF5 snapshots:

```toml
[checkpoint]
path = "results/part1.h5"
every_steps = 100
```

To continue, retain model and numerical settings, increase `beta_max`, select fresh JSON and
HDF5 destinations, and add:

```toml
[restart]
path = "results/part1.h5"
# snapshot = 3
```

Omitting `snapshot` selects the latest complete snapshot. `[restart]` requires
`[checkpoint]`. Compatibility checks cover Hamiltonian values and storage, time step, Trotter
order, truncation, canonicalization cadence, and any saved observation interval. An `aklt`
checkpoint cannot become `aklt_projector` by changing its label.

A resumed invocation writes a separate trajectory and an observation segment after the selected
step, leaving source files unchanged. Every destination must be unused. JSON and HDF5 are not
one transaction; after interruption, inspect complete HDF5 snapshots before joining segments.
See [numerical conventions](numerical-conventions.md) and [limitations](limitations.md).

```bash
cargo run --release --bin solve -- configs/tfim_checkpoint.toml
cargo run --release --bin solve -- configs/tfim_restart.toml
cargo run --release --bin solve -- configs/phase_tfim_checkpoint.toml
cargo run --release --bin solve -- configs/phase_tfim_restart.json
```

Edit their output and checkpoint paths before repeating them.

## Reproduce figures

```bash
cargo run --release --bin solve -- configs/tfim.toml
cargo run --release --bin solve -- configs/xy.toml
cargo run --release --bin solve -- configs/aklt.toml
cargo run --release --bin solve -- configs/aklt_projector.toml

python3 scripts/plot_run.py results/tfim.json figures/tfim_observables.png
python3 scripts/plot_run.py results/xy.json figures/xy_observables.png
python3 scripts/plot_run.py results/aklt.json figures/aklt_observables.png
python3 scripts/plot_run.py results/aklt_projector.json figures/aklt_projector_observables.png
```

TFIM and XY overlay exact curves. AKLT guides are limiting values, not complete exact
finite-temperature curves. Consult [validation](validation.md) and [limitations](limitations.md)
before extending the checked-in evidence.
