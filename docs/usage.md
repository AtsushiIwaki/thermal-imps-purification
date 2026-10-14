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

`include_exact = true` is accepted only for TFIM and XY. The precise relation is in the
[AKLT normalization guide](../knowledge/topics/aklt-normalization.md).

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
- `beta_f`: dimensionless free energy per site, equal to `beta * f` at positive beta
- `magnetization`: the configured local observable; the name remains for compatibility
- `max_bond`: largest bond dimension reached
- `exact`: exact reference fields when requested, otherwise `null`

For matrix models, the observable name and matrix remain in `metadata.model`.

Fresh runs include an initial `beta = 0` record before the scheduled positive-beta
observations. At infinite temperature, `u = Tr(site_energy) / d²`, the local observable
is `Tr(observable) / d`, `c = 0`, and `max_bond = 1`, where `d` is the local dimension.
The free energy diverges, so `f` (including `exact.f` when present) is `null`;
`beta_f = -ln(d)` remains finite. A run whose rounded evolution schedule has zero
steps still outputs this initial record. The existing requirement `beta_max > 0` remains.
The JSON reader and plotting script also accept historical positive-beta records without
`beta_f`, reconstructing it as `beta * f`.

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
step, leaving source files unchanged. It does not repeat the observation at the selected
step, including when restarting from the beta-zero snapshot. Every destination must be unused. JSON and HDF5 are not
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

## Fixed mixed-field Ising reference worker

`ising_reference` exposes four argument forms:

```text
ising_reference --preflight REQUEST_JSON
ising_reference REQUEST_JSON OUTPUT_DIRECTORY
ising_reference --verify OUTPUT_DIRECTORY
ising_reference --finalize-interrupted OUTPUT_DIRECTORY REASON
```

A schema-1 request contains only `schema_version: 1`,
`model: {"name":"ising_mixed","g":1.05,"h":0.5}`, `variant` (`fine`, `time-step`, or
`bond-cap`), `source_revision` (40 hexadecimal characters), and `worker_sha256`
(64 hexadecimal characters). Unknown fields and unsupported models/settings fail. The caller
must independently bind the actual source, worker executable and request bytes; these strings
are syntax-checked claims, not an executable attestation. Build with
`cargo build --release --offline --locked --bin ising_reference` using the README's native
static-HDF5 environment. Preflight emits all resolved settings/matrices/grid and creates no output.

Each run starts at beta zero with real two-site purification of
`H = -sum ZZ - 1.05 sum X - 0.5 sum Z` (Pauli operators, `J=k_B=1`). Targets are
`T=j/10`, `beta=10/j`, for decreasing integer `j=20,...,1`, with no interpolation.
The coarse grid uses 507 steps; both finer grids split each coarse step in two (1,014 steps).
Fine uses maximum `dtau=.005`, cap128; time-step uses `.01`, cap128; bond-cap uses `.005`, cap64.
All use cutoff `1e-13`, second-order Strang, and canonicalization after every complete step.
Canonicalizer tolerance/max-iterations/support-floor are `1e-13`/4,000/`1e-12`.
RDM tolerance/max-iterations are `1e-12`/10,000; trace/Hermiticity/positivity tolerance is `1e-10`;
output/intermediate element caps are 1,048,576/16,777,216. No wall cap or automatic retry is added.

The output directory must not exist. Its only members are `conditions.json`, `samples.csv`,
`summary.json` and, when a valid endpoint exists, one `state.h5`. Conditions are exactly the
resolved preflight object. Summary schema1 repeats `request` and records `status` (`complete`
or `partial`), actual `reason`, `completed_steps`, `completed_steps_kind`, `available_samples`,
`elapsed_seconds`, and
`checkpoint_available`. Every target boundary atomically replaces the 20-row scalar table;
staging files are temporary siblings outside the bundle. Only one terminal checkpoint is written,
never per-step tensors or checkpoints. A complete summary is published after consistency checks.
An I/O failure can leave diagnostic evidence and no summary; normal verification rejects that
incomplete publication. Atomic replacement and HDF5 flush do not guarantee power-loss durability.

The CSV has these ordered columns:

```text
temperature_tick,temperature,beta_target,beta_actual,completed_steps,available,reason,energy,mx,mz,zz,free_energy,entropy,accumulated_log_norm,parity_delta_energy,parity_delta_mx,parity_delta_mz,parity_delta_zz,direct_energy_difference,maximum_rdm_fixed_point_residual,maximum_rdm_trace_residual,maximum_rdm_hermiticity_residual,maximum_rdm_iterations,minimum_rdm_eigenvalue,bond_dimension_ab,bond_dimension_ba
```

Missing measurement/diagnostic values are empty fields (load as JSON null); availability is
`true`/`false` and unavailable rows retain a quoted CSV reason. Reasons may contain commas,
quotes or newlines, so use a CSV parser. Progress and accumulated log norm remain finite known
ledger values even when measurement values are missing. Untouched slots have beta/steps/norm zero.
`energy` averages A/B two-site physical-RDM expectations and agrees with
`-zz-1.05*mx-.5*mz`; `direct_energy_difference` is that energy minus `energy_density_auto`.
Parity differences are A minus B. The norm ledger includes both evolution and canonicalization:
`f=-(ln(2)+accumulated_log_norm/2)/beta`, `s=(E-f)/T`, per physical site. This entropy is neither
purification entanglement nor block Markov entropy. Map these fields to ordinary `f_ref`/`s_ref`.

Only after observing worker termination may a supervisor call `--finalize-interrupted`.
This preserves the scalar table and never evolves or replaces state. An absent endpoint remains
absent; an existing endpoint must pass native integrity and applicable terminal remeasurement.
Corrupt/incomplete checkpoints or inconsistent existing summaries leave finalization unsuccessful
and evidence unchanged. A finalized interruption is always `partial`, with the supervisor's reason
and `elapsed_seconds: null` (unknown); without a checkpoint, completed steps are the largest
retained scalar-ledger count, explicitly labeled `completed_steps_kind: "retained_lower_bound"`.
Ordinary returned progress and native endpoints use `completed_steps_kind: "exact"`.
Native panics remain process
failures, not successful partial evolutions. Ordinary evolution errors also omit the possibly
mutated endpoint. RDM errors may leave gaps alongside a valid terminal endpoint.

Verification emits one schema1 JSON object with `status`, `sample_count`, `available_samples`,
`checkpoint_available`, `checkpoint_verified`, `terminal_consistency_passed` (null when no available
sample matches the endpoint), `completed_steps`, `completed_steps_kind`, and `request`.
Errors exit nonzero with file/field context.
The caller retains independent file hashes and rejects later mutation. A verified table/endpoint
does not certify physical convergence, accuracy at earlier beta, or a refinement comparison.
Scheduled native checkpoints use schema2 with exact segments; fixed-step checkpoints retain
schema1. The fixed-step `solve` restart interface explicitly rejects scheduled checkpoints.
