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

## Fixed AKLT projector reference worker

`aklt_reference` has the same four argument forms as `ising_reference`, with its own
strict schema-1 request: `contract: "aklt-projector-reference-v1"`,
`model: {"name":"aklt_projector"}`, `variant: "fine"` or `"time-step"`,
`max_seconds: 3600.0`, and the same source-revision/worker-hash fields. There is no
bond-cap variant or restart. Build `--bin aklt_reference` with the native environment
above. Both variants start independently at beta zero. Fine uses maximum dtau0.005;
time-step uses0.01. Both use cap128 and cutoff1e-13; all other grid, canonicalization,
RDM and element limits match the fixed Ising worker. Source/worker identity must still
be independently bound by the caller.

The Hamiltonian is the coefficient-one spin-2 projector, with d=3 and ground energy0.
Energy averages the two physical-RDM parities. `mx` and `mz` use spin-one operators,
not Pauli matrices. Norms from both evolution and canonicalization contribute to
`f=-(ln(3)+accumulated_log_norm/2)/beta` and `s=beta*(E-f)`. No exact finite-temperature
reference or bond-convergence certificate is claimed.

The monotonic3600-second trajectory allowance starts before setup/publication. It is
checked between complete native operations and around measurements/callbacks; a long
operation can overrun. A valid measurement completed during overrun is retained, then
no next evolution begins. Native failures keep their actual reason. A failed mutating
operation invalidates the endpoint; read-only measurement failure may retain it.
Terminal publication and native verification finish after the trajectory stops.

The compact bundle contains the same three scalar files and optional single native
state as the Ising contract, with exactly20 target rows. Its ordered CSV columns are:

```text
temperature_tick,temperature,beta_target,beta_actual,completed_steps,available,reason,energy,mx,mz,free_energy,entropy,accumulated_log_norm,parity_delta_energy,parity_delta_mx,parity_delta_mz,direct_energy_difference,maximum_rdm_fixed_point_residual,maximum_rdm_trace_residual,maximum_rdm_hermiticity_residual,maximum_rdm_iterations,minimum_rdm_eigenvalue,bond_dimension_ab,bond_dimension_ba,cap_reached_now,maximum_dimension_seen,cap_ever_reached,cap_observed_strang_steps,cap_coverage
```

`cap_reached_now` describes the sampled canonical dimensions. Cumulative cap evidence
also observes each complete Strang StepInfo (including its substep maximum), so a
previous cap hit survives later shrinking. `cap_coverage` is `complete_through_stop`
or `unobserved_tail`. If an in-place failure or dead process leaves an unobserved tail,
`cap_ever_reached` remains true if proven; otherwise it is empty/unknown, never false.
The summary repeats this evidence as `cap`, alongside actual `beta_actual`,
`accumulated_log_norm`, `budget_overrun_seconds` and the Ising-style progress/status
fields. In v1, elapsed/overrun are null only for an unobserved interruption.
In v2, overrun is always null because no deadline exists.

Native verification checks exact settings, schedule, model matrices, endpoint bits,
thermodynamic identities, cap monotonicity and applicable terminal remeasurement.
Off-target beta is never rounded to a target. Interrupted finalization is allowed
only after worker death; corrupt checkpoints remain unsuccessful integrity evidence.
The publication report adds disjoint `timing.trajectory_seconds`,
`timing.terminal_publication_seconds` and `timing.native_verification_seconds`.
A caller's inclusive process wall time is an additional measurement, not an amount
to add to these intervals. Earlier sampled states are not retained or replayed.

### AKLT reference without a deadline

The fixed `aklt-projector-reference-v2` contract uses `schema_version: 2` and
a required `max_seconds: null`. Omission, a finite number, and a string are
rejected. The original v1 contract still requires schema 1 and 3600 seconds.
Both use the same independent beta=0 starts, cap 128, cutoff 1e-13,
max_dtau 0.005/0.01 and 1014/507 Strang steps for fine/time-step respectively.
No Ising policy or numerical fallback is substituted.

All v2 preflight, conditions, summary and verification envelopes preserve v2
identity. Elapsed time remains measured (or null if unknown after interruption);
`budget_overrun_seconds` is always null. Cap observations and unknown-tail
coverage remain independent of deadline selection. Output creation remains
exclusive and interruption finalization does not resume evolution.


The2026-10-27 single v2 operation exercised both variants. Fine completed20 targets
and its endpoint verification; time-step retained18 targets and was finalized as
partial after an SVD backend failure, without a checkpoint or retry. Partial native
verification validates retained evidence, not an unavailable physical endpoint.
See [the measured outcome and limits](limitations.md#new-aklt-reference-adapter-2026-10-24-implementation).

### AKLT cap256 target snapshots and variance heat

The explicit `aklt-projector-reference-v3` contract (schema3) adds one `fine`
trajectory with `max_bond:256`, required `max_seconds:null`,
`recording_policy:"target-snapshots-v1"` and
`heat_capacity_policy:"variance-default-v1"`. Model/source/worker fields retain
v2 meanings. Other caps, variants, missing deadlines, policies and unknown
request fields are rejected. V1/v2 interfaces and retained output meanings remain.

```json
{"schema_version":3,"contract":"aklt-projector-reference-v3","model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,"max_seconds":null,"recording_policy":"target-snapshots-v1","heat_capacity_policy":"variance-default-v1","source_revision":"<40 lowercase hex digits>","worker_sha256":"<64 lowercase hex digits>"}
```

Use the same `aklt_reference --preflight request.json`,
`aklt_reference request.json fresh-directory`, `--verify directory` and
`--finalize-interrupted directory reason` forms. `--runtime-info` is read-only
and reports provider, visible CPUs, actual native thread hint and five thread
environment variables. It neither evolves a state nor changes a default.
The cap256 controller must preserve the independent purification environment;
MST's four-thread baseline does not apply. Verification may use thread1 without
rewriting the recorded launch settings.

The existing 1014-step fine grid and all cutoff/canonicalizer/RDM settings are
reused. One `states.h5` contains up to20 scheduled snapshots at T=2.0,...,0.1.
No beta0/per-step/off-target/duplicate terminal state is retained. Each target is
appended, flushed and read back before physical and variance measurements.
Variance uses `SpecificHeatOptions::default()` (distance200, rtol1e-6,
atol1e-12,3 small shells, reality1e-10). Failure preserves the saved state and
completed physical values, records its cause, stops evolution and leaves later
points unavailable.

CSV adds snapshot availability/index/reason, heat availability/reason,
`heat_capacity_variance`, and compact scalar `heat_diagnostics` to the v2 fields.
Conditions and summary retain actual `runtime_at_launch`. Summary binds file
hashes, snapshot mapping/count and separate physical/heat counts. Native
verification loads all usable snapshots and checks grid, progress, source,
model, cap256, physical scalars and beta²-variance normalization. It does not
repeat variance sums. Returned read-back evidence is verification of retained
states, not a bond/time-step convergence certificate.

After a reaped worker's interruption, finalization recovers completed snapshots
even if scalar CSV publication lagged, marking absent measurements unavailable.
A mutation-failed or flush-uncertain snapshot remains uncertified diagnostic
content; usable snapshots must be an ordered prefix. Corrupt complete states,
extra indices, swapped targets and existing inconsistent summaries are rejected.
Finalization cannot resume evolution and refuses to overwrite an existing summary.
The controlling Infinite_MST process consumes one launch and handles report-only
recovery; it must not delete guards or issue an automatic second trajectory.

The selected 2026-10-29 operation is partial:14 target snapshots and physical/
variance rows through T=0.7 were verified, then SVD failed before the next target.
The retained154 steps are a lower bound, not the exact failure step; missing
T=0.6,...,0.1 values remain unavailable. No retry was performed. The controlling
Infinite_MST record owns the comparison, runtime and cleanup evidence.

### AKLT cap256 continuation after the SVD repair

The explicit `aklt-projector-reference-resume-v4` request (schema4) continues the
original v3 fourteen-target bundle from snapshot13 at T=0.7. It keeps every v3
fine control and requires `restart` with a relative `directory`, the four
parent file pins (`bytes`, `sha256` for conditions.json, samples.csv, states.h5,
summary.json), `snapshot_index:13`, and exact saved `progress`
(`completed_steps:154`, `beta`, `accumulated_log_norm`). Its `cap` comes from
the saved boundary row, with `coverage:"complete_through_stop"` and
`observed_strang_steps:154`; the failed parent's unobserved tail is not inherited.
Unknown fields and control changes are rejected. Preflight resolves the request
without evolving or opening the parent; launch verifies the parent before
reserving output, including pins, native physical consistency, raw progress,
normalization and the complete original segmented schedule.

The loaded state goes directly into the remaining six original segments:
860 new full Strang steps, absolute endpoint1014. No beta0 setup or initial
canonicalization is repeated. The accumulated log normalization and observed cap
history continue from the checkpoint. One new `states.h5` contains only the six
new snapshots, T=0.6,0.5,...,0.1, local indices0..5; T=0.7 is not copied.
Conditions preserve the full1014-step schedule, remaining_steps860, initial
progress and source pins. Summary and verification sample/snapshot/physical/heat
counts refer to these six new targets; completed_steps remains absolute.
Interrupted finalization may retain zero new snapshots and progress154.
Verification also requires the pinned original v3 parent. A compatible rebuilt
reader can inspect either producer without requiring the original executable.

The controlling report combines the original14 scalar/variance rows with the
six continuation rows, retaining both producers and local snapshot mappings.
Old variance sums are not evaluated again. That report uses the explicit
`aklt-purification-bond-comparison-report-v2` frozen format; its original v1
report remains readable. Continuation plots identify the T=0.7 boundary.

Faer0.24.4 is bound to the retained terminal2x2 SVD repair; see
[vendoring provenance](../vendor/faer/VENDORING.md). The tensor4all/tenferro pins,
cutoff, schedule, canonicalizer and variance defaults remain unchanged. Native
production thread defaults are preserved and recorded. One selected invocation
has no elapsed-time cap, automatic retry or fallback.

Only on a returned real bond-SVD error during this continuation, a scoped hook
attempts exclusive creation of `../svd-failure.h5`, at most once. It records the
exact column-major f64 input with raw-bit hash, shape and tensor split, absolute
step, beta, SVD call1..3 within that full step, error and producer identity.
The bound is2304x2304 entries (42,467,328 matrix bytes). It records no per-step
history and does not certify a physical state. Capture failure preserves the
original SVD panic; unrelated backend panics are not promised matrix capture.
`aklt_reference --verify-svd-failure FILE` checks schema, completeness, dimensions,
layout, hash and producer fields and returns small diagnostic metadata, including
nonfinite count. It does not replay SVD or emit matrix JSON. Existing files are
never overwritten. A passing finite control does not identify the original
uncaptured thermal failure's cause.
