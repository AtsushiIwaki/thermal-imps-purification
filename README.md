# Thermal iMPS Purification

Finite-temperature properties of infinite 1D quantum spin chains via **iMPS purification +
iTEBD**. The thermal state ρ(β) ∝ e^{−βH/2}|I⟩⟨I|e^{−βH/2} is represented as an infinite
matrix-product state on a 2-site unit cell; imaginary-time evolution of the purified
maximally-mixed state cools it from T = ∞ down to the target temperature. Observables
(energy density, specific heat, free energy, magnetization) are read off along the way.

## Build

Requirements are Rust 1.96.1, a C compiler/build tool, and CMake 3.26 or newer. The lockfile
pins all Rust dependencies; the enabled `hdf5-metno` `static` feature builds vendored HDF5, so a
system HDF5, BLAS/LAPACK, Fortran, MPI, and zlib installation is not required. On Ubuntu 24.04,
install `build-essential cmake`; on macOS, install Xcode Command Line Tools and a current CMake.

```bash
cargo build --release --locked
```

Large-β or spin-1 (d = 3) runs are slow in debug — use `--release` for real sweeps.

## Quick start

```bash
cargo run --release --bin solve -- configs/quickstart.toml
```

This short TFIM run writes `results/quickstart.json`. Output creation is exclusive: move the
result or change `output.path` before repeating a run. To plot it in an isolated Python
environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-plot.txt
python3 scripts/plot_run.py results/quickstart.json figures/quickstart.png
```

The two-case smoke command runs fresh real TFIM and genuinely complex phase-rotated TFIM inputs.
Its second argument must name a directory that does not exist:

```bash
cargo build --release --locked --bin solve
python3 scripts/smoke.py target/release/solve results/smoke-001
```

Use a new directory such as `results/smoke-002` for every repeat. The smoke cases are short
interface checks, not convergence measurements.

Numerical conventions, qualified limits, and extraction/dependency provenance are documented in
[`docs/numerical-conventions.md`](docs/numerical-conventions.md),
[`docs/limitations.md`](docs/limitations.md), and [`docs/provenance.md`](docs/provenance.md).

## Configuration (TOML or JSON)

`solve` accepts `.toml` and `.json` configuration paths; suffix matching is
case-insensitive. The two formats share the same validated schema. For example:

```bash
cargo run --release --bin solve -- configs/phase_tfim.toml
cargo run --release --bin solve -- configs/phase_tfim.json
```

The first command writes `results/phase_tfim.json`; the second uses the distinct
`results/phase_tfim_json.json` destination configured in the JSON file. Output and checkpoint
paths are resolved relative to the working directory. Checkpoint-mode destinations use exclusive
creation; choose unused output paths before repeating a checked-in checkpoint/restart example.

```toml
[model]
type = "aklt"        # tfim | xy | bilinear_biquadratic | aklt | aklt_projector | heisenberg

[evolution]
dtau = 0.01              # imaginary-time step (β advances by 2·dtau per step)
trotter_order = 2        # optional: 1 or 2; omission defaults to second-order Strang
beta_max = 8.0           # final inverse temperature
record_every_beta = 0.2  # record observables at this β spacing

[truncation]
epsilon = 1e-12          # relative discarded weight per bond
max_bond = 64            # optional bond-dimension cap (omit for none)

[run]
canonicalize_every = 1   # optional; canonicalize every N steps (default 1)

[output]
path = "results/aklt.json"
include_exact = false    # tfim/xy only: also emit closed-form reference columns
```

### Models and their parameters

| `type`                 | parameters   | Hamiltonian                                        | exact reference |
|------------------------|--------------|----------------------------------------------------|-----------------|
| `tfim`                 | `j`, `g`     | −j·ZZ − g·X (transverse-field Ising)               | yes (free fermion) |
| `xy`                   | `gamma`, `h` | anisotropic XY in field h                           | yes (free fermion) |
| `bilinear_biquadratic` | `j1`, `j2`   | j1·(S·S) + j2·(S·S)² (spin-1)                       | no              |
| `aklt`                 | none         | preset (j1, j2) = (1, 1/3)                          | no              |
| `aklt_projector`       | none         | coefficient-one P₂ projector (spin-1)               | no              |
| `heisenberg`           | none         | preset (j1, j2) = (1, 0) (spin-1)                   | no              |
| `matrix`               | inline matrices and observable | user-supplied Hermitian nearest-neighbor model | no |

`include_exact = true` is rejected for models without a closed form (everything except `tfim`/`xy`).
For `x = S_i·S_(i+1)`, the projector preset uses
`P₂ = (x² + 3x + 2I)/6` in both the evolution Hamiltonian and reported site energy. The existing
`aklt` preset remains `h_old = x + x²/3 = 2P₂ - 2I/3`; its normalization and old result tags are
unchanged. At the same physical Gibbs state,
`beta_old = beta_projector/2`, `dtau_old = dtau_projector/2`,
`u_projector = u_old/2 + 1/3`, `f_projector = f_old/2 + 1/3`, and specific heat,
single-site `Sz`, and normalized RDMs agree. Both AKLT presets report the fixed spin-1
`Sz = diag(1,0,-1)` observable. Neither has a full finite-temperature exact curve, so their plots
show only analytic limiting guides and their records keep `exact: null`.

Both Trotter orders advance β by `2 * dtau` per sweep step; configurations that omit
`trotter_order` use the second-order Strang sequence `AB/2 -> BA -> AB/2`. Set
`trotter_order = 1` explicitly to reproduce historical first-order runs. Historical result JSON
whose metadata predates the order field is still interpreted as first order.

### Matrix models

A matrix model must provide `version = 1`, a positive `local_dim = d`, and
`basis_order = "first_site_fastest"`, followed by three Hermitian matrices:

- `two_site_h`: the repeated two-site evolution Hamiltonian, with shape `d² × d²`;
- `site_energy`: the two-site operator used to report per-site energy, also `d² × d²`;
- `observable.matrix`: the `d × d` local observable whose expectation is recorded.

`observable.name` must be nonblank. It describes the observable in result metadata; the optional
model `label` is descriptive only. Every matrix has a required `real` array and an optional
`imag` array. Each array is written as matrix rows containing columns. This row-array layout is
separate from the physical tuple convention: a two-site basis tuple `(s1, s2)` has index
`s1 + d*s2`, so the first site is fastest. The loader copies entries at the stated row and column
without transposing or conjugating them.

Omitting `imag` means an exactly zero imaginary array of the same shape. If `imag` is present, it
must be a full array with the same dimensions; `null`, ragged arrays, nonfinite values, wrong
sizes, unknown keys, unsupported versions/basis orders, and mixed preset parameters are rejected.
All three matrices are checked for Hermiticity with
`||M-M†||F <= 1e-12 * max(||M||F, 1)` and are never silently symmetrized. Matrix models require
`include_exact = false`.

Fresh storage classification examines both Hamiltonian matrices. The real backend is selected
only when every imaginary entry of `two_site_h` and `site_energy` is exactly zero, including
signed zero; any nonzero Hamiltonian imaginary entry selects complex storage. A complex
observable by itself does not select a complex state. See
[`configs/phase_tfim.toml`](configs/phase_tfim.toml) and
[`configs/phase_tfim.json`](configs/phase_tfim.json) for complete, runnable encodings of the same
phase-rotated TFIM model.

## Output (JSON)

```json
{
  "metadata": {
    "model": { "type": "aklt" },
    "evolution": { "dtau": 0.01, "trotter_order": 2, "beta_max": 8.0, "record_every_beta": 0.2 },
    "truncation": { "epsilon": 1e-12, "max_bond": 64 },
    "canonicalize_every": 1,
    "local_dim": 3,
    "git_revision": "edbab9d"
  },
  "records": [
    { "beta": 0.2, "u": -0.13, "c": 0.21, "f": -1.05,
      "magnetization": 0.0, "max_bond": 12, "exact": null }
  ]
}
```

Per-record columns:

- `beta` — inverse temperature.
- `u` — energy density (per site).
- `c` — specific heat (per site), connected-variance estimator.
- `f` — free-energy density (per site).
- `magnetization` — expectation of the configured single-site operator: TFIM ⟨σx⟩, XY ⟨σz⟩,
  spin-1 ⟨Sz⟩, and `model.observable.matrix` for a matrix model. For matrix models this field name
  is retained for result compatibility and need not represent physical magnetization; the
  observable name and full matrix are preserved in `metadata.model`.
- `max_bond` — bond dimension reached at that step.
- `exact` — `{u, c, f, magnetization}` closed-form references (only when `include_exact`; else `null`).

`metadata.local_dim` is the derived local dimension `d`; `git_revision` is the code revision
that produced the file (`null` if unavailable).

## Two-site iTEBD library APIs

The original direct `f64` API remains available without complex allocation or enum dispatch:

```rust
use thermal_imps_purification::itebd::imaginary_time_step_second_order;
use thermal_imps_purification::model::{LocalHamiltonian, Tfim};
use thermal_imps_purification::purified_mps::infinite_temperature;
use thermal_imps_purification::tensor::Truncation;

fn evolve_real_step() {
    let hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let mut state = infinite_temperature(hamiltonian.dim());
    let step = imaginary_time_step_second_order(
        &mut state,
        &hamiltonian,
        0.01,
        &Truncation {
            epsilon: 1e-12,
            max_bond: Some(64),
        },
    );
    assert!(step.log_norm.is_finite());
}
```

Use the automatic facade when the Hamiltonian matrices may be complex. The matrices below are a
small genuinely complex Hermitian example; the second matrix supplies the per-site-energy bond
operator in the same first-physical-index-fastest basis convention.

```rust
use thermal_imps_purification::itebd_auto::{
    imaginary_time_step_second_order_auto, ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::tensor::Truncation;
use nalgebra::DMatrix;
use num_complex::Complex64;

fn evolve_complex_step() -> Result<(), Box<dyn std::error::Error>> {
    let mut two_site_h = DMatrix::<Complex64>::identity(4, 4);
    two_site_h[(0, 1)] = Complex64::new(0.0, 0.2);
    two_site_h[(1, 0)] = Complex64::new(0.0, -0.2);
    let site_energy = two_site_h.clone();

    let hamiltonian = ItebdHamiltonian::try_from_complex(two_site_h, site_energy)?;
    assert!(matches!(hamiltonian, ItebdHamiltonian::Complex(_)));
    let mut state = ItebdState::infinite_temperature(&hamiltonian)?;
    let step = imaginary_time_step_second_order_auto(
        &mut state,
        &hamiltonian,
        0.01,
        &Truncation {
            epsilon: 1e-12,
            max_bond: Some(64),
        },
    )?;
    assert!(step.log_norm.is_finite());
    Ok(())
}
```

`ItebdHamiltonian::try_from_complex` validates dimensions, finite entries, and Hermiticity before
choosing a backend. It selects the real backend only when every imaginary matrix component is
exactly zero (`im == 0.0`, including signed zero); any nonzero imaginary component selects the
complex backend. `ItebdState::infinite_temperature` fixes that choice for the lifetime of the
state, and mixing real and complex variants returns a typed backend-mismatch error rather than
switching mid-run.

Hermiticity is checked without silently symmetrizing the input, using the default scaled
Frobenius tolerance `1e-12`. Complex local observables must also be finite, dimensionally valid,
and Hermitian, and their expectation must be real within the requested scaled tolerance. These
validation failures are reported through `ItebdError`.

`ModelSpec` model construction is now fallible for source callers because matrix input may be
complex or invalid. Migrate direct real-only callers to propagate the result:

```rust
let hamiltonian = model.hamiltonian()?;
let observable = model.magnetization_op()?;
```

Those helpers reject nonzero imaginary entries rather than projecting them away. Code that should
follow automatic real/complex dispatch uses `let resolved = model.resolve()?;` and then consumes
`resolved.hamiltonian`, `resolved.observable`, and `resolved.observable_name`.

For direct coefficient-one AKLT construction, use the public unit struct and its local builder:

```rust
use thermal_imps_purification::config::ModelSpec;
use thermal_imps_purification::model::AkltProjector;

let local = AkltProjector.local();
assert_eq!(local.two_site_h, local.site_energy);
let configured = ModelSpec::AkltProjector;
```

`ModelSpec::AkltProjector` is a new public enum variant. External exhaustive matches over
`ModelSpec` must add an `AkltProjector` arm. `BilinearBiquadratic::aklt()` and `ModelSpec::Aklt`
continue to construct the historical normalization.

The Rust library API supports first- and second-order evolution, canonicalization, energy
density, Hermitian local expectations, and fallible bidirectional specific heat. The direct real
scalar API returns a `Result`, so callers must handle invalid beta, tensor, numerical-reality, or
tail-convergence failures:

```rust
use thermal_imps_purification::variance::specific_heat;

fn real_specific_heat(
    state: thermal_imps_purification::purified_mps::PurifiedMps,
    hamiltonian: thermal_imps_purification::model::LocalHamiltonian,
    beta: f64,
) -> Result<f64, thermal_imps_purification::itebd_error::ItebdError> {
    let heat = specific_heat(&state, &hamiltonian, beta)?;
    Ok(heat)
}
```

The direct complex API has the same fallible scalar boundary. Use the separate report-producing
function to configure the tail policy and inspect how the result was accumulated:

```rust
use thermal_imps_purification::itebd_complex::{
    specific_heat_complex, specific_heat_complex_with_options, ComplexLocalHamiltonian,
    ComplexPurifiedMps,
};
use thermal_imps_purification::specific_heat::SpecificHeatOptions;

fn complex_specific_heat(
    state: ComplexPurifiedMps,
    hamiltonian: ComplexLocalHamiltonian,
    beta: f64,
) -> Result<f64, thermal_imps_purification::itebd_error::ItebdError> {
    let scalar = specific_heat_complex(&state, &hamiltonian, beta)?;
    let options = SpecificHeatOptions {
        max_distance: 300,
        ..SpecificHeatOptions::default()
    };
    let report = specific_heat_complex_with_options(&state, &hamiltonian, beta, &options)?;
    assert!(scalar.is_finite());
    Ok(report.specific_heat_per_site)
}
```

`specific_heat_with_options` and `specific_heat_with_options_auto` provide the corresponding
real and automatic-facade report calls. The estimator evaluates positive and negative correlation
directions independently; it does not use a reflection-symmetry shortcut. By default,
`SpecificHeatOptions` caps the correlation distance at 200 and accepts a tail only after three
consecutive small shells. If that does not happen, the call returns the typed
`ItebdError::SpecificHeatTailNonConvergence` error rather than a partial scalar. A
`SpecificHeatReport` contains the raw and validated variance, specific heat, onsite,
positive-direction, negative-direction, and parity decompositions, final distance, tail stop
reason, final shell magnitudes, and maximum imaginary residual. Both decompositions
reconstruct `raw_energy_variance_per_site`. The validated `energy_variance_per_site` differs only
when a small negative raw value is projected to zero within the documented tolerance.

Reviewed TOML/JSON matrix input and restart support are included in this package. Their public
contracts and evidence bounds are summarized in
[`docs/numerical-conventions.md`](docs/numerical-conventions.md) and
[`docs/limitations.md`](docs/limitations.md).

The complex backend was validated with the exactly equivalent phase rotation
`U = diag(1, i)` of the TFIM at `J=1`, `g=0.7`, and `beta=1`. On the selected
`dtau=0.1 -> 0.05` window, the measured second-order refinement values are `1.982975` for energy
and `2.072136` for free energy; the largest cutoff/cap endpoint change is `1.9198%`, and the
largest selected bond is 10. The finest energy window is not used as a convergence claim because
its observed order falls to `0.829` as other numerical floors compete. See
[`docs/limitations.md`](docs/limitations.md) before extending this finite-window evidence.

## Two-site iTEBD checkpoints and physical interval RDMs

Run the complete save/load/continue example with a new destination:

```bash
cargo run --release --example itebd_checkpoint_rdm -- /tmp/itebd-checkpoint-rdm-example.h5
```

`examples/itebd_checkpoint_rdm.rs` records beta=0, evolves five steps at `dtau=0.05`
with canonicalization cadence 3, saves snapshot 1, closes the writer, loads that snapshot,
and evolves five more steps. It prints beta, energy, free energy, and A/B length-2 RDM
reports. The caller must supply a path; creation is exclusive and an existing path returns
an error without overwriting it. A run may also begin with a finite-beta first snapshot.
Persist the actual completed step count and accumulated log norm, including actual
canonicalization returns, so an off-cadence restart preserves the next canonicalization step.
Free energy uses `free_energy_from_log_norm(accumulated_log_norm / 2.0, beta, d)`.

`itebd_checkpoint::{ItebdTrajectoryWriter, list_itebd_checkpoints, load_itebd_checkpoint}`
support real and complex two-site states through borrowed `ItebdStateRef` views. The distinct
`TwoSiteItebdPurificationRun` schema (version 1, unit cell size 2) stores Gamma tensors,
Schmidt values, index identities/dimensions, authoritative Hamiltonian matrices, numerical
settings, progress, and diagnostics. Saving/loading does not canonicalize or normalize a
state, and a stored complex backend remains complex even if every imaginary component is
zero. Metadata's Hermiticity tolerance preserves source provenance; load options independently
control Hamiltonian acceptance. Before starting a new trajectory from a loaded complex
Hamiltonian, reconcile the new metadata tolerance with that Hamiltonian's tolerance.

The completion marker excludes incomplete snapshots from listing; loading one explicitly
returns an error. A partial append I/O failure poisons that writer. Markers and flushes are
not filesystem transactions or power-loss guarantees: a final flush failure can leave a
complete marker whose durability is uncertain.

Use `itebd_rdm::{reduced_density_matrix, reduced_density_matrix_complex,
reduced_density_matrix_auto}` for real, complex, or automatic dispatch. Specify `RdmParity::A`
or `RdmParity::B` for the first site, and a positive interval length. The returned normalized
`DMatrix` is column-major, with the first physical site fastest in both row and column
indices: `p0 + d*p1 + ...`. Ancillas are traced out. `RdmReport` includes each boundary's
fixed-point residual, iteration count and eigenvalue, raw trace, normalized trace residual,
Hermiticity residual, minimum eigenvalue, and largest retained tensor size. No negative
spectrum clipping or Hermitian projection repairs the returned matrix.

`RdmOptions` defaults to fixed-point tolerance `1e-12`, at most 10,000 iterations,
trace/Hermiticity/positivity tolerances `1e-10`, 1,048,576 output elements and 16,777,216
intermediate elements. Tolerances must be finite and nonnegative; budgets must be positive.
Nonconvergence, invalid boundaries/matrices and resource overruns return typed errors.
The element caps check planned tensor sizes before allocation; the intermediate cap also
bounds the `2*length+4` occurrence-index metadata slots. The reported tensor peak excludes
metadata and does not measure aggregate memory or hidden backend workspace. With deliberately
corrupted underreported public index dimensions, the pinned dependency may materialize storage
while validating its authoritative shape, so this is not a global allocation bound.

Boundaries are independently solved from identity seeds. Degenerate dominant sectors can make
that selected boundary state sector-dependent; residual convergence does not prove uniqueness.
Check model-specific step-size/cutoff/bond convergence separately from RDM matrix invariants.
The resource and sector qualifications are summarized in
[`docs/limitations.md`](docs/limitations.md).

For genuinely complex input, rotate both model matrices by the same `U = diag(1,i)`:

```rust
use thermal_imps_purification::itebd_auto::{ItebdHamiltonian, ItebdState};
use thermal_imps_purification::itebd_rdm::{reduced_density_matrix_auto, RdmOptions, RdmParity};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

fn phase_rotated_rdm() -> Result<(), Box<dyn std::error::Error>> {
    let h = thermal_imps_purification::model::Tfim { j: 1.0, g: 0.7 }.local();
    let u = DMatrix::from_diagonal(&DVector::from_vec(vec![
        Complex64::from(1.0), Complex64::new(0.0, 1.0),
    ]));
    let u2 = u.kronecker(&u);
    let rotate = |m: &DMatrix<f64>| &u2 * m.map(Complex64::from) * u2.adjoint();
    let h = ItebdHamiltonian::try_from_complex(rotate(&h.two_site_h), rotate(&h.site_energy))?;
    let state = ItebdState::infinite_temperature(&h)?;
    let rho = reduced_density_matrix_auto(&state, RdmParity::B, 2, &RdmOptions::default())?;
    println!("{rho:?}");
    Ok(())
}
```

Automatic save/restart for built-in real models is described next; standalone crate extraction
remains separate work.

## Automatic solve checkpoints and restart

For a fresh built-in run and then a continuation into separate files:

```bash
cargo run --release --bin solve -- configs/tfim_checkpoint.toml
cargo run --release --bin solve -- configs/tfim_restart.toml
```

The repository also includes a genuinely complex TOML fresh/checkpoint run and JSON restart:

```bash
cargo run --release --bin solve -- configs/phase_tfim.toml
cargo run --release --bin solve -- configs/phase_tfim_checkpoint.toml
cargo run --release --bin solve -- configs/phase_tfim_restart.json
```

The built-in `tfim_checkpoint.toml` / `tfim_restart.toml` pair uses `dtau=0.01`,
canonicalization every 3 steps and observations every `beta=0.06`. The checkpoint config runs to
step 5 (`beta=0.1`); the restart config continues to step 10 (`beta=0.2`). The phase-TFIM
checkpoint/restart pair instead uses `dtau=0.05`, canonicalization every step, observations every
`beta=0.2` and checkpoint cadence 3. Its checkpoint config runs through step 4 (`beta=0.4`), and
the JSON restart targets step 10 (`beta=1`). Every checkpoint/restart invocation requires unused
configured JSON and HDF5 destinations. All paths are relative to the working directory. Choose
new paths to rerun the examples.

Add this optional section to enable initial, periodic and final state saving:

```toml
[checkpoint]
path = "results/part1.h5"
every_steps = 100
```

For restart, retain the model and numerical settings, increase `evolution.beta_max`, set new
`output.path` and `checkpoint.path`, and add:

```toml
[restart]
path = "results/part1.h5"
# snapshot = 3
```

The projector preset follows the same pattern. Starting from `configs/aklt_projector.toml`, use
fresh destinations for the first segment:

```toml
[output]
path = "results/aklt-projector-part1.json"
include_exact = false

[checkpoint]
path = "results/aklt-projector-part1.h5"
every_steps = 4
```

For a continuation, keep `model.type = "aklt_projector"`, `dtau`, Trotter order, truncation and
cadence settings, increase `beta_max`, and use another pair of unused destinations:

```toml
[output]
path = "results/aklt-projector-part2.json"
include_exact = false

[checkpoint]
path = "results/aklt-projector-part2.h5"
every_steps = 4

[restart]
path = "results/aklt-projector-part1.h5"
```

Omitting `snapshot` selects the latest complete checkpoint. An explicitly incomplete checkpoint,
a malformed complete snapshot, or an incompatible configuration is an error. The loader verifies
the dimensions and exact finite values of both saved Hamiltonian matrices, `dtau`, Trotter order,
truncation settings, canonicalization cadence and the saved observation interval when present.
Real and complex Hamiltonian storage compare as compatible only when the complex imaginary
components are all exactly zero. Model labels alone do not establish compatibility.
In particular, an old `aklt` checkpoint contains different Hamiltonian values and cannot become
an `aklt_projector` run by renaming a model label or tag; use a fresh projector run instead.
Library-created checkpoints without an observation interval use the interval from the new config.
The loaded state, Hamiltonian storage, progress and accumulated normalization remain authoritative:
an all-real Hamiltonian saved with complex storage stays complex after restart.

Each invocation creates one HDF5 trajectory with multiple snapshots and one JSON observation
segment. A resumed HDF5 begins with the selected saved state; its JSON contains only observations
after that start step. Original files remain untouched. Saving uses global step multiples, so an
off-cadence restart preserves canonicalization and observation schedules. Initial/final snapshots
are saved once even when they coincide with a periodic save. Targets/observation strides retain
the existing nearest-step rounding; the CLI prints actual final beta as well as requested beta.
The configured observable may change on restart because it does not change the evolved state.
Each segment records its own complete observable name and matrix; segments with different
observables must not be merged as one unchanged measurement series.

Checkpoint mode publishes complete JSON after each successful observation and checkpoint, using
same-directory temporary files and atomic replacement. `segment` records version, source
path/snapshot, start step/beta, completed steps, last checkpoint position and `finished`. Missing
segment metadata in old JSON remains supported. The final step may be later than the last
observation; no extra off-cadence observation is forced. On failure, earlier observations and
complete checkpoints remain, and `finished` stays false. Configurations without `[checkpoint]`
retain the previous final-only JSON output behavior. `[restart]` requires `[checkpoint]`.

JSON and HDF5 are separate files, not a shared transaction. A crash can leave one ahead of the
other; inspect the HDF5 complete snapshots to select a restart point. When manually joining JSON
segments, keep old records only through the selected restart step and take later records from
the resumed segment. Automatic merging/backfilling is not provided. HDF5 completion markers,
flushes and atomic JSON replacement do not guarantee recovery from a kill during HDF5 writing
or power loss. Concurrent writers to the same destinations are unsupported.

On macOS, runtime Git revision queries use descriptor-isolated process creation so Git cannot
retain checkpoint file locks through inherited file descriptors. Short revision metadata for
`solve` is still queried at runtime. This protection
covers the library's Git queries; callers that launch other subprocesses while HDF5 files are
open must manage those subprocesses' descriptor inheritance themselves. Other platforms retain
the existing standard process-launch path.

For Rust callers, `solve_run::run_checkpointed_sweep(&cfg, config_path)` performs the file I/O.
The existing `runner::run_sweep` remains computation-only and rejects checkpoint/restart options
with `ItebdError::InvalidRunConfig`. Public `RunConfig` and `SweepResult` gained optional fields;
external Rust struct-literal callers must initialize them (`None` for legacy behavior).
Continuation and independent-publication boundaries are summarized in
[`docs/numerical-conventions.md`](docs/numerical-conventions.md) and
[`docs/limitations.md`](docs/limitations.md).

## Reproducing the figures

```bash
cargo run --release --bin solve -- configs/tfim.toml
cargo run --release --bin solve -- configs/xy.toml
cargo run --release --bin solve -- configs/aklt.toml
cargo run --release --bin solve -- configs/aklt_projector.toml

python scripts/plot_run.py results/tfim.json figures/tfim_observables.png
python scripts/plot_run.py results/xy.json   figures/xy_observables.png
python scripts/plot_run.py results/aklt.json figures/aklt_observables.png
python scripts/plot_run.py results/aklt_projector.json figures/aklt_projector_observables.png
```

Each figure shows `u`, `C`, and `βf` vs β; TFIM/XY overlay the exact free-fermion curves,
the historical AKLT plot marks the ground-state energy `−2/3`, and the projector plot marks zero.
Both AKLT plots mark `βf → −ln 3` at high temperature. These guides are limiting values rather
than full exact finite-temperature curves.

## Physics validation

The test suite pins the method against known limits:

- **TFIM / XY** — full finite-T agreement with the free-fermion (Jordan–Wigner) exact
  solutions (`src/exact.rs`, `tests/`).
- **AKLT** (`tests/aklt_finite_t.rs`) — `u(T=∞) = 4/9` (the bond Hamiltonian is not traceless),
  `βf → −ln 3` as β → 0, and `u → −2/3`, `f → −2/3` as T → 0 (the AKLT ground-state energy).
- **AKLT projector** (`tests/aklt_projector.rs`, `tests/aklt_projector_evidence.rs`) — the local
  term is a rank-5 Hermitian projector, `u(T=∞) = 5/9`, `βf → −ln 3` as β → 0, and the periodic
  AKLT ground state has zero energy. The measured normalization comparison covers only the
  documented finite beta, step, cutoff and bond-cap grid.

Heavy d = 3 / large-β tests are `#[ignore]`d; run them with
`cargo test --release -- --ignored`.

## Verification

Run python3 scripts/verify.py routine for ordinary full verification. It runs
cargo test --all-targets and does not reduce verification scope. Run cargo test --doc
and ignored release benchmarks separately. Unknown project warnings fail verification.
