# Library API

The crate exposes direct real and complex two-site iTEBD interfaces and an automatic facade,
along with fallible observables, HDF5 checkpoints, and physical interval reduced density
matrices (RDMs). See [numerical conventions](numerical-conventions.md),
[validation](validation.md), and [limitations](limitations.md) for qualified numerical claims.

## Real and automatic evolution

The direct `f64` path avoids complex allocation and enum dispatch:

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
        &Truncation { epsilon: 1e-12, max_bond: Some(64) },
    );
    assert!(step.log_norm.is_finite());
}
```

Use automatic dispatch when Hamiltonian matrices may be complex:

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

`ItebdHamiltonian::try_from_complex` validates dimensions, finite entries, and Hermiticity.
It selects real storage only when every imaginary Hamiltonian component is exactly zero. Backend
mismatches return typed errors, and inputs are never silently symmetrized.

`ModelSpec` construction is fallible. Real callers can use `model.hamiltonian()?` and
`model.magnetization_op()?`; automatic callers use `let resolved = model.resolve()?;`.
`ModelSpec::AkltProjector` is public, so external exhaustive matches must include it.

## Specific heat

Scalar APIs are fallible because invalid beta, malformed tensors, numerical-reality failures,
and unconverged correlation tails are errors:

```rust
use thermal_imps_purification::variance::{specific_heat, specific_heat_with_options};
use thermal_imps_purification::specific_heat::SpecificHeatOptions;

fn real_specific_heat(
    state: thermal_imps_purification::purified_mps::PurifiedMps,
    hamiltonian: thermal_imps_purification::model::LocalHamiltonian,
    beta: f64,
) -> Result<f64, thermal_imps_purification::itebd_error::ItebdError> {
    let heat = specific_heat(&state, &hamiltonian, beta)?;
    let report = specific_heat_with_options(
        &state, &hamiltonian, beta, &SpecificHeatOptions::default(),
    )?;
    assert_eq!(heat, report.specific_heat_per_site);
    Ok(heat)
}
```

The complex interface follows the same boundary:

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

`specific_heat_with_options_auto` provides the automatic report call. Positive and negative
correlation directions are evaluated independently. Defaults cap distance at 200 and require
three consecutive small shells; failure returns
`ItebdError::SpecificHeatTailNonConvergence` rather than a partial scalar. Both the direction
and parity decompositions reconstruct `raw_energy_variance_per_site`. The validated variance
differs only when a small negative raw value is within the documented projection tolerance.

## Checkpoints

Run the complete save/load/continue example with a fresh path:

```bash
cargo run --release --example itebd_checkpoint_rdm -- /tmp/itebd-checkpoint-rdm-example.h5
```

The example records beta zero, evolves five steps, saves and reloads snapshot 1, then evolves
five more steps while printing energy, free energy, and A/B length-2 RDM reports. Creation is
exclusive; supply a new path for every run.

`itebd_checkpoint::{ItebdTrajectoryWriter, list_itebd_checkpoints, load_itebd_checkpoint}`
stores real and complex states through borrowed `ItebdStateRef` views. The version-1 two-site
schema stores Gamma tensors, Schmidt values, index identities and dimensions, Hamiltonian
matrices, settings, progress, and diagnostics. Save/load does not canonicalize or normalize.
Persist completed steps and accumulated log norm so off-cadence restarts retain their schedule.
A stored complex backend remains complex even when every component is real.

Incomplete snapshots are excluded from listing; explicit loading fails. An append I/O failure
poisons the writer. Completion markers and flushes are not filesystem transactions or power-loss
guarantees. See [CLI restart configuration](usage.md#automatic-checkpoints-and-restart).

## Physical interval RDMs

Use `itebd_rdm::{reduced_density_matrix, reduced_density_matrix_complex,
reduced_density_matrix_auto}` for real, complex, or automatic dispatch. Choose `RdmParity::A`
or `RdmParity::B` for the first site and a positive interval length. Ancillas are traced out.
The normalized `DMatrix` uses first-physical-site-fastest row and column indices:
`p0 + d*p1 + ...`.

`RdmReport` includes boundary fixed-point residuals, iteration counts and eigenvalues, trace
diagnostics, Hermiticity residual, minimum eigenvalue, and largest retained tensor size. The API
does not repair results by spectrum clipping or Hermitian projection.

`RdmOptions` defaults to fixed-point tolerance `1e-12`, at most 10,000 iterations,
trace/Hermiticity/positivity tolerances `1e-10`, 1,048,576 output elements, and 16,777,216
intermediate elements. Invalid options, nonconvergence, malformed boundaries, and resource
overruns return typed errors. Caps bound planned sizes rather than aggregate memory or hidden
backend workspace. Degenerate dominant transfer sectors can make identity-seeded boundaries
sector-dependent even when residual checks pass.

For automatic file I/O, `solve_run::run_checkpointed_sweep(&cfg, config_path)` drives the CLI
checkpoint path. `runner::run_sweep` remains computation-only and rejects checkpoint/restart
options. External struct-literal users must initialize the optional fields added to public
`RunConfig` and `SweepResult`; `None` preserves legacy behavior.

Fresh sweeps return a beta-zero observation as the first `runner::Record`, followed by
the existing observation schedule. Restarted sweeps return only observations after the
selected checkpoint step. `Record::f` and `config::ExactRefs::f` are `Option<f64>`:
`None` represents the divergent free energy at beta zero and `Some(f)` the value at
positive beta. `Record::beta_f` is finite, with initial value `-ln(local_dim)`.
External struct literals must supply `beta_f` and wrap positive-beta free energies in
`Some`; consumers should handle the initial `None` explicitly.

`runner::read_result` accepts older positive-beta JSON records lacking `beta_f` and derives
it from `beta * f`. New beta-zero records require `f: null` and a finite `beta_f`.
Missing `f`, a null positive-beta `f`, explicit null `beta_f`, and a positive-beta
`beta_f` inconsistent with `beta * f` are rejected during deserialization.

## Scheduled checkpoints and fixed Ising reference

`ItebdTrajectoryWriter::create_scheduled(path, metadata, hamiltonian, segments)` writes schema2
with `Vec<ItebdStepSegment { steps, dtau }>` and metadata `dtau` as the maximum allowed step.
`LoadedItebdCheckpoint::time_grid` distinguishes `ItebdTimeGrid::Fixed` (historical schema1)
from `Segmented`. Progress validates against the actual prefix sum, including a partial segment;
never reconstruct beta from average or maximum dtau. The fixed-step solve CLI cannot restart it.

`ising_reference::{resolve_request, preflight, run_reference, measure_reference}` supplies the
strict fixed mixed-field request, immutable resolved controls, scalar target callback, terminal
outcome and ordinary physical thermodynamics. The output module exposes:

- `run_to_directory(&ReferenceRequest, &Path) -> anyhow::Result<serde_json::Value>`
- `verify_directory(&Path) -> anyhow::Result<serde_json::Value>` (read-only)
- `finalize_interrupted_directory(&Path, &str) -> anyhow::Result<serde_json::Value>`

The finalizer is only for a supervisor-observed dead process and cannot restart/evolve/replace
an endpoint. A callback or publication error propagates; native panics are not caught. On an
ordinary evolution error, `ReferenceOutcome::terminal_state` is absent because a failed native
call may have mutated tensors. `ReferenceSample::values` is optional and serde-flattened; CSV
publication explicitly emits every column for gaps. See the [worker contract](usage.md#fixed-mixed-field-ising-reference-worker)
for complete controls, schema, normalization, integrity checks, and limitations.


## Fixed AKLT projector reference

`aklt_reference::{resolve_request, preflight, run_reference, measure_reference}` uses
`AkltReferenceRequest`, `AkltReferenceSettings`, `AkltSample` and `AkltOutcome`.
The target callback receives the fixed scalar row and cumulative `CapEvidence`.
`AkltSampleValues` contains spin-one mx/mz, ordinary energy/free-energy/entropy,
RDM diagnostics and `cap_reached_now`; it has no zz field. `AkltOutcome` records
actual progress, nullable terminal state, cap coverage and cooperative budget overrun.
The shared scheduled runner is private; existing Ising public types remain supported.

`aklt_reference::output` exposes `run_to_directory(&AkltReferenceRequest, &Path)`,
`verify_directory(&Path)` and `finalize_interrupted_directory(&Path, &str)`, each
returning `anyhow::Result<serde_json::Value>`. The finalizer never evolves and requires
an externally observed dead process. See the [AKLT worker contract](usage.md#fixed-aklt-projector-reference-worker)
for exact fields, deadlines, normalization and the limits of cap evidence.

## AKLT reference v3 publication

`aklt_reference::v3::{AkltReferenceRequestV3, resolve_v3_request, preflight_v3,
measure_v3_reference, capture_runtime}` defines fixed cap256 controls, variance
measurement evidence and native launch metadata. `AkltValuesV3` retains physical
values when variance measurement fails, with typed scalar diagnostics or the
actual failure reason. Existing `measure_reference` remains cap128.

`aklt_reference::v3_output::{run_to_directory, verify_directory,
finalize_interrupted_directory}` owns the20-target native HDF5 bundle.
The existing parent `output::{verify_directory, finalize_interrupted_directory}`
dispatches by explicit contract. `parse_worker_request` returns the explicit
`AkltWorkerRequest::{Legacy,V3}` union, and `preflight_worker` resolves it.
The shared scheduled trajectory loop uses an internal target observer before
measurement; legacy scalar callbacks remain adapters to the same evolution.
V3 verification returns the summary plus sample/snapshot counts,
`checkpoint_verified`, `scalar_state_consistency_passed`, and
`variance_remeasured:false`. A later verifier's environment never replaces
`runtime_at_launch`. See [usage](usage.md#aklt-cap256-target-snapshots-and-variance-heat).
