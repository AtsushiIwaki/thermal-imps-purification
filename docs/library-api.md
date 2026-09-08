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
