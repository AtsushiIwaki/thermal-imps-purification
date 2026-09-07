#[path = "support/itebd_checkpoint_rdm.rs"]
mod support;

use thermal_imps_purification::config::TrotterOrder;
use thermal_imps_purification::itebd::{free_energy_from_log_norm, StepInfo};
use thermal_imps_purification::itebd_auto::{energy_density_auto, ItebdHamiltonian, ItebdState};
use thermal_imps_purification::itebd_checkpoint::{
    load_itebd_checkpoint, ItebdCheckpointLoadOptions, ItebdCheckpointProgress,
    ItebdTrajectoryWriter,
};
use thermal_imps_purification::model::Tfim;

#[test]
fn off_cadence_checkpoint_continues_real_and_complex_first_and_second_order_exactly() {
    for complex in [false, true] {
        for trotter_order in [TrotterOrder::First, TrotterOrder::Second] {
            exercise_split_run(complex, trotter_order);
        }
    }
}

fn exercise_split_run(complex: bool, trotter_order: TrotterOrder) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!(
        "{}-{:?}.h5",
        if complex { "complex" } else { "real" },
        trotter_order
    ));
    let h = if complex {
        support::phase_tfim()
    } else {
        ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local())
    };
    let mut metadata = support::real_tfim_metadata();
    metadata.trotter_order = trotter_order;
    let mut uninterrupted = ItebdState::infinite_temperature(&h).unwrap();
    let mut saved_progress = zero_progress();
    support::advance(&mut uninterrupted, &h, &metadata, &mut saved_progress, 5).unwrap();

    let mut writer = ItebdTrajectoryWriter::create(&path, metadata.clone(), &h).unwrap();
    let before_append = support::snapshot(&uninterrupted, &saved_progress);
    writer
        .append((&uninterrupted).into(), &saved_progress)
        .unwrap();
    assert_eq!(
        before_append,
        support::snapshot(&uninterrupted, &saved_progress)
    );
    writer.finish().unwrap();

    let loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    compare_rdms(&loaded.state, &uninterrupted, "loaded checkpoint");
    let mut resumed = loaded.state;
    let mut resumed_progress = loaded.progress;
    let resumed_h = loaded.hamiltonian;
    let resumed_metadata = loaded.metadata;
    let loaded_for_negative_control =
        load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    let mut reset = loaded_for_negative_control.state;
    let mut reset_progress = loaded_for_negative_control.progress;
    let reset_h = loaded_for_negative_control.hamiltonian;
    let reset_metadata = loaded_for_negative_control.metadata;
    reset_progress.completed_steps = 0;

    support::advance(&mut uninterrupted, &h, &metadata, &mut saved_progress, 5).unwrap();
    support::advance(
        &mut resumed,
        &resumed_h,
        &resumed_metadata,
        &mut resumed_progress,
        5,
    )
    .unwrap();
    support::advance(
        &mut reset,
        &reset_h,
        &reset_metadata,
        &mut reset_progress,
        5,
    )
    .unwrap();

    assert_eq!(saved_progress.completed_steps, 10);
    assert_eq!(resumed_progress.completed_steps, 10);
    assert_eq!(saved_progress.beta.to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        resumed_progress.beta.to_bits(),
        saved_progress.beta.to_bits()
    );
    assert_close(
        resumed_progress.accumulated_log_norm,
        saved_progress.accumulated_log_norm,
        "accumulated log norm",
    );
    for (actual, expected) in schmidt_spectra(&resumed)
        .into_iter()
        .zip(schmidt_spectra(&uninterrupted))
    {
        assert_slices_close(actual, expected, "Schmidt spectrum");
    }
    compare_rdms(&resumed, &uninterrupted, "continued checkpoint");
    let expected_energy = energy_density_auto(&uninterrupted, &h).unwrap();
    let actual_energy = energy_density_auto(&resumed, &resumed_h).unwrap();
    assert_close(actual_energy, expected_energy, "energy density");
    let expected_free_energy = free_energy_from_log_norm(
        saved_progress.accumulated_log_norm / 2.0,
        saved_progress.beta,
        2,
    );
    let actual_free_energy = free_energy_from_log_norm(
        resumed_progress.accumulated_log_norm / 2.0,
        resumed_progress.beta,
        2,
    );
    assert_close(actual_free_energy, expected_free_energy, "free energy");
    eprintln!(
        "backend={} order={trotter_order:?} norm_abs={:.3e} schmidt_max_abs={:.3e} energy_abs={:.3e} free_energy_abs={:.3e}",
        if complex { "complex" } else { "real" },
        (resumed_progress.accumulated_log_norm - saved_progress.accumulated_log_norm).abs(),
        max_schmidt_difference(&resumed, &uninterrupted),
        (actual_energy - expected_energy).abs(),
        (actual_free_energy - expected_free_energy).abs(),
    );

    assert_eq!(reset_progress.completed_steps, 5);
    assert_eq!(reset_progress.beta.to_bits(), 0.5_f64.to_bits());
    assert_eq!(scheduled_events(5, 10, metadata.canonicalize_every), 2);
    assert_eq!(scheduled_events(0, 5, reset_metadata.canonicalize_every), 1);
    let reset_free_energy = free_energy_from_log_norm(
        reset_progress.accumulated_log_norm / 2.0,
        reset_progress.beta,
        2,
    );
    assert!(
        (reset_free_energy - expected_free_energy).abs() > 1e-10,
        "reset progress unexpectedly reproduced the continuation free energy"
    );
}

fn scheduled_events(start_step: u64, end_step: u64, cadence: usize) -> u64 {
    end_step / cadence as u64 - start_step / cadence as u64
}

fn zero_progress() -> ItebdCheckpointProgress {
    ItebdCheckpointProgress {
        beta: 0.0,
        completed_steps: 0,
        accumulated_log_norm: 0.0,
        last_step: None::<StepInfo>,
    }
}

fn schmidt_spectra(state: &ItebdState) -> [&[f64]; 2] {
    match state {
        ItebdState::Real(state) => [&state.lambda_ab, &state.lambda_ba],
        ItebdState::Complex(state) => [&state.lambda_ab, &state.lambda_ba],
    }
}

fn assert_slices_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} length");
    for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let scale = 1.0_f64.max(actual.abs()).max(expected.abs());
        assert!(
            (actual - expected).abs() <= 1e-10 * scale,
            "{context}[{position}]: {actual} != {expected}"
        );
    }
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    let scale = 1.0_f64.max(actual.abs()).max(expected.abs());
    assert!(
        (actual - expected).abs() <= 1e-10 * scale,
        "{context}: {actual} != {expected}"
    );
}

fn max_schmidt_difference(actual: &ItebdState, expected: &ItebdState) -> f64 {
    schmidt_spectra(actual)
        .into_iter()
        .zip(schmidt_spectra(expected))
        .flat_map(|(actual, expected)| actual.iter().zip(expected))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

use thermal_imps_purification::itebd_rdm::RdmParity;
use thermal_imps_purification::{itebd_auto, itebd_complex, itebd_state_view, purified_mps, tensor};
#[allow(dead_code)]
#[path = "support/itebd_rdm_invariants.rs"]
mod invariants;
#[allow(dead_code)]
#[path = "../src/itebd_rdm/oracles.rs"]
mod oracles;

fn compare_rdms(actual: &ItebdState, expected: &ItebdState, stage: &str) {
    for start in [RdmParity::A, RdmParity::B] {
        for length in 1..=2 {
            let error = (support::rdm(actual, start, length)
                - support::rdm(expected, start, length))
            .norm();
            eprintln!("{stage} start={start:?} n={length} rdm_error={error:e}");
            assert!(error <= 1e-10);
        }
    }
}

#[test]
fn append_is_read_only_for_noncanonical_states_and_loaded_complex_matches_oracle() {
    for complex in [false, true] {
        let state = oracles::fixture(complex, 2, 3);
        let h = if complex {
            support::phase_tfim()
        } else {
            ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local())
        };
        let progress = ItebdCheckpointProgress {
            beta: 0.5,
            completed_steps: 5,
            accumulated_log_norm: -0.125,
            last_step: Some(StepInfo {
                max_bond: 3,
                min_singular_value: 0.25,
                log_norm: -0.03125,
            }),
        };
        let view = itebd_state_view::ItebdStateRef::from(&state);
        // Unit lambdas with unequal bonds have norms sqrt(2) and sqrt(3), so
        // canonicalization/normalization cannot be hidden by a canonical input.
        assert_eq!(view.lambda_ab(), &[1.0, 1.0]);
        assert_eq!(view.lambda_ba(), &[1.0, 1.0, 1.0]);
        let before = support::snapshot(&state, &progress);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noncanonical.h5");
        let mut writer =
            ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
        writer.append((&state).into(), &progress).unwrap();
        assert_eq!(before, support::snapshot(&state, &progress));
        writer.finish().unwrap();
        assert_eq!(before, support::snapshot(&state, &progress));
        let loaded =
            load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
        // Loading preserves serialized IDs. Only subsequent independent evolution
        // may create different new IDs; physical RDM comparisons remain meaningful.
        assert_eq!(
            view.a().gamma.is_complex(),
            itebd_state_view::ItebdStateRef::from(&loaded.state)
                .a()
                .gamma
                .is_complex()
        );
        for start in [RdmParity::A, RdmParity::B] {
            for length in 1..=4 {
                let expected =
                    invariants::oracle_rdm(&state, usize::from(start == RdmParity::B), length);
                let error = (support::rdm(&loaded.state, start, length) - expected).norm();
                eprintln!(
                    "loaded oracle complex={complex} start={start:?} n={length} error={error:e}"
                );
                assert!(error <= 1e-10);
            }
        }
    }
}
