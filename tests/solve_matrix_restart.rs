#[path = "support/itebd_checkpoint_rdm.rs"]
#[allow(dead_code)]
mod checkpoint_support;
#[path = "support/solve_matrix.rs"]
mod matrix_support;

use thermal_imps_purification::config::RunConfig;
use thermal_imps_purification::itebd_auto::{ItebdHamiltonian, ItebdState};
use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions,
    ItebdCheckpointProgress, ItebdRunMetadata, ItebdTrajectoryWriter,
};
use thermal_imps_purification::itebd_complex::ComplexLocalHamiltonian;
use thermal_imps_purification::itebd_rdm::RdmParity;
use thermal_imps_purification::runner::{read_result, Record};
use thermal_imps_purification::solve_run::{run_checkpointed_sweep, SolveRunError};
use num_complex::Complex64;
use serde_json::{json, Value};
use std::path::Path;

static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn checkpoint_config(
    mut value: Value,
    dir: &Path,
    name: &str,
    beta_max: f64,
    restart: Option<&Path>,
) -> RunConfig {
    value["evolution"]["beta_max"] = json!(beta_max);
    value["evolution"]["record_every_beta"] = json!(0.1);
    value["output"]["path"] = json!(dir.join(format!("{name}.json")));
    value["checkpoint"] = json!({
        "path": dir.join(format!("{name}.h5")),
        "every_steps": 1
    });
    if let Some(path) = restart {
        value["restart"] = json!({"path": path});
    }
    RunConfig::from_json_str(&value.to_string()).unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-10 * actual.abs().max(expected.abs()).max(1.0),
        "{actual:.16e} != {expected:.16e}"
    );
}

fn assert_records_close(actual: &Record, expected: &Record) {
    assert_close(actual.beta, expected.beta);
    assert_close(actual.u, expected.u);
    assert_close(actual.c, expected.c);
    assert_close(actual.f, expected.f);
    assert_close(actual.magnetization, expected.magnetization);
    assert_eq!(actual.max_bond, expected.max_bond);
}

fn assert_progress_exact(
    actual: &thermal_imps_purification::itebd_checkpoint::ItebdCheckpointProgress,
    expected: &thermal_imps_purification::itebd_checkpoint::ItebdCheckpointProgress,
) {
    assert_eq!(actual.beta.to_bits(), expected.beta.to_bits());
    assert_eq!(actual.completed_steps, expected.completed_steps);
    assert_eq!(
        actual.accumulated_log_norm.to_bits(),
        expected.accumulated_log_norm.to_bits()
    );
    match (&actual.last_step, &expected.last_step) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.max_bond, expected.max_bond);
            assert_eq!(
                actual.min_singular_value.to_bits(),
                expected.min_singular_value.to_bits()
            );
            assert_eq!(actual.log_norm.to_bits(), expected.log_norm.to_bits());
        }
        (None, None) => {}
        _ => panic!("last_step presence changed"),
    }
}

fn assert_spectra_close(actual: &ItebdState, expected: &ItebdState) {
    let (actual, expected) = match (actual, expected) {
        (ItebdState::Real(actual), ItebdState::Real(expected)) => (
            [&actual.lambda_ab, &actual.lambda_ba],
            [&expected.lambda_ab, &expected.lambda_ba],
        ),
        (ItebdState::Complex(actual), ItebdState::Complex(expected)) => (
            [&actual.lambda_ab, &actual.lambda_ba],
            [&expected.lambda_ab, &expected.lambda_ba],
        ),
        _ => panic!("state backend changed"),
    };
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }
}

fn latest(path: &Path) -> thermal_imps_purification::itebd_checkpoint::LoadedItebdCheckpoint {
    let index = list_itebd_checkpoints(path).unwrap().last().unwrap().index;
    load_itebd_checkpoint(path, index, &ItebdCheckpointLoadOptions::default()).unwrap()
}

fn assert_restart_mismatch(cfg: &RunConfig, field: &'static str) {
    let output = Path::new(&cfg.output.path);
    let checkpoint = Path::new(&cfg.checkpoint.as_ref().unwrap().path);
    let error = run_checkpointed_sweep(cfg, None).unwrap_err();
    assert!(
        matches!(error, SolveRunError::RestartMismatch { field: actual } if actual == field),
        "{error}"
    );
    assert!(!output.exists());
    assert!(!checkpoint.exists());
}

#[test]
fn genuinely_complex_whole_split_restart_preserves_loaded_state_and_progress() {
    // Mutation caught: rejecting loaded Complex variants, reconstructing the backend, or resetting
    // loaded progress makes this public-driver split differ from the uninterrupted trajectory.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let whole_cfg = checkpoint_config(
        matrix_support::phase_config(),
        dir.path(),
        "whole",
        0.2,
        None,
    );
    let whole = run_checkpointed_sweep(&whole_cfg, None).unwrap();

    let split_cfg = checkpoint_config(
        matrix_support::phase_config(),
        dir.path(),
        "split",
        0.1,
        None,
    );
    run_checkpointed_sweep(&split_cfg, None).unwrap();
    let source_path = dir.path().join("split.h5");
    let source_before = std::fs::read(&source_path).unwrap();
    let source_index = list_itebd_checkpoints(&source_path)
        .unwrap()
        .last()
        .unwrap()
        .index;
    let source = load_itebd_checkpoint(
        &source_path,
        source_index,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap();
    assert!(matches!(source.state, ItebdState::Complex(_)));
    assert!(matches!(source.hamiltonian, ItebdHamiltonian::Complex(_)));

    let resumed_cfg = checkpoint_config(
        matrix_support::phase_config(),
        dir.path(),
        "resumed",
        0.2,
        Some(&source_path),
    );
    let resumed = run_checkpointed_sweep(&resumed_cfg, None).unwrap();
    assert_eq!(std::fs::read(&source_path).unwrap(), source_before);
    assert_eq!(resumed.segment.as_ref().unwrap().start_step, 1);
    assert_eq!(resumed.records.len(), 1);
    assert_records_close(&resumed.records[0], whole.records.last().unwrap());

    let destination = dir.path().join("resumed.h5");
    let resumed_initial =
        load_itebd_checkpoint(&destination, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    assert_eq!(
        checkpoint_support::snapshot(&resumed_initial.state, &resumed_initial.progress),
        checkpoint_support::snapshot(&source.state, &source.progress)
    );
    let latest = list_itebd_checkpoints(&destination)
        .unwrap()
        .last()
        .unwrap()
        .index;
    let resumed_final =
        load_itebd_checkpoint(&destination, latest, &ItebdCheckpointLoadOptions::default())
            .unwrap();
    let whole_path = dir.path().join("whole.h5");
    let whole_latest = list_itebd_checkpoints(&whole_path)
        .unwrap()
        .last()
        .unwrap()
        .index;
    let whole_final = load_itebd_checkpoint(
        &whole_path,
        whole_latest,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap();
    assert_progress_exact(&resumed_final.progress, &whole_final.progress);
    assert_spectra_close(&resumed_final.state, &whole_final.state);
    for parity in [RdmParity::A, RdmParity::B] {
        for length in [1, 2] {
            let residual = (checkpoint_support::rdm(&resumed_final.state, parity, length)
                - checkpoint_support::rdm(&whole_final.state, parity, length))
            .norm();
            assert!(residual <= 1e-10, "{parity:?} length {length}: {residual}");
        }
    }
}

#[test]
fn genuinely_complex_restart_compares_each_authoritative_matrix_component() {
    // Mutations caught: checking only the descriptive label, one matrix, or only real/imaginary
    // components would allow at least one altered evolution operator through this boundary.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let source_cfg = checkpoint_config(
        matrix_support::phase_config(),
        dir.path(),
        "source",
        0.1,
        None,
    );
    run_checkpointed_sweep(&source_cfg, None).unwrap();
    let source = dir.path().join("source.h5");

    for (name, field, mut value) in [
        (
            "two_real",
            "Hamiltonian.two_site_h",
            matrix_support::phase_config(),
        ),
        (
            "two_imag",
            "Hamiltonian.two_site_h",
            matrix_support::phase_config(),
        ),
        (
            "site_real",
            "Hamiltonian.site_energy",
            matrix_support::phase_config(),
        ),
        (
            "site_imag",
            "Hamiltonian.site_energy",
            matrix_support::phase_config(),
        ),
    ] {
        match name {
            "two_real" => value["model"]["two_site_h"]["real"][0][0] = json!(-0.9),
            "two_imag" => {
                value["model"]["two_site_h"]["imag"][0][1] = json!(0.36);
                value["model"]["two_site_h"]["imag"][1][0] = json!(-0.36);
            }
            "site_real" => value["model"]["site_energy"]["real"][0][0] = json!(-0.9),
            "site_imag" => {
                value["model"]["site_energy"]["imag"][0][1] = json!(0.71);
                value["model"]["site_energy"]["imag"][1][0] = json!(-0.71);
            }
            _ => unreachable!(),
        }
        assert_eq!(value["model"]["label"], "phase TFIM");
        let cfg = checkpoint_config(value, dir.path(), name, 0.2, Some(&source));
        assert_restart_mismatch(&cfg, field);
    }
}

#[test]
fn restart_rejects_shape_and_real_source_to_nonzero_imaginary_changes() {
    // Mutations caught: treating storage backend alone as compatibility, or zipping matrices
    // without a shape check, would accept one of these different Hamiltonians.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let source_cfg = checkpoint_config(
        matrix_support::real_tfim_config(),
        dir.path(),
        "real_source",
        0.1,
        None,
    );
    run_checkpointed_sweep(&source_cfg, None).unwrap();
    let source = dir.path().join("real_source.h5");

    let complex_cfg = checkpoint_config(
        matrix_support::phase_config(),
        dir.path(),
        "nonzero_imaginary",
        0.2,
        Some(&source),
    );
    assert_restart_mismatch(&complex_cfg, "Hamiltonian.two_site_h");

    let mut shape = matrix_support::real_tfim_config();
    shape["model"] = json!({"type": "bilinear_biquadratic", "j1": 1.0, "j2": 0.0});
    let shape_cfg = checkpoint_config(shape, dir.path(), "shape", 0.2, Some(&source));
    assert_restart_mismatch(&shape_cfg, "Hamiltonian.two_site_h");
}

#[test]
fn all_real_complex_storage_matches_matrix_values_without_reclassification() {
    // Mutation caught: requiring matching enum variants rejects a value-compatible source, while
    // rebuilding from the configured matrix silently changes the authoritative stored backend.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let cfg = checkpoint_config(
        matrix_support::real_tfim_config(),
        dir.path(),
        "resumed",
        0.1,
        Some(&dir.path().join("library.h5")),
    );
    let resolved = cfg.model.resolve().unwrap();
    let ItebdHamiltonian::Real(real) = resolved.hamiltonian else {
        panic!("exact-real matrix fixture selected complex storage")
    };
    let complex = ItebdHamiltonian::Complex(
        ComplexLocalHamiltonian::try_new_with_tolerance(
            real.two_site_h.map(|value| Complex64::new(value, -0.0)),
            real.site_energy.map(|value| Complex64::new(value, -0.0)),
            1e-12,
        )
        .unwrap(),
    );
    let state = ItebdState::infinite_temperature(&complex).unwrap();
    let metadata = ItebdRunMetadata {
        dtau: cfg.evolution.dtau,
        trotter_order: cfg.evolution.trotter_order,
        truncation: cfg.truncation.clone(),
        canonicalize_every: cfg.run.canonicalize_every,
        record_every_beta: Some(cfg.evolution.record_every_beta),
        model_label: Some("descriptive library label".into()),
        git_revision: None,
        hermiticity_tolerance: 1e-12,
    };
    let source = dir.path().join("library.h5");
    let mut writer = ItebdTrajectoryWriter::create(&source, metadata, &complex).unwrap();
    writer
        .append(
            (&state).into(),
            &ItebdCheckpointProgress {
                beta: 0.0,
                completed_steps: 0,
                accumulated_log_norm: 0.0,
                last_step: None,
            },
        )
        .unwrap();
    writer.finish().unwrap();

    run_checkpointed_sweep(&cfg, None).unwrap();
    let loaded = latest(&dir.path().join("resumed.h5"));
    assert!(matches!(loaded.state, ItebdState::Complex(_)));
    assert!(matches!(loaded.hamiltonian, ItebdHamiltonian::Complex(_)));
}

#[test]
fn complex_restart_allows_observable_change_and_records_loaded_tolerance() {
    // Mutations caught: authenticating the observable rejects this restart; using source metadata
    // tolerance instead of the loaded Hamiltonian tolerance prevents the new writer from opening.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let split_cfg = checkpoint_config(
        matrix_support::phase_config(),
        dir.path(),
        "split",
        0.1,
        None,
    );
    run_checkpointed_sweep(&split_cfg, None).unwrap();
    let split = latest(&dir.path().join("split.h5"));
    let ItebdHamiltonian::Complex(split_hamiltonian) = &split.hamiltonian else {
        panic!("genuinely complex fixture selected real storage")
    };
    let source_hamiltonian = ItebdHamiltonian::Complex(
        ComplexLocalHamiltonian::try_new_with_tolerance(
            split_hamiltonian.two_site_h().clone(),
            split_hamiltonian.site_energy().clone(),
            1e-9,
        )
        .unwrap(),
    );
    let mut source_metadata = split.metadata.clone();
    source_metadata.hermiticity_tolerance = 1e-9;
    source_metadata.model_label = Some("source observable is not authentication".into());
    let source = dir.path().join("library.h5");
    let mut writer =
        ItebdTrajectoryWriter::create(&source, source_metadata, &source_hamiltonian).unwrap();
    writer
        .append((&split.state).into(), &split.progress)
        .unwrap();
    writer.finish().unwrap();
    let source_before = std::fs::read(&source).unwrap();

    let mut changed = matrix_support::phase_config();
    changed["model"]["observable"] = json!({
        "name": "sigma_z",
        "matrix": {"real": [[1.0, 0.0], [0.0, -1.0]]}
    });
    let resumed_cfg = checkpoint_config(changed.clone(), dir.path(), "resumed", 0.2, Some(&source));
    let resumed = run_checkpointed_sweep(&resumed_cfg, None).unwrap();
    assert_eq!(std::fs::read(&source).unwrap(), source_before);
    assert_eq!(
        latest(&source).metadata.hermiticity_tolerance.to_bits(),
        1e-9f64.to_bits()
    );

    let direct_cfg = checkpoint_config(changed, dir.path(), "direct", 0.2, None);
    let direct = run_checkpointed_sweep(&direct_cfg, None).unwrap();
    assert_eq!(resumed.records.len(), 1);
    assert_records_close(&resumed.records[0], direct.records.last().unwrap());
    assert_eq!(
        serde_json::to_value(&resumed.metadata.model).unwrap(),
        serde_json::to_value(&resumed_cfg.model).unwrap()
    );
    let published = read_result(&dir.path().join("resumed.json")).unwrap();
    assert_eq!(
        serde_json::to_value(&published.metadata.model).unwrap(),
        serde_json::to_value(&resumed_cfg.model).unwrap()
    );
    let continued = latest(&dir.path().join("resumed.h5"));
    assert_eq!(
        continued.metadata.hermiticity_tolerance.to_bits(),
        ItebdCheckpointLoadOptions::default()
            .hermiticity_tolerance
            .to_bits()
    );
    let stored_model: thermal_imps_purification::config::ModelSpec =
        serde_json::from_str(continued.metadata.model_label.as_deref().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(stored_model).unwrap(),
        serde_json::to_value(&resumed_cfg.model).unwrap()
    );
}
