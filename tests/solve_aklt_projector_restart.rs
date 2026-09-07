#[path = "support/itebd_checkpoint_rdm.rs"]
#[allow(dead_code)]
mod checkpoint_support;
#[path = "support/aklt_projector_solve.rs"]
mod support;

use thermal_imps_purification::config::{ModelSpec, RunConfig};
use thermal_imps_purification::itebd_auto::{ItebdHamiltonian, ItebdState};
use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions, ItebdRunMetadata,
    ItebdTrajectoryWriter,
};
use thermal_imps_purification::itebd_rdm::RdmParity;
use thermal_imps_purification::itebd_state_view::{ItebdScalarType, ItebdStateRef};
use thermal_imps_purification::model::{sz1, AkltProjector, LocalHamiltonian};
use thermal_imps_purification::runner::read_result;
use thermal_imps_purification::solve_run::{run_checkpointed_sweep, SolveRunError};
use nalgebra::DMatrix;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

// HDF5 native descriptors can be inherited by concurrently launched solve processes.
// Every CLI launch and direct HDF5 operation in this target shares this lock.
static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn checkpoint_value(
    dir: &Path,
    name: &str,
    kind: &str,
    order: u8,
    beta: f64,
    restart: Option<&Path>,
) -> Value {
    let mut value = support::config(dir, name, kind, order, beta);
    value["checkpoint"] = json!({
        "path": dir.join(format!("{name}.h5")),
        "every_steps": 1
    });
    if let Some(path) = restart {
        value["restart"] = json!({"path": path});
    }
    value
}

fn checkpoint_config(
    dir: &Path,
    name: &str,
    kind: &str,
    beta: f64,
    restart: Option<&Path>,
) -> RunConfig {
    RunConfig::from_json_str(&checkpoint_value(dir, name, kind, 2, beta, restart).to_string())
        .unwrap()
}

fn run_cli_success(path: &Path) {
    let output = support::run_cli(path);
    assert!(
        output.status.success(),
        "config={} status={} stdout={} stderr={}",
        path.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn checkpoint_steps(path: &Path) -> Vec<u64> {
    list_itebd_checkpoints(path)
        .unwrap()
        .iter()
        .map(|entry| entry.completed_steps)
        .collect()
}

fn parse_model_label(metadata: &ItebdRunMetadata) -> ModelSpec {
    serde_json::from_str(metadata.model_label.as_deref().unwrap()).unwrap()
}

fn real_hamiltonian(hamiltonian: &ItebdHamiltonian) -> &LocalHamiltonian {
    match hamiltonian {
        ItebdHamiltonian::Real(hamiltonian) => hamiltonian,
        ItebdHamiltonian::Complex(_) => {
            panic!("AKLT projector checkpoint selected complex storage")
        }
    }
}

fn assert_finite_matrix(matrix: &DMatrix<f64>) {
    assert!(matrix.iter().all(|value| value.is_finite()));
}

fn assert_projector_hamiltonian(hamiltonian: &ItebdHamiltonian) {
    let actual = real_hamiltonian(hamiltonian);
    let expected = AkltProjector.local();
    assert_finite_matrix(&actual.two_site_h);
    assert_finite_matrix(&actual.site_energy);
    assert_eq!(actual.two_site_h, expected.two_site_h);
    assert_eq!(actual.site_energy, expected.site_energy);
}

fn assert_hamiltonian_exact(actual: &ItebdHamiltonian, expected: &ItebdHamiltonian) {
    let actual = real_hamiltonian(actual);
    let expected = real_hamiltonian(expected);
    assert_finite_matrix(&actual.two_site_h);
    assert_finite_matrix(&actual.site_energy);
    assert_finite_matrix(&expected.two_site_h);
    assert_finite_matrix(&expected.site_energy);
    assert_eq!(actual.two_site_h, expected.two_site_h);
    assert_eq!(actual.site_energy, expected.site_energy);
}

fn assert_metadata_exact(actual: &ItebdRunMetadata, expected: &ItebdRunMetadata) {
    assert!(actual.dtau.is_finite() && expected.dtau.is_finite());
    assert!(actual.truncation.epsilon.is_finite() && expected.truncation.epsilon.is_finite());
    assert!(actual.hermiticity_tolerance.is_finite() && expected.hermiticity_tolerance.is_finite());
    assert!(
        actual.record_every_beta.into_iter().all(f64::is_finite)
            && expected.record_every_beta.into_iter().all(f64::is_finite)
    );
    assert_eq!(actual.dtau.to_bits(), expected.dtau.to_bits());
    assert_eq!(actual.trotter_order, expected.trotter_order);
    assert_eq!(
        actual.truncation.epsilon.to_bits(),
        expected.truncation.epsilon.to_bits()
    );
    assert_eq!(actual.truncation.max_bond, expected.truncation.max_bond);
    assert_eq!(actual.canonicalize_every, expected.canonicalize_every);
    assert_eq!(
        actual.record_every_beta.map(f64::to_bits),
        expected.record_every_beta.map(f64::to_bits)
    );
    assert_eq!(actual.model_label, expected.model_label);
    assert_eq!(actual.git_revision, expected.git_revision);
    assert_eq!(
        actual.hermiticity_tolerance.to_bits(),
        expected.hermiticity_tolerance.to_bits()
    );
}

fn assert_spectra_close(actual: &ItebdState, expected: &ItebdState) {
    let actual = ItebdStateRef::from(actual);
    let expected = ItebdStateRef::from(expected);
    assert_eq!(actual.scalar_type(), expected.scalar_type());
    for (actual, expected) in [actual.lambda_ab(), actual.lambda_ba()]
        .into_iter()
        .zip([expected.lambda_ab(), expected.lambda_ba()])
    {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            support::close(*actual, *expected, 1e-10);
        }
    }
}

fn copy_checkpoint(source: &Path, destination: &Path, model_label: ModelSpec) {
    let loaded = support::latest(source);
    let mut metadata = loaded.metadata.clone();
    metadata.model_label = Some(serde_json::to_string(&model_label).unwrap());
    let mut writer =
        ItebdTrajectoryWriter::create(destination, metadata, &loaded.hamiltonian).unwrap();
    writer
        .append((&loaded.state).into(), &loaded.progress)
        .unwrap();
    writer.finish().unwrap();
}

fn assert_restart_mismatch(cfg: &RunConfig) {
    let output = Path::new(&cfg.output.path);
    let checkpoint = Path::new(&cfg.checkpoint.as_ref().unwrap().path);
    let error = run_checkpointed_sweep(cfg, None).unwrap_err();
    assert!(matches!(
        error,
        SolveRunError::RestartMismatch {
            field: "Hamiltonian.two_site_h"
        }
    ));
    assert!(!output.exists());
    assert!(!checkpoint.exists());
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}

fn inline_projector_value(dir: &Path, name: &str, source: &ItebdHamiltonian) -> Value {
    let source = real_hamiltonian(source);
    let mut value = checkpoint_value(dir, name, "aklt_projector", 2, 0.1, None);
    value["model"] = json!({
        "type": "matrix",
        "version": 1,
        "local_dim": 3,
        "basis_order": "first_site_fastest",
        "label": "exact stored AKLT projector matrix",
        "two_site_h": {"real": matrix_rows(&source.two_site_h)},
        "site_energy": {"real": matrix_rows(&source.site_energy)},
        "observable": {"name": "sz1", "matrix": {"real": matrix_rows(&sz1())}}
    });
    value
}

#[test]
fn projector_cli_whole_split_restart_preserves_normalization_and_state() {
    // Mutations caught: resetting progress, changing the projector normalization, reclassifying
    // real storage, or serializing a different state breaks an endpoint or lineage assertion.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let root = tempfile::tempdir().unwrap();
    let mut process_count = 0;

    for format in ["toml", "json"] {
        for order in [1, 2] {
            let dir = root.path().join(format!("{format}-order-{order}"));
            std::fs::create_dir(&dir).unwrap();
            let whole_value = checkpoint_value(&dir, "whole", "aklt_projector", order, 0.2, None);
            let split_value = checkpoint_value(&dir, "split", "aklt_projector", order, 0.1, None);
            let restart_value = checkpoint_value(
                &dir,
                "restart",
                "aklt_projector",
                order,
                0.2,
                Some(&dir.join("split.h5")),
            );
            let whole_input = dir.join(format!("whole.{format}"));
            let split_input = dir.join(format!("split.{format}"));
            let restart_input = dir.join(format!("restart.{format}"));
            support::write_input(&whole_value, &whole_input, format);
            support::write_input(&split_value, &split_input, format);
            support::write_input(&restart_value, &restart_input, format);

            run_cli_success(&whole_input);
            process_count += 1;
            run_cli_success(&split_input);
            process_count += 1;
            let source_path = dir.join("split.h5");
            let source_before = std::fs::read(&source_path).unwrap();
            run_cli_success(&restart_input);
            process_count += 1;
            assert_eq!(std::fs::read(&source_path).unwrap(), source_before);

            let whole_path = dir.join("whole.h5");
            let restart_path = dir.join("restart.h5");
            assert_eq!(checkpoint_steps(&whole_path), vec![0, 1, 2]);
            assert_eq!(checkpoint_steps(&source_path), vec![0, 1]);
            assert_eq!(checkpoint_steps(&restart_path), vec![1, 2]);

            let whole_result =
                read_result(Path::new(whole_value["output"]["path"].as_str().unwrap())).unwrap();
            let restart_result =
                read_result(Path::new(restart_value["output"]["path"].as_str().unwrap())).unwrap();
            assert_eq!(restart_result.records.len(), 1);
            support::close(restart_result.records[0].beta, 0.2, 1e-10);
            support::records_close(
                &restart_result.records[0],
                whole_result.records.last().unwrap(),
            );

            let whole_final = support::latest(&whole_path);
            let split_final = support::latest(&source_path);
            let restart_initial =
                load_itebd_checkpoint(&restart_path, 0, &ItebdCheckpointLoadOptions::default())
                    .unwrap();
            let restart_final = support::latest(&restart_path);
            support::close(
                restart_final.progress.accumulated_log_norm,
                whole_final.progress.accumulated_log_norm,
                1e-10,
            );
            assert_eq!(
                ItebdStateRef::from(&whole_final.state).scalar_type(),
                ItebdScalarType::Real
            );
            assert_eq!(
                ItebdStateRef::from(&restart_final.state).scalar_type(),
                ItebdScalarType::Real
            );
            for loaded in [&whole_final, &restart_final] {
                assert!(matches!(
                    parse_model_label(&loaded.metadata),
                    ModelSpec::AkltProjector
                ));
                assert_projector_hamiltonian(&loaded.hamiltonian);
            }
            assert_eq!(
                checkpoint_support::snapshot(&split_final.state, &split_final.progress),
                checkpoint_support::snapshot(&restart_initial.state, &restart_initial.progress)
            );

            let roundtrip_path = dir.join("roundtrip.h5");
            let mut writer = ItebdTrajectoryWriter::create(
                &roundtrip_path,
                restart_final.metadata.clone(),
                &restart_final.hamiltonian,
            )
            .unwrap();
            writer
                .append((&restart_final.state).into(), &restart_final.progress)
                .unwrap();
            writer.finish().unwrap();
            let roundtrip = support::latest(&roundtrip_path);
            assert_eq!(
                checkpoint_support::snapshot(&roundtrip.state, &roundtrip.progress),
                checkpoint_support::snapshot(&restart_final.state, &restart_final.progress)
            );
            assert_hamiltonian_exact(&roundtrip.hamiltonian, &restart_final.hamiltonian);
            assert_metadata_exact(&roundtrip.metadata, &restart_final.metadata);

            assert_spectra_close(&restart_final.state, &whole_final.state);
            for parity in [RdmParity::A, RdmParity::B] {
                let residual = (checkpoint_support::rdm(&restart_final.state, parity, 1)
                    - checkpoint_support::rdm(&whole_final.state, parity, 1))
                .norm();
                assert!(
                    residual <= 1e-10,
                    "{format} order {order} {parity:?}: {residual}"
                );
            }
        }
    }
    assert_eq!(process_count, 12);
}

#[test]
fn restart_rejects_old_and_projector_normalizations_even_when_relabelled() {
    // Mutation caught: comparing model labels instead of both authoritative matrices would accept
    // at least one relabelled checkpoint with the wrong Hamiltonian normalization.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let old_cfg = checkpoint_config(dir.path(), "old-source", "aklt", 0.1, None);
    let projector_cfg =
        checkpoint_config(dir.path(), "projector-source", "aklt_projector", 0.1, None);
    run_checkpointed_sweep(&old_cfg, None).unwrap();
    run_checkpointed_sweep(&projector_cfg, None).unwrap();

    let old_source = dir.path().join("old-source.h5");
    let projector_source = dir.path().join("projector-source.h5");
    let old_relabelled = dir.path().join("old-as-projector.h5");
    let projector_relabelled = dir.path().join("projector-as-old.h5");
    copy_checkpoint(&old_source, &old_relabelled, ModelSpec::AkltProjector);
    copy_checkpoint(&projector_source, &projector_relabelled, ModelSpec::Aklt);

    for (name, kind, source) in [
        ("old-original-to-projector", "aklt_projector", &old_source),
        (
            "old-relabelled-to-projector",
            "aklt_projector",
            &old_relabelled,
        ),
        ("projector-original-to-old", "aklt", &projector_source),
        ("projector-relabelled-to-old", "aklt", &projector_relabelled),
    ] {
        let before = std::fs::read(source).unwrap();
        let cfg = checkpoint_config(dir.path(), name, kind, 0.2, Some(source));
        assert_restart_mismatch(&cfg);
        assert_eq!(std::fs::read(source).unwrap(), before);
    }
}

#[test]
fn projector_restart_accepts_descriptive_relabel_and_exact_inline_matrix_source() {
    // Mutation caught: authenticating model labels or requiring the same config variant would
    // reject a checkpoint whose authoritative matrices exactly match the projector preset.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let projector_cfg =
        checkpoint_config(dir.path(), "projector-source", "aklt_projector", 0.1, None);
    run_checkpointed_sweep(&projector_cfg, None).unwrap();
    let projector_source = dir.path().join("projector-source.h5");
    let projector_as_old = dir.path().join("projector-as-old.h5");
    copy_checkpoint(&projector_source, &projector_as_old, ModelSpec::Aklt);
    assert!(matches!(
        parse_model_label(&support::latest(&projector_as_old).metadata),
        ModelSpec::Aklt
    ));
    let relabelled_before = std::fs::read(&projector_as_old).unwrap();
    let relabelled_cfg = checkpoint_config(
        dir.path(),
        "relabelled-resume",
        "aklt_projector",
        0.2,
        Some(&projector_as_old),
    );
    run_checkpointed_sweep(&relabelled_cfg, None).unwrap();
    assert_eq!(std::fs::read(&projector_as_old).unwrap(), relabelled_before);

    let stored = support::latest(&projector_source);
    let inline_value = inline_projector_value(dir.path(), "inline-source", &stored.hamiltonian);
    let inline_cfg = RunConfig::from_json_str(&inline_value.to_string()).unwrap();
    assert_hamiltonian_exact(
        &inline_cfg.model.resolve().unwrap().hamiltonian,
        &stored.hamiltonian,
    );
    run_checkpointed_sweep(&inline_cfg, None).unwrap();
    let inline_source = dir.path().join("inline-source.h5");
    let inline_before = std::fs::read(&inline_source).unwrap();
    let inline_resume_cfg = checkpoint_config(
        dir.path(),
        "inline-resume",
        "aklt_projector",
        0.2,
        Some(&inline_source),
    );
    run_checkpointed_sweep(&inline_resume_cfg, None).unwrap();
    assert_eq!(std::fs::read(&inline_source).unwrap(), inline_before);
}

#[test]
fn existing_checkpoint_destination_is_configuration_error_and_preserves_inputs() {
    // Mutation caught: deferring destination collision handling to HDF5 creation would change the
    // typed error and could modify a valid restart source or the sentinel destination.
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let source_cfg = checkpoint_config(dir.path(), "source", "aklt_projector", 0.1, None);
    run_checkpointed_sweep(&source_cfg, None).unwrap();
    let source = dir.path().join("source.h5");
    let source_before = std::fs::read(&source).unwrap();

    let value = checkpoint_value(
        dir.path(),
        "collision",
        "aklt_projector",
        2,
        0.2,
        Some(&source),
    );
    let config_path = dir.path().join("collision.json");
    support::write_input(&value, &config_path, "json");
    let config_before = std::fs::read(&config_path).unwrap();
    let cfg = RunConfig::from_json_str(std::str::from_utf8(&config_before).unwrap()).unwrap();
    let checkpoint_path = PathBuf::from(&cfg.checkpoint.as_ref().unwrap().path);
    let output_path = PathBuf::from(&cfg.output.path);
    let sentinel = b"pre-existing destination sentinel";
    std::fs::write(&checkpoint_path, sentinel).unwrap();

    let error = run_checkpointed_sweep(&cfg, Some(&config_path)).unwrap_err();
    assert!(
        matches!(&error, SolveRunError::Configuration(message) if message.contains("destination already exists")),
        "{error}"
    );
    assert_eq!(std::fs::read(&checkpoint_path).unwrap(), sentinel);
    assert_eq!(std::fs::read(&source).unwrap(), source_before);
    assert_eq!(std::fs::read(&config_path).unwrap(), config_before);
    assert!(!output_path.exists());
}
