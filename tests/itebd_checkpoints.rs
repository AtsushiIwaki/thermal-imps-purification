#[path = "support/itebd_checkpoint_rdm.rs"]
mod support;

use thermal_imps_purification::itebd_auto::{ItebdHamiltonian, ItebdState};
use thermal_imps_purification::itebd_checkpoint::*;
use thermal_imps_purification::itebd_complex::{ComplexPurifiedMps, ComplexSite};
use thermal_imps_purification::itebd_state_view::ItebdStateRef;
use thermal_imps_purification::model::Tfim;
use thermal_imps_purification::purified_mps::{infinite_temperature, PurifiedMps, Site};
use thermal_imps_purification::tensor::Tensor;
use num_complex::Complex64;
use std::str::FromStr;

#[test]
fn beta_zero_round_trip_retains_exact_payload_and_roles() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two-site.h5");
    let state = infinite_temperature(2);
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let progress = ItebdCheckpointProgress {
        beta: 0.0,
        completed_steps: 0,
        accumulated_log_norm: 0.0,
        last_step: None,
    };
    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &progress).unwrap();
    writer.finish().unwrap();
    let loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    let ItebdState::Real(restored) = loaded.state else {
        panic!("backend changed")
    };
    assert_eq!(
        state.a.gamma.to_vec::<f64>().unwrap(),
        restored.a.gamma.to_vec::<f64>().unwrap()
    );
    assert_eq!(state.a.gamma.indices, restored.a.gamma.indices);
    assert_eq!(restored.a.right, restored.b.left);
    assert_eq!(restored.a.left, restored.b.right);
    assert_eq!(state.lambda_ab, restored.lambda_ab);
    assert_eq!(state.lambda_ba, restored.lambda_ba);
    assert_eq!(
        loaded.progress.accumulated_log_norm.to_bits(),
        0.0_f64.to_bits()
    );
    assert_real_state_exact(&state, &restored);
    assert_hamiltonian_exact(&h, &loaded.hamiltonian);
    assert_metadata(&support::real_tfim_metadata(), &loaded.metadata);
    assert_eq!(loaded.progress.beta.to_bits(), 0.0_f64.to_bits());
    assert_eq!(loaded.progress.completed_steps, 0);
    assert!(loaded.progress.last_step.is_none());
    assert_eq!(loaded.diagnostics.bond_dimensions, [1, 1]);
    assert_eq!(loaded.diagnostics.schmidt_norms, [1.0, 1.0]);
    assert!(loaded.diagnostics.last_step.is_none());
}

#[test]
fn file_schema_uses_exact_root_and_snapshot_types_and_shapes() {
    use hdf5_metno::types::VarLenUnicode;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &zero_progress()).unwrap();
    writer.finish().unwrap();

    let file = hdf5_metno::File::open(&path).unwrap();
    for name in ["type", "scalar_type", "basis_order"] {
        let attribute = file.attr(name).unwrap();
        assert!(attribute.shape().is_empty(), "{name}");
        assert!(attribute.dtype().unwrap().is::<VarLenUnicode>(), "{name}");
    }
    assert_eq!(
        file.attr("type")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str(),
        "TwoSiteItebdPurificationRun"
    );
    assert_scalar_attribute_type::<u32>(&file, "schema_version");
    assert_scalar_attribute_type::<u32>(&file, "unit_cell_size");
    assert_scalar_attribute_type::<u64>(&file, "physical_dim");
    assert_scalar_attribute_type::<u64>(&file, "ancilla_dim");

    let run = file.group("run").unwrap();
    let metadata = run.dataset("metadata_json").unwrap();
    assert!(metadata.shape().is_empty());
    assert!(metadata.dtype().unwrap().is::<VarLenUnicode>());
    let metadata_json = metadata.read_scalar::<VarLenUnicode>().unwrap();
    let metadata_value: serde_json::Value = serde_json::from_str(metadata_json.as_str()).unwrap();
    assert_eq!(metadata_value["trotter_order"], 2);
    assert_eq!(metadata_value["truncation"]["max_bond"], 64);
    let hamiltonian = run.group("hamiltonian").unwrap();
    for name in ["two_site_h_real", "site_energy_real"] {
        let dataset = hamiltonian.dataset(name).unwrap();
        assert_eq!(dataset.shape(), [16], "{name}");
        assert!(dataset.dtype().unwrap().is::<f64>(), "{name}");
    }

    let snapshot = file.group("states/000000").unwrap();
    assert_scalar_attribute_type::<u8>(&snapshot, "complete");
    assert_scalar_dataset_type::<f64>(&snapshot, "beta");
    assert_scalar_dataset_type::<u64>(&snapshot, "completed_steps");
    assert_scalar_dataset_type::<f64>(&snapshot, "accumulated_log_norm");
    let diagnostics = snapshot.dataset("diagnostics_json").unwrap();
    assert!(diagnostics.shape().is_empty());
    assert!(diagnostics.dtype().unwrap().is::<VarLenUnicode>());
    for name in ["a_role_axes", "b_role_axes"] {
        let dataset = snapshot.dataset(name).unwrap();
        assert_eq!(dataset.shape(), [4], "{name}");
        assert!(dataset.dtype().unwrap().is::<u8>(), "{name}");
    }
    for name in ["lambdaAB", "lambdaBA"] {
        let dataset = snapshot.dataset(name).unwrap();
        assert_eq!(dataset.shape(), [1], "{name}");
        assert!(dataset.dtype().unwrap().is::<f64>(), "{name}");
    }
    snapshot.group("GammaA").unwrap();
    snapshot.group("GammaB").unwrap();
}

#[test]
fn complex_zero_imaginary_payload_remains_complex_and_preserves_component_bits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("complex-zero-imaginary.h5");
    let real = infinite_temperature(2);
    let mut state = support::promote_real_state(&real);
    let mut a_values = state.a.gamma.to_vec::<Complex64>().unwrap();
    a_values[0] = Complex64::new(-0.0, -0.0);
    state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), a_values).unwrap();
    let h = support::phase_tfim();
    let progress = zero_progress();

    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &progress).unwrap();
    writer.finish().unwrap();

    let loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    let ItebdState::Complex(restored) = loaded.state else {
        panic!("complex backend was reclassified")
    };
    assert_complex_state_exact(&state, &restored);
    assert_hamiltonian_exact(&h, &loaded.hamiltonian);
    let file = hdf5_metno::File::open(&path).unwrap();
    let hamiltonian = file.group("run/hamiltonian").unwrap();
    for name in [
        "two_site_h_real",
        "two_site_h_imag",
        "site_energy_real",
        "site_energy_imag",
    ] {
        let dataset = hamiltonian.dataset(name).unwrap();
        assert_eq!(dataset.shape(), [16], "{name}");
        assert!(dataset.dtype().unwrap().is::<f64>(), "{name}");
    }
}

#[test]
fn finite_complex_snapshot_preserves_permuted_axes_unequal_bonds_and_lists_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("finite-complex.h5");
    let mut state = support::promote_real_state(&infinite_temperature(2));
    expand_ab_bond_complex(&mut state);
    state.lambda_ab = vec![0.8, 0.6];
    let a_values = (0..8)
        .map(|value| {
            let value = value as f64;
            Complex64::new(if value == 0.0 { -0.0 } else { value / 7.0 }, value - 2.5)
        })
        .collect();
    let b_values = (0..8)
        .map(|value| {
            let value = value as f64;
            Complex64::new(-value / 9.0, if value == 0.0 { -0.0 } else { value + 0.25 })
        })
        .collect();
    state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), a_values).unwrap();
    state.b.gamma = Tensor::from_dense(state.b.gamma.indices.clone(), b_values).unwrap();
    permute_complex(&mut state.a.gamma, [2, 0, 3, 1]);
    permute_complex(&mut state.b.gamma, [3, 1, 0, 2]);
    let h = support::phase_tfim();
    let progress = ItebdCheckpointProgress {
        beta: 0.1,
        completed_steps: 1,
        accumulated_log_norm: -0.375,
        last_step: Some(thermal_imps_purification::itebd::StepInfo {
            max_bond: 2,
            min_singular_value: 0.125,
            log_norm: -0.375,
        }),
    };

    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    let entry = writer.append((&state).into(), &progress).unwrap();
    writer.finish().unwrap();
    assert_eq!(entry.index, 0);
    assert_eq!(entry.beta.to_bits(), 0.1_f64.to_bits());
    assert_eq!(entry.completed_steps, 1);

    let entries = list_itebd_checkpoints(&path).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].index, 0);
    assert_eq!(entries[0].completed_steps, 1);
    for _ in 0..2 {
        let loaded =
            load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
        let ItebdState::Complex(restored) = loaded.state else {
            panic!("complex backend changed")
        };
        assert_complex_state_exact(&state, &restored);
        assert_hamiltonian_exact(&h, &loaded.hamiltonian);
        assert_eq!(loaded.progress.beta.to_bits(), progress.beta.to_bits());
        assert_eq!(loaded.progress.completed_steps, 1);
        assert_eq!(
            loaded.progress.accumulated_log_norm.to_bits(),
            progress.accumulated_log_norm.to_bits()
        );
        let step = loaded.progress.last_step.unwrap();
        assert_eq!(step.max_bond, 2);
        assert_eq!(step.min_singular_value.to_bits(), 0.125_f64.to_bits());
        assert_eq!(step.log_norm.to_bits(), (-0.375_f64).to_bits());
        assert_eq!(loaded.diagnostics.bond_dimensions, [2, 1]);
        assert_eq!(loaded.diagnostics.schmidt_norms, [1.0, 1.0]);
    }
}

#[test]
fn create_is_exclusive_and_rejects_invalid_metadata_before_creating_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let existing = dir.path().join("existing.h5");
    std::fs::write(&existing, b"keep").unwrap();
    assert!(ItebdTrajectoryWriter::create(&existing, support::real_tfim_metadata(), &h).is_err());
    assert_eq!(std::fs::read(&existing).unwrap(), b"keep");

    let invalid = [
        (f64::NAN, 1e-13, Some(64), 3, None, 1e-12),
        (0.0, 1e-13, Some(64), 3, None, 1e-12),
        (0.05, 0.0, Some(64), 3, None, 1e-12),
        (0.05, 1e-13, Some(0), 3, None, 1e-12),
        (0.05, 1e-13, Some(64), 0, None, 1e-12),
        (0.05, 1e-13, Some(64), 3, Some(0.0), 1e-12),
        (0.05, 1e-13, Some(64), 3, None, -1.0),
    ];
    for (case, (dtau, epsilon, max_bond, cadence, interval, tolerance)) in
        invalid.into_iter().enumerate()
    {
        let path = dir.path().join(format!("invalid-{case}.h5"));
        let mut metadata = support::real_tfim_metadata();
        metadata.dtau = dtau;
        metadata.truncation.epsilon = epsilon;
        metadata.truncation.max_bond = max_bond;
        metadata.canonicalize_every = cadence;
        metadata.record_every_beta = interval;
        metadata.hermiticity_tolerance = tolerance;
        assert!(ItebdTrajectoryWriter::create(&path, metadata, &h).is_err());
        assert!(!path.exists(), "invalid metadata created {path:?}");
    }
}

#[test]
fn create_rejects_invalid_hamiltonian_before_creating_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        thermal_imps_purification::model::LocalHamiltonian {
            two_site_h: nalgebra::DMatrix::zeros(3, 3),
            site_energy: nalgebra::DMatrix::zeros(3, 3),
        },
        thermal_imps_purification::model::LocalHamiltonian {
            two_site_h: nalgebra::DMatrix::from_row_slice(
                4,
                4,
                &[
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            site_energy: nalgebra::DMatrix::zeros(4, 4),
        },
    ];
    for (case, h) in cases.into_iter().enumerate() {
        let path = dir.path().join(format!("bad-hamiltonian-{case}.h5"));
        assert!(ItebdTrajectoryWriter::create(
            &path,
            support::real_tfim_metadata(),
            &ItebdHamiltonian::Real(h),
        )
        .is_err());
        assert!(!path.exists());
    }
}

#[test]
fn real_hamiltonian_uses_the_scaled_frobenius_hermiticity_tolerance() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("near-symmetric.h5");
    let mut matrix = nalgebra::DMatrix::<f64>::identity(4, 4);
    matrix[(0, 1)] = 1e-6;
    let h = ItebdHamiltonian::Real(thermal_imps_purification::model::LocalHamiltonian {
        two_site_h: matrix.clone(),
        site_energy: matrix,
    });
    let mut metadata = support::real_tfim_metadata();
    metadata.hermiticity_tolerance = 8e-7;

    ItebdTrajectoryWriter::create(&path, metadata, &h)
        .unwrap()
        .finish()
        .unwrap();
}

#[test]
fn append_rejects_invalid_progress_backend_and_cap_without_consuming_an_index() {
    let dir = tempfile::tempdir().unwrap();
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());

    for (case, progress) in [
        ItebdCheckpointProgress {
            beta: 0.2,
            completed_steps: 1,
            accumulated_log_norm: 0.0,
            last_step: None,
        },
        ItebdCheckpointProgress {
            beta: 0.0,
            completed_steps: 0,
            accumulated_log_norm: 1.0,
            last_step: None,
        },
        ItebdCheckpointProgress {
            beta: 0.0,
            completed_steps: 0,
            accumulated_log_norm: 0.0,
            last_step: Some(thermal_imps_purification::itebd::StepInfo {
                max_bond: 1,
                min_singular_value: 1.0,
                log_norm: 0.0,
            }),
        },
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("bad-progress-{case}.h5"));
        let mut writer =
            ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
        assert!(writer
            .append(ItebdStateRef::from(&infinite_temperature(2)), &progress)
            .is_err());
        let entry = writer
            .append(
                ItebdStateRef::from(&infinite_temperature(2)),
                &zero_progress(),
            )
            .unwrap();
        assert_eq!(entry.index, 0);
    }

    let mismatch_path = dir.path().join("backend-mismatch.h5");
    let mut writer =
        ItebdTrajectoryWriter::create(&mismatch_path, support::real_tfim_metadata(), &h).unwrap();
    let complex = support::promote_real_state(&infinite_temperature(2));
    assert!(writer.append((&complex).into(), &zero_progress()).is_err());

    let cap_path = dir.path().join("cap.h5");
    let mut metadata = support::real_tfim_metadata();
    metadata.truncation.max_bond = Some(1);
    let mut writer = ItebdTrajectoryWriter::create(&cap_path, metadata, &h).unwrap();
    let mut state = infinite_temperature(2);
    expand_ab_bond_real(&mut state);
    state.lambda_ab = vec![0.8, 0.6];
    assert!(writer.append((&state).into(), &zero_progress()).is_err());
}

#[test]
fn later_snapshots_require_strictly_increasing_steps_and_beta() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("monotonic.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &zero_progress()).unwrap();
    assert!(writer.append((&state).into(), &zero_progress()).is_err());
    let finite = ItebdCheckpointProgress {
        beta: 0.1,
        completed_steps: 1,
        accumulated_log_norm: -0.1,
        last_step: None,
    };
    let entry = writer.append((&state).into(), &finite).unwrap();
    assert_eq!(entry.index, 1);
    let entries = list_itebd_checkpoints(&path).unwrap();
    assert_eq!(
        entries.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn empty_trajectory_still_rejects_different_root_physical_and_ancilla_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root-dimensions.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h)
        .unwrap()
        .finish()
        .unwrap();
    let file = hdf5_metno::File::open_rw(&path).unwrap();
    file.attr("ancilla_dim")
        .unwrap()
        .write_scalar(&3_u64)
        .unwrap();
    file.flush().unwrap();
    drop(file);

    assert!(matches!(
        list_itebd_checkpoints(&path),
        Err(ItebdCheckpointError::Schema { field, .. }) if field == "ancilla_dim"
    ));
}

#[test]
fn invalid_complete_marker_is_not_reported_as_an_incomplete_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid-marker.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &zero_progress()).unwrap();
    writer.finish().unwrap();
    let file = hdf5_metno::File::open_rw(&path).unwrap();
    file.group("states/000000")
        .unwrap()
        .attr("complete")
        .unwrap()
        .write_scalar(&2_u8)
        .unwrap();
    file.flush().unwrap();
    drop(file);

    assert!(matches!(
        load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::Snapshot { field, .. }) if field == "complete"
    ));
    assert!(matches!(
        list_itebd_checkpoints(&path),
        Err(ItebdCheckpointError::Snapshot { field, .. }) if field == "complete"
    ));
}

#[test]
fn incomplete_or_missing_complete_marker_is_hidden_from_listing_and_rejected_by_load() {
    for (case, remove_marker) in [("zero", false), ("missing", true)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("incomplete-{case}.h5"));
        write_real_zero_checkpoint(&path);
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        let snapshot = file.group("states/000000").unwrap();
        if remove_marker {
            snapshot.delete_attr("complete").unwrap();
        } else {
            snapshot
                .attr("complete")
                .unwrap()
                .write_scalar(&0_u8)
                .unwrap();
        }
        file.flush().unwrap();
        drop(snapshot);
        drop(file);

        assert!(list_itebd_checkpoints(&path).unwrap().is_empty());
        assert!(matches!(
            load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()),
            Err(ItebdCheckpointError::Incomplete { path: error_path, index: 0 })
                if error_path == path
        ));
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("malformed-complete.h5");
    write_real_zero_checkpoint(&path);
    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let snapshot = file.group("states/000000").unwrap();
    snapshot.delete_attr("complete").unwrap();
    snapshot
        .new_attr::<u16>()
        .shape(())
        .create("complete")
        .unwrap()
        .write_scalar(&1_u16)
        .unwrap();
    file.flush().unwrap();
    drop(snapshot);
    drop(file);
    assert_snapshot_field(&path, 0, "complete");
}

#[test]
fn root_discriminators_and_authoritative_hamiltonian_are_strictly_validated() {
    use hdf5_metno::types::VarLenUnicode;

    for (case, attribute, value, expected_field) in [
        ("type", "type", "UnknownRun", "type"),
        ("backend", "scalar_type", "other", "scalar_type"),
        ("basis", "basis_order", "reversed", "basis_order"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("bad-{case}.h5"));
        write_real_zero_checkpoint(&path);
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        file.attr(attribute)
            .unwrap()
            .write_scalar(&VarLenUnicode::from_str(value).unwrap())
            .unwrap();
        file.flush().unwrap();
        drop(file);
        assert_schema_field(&path, expected_field);
    }

    for (case, attribute, value, expected_field) in [
        ("version", "schema_version", 2_u32, "schema_version"),
        ("unit-cell", "unit_cell_size", 3_u32, "unit_cell_size"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("bad-{case}.h5"));
        write_real_zero_checkpoint(&path);
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        file.attr(attribute).unwrap().write_scalar(&value).unwrap();
        file.flush().unwrap();
        drop(file);
        assert_schema_field(&path, expected_field);
    }

    let dir = tempfile::tempdir().unwrap();
    let missing_path = dir.path().join("missing-two-site-h.h5");
    write_real_zero_checkpoint(&missing_path);
    let file = hdf5_metno::File::open_rw(&missing_path).unwrap();
    file.group("run/hamiltonian")
        .unwrap()
        .unlink("two_site_h_real")
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_schema_field(&missing_path, "run/hamiltonian/two_site_h_real");

    let malformed_path = dir.path().join("malformed-site-energy.h5");
    write_real_zero_checkpoint(&malformed_path);
    let file = hdf5_metno::File::open_rw(&malformed_path).unwrap();
    let group = file.group("run/hamiltonian").unwrap();
    group.unlink("site_energy_real").unwrap();
    group
        .new_dataset::<f64>()
        .shape(15)
        .create("site_energy_real")
        .unwrap()
        .write_raw(&[0.0; 15])
        .unwrap();
    file.flush().unwrap();
    drop(group);
    drop(file);
    assert_schema_field(&malformed_path, "run/hamiltonian/site_energy_real");

    let wrong_backend_path = dir.path().join("wrong-recognized-backend.h5");
    write_real_zero_checkpoint(&wrong_backend_path);
    let file = hdf5_metno::File::open_rw(&wrong_backend_path).unwrap();
    file.attr("scalar_type")
        .unwrap()
        .write_scalar(&VarLenUnicode::from_str("complex").unwrap())
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_schema_field(&wrong_backend_path, "run/hamiltonian/two_site_h_imag");
}

#[test]
fn snapshot_requires_gamma_lambda_numeric_shape_and_role_schema() {
    let dir = tempfile::tempdir().unwrap();

    let missing_gamma = dir.path().join("missing-gamma.h5");
    write_real_zero_checkpoint(&missing_gamma);
    let file = hdf5_metno::File::open_rw(&missing_gamma).unwrap();
    file.group("states/000000")
        .unwrap()
        .unlink("GammaB")
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_snapshot_field(&missing_gamma, 0, "GammaB");

    let missing_lambda = dir.path().join("missing-lambda.h5");
    write_real_zero_checkpoint(&missing_lambda);
    let file = hdf5_metno::File::open_rw(&missing_lambda).unwrap();
    file.group("states/000000")
        .unwrap()
        .unlink("lambdaAB")
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_snapshot_field(&missing_lambda, 0, "lambdaAB");

    let wrong_lambda_type = dir.path().join("wrong-lambda-type.h5");
    write_real_zero_checkpoint(&wrong_lambda_type);
    let file = hdf5_metno::File::open_rw(&wrong_lambda_type).unwrap();
    let snapshot = file.group("states/000000").unwrap();
    snapshot.unlink("lambdaBA").unwrap();
    snapshot
        .new_dataset::<f32>()
        .shape(1)
        .create("lambdaBA")
        .unwrap()
        .write_raw(&[1.0_f32])
        .unwrap();
    file.flush().unwrap();
    drop(snapshot);
    drop(file);
    assert_snapshot_field(&wrong_lambda_type, 0, "lambdaBA");

    let wrong_lambda_shape = dir.path().join("wrong-lambda-shape.h5");
    write_real_zero_checkpoint(&wrong_lambda_shape);
    let file = hdf5_metno::File::open_rw(&wrong_lambda_shape).unwrap();
    let snapshot = file.group("states/000000").unwrap();
    snapshot.unlink("lambdaAB").unwrap();
    snapshot
        .new_dataset::<f64>()
        .shape(2)
        .create("lambdaAB")
        .unwrap()
        .write_raw(&[1.0_f64, 0.0])
        .unwrap();
    file.flush().unwrap();
    drop(snapshot);
    drop(file);
    assert_snapshot_field(&wrong_lambda_shape, 0, "lambdaAB");

    let bad_roles = dir.path().join("bad-role-permutation.h5");
    write_real_zero_checkpoint(&bad_roles);
    let file = hdf5_metno::File::open_rw(&bad_roles).unwrap();
    file.group("states/000000")
        .unwrap()
        .dataset("a_role_axes")
        .unwrap()
        .write_raw(&[0_u8, 0, 2, 3])
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_snapshot_field(&bad_roles, 0, "a_role_axes");
}

#[test]
fn snapshot_rejects_mismatched_links_nonfinite_values_and_diagnostic_disagreement() {
    let dir = tempfile::tempdir().unwrap();

    let mismatched_id = dir.path().join("mismatched-link-id.h5");
    write_real_zero_checkpoint(&mismatched_id);
    let file = hdf5_metno::File::open_rw(&mismatched_id).unwrap();
    let snapshot = file.group("states/000000").unwrap();
    let b_axes = snapshot
        .dataset("b_role_axes")
        .unwrap()
        .read_raw::<u8>()
        .unwrap();
    let b_left = b_axes[0] as usize + 1;
    snapshot
        .group(&format!("GammaB/inds/index_{b_left}"))
        .unwrap()
        .dataset("id")
        .unwrap()
        .write_scalar(&u64::MAX)
        .unwrap();
    file.flush().unwrap();
    drop(snapshot);
    drop(file);
    assert_state_field(&mismatched_id, 0, "a.right/b.left");

    let mismatched_dimension = dir.path().join("mismatched-link-dimension.h5");
    write_real_zero_checkpoint(&mismatched_dimension);
    let file = hdf5_metno::File::open_rw(&mismatched_dimension).unwrap();
    let snapshot = file.group("states/000000").unwrap();
    let b_axes = snapshot
        .dataset("b_role_axes")
        .unwrap()
        .read_raw::<u8>()
        .unwrap();
    let b_left = b_axes[0] as usize + 1;
    snapshot
        .group(&format!("GammaB/inds/index_{b_left}"))
        .unwrap()
        .dataset("dim")
        .unwrap()
        .write_scalar(&2_i64)
        .unwrap();
    let storage = snapshot.group("GammaB/storage").unwrap();
    storage.unlink("data").unwrap();
    storage
        .new_dataset::<f64>()
        .shape(8)
        .create("data")
        .unwrap()
        .write_raw(&[1.0_f64; 8])
        .unwrap();
    file.flush().unwrap();
    drop(storage);
    drop(snapshot);
    drop(file);
    assert_state_field(&mismatched_dimension, 0, "a.right/b.left");

    let nonfinite_gamma = dir.path().join("nonfinite-gamma.h5");
    write_real_zero_checkpoint(&nonfinite_gamma);
    let file = hdf5_metno::File::open_rw(&nonfinite_gamma).unwrap();
    let data = file
        .group("states/000000/GammaA/storage")
        .unwrap()
        .dataset("data")
        .unwrap();
    let mut values = data.read_raw::<f64>().unwrap();
    values[0] = f64::NAN;
    data.write_raw(&values).unwrap();
    file.flush().unwrap();
    drop(data);
    drop(file);
    assert_state_field(&nonfinite_gamma, 0, "a.gamma");

    let nonfinite_lambda = dir.path().join("nonfinite-lambda.h5");
    write_real_zero_checkpoint(&nonfinite_lambda);
    let file = hdf5_metno::File::open_rw(&nonfinite_lambda).unwrap();
    file.group("states/000000")
        .unwrap()
        .dataset("lambdaBA")
        .unwrap()
        .write_raw(&[f64::INFINITY])
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_state_field(&nonfinite_lambda, 0, "lambda_ba[0]");

    let nonfinite_norm = dir.path().join("nonfinite-accumulated-norm.h5");
    write_real_zero_checkpoint(&nonfinite_norm);
    let file = hdf5_metno::File::open_rw(&nonfinite_norm).unwrap();
    file.group("states/000000")
        .unwrap()
        .dataset("accumulated_log_norm")
        .unwrap()
        .write_scalar(&f64::NEG_INFINITY)
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_snapshot_field(&nonfinite_norm, 0, "accumulated_log_norm");

    let nonfinite_beta = dir.path().join("nonfinite-beta.h5");
    write_real_zero_checkpoint(&nonfinite_beta);
    let file = hdf5_metno::File::open_rw(&nonfinite_beta).unwrap();
    file.group("states/000000")
        .unwrap()
        .dataset("beta")
        .unwrap()
        .write_scalar(&f64::NAN)
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_snapshot_field(&nonfinite_beta, 0, "beta");

    let diagnostics = dir.path().join("diagnostics-disagree.h5");
    write_real_zero_checkpoint(&diagnostics);
    rewrite_diagnostics(
        &diagnostics,
        r#"{"bond_dimensions":[1,1],"schmidt_norms":[2.0,1.0],"last_step":null}"#,
    );
    assert_snapshot_field(&diagnostics, 0, "diagnostics_json.schmidt_norms[0]");

    let beta_steps = dir.path().join("beta-step-disagree.h5");
    write_real_zero_checkpoint(&beta_steps);
    let file = hdf5_metno::File::open_rw(&beta_steps).unwrap();
    file.group("states/000000")
        .unwrap()
        .dataset("completed_steps")
        .unwrap()
        .write_scalar(&1_u64)
        .unwrap();
    file.flush().unwrap();
    drop(file);
    assert_snapshot_field(&beta_steps, 0, "beta");
}

#[test]
fn step_zero_preserves_signed_zero_progress_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("signed-zero-progress.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let progress = ItebdCheckpointProgress {
        beta: -0.0,
        completed_steps: 0,
        accumulated_log_norm: -0.0,
        last_step: None,
    };
    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &progress).unwrap();
    writer.finish().unwrap();

    let loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    assert_eq!(loaded.progress.beta.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        loaded.progress.accumulated_log_norm.to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn listing_revalidates_strict_progress_order_across_complete_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt-order.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let mut writer =
        ItebdTrajectoryWriter::create(&path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &zero_progress()).unwrap();
    writer
        .append(
            (&state).into(),
            &ItebdCheckpointProgress {
                beta: 0.1,
                completed_steps: 1,
                accumulated_log_norm: -0.25,
                last_step: None,
            },
        )
        .unwrap();
    writer.finish().unwrap();

    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let second = file.group("states/000001").unwrap();
    second
        .dataset("beta")
        .unwrap()
        .write_scalar(&0.0_f64)
        .unwrap();
    second
        .dataset("completed_steps")
        .unwrap()
        .write_scalar(&0_u64)
        .unwrap();
    second
        .dataset("accumulated_log_norm")
        .unwrap()
        .write_scalar(&0.0_f64)
        .unwrap();
    file.flush().unwrap();
    drop(second);
    drop(file);

    assert!(matches!(
        list_itebd_checkpoints(&path),
        Err(ItebdCheckpointError::Snapshot { index: 1, field, .. })
            if field == "completed_steps"
    ));
}

#[test]
fn load_rejects_gamma_payload_numeric_coercion_before_tensor_reconstruction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gamma-f32-data.h5");
    write_real_zero_checkpoint(&path);
    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let storage = file.group("states/000000/GammaA/storage").unwrap();
    let values = storage.dataset("data").unwrap().read_raw::<f64>().unwrap();
    storage.unlink("data").unwrap();
    storage
        .new_dataset::<f32>()
        .shape(values.len())
        .create("data")
        .unwrap()
        .write_raw(
            &values
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    file.flush().unwrap();
    drop(storage);
    drop(file);

    assert!(matches!(
        load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::Snapshot { field, .. })
            if field == "GammaA/storage/data"
    ));
}

#[test]
fn load_rejects_gamma_index_numeric_coercion_before_tensor_reconstruction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gamma-i32-index-dim.h5");
    write_real_zero_checkpoint(&path);
    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let index = file.group("states/000000/GammaB/inds/index_1").unwrap();
    let dimension = index.dataset("dim").unwrap().read_scalar::<i64>().unwrap();
    index.unlink("dim").unwrap();
    index
        .new_dataset::<i32>()
        .shape(())
        .create("dim")
        .unwrap()
        .write_scalar(&(dimension as i32))
        .unwrap();
    file.flush().unwrap();
    drop(index);
    drop(file);

    assert!(matches!(
        load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::Snapshot { field, .. })
            if field == "GammaB/inds/index_1/dim"
    ));
}

#[test]
fn load_requires_exact_gamma_storage_identity_and_version() {
    use hdf5_metno::types::VarLenUnicode;

    let dir = tempfile::tempdir().unwrap();
    let type_path = dir.path().join("gamma-storage-type.h5");
    write_real_zero_checkpoint(&type_path);
    let file = hdf5_metno::File::open_rw(&type_path).unwrap();
    let storage = file.group("states/000000/GammaA/storage").unwrap();
    storage
        .attr("type")
        .unwrap()
        .write_scalar(&VarLenUnicode::from_str("prefix Dense{Float64} suffix").unwrap())
        .unwrap();
    file.flush().unwrap();
    drop(storage);
    drop(file);
    assert!(matches!(
        load_itebd_checkpoint(&type_path, 0, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::Snapshot { field, .. })
            if field == "GammaA/storage/type"
    ));

    let version_path = dir.path().join("gamma-storage-version.h5");
    write_real_zero_checkpoint(&version_path);
    let file = hdf5_metno::File::open_rw(&version_path).unwrap();
    let storage = file.group("states/000000/GammaA/storage").unwrap();
    storage
        .attr("version")
        .unwrap()
        .write_scalar(&2_i64)
        .unwrap();
    file.flush().unwrap();
    drop(storage);
    drop(file);
    assert!(matches!(
        load_itebd_checkpoint(&version_path, 0, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::Snapshot { field, .. })
            if field == "GammaA/storage/version"
    ));
}

fn write_real_zero_checkpoint(path: &std::path::Path) {
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let mut writer =
        ItebdTrajectoryWriter::create(path, support::real_tfim_metadata(), &h).unwrap();
    writer.append((&state).into(), &zero_progress()).unwrap();
    writer.finish().unwrap();
}

fn rewrite_diagnostics(path: &std::path::Path, json: &str) {
    use hdf5_metno::types::VarLenUnicode;

    let file = hdf5_metno::File::open_rw(path).unwrap();
    file.group("states/000000")
        .unwrap()
        .dataset("diagnostics_json")
        .unwrap()
        .write_scalar(&VarLenUnicode::from_str(json).unwrap())
        .unwrap();
    file.flush().unwrap();
}

fn assert_schema_field(path: &std::path::Path, expected_field: &str) {
    assert!(matches!(
        list_itebd_checkpoints(path),
        Err(ItebdCheckpointError::Schema { path: error_path, field, .. })
            if error_path == path && field == expected_field
    ));
}

fn assert_snapshot_field(path: &std::path::Path, expected_index: u64, expected_field: &str) {
    assert!(matches!(
        load_itebd_checkpoint(path, expected_index, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::Snapshot { path: error_path, index, field, .. })
            if error_path == path && index == expected_index && field == expected_field
    ));
}

fn assert_state_field(path: &std::path::Path, expected_index: u64, expected_field: &str) {
    assert!(matches!(
        load_itebd_checkpoint(path, expected_index, &ItebdCheckpointLoadOptions::default()),
        Err(ItebdCheckpointError::State {
            path: error_path,
            index,
            source: thermal_imps_purification::itebd_state_view::ItebdStateValidationError::InvalidField {
                field,
                ..
            },
        }) if error_path == path && index == expected_index && field == expected_field
    ));
}

fn zero_progress() -> ItebdCheckpointProgress {
    ItebdCheckpointProgress {
        beta: 0.0,
        completed_steps: 0,
        accumulated_log_norm: 0.0,
        last_step: None,
    }
}

fn assert_metadata(expected: &ItebdRunMetadata, actual: &ItebdRunMetadata) {
    assert_eq!(actual.dtau.to_bits(), expected.dtau.to_bits());
    assert_eq!(actual.trotter_order, expected.trotter_order);
    assert_eq!(
        actual.truncation.epsilon.to_bits(),
        expected.truncation.epsilon.to_bits()
    );
    assert_eq!(actual.truncation.max_bond, expected.truncation.max_bond);
    assert_eq!(actual.canonicalize_every, expected.canonicalize_every);
    assert_eq!(actual.record_every_beta, expected.record_every_beta);
    assert_eq!(actual.model_label, expected.model_label);
    assert_eq!(actual.git_revision, expected.git_revision);
    assert_eq!(
        actual.hermiticity_tolerance.to_bits(),
        expected.hermiticity_tolerance.to_bits()
    );
}

fn assert_real_state_exact(expected: &PurifiedMps, actual: &PurifiedMps) {
    assert_real_site_exact(&expected.a, &actual.a);
    assert_real_site_exact(&expected.b, &actual.b);
    assert_eq!(actual.lambda_ab, expected.lambda_ab);
    assert_eq!(actual.lambda_bond_ab, expected.lambda_bond_ab);
    assert_eq!(actual.lambda_ba, expected.lambda_ba);
    assert_eq!(actual.lambda_bond_ba, expected.lambda_bond_ba);
}

fn assert_real_site_exact(expected: &Site, actual: &Site) {
    assert_eq!(actual.gamma.indices, expected.gamma.indices);
    assert_eq!(
        actual.gamma.to_vec::<f64>().unwrap(),
        expected.gamma.to_vec::<f64>().unwrap()
    );
    assert_eq!(actual.left, expected.left);
    assert_eq!(actual.phys, expected.phys);
    assert_eq!(actual.anc, expected.anc);
    assert_eq!(actual.right, expected.right);
}

fn assert_complex_state_exact(expected: &ComplexPurifiedMps, actual: &ComplexPurifiedMps) {
    assert_complex_site_exact(&expected.a, &actual.a);
    assert_complex_site_exact(&expected.b, &actual.b);
    assert_eq!(actual.lambda_ab, expected.lambda_ab);
    assert_eq!(actual.lambda_bond_ab, expected.lambda_bond_ab);
    assert_eq!(actual.lambda_ba, expected.lambda_ba);
    assert_eq!(actual.lambda_bond_ba, expected.lambda_bond_ba);
}

fn assert_complex_site_exact(expected: &ComplexSite, actual: &ComplexSite) {
    assert_eq!(actual.gamma.indices, expected.gamma.indices);
    let expected_values = expected.gamma.to_vec::<Complex64>().unwrap();
    let actual_values = actual.gamma.to_vec::<Complex64>().unwrap();
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
    assert_eq!(actual.left, expected.left);
    assert_eq!(actual.phys, expected.phys);
    assert_eq!(actual.anc, expected.anc);
    assert_eq!(actual.right, expected.right);
}

fn assert_hamiltonian_exact(expected: &ItebdHamiltonian, actual: &ItebdHamiltonian) {
    match (expected, actual) {
        (ItebdHamiltonian::Real(expected), ItebdHamiltonian::Real(actual)) => {
            assert_real_matrix_bits(&expected.two_site_h, &actual.two_site_h);
            assert_real_matrix_bits(&expected.site_energy, &actual.site_energy);
        }
        (ItebdHamiltonian::Complex(expected), ItebdHamiltonian::Complex(actual)) => {
            assert_complex_matrix_bits(expected.two_site_h(), actual.two_site_h());
            assert_complex_matrix_bits(expected.site_energy(), actual.site_energy());
            assert_eq!(
                actual.hermiticity_tolerance().to_bits(),
                ItebdCheckpointLoadOptions::default()
                    .hermiticity_tolerance
                    .to_bits()
            );
        }
        _ => panic!("Hamiltonian backend changed"),
    }
}

fn assert_real_matrix_bits(expected: &nalgebra::DMatrix<f64>, actual: &nalgebra::DMatrix<f64>) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

fn assert_complex_matrix_bits(
    expected: &nalgebra::DMatrix<Complex64>,
    actual: &nalgebra::DMatrix<Complex64>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

fn assert_scalar_attribute_type<T: hdf5_metno::H5Type>(
    location: &hdf5_metno::Location,
    name: &str,
) {
    let attribute = location.attr(name).unwrap();
    assert!(attribute.shape().is_empty(), "{name}");
    assert!(attribute.dtype().unwrap().is::<T>(), "{name}");
}

fn assert_scalar_dataset_type<T: hdf5_metno::H5Type>(group: &hdf5_metno::Group, name: &str) {
    let dataset = group.dataset(name).unwrap();
    assert!(dataset.shape().is_empty(), "{name}");
    assert!(dataset.dtype().unwrap().is::<T>(), "{name}");
}

fn expand_ab_bond_real(state: &mut PurifiedMps) {
    state.lambda_bond_ab.dim = 2;
    state.a.right.dim = 2;
    state.b.left.dim = 2;
    let a_axis = state
        .a
        .gamma
        .indices
        .iter()
        .position(|index| index.id == state.a.right.id)
        .unwrap();
    let b_axis = state
        .b
        .gamma
        .indices
        .iter()
        .position(|index| index.id == state.b.left.id)
        .unwrap();
    state.a.gamma.indices[a_axis].dim = 2;
    state.b.gamma.indices[b_axis].dim = 2;
    state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), vec![1.0; 8]).unwrap();
    state.b.gamma = Tensor::from_dense(state.b.gamma.indices.clone(), vec![1.0; 8]).unwrap();
}

fn expand_ab_bond_complex(state: &mut ComplexPurifiedMps) {
    state.lambda_bond_ab.dim = 2;
    state.a.right.dim = 2;
    state.b.left.dim = 2;
    let a_axis = state
        .a
        .gamma
        .indices
        .iter()
        .position(|index| index.id == state.a.right.id)
        .unwrap();
    let b_axis = state
        .b
        .gamma
        .indices
        .iter()
        .position(|index| index.id == state.b.left.id)
        .unwrap();
    state.a.gamma.indices[a_axis].dim = 2;
    state.b.gamma.indices[b_axis].dim = 2;
}

fn permute_complex(tensor: &mut Tensor, order: [usize; 4]) {
    let old_indices = tensor.indices.clone();
    let old_values = tensor.to_vec::<Complex64>().unwrap();
    let old_dims: Vec<_> = old_indices.iter().map(|index| index.dim).collect();
    let new_dims = order.map(|old_axis| old_dims[old_axis]);
    let new_values = (0..new_dims.iter().product())
        .map(|new_offset| {
            let mut remainder = new_offset;
            let mut old_coordinates = [0_usize; 4];
            for (new_axis, dimension) in new_dims.into_iter().enumerate() {
                old_coordinates[order[new_axis]] = remainder % dimension;
                remainder /= dimension;
            }
            let mut old_offset = 0;
            let mut stride = 1;
            for (coordinate, dimension) in old_coordinates.into_iter().zip(old_dims.iter()) {
                old_offset += coordinate * stride;
                stride *= dimension;
            }
            old_values[old_offset]
        })
        .collect();
    *tensor = Tensor::from_dense(
        order.map(|old_axis| old_indices[old_axis].clone()).to_vec(),
        new_values,
    )
    .unwrap();
}

/// This decimal's default JSON parser previously lost one ULP in persisted step diagnostics.
#[test]
fn json_checkpoint_step_log_norm_preserves_exact_bits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("float-bits.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let metadata = support::real_tfim_metadata();
    let value = f64::from_bits(4595195715801308216);
    let progress = ItebdCheckpointProgress {
        beta: 2.0 * metadata.dtau,
        completed_steps: 1,
        accumulated_log_norm: value,
        last_step: Some(thermal_imps_purification::itebd::StepInfo {
            max_bond: 1, min_singular_value: 1.0, log_norm: value,
        }),
    };
    let mut writer = ItebdTrajectoryWriter::create(&path, metadata, &h).unwrap();
    writer.append((&state).into(), &progress).unwrap();
    writer.finish().unwrap();
    let loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    assert_eq!(loaded.progress.last_step.unwrap().log_norm.to_bits(), 4595195715801308216);
    assert_eq!(loaded.diagnostics.last_step.unwrap().log_norm.to_bits(), 4595195715801308216);
}

#[test]
fn json_checkpoint_metadata_preserves_exact_float_bits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata-float-bits.h5");
    let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
    let state = infinite_temperature(2);
    let mut metadata = support::real_tfim_metadata();
    metadata.record_every_beta = Some(f64::from_bits(4595195715801308216));
    let mut writer = ItebdTrajectoryWriter::create(&path, metadata, &h).unwrap();
    writer.append((&state).into(), &zero_progress()).unwrap();
    writer.finish().unwrap();
    let loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
    assert_eq!(loaded.metadata.record_every_beta.unwrap().to_bits(), 4595195715801308216);
}
