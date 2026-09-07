#[path = "support/solve_matrix.rs"]
mod support;

use thermal_imps_purification::config::{ModelSpec, RunConfig, TrotterOrder};
use thermal_imps_purification::itebd_auto::{ItebdHamiltonian, ItebdState};
use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions,
};
use thermal_imps_purification::runner::{read_result, run_sweep, Record};
use thermal_imps_purification::solve_run::{run_checkpointed_sweep, SolveRunError};
use serde_json::json;

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

fn assert_record_matches_oracle(actual: &Record, expected: support::OracleRecord) {
    assert_close(actual.u, expected.u);
    assert_close(actual.c, expected.c);
    assert_close(actual.f, expected.f);
    assert_close(actual.magnetization, expected.local);
    assert_eq!(actual.max_bond, expected.max_bond);
}

#[test]
fn fresh_complex_sweep_and_checkpoint_publications_have_the_same_records() {
    // Mutation caught: resolving through the legacy real-only runner rejects this input, and
    // writing a fresh real-only checkpoint rejects the complex state/Hamiltonian pair.
    let mut value = support::phase_config();
    let dir = tempfile::tempdir().unwrap();
    let cfg = RunConfig::from_json_str(&value.to_string()).unwrap();
    let direct = run_sweep(&cfg).unwrap();

    value["output"]["path"] = json!(dir.path().join("run.json"));
    value["checkpoint"] = json!({
        "path": dir.path().join("run.h5"),
        "every_steps": 1
    });
    let saved_cfg = RunConfig::from_json_str(&value.to_string()).unwrap();
    let saved = run_checkpointed_sweep(&saved_cfg, None).unwrap();

    assert_eq!(direct.records.len(), saved.records.len());
    for (actual, expected) in direct.records.iter().zip(&saved.records) {
        assert_records_close(actual, expected);
    }
    let published = read_result(&dir.path().join("run.json")).unwrap();
    assert_eq!(published.records.len(), saved.records.len());
    for (actual, expected) in published.records.iter().zip(&saved.records) {
        assert_records_close(actual, expected);
    }
    assert_eq!(
        serde_json::to_value(&published.metadata.model).unwrap(),
        serde_json::to_value(&saved_cfg.model).unwrap()
    );

    let entries = list_itebd_checkpoints(&dir.path().join("run.h5")).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.completed_steps)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let loaded = load_itebd_checkpoint(
        &dir.path().join("run.h5"),
        entries.last().unwrap().index,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap();
    assert!(matches!(loaded.state, ItebdState::Complex(_)));
    let ItebdHamiltonian::Complex(hamiltonian) = loaded.hamiltonian else {
        panic!("fresh checkpoint lost complex Hamiltonian storage")
    };
    assert_eq!(
        loaded.metadata.hermiticity_tolerance.to_bits(),
        hamiltonian.hermiticity_tolerance().to_bits()
    );
    let stored_model: ModelSpec =
        serde_json::from_str(loaded.metadata.model_label.as_deref().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(stored_model).unwrap(),
        serde_json::to_value(saved_cfg.model).unwrap()
    );
}

#[test]
fn both_orders_match_an_independent_direct_complex_oracle() {
    // Mutation caught: wrong automatic order dispatch, missing canonicalization contribution,
    // or using the configured matrix observable with the wrong phase changes these values.
    for order in [TrotterOrder::First, TrotterOrder::Second] {
        let mut value = support::phase_config();
        value["evolution"]["trotter_order"] = json!(match order {
            TrotterOrder::First => 1,
            TrotterOrder::Second => 2,
        });
        let cfg = RunConfig::from_json_str(&value.to_string()).unwrap();
        let result = run_sweep(&cfg).unwrap();
        assert_eq!(result.records.len(), 1);
        assert_record_matches_oracle(&result.records[0], support::phase_oracle(order));
    }
}

#[test]
fn exact_real_matrix_matches_preset_and_direct_real_api_bitwise() {
    // Mutation caught: routing exactly-real matrix input through a different numerical path
    // changes at least one recorded floating-point value or bond dimension.
    for order in [TrotterOrder::First, TrotterOrder::Second] {
        let order_value = match order {
            TrotterOrder::First => 1,
            TrotterOrder::Second => 2,
        };
        let mut matrix_value = support::real_tfim_config();
        matrix_value["evolution"]["trotter_order"] = json!(order_value);
        let matrix_cfg = RunConfig::from_json_str(&matrix_value.to_string()).unwrap();
        let matrix_result = run_sweep(&matrix_cfg).unwrap();

        let mut preset_value = support::real_tfim_config();
        preset_value["model"] = json!({"type": "tfim", "j": 1.0, "g": 0.7});
        preset_value["evolution"]["trotter_order"] = json!(order_value);
        let preset_cfg = RunConfig::from_json_str(&preset_value.to_string()).unwrap();
        let preset_result = run_sweep(&preset_cfg).unwrap();

        let matrix = &matrix_result.records[0];
        let preset = &preset_result.records[0];
        for (actual, expected) in [
            (matrix.u, preset.u),
            (matrix.c, preset.c),
            (matrix.f, preset.f),
            (matrix.magnetization, preset.magnetization),
        ] {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(matrix.max_bond, preset.max_bond);

        let oracle = support::real_oracle(order);
        for (actual, expected) in [
            (matrix.u, oracle.u),
            (matrix.c, oracle.c),
            (matrix.f, oracle.f),
            (matrix.magnetization, oracle.local),
        ] {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(matrix.max_bond, oracle.max_bond);
    }
}

#[test]
fn real_state_accepts_a_complex_hermitian_observable() {
    // Mutation caught: asking the legacy real-only observable resolver rejects the matrix even
    // though the Hamiltonian correctly selects real state storage.
    let mut value = support::real_tfim_config();
    value["model"]["observable"] = json!({
        "name": "sigma_x_plus_sigma_y",
        "matrix": {
            "real": [[0.0, 1.0], [1.0, 0.0]],
            "imag": [[0.0, -1.0], [1.0, 0.0]]
        }
    });
    let cfg = RunConfig::from_json_str(&value.to_string()).unwrap();
    let resolved = cfg.model.resolve().unwrap();
    assert!(matches!(resolved.hamiltonian, ItebdHamiltonian::Real(_)));
    let actual = run_sweep(&cfg).unwrap();

    let mut preset_value = support::real_tfim_config();
    preset_value["model"] = json!({"type": "tfim", "j": 1.0, "g": 0.7});
    let expected =
        run_sweep(&RunConfig::from_json_str(&preset_value.to_string()).unwrap()).unwrap();
    assert_eq!(
        actual.records[0].magnetization.to_bits(),
        expected.records[0].magnetization.to_bits()
    );
    let ModelSpec::Matrix(model) = &actual.metadata.model else {
        panic!("configured matrix metadata was not retained")
    };
    assert_eq!(model.observable.name, "sigma_x_plus_sigma_y");
    assert!(model.observable.matrix.imag.is_some());
}

#[test]
fn real_checkpoint_restart_uses_the_newly_configured_complex_observable() {
    // Mutation caught: deriving the observable from the loaded real Hamiltonian reports the
    // built-in sigma-x value instead of preserving this segment's configured matrix observable.
    let dir = tempfile::tempdir().unwrap();
    let mut source_value = support::real_tfim_config();
    source_value["model"] = json!({"type": "tfim", "j": 1.0, "g": 0.7});
    source_value["evolution"]["beta_max"] = json!(0.1);
    source_value["evolution"]["record_every_beta"] = json!(0.1);
    source_value["output"]["path"] = json!(dir.path().join("source.json"));
    source_value["checkpoint"] = json!({
        "path": dir.path().join("source.h5"),
        "every_steps": 1
    });
    run_checkpointed_sweep(
        &RunConfig::from_json_str(&source_value.to_string()).unwrap(),
        None,
    )
    .unwrap();

    let mut resumed_value = support::real_tfim_config();
    resumed_value["evolution"]["record_every_beta"] = json!(0.1);
    resumed_value["model"]["observable"] = json!({
        "name": "sigma_y",
        "matrix": {"real": [[0.0, 0.0], [0.0, 0.0]], "imag": [[0.0, -1.0], [1.0, 0.0]]}
    });
    resumed_value["output"]["path"] = json!(dir.path().join("resumed.json"));
    resumed_value["checkpoint"] = json!({
        "path": dir.path().join("resumed.h5"),
        "every_steps": 1
    });
    resumed_value["restart"] = json!({"path": dir.path().join("source.h5")});
    let resumed_cfg = RunConfig::from_json_str(&resumed_value.to_string()).unwrap();
    let resumed = run_checkpointed_sweep(&resumed_cfg, None).unwrap();

    let mut direct_value = resumed_value;
    direct_value.as_object_mut().unwrap().remove("checkpoint");
    direct_value.as_object_mut().unwrap().remove("restart");
    let direct = run_sweep(&RunConfig::from_json_str(&direct_value.to_string()).unwrap()).unwrap();
    assert_eq!(resumed.records.len(), 1);
    assert_records_close(&resumed.records[0], direct.records.last().unwrap());
    assert_close(resumed.records[0].magnetization, 0.0);
    let ModelSpec::Matrix(model) = &resumed.metadata.model else {
        panic!("restart did not retain configured matrix metadata")
    };
    assert_eq!(model.observable.name, "sigma_y");
    assert!(model.observable.matrix.imag.is_some());
}

#[test]
fn generated_nonfinite_builtin_hamiltonian_is_a_configuration_error_before_output() {
    // Mutation caught: blindly converting ModelSpec::resolve errors through `?` exposes this
    // established input failure as SolveRunError::Physics(NonFiniteMatrix).
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("overflow.json");
    let checkpoint = dir.path().join("overflow.h5");
    let value = json!({
        "model": {"type": "bilinear_biquadratic", "j1": f64::MAX, "j2": f64::MAX},
        "evolution": {
            "dtau": 0.05, "beta_max": 0.1, "record_every_beta": 0.1,
            "trotter_order": 2
        },
        "truncation": {"epsilon": 1e-12, "max_bond": 8},
        "output": {"path": &output, "include_exact": false},
        "checkpoint": {"path": &checkpoint, "every_steps": 1}
    });
    let cfg = RunConfig::from_json_str(&value.to_string()).unwrap();

    let error = run_checkpointed_sweep(&cfg, None).unwrap_err();
    assert!(matches!(
        error,
        SolveRunError::Configuration(ref message)
            if message == "model Hamiltonian contains non-finite entries"
    ));
    assert!(!output.exists());
    assert!(!checkpoint.exists());
}
