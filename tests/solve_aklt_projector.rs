#[path = "support/aklt_projector_solve.rs"]
mod support;

use serde_json::json;
use std::path::Path;
use thermal_imps_purification::config::{ModelSpec, RunConfig, TrotterOrder};
use thermal_imps_purification::itebd_auto::ItebdHamiltonian;
use thermal_imps_purification::model::{sz1, AkltProjector, BilinearBiquadratic};
use thermal_imps_purification::runner::read_result;

// HDF5 native descriptors can be inherited by concurrently launched solve processes.
static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn projector_preset_parses() {
    let dir = tempfile::tempdir().unwrap();
    for format in ["json", "toml"] {
        let value = support::config(
            dir.path(),
            &format!("parse-{format}"),
            "aklt_projector",
            2,
            0.2,
        );
        let text = match format {
            "json" => value.to_string(),
            "toml" => toml::to_string(&value).unwrap(),
            _ => unreachable!(),
        };
        let parsed = match format {
            "json" => RunConfig::from_json_str(&text),
            "toml" => RunConfig::from_toml_str(&text),
            _ => unreachable!(),
        };
        assert!(parsed.is_ok(), "{format}: {parsed:?}");
    }
}

#[test]
fn projector_preset_resolves_task_one_model_and_observable() {
    let expected = AkltProjector.local();
    let model = ModelSpec::AkltProjector;
    let actual = model.hamiltonian().unwrap();

    assert_eq!(actual.two_site_h, expected.two_site_h);
    assert_eq!(actual.site_energy, expected.site_energy);
    assert_eq!(actual.dim(), 3);
    assert_eq!(model.magnetization_op().unwrap(), sz1());
    assert!(matches!(
        model.resolve().unwrap().hamiltonian,
        ItebdHamiltonian::Real(_)
    ));
    assert!(!model.has_exact());
    assert!(model.exact(0.2).is_none());
}

#[test]
fn projector_model_spec_json_round_trip_retains_tag() {
    let value = serde_json::to_value(ModelSpec::AkltProjector).unwrap();
    assert_eq!(value, json!({"type": "aklt_projector"}));

    let decoded: ModelSpec = serde_json::from_value(value).unwrap();
    assert!(matches!(decoded, ModelSpec::AkltProjector));
}

#[test]
fn old_aklt_preset_retains_matrix_and_tag() {
    let dir = tempfile::tempdir().unwrap();
    let value = support::config(dir.path(), "old-aklt", "aklt", 2, 0.2);
    let cfg = RunConfig::from_json_str(&value.to_string()).unwrap();

    assert_eq!(
        serde_json::to_value(&cfg.model).unwrap(),
        json!({"type": "aklt"})
    );
    let actual = cfg.model.hamiltonian().unwrap();
    let expected = BilinearBiquadratic::aklt().local();
    assert_eq!(actual.two_site_h, expected.two_site_h);
    assert_eq!(actual.site_energy, expected.site_energy);
}

#[test]
fn projector_preset_parameters_do_not_tune_fixed_model() {
    let dir = tempfile::tempdir().unwrap();
    let mut value = support::config(dir.path(), "fixed", "aklt_projector", 2, 0.2);
    value["model"]["j"] = json!(-17.0);
    value["model"]["j1"] = json!(41.0);
    value["model"]["j2"] = json!(-23.0);

    let expected = AkltProjector.local();
    for format in ["json", "toml"] {
        let text = match format {
            "json" => value.to_string(),
            "toml" => toml::to_string(&value).unwrap(),
            _ => unreachable!(),
        };
        let cfg = match format {
            "json" => RunConfig::from_json_str(&text),
            "toml" => RunConfig::from_toml_str(&text),
            _ => unreachable!(),
        }
        .unwrap();
        let actual = cfg.model.hamiltonian().unwrap();
        assert_eq!(actual.two_site_h, expected.two_site_h);
        assert_eq!(actual.site_energy, expected.site_energy);
    }
}

#[test]
fn checked_in_projector_configs_are_equivalent() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let toml_text = std::fs::read_to_string(root.join("configs/aklt_projector.toml")).unwrap();
    let json_text = std::fs::read_to_string(root.join("configs/aklt_projector.json")).unwrap();
    let toml_cfg = RunConfig::from_toml_str(&toml_text).unwrap();
    let json_cfg = RunConfig::from_json_str(&json_text).unwrap();

    for cfg in [&toml_cfg, &json_cfg] {
        assert!(matches!(cfg.model, ModelSpec::AkltProjector));
        assert_eq!(cfg.evolution.dtau, 0.025);
        assert_eq!(cfg.evolution.beta_max, 0.4);
        assert_eq!(cfg.evolution.record_every_beta, 0.2);
        assert_eq!(cfg.evolution.trotter_order, TrotterOrder::Second);
        assert_eq!(cfg.truncation.epsilon, 1e-12);
        assert_eq!(cfg.truncation.max_bond, Some(32));
        assert_eq!(cfg.run.canonicalize_every, 1);
        assert!(!cfg.output.include_exact);
    }
    assert_eq!(toml_cfg.output.path, "results/aklt_projector.json");
    assert_eq!(json_cfg.output.path, "results/aklt_projector_json.json");

    let ItebdHamiltonian::Real(toml_h) = toml_cfg.model.resolve().unwrap().hamiltonian else {
        panic!("checked-in TOML selected the complex backend")
    };
    let ItebdHamiltonian::Real(json_h) = json_cfg.model.resolve().unwrap().hamiltonian else {
        panic!("checked-in JSON selected the complex backend")
    };
    assert_eq!(toml_h.two_site_h, json_h.two_site_h);
    assert_eq!(toml_h.site_energy, json_h.site_energy);
}

#[test]
fn projector_cli_json_and_toml_emit_finite_tagged_results() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();

    for format in ["json", "toml"] {
        let value = support::config(dir.path(), format, "aklt_projector", 2, 0.2);
        let input_path = dir.path().join(format!("input.{format}"));
        support::write_input(&value, &input_path, format);

        let output = support::run_cli(&input_path);
        assert!(
            output.status.success(),
            "{format}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result = read_result(Path::new(value["output"]["path"].as_str().unwrap())).unwrap();
        assert_eq!(
            serde_json::to_value(&result.metadata.model).unwrap(),
            json!({"type": "aklt_projector"})
        );
        assert_eq!(result.metadata.local_dim, 3);
        assert!(!result.records.is_empty());
        for record in &result.records {
            assert_eq!(record.f.is_none(), record.beta == 0.0);
            assert!(record.f.is_none_or(f64::is_finite));
            for observation in [
                record.beta,
                record.u,
                record.c,
                record.beta_f,
                record.magnetization,
            ] {
                assert!(observation.is_finite(), "non-finite {format} observation");
            }
        }
        support::close(result.records.last().unwrap().beta, 0.2, 1e-12);
    }
}

#[test]
fn projector_cli_rejects_exact_without_creating_outputs_or_changing_input() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let mut value = support::config(dir.path(), "exact", "aklt_projector", 2, 0.2);
    let result_path = dir.path().join("must-not-exist.result.json");
    let checkpoint_path = dir.path().join("must-not-exist.h5");
    value["output"]["path"] = json!(result_path);
    value["output"]["include_exact"] = json!(true);
    value["checkpoint"] = json!({"path": checkpoint_path, "every_steps": 1});
    let input_path = dir.path().join("include-exact.json");
    support::write_input(&value, &input_path, "json");
    let original = std::fs::read(&input_path).unwrap();

    let output = support::run_cli(&input_path);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("include_exact"));
    assert_eq!(std::fs::read(&input_path).unwrap(), original);
    assert!(!result_path.exists());
    assert!(!checkpoint_path.exists());
}
