use serde_json::{json, Value};
use thermal_imps_purification::config::RunConfig;
use thermal_imps_purification::runner::{read_result, write_result, Record};
use thermal_imps_purification::runner::{run_sweep, SweepResult};
use thermal_imps_purification::solve_run::run_checkpointed_sweep;

#[path = "support/solve_matrix.rs"]
mod support;

fn config(model: &str, exact: bool) -> RunConfig {
    RunConfig::from_toml_str(&format!(
        "[model]\n{model}\n[evolution]\ndtau=0.05\nbeta_max=0.1\nrecord_every_beta=0.1\n\
         [truncation]\nepsilon=1e-10\nmax_bond=16\n\
         [output]\npath='unused.json'\ninclude_exact={exact}\n"
    ))
    .unwrap()
}

#[test]
fn fresh_runs_emit_analytic_initial_observation_before_evolution() {
    for (model, exact, energy, dim) in [
        ("type='tfim'\nj=1.0\ng=0.7", true, 0.0, 2.0_f64),
        ("type='xy'\ngamma=0.5\nh=0.7", true, 0.0, 2.0),
        ("type='aklt_projector'", false, 5.0 / 9.0, 3.0),
        ("type='aklt'", false, 4.0 / 9.0, 3.0),
    ] {
        let result = run_sweep(&config(model, exact)).unwrap();
        let value = serde_json::to_value(&result).unwrap();
        let initial = &value["records"][0];
        assert_eq!(initial["beta"], 0.0, "{model}");
        assert_eq!(result.records.len(), 2);
        assert!((initial["u"].as_f64().unwrap() - energy).abs() < 1e-14);
        assert_eq!(initial["c"], 0.0);
        assert_eq!(initial["magnetization"], 0.0);
        assert_eq!(initial["max_bond"], 1);
        assert_eq!(initial["f"], Value::Null);
        assert!((initial["beta_f"].as_f64().unwrap() + dim.ln()).abs() < 1e-14);
        if exact {
            assert_eq!(initial["exact"]["u"], 0.0);
            assert_eq!(initial["exact"]["c"], 0.0);
            assert_eq!(initial["exact"]["f"], Value::Null);
        }
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: SweepResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), value);
        assert_eq!(value["records"][1]["beta"], 0.1);
        assert!(value["records"][1]["f"].is_number());
    }
}

#[test]
fn legacy_positive_beta_records_remain_readable_and_gain_beta_f() {
    let mut value =
        serde_json::to_value(run_sweep(&config("type='tfim'\nj=1.0\ng=0.7", true)).unwrap())
            .unwrap();
    let last = value["records"].as_array_mut().unwrap().pop().unwrap();
    value["records"] = json!([last]);
    value["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("beta_f");
    let decoded: SweepResult = serde_json::from_value(value.clone()).unwrap();
    let encoded = serde_json::to_value(decoded).unwrap();
    let want =
        value["records"][0]["beta"].as_f64().unwrap() * value["records"][0]["f"].as_f64().unwrap();
    assert_eq!(encoded["records"][0]["beta_f"], want);
}

#[test]
fn initial_matrix_observables_use_normalized_traces_in_both_backends() {
    for complex in [false, true] {
        let mut value = if complex {
            support::phase_config()
        } else {
            support::real_tfim_config()
        };
        // Independent trace oracle: a uniform identity shift contributes +2 per site;
        // the observable has diagonal (1, 3), hence infinite-temperature mean 2.
        for field in ["two_site_h", "site_energy"] {
            for i in 0..4 {
                let entry = &mut value["model"][field]["real"][i][i];
                *entry = json!(entry.as_f64().unwrap() + 2.0);
            }
        }
        value["model"]["observable"]["matrix"]["real"][0][0] = json!(1.0);
        value["model"]["observable"]["matrix"]["real"][1][1] = json!(3.0);
        let cfg = RunConfig::from_json_str(&value.to_string()).unwrap();
        let result = run_sweep(&cfg).unwrap();
        let initial = &result.records[0];
        assert_eq!(initial.beta, 0.0);
        assert_eq!(initial.u, 2.0);
        assert_eq!(initial.magnetization, 2.0);
        assert_eq!(initial.c, 0.0);
        assert_eq!(initial.f, None);
        assert_eq!(initial.max_bond, 1);
        assert!(initial.exact.is_none());
    }
}

#[test]
fn checkpoint_restart_from_zero_does_not_repeat_the_initial_observation() {
    let dir = tempfile::tempdir().unwrap();
    for complex in [false, true] {
        let mut value = if complex {
            support::phase_config()
        } else {
            support::real_tfim_config()
        };
        let stem = if complex { "complex" } else { "real" };
        let source_json = dir.path().join(format!("{stem}.json"));
        let source_h5 = dir.path().join(format!("{stem}.h5"));
        value["output"]["path"] = json!(source_json);
        value["checkpoint"] = json!({"path": source_h5, "every_steps": 1});
        let fresh =
            run_checkpointed_sweep(&RunConfig::from_json_str(&value.to_string()).unwrap(), None)
                .unwrap();
        assert_eq!(fresh.records[0].beta, 0.0);
        let bytes = std::fs::read(&source_json).unwrap();
        let round_trip = read_result(&source_json).unwrap();
        assert_eq!(
            serde_json::to_value(&fresh).unwrap(),
            serde_json::to_value(round_trip).unwrap()
        );
        value["output"]["path"] = json!(dir.path().join(format!("{stem}-restart.json")));
        value["checkpoint"]["path"] = json!(dir.path().join(format!("{stem}-restart.h5")));
        value["restart"] = json!({"path": source_h5, "snapshot": 0});
        let resumed =
            run_checkpointed_sweep(&RunConfig::from_json_str(&value.to_string()).unwrap(), None)
                .unwrap();
        assert_eq!(resumed.records.len(), fresh.records.len() - 1);
        assert!(resumed.records.iter().all(|r| r.beta > 0.0));
        for (a, b) in fresh.records[1..].iter().zip(&resumed.records) {
            assert_eq!(
                serde_json::to_value(a).unwrap(),
                serde_json::to_value(b).unwrap()
            );
        }
        assert_eq!(std::fs::read(&source_json).unwrap(), bytes);
    }
}

#[test]
fn initial_observations_round_trip_through_real_result_io() {
    let dir = tempfile::tempdir().unwrap();
    let result = run_sweep(&config("type='tfim'\nj=1.0\ng=0.7", true)).unwrap();
    let path = dir.path().join("nested/run.json");
    write_result(&path, &result).unwrap();
    let decoded = read_result(&path).unwrap();
    assert_eq!(
        serde_json::to_value(decoded).unwrap(),
        serde_json::to_value(result).unwrap()
    );
}

#[test]
fn malformed_free_energy_records_are_rejected_with_field_context() {
    let zero = json!({"beta":0.0,"u":0.0,"c":0.0,"f":null,
        "beta_f":-0.6931471805599453,"magnetization":0.0,"max_bond":1,"exact":null});
    let positive = json!({"beta":0.5,"u":-0.2,"c":0.1,"f":-1.5,
        "beta_f":-0.75,"magnetization":0.1,"max_bond":4,"exact":null});
    let mut cases = Vec::new();
    for source in [&zero, &positive] {
        let mut missing = source.clone();
        missing.as_object_mut().unwrap().remove("f");
        cases.push((missing, "f"));
        let mut wrong_exact = source.clone();
        wrong_exact["exact"] = json!({"u":0.0,"c":0.0,"f":null,"magnetization":0.0});
        if source["beta"] == 0.0 {
            wrong_exact["exact"]["f"] = json!(0.0);
        }
        cases.push((wrong_exact.clone(), "exact.f"));
        wrong_exact["exact"].as_object_mut().unwrap().remove("f");
        cases.push((wrong_exact, "f"));
    }
    let mut missing_beta_f = zero.clone();
    missing_beta_f.as_object_mut().unwrap().remove("beta_f");
    cases.push((missing_beta_f, "beta_f"));
    let mut finite_zero = zero;
    finite_zero["f"] = json!(0.0);
    cases.push((finite_zero, "f"));
    let mut null_positive = positive.clone();
    null_positive["f"] = Value::Null;
    cases.push((null_positive, "f"));
    let mut null_beta_f = positive.clone();
    null_beta_f["beta_f"] = Value::Null;
    cases.push((null_beta_f, "f64"));
    let mut inconsistent = positive;
    inconsistent["beta_f"] = json!(17.0);
    cases.push((inconsistent, "beta_f"));
    for (input, context) in cases {
        let error = serde_json::from_value::<Record>(input.clone()).unwrap_err();
        assert!(error.to_string().contains(context), "{input}: {error}");
    }
}
