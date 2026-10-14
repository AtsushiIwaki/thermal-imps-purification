use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};
use thermal_imps_purification::{
    config::{TrotterOrder, TruncationCfg},
    ising_reference::output::{finalize_interrupted_directory, run_to_directory, verify_directory},
    ising_reference::{preflight, resolve_request, ReferenceRequest},
    itebd_auto::ItebdState,
    itebd_checkpoint::*,
};
const HEADER: &str = "temperature_tick,temperature,beta_target,beta_actual,completed_steps,available,reason,energy,mx,mz,zz,free_energy,entropy,accumulated_log_norm,parity_delta_energy,parity_delta_mx,parity_delta_mz,parity_delta_zz,direct_energy_difference,maximum_rdm_fixed_point_residual,maximum_rdm_trace_residual,maximum_rdm_hermiticity_residual,maximum_rdm_iterations,minimum_rdm_eigenvalue,bond_dimension_ab,bond_dimension_ba";
fn request() -> ReferenceRequest {
    serde_json::from_value(json!({"schema_version":1,"model":{"name":"ising_mixed","g":1.05,"h":0.5},"variant":"fine","source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})).unwrap()
}
fn put(path: &Path, v: &Value) {
    fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}
// An analytical infinite-temperature state labeled with a scheduled progress is an
// integrity fixture only; it does not claim to be the physical evolved trajectory.
fn fixture(path: &Path, checkpoint: bool, available: usize) {
    fs::create_dir(path).unwrap();
    let req = request();
    let settings = resolve_request(&req).unwrap();
    put(&path.join("conditions.json"), &preflight(&req).unwrap());
    let mut csv = format!("{HEADER}\n");
    for (i, t) in settings.targets().iter().enumerate() {
        let mut row = vec![String::new(); 26];
        row[0] = t.temperature_tick.to_string();
        row[1] = t.temperature.to_string();
        row[2] = t.beta.to_string();
        row[3] = if i < available { t.beta } else { 0. }.to_string();
        row[4] = if i < available { t.completed_steps } else { 0 }.to_string();
        row[5] = (i < available).to_string();
        row[13] = "0".into();
        if i < available {
            for index in [7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21] {
                row[index] = "0".into();
            }
            row[11] = (-std::f64::consts::LN_2 / t.beta).to_string();
            row[12] = std::f64::consts::LN_2.to_string();
            row[22] = "1".into();
            row[23] = "0.25".into();
            row[24] = "1".into();
            row[25] = "1".into();
        } else {
            row[6] = "\"process said \"\"stop\"\",\nnot completed\"".into();
        }
        csv.push_str(&row.join(","));
        csv.push('\n');
    }
    fs::write(path.join("samples.csv"), csv).unwrap();
    if checkpoint {
        let state = ItebdState::infinite_temperature(settings.hamiltonian()).unwrap();
        let meta = ItebdRunMetadata {
            dtau: 0.005,
            trotter_order: TrotterOrder::Second,
            truncation: TruncationCfg {
                epsilon: 1e-13,
                max_bond: Some(128),
            },
            canonicalize_every: 1,
            record_every_beta: None,
            model_label: Some("ising_mixed".into()),
            git_revision: Some(req.source_revision.clone()),
            hermiticity_tolerance: 1e-12,
        };
        let mut writer = ItebdTrajectoryWriter::create_scheduled(
            path.join("state.h5"),
            meta,
            settings.hamiltonian(),
            settings.segments().to_vec(),
        )
        .unwrap();
        writer
            .append(
                (&state).into(),
                &ItebdCheckpointProgress {
                    beta: 10.,
                    completed_steps: 1014,
                    accumulated_log_norm: 0.,
                    last_step: None,
                },
            )
            .unwrap();
        writer.finish().unwrap();
    }
    put(
        &path.join("summary.json"),
        &json!({"schema_version":1,"request":req,"status":if available==20 && checkpoint {"complete"} else {"partial"},"reason":"fixture","completed_steps":if checkpoint {1014} else {settings.targets().get(available.wrapping_sub(1)).map_or(0,|t|t.completed_steps)},"completed_steps_kind":"exact","available_samples":available,"elapsed_seconds":0.0,"checkpoint_available":checkpoint}),
    );
}
#[test]
fn reference_cli_preflight_creates_no_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("request.json");
    put(&input, &serde_json::to_value(request()).unwrap());
    let out = Command::new(env!("CARGO_BIN_EXE_ising_reference"))
        .args(["--preflight"])
        .arg(&input)
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["total_steps"], 1014);
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    for args in [
        vec!["--verify"],
        vec!["--preflight", "x", "extra"],
        vec!["--bad", "x"],
    ] {
        assert!(!Command::new(env!("CARGO_BIN_EXE_ising_reference"))
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
    }
    put(&input, &json!({"schema_version":1,"unexpected":true}));
    assert!(!Command::new(env!("CARGO_BIN_EXE_ising_reference"))
        .arg("--preflight")
        .arg(input)
        .output()
        .unwrap()
        .status
        .success());
}
#[test]
fn reference_bundle_has_only_targets_and_one_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("bundle");
    fixture(&bundle, true, 20);
    let report = verify_directory(&bundle).unwrap();
    assert_eq!(report["sample_count"], 20);
    assert_eq!(report["available_samples"], 20);
    assert_eq!(report["terminal_consistency_passed"], true);
    let checkpoint_entries = list_itebd_checkpoints(&bundle.join("state.h5")).unwrap();
    assert_eq!(checkpoint_entries.len(), 1);
    assert_eq!(fs::read_dir(&bundle).unwrap().count(), 4);
    let loaded = load_itebd_checkpoint(&bundle.join("state.h5"), 0, &Default::default()).unwrap();
    assert_eq!(
        loaded.time_grid,
        ItebdTimeGrid::Segmented(resolve_request(&request()).unwrap().segments().to_vec())
    );
    let ItebdState::Real(state) = loaded.state else {
        panic!("backend")
    };
    let initial = thermal_imps_purification::purified_mps::infinite_temperature(2);
    for (a, b) in [
        (&state.a.gamma, &initial.a.gamma),
        (&state.b.gamma, &initial.b.gamma),
    ] {
        assert_eq!(
            a.to_vec::<f64>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            b.to_vec::<f64>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        state
            .lambda_ab
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        initial
            .lambda_ab
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        state
            .lambda_ba
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        initial
            .lambda_ba
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_ising_reference"))
        .arg("--verify")
        .arg(&bundle)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
#[test]
fn reference_existing_destination_is_not_overwritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing");
    fs::write(&path, b"keep").unwrap();
    let error = format!("{:#}", run_to_directory(&request(), &path).unwrap_err());
    assert!(
        error.contains("existing") && error.contains("reserve"),
        "{error}"
    );
    assert_eq!(fs::read(&path).unwrap(), b"keep");
    let error = format!(
        "{:#}",
        run_to_directory(&request(), &path.join("child")).unwrap_err()
    );
    assert!(error.contains("reserve"), "{error}");
}
#[test]
fn reference_partial_rows_survive_failed_process() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("bundle");
    fixture(&bundle, false, 1);
    fs::remove_file(bundle.join("summary.json")).unwrap();
    let before = fs::read(bundle.join("samples.csv")).unwrap();
    let report = finalize_interrupted_directory(
        &bundle,
        "native panic, \"failure\"\nobserved by supervisor",
    )
    .unwrap();
    assert_eq!(report["status"], "partial");
    assert_eq!(report["available_samples"], 1);
    assert_eq!(report["sample_count"], 20);
    assert_eq!(report["checkpoint_available"], false);
    assert_eq!(report["completed_steps_kind"], "retained_lower_bound");
    assert_eq!(report["terminal_consistency_passed"], Value::Null);
    assert_eq!(fs::read(bundle.join("samples.csv")).unwrap(), before);
    assert!(!bundle.join("state.h5").exists());
    verify_directory(&bundle).unwrap();
    fs::write(bundle.join("state.h5"), b"corrupt native evidence").unwrap();
    let error = format!(
        "{:#}",
        finalize_interrupted_directory(&bundle, "dead").unwrap_err()
    );
    assert!(
        error.contains("state.h5") && error.contains("integrity"),
        "{error}"
    );
    assert_eq!(
        fs::read(bundle.join("state.h5")).unwrap(),
        b"corrupt native evidence"
    );
    assert_eq!(fs::read(bundle.join("samples.csv")).unwrap(), before);
}
#[test]
fn reference_verify_rejects_mismatched_conditions_or_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("bundle");
    fixture(&bundle, true, 20);
    let conditions: Value =
        serde_json::from_slice(&fs::read(bundle.join("conditions.json")).unwrap()).unwrap();
    for field in ["model", "source_revision"] {
        let mut altered = conditions.clone();
        if field == "model" {
            altered["request"]["model"]["g"] = json!(1.04);
        } else {
            altered["request"]["source_revision"] = json!("c".repeat(40));
        }
        put(&bundle.join("conditions.json"), &altered);
        let error = format!("{:#}", verify_directory(&bundle).unwrap_err());
        assert!(
            error.contains("conditions.json")
                && (error.contains(field) || error.contains("identity")),
            "{error}"
        );
    }
    put(&bundle.join("conditions.json"), &conditions);
    fs::write(bundle.join("extra"), b"extra").unwrap();
    assert!(verify_directory(&bundle).is_err());
    fs::remove_file(bundle.join("extra")).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("conditions.json", bundle.join("extra")).unwrap();
        assert!(verify_directory(&bundle).is_err());
        fs::remove_file(bundle.join("extra")).unwrap();
    }
    let summary: Value =
        serde_json::from_slice(&fs::read(bundle.join("summary.json")).unwrap()).unwrap();
    let mut altered = summary.clone();
    altered["available_samples"] = json!(19);
    put(&bundle.join("summary.json"), &altered);
    assert!(verify_directory(&bundle).is_err());
    put(&bundle.join("summary.json"), &summary);
    let checkpoint_bytes = fs::read(bundle.join("state.h5")).unwrap();
    {
        let file = hdf5_metno::File::open_rw(bundle.join("state.h5")).unwrap();
        let dataset = file.dataset("run/hamiltonian/two_site_h_real").unwrap();
        let mut values = dataset.read_raw::<f64>().unwrap();
        values[0] += 0.25;
        dataset.write_raw(&values).unwrap();
        file.flush().unwrap();
    }
    let error = format!("{:#}", verify_directory(&bundle).unwrap_err());
    assert!(
        error.contains("state.h5") && error.contains("model matrices"),
        "{error}"
    );
    fs::write(bundle.join("state.h5"), &checkpoint_bytes).unwrap();
    {
        let file = hdf5_metno::File::open_rw(bundle.join("state.h5")).unwrap();
        file.create_group("states/000001").unwrap();
        file.flush().unwrap();
    }
    let error = format!(
        "{:#}",
        finalize_interrupted_directory(&bundle, "died").unwrap_err()
    );
    assert!(
        error.contains("integrity") && error.contains("snapshot"),
        "{error}"
    );
    fs::write(bundle.join("state.h5"), &checkpoint_bytes).unwrap();
    fs::remove_file(bundle.join("state.h5")).unwrap();
    let error = format!("{:#}", verify_directory(&bundle).unwrap_err());
    assert!(error.contains("availability"), "{error}");
    fs::write(bundle.join("state.h5"), &checkpoint_bytes).unwrap();
    let csv = fs::read_to_string(bundle.join("samples.csv")).unwrap();
    for (column, replacement) in [
        (0, "19"),
        (3, "0.49"),
        (7, "1.0"),
        (11, "0.0"),
        (19, "1e-3"),
        (24, "129"),
        (8, "NaN"),
    ] {
        let mut lines = csv.lines().map(str::to_owned).collect::<Vec<_>>();
        let mut row = lines[1].split(',').map(str::to_owned).collect::<Vec<_>>();
        row[column] = replacement.into();
        lines[1] = row.join(",");
        fs::write(bundle.join("samples.csv"), lines.join("\n") + "\n").unwrap();
        let error = format!("{:#}", verify_directory(&bundle).unwrap_err());
        assert!(error.contains("samples.csv"), "{error}");
    }
    fs::write(bundle.join("samples.csv"), csv).unwrap();
    fs::write(bundle.join("state.h5"), b"corrupt").unwrap();
    let error = format!("{:#}", verify_directory(&bundle).unwrap_err());
    assert!(error.contains("state.h5"), "{error}");
}
#[test]
fn interrupted_finalization_rejects_existing_inconsistent_summary() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("bundle");
    fixture(&bundle, false, 1);
    let mut summary: Value =
        serde_json::from_slice(&fs::read(bundle.join("summary.json")).unwrap()).unwrap();
    summary["request"]["source_revision"] = json!("c".repeat(40));
    put(&bundle.join("summary.json"), &summary);
    let before = fs::read(bundle.join("summary.json")).unwrap();
    let error = format!(
        "{:#}",
        finalize_interrupted_directory(&bundle, "observed death").unwrap_err()
    );
    assert!(
        error.contains("summary.json") && error.contains("identity"),
        "{error}"
    );
    assert_eq!(fs::read(bundle.join("summary.json")).unwrap(), before);
}
