use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};
use thermal_imps_purification::{
    aklt_reference::output::{finalize_interrupted_directory, run_to_directory, verify_directory},
    aklt_reference::{preflight, resolve_request, AkltReferenceRequest},
    config::{TrotterOrder, TruncationCfg},
    itebd_auto::ItebdState,
    itebd_checkpoint::*,
};
// Match the existing native-descriptor/subprocess coordination discipline.
static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());
const HEADER:&str="temperature_tick,temperature,beta_target,beta_actual,completed_steps,available,reason,energy,mx,mz,free_energy,entropy,accumulated_log_norm,parity_delta_energy,parity_delta_mx,parity_delta_mz,direct_energy_difference,maximum_rdm_fixed_point_residual,maximum_rdm_trace_residual,maximum_rdm_hermiticity_residual,maximum_rdm_iterations,minimum_rdm_eigenvalue,bond_dimension_ab,bond_dimension_ba,cap_reached_now,maximum_dimension_seen,cap_ever_reached,cap_observed_strang_steps,cap_coverage";
fn request() -> AkltReferenceRequest {
    serde_json::from_value(json!({"schema_version":1,"contract":"aklt-projector-reference-v1","model":{"name":"aklt_projector"},"variant":"fine","max_seconds":3600.,"source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})).unwrap()
}
fn put(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
// Analytical infinite-temperature tensors with labeled progress are integrity fixtures,
// not evidence of a physical finite-temperature trajectory.
fn fixture(
    path: &Path,
    checkpoint: bool,
    available: usize,
    off_target: bool,
    hit: bool,
) -> ItebdCheckpointProgress {
    fixture_request(path, checkpoint, available, off_target, hit, request())
}
fn fixture_request(
    path: &Path,
    checkpoint: bool,
    available: usize,
    off_target: bool,
    hit: bool,
    req: AkltReferenceRequest,
) -> ItebdCheckpointProgress {
    fs::create_dir(path).unwrap();
    let settings = resolve_request(&req).unwrap();
    put(&path.join("conditions.json"), &preflight(&req).unwrap());
    let steps = if off_target {
        51
    } else {
        settings
            .targets()
            .get(available.wrapping_sub(1))
            .map_or(0, |t| t.completed_steps)
    };
    let mut left = steps;
    let mut beta = 0.;
    for segment in settings.segments() {
        let n = left.min(segment.steps);
        beta += 2. * n as f64 * segment.dtau;
        left -= n;
    }
    let progress = ItebdCheckpointProgress {
        beta,
        completed_steps: steps,
        accumulated_log_norm: 0.,
        last_step: None,
    };
    let mut csv = format!("{HEADER}\n");
    for (i, t) in settings.targets().iter().enumerate() {
        let mut row = vec![String::new(); 29];
        row[0] = t.temperature_tick.to_string();
        row[1] = t.temperature.to_string();
        row[2] = t.beta.to_string();
        let observed = i < available || off_target;
        let actual_steps = if i < available {
            t.completed_steps
        } else if off_target {
            steps
        } else {
            0
        };
        row[3] = if i < available {
            t.beta
        } else if off_target {
            beta
        } else {
            0.
        }
        .to_string();
        row[4] = actual_steps.to_string();
        row[5] = (i < available).to_string();
        row[12] = "0".into();
        if i < available {
            row[7] = (5. / 9.).to_string();
            row[10] = (-3f64.ln() / t.beta).to_string();
            row[11] = (3f64.ln() + t.beta * 5. / 9.).to_string();
            for n in [8, 9, 13, 14, 15, 16, 17, 18, 19] {
                row[n] = "0".into();
            }
            row[20] = "1".into();
            row[21] = (1. / 9.).to_string();
            row[22] = "1".into();
            row[23] = "1".into();
            row[24] = "false".into();
        } else {
            row[6] = "\"time limit, \"\"stop\"\"\nobserved\"".into();
        }
        row[27] = actual_steps.to_string();
        row[28] = if observed {
            "complete_through_stop"
        } else {
            "unobserved_tail"
        }
        .into();
        if observed {
            row[25] = if hit { "128" } else { "1" }.into();
            row[26] = hit.to_string();
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
            model_label: Some("aklt_projector".into()),
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
        writer.append((&state).into(), &progress).unwrap();
        writer.finish().unwrap();
    }
    let contract_name = &req.contract;
    put(
        &path.join("summary.json"),
        &json!({"schema_version":req.schema_version,"contract":contract_name,"request":req,"status":if available==20 && checkpoint{"complete"}else{"partial"},"reason":"integrity fixture","completed_steps":steps,"beta_actual":beta,"accumulated_log_norm":0.,"completed_steps_kind":"exact","available_samples":available,"elapsed_seconds":if req.schema_version==2 {4001.}else{0.},"budget_overrun_seconds":if req.schema_version==2 {Value::Null}else{json!(0.)},"checkpoint_available":checkpoint,
        "cap":{"maximum_dimension_seen":if hit{128}else{1},"ever_reached":hit,"observed_strang_steps":steps,"coverage":"complete_through_stop"}}),
    );
    progress
}
#[test]
fn aklt_cli_preflight_is_read_only_and_strict() {
    let _guard = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("request.json");
    put(&input, &serde_json::to_value(request()).unwrap());
    let out = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .arg("--preflight")
        .arg(&input)
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let p: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(p["total_steps"], 1014);
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    for args in [
        vec!["--verify"],
        vec!["--preflight", "x", "extra"],
        vec!["--resume", "x"],
    ] {
        assert!(!Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
    }
}
#[test]
fn aklt_bundle_exact_round_trip_and_off_target_budget_stop() {
    let _guard = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    for off_target in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("bundle");
        let p = fixture(
            &bundle,
            true,
            if off_target { 1 } else { 20 },
            off_target,
            false,
        );
        let report = verify_directory(&bundle).unwrap();
        assert_eq!(report["sample_count"], 20);
        assert_eq!(report["checkpoint_verified"], true);
        assert_eq!(fs::read_dir(&bundle).unwrap().count(), 4);
        let loaded =
            load_itebd_checkpoint(&bundle.join("state.h5"), 0, &Default::default()).unwrap();
        assert_eq!(loaded.progress.beta.to_bits(), p.beta.to_bits());
        assert_eq!(
            loaded.progress.accumulated_log_norm.to_bits(),
            p.accumulated_log_norm.to_bits()
        );
        assert_eq!(
            list_itebd_checkpoints(&bundle.join("state.h5"))
                .unwrap()
                .len(),
            1
        );
        let ItebdState::Real(s) = loaded.state else {
            panic!("real")
        };
        let initial = thermal_imps_purification::purified_mps::infinite_temperature(3);
        for (a, b) in [
            (&s.a.gamma, &initial.a.gamma),
            (&s.b.gamma, &initial.b.gamma),
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
        assert_eq!(s.lambda_ab, initial.lambda_ab);
        assert_eq!(s.lambda_ba, initial.lambda_ba);
        assert_eq!(
            report["beta_actual"].as_f64().unwrap().to_bits(),
            p.beta.to_bits()
        );
        if off_target {
            assert!(p.beta > 0.5 && p.beta < 10. / 19.);
            assert_eq!(report["available_samples"], 1);
            assert!(report["terminal_consistency_passed"].is_null());
        }
    }
}
#[test]
fn aklt_bundle_rejects_inconsistent_evidence() {
    let _guard = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("bundle");
    fixture(&bundle, true, 20, false, true);
    let cp = bundle.join("conditions.json");
    let conditions: Value = serde_json::from_slice(&fs::read(&cp).unwrap()).unwrap();
    for pointer in [
        "/request/model/name",
        "/request/source_revision",
        "/time_grid/segments/0/dtau",
    ] {
        let mut v = conditions.clone();
        *v.pointer_mut(pointer).unwrap() = if pointer.ends_with("dtau") {
            json!(0.01)
        } else {
            json!("changed")
        };
        put(&cp, &v);
        let e = format!("{:#}", verify_directory(&bundle).unwrap_err());
        assert!(e.contains("conditions.json"), "{e}");
    }
    put(&cp, &conditions);
    let sp = bundle.join("summary.json");
    let summary: Value = serde_json::from_slice(&fs::read(&sp).unwrap()).unwrap();
    for (pointer, value) in [
        ("/beta_actual", json!(9.)),
        ("/accumulated_log_norm", json!(1.)),
        ("/available_samples", json!(19)),
        ("/cap/ever_reached", json!(false)),
        ("/cap/observed_strang_steps", json!(1)),
        ("/cap/coverage", json!("unobserved_tail")),
        ("/budget_overrun_seconds", json!(1.)),
    ] {
        let mut v = summary.clone();
        *v.pointer_mut(pointer).unwrap() = value;
        put(&sp, &v);
        assert!(verify_directory(&bundle).is_err(), "{pointer}");
    }
    put(&sp, &summary);
    let csvp = bundle.join("samples.csv");
    let original = fs::read_to_string(&csvp).unwrap();
    let mut lines = original.lines().map(str::to_owned).collect::<Vec<_>>();
    let mut row = lines[2].split(',').map(str::to_owned).collect::<Vec<_>>();
    row[25] = "1".into();
    row[26] = "false".into();
    lines[2] = row.join(",");
    fs::write(&csvp, lines.join("\n") + "\n").unwrap();
    assert!(verify_directory(&bundle).is_err());
    fs::write(&csvp, &original).unwrap();
    let hp = bundle.join("state.h5");
    let bytes = fs::read(&hp).unwrap();
    {
        let f = hdf5_metno::File::open_rw(&hp).unwrap();
        let d = f.dataset("run/hamiltonian/two_site_h_real").unwrap();
        let mut v = d.read_raw::<f64>().unwrap();
        v[0] += 0.1;
        d.write_raw(&v).unwrap();
        f.flush().unwrap();
    }
    let e = format!("{:#}", verify_directory(&bundle).unwrap_err());
    assert!(
        e.contains("state.h5") && e.contains("model matrices"),
        "{e}"
    );
    fs::write(&hp, bytes).unwrap();
    fs::write(bundle.join("extra"), b"extra").unwrap();
    assert!(verify_directory(&bundle).is_err());
    fs::remove_file(bundle.join("extra")).unwrap();
    #[cfg(unix)]
    {
        fs::remove_file(&csvp).unwrap();
        std::os::unix::fs::symlink("conditions.json", &csvp).unwrap();
        let e = format!("{:#}", verify_directory(&bundle).unwrap_err());
        assert!(e.contains("symlink"), "{e}");
    }
}
#[test]
fn aklt_interruption_preserves_unknown_tail() {
    let _guard = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    for hit in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("bundle");
        fixture(&bundle, false, 1, false, hit);
        fs::remove_file(bundle.join("summary.json")).unwrap();
        let before = fs::read(bundle.join("samples.csv")).unwrap();
        let report = finalize_interrupted_directory(&bundle, "native process failed").unwrap();
        assert_eq!(report["status"], "partial");
        assert_eq!(report["completed_steps_kind"], "retained_lower_bound");
        assert_eq!(report["cap"]["coverage"], "unobserved_tail");
        assert_eq!(
            report["cap"]["ever_reached"],
            if hit { json!(true) } else { Value::Null }
        );
        assert_eq!(fs::read(bundle.join("samples.csv")).unwrap(), before);
        verify_directory(&bundle).unwrap();
        fs::write(bundle.join("state.h5"), b"corrupt native evidence").unwrap();
        let error = format!(
            "{:#}",
            finalize_interrupted_directory(&bundle, "dead").unwrap_err()
        );
        assert!(
            error.contains("integrity") && error.contains("state.h5"),
            "{error}"
        );
        assert_eq!(
            fs::read(bundle.join("state.h5")).unwrap(),
            b"corrupt native evidence"
        );
    }
}
#[test]
fn aklt_output_collisions_are_real_errors() {
    let _guard = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("existing");
    fs::write(&file, b"keep").unwrap();
    let e = format!("{:#}", run_to_directory(&request(), &file).unwrap_err());
    assert!(e.contains("reserve") && e.contains("existing"), "{e}");
    assert_eq!(fs::read(&file).unwrap(), b"keep");
    let existing = dir.path().join("directory");
    fs::create_dir(&existing).unwrap();
    assert!(run_to_directory(&request(), &existing).is_err());
    assert!(run_to_directory(&request(), &file.join("child")).is_err());
}

#[test]
fn aklt_v2_bundle_roundtrip_and_interrupted_tail() {
    let _guard = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let mut value = serde_json::to_value(request()).unwrap();
    value["schema_version"] = json!(2);
    value["contract"] = json!("aklt-projector-reference-v2");
    value["max_seconds"] = Value::Null;
    let req: AkltReferenceRequest = serde_json::from_value(value).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("v2");
    let expected = fixture_request(&bundle, true, 20, false, true, req);
    let state_bytes = fs::read(bundle.join("state.h5")).unwrap();
    let verified = verify_directory(&bundle).unwrap();
    assert_eq!(verified["schema_version"], 2);
    assert_eq!(verified["contract"], "aklt-projector-reference-v2");
    assert_eq!(verified["elapsed_seconds"], 4001.);
    assert!(verified["budget_overrun_seconds"].is_null());
    let loaded = load_itebd_checkpoint(&bundle.join("state.h5"), 0, &Default::default()).unwrap();
    assert_eq!(loaded.progress.beta.to_bits(), expected.beta.to_bits());
    assert_eq!(verified["checkpoint_verified"], true);
    let output = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .arg("--verify")
        .arg(&bundle)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cli: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cli, verified);

    let sp = bundle.join("summary.json");
    let original: Value = serde_json::from_slice(&fs::read(&sp).unwrap()).unwrap();
    for (key, value) in [
        ("budget_overrun_seconds", json!(0.)),
        ("schema_version", json!(1)),
        ("contract", json!("aklt-projector-reference-v1")),
    ] {
        let mut bad = original.clone();
        bad[key] = value;
        put(&sp, &bad);
        assert!(verify_directory(&bundle).is_err(), "{key}");
    }
    fs::remove_file(&sp).unwrap();
    let finalised = finalize_interrupted_directory(&bundle, "fixture process ended").unwrap();
    assert_eq!(finalised["schema_version"], 2);
    assert_eq!(finalised["cap"]["coverage"], "unobserved_tail");
    assert_eq!(finalised["cap"]["ever_reached"], true);
    assert!(finalised["elapsed_seconds"].is_null());
    assert!(finalised["budget_overrun_seconds"].is_null());
    assert_eq!(fs::read(bundle.join("state.h5")).unwrap(), state_bytes);
    verify_directory(&bundle).unwrap();
}
