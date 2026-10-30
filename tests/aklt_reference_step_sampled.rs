use serde_json::{json, Value};
use std::process::Command;
use thermal_imps_purification::aklt_reference::{parse_worker_request, preflight_worker};

fn request_json() -> Value {
    json!({"schema_version":5,"contract":"aklt-projector-reference-step-sampled-v5",
        "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,
        "max_seconds":null,"measurement_policy":"every-two-complete-steps-v1",
        "recording_policy":"observables-only-v1","heat_capacity_policy":"variance-default-v1",
        "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64),
        "restart":{"directory":"../parent","snapshot_index":0,
            "files":{"conditions.json":{"bytes":1,"sha256":"c".repeat(64)},
                "samples.csv":{"bytes":1,"sha256":"d".repeat(64)},
                "states.h5":{"bytes":1,"sha256":"e".repeat(64)},
                "summary.json":{"bytes":1,"sha256":"f".repeat(64)}},
            "progress":{"completed_steps":50,"beta":0.5000000000000002,
                "accumulated_log_norm":-0.4937122532815281},
            "cap":{"maximum_dimension_seen":30,"ever_reached":false,
                "observed_strang_steps":50,"coverage":"complete_through_stop"}}})
}

// A temperature-target adapter must not silently replace step-based observations.
#[test]
fn step_preflight_retains_legacy_evolution_and_483_step_identities() {
    let parsed = parse_worker_request(&serde_json::to_vec(&request_json()).unwrap());
    assert!(parsed.is_ok(), "step request rejected: {:?}", parsed.err());
    let p = preflight_worker(&parsed.unwrap()).unwrap();
    assert_eq!(p["remaining_steps"], 964);
    assert_eq!(p["total_steps"], 1014);
    assert_eq!(p["initial_beta"], 0.5000000000000002);
    let points = p["sample_points"].as_array().unwrap();
    assert_eq!(points.len(), 483);
    assert_eq!(points[0]["completed_steps"], 50);
    assert_eq!(points[1]["completed_steps"], 52);
    assert_eq!(points[482]["completed_steps"], 1014);
    assert_eq!(points[482]["sample_index"], 482);
    assert_eq!(p["recording_policy"], "observables-only-v1");
    assert!(!p.as_object().unwrap().contains_key("targets"));
    assert!(points.iter().all(|v| v.get("temperature_tick").is_none()));
}

#[test]
fn step_profile_refuses_incompatible_controls() {
    for (key, value) in [
        ("max_bond", json!(128)),
        ("measurement_policy", json!("every-one-complete-step-v1")),
        ("recording_policy", json!("target-snapshots-v1")),
        ("variant", json!("time-step")),
        ("max_seconds", json!(3600)),
        ("unexpected", json!(true)),
    ] {
        let mut r = request_json();
        r[key] = value;
        if let Ok(parsed) = parse_worker_request(&serde_json::to_vec(&r).unwrap()) {
            assert!(preflight_worker(&parsed).is_err(), "accepted {key}");
        }
    }
    let mut r = request_json();
    r.as_object_mut().unwrap().remove("max_seconds");
    assert!(parse_worker_request(&serde_json::to_vec(&r).unwrap()).is_err());
    let mut r = request_json();
    r["restart"]["progress"]["completed_steps"] = 154.into();
    if let Ok(parsed) = parse_worker_request(&serde_json::to_vec(&r).unwrap()) {
        assert!(preflight_worker(&parsed).is_err());
    }
}

#[test]
fn selected_preflight_reports_missing_parent_without_creating_output() {
    let root = tempfile::tempdir().unwrap();
    let request = root.path().join("request.json");
    let output = root.path().join("output");
    std::fs::write(&request, serde_json::to_vec(&request_json()).unwrap()).unwrap();
    let reply = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .arg("--preflight")
        .arg(&request)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!reply.status.success());
    let error = String::from_utf8_lossy(&reply.stderr);
    assert!(error.contains("parent"), "wrong admission error: {error}");
    assert!(!output.exists());
}

#[test]
fn scalar_verifier_dispatch_keeps_v5_error_context() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("conditions.json"),
        serde_json::to_vec(&json!({"contract":"aklt-projector-reference-step-sampled-v5"}))
            .unwrap(),
    )
    .unwrap();
    std::fs::write(root.path().join("samples.csv"), b"broken\n").unwrap();
    std::fs::write(root.path().join("summary.json"), b"{}").unwrap();
    let reply = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .arg("--verify")
        .arg(root.path())
        .output()
        .unwrap();
    assert!(!reply.status.success());
    let error = String::from_utf8_lossy(&reply.stderr);
    assert!(
        error.contains("verify step-sampled scalars"),
        "wrong verifier dispatch: {error}"
    );
    assert!(!root.path().join("states.h5").exists());
}

#[test]
fn scalar_production_refuses_thread_override_before_output_creation() {
    let root = tempfile::tempdir().unwrap();
    let request = root.path().join("request.json");
    let output = root.path().join("out");
    std::fs::write(&request, serde_json::to_vec(&request_json()).unwrap()).unwrap();
    let reply = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .env("RAYON_NUM_THREADS", "1")
        .arg(&request)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!reply.status.success());
    let error = String::from_utf8_lossy(&reply.stderr);
    assert!(
        error.contains("requires native defaults"),
        "wrong production refusal: {error}"
    );
    assert!(!output.exists());
}

#[test]
fn scalar_missing_summary_reports_the_actual_metadata_path() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("conditions.json"),
        serde_json::to_vec(&json!({"contract":"aklt-projector-reference-step-sampled-v5"}))
            .unwrap(),
    )
    .unwrap();
    let reply = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .arg("--verify")
        .arg(root.path())
        .output()
        .unwrap();
    assert!(!reply.status.success());
    let error = String::from_utf8_lossy(&reply.stderr);
    assert!(
        error.contains("summary.json"),
        "missing I/O path context: {error}"
    );
}
