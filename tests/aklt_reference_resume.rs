use serde_json::{json, Value};
use thermal_imps_purification::aklt_reference::{parse_worker_request, preflight_worker};

fn request() -> Value {
    json!({"schema_version":4,"contract":"aklt-projector-reference-resume-v4",
        "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,
        "max_seconds":null,"recording_policy":"target-snapshots-v1","heat_capacity_policy":"variance-default-v1",
        "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64),
        "restart":{"directory":"../../aklt-projector-dmax256-reference/fine","snapshot_index":13,
            "files":{"conditions.json":{"bytes":1,"sha256":"c".repeat(64)},"samples.csv":{"bytes":1,"sha256":"d".repeat(64)},"states.h5":{"bytes":1,"sha256":"e".repeat(64)},"summary.json":{"bytes":1,"sha256":"f".repeat(64)}},
            "progress":{"completed_steps":154,"beta":1.4285714285714268,"accumulated_log_norm":-1.1173122177313783},
            "cap":{"maximum_dimension_seen":68,"ever_reached":false,"observed_strang_steps":154,"coverage":"complete_through_stop"}}})
}

// Resetting beta/norm, replaying the prefix, or recording the boundary twice breaks continuation.
#[test]
fn resume_preflight_keeps_absolute_grid_and_only_six_new_targets() {
    let parsed = parse_worker_request(&serde_json::to_vec(&request()).unwrap());
    assert!(
        parsed.is_ok(),
        "explicit resume request rejected: {:?}",
        parsed.err()
    );
    let p = preflight_worker(&parsed.unwrap()).unwrap();
    assert_eq!(p["contract"], "aklt-projector-reference-resume-v4");
    assert_eq!(p["initial_beta"], request()["restart"]["progress"]["beta"]);
    assert_eq!(p["remaining_steps"], 860);
    assert_eq!(p["total_steps"], 1014);
    assert_eq!(p["targets"].as_array().unwrap().len(), 6);
    assert_eq!(p["targets"][0]["temperature_tick"], 6);
    assert_eq!(p["targets"][0]["completed_steps"], 178);
    assert_eq!(p["targets"][5]["completed_steps"], 1014);
    assert_eq!(p["snapshot_policy"]["maximum_count"], 6);
}

#[test]
fn resume_rejects_boundary_and_source_drift() {
    for (key, value) in [
        ("snapshot_index", json!(12)),
        ("directory", json!("/absolute/parent")),
        (
            "progress",
            json!({"completed_steps":0,"beta":0.,"accumulated_log_norm":0.}),
        ),
    ] {
        let mut r = request();
        r["restart"][key] = value;
        if let Ok(parsed) = parse_worker_request(&serde_json::to_vec(&r).unwrap()) {
            assert!(preflight_worker(&parsed).is_err());
        }
    }
}

// A T0.3 restart must not replay the four already retained continuation targets.
#[test]
fn t03_resume_preflight_has_only_two_cold_targets() {
    let mut r = request();
    r["restart"]["snapshot_index"] = 3.into();
    r["restart"]["progress"] =
        json!({"completed_steps":346,"beta":10./3.,"accumulated_log_norm":-3.46});
    r["restart"]["cap"]["observed_strang_steps"] = 346.into();
    let parsed = parse_worker_request(&serde_json::to_vec(&r).unwrap()).unwrap();
    let result = preflight_worker(&parsed);
    assert!(result.is_ok(), "T0.3 boundary rejected: {:?}", result.err());
    let p = result.unwrap();
    assert_eq!(p["remaining_steps"], 668);
    assert_eq!(p["snapshot_policy"]["maximum_count"], 2);
    assert_eq!(p["targets"].as_array().unwrap().len(), 2);
    assert_eq!(p["targets"][0]["temperature_tick"], 2);
    assert_eq!(p["targets"][0]["completed_steps"], 514);
    assert_eq!(p["targets"][1]["completed_steps"], 1014);
    for (key, value) in [
        ("snapshot_index", json!(2)),
        (
            "progress",
            json!({"completed_steps":345,"beta":10./3.,"accumulated_log_norm":-3.46}),
        ),
    ] {
        let mut bad = r.clone();
        bad["restart"][key] = value;
        let parsed = parse_worker_request(&serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(preflight_worker(&parsed).is_err());
    }
}
