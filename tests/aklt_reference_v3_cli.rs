use serde_json::{json, Value};
use std::{fs, process::Command};

fn request() -> Value {
    json!({"schema_version":3,"contract":"aklt-projector-reference-v3",
        "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,
        "max_seconds":null,"recording_policy":"target-snapshots-v1",
        "heat_capacity_policy":"variance-default-v1",
        "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})
}

#[test]
fn v3_cli_preflight_is_read_only() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("request.json");
    fs::write(&input, serde_json::to_vec(&request()).unwrap()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_aklt_reference"))
        .args(["--preflight", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let p: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(p["contract"], "aklt-projector-reference-v3");
    assert_eq!(p["total_steps"], 1014);
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn runtime_info_honors_native_environment() {
    let executable = env!("CARGO_BIN_EXE_aklt_reference");
    for explicit in [Some("2"), None] {
        let mut command = Command::new(executable);
        command.arg("--runtime-info");
        if let Some(value) = explicit {
            command.env("RAYON_NUM_THREADS", value);
        } else {
            command.env_remove("RAYON_NUM_THREADS");
        }
        let out = command.output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let r: Value = serde_json::from_slice(&out.stdout).unwrap();
        let want = explicit.map_or_else(tenferro_cpu::available_parallelism, |_| 2);
        assert_eq!(r["thread_hint"], want);
        assert_eq!(
            r["environment"]["RAYON_NUM_THREADS"],
            explicit.map(Value::from).unwrap_or(Value::Null)
        );
    }
}
