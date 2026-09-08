#[path = "support/solve_matrix.rs"]
mod support;

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use thermal_imps_purification::config::RunConfig;
use thermal_imps_purification::runner::{read_result, SweepResult};

// HDF5 native descriptors can be inherited by concurrently launched solve processes.
static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(path)
        .output()
        .unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_finite_result(path: &Path, expected_model: &Value) -> SweepResult {
    let output = read_result(path).unwrap();
    assert_eq!(
        serde_json::to_value(&output.metadata.model).unwrap(),
        *expected_model
    );
    let encoded = serde_json::to_value(&output.metadata.model).unwrap();
    assert_eq!(encoded["observable"]["name"], "sigma_y");
    assert_eq!(encoded["site_energy"]["imag"][0][1], 0.7);
    assert!(!output.records.is_empty());
    assert!(output.records.iter().all(|record| {
        assert_eq!(record.f.is_none(), record.beta == 0.0);
        assert!(record.f.is_none_or(f64::is_finite));
        [
            record.beta,
            record.u,
            record.c,
            record.beta_f,
            record.magnetization,
        ]
        .iter()
        .all(|value| value.is_finite())
    }));
    output
}

fn phase_paths(directory: &Path, stem: &str) -> (Value, PathBuf, PathBuf) {
    let input = directory.join(stem);
    let output = directory.join(format!("{stem}.result.json"));
    let mut config = support::phase_config();
    config["output"]["path"] = json!(output);
    (config, input, output)
}

#[test]
fn uppercase_json_and_toml_dispatch_preserve_matrix_metadata() {
    let directory = tempfile::tempdir().unwrap();

    let (json_config, json_input, json_output) = phase_paths(directory.path(), "input.JSON");
    std::fs::write(
        &json_input,
        serde_json::to_vec_pretty(&json_config).unwrap(),
    )
    .unwrap();
    assert_success(&run(&json_input));
    assert_finite_result(&json_output, &json_config["model"]);

    let (toml_value, toml_input, toml_output) = phase_paths(directory.path(), "input.ToMl");
    let toml_config = RunConfig::from_json_str(&toml_value.to_string()).unwrap();
    std::fs::write(&toml_input, toml::to_string(&toml_config).unwrap()).unwrap();
    assert_success(&run(&toml_input));
    assert_finite_result(&toml_output, &toml_value["model"]);
}

#[test]
fn checkpoint_enabled_json_dispatch_publishes_complete_matrix_provenance() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let (mut config, input, output_path) = phase_paths(directory.path(), "checkpoint.json");
    let checkpoint_path = directory.path().join("checkpoint.h5");
    config["checkpoint"] = json!({"path": checkpoint_path, "every_steps": 1});
    std::fs::write(&input, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

    assert_success(&run(&input));
    let output = assert_finite_result(&output_path, &config["model"]);
    assert!(output.segment.unwrap().finished);
    assert!(checkpoint_path.exists());
}

#[test]
fn unsupported_and_missing_extensions_are_configuration_errors() {
    let directory = tempfile::tempdir().unwrap();
    for name in ["input.yaml", "input"] {
        let path = directory.path().join(name);
        let output_path = directory.path().join(format!("{name}.result.json"));
        let checkpoint_path = directory.path().join(format!("{name}.h5"));
        let mut config = support::phase_config();
        config["output"]["path"] = json!(output_path);
        config["checkpoint"] = json!({"path": checkpoint_path, "every_steps": 1});
        let bytes = serde_json::to_vec(&config).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let output = run(&path);
        assert_eq!(output.status.code(), Some(1));
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(path.to_str().unwrap()), "{stderr}");
        assert!(
            stderr.contains("toml") && stderr.contains("json"),
            "{stderr}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(!output_path.exists());
        assert!(!checkpoint_path.exists());
    }
}

#[test]
fn usage_and_read_failures_keep_their_exit_codes() {
    let no_argument = Command::new(env!("CARGO_BIN_EXE_solve")).output().unwrap();
    assert_eq!(no_argument.status.code(), Some(2));

    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("unreadable.json");
    let output = run(&missing);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot read") && stderr.contains(missing.to_str().unwrap()));
}

#[test]
fn malformed_and_invalid_configs_create_no_destinations_and_preserve_sources() {
    let directory = tempfile::tempdir().unwrap();
    let make = |name: &str| {
        let mut config = support::phase_config();
        config["output"]["path"] = json!(directory.path().join(format!("{name}.result.json")));
        config["checkpoint"] = json!({
            "path": directory.path().join(format!("{name}.h5")),
            "every_steps": 1
        });
        config
    };
    let invalid_json = make("invalid.json");
    let invalid_toml = make("invalid.toml");
    let invalid_toml = RunConfig::from_json_str(&invalid_toml.to_string()).unwrap();
    let mut imag_null = make("imag-null.json");
    imag_null["model"]["two_site_h"]["imag"] = Value::Null;
    let cases = [
        ("invalid.json", format!("{} trailing", invalid_json)),
        (
            "invalid.toml",
            format!("{}\n[", toml::to_string(&invalid_toml).unwrap()),
        ),
        (
            "duplicate.json",
            make("duplicate.json").to_string().replacen(
                "\"version\":1",
                "\"version\":1,\"version\":1",
                1,
            ),
        ),
        (
            "unknown.json",
            make("unknown.json").to_string().replacen(
                "\"version\":1",
                "\"surprise\":true,\"version\":1",
                1,
            ),
        ),
        (
            "include-exact.json",
            make("include-exact.json")
                .to_string()
                .replace("\"include_exact\":false", "\"include_exact\":true"),
        ),
        ("imag-null.json", imag_null.to_string()),
    ];
    for (name, text) in cases {
        let input = directory.path().join(name);
        std::fs::write(&input, &text).unwrap();
        let output = run(&input);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(std::fs::read_to_string(&input).unwrap(), text);
        assert!(!directory
            .path()
            .join(format!("{name}.result.json"))
            .exists());
        assert!(!directory.path().join(format!("{name}.h5")).exists());
    }
}

#[test]
fn null_preset_only_keys_create_no_destinations_and_preserve_sources() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for key in ["j", "g", "gamma", "h", "j1", "j2"] {
        let input = directory.path().join(format!("null-{key}.json"));
        let output_path = directory.path().join(format!("null-{key}.result.json"));
        let checkpoint_path = directory.path().join(format!("null-{key}.h5"));
        let mut config = support::phase_config();
        config["model"][key] = Value::Null;
        config["output"]["path"] = json!(output_path);
        config["checkpoint"] = json!({"path": checkpoint_path, "every_steps": 1});
        let bytes = serde_json::to_vec_pretty(&config).unwrap();
        std::fs::write(&input, &bytes).unwrap();

        let failed = run(&input);
        let stderr = String::from_utf8(failed.stderr).unwrap();
        let source_preserved = std::fs::read(&input).unwrap() == bytes;
        let json_absent = !output_path.exists();
        let checkpoint_absent = !checkpoint_path.exists();
        if failed.status.code() != Some(1)
            || !stderr.contains(key)
            || !stderr.contains("preset-only key")
            || !source_preserved
            || !json_absent
            || !checkpoint_absent
        {
            failures.push(format!(
                "{key}: status={:?}, stderr={stderr:?}, source_preserved={source_preserved}, json_absent={json_absent}, checkpoint_absent={checkpoint_absent}",
                failed.status.code()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn checkpoint_input_alias_is_rejected_without_changing_the_config() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("aliased.json");
    let checkpoint = directory.path().join("aliased.h5");
    let mut config = support::phase_config();
    config["output"]["path"] = json!(input);
    config["checkpoint"] = json!({"path": checkpoint, "every_steps": 1});
    let bytes = serde_json::to_vec_pretty(&config).unwrap();
    std::fs::write(&input, &bytes).unwrap();

    let output = run(&input);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid checkpoint run"));
    assert_eq!(std::fs::read(&input).unwrap(), bytes);
    assert!(!checkpoint.exists());
}

#[test]
fn incompatible_json_restart_preserves_source_and_creates_no_destinations() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let (mut source_config, source_input, source_output) =
        phase_paths(directory.path(), "source.json");
    let source_checkpoint = directory.path().join("source.h5");
    source_config["checkpoint"] = json!({"path": source_checkpoint, "every_steps": 1});
    std::fs::write(
        &source_input,
        serde_json::to_vec_pretty(&source_config).unwrap(),
    )
    .unwrap();
    assert_success(&run(&source_input));
    assert!(source_output.exists());
    let source_bytes = std::fs::read(&source_checkpoint).unwrap();

    let destination_input = directory.path().join("resume.JSON");
    let destination_output = directory.path().join("resume.result.json");
    let destination_checkpoint = directory.path().join("resume.h5");
    let mut resume = support::phase_config();
    resume["evolution"]["beta_max"] = json!(0.4);
    resume["evolution"]["dtau"] = json!(0.025);
    resume["output"]["path"] = json!(destination_output);
    resume["checkpoint"] = json!({"path": destination_checkpoint, "every_steps": 1});
    resume["restart"] = json!({"path": source_checkpoint});
    let input_bytes = serde_json::to_vec_pretty(&resume).unwrap();
    std::fs::write(&destination_input, &input_bytes).unwrap();

    let failed = run(&destination_input);
    assert_eq!(failed.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failed.stderr).contains("restart configuration mismatch"));
    assert_eq!(std::fs::read(&source_checkpoint).unwrap(), source_bytes);
    assert_eq!(std::fs::read(&destination_input).unwrap(), input_bytes);
    assert!(!destination_output.exists());
    assert!(!destination_checkpoint.exists());
}
