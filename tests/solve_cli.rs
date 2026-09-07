use std::process::Command;

use thermal_imps_purification::config::TrotterOrder;
use thermal_imps_purification::runner::read_result;

#[test]
fn solve_writes_parseable_json() {
    let dir = std::env::temp_dir().join("imps_solve_cli_test");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("tfim.toml");
    let out_path = dir.join("out.json");

    let toml = format!(
        "[model]\ntype = \"tfim\"\nj = 1.0\ng = 1.0\n\
         [evolution]\ndtau = 0.02\nbeta_max = 0.2\nrecord_every_beta = 0.1\n\
         [truncation]\nepsilon = 1e-10\nmax_bond = 16\n\
         [output]\npath = \"{}\"\ninclude_exact = true\n",
        out_path.to_string_lossy().replace('\\', "\\\\")
    );
    std::fs::write(&cfg_path, toml).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(&cfg_path)
        .status()
        .expect("run solve");
    assert!(status.success(), "solve exited with {status}");

    let res = read_result(&out_path).expect("parse output json");
    assert!(!res.records.is_empty());
    assert_eq!(res.metadata.local_dim, 2);
    assert_eq!(res.metadata.evolution.trotter_order, TrotterOrder::Second);
    assert!(res.records[0].exact.is_some());
}

#[test]
fn solve_rejects_bad_config() {
    let dir = std::env::temp_dir().join("imps_solve_cli_test");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("bad.toml");
    // Unsupported Trotter order → validation error → nonzero exit.
    std::fs::write(
        &cfg_path,
        "[model]\ntype = \"tfim\"\nj = 1.0\ng = 1.0\n\
         [evolution]\ndtau = 0.01\ntrotter_order = 4\nbeta_max = 0.2\nrecord_every_beta = 0.1\n\
         [truncation]\nepsilon = 1e-10\n\
         [output]\npath = \"unused.json\"\ninclude_exact = false\n",
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(&cfg_path)
        .status()
        .expect("run solve");
    assert!(!status.success(), "expected nonzero exit on bad config");
}

#[test]
fn solve_reports_specific_heat_sweep_failure_without_writing_output() {
    let dir = std::env::temp_dir().join("imps_solve_cli_sweep_failure_test");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("noncanonical.toml");
    let out_path = dir.join("must_not_exist.json");
    if out_path.exists() {
        std::fs::remove_file(&out_path).unwrap();
    }
    let toml = format!(
        "[model]\ntype = \"tfim\"\nj = 1.0\ng = 1.0\n\
         [evolution]\ndtau = 0.01\nbeta_max = 1.0\nrecord_every_beta = 1.0\n\
         [truncation]\nepsilon = 1e-12\nmax_bond = 48\n\
         [run]\ncanonicalize_every = 1000\n\
         [output]\npath = \"{}\"\ninclude_exact = false\n",
        out_path.to_string_lossy().replace('\\', "\\\\")
    );
    std::fs::write(&cfg_path, toml).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(&cfg_path)
        .output()
        .expect("run solve");
    assert!(!output.status.success(), "expected sweep failure");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains(&format!("error: sweep failed for {}:", cfg_path.display())),
        "stderr = {stderr}"
    );
    assert!(
        stderr.contains("specific-heat tail did not converge"),
        "stderr = {stderr}"
    );
    assert!(
        !out_path.exists(),
        "failed sweep must not write partial output"
    );
}
