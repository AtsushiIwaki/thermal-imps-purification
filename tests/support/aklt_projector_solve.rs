use serde_json::{json, Value};
use std::path::Path;

use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions,
    LoadedItebdCheckpoint,
};
use thermal_imps_purification::runner::Record;

pub fn config(dir: &Path, name: &str, kind: &str, order: u8, beta: f64) -> Value {
    json!({
        "model":{"type":kind},
        "evolution":{"dtau":0.05,"beta_max":beta,
            "record_every_beta":0.1,"trotter_order":order},
        "truncation":{"epsilon":1e-12,"max_bond":32},
        "run":{"canonicalize_every":1},
        "output":{"path":dir.join(format!("{name}.result.json")),"include_exact":false}
    })
}

pub fn write_input(value: &Value, path: &Path, format: &str) {
    let text = match format {
        "json" => serde_json::to_string_pretty(value).unwrap(),
        "toml" => toml::to_string(value).unwrap(),
        _ => panic!("unsupported test input format {format}"),
    };
    std::fs::write(path, text).unwrap();
}

pub fn run_cli(path: &Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(path)
        .output()
        .unwrap()
}

pub fn close(a: f64, b: f64, tolerance: f64) {
    assert!(a.is_finite() && b.is_finite());
    assert!(
        (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0),
        "actual={a:.16e} expected={b:.16e} tolerance={tolerance:e}"
    );
}

#[allow(dead_code)]
pub fn latest(path: &Path) -> LoadedItebdCheckpoint {
    let index = list_itebd_checkpoints(path).unwrap().last().unwrap().index;
    load_itebd_checkpoint(path, index, &ItebdCheckpointLoadOptions::default()).unwrap()
}

#[allow(dead_code)]
pub fn records_close(a: &Record, b: &Record) {
    close(a.beta, b.beta, 1e-10);
    close(a.u, b.u, 1e-10);
    close(a.c, b.c, 1e-10);
    close(a.f, b.f, 1e-10);
    close(a.magnetization, b.magnetization, 1e-10);
    assert_eq!(a.max_bond, b.max_bond);
}
