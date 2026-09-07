//! `solve <config.toml|config.json>` — run a finite-T imaginary-time sweep and write JSON.

use std::path::Path;
use std::process::exit;

use thermal_imps_purification::config::RunConfig;
use thermal_imps_purification::runner::{run_sweep, write_result};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        let prog = args.first().map(String::as_str).unwrap_or("solve");
        eprintln!("usage: {prog} <config.toml|config.json>");
        exit(2);
    }
    let cfg_path = &args[1];

    let format = Path::new(cfg_path)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    let parse = match format.as_deref() {
        Some("toml") => RunConfig::from_toml_str,
        Some("json") => RunConfig::from_json_str,
        _ => {
            eprintln!("error: invalid config {cfg_path}: supported formats are .toml and .json");
            exit(1);
        }
    };

    let text = match std::fs::read_to_string(cfg_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read {cfg_path}: {e}");
            exit(1);
        }
    };
    let cfg = match parse(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: invalid config {cfg_path}: {e}");
            exit(1);
        }
    };

    if cfg.checkpoint.is_some() || cfg.restart.is_some() {
        match thermal_imps_purification::solve_run::run_checkpointed_sweep(&cfg, Some(Path::new(cfg_path)))
        {
            Ok(result) => {
                let segment = result.segment.as_ref().expect("checkpoint segment");
                println!("solve: {} -> {} (start step {}, target step {}, every {} steps, restart {:?}; requested beta {}, actual beta {})",
                    cfg_path, cfg.output.path, segment.start_step, segment.completed_steps,
                    cfg.checkpoint.as_ref().unwrap().every_steps,
                    segment.source.as_ref().map(|s| s.snapshot), cfg.evolution.beta_max,
                    2.0 * segment.completed_steps as f64 * cfg.evolution.dtau);
                return;
            }
            Err(error) => {
                eprintln!("error: sweep failed for {}: {}", cfg_path, error);
                exit(1);
            }
        }
    }
    let result = match run_sweep(&cfg) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: sweep failed for {cfg_path}: {error}");
            exit(1);
        }
    };

    if let Err(e) = write_result(Path::new(&cfg.output.path), &result) {
        eprintln!("error: cannot write {}: {e}", cfg.output.path);
        exit(1);
    }

    let n = result.records.len();
    let last_beta = result.records.last().map(|r| r.beta).unwrap_or(0.0);
    println!(
        "solve: {} -> {} ({} records, beta up to {:.4}, d={})",
        cfg_path, cfg.output.path, n, last_beta, result.metadata.local_dim
    );
}
