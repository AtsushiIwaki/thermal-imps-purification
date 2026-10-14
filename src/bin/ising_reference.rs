//! Fixed mixed-field reference worker; no restart or automatic retry.
use anyhow::{bail, Context, Result};
use std::path::Path;
use thermal_imps_purification::ising_reference::output::{
    finalize_interrupted_directory, run_to_directory, verify_directory,
};
use thermal_imps_purification::ising_reference::{preflight, ReferenceRequest};
fn request(path: &str) -> Result<ReferenceRequest> {
    serde_json::from_slice(&std::fs::read(path).with_context(|| format!("read request {path}"))?)
        .with_context(|| format!("parse strict request {path}"))
}
fn execute(args: &[String]) -> Result<serde_json::Value> {
    match args {
        [flag,path] if flag=="--preflight"=>preflight(&request(path)?),
        [flag,path] if flag=="--verify"=>verify_directory(Path::new(path)),
        [flag,path,reason] if flag=="--finalize-interrupted"=>finalize_interrupted_directory(Path::new(path),reason),
        [input,output] if !input.starts_with('-')=>run_to_directory(&request(input)?,Path::new(output)),
        _=>bail!("usage: ising_reference --preflight REQUEST_JSON | REQUEST_JSON OUTPUT_DIRECTORY | --verify OUTPUT_DIRECTORY | --finalize-interrupted OUTPUT_DIRECTORY REASON"),
    }
}
fn main() {
    match execute(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(report) => println!("{}", report),
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(1);
        }
    }
}
