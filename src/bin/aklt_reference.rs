//! Fixed AKLT reference worker with explicit pinned continuation; no automatic retry.
use anyhow::{bail, Context, Result};
use std::path::Path;
use thermal_imps_purification::aklt_reference::output::{
    finalize_interrupted_directory, run_to_directory, verify_directory,
};
use thermal_imps_purification::aklt_reference::{
    parse_worker_request, preflight_worker, v3, v3_output, AkltWorkerRequest,
};
fn request(path: &str) -> Result<AkltWorkerRequest> {
    parse_worker_request(&std::fs::read(path).with_context(|| format!("read request {path}"))?)
        .with_context(|| format!("parse strict request {path}"))
}
fn execute(args: &[String]) -> Result<serde_json::Value> {
    match args {
        [flag] if flag=="--runtime-info"=>Ok(serde_json::to_value(v3::capture_runtime()?)?),
        [flag,path] if flag=="--preflight"=>preflight_worker(&request(path)?),
        [flag,path,destination] if flag=="--preflight"=>match request(path)? {
            AkltWorkerRequest::StepSampled(r)=>thermal_imps_purification::aklt_reference::v5_input::preflight_start(&r,Path::new(destination)),
            _=>bail!("destination-bound preflight requires step-sampled v5"),
        },
        [flag,path] if flag=="--verify"=>verify_directory(Path::new(path)),
        [flag,path] if flag=="--verify-svd-failure"=>thermal_imps_purification::aklt_reference::v4::verify_failure_file(Path::new(path)),
        [flag,path,reason] if flag=="--finalize-interrupted"=>finalize_interrupted_directory(Path::new(path),reason),
        [input,output] if !input.starts_with('-')=>match request(input)? {
            AkltWorkerRequest::Legacy(r)=>run_to_directory(&r,Path::new(output)),
            AkltWorkerRequest::V3(r)=>v3_output::run_to_directory(&r,Path::new(output)),
            AkltWorkerRequest::Resume(r)=>thermal_imps_purification::aklt_reference::v4::run_to_directory(&r,Path::new(output)),
            AkltWorkerRequest::StepSampled(_)=>bail!("step-sampled scalar publisher is not yet available"),
        },
        _=>bail!("usage: aklt_reference --preflight REQUEST_JSON | REQUEST_JSON OUTPUT_DIRECTORY | --verify OUTPUT_DIRECTORY | --verify-svd-failure FILE | --finalize-interrupted OUTPUT_DIRECTORY REASON"),
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
