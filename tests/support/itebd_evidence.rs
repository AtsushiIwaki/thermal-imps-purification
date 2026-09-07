//! Shared test-only evidence output. Reserve destinations before doing expensive work.
use serde_json::{json, Value};
use std::io::Write;

pub fn destination() -> std::io::Result<Option<std::fs::File>> {
    std::env::var_os("ITEBD_EVIDENCE_PATH")
        .map(|path| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
        })
        .transpose()
}
pub fn context() -> Value {
    let command = |program: &str, args: &[&str]| {
        let output = std::process::Command::new(program)
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    };
    json!({"source_commit": command("git", &["rev-parse", "HEAD"]),
        "rustc": command("rustc", &["-Vv"]), "host": command("uname", &["-a"]),
        "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
        "pid": std::process::id()})
}
pub fn emit(mut destination: Option<std::fs::File>, report: &Value) {
    let data = serde_json::to_string_pretty(report).unwrap();
    if let Some(file) = destination.as_mut() {
        file.write_all(data.as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
        file.sync_all().unwrap();
    }
    println!("{data}");
}

pub fn finite(field: &str, values: &[f64]) -> std::io::Result<()> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("nonfinite evidence scalar in {field}: {values:?}"),
        ))
    }
}
