//! Private runtime Git provenance capture.
use std::ffi::OsStr;
#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
use std::io;

pub(crate) fn git_stdout(args: &[&str]) -> Option<Vec<u8>> {
    let args: Vec<_> = args.iter().map(OsStr::new).collect();
    stdout(OsStr::new("git"), &args).ok().flatten()
}

#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
fn stdout(program: &OsStr, args: &[&OsStr]) -> io::Result<Option<Vec<u8>>> {
    let output = std::process::Command::new(program).args(args).output()?;
    Ok(output.status.success().then_some(output.stdout))
}

#[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
mod posix;
#[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
use posix::stdout;

#[cfg(all(test, unix))]
pub(crate) mod tests;
