use super::*;
use std::io;
use std::path::{Path, PathBuf};

fn capture(program: &str, args: &[&str]) -> io::Result<Option<Vec<u8>>> {
    stdout(
        OsStr::new(program),
        &args.iter().map(OsStr::new).collect::<Vec<_>>(),
    )
}

#[test]
fn captures_success_and_discards_nonzero_stdout() {
    assert_eq!(
        capture("/bin/sh", &["-c", "printf 'hello\\n'"]).unwrap(),
        Some(b"hello\n".to_vec())
    );
    assert_eq!(
        capture("/bin/sh", &["-c", "printf partial; exit 7"]).unwrap(),
        None
    );
}

#[test]
fn reports_spawn_and_input_errors() {
    assert_eq!(
        capture("/definitely-missing-git-process-test", &[])
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(
        capture("bad\0program", &[]).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        capture("/bin/sh", &["bad\0argument"]).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn drains_stdout_larger_than_pipe_capacity() {
    let bytes = capture(
        "/bin/sh",
        &[
            "-c",
            "i=0; while [ $i -lt 20000 ]; do printf '0123456789abcdef'; i=$((i+1)); done",
        ],
    )
    .unwrap()
    .unwrap();
    assert_eq!(bytes, b"0123456789abcdef".repeat(20000));
}

fn marker(name: &str) -> Option<PathBuf> {
    let prefix = format!("{name}=");
    std::env::args().find_map(|arg| arg.strip_prefix(&prefix).map(PathBuf::from))
}

#[test]
fn inherits_cwd_environment_and_path_in_isolated_process() {
    if let Some(dir) = marker("inheritance") {
        assert_eq!(
            capture("fixture-command", &[]).unwrap().unwrap(),
            dir.as_os_str().as_encoded_bytes()
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    write_executable(&dir.path().join("fixture-command"), "#!/bin/sh\nprintf '%s' \"$GIT_PROCESS_TEST_VALUE\"\n[ \"$PWD\" = \"$GIT_PROCESS_TEST_VALUE\" ]\n");
    let path = dir.path().canonicalize().unwrap();
    isolated_test(
        "git_process::tests::inherits_cwd_environment_and_path_in_isolated_process",
        "inheritance",
        &path,
        true,
    );
}

#[cfg(unix)]
fn write_executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}

#[cfg(target_os = "macos")]
fn holds_path(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    // Independent test observation of this child's open descriptors, never parent mutation.
    let limit = unsafe { libc::sysconf(libc::_SC_OPEN_MAX) };
    assert!(limit > 0);
    (0..limit as i32).any(|fd| {
        let mut buffer = [0u8; libc::PATH_MAX as usize];
        let ok = unsafe { libc::fcntl(fd, libc::F_GETPATH, buffer.as_mut_ptr()) } == 0;
        ok && buffer.split(|b| *b == 0).next().unwrap() == path.as_os_str().as_bytes()
    })
}

#[test]
fn subprocess_entry() {
    #[cfg(target_os = "macos")]
    if let Some(target) = marker("target") {
        let inherited = holds_path(&target);
        if let Some(dir) = marker("hold") {
            std::fs::write(
                dir.join("ready"),
                if inherited { "inherited" } else { "isolated" },
            )
            .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while !dir.join("release").exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "parent never released child"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        } else {
            std::process::exit(if inherited { 42 } else { 0 });
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn open_hdf5_file_is_not_inherited_and_reopens_before_child_exit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checkpoint.h5");
    let file = hdf5_metno::File::create(&path).unwrap();
    let path = path.canonicalize().unwrap();
    assert!(
        holds_path(&path),
        "fixture must have an actual open HDF5 descriptor"
    );
    let child_dir = dir.path().to_owned();
    let child_path = path.clone();
    let join = std::thread::spawn(move || {
        let exe = std::env::current_exe().unwrap();
        let args = [
            "--exact".to_owned(),
            "git_process::tests::subprocess_entry".to_owned(),
            "--skip".to_owned(),
            format!("target={}", child_path.display()),
            "--skip".to_owned(),
            format!("hold={}", child_dir.display()),
        ];
        stdout(
            exe.as_os_str(),
            &args.iter().map(OsStr::new).collect::<Vec<_>>(),
        )
    });
    struct Release {
        dir: PathBuf,
        join: Option<std::thread::JoinHandle<io::Result<Option<Vec<u8>>>>>,
    }
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(self.dir.join("release"), "release");
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }
    let mut release = Release {
        dir: dir.path().to_owned(),
        join: Some(join),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !dir.path().join("ready").exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "child did not become ready"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let descriptors = std::fs::read_to_string(dir.path().join("ready")).unwrap();
    file.close().unwrap();
    let reopened = hdf5_metno::File::open_rw(&path);
    assert!(reopened.is_ok(), "HDF5 reopen before child exit failed; child descriptor observation={descriptors}: {reopened:?}");
    assert_eq!(
        descriptors, "isolated",
        "child retained the target HDF5 path"
    );
    assert!(
        !release.join.as_ref().unwrap().is_finished(),
        "child exited before reopen"
    );
    std::fs::write(dir.path().join("release"), "release").unwrap();
    assert!(release
        .join
        .take()
        .unwrap()
        .join()
        .unwrap()
        .unwrap()
        .is_some());
}

pub(crate) fn revision_contract(test_name: &str, revision: fn() -> Option<String>, short: bool) {
    if let Some(dir) = marker("revision") {
        let path = dir.join("checkpoint.h5");
        let file = hdf5_metno::File::create(&path).unwrap();
        let expected_args = if short {
            "rev-parse --short HEAD"
        } else {
            "rev-parse HEAD"
        };
        std::fs::write(dir.join("expected-args"), expected_args).unwrap();
        for (output, expected) in [
            (b"  abc123\n".as_slice(), Some("abc123")),
            (b" \n\t", if short { None } else { Some("") }),
            (b" \xff\n", if short { None } else { Some("\u{fffd}") }),
        ] {
            std::fs::write(dir.join("output"), output).unwrap();
            assert_eq!(revision().as_deref(), expected);
        }
        std::fs::write(dir.join("exit"), "7").unwrap();
        assert_eq!(revision(), None);
        file.close().unwrap();
        std::fs::remove_file(dir.join("git")).unwrap();
        assert_eq!(revision(), None);
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().canonicalize().unwrap();
    let scan = if cfg!(target_os = "macos") {
        "\"$GIT_PROCESS_TEST_EXE\" --exact git_process::tests::subprocess_entry --skip \"target=$GIT_PROCESS_TEST_DIR/checkpoint.h5\" >/dev/null || exit $?\n"
    } else {
        ""
    };
    write_executable(&path.join("git"), &format!("#!/bin/sh\n[ \"$*\" = \"$(/bin/cat \"$GIT_PROCESS_TEST_DIR/expected-args\")\" ] || exit 41\n{scan}/bin/cat \"$GIT_PROCESS_TEST_DIR/output\"\nexit $(/bin/cat \"$GIT_PROCESS_TEST_DIR/exit\")\n"));
    std::fs::write(path.join("exit"), "0").unwrap();
    isolated_test(test_name, "revision", &path, true);
}

#[test]
fn revision_changes_when_disposable_repository_head_changes() {
    if let Some(dir) = marker("repository") {
        for message in ["first", "second"] {
            assert!(capture(
                "git",
                &[
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "--allow-empty",
                    "-m",
                    message
                ]
            )
            .unwrap()
            .is_some());
            let expected = capture("git", &["rev-parse", "HEAD"]).unwrap().unwrap();
            assert_eq!(git_stdout(&["rev-parse", "HEAD"]).unwrap(), expected);
            if message == "first" {
                std::fs::write(dir.join("first"), &expected).unwrap();
            } else {
                assert_ne!(std::fs::read(dir.join("first")).unwrap(), expected);
            }
        }
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    assert!(stdout(
        OsStr::new("git"),
        &[OsStr::new("init"), dir.path().as_os_str()]
    )
    .unwrap()
    .is_some());
    isolated_test(
        "git_process::tests::revision_changes_when_disposable_repository_head_changes",
        "repository",
        dir.path(),
        false,
    );
}

fn isolated_test(test: &str, marker: &str, dir: &Path, fixture_path: bool) {
    let exe = std::env::current_exe().unwrap();
    let role = format!("{marker}={}", dir.display());
    let script = r#"cd "$1" || exit 90
export GIT_PROCESS_TEST_DIR="$1" GIT_PROCESS_TEST_VALUE="$1" GIT_PROCESS_TEST_EXE="$2"
if [ "$5" = fixture ]; then export PATH="$1"; fi
exec "$2" --exact "$3" --skip "$4" > "$1/child.log" 2>&1
"#;
    let result = stdout(
        OsStr::new("/bin/sh"),
        &[
            OsStr::new("-c"),
            OsStr::new(script),
            OsStr::new("sh"),
            dir.as_os_str(),
            exe.as_os_str(),
            OsStr::new(test),
            OsStr::new(&role),
            OsStr::new(if fixture_path { "fixture" } else { "inherited" }),
        ],
    )
    .unwrap();
    assert!(
        result.is_some(),
        "isolated test {test} failed: {}",
        std::fs::read_to_string(dir.join("child.log")).unwrap_or_default()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn handles_closed_parent_standard_descriptors() {
    if marker("closed-stdio").is_some() {
        for fd in 0..=2 {
            unsafe {
                libc::close(fd);
            }
        }
        let result = capture("/bin/sh", &["-c", "printf pipe-ok"]);
        std::process::exit(
            if matches!(result, Ok(Some(ref bytes)) if bytes == b"pipe-ok") {
                0
            } else {
                43
            },
        );
    }
    let dir = tempfile::tempdir().unwrap();
    isolated_test(
        "git_process::tests::handles_closed_parent_standard_descriptors",
        "closed-stdio",
        dir.path(),
        false,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn resets_sigpipe_to_default_in_child() {
    // Rust ignores SIGPIPE; a child must terminate here rather than reach exit 0.
    assert_eq!(
        capture("/bin/sh", &["-c", "kill -PIPE $$; exit 0"]).unwrap(),
        None
    );
}

/// Run a test body in a process without unrelated concurrent libtest descriptors.
/// The return value tells the caller whether it is already inside that process.
pub(crate) fn in_isolated_test(test: &str) -> bool {
    if marker("isolated-body").is_some() {
        return true;
    }
    let dir = tempfile::tempdir().unwrap();
    isolated_test(test, "isolated-body", dir.path(), false);
    false
}

#[cfg(target_os = "macos")]
thread_local! {
    static SPAWN_FILE_ACTION_GATE: std::cell::RefCell<Option<(PathBuf, PathBuf)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(target_os = "macos")]
pub(super) fn configure_spawn_file_actions(
    actions: &mut libc::posix_spawn_file_actions_t,
) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let Some((marker, fifo)) = SPAWN_FILE_ACTION_GATE.with(|gate| gate.borrow_mut().take()) else {
        return Ok(());
    };
    let marker = CString::new(marker.as_os_str().as_bytes())?;
    let fifo = CString::new(fifo.as_os_str().as_bytes())?;
    for (path, flags) in [
        (
            marker.as_c_str(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        ),
        (fifo.as_c_str(), libc::O_RDONLY),
        (c"/dev/null", libc::O_RDONLY),
    ] {
        // SAFETY: initialized caller-owned actions; addopen copies its valid path.
        let error = unsafe {
            libc::posix_spawn_file_actions_addopen(actions, 0, path.as_ptr(), flags, 0o600)
        };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn hdf5_close_reopen_is_safe_during_native_spawn() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    if !in_isolated_test("git_process::tests::hdf5_close_reopen_is_safe_during_native_spawn") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checkpoint.h5");
    let file = hdf5_metno::File::create(&path).unwrap();
    let target = path.canonicalize().unwrap();
    let fifo = dir.path().join("spawn.fifo");
    let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
    let entered = dir.path().join("entered");
    let worker_dir = dir.path().to_owned();
    let worker_fifo = fifo.clone();
    let worker_target = target.clone();
    let spawn = std::thread::spawn(move || {
        SPAWN_FILE_ACTION_GATE.with(|gate| *gate.borrow_mut() = Some((entered, worker_fifo)));
        let exe = std::env::current_exe().unwrap();
        let args = [
            "--exact".to_owned(),
            "git_process::tests::subprocess_entry".to_owned(),
            "--skip".to_owned(),
            format!("target={}", worker_target.display()),
            "--skip".to_owned(),
            format!("hold={}", worker_dir.display()),
        ];
        stdout(
            exe.as_os_str(),
            &args.iter().map(OsStr::new).collect::<Vec<_>>(),
        )
    });
    struct WindowGuard {
        dir: PathBuf,
        spawn: Option<std::thread::JoinHandle<io::Result<Option<Vec<u8>>>>>,
        close: Option<std::thread::JoinHandle<Result<(), String>>>,
        fifo_writer: Option<std::fs::File>,
    }
    impl WindowGuard {
        fn release_spawn(&mut self) {
            // O_RDWR never waits for a counterpart; this also safely releases the
            // kernel open on assertion failure before the reader reaches the FIFO.
            self.fifo_writer = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(self.dir.join("spawn.fifo"))
                .ok();
        }
    }
    impl Drop for WindowGuard {
        fn drop(&mut self) {
            self.release_spawn();
            let _ = std::fs::write(self.dir.join("release"), "release");
            if let Some(spawn) = self.spawn.take() {
                let _ = spawn.join();
            }
            if let Some(close) = self.close.take() {
                let _ = close.join();
            }
        }
    }
    let mut guard = WindowGuard {
        dir: dir.path().to_owned(),
        spawn: Some(spawn),
        close: None,
        fifo_writer: None,
    };
    let deadline = Instant::now() + Duration::from_secs(20);
    while !dir.path().join("entered").exists() {
        assert!(
            Instant::now() < deadline,
            "native spawn never entered file actions"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        !dir.path().join("ready").exists(),
        "child executed before FIFO release"
    );
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    guard.close = Some(std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = file
            .close()
            .map_err(|e| format!("close: {e}"))
            .and_then(|()| hdf5_metno::File::open_rw(target).map_err(|e| format!("reopen: {e}")))
            .and_then(|file| file.close().map_err(|e| format!("reopened close: {e}")));
        let _ = done_tx.send(result.clone());
        result
    }));
    started_rx.recv_timeout(Duration::from_secs(20)).unwrap();
    // Give the real close/reopen request an interval while native spawn is held.
    // Without coordination it completes with errno 35; with coordination it is
    // blocked until release. The substantive assertion below is actual HDF5 I/O.
    let early_result = done_rx.recv_timeout(Duration::from_millis(250));
    guard.release_spawn();
    let result = guard.close.take().unwrap().join().unwrap();
    assert!(result.is_ok(), "real HDF5 close/reopen raced native spawn: {result:?}; during-spawn result={early_result:?}");
    assert!(
        matches!(early_result, Err(mpsc::RecvTimeoutError::Timeout)),
        "HDF5 operations completed before spawn was released: {early_result:?}"
    );
    while !dir.path().join("ready").exists() {
        assert!(
            Instant::now() < deadline,
            "child did not reach post-exec handshake"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        std::fs::read_to_string(dir.path().join("ready")).unwrap(),
        "isolated"
    );
    assert!(
        !guard.spawn.as_ref().unwrap().is_finished(),
        "child exited before close/reopen completed"
    );
    std::fs::write(dir.path().join("release"), "release").unwrap();
    assert!(guard
        .spawn
        .take()
        .unwrap()
        .join()
        .unwrap()
        .unwrap()
        .is_some());
}
