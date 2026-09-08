use super::*;
use std::fs;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn cfg(dir: &Path, name: &str, restart: bool) -> RunConfig {
    let source = if restart {
        format!("[restart]\npath='{}'", dir.join("child.h5").display())
    } else {
        String::new()
    };
    RunConfig::from_toml_str(&format!("[model]\ntype='tfim'\nj=1.0\ng=0.7\n[evolution]\ndtau=0.01\nbeta_max=0.16\nrecord_every_beta=0.04\n[truncation]\nepsilon=1e-12\nmax_bond=16\n[output]\npath='{}'\n[checkpoint]\npath='{}'\nevery_steps=3\n{source}", dir.join(format!("{name}.json")).display(),dir.join(format!("{name}.h5")).display())).unwrap()
}

#[test]
#[ignore = "subprocess helper invoked by quiescent interruption test"]
fn checkpoint_child() {
    let dir = PathBuf::from(std::env::var_os("SOLVE_TEST_CHILD_DIR").unwrap());
    run_with_checkpoint_observer(&cfg(&dir, "child", false), None, |step| {
        if step == 3 {
            fs::write(dir.join("ready"), b"checkpoint complete").unwrap();
            let _ = std::io::stdin().read_exact(&mut [0u8]);
            panic!("parent must terminate at the checkpoint handshake");
        }
        Ok(())
    })
    .unwrap();
}
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn quiescent_process_interruption_restarts_from_last_complete_snapshot() {
    #[cfg(unix)]
    if !crate::git_process::tests::in_isolated_test(
        "solve_run::tests::quiescent_process_interruption_restarts_from_last_complete_snapshot",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let log = fs::File::create(dir.path().join("child.log")).unwrap();
    let mut child = ChildGuard(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "solve_run::tests::checkpoint_child",
                "--ignored",
                "--nocapture",
            ])
            .env("SOLVE_TEST_CHILD_DIR", dir.path())
            .stdin(Stdio::piped())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while !dir.path().join("ready").exists() {
        assert!(
            Instant::now() < deadline,
            "child timeout: {}",
            fs::read_to_string(dir.path().join("child.log")).unwrap()
        );
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "child exited: {}",
            fs::read_to_string(dir.path().join("child.log")).unwrap()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    child.0.kill().unwrap();
    child.0.wait().unwrap();
    let before = fs::read(dir.path().join("child.h5")).unwrap();
    let partial = crate::runner::read_result(&dir.path().join("child.json")).unwrap();
    assert_eq!(
        partial
            .segment
            .as_ref()
            .unwrap()
            .checkpoint
            .as_ref()
            .unwrap()
            .completed_steps,
        3
    );
    assert!(!partial.segment.unwrap().finished);
    let resumed = run_checkpointed_sweep(&cfg(dir.path(), "resumed", true), None).unwrap();
    let whole = run_checkpointed_sweep(&cfg(dir.path(), "whole", false), None).unwrap();
    assert_eq!(fs::read(dir.path().join("child.h5")).unwrap(), before);
    assert_eq!(resumed.segment.as_ref().unwrap().start_step, 3);
    for (actual, expected) in resumed
        .records
        .iter()
        .zip(whole.records.iter().filter(|r| r.beta > 0.06))
    {
        for (a, b) in [
            (actual.beta, expected.beta),
            (actual.u, expected.u),
            (actual.c, expected.c),
            (actual.f.unwrap(), expected.f.unwrap()),
            (actual.magnetization, expected.magnetization),
        ] {
            assert!(
                (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0),
                "{a} != {b}"
            );
        }
    }
    assert_eq!(resumed.records.len(), 3);
}

#[test]
fn regular_file_destination_parent_maps_path_source_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = cfg(dir.path(), "blocked_parent", false);
    let parent = dir.path().join("ordinary-file");
    fs::write(&parent, b"not a directory").unwrap();
    let checkpoint = parent.join("trajectory.h5");
    cfg.checkpoint.as_mut().unwrap().path = checkpoint.display().to_string();

    let error = run_checkpointed_sweep(&cfg, None).unwrap_err();
    let SolveRunError::Path { path, source } = error else {
        panic!("{error}")
    };
    assert_eq!(path, checkpoint);
    assert_eq!(source.kind(), std::io::ErrorKind::NotADirectory);
    assert!(source.raw_os_error().is_some());
    assert!(!Path::new(&cfg.output.path).exists());
    assert_eq!(fs::read(&parent).unwrap(), b"not a directory");
}

#[test]
fn real_append_collision_retains_context_and_previous_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(dir.path(), "collision", false);
    let path = Path::new(&cfg.checkpoint.as_ref().unwrap().path);
    let error = run_with_checkpoint_observer(&cfg, None, |step| {
        if step == 0 {
            let f = hdf5_metno::File::open_rw(path).unwrap();
            f.group("states").unwrap().create_group("000001").unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(
        error,
        SolveRunError::Checkpoint(ItebdCheckpointError::Io { .. })
    ));
    let text = error.to_string();
    assert!(
        text.contains(path.to_str().unwrap())
            && text.contains("snapshot 1")
            && text.contains("states/000001"),
        "{text}"
    );
    assert!(text.contains("name already exists"), "{text}");
    assert_eq!(list_itebd_checkpoints(path).unwrap().len(), 1);
    assert!(
        !crate::runner::read_result(Path::new(&cfg.output.path))
            .unwrap()
            .segment
            .unwrap()
            .finished
    );
}

#[test]
fn final_flush_open_failure_does_not_publish_finished() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(dir.path(), "finish", false);
    let path = Path::new(&cfg.checkpoint.as_ref().unwrap().path);
    let backup = dir.path().join("preserved.h5");
    let error = run_with_checkpoint_observer(&cfg, None, |step| {
        if step == 8 {
            fs::rename(path, &backup).unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    let text = error.to_string();
    assert!(matches!(
        error,
        SolveRunError::Checkpoint(ItebdCheckpointError::Io { .. })
    ));
    assert!(
        text.contains(path.to_str().unwrap()) && text.contains("No such file"),
        "{text}"
    );
    assert_eq!(
        list_itebd_checkpoints(&backup)
            .unwrap()
            .last()
            .unwrap()
            .completed_steps,
        8
    );
    assert!(
        !crate::runner::read_result(Path::new(&cfg.output.path))
            .unwrap()
            .segment
            .unwrap()
            .finished
    );
}

#[test]
fn actual_json_replacement_failure_maps_through_solve_and_preserves_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(dir.path(), "json_failure", false);
    let json = Path::new(&cfg.output.path);
    let backup = dir.path().join("prior.json");
    let checkpoint = Path::new(&cfg.checkpoint.as_ref().unwrap().path);
    let error = run_with_checkpoint_observer(&cfg, None, |step| {
        if step == 0 {
            fs::rename(json, &backup).unwrap();
            fs::create_dir(json).unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    let SolveRunError::Publication(ref publication) = error else {
        panic!("{error}");
    };
    assert_eq!(publication.path, json);
    assert_eq!(publication.stage, "replace");
    let source = std::error::Error::source(publication)
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .unwrap();
    assert!(source.raw_os_error().is_some());
    assert!(error.to_string().contains(&source.to_string()));
    let published = crate::runner::read_result(&backup).unwrap();
    assert!(!published.segment.as_ref().unwrap().finished);
    assert_eq!(
        published
            .segment
            .unwrap()
            .checkpoint
            .unwrap()
            .completed_steps,
        0
    );
    assert_eq!(published.records.len(), 1);
    assert_eq!(published.records[0].beta, 0.0);
    assert_eq!(published.records[0].f, None);
    assert!(json.is_dir());
    assert_eq!(
        list_itebd_checkpoints(checkpoint)
            .unwrap()
            .iter()
            .map(|e| e.completed_steps)
            .collect::<Vec<_>>(),
        vec![0]
    );
    assert!(!fs::read_dir(dir.path()).unwrap().any(|p| p
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".solve-json-")));
}
