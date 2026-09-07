use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions,
};
use thermal_imps_purification::runner::read_result;
use std::path::Path;
use std::process::{Command, Output};

// HDF5 native descriptors can be inherited by concurrently spawned CLI processes.
// Serialize this integration harness's direct HDF5 access with its subprocess launches.
static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn config(dir: &Path, name: &str, beta: f64, restart: &str) -> std::path::PathBuf {
    let file = dir.join(format!("{name}.toml"));
    let text = format!("[model]\ntype='tfim'\nj=1.0\ng=0.7\n[evolution]\ndtau=0.05\nbeta_max={beta}\nrecord_every_beta=0.2\n[truncation]\nepsilon=1e-12\nmax_bond=16\n[run]\ncanonicalize_every=1\n[output]\npath='{}'\n[checkpoint]\npath='{}'\nevery_steps=3\n{restart}", dir.join(format!("{name}.json")).display(), dir.join(format!("{name}.h5")).display());
    std::fs::write(&file, text).unwrap();
    file
}
fn run(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(path)
        .output()
        .unwrap()
}
fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_saves_global_cadence_and_restarts_into_new_files() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let first = config(dir.path(), "first", 0.4, "");
    success(&run(&first));
    let source = dir.path().join("first.h5");
    let before = std::fs::read(&source).unwrap();
    let entries = list_itebd_checkpoints(&source).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|e| e.completed_steps)
            .collect::<Vec<_>>(),
        vec![0, 3, 4]
    );
    let second = config(
        dir.path(),
        "second",
        0.8,
        &format!("[restart]\npath='{}'\nsnapshot=1\n", source.display()),
    );
    success(&run(&second));
    assert_eq!(std::fs::read(&source).unwrap(), before);
    let entries = list_itebd_checkpoints(&dir.path().join("second.h5")).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|e| e.completed_steps)
            .collect::<Vec<_>>(),
        vec![3, 6, 8]
    );
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("second.json")).unwrap()).unwrap();
    assert_eq!(json["segment"]["source"]["snapshot"], 1);
    assert_eq!(json["segment"]["start_step"], 3);
    assert_eq!(json["segment"]["completed_steps"], 8);
    assert_eq!(json["segment"]["finished"], true);
    let result = read_result(&dir.path().join("second.json")).unwrap();
    assert_eq!(
        result.records.iter().map(|r| r.beta).collect::<Vec<_>>(),
        vec![0.4, 0.6000000000000001, 0.8]
    );
    let final_state = load_itebd_checkpoint(
        &dir.path().join("second.h5"),
        2,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap();
    assert_eq!(final_state.progress.completed_steps, 8);
}

#[test]
fn observation_failure_preserves_checkpoint_and_partial_json() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let path = config(dir.path(), "failure", 0.4, "");
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("canonicalize_every=1", "canonicalize_every=3");
    std::fs::write(&path, &text).unwrap();
    let failed = run(&path);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("specific-heat tail did not converge"));
    let prior = read_result(&dir.path().join("failure.json")).unwrap();
    assert_eq!(prior.records.len(), 1);
    assert_eq!(prior.records[0].beta, 0.2);
    assert!(!prior.segment.unwrap().finished);
    assert_eq!(
        list_itebd_checkpoints(&dir.path().join("failure.h5"))
            .unwrap()
            .len(),
        2
    );
    // The same noncanonical observation fails without persistence: no tolerance is relaxed.
    let legacy = text
        .split("[checkpoint]")
        .next()
        .unwrap()
        .replace("failure.json", "legacy.json");
    std::fs::write(&path, legacy).unwrap();
    let failed = run(&path);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("specific-heat tail did not converge"));
    assert!(!dir.path().join("legacy.json").exists());
}

#[test]
fn mismatch_and_existing_paths_fail_before_creating_outputs() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let path = config(dir.path(), "source", 0.4, "");
    success(&run(&path));
    let bytes = std::fs::read(dir.path().join("source.json")).unwrap();
    assert!(!run(&path).status.success());
    assert_eq!(
        std::fs::read(dir.path().join("source.json")).unwrap(),
        bytes
    );
    for (field, from, to) in [
        ("Hamiltonian", "g=0.7", "g=0.8"),
        ("dtau", "dtau=0.05", "dtau=0.025"),
        ("epsilon", "epsilon=1e-12", "epsilon=1e-10"),
        ("max_bond", "max_bond=16", "max_bond=32"),
        (
            "canonicalize_every",
            "canonicalize_every=1",
            "canonicalize_every=2",
        ),
        (
            "record_every_beta",
            "record_every_beta=0.2",
            "record_every_beta=0.3",
        ),
    ] {
        let resume = config(
            dir.path(),
            "rejected",
            0.8,
            &format!(
                "[restart]\npath='{}'",
                dir.path().join("source.h5").display()
            ),
        );
        let text = std::fs::read_to_string(&resume).unwrap().replace(from, to);
        std::fs::write(&resume, text).unwrap();
        let failed = run(&resume);
        assert!(!failed.status.success());
        assert!(
            String::from_utf8_lossy(&failed.stderr).contains(field),
            "{}",
            String::from_utf8_lossy(&failed.stderr)
        );
        assert!(!dir.path().join("rejected.json").exists());
        assert!(!dir.path().join("rejected.h5").exists());
    }
}

#[test]
fn latest_skips_incomplete_tail_and_explicit_incomplete_is_an_error() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let path = config(dir.path(), "source", 0.4, "");
    success(&run(&path));
    let source = dir.path().join("source.h5");
    {
        let f = hdf5_metno::File::open_rw(&source).unwrap();
        let group = f.group("states/000002").unwrap();
        group.attr("complete").unwrap().write_scalar(&0u8).unwrap();
    }
    let latest = config(
        dir.path(),
        "latest",
        0.8,
        &format!("[restart]\npath='{}'", source.display()),
    );
    success(&run(&latest));
    assert_eq!(
        read_result(&dir.path().join("latest.json"))
            .unwrap()
            .segment
            .unwrap()
            .start_step,
        3
    );
    let explicit = config(
        dir.path(),
        "explicit",
        0.8,
        &format!("[restart]\npath='{}'\nsnapshot=2", source.display()),
    );
    let failed = run(&explicit);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("incomplete"));
    assert!(!dir.path().join("explicit.h5").exists());
}

#[test]
fn rounded_zero_step_and_restart_target_bounds_are_explicit() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let zero = config(dir.path(), "zero", 0.01, "");
    success(&run(&zero));
    assert_eq!(
        list_itebd_checkpoints(&dir.path().join("zero.h5"))
            .unwrap()
            .iter()
            .map(|e| e.completed_steps)
            .collect::<Vec<_>>(),
        vec![0]
    );
    let json = read_result(&dir.path().join("zero.json")).unwrap();
    assert!(json.records.is_empty() && json.segment.unwrap().finished);
    let source = config(dir.path(), "source", 0.36, "");
    success(&run(&source));
    assert_eq!(
        list_itebd_checkpoints(&dir.path().join("source.h5"))
            .unwrap()
            .last()
            .unwrap()
            .completed_steps,
        4
    );
    let restart = config(
        dir.path(),
        "reject",
        0.4,
        &format!(
            "[restart]\npath='{}'",
            dir.path().join("source.h5").display()
        ),
    );
    let failed = run(&restart);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("must exceed restart step"));
    assert!(!dir.path().join("reject.json").exists());
}

#[test]
fn output_aliases_and_config_collision_preserve_inputs() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    for (case, json, h5) in [
        ("same", dir.path().join("out"), dir.path().join("out")),
        (
            "normalized",
            dir.path().join("unused/../out"),
            dir.path().join("out"),
        ),
        (
            "nested",
            dir.path().join("out"),
            dir.path().join("out/file.h5"),
        ),
    ] {
        let path = config(dir.path(), case, 0.4, "");
        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replace(
                &dir.path()
                    .join(format!("{case}.json"))
                    .display()
                    .to_string(),
                &json.display().to_string(),
            )
            .replace(
                &dir.path().join(format!("{case}.h5")).display().to_string(),
                &h5.display().to_string(),
            );
        std::fs::write(&path, &text).unwrap();
        let failed = run(&path);
        assert!(!failed.status.success());
        assert!(!dir.path().join("out").exists());
        assert_eq!(std::fs::read_to_string(path).unwrap(), text);
    }
    let path = config(dir.path(), "input", 0.4, "");
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("input.json", "input.toml");
    std::fs::write(&path, &text).unwrap();
    assert!(!run(&path).status.success());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    assert!(!dir.path().join("input.h5").exists());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(dir.path(), dir.path().join("alias")).unwrap();
        let path = config(dir.path(), "symlink", 0.4, "");
        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replace("symlink.json", "alias/shared")
            .replace("symlink.h5", "shared");
        std::fs::write(&path, text).unwrap();
        assert!(!run(&path).status.success());
        assert!(!dir.path().join("shared").exists());
    }
}

#[test]
fn library_checkpoint_without_observation_metadata_preserves_complex_storage() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    use thermal_imps_purification::itebd_auto::{ItebdHamiltonian, ItebdState};
    use thermal_imps_purification::itebd_checkpoint::{ItebdCheckpointProgress, ItebdTrajectoryWriter};
    use thermal_imps_purification::itebd_complex::ComplexLocalHamiltonian;
    let dir = tempfile::tempdir().unwrap();
    success(&run(&config(dir.path(), "source", 0.4, "")));
    let loaded = load_itebd_checkpoint(
        &dir.path().join("source.h5"),
        2,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap();
    let mut metadata = loaded.metadata;
    metadata.record_every_beta = None;
    metadata.model_label = Some("descriptive label is not a model contract".into());
    let source = dir.path().join("library.h5");
    let mut writer =
        ItebdTrajectoryWriter::create(&source, metadata.clone(), &loaded.hamiltonian).unwrap();
    writer
        .append((&loaded.state).into(), &loaded.progress)
        .unwrap();
    writer.finish().unwrap();
    let path = config(
        dir.path(),
        "library_resume",
        0.8,
        &format!("[restart]\npath='{}'", source.display()),
    );
    success(&run(&path));
    assert_eq!(
        read_result(&dir.path().join("library_resume.json"))
            .unwrap()
            .segment
            .unwrap()
            .start_step,
        4
    );
    let ItebdHamiltonian::Real(h) = loaded.hamiltonian else {
        panic!("real fixture")
    };
    let complex = ItebdHamiltonian::Complex(
        ComplexLocalHamiltonian::try_new_with_tolerance(
            h.two_site_h.map(num_complex::Complex64::from),
            h.site_energy.map(num_complex::Complex64::from),
            1e-12,
        )
        .unwrap(),
    );
    let state = ItebdState::infinite_temperature(&complex).unwrap();
    let complex_path = dir.path().join("complex.h5");
    let mut writer = ItebdTrajectoryWriter::create(&complex_path, metadata, &complex).unwrap();
    writer
        .append(
            (&state).into(),
            &ItebdCheckpointProgress {
                beta: 0.0,
                completed_steps: 0,
                accumulated_log_norm: 0.0,
                last_step: None,
            },
        )
        .unwrap();
    writer.finish().unwrap();
    let path = config(
        dir.path(),
        "resume_complex",
        0.8,
        &format!("[restart]\npath='{}'", complex_path.display()),
    );
    success(&run(&path));
    let destination = dir.path().join("resume_complex.h5");
    let latest = list_itebd_checkpoints(&destination)
        .unwrap()
        .last()
        .unwrap()
        .index;
    let loaded =
        load_itebd_checkpoint(&destination, latest, &ItebdCheckpointLoadOptions::default())
            .unwrap();
    assert!(matches!(loaded.state, ItebdState::Complex(_)));
    assert!(matches!(loaded.hamiltonian, ItebdHamiltonian::Complex(_)));
}

#[test]
fn malformed_complete_source_does_not_fall_back_to_older_snapshot() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    success(&run(&config(dir.path(), "source", 0.4, "")));
    let source = dir.path().join("source.h5");
    {
        let f = hdf5_metno::File::open_rw(&source).unwrap();
        f.group("states/000002")
            .unwrap()
            .dataset("accumulated_log_norm")
            .unwrap()
            .write_scalar(&f64::NAN)
            .unwrap();
    }
    let path = config(
        dir.path(),
        "reject",
        0.8,
        &format!("[restart]\npath='{}'", source.display()),
    );
    let failed = run(&path);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("accumulated_log_norm"));
    assert!(!dir.path().join("reject.json").exists());
}
