use thermal_imps_purification::config::RunConfig;
use thermal_imps_purification::runner::{CheckpointPosition, RestartSource, SegmentMetadata, SweepResult};

fn base() -> String {
    "[model]\ntype='tfim'\nj=1.0\ng=0.7\n\
     [evolution]\ndtau=0.05\nbeta_max=0.4\nrecord_every_beta=0.2\n\
     [truncation]\nepsilon=1e-12\nmax_bond=16\n\
     [output]\npath='out.json'\n"
        .to_owned()
}

#[test]
fn rejects_zero_checkpoint_cadence() {
    let text = base() + "\n[checkpoint]\npath='x.h5'\nevery_steps=0\n";
    assert!(RunConfig::from_toml_str(&text).is_err());
}

#[test]
fn rejects_restart_without_checkpoint_destination() {
    let text = base() + "\n[restart]\npath='source.h5'\nsnapshot=2\n";
    assert!(RunConfig::from_toml_str(&text).is_err());
}

#[test]
fn rejects_unknown_checkpoint_and_restart_fields() {
    let checkpoint = base() + "\n[checkpoint]\npath='x.h5'\nevery_steps=2\nunexpected=true\n";
    assert!(RunConfig::from_toml_str(&checkpoint).is_err());

    let restart = base()
        + "\n[checkpoint]\npath='new.h5'\nevery_steps=2\n\
           [restart]\npath='old.h5'\nsnapshot=1\nunexpected=true\n";
    assert!(RunConfig::from_toml_str(&restart).is_err());
}

#[test]
fn checkpoint_mode_rejects_blank_paths() {
    for text in [
        base().replace("path='out.json'", "path='  '")
            + "\n[checkpoint]\npath='new.h5'\nevery_steps=2\n",
        base() + "\n[checkpoint]\npath='  '\nevery_steps=2\n",
        base()
            + "\n[checkpoint]\npath='new.h5'\nevery_steps=2\n\
               [restart]\npath='  '\nsnapshot=1\n",
    ] {
        assert!(RunConfig::from_toml_str(&text).is_err(), "accepted {text}");
    }

    assert!(RunConfig::from_toml_str(&base().replace("path='out.json'", "path='  '")).is_ok());
}

#[test]
fn rejects_step_denominator_overflow() {
    let text = base().replace("dtau=0.05", "dtau=1e308");
    assert!(RunConfig::from_toml_str(&text).is_err());
}

#[test]
fn rejects_nonfinite_values_and_unsafe_schedules() {
    for text in [
        base().replace("j=1.0", "j=inf"),
        base().replace("dtau=0.05", "dtau=inf"),
        base().replace("beta_max=0.4", "beta_max=inf"),
        base().replace("record_every_beta=0.2", "record_every_beta=inf"),
        base().replace("epsilon=1e-12", "epsilon=inf"),
        base().replace("beta_max=0.4", "beta_max=9007199254740992.0"),
    ] {
        assert!(RunConfig::from_toml_str(&text).is_err(), "accepted {text}");
    }
}

#[test]
fn parses_and_serializes_checkpoint_and_restart_fields() {
    let text = base()
        + "\n[checkpoint]\npath='new.h5'\nevery_steps=3\n\
           [restart]\npath='old.h5'\nsnapshot=7\n";
    let cfg = RunConfig::from_toml_str(&text).unwrap();
    let checkpoint = cfg.checkpoint.as_ref().unwrap();
    assert_eq!(checkpoint.path, "new.h5");
    assert_eq!(checkpoint.every_steps, 3);
    let restart = cfg.restart.as_ref().unwrap();
    assert_eq!(restart.path, "old.h5");
    assert_eq!(restart.snapshot, Some(7));

    let json = serde_json::to_value(&cfg).unwrap();
    assert_eq!(json["checkpoint"]["every_steps"], 3);
    assert_eq!(json["restart"]["snapshot"], 7);
}

#[test]
fn segment_metadata_round_trips_and_legacy_results_default_to_none() {
    let mut result: SweepResult = serde_json::from_str(
        r#"{"metadata":{"model":{"type":"tfim","j":1.0,"g":0.7},"evolution":{"dtau":0.05,"beta_max":0.4,"record_every_beta":0.2},"truncation":{"epsilon":1e-12,"max_bond":16},"canonicalize_every":1,"local_dim":2,"git_revision":null},"records":[]}"#,
    )
    .unwrap();
    assert!(result.segment.is_none());

    result.segment = Some(SegmentMetadata {
        version: 1,
        source: Some(RestartSource {
            path: "old.h5".into(),
            snapshot: 7,
        }),
        start_step: 7,
        start_beta: 0.7,
        completed_steps: 9,
        checkpoint: Some(CheckpointPosition {
            index: 2,
            completed_steps: 9,
        }),
        finished: false,
    });
    let json = serde_json::to_string(&result).unwrap();
    let back: SweepResult = serde_json::from_str(&json).unwrap();
    let segment = back.segment.unwrap();
    assert_eq!(segment.source.unwrap().snapshot, 7);
    assert_eq!(segment.checkpoint.unwrap().completed_steps, 9);
}
