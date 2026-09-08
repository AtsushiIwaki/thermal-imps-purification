//! Opt-in filesystem orchestration for solve trajectories.
mod output;
mod validation;

use crate::config::{ModelSpec, ResolvedModel, RunConfig};
use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
use crate::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointError,
    ItebdCheckpointLoadOptions, ItebdCheckpointProgress, ItebdRunMetadata, ItebdTrajectoryWriter,
};
use crate::itebd_error::ItebdError;
use crate::runner::{
    drive_sweep, empty_result, initial_record, CheckpointPosition, RestartSource, SegmentMetadata,
    SweepResult,
};
use output::JsonPublisher;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum SolveRunError {
    #[error("invalid checkpoint run: {0}")]
    Configuration(String),
    #[error("restart configuration mismatch: {field}")]
    RestartMismatch { field: &'static str },
    #[error("checkpoint run path validation failed for {path}: {source}")]
    Path {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Physics(#[from] ItebdError),
    #[error(transparent)]
    Checkpoint(#[from] ItebdCheckpointError),
    #[error(transparent)]
    Publication(#[from] output::PublicationError),
}

/// Save initial, periodic and final states into a new trajectory, optionally continuing a
/// compatible checkpoint. Fresh trajectories may use either scalar backend.
/// `config_path` protects the CLI input from output aliasing.
/// Fresh runs include beta=0. Restarts cover only steps after the selected start.
pub fn run_checkpointed_sweep(
    cfg: &RunConfig,
    config_path: Option<&Path>,
) -> Result<SweepResult, SolveRunError> {
    run_with_checkpoint_observer(cfg, config_path, |_| Ok(()))
}

fn resolve_configured_model(cfg: &RunConfig) -> Result<ResolvedModel, SolveRunError> {
    match cfg.model.resolve() {
        Err(ItebdError::NonFiniteMatrix {
            field: "two_site_h" | "site_energy",
            ..
        }) if !matches!(&cfg.model, ModelSpec::Matrix(_)) => Err(SolveRunError::Configuration(
            "model Hamiltonian contains non-finite entries".into(),
        )),
        resolved => resolved.map_err(SolveRunError::Physics),
    }
}

fn run_with_checkpoint_observer(
    cfg: &RunConfig,
    config_path: Option<&Path>,
    mut checkpoint_observer: impl FnMut(u64) -> Result<(), SolveRunError>,
) -> Result<SweepResult, SolveRunError> {
    cfg.validate().map_err(SolveRunError::Configuration)?;
    let checkpoint = cfg
        .checkpoint
        .as_ref()
        .ok_or_else(|| SolveRunError::Configuration("checkpoint settings are required".into()))?;
    let (target, _) = cfg.schedule().map_err(SolveRunError::Configuration)?;
    validation::paths(cfg, config_path)?;
    let ResolvedModel {
        hamiltonian: configured_hamiltonian,
        observable,
        ..
    } = resolve_configured_model(cfg)?;
    let (mut state, ham, mut progress, source) = if let Some(restart) = &cfg.restart {
        let path = Path::new(&restart.path);
        let snapshot = match restart.snapshot {
            Some(index) => index,
            None => {
                list_itebd_checkpoints(path)?
                    .last()
                    .ok_or_else(|| {
                        SolveRunError::Configuration(format!(
                            "no complete checkpoints in {}",
                            path.display()
                        ))
                    })?
                    .index
            }
        };
        let loaded = load_itebd_checkpoint(path, snapshot, &ItebdCheckpointLoadOptions::default())?;
        validation::compatible(
            cfg,
            &configured_hamiltonian,
            &loaded.hamiltonian,
            &loaded.metadata,
        )?;
        if target <= loaded.progress.completed_steps {
            return Err(SolveRunError::Configuration(format!(
                "target step {target} must exceed restart step {}",
                loaded.progress.completed_steps
            )));
        }
        (
            loaded.state,
            loaded.hamiltonian,
            loaded.progress,
            Some(RestartSource {
                path: restart.path.clone(),
                snapshot,
            }),
        )
    } else {
        let state = ItebdState::infinite_temperature(&configured_hamiltonian)?;
        (
            state,
            configured_hamiltonian,
            ItebdCheckpointProgress {
                beta: 0.0,
                completed_steps: 0,
                accumulated_log_norm: 0.0,
                last_step: None,
            },
            None,
        )
    };
    let mut result = empty_result(cfg, ham.dim());
    if cfg.restart.is_none() {
        result.records.push(initial_record(cfg, &ham, &observable));
    }
    result.segment = Some(SegmentMetadata {
        version: 1,
        source,
        start_step: progress.completed_steps,
        start_beta: progress.beta,
        completed_steps: progress.completed_steps,
        checkpoint: None,
        finished: false,
    });
    let metadata = ItebdRunMetadata {
        dtau: cfg.evolution.dtau,
        trotter_order: cfg.evolution.trotter_order,
        truncation: cfg.truncation.clone(),
        canonicalize_every: cfg.run.canonicalize_every,
        record_every_beta: Some(cfg.evolution.record_every_beta),
        model_label: Some(
            serde_json::to_string(&cfg.model)
                .map_err(|e| SolveRunError::Configuration(e.to_string()))?,
        ),
        git_revision: result.metadata.git_revision.clone(),
        hermiticity_tolerance: match &ham {
            ItebdHamiltonian::Real(_) => 1e-12,
            ItebdHamiltonian::Complex(hamiltonian) => hamiltonian.hermiticity_tolerance(),
        },
    };
    let mut publisher = JsonPublisher::create(Path::new(&cfg.output.path), &result)?;
    let mut writer = ItebdTrajectoryWriter::create(&checkpoint.path, metadata, &ham)?;
    let entry = writer.append((&state).into(), &progress)?;
    result.segment.as_mut().unwrap().checkpoint = Some(CheckpointPosition {
        index: entry.index,
        completed_steps: entry.completed_steps,
    });
    publisher.publish(&result)?;
    checkpoint_observer(progress.completed_steps)?;
    drive_sweep::<SolveRunError>(
        cfg,
        &ham,
        &observable,
        &mut state,
        &mut progress,
        &mut result,
        None,
        |state, progress, result, observed| {
            result.segment.as_mut().unwrap().completed_steps = progress.completed_steps;
            if observed {
                publisher.publish(result)?;
            }
            if progress.completed_steps % checkpoint.every_steps == 0
                || progress.completed_steps == target
            {
                let entry = writer.append(state.into(), progress)?;
                result.segment.as_mut().unwrap().checkpoint = Some(CheckpointPosition {
                    index: entry.index,
                    completed_steps: entry.completed_steps,
                });
                publisher.publish(result)?;
                checkpoint_observer(progress.completed_steps)?;
            }
            Ok(())
        },
    )?;
    writer.finish()?;
    result.segment.as_mut().unwrap().finished = true;
    publisher.publish(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests;
