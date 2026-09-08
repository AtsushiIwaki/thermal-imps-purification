//! Imaginary-time sweep driver: a `RunConfig` in, a structured `SweepResult` out.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{
    Evolution, ExactRefs, ModelSpec, ResolvedModel, RunConfig, TrotterOrder, TruncationCfg,
};
use crate::itebd::free_energy_from_log_norm;
use crate::itebd_auto::{
    canonicalize_auto, energy_density_auto, imaginary_time_step_auto,
    imaginary_time_step_second_order_auto, local_expectation_auto, specific_heat_auto,
    specific_heat_with_options_auto, ItebdHamiltonian, ItebdState,
};
use crate::itebd_checkpoint::ItebdCheckpointProgress;
use crate::itebd_error::ItebdError;
use crate::specific_heat::SpecificHeatOptions;
use crate::tensor::Truncation;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub metadata: Metadata,
    pub records: Vec<Record>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment: Option<SegmentMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetadata {
    pub version: u32,
    pub source: Option<RestartSource>,
    pub start_step: u64,
    pub start_beta: f64,
    pub completed_steps: u64,
    pub checkpoint: Option<CheckpointPosition>,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartSource {
    pub path: String,
    pub snapshot: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointPosition {
    pub index: u64,
    pub completed_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub model: ModelSpec,
    #[serde(deserialize_with = "deserialize_recorded_evolution")]
    pub evolution: Evolution,
    pub truncation: TruncationCfg,
    pub canonicalize_every: usize,
    pub local_dim: usize,
    pub git_revision: Option<String>,
}

#[derive(Deserialize)]
struct RecordedEvolution {
    dtau: f64,
    beta_max: f64,
    record_every_beta: f64,
    #[serde(default = "legacy_first_order")]
    trotter_order: TrotterOrder,
}

fn legacy_first_order() -> TrotterOrder {
    TrotterOrder::First
}

fn deserialize_recorded_evolution<'de, D>(deserializer: D) -> Result<Evolution, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = RecordedEvolution::deserialize(deserializer)?;
    Ok(Evolution {
        dtau: value.dtau,
        beta_max: value.beta_max,
        record_every_beta: value.record_every_beta,
        trotter_order: value.trotter_order,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "StoredRecord")]
pub struct Record {
    pub beta: f64,
    pub u: f64,
    pub c: f64,
    /// None only at beta=0; positive-temperature records contain Some(f).
    pub f: Option<f64>,
    /// Dimensionless free energy, including its finite beta=0 limit -ln(d).
    pub beta_f: f64,
    pub magnetization: f64,
    pub max_bond: usize,
    pub exact: Option<ExactRefs>,
}

#[derive(Deserialize)]
struct StoredRecord {
    beta: f64,
    u: f64,
    c: f64,
    #[serde(deserialize_with = "crate::config::deserialize_required_free_energy")]
    f: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_beta_f")]
    beta_f: Option<f64>,
    magnetization: f64,
    max_bond: usize,
    exact: Option<ExactRefs>,
}

fn deserialize_beta_f<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    f64::deserialize(deserializer).map(Some)
}

impl TryFrom<StoredRecord> for Record {
    type Error = String;

    fn try_from(value: StoredRecord) -> Result<Self, Self::Error> {
        if !value.beta.is_finite() || value.beta < 0.0 {
            return Err("record beta must be finite and nonnegative".into());
        }
        if (value.beta == 0.0) != value.f.is_none() {
            return Err("record f must be null exactly at beta=0".into());
        }
        if let Some(exact) = &value.exact {
            if (value.beta == 0.0) != exact.f.is_none() {
                return Err("record exact.f must be null exactly at beta=0".into());
            }
        }
        let beta_f = value
            .beta_f
            .or_else(|| value.f.map(|f| value.beta * f))
            .ok_or("record beta_f is required at beta=0")?;
        if !beta_f.is_finite() {
            return Err("record beta_f must be finite".into());
        }
        if let Some(f) = value.f {
            let expected = value.beta * f;
            if !expected.is_finite()
                || (beta_f - expected).abs() > 8.0 * f64::EPSILON * expected.abs().max(1.0)
            {
                return Err("record beta_f must agree with beta*f".into());
            }
        }
        Ok(Self {
            beta: value.beta,
            u: value.u,
            c: value.c,
            f: value.f,
            beta_f,
            magnetization: value.magnetization,
            max_bond: value.max_bond,
            exact: value.exact,
        })
    }
}

/// Analytic traces of the maximally mixed physical state; no evolution or normalization.
pub(crate) fn initial_record(
    cfg: &RunConfig,
    ham: &ItebdHamiltonian,
    observable: &DMatrix<Complex64>,
) -> Record {
    let d = ham.dim() as f64;
    let u = match ham {
        ItebdHamiltonian::Real(ham) => ham.site_energy.trace() / (d * d),
        ItebdHamiltonian::Complex(ham) => ham.site_energy().trace().re / (d * d),
    };
    Record {
        beta: 0.0,
        u,
        c: 0.0,
        f: None,
        beta_f: -d.ln(),
        magnetization: observable.trace().re / d,
        max_bond: 1,
        exact: if cfg.output.include_exact {
            cfg.model.exact(0.0)
        } else {
            None
        },
    }
}

/// Run the imaginary-time sweep described by `cfg`, recording observables at the
/// configured β cadence. Mirrors the validated demo loop exactly.
pub fn run_sweep(cfg: &RunConfig) -> Result<SweepResult, ItebdError> {
    run_sweep_impl(cfg, None)
}

#[cfg(test)]
pub(crate) fn run_sweep_with_specific_heat_options(
    cfg: &RunConfig,
    options: &SpecificHeatOptions,
) -> Result<SweepResult, ItebdError> {
    run_sweep_impl(cfg, Some(options))
}

fn run_sweep_impl(
    cfg: &RunConfig,
    specific_heat_options: Option<&SpecificHeatOptions>,
) -> Result<SweepResult, ItebdError> {
    if cfg.checkpoint.is_some() || cfg.restart.is_some() {
        return Err(ItebdError::InvalidRunConfig(
            "run_sweep does not perform checkpoint or restart I/O".into(),
        ));
    }
    cfg.validate().map_err(ItebdError::InvalidRunConfig)?;
    let ResolvedModel {
        hamiltonian,
        observable,
        ..
    } = cfg.model.resolve()?;
    let mut state = ItebdState::infinite_temperature(&hamiltonian)?;
    let mut progress = ItebdCheckpointProgress {
        beta: 0.0,
        completed_steps: 0,
        accumulated_log_norm: 0.0,
        last_step: None,
    };
    let mut result = empty_result(cfg, hamiltonian.dim());
    result
        .records
        .push(initial_record(cfg, &hamiltonian, &observable));
    drive_sweep(
        cfg,
        &hamiltonian,
        &observable,
        &mut state,
        &mut progress,
        &mut result,
        specific_heat_options,
        |_, _, _, _| Ok(()),
    )?;
    Ok(result)
}

pub(crate) fn empty_result(cfg: &RunConfig, local_dim: usize) -> SweepResult {
    SweepResult {
        metadata: Metadata {
            model: cfg.model.clone(),
            evolution: cfg.evolution.clone(),
            truncation: cfg.truncation.clone(),
            canonicalize_every: cfg.run.canonicalize_every,
            local_dim,
            git_revision: git_revision(),
        },
        records: Vec::new(),
        segment: None,
    }
}

pub(crate) fn drive_sweep<E: From<ItebdError>>(
    cfg: &RunConfig,
    ham: &ItebdHamiltonian,
    observable: &DMatrix<Complex64>,
    state: &mut ItebdState,
    progress: &mut ItebdCheckpointProgress,
    result: &mut SweepResult,
    heat: Option<&SpecificHeatOptions>,
    mut on_step: impl FnMut(
        &ItebdState,
        &ItebdCheckpointProgress,
        &mut SweepResult,
        bool,
    ) -> Result<(), E>,
) -> Result<(), E> {
    cfg.validate()
        .map_err(|error| E::from(ItebdError::InvalidRunConfig(error)))?;
    let (target_steps, record_stride) = cfg
        .schedule()
        .map_err(|error| E::from(ItebdError::InvalidRunConfig(error)))?;
    let trunc = Truncation {
        epsilon: cfg.truncation.epsilon,
        max_bond: cfg.truncation.max_bond,
    };
    let dtau = cfg.evolution.dtau;
    let canon_every = cfg.run.canonicalize_every as u64;
    if progress.completed_steps >= target_steps {
        return Ok(());
    }
    for step in (progress.completed_steps + 1)..=target_steps {
        let info = match cfg.evolution.trotter_order {
            TrotterOrder::First => imaginary_time_step_auto(state, ham, dtau, &trunc),
            TrotterOrder::Second => imaginary_time_step_second_order_auto(state, ham, dtau, &trunc),
        }
        .map_err(E::from)?;
        progress.accumulated_log_norm += info.log_norm;
        if step % canon_every == 0 {
            progress.accumulated_log_norm += canonicalize_auto(state, ham).map_err(E::from)?;
        }
        progress.completed_steps = step;
        progress.beta = 2.0 * (step as f64) * dtau;
        progress.last_step = Some(info.clone());
        let mut observed = false;
        if step % record_stride == 0 {
            let beta = 2.0 * (step as f64) * dtau;
            let u = energy_density_auto(state, ham).map_err(E::from)?;
            let c = match heat {
                Some(options) => {
                    specific_heat_with_options_auto(state, ham, beta, options)
                        .map_err(E::from)?
                        .specific_heat_per_site
                }
                None => specific_heat_auto(state, ham, beta).map_err(E::from)?,
            };
            let f = free_energy_from_log_norm(progress.accumulated_log_norm / 2.0, beta, ham.dim());
            let m = local_expectation_auto(state, ham, observable, 1e-12).map_err(E::from)?;
            let exact = if cfg.output.include_exact {
                cfg.model.exact(beta)
            } else {
                None
            };
            result.records.push(Record {
                beta,
                u,
                c,
                f: Some(f),
                beta_f: beta * f,
                magnetization: m,
                max_bond: info.max_bond,
                exact,
            });
            observed = true;
        }
        on_step(state, progress, result, observed)?;
    }
    Ok(())
}

/// Write a result as pretty JSON, creating the parent directory if needed.
pub fn write_result(path: &Path, result: &SweepResult) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(&file, result)?;
    Ok(())
}

/// Read a result back from a JSON file.
pub fn read_result(path: &Path) -> Result<SweepResult, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let result: SweepResult = serde_json::from_reader(file)?;
    Ok(result)
}

/// Best-effort short git revision; `None` if not in a repo or git is unavailable.
fn git_revision() -> Option<String> {
    crate::git_process::git_stdout(&["rev-parse", "--short", "HEAD"])
        .and_then(|stdout| String::from_utf8(stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RunConfig, TrotterOrder};

    #[cfg(unix)]
    #[test]
    fn git_revision_contract() {
        crate::git_process::tests::revision_contract(
            "runner::tests::git_revision_contract",
            git_revision,
            true,
        );
    }

    fn xy_cfg(beta_max: f64, include_exact: bool) -> RunConfig {
        let toml = format!(
            "[model]\ntype = \"xy\"\ngamma = 0.5\nh = 0.5\n\
             [evolution]\ndtau = 0.01\nbeta_max = {beta_max}\nrecord_every_beta = {beta_max}\n\
             [truncation]\nepsilon = 1e-12\nmax_bond = 48\n\
             [output]\npath = \"ignored.json\"\ninclude_exact = {include_exact}\n"
        );
        RunConfig::from_toml_str(&toml).unwrap()
    }

    #[test]
    fn xy_sweep_records_and_tracks_exact() {
        let cfg = xy_cfg(0.5, true);
        let res = run_sweep(&cfg).unwrap();
        assert_eq!(res.metadata.local_dim, 2);
        assert!(!res.records.is_empty());
        let last = res.records.last().unwrap();
        assert!((last.beta - 0.5).abs() < 1e-9, "beta = {}", last.beta);
        let ex = last.exact.as_ref().expect("exact present");
        assert!(
            (last.u - ex.u).abs() < 1e-3,
            "u {} vs exact {}",
            last.u,
            ex.u
        );
        assert!(
            (last.f.unwrap() - ex.f.unwrap()).abs() < 1e-3,
            "f {} vs exact {}",
            last.f.unwrap(),
            ex.f.unwrap()
        );
    }

    #[test]
    fn explicit_first_order_xy_sweep_records_initial_and_evolved_exact_references() {
        let toml = "[model]\ntype = \"xy\"\ngamma = 0.5\nh = 0.5\n\
                    [evolution]\ndtau = 0.01\ntrotter_order = 1\nbeta_max = 0.1\nrecord_every_beta = 0.1\n\
                    [truncation]\nepsilon = 1e-12\nmax_bond = 48\n\
                    [output]\npath = \"ignored.json\"\ninclude_exact = true\n";
        let res = run_sweep(&RunConfig::from_toml_str(toml).unwrap()).unwrap();

        assert_eq!(res.records.len(), 2);
        assert!(res.records[0].exact.is_some());
        assert_eq!(res.metadata.evolution.trotter_order, TrotterOrder::First);

        let second_order = run_sweep(&xy_cfg(0.1, true)).unwrap();
        assert_ne!(
            res.records[1].u.to_bits(),
            second_order.records[1].u.to_bits()
        );
    }

    #[test]
    fn omitted_and_explicit_second_order_are_bitwise_compatible() {
        let omitted = xy_cfg(0.2, false);
        let explicit = RunConfig::from_toml_str(
            "[model]\ntype = \"xy\"\ngamma = 0.5\nh = 0.5\n\
             [evolution]\ndtau = 0.01\ntrotter_order = 2\nbeta_max = 0.2\nrecord_every_beta = 0.2\n\
             [truncation]\nepsilon = 1e-12\nmax_bond = 48\n\
             [output]\npath = \"ignored.json\"\ninclude_exact = false\n",
        )
        .unwrap();

        let omitted_result = run_sweep(&omitted).unwrap();
        let explicit_result = run_sweep(&explicit).unwrap();
        assert_eq!(omitted_result.records.len(), explicit_result.records.len());
        for (omitted, explicit) in omitted_result.records.iter().zip(&explicit_result.records) {
            assert_eq!(omitted.u.to_bits(), explicit.u.to_bits());
            assert_eq!(omitted.f.map(f64::to_bits), explicit.f.map(f64::to_bits));
            assert_eq!(omitted.beta_f.to_bits(), explicit.beta_f.to_bits());
            assert_eq!(omitted.max_bond, explicit.max_bond);
        }
    }

    #[test]
    fn legacy_result_without_trotter_order_remains_first_order() {
        let json = r#"{
          "metadata": {
            "model":{"type":"tfim","j":1.0,"g":0.7},
            "evolution":{"dtau":0.1,"beta_max":0.2,"record_every_beta":0.2},
            "truncation":{"epsilon":1e-12,"max_bond":8},
            "canonicalize_every":1,"local_dim":2,"git_revision":null
          },
          "records":[]
        }"#;
        let parsed: SweepResult = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.metadata.evolution.trotter_order, TrotterOrder::First);
    }

    #[test]
    fn exact_absent_when_not_requested() {
        let res = run_sweep(&xy_cfg(0.2, false)).unwrap();
        assert!(res.records.iter().all(|r| r.exact.is_none()));
    }

    #[test]
    fn computation_only_runner_rejects_io_options() {
        let toml = "[model]\ntype = 'xy'\ngamma = 0.5\nh = 0.5\n\
                    [evolution]\ndtau = 0.05\nbeta_max = 0.1\nrecord_every_beta = 0.1\n\
                    [truncation]\nepsilon = 1e-12\nmax_bond = 8\n\
                    [output]\npath = 'ignored.json'\n\
                    [checkpoint]\npath = 'ignored.h5'\nevery_steps = 1\n";
        let error = run_sweep(&RunConfig::from_toml_str(toml).unwrap()).unwrap_err();
        assert!(matches!(error, ItebdError::InvalidRunConfig(_)));
    }

    #[test]
    fn shared_driver_reports_global_progress_after_published_observation() {
        let cfg = xy_cfg(0.2, false);
        let resolved = cfg.model.resolve().unwrap();
        let mut state = ItebdState::infinite_temperature(&resolved.hamiltonian).unwrap();
        let mut progress = ItebdCheckpointProgress {
            beta: 0.0,
            completed_steps: 0,
            accumulated_log_norm: 0.0,
            last_step: None,
        };
        let mut result = empty_result(&cfg, resolved.hamiltonian.dim());
        let mut callbacks = Vec::new();
        drive_sweep::<ItebdError>(
            &cfg,
            &resolved.hamiltonian,
            &resolved.observable,
            &mut state,
            &mut progress,
            &mut result,
            None,
            |_, progress, result, observed| {
                callbacks.push((progress.completed_steps, observed, result.records.len()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            callbacks,
            vec![
                (1, false, 0),
                (2, false, 0),
                (3, false, 0),
                (4, false, 0),
                (5, false, 0),
                (6, false, 0),
                (7, false, 0),
                (8, false, 0),
                (9, false, 0),
                (10, true, 1),
            ]
        );
        assert_eq!(result.records.len(), 1);
        assert_eq!(progress.beta.to_bits(), 0.2_f64.to_bits());
    }

    #[test]
    fn shared_driver_does_not_publish_a_partial_record_when_heat_fails() {
        let cfg = xy_cfg(0.1, false);
        let resolved = cfg.model.resolve().unwrap();
        let mut state = ItebdState::infinite_temperature(&resolved.hamiltonian).unwrap();
        let mut progress = ItebdCheckpointProgress {
            beta: 0.0,
            completed_steps: 0,
            accumulated_log_norm: 0.0,
            last_step: None,
        };
        let mut result = empty_result(&cfg, resolved.hamiltonian.dim());
        let options = SpecificHeatOptions {
            max_distance: 4,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        let mut callbacks = Vec::new();
        let error = drive_sweep::<ItebdError>(
            &cfg,
            &resolved.hamiltonian,
            &resolved.observable,
            &mut state,
            &mut progress,
            &mut result,
            Some(&options),
            |_, progress, result, observed| {
                callbacks.push((progress.completed_steps, observed, result.records.len()));
                Ok(())
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::SpecificHeatTailNonConvergence {
                max_distance: 4,
                ..
            }
        ));
        assert!(result.records.is_empty());
        assert_eq!(
            callbacks,
            vec![(1, false, 0), (2, false, 0), (3, false, 0), (4, false, 0)]
        );
        assert_eq!(progress.completed_steps, 5);
    }

    #[test]
    fn json_round_trip() {
        let res = run_sweep(&xy_cfg(0.2, false)).unwrap();
        let json = serde_json::to_string(&res).unwrap();
        let back: SweepResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.records.len(), res.records.len());
        assert_eq!(back.metadata.local_dim, res.metadata.local_dim);
        assert!(matches!(
            back.metadata.model,
            crate::config::ModelSpec::Xy { .. }
        ));
    }

    #[test]
    #[ignore = "heavy: evolves to β=8 (d=3); run in release: cargo test --release runner:: -- --ignored"]
    fn aklt_sweep_matches_recorded_demo() {
        let toml = "[model]\ntype = \"aklt\"\n\
                    [evolution]\ndtau = 0.01\nbeta_max = 8.0\nrecord_every_beta = 8.0\n\
                    [truncation]\nepsilon = 1e-12\nmax_bond = 64\n\
                    [output]\npath = \"ignored.json\"\n";
        let cfg = RunConfig::from_toml_str(toml).unwrap();
        let res = run_sweep(&cfg).unwrap();
        let last = res.records.last().unwrap();
        assert!((last.beta - 8.0).abs() < 1e-9, "beta = {}", last.beta);
        assert!((last.u - (-0.66547)).abs() < 5e-4, "u(8) = {}", last.u);
        assert!(
            last.magnetization.abs() < 1e-6,
            "sz = {}",
            last.magnetization
        );
    }

    #[test]
    fn sweep_returns_no_partial_result_when_specific_heat_tail_does_not_converge() {
        let cfg = xy_cfg(0.1, false);
        let options = SpecificHeatOptions {
            max_distance: 4,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        let error = run_sweep_with_specific_heat_options(&cfg, &options).unwrap_err();
        assert!(matches!(
            error,
            ItebdError::SpecificHeatTailNonConvergence {
                max_distance: 4,
                ..
            }
        ));
    }
}
