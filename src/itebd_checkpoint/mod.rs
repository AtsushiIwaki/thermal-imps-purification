mod io;
mod schema;

use crate::config::{TrotterOrder, TruncationCfg};
use crate::itebd::StepInfo;
use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
use crate::itebd_state_view::ItebdStateValidationError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use io::{list_itebd_checkpoints, load_itebd_checkpoint, ItebdTrajectoryWriter};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItebdRunMetadata {
    pub dtau: f64,
    pub trotter_order: TrotterOrder,
    pub truncation: TruncationCfg,
    pub canonicalize_every: usize,
    pub record_every_beta: Option<f64>,
    pub model_label: Option<String>,
    pub git_revision: Option<String>,
    pub hermiticity_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct ItebdCheckpointProgress {
    pub beta: f64,
    pub completed_steps: u64,
    pub accumulated_log_norm: f64,
    pub last_step: Option<StepInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItebdStepDiagnostics {
    pub max_bond: usize,
    pub min_singular_value: f64,
    pub log_norm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItebdSnapshotDiagnostics {
    pub bond_dimensions: [usize; 2],
    pub schmidt_norms: [f64; 2],
    pub last_step: Option<ItebdStepDiagnostics>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItebdCheckpointEntry {
    pub index: u64,
    pub beta: f64,
    pub completed_steps: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ItebdCheckpointLoadOptions {
    /// Acceptance tolerance used to reconstruct and validate a complex Hamiltonian.
    ///
    /// This does not replace the source tolerance retained in the run metadata.
    pub hermiticity_tolerance: f64,
    pub diagnostic_relative_tolerance: f64,
}

impl Default for ItebdCheckpointLoadOptions {
    fn default() -> Self {
        Self {
            hermiticity_tolerance: 1e-12,
            diagnostic_relative_tolerance: 1e-12,
        }
    }
}

/// A loaded checkpoint with separate source provenance and load acceptance tolerances.
///
/// The metadata Hermiticity tolerance records the tolerance of the original trajectory, while a
/// complex Hamiltonian carries the caller's load acceptance tolerance. Before creating a new
/// trajectory from these values, set the metadata tolerance to the loaded complex Hamiltonian's
/// hermiticity_tolerance value. Creation intentionally requires an exact match.
pub struct LoadedItebdCheckpoint {
    pub state: ItebdState,
    pub hamiltonian: ItebdHamiltonian,
    pub metadata: ItebdRunMetadata,
    pub progress: ItebdCheckpointProgress,
    pub diagnostics: ItebdSnapshotDiagnostics,
}

#[derive(Debug, thiserror::Error)]
pub enum ItebdCheckpointError {
    #[error("iTEBD checkpoint I/O failed for {path}: {reason}")]
    Io { path: PathBuf, reason: String },
    #[error("invalid iTEBD checkpoint schema field {field:?} in {path}: {reason}")]
    Schema {
        path: PathBuf,
        field: String,
        reason: String,
    },
    #[error("invalid iTEBD checkpoint snapshot {index} field {field:?} in {path}: {reason}")]
    Snapshot {
        path: PathBuf,
        index: u64,
        field: String,
        reason: String,
    },
    #[error("iTEBD checkpoint snapshot {index} in {path} failed state validation: {source}")]
    State {
        path: PathBuf,
        index: u64,
        #[source]
        source: ItebdStateValidationError,
    },
    #[error("iTEBD checkpoint snapshot {index} in {path} is incomplete")]
    Incomplete { path: PathBuf, index: u64 },
    #[error("iTEBD checkpoint writer for {path} is poisoned by an earlier partial write")]
    WriterPoisoned { path: PathBuf },
}
