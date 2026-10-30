//! Step identity is independent of historical temperature targets.
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct StepMeasurementPoint {
    pub sample_index: u32,
    pub completed_steps: u64,
    pub beta_expected: f64,
}
