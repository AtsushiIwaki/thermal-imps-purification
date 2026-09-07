use crate::itebd_state_view::ItebdStateValidationError;

#[derive(Debug, thiserror::Error)]
pub enum RdmError {
    #[error(transparent)]
    State(#[from] ItebdStateValidationError),
    #[error("invalid RDM option {name}: {reason}")]
    InvalidOption { name: &'static str, reason: String },
    #[error("RDM resource limit at {stage}: {requested} elements exceed {limit}")]
    ResourceLimit {
        stage: &'static str,
        requested: usize,
        limit: usize,
    },
    #[error("RDM dimension overflow at {stage}")]
    DimensionOverflow { stage: &'static str },
    #[error(
        "{side} fixed point did not converge in {iterations} iterations: residual {residual:e}"
    )]
    FixedPointNonConvergence {
        side: &'static str,
        iterations: usize,
        residual: f64,
    },
    #[error("invalid {side} environment: {reason}")]
    InvalidEnvironment { side: &'static str, reason: String },
    #[error("invalid density matrix {invariant}: {value:e} (tolerance {tolerance:e})")]
    InvalidDensityMatrix {
        invariant: &'static str,
        value: f64,
        tolerance: f64,
    },
    #[error("RDM tensor failure at {stage}: {reason}")]
    Tensor { stage: &'static str, reason: String },
}
