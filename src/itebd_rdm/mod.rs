//! Checked reduced-density-matrix options and numerical diagnostics for two-site iTEBD.
#[cfg(test)]
mod benchmarks;
mod environment;
mod error;
mod interval;
#[cfg(test)]
mod oracles;
mod tensor;

pub use error::RdmError;
use num_complex::Complex64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RdmParity {
    A,
    B,
}

#[derive(Clone, Debug)]
pub struct RdmOptions {
    pub fixed_point_tolerance: f64,
    pub max_iterations: usize,
    pub hermiticity_tolerance: f64,
    pub trace_tolerance: f64,
    pub positivity_tolerance: f64,
    pub max_output_elements: usize,
    /// Maximum logical entries in a retained tensor; also bounds the interval
    /// occurrence-index slots (2 * length + 4), including physical dimension one.
    /// Does not bound aggregate memory or hidden backend workspace.
    pub max_intermediate_elements: usize,
}
impl Default for RdmOptions {
    fn default() -> Self {
        Self {
            fixed_point_tolerance: 1e-12,
            max_iterations: 10_000,
            hermiticity_tolerance: 1e-10,
            trace_tolerance: 1e-10,
            positivity_tolerance: 1e-10,
            max_output_elements: 1_048_576,
            max_intermediate_elements: 16_777_216,
        }
    }
}
impl RdmOptions {
    pub(super) fn validate(&self) -> Result<(), RdmError> {
        for (name, value) in [
            ("fixed_point_tolerance", self.fixed_point_tolerance),
            ("hermiticity_tolerance", self.hermiticity_tolerance),
            ("trace_tolerance", self.trace_tolerance),
            ("positivity_tolerance", self.positivity_tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RdmError::InvalidOption {
                    name,
                    reason: "must be finite and nonnegative".into(),
                });
            }
        }
        for (name, value) in [
            ("max_iterations", self.max_iterations),
            ("max_output_elements", self.max_output_elements),
            ("max_intermediate_elements", self.max_intermediate_elements),
        ] {
            if value == 0 {
                return Err(RdmError::InvalidOption {
                    name,
                    reason: "must be positive".into(),
                });
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct RdmEnvironmentDiagnostics {
    pub eigenvalue: f64,
    pub iterations: usize,
    pub relative_residual: f64,
}
/// Diagnostics for identity-seeded boundaries, without asserting sector uniqueness.
#[derive(Clone, Debug)]
pub struct RdmReport {
    pub start: RdmParity,
    pub length: usize,
    pub left: RdmEnvironmentDiagnostics,
    pub right: RdmEnvironmentDiagnostics,
    pub raw_trace: Complex64,
    pub trace_residual: f64,
    pub hermiticity_residual: f64,
    pub minimum_eigenvalue: f64,
    pub largest_intermediate_elements: usize,
}

/// A normalized physical interval matrix and its numerical diagnostics.
#[derive(Clone, Debug)]
pub struct RdmResult<T: nalgebra::Scalar> {
    pub density_matrix: nalgebra::DMatrix<T>,
    pub report: RdmReport,
}
#[derive(Clone, Debug)]
pub enum AutoRdmResult {
    Real(RdmResult<f64>),
    Complex(RdmResult<Complex64>),
}
pub use interval::{
    reduced_density_matrix, reduced_density_matrix_auto, reduced_density_matrix_complex,
};
