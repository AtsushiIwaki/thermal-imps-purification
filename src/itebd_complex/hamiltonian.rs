use crate::itebd_error::ItebdError;
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;

const DEFAULT_HERMITICITY_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct ComplexLocalHamiltonian {
    two_site_h: DMatrix<Complex64>,
    site_energy: DMatrix<Complex64>,
    physical_dim: usize,
    hermiticity_tolerance: f64,
}

impl ComplexLocalHamiltonian {
    pub fn try_new(
        two_site_h: DMatrix<Complex64>,
        site_energy: DMatrix<Complex64>,
    ) -> Result<Self, ItebdError> {
        Self::try_new_with_tolerance(two_site_h, site_energy, DEFAULT_HERMITICITY_TOLERANCE)
    }

    pub fn try_new_with_tolerance(
        two_site_h: DMatrix<Complex64>,
        site_energy: DMatrix<Complex64>,
        hermiticity_tolerance: f64,
    ) -> Result<Self, ItebdError> {
        validate_square("two_site_h", &two_site_h)?;
        validate_square("site_energy", &site_energy)?;
        if two_site_h.nrows() != site_energy.nrows() {
            return Err(ItebdError::MatrixDimensionMismatch {
                two_site: two_site_h.nrows(),
                site_energy: site_energy.nrows(),
            });
        }

        let physical_dim = physical_dimension(two_site_h.nrows())?;

        validate_finite("two_site_h", &two_site_h)?;
        validate_finite("site_energy", &site_energy)?;
        validate_tolerance(hermiticity_tolerance)?;
        validate_hermitian("two_site_h", &two_site_h, hermiticity_tolerance)?;
        validate_hermitian("site_energy", &site_energy, hermiticity_tolerance)?;

        Ok(Self {
            two_site_h,
            site_energy,
            physical_dim,
            hermiticity_tolerance,
        })
    }

    pub fn two_site_h(&self) -> &DMatrix<Complex64> {
        &self.two_site_h
    }

    pub fn site_energy(&self) -> &DMatrix<Complex64> {
        &self.site_energy
    }

    pub fn dim(&self) -> usize {
        self.physical_dim
    }

    pub fn hermiticity_tolerance(&self) -> f64 {
        self.hermiticity_tolerance
    }

    pub fn trotter_gate(&self, tau: f64) -> Result<DMatrix<Complex64>, ItebdError> {
        complex_trotter_gate_with_tolerance(&self.two_site_h, tau, self.hermiticity_tolerance)
    }

    pub(crate) fn into_parts(self) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
        (self.two_site_h, self.site_energy)
    }
}

pub fn complex_trotter_gate(
    h: &DMatrix<Complex64>,
    tau: f64,
) -> Result<DMatrix<Complex64>, ItebdError> {
    complex_trotter_gate_with_tolerance(h, tau, DEFAULT_HERMITICITY_TOLERANCE)
}

fn complex_trotter_gate_with_tolerance(
    h: &DMatrix<Complex64>,
    tau: f64,
    hermiticity_tolerance: f64,
) -> Result<DMatrix<Complex64>, ItebdError> {
    if !tau.is_finite() {
        return Err(ItebdError::InvalidTimeStep { value: tau });
    }
    validate_square("two_site_h", h)?;
    physical_dimension(h.nrows())?;
    validate_finite("two_site_h", h)?;
    validate_tolerance(hermiticity_tolerance)?;
    validate_hermitian("two_site_h", h, hermiticity_tolerance)?;

    let eig = SymmetricEigen::new(h.clone());
    for (index, eigenvalue) in eig.eigenvalues.iter().enumerate() {
        if !eigenvalue.is_finite() {
            return Err(ItebdError::NonFiniteEigenvalue { index });
        }
    }

    let exponential = eig
        .eigenvalues
        .map(|eigenvalue| Complex64::new((-tau * eigenvalue).exp(), 0.0));
    let gate =
        &eig.eigenvectors * DMatrix::from_diagonal(&exponential) * eig.eigenvectors.adjoint();
    for column in 0..gate.ncols() {
        for row in 0..gate.nrows() {
            if !is_finite(gate[(row, column)]) {
                return Err(ItebdError::NonFiniteGate { row, column });
            }
        }
    }
    Ok(gate)
}

fn physical_dimension(matrix_dim: usize) -> Result<usize, ItebdError> {
    let physical_dim = (matrix_dim as f64).sqrt() as usize;
    if physical_dim == 0 || physical_dim.checked_mul(physical_dim) != Some(matrix_dim) {
        return Err(ItebdError::InvalidPhysicalDimension { matrix_dim });
    }
    Ok(physical_dim)
}

fn validate_square(field: &'static str, matrix: &DMatrix<Complex64>) -> Result<(), ItebdError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(ItebdError::NonSquare {
            field,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, matrix: &DMatrix<Complex64>) -> Result<(), ItebdError> {
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            if !is_finite(matrix[(row, column)]) {
                return Err(ItebdError::NonFiniteMatrix { field, row, column });
            }
        }
    }
    Ok(())
}

fn validate_tolerance(tolerance: f64) -> Result<(), ItebdError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ItebdError::InvalidTolerance {
            name: "hermiticity tolerance",
            value: tolerance,
        });
    }
    Ok(())
}

fn validate_hermitian(
    field: &'static str,
    matrix: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<(), ItebdError> {
    let residual = scaled_hermiticity_residual(matrix);
    if !residual.is_finite() || residual > tolerance {
        return Err(ItebdError::NonHermitian {
            field,
            residual,
            tolerance,
        });
    }
    Ok(())
}

fn scaled_hermiticity_residual(matrix: &DMatrix<Complex64>) -> f64 {
    let scale = matrix
        .iter()
        .flat_map(|value| [value.re.abs(), value.im.abs()])
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }

    let scaled_matrix = matrix.map(|value| value / scale);
    let mut scaled_anti_hermitian = DMatrix::zeros(matrix.nrows(), matrix.ncols());
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[(row, column)];
            let adjoint_value = matrix[(column, row)];
            scaled_anti_hermitian[(row, column)] = Complex64::new(
                value.re / scale - adjoint_value.re / scale,
                value.im / scale + adjoint_value.im / scale,
            );
        }
    }

    let scaled_matrix_norm = scaled_frobenius_norm(scaled_matrix.as_slice());
    let scaled_residual_norm = scaled_frobenius_norm(scaled_anti_hermitian.as_slice());
    if scale >= scaled_matrix_norm.recip() {
        scaled_residual_norm / scaled_matrix_norm
    } else {
        scale * scaled_residual_norm
    }
}

fn is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn scaled_frobenius_norm(data: &[Complex64]) -> f64 {
    let mut scale = 0.0;
    let mut scaled_sum_squares = 1.0;
    for component in data
        .iter()
        .flat_map(|value| [value.re.abs(), value.im.abs()])
        .filter(|component| *component != 0.0)
    {
        if component.is_nan() {
            return f64::NAN;
        }
        if component.is_infinite() {
            return f64::INFINITY;
        }
        if scale < component {
            scaled_sum_squares = 1.0 + scaled_sum_squares * (scale / component).powi(2);
            scale = component;
        } else {
            scaled_sum_squares += (component / scale).powi(2);
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * scaled_sum_squares.sqrt()
    }
}
