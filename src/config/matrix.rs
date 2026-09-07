use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::itebd_auto::ItebdHamiltonian;
use crate::itebd_complex::validate_local_operator;
use crate::itebd_error::ItebdError;

const HERMITICITY_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixInput {
    pub real: Vec<Vec<f64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_imag",
        skip_serializing_if = "Option::is_none"
    )]
    pub imag: Option<Vec<Vec<f64>>>,
}

fn deserialize_present_imag<'de, D>(deserializer: D) -> Result<Option<Vec<Vec<f64>>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<Vec<f64>>::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservableInput {
    pub name: String,
    pub matrix: MatrixInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixModelInput {
    pub version: u32,
    pub local_dim: usize,
    pub basis_order: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub two_site_h: MatrixInput,
    pub site_energy: MatrixInput,
    pub observable: ObservableInput,
}

pub struct ResolvedModel {
    pub hamiltonian: ItebdHamiltonian,
    pub observable: DMatrix<Complex64>,
    pub observable_name: String,
}

fn invalid(message: impl Into<String>) -> ItebdError {
    ItebdError::InvalidRunConfig(message.into())
}

fn checked_matrix_len(n: usize, field: &str) -> Result<usize, ItebdError> {
    let elements = n
        .checked_mul(n)
        .ok_or_else(|| invalid(format!("{field} dimension {n} overflows n*n")))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or_else(|| invalid(format!("{field} allocation size overflows")))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(format!(
            "{field} allocation requires {bytes} bytes, exceeding isize::MAX"
        )));
    }
    Ok(elements)
}

impl MatrixInput {
    pub fn to_matrix(&self, n: usize, field: &str) -> Result<DMatrix<Complex64>, ItebdError> {
        self.validate(n, field)?;
        Ok(self.to_matrix_after_validation(n))
    }

    fn validate(&self, n: usize, field: &str) -> Result<(), ItebdError> {
        checked_matrix_len(n, field)?;
        validate_component(&self.real, n, &format!("{field}.real"))?;
        if let Some(imag) = &self.imag {
            validate_component(imag, n, &format!("{field}.imag"))?;
        }
        Ok(())
    }

    fn to_matrix_after_validation(&self, n: usize) -> DMatrix<Complex64> {
        DMatrix::from_fn(n, n, |row, column| {
            Complex64::new(
                self.real[row][column],
                self.imag.as_ref().map_or(0.0, |imag| imag[row][column]),
            )
        })
    }
}

fn validate_component(rows: &[Vec<f64>], n: usize, field: &str) -> Result<(), ItebdError> {
    if rows.len() != n {
        return Err(invalid(format!(
            "{field} must contain exactly {n} rows, got {}",
            rows.len()
        )));
    }
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != n {
            return Err(invalid(format!(
                "{field}[{row_index}] must contain exactly {n} columns, got {}",
                row.len()
            )));
        }
        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "{field}[{row_index}][{column_index}] must be finite"
                )));
            }
        }
    }
    Ok(())
}

impl MatrixModelInput {
    pub(crate) fn resolve(&self) -> Result<ResolvedModel, ItebdError> {
        self.validate_header()?;
        let matrix_dim = self
            .local_dim
            .checked_mul(self.local_dim)
            .ok_or_else(|| invalid(format!("model.local_dim {} overflows d*d", self.local_dim)))?;
        self.two_site_h.validate(matrix_dim, "model.two_site_h")?;
        self.site_energy.validate(matrix_dim, "model.site_energy")?;
        self.observable
            .matrix
            .validate(self.local_dim, "model.observable.matrix")?;

        let two_site_h = self.two_site_h.to_matrix_after_validation(matrix_dim);
        let site_energy = self.site_energy.to_matrix_after_validation(matrix_dim);
        let observable = self
            .observable
            .matrix
            .to_matrix_after_validation(self.local_dim);
        let hamiltonian = ItebdHamiltonian::try_from_complex(two_site_h, site_energy)
            .map_err(qualify_hamiltonian_error)?;
        validate_local_operator(&observable, self.local_dim, HERMITICITY_TOLERANCE)
            .map_err(qualify_observable_error)?;
        Ok(ResolvedModel {
            hamiltonian,
            observable,
            observable_name: self.observable.name.clone(),
        })
    }

    pub(crate) fn real_observable(&self) -> Result<DMatrix<f64>, ItebdError> {
        self.validate_header()?;
        let observable = self
            .observable
            .matrix
            .to_matrix(self.local_dim, "model.observable.matrix")?;
        ensure_exactly_real(&observable, "model.observable.matrix")?;
        validate_local_operator(&observable, self.local_dim, HERMITICITY_TOLERANCE)
            .map_err(qualify_observable_error)?;
        Ok(observable.map(|value| value.re))
    }

    pub(crate) fn real_hamiltonian(&self) -> Result<crate::model::LocalHamiltonian, ItebdError> {
        self.validate_header()?;
        let matrix_dim = self
            .local_dim
            .checked_mul(self.local_dim)
            .ok_or_else(|| invalid(format!("model.local_dim {} overflows d*d", self.local_dim)))?;
        self.two_site_h.validate(matrix_dim, "model.two_site_h")?;
        self.site_energy.validate(matrix_dim, "model.site_energy")?;

        let two_site_h = self.two_site_h.to_matrix_after_validation(matrix_dim);
        let site_energy = self.site_energy.to_matrix_after_validation(matrix_dim);
        ensure_exactly_real(&two_site_h, "model.two_site_h")?;
        ensure_exactly_real(&site_energy, "model.site_energy")?;
        match ItebdHamiltonian::try_from_complex(two_site_h, site_energy)
            .map_err(qualify_hamiltonian_error)?
        {
            ItebdHamiltonian::Real(hamiltonian) => Ok(hamiltonian),
            ItebdHamiltonian::Complex(_) => unreachable!("exact-real check controls dispatch"),
        }
    }

    fn validate_header(&self) -> Result<(), ItebdError> {
        if self.version != 1 {
            return Err(invalid(format!(
                "model.version must be 1, got {}",
                self.version
            )));
        }
        if self.local_dim == 0 {
            return Err(invalid("model.local_dim must be >= 1"));
        }
        if self.basis_order != "first_site_fastest" {
            return Err(invalid(format!(
                "model.basis_order must be \"first_site_fastest\", got {:?}",
                self.basis_order
            )));
        }
        if self.observable.name.trim().is_empty() {
            return Err(invalid("model.observable.name must not be blank"));
        }
        Ok(())
    }
}

fn qualify_hamiltonian_field(field: &'static str) -> &'static str {
    match field {
        "two_site_h" => "model.two_site_h",
        "site_energy" => "model.site_energy",
        _ => field,
    }
}

fn qualify_hamiltonian_error(error: ItebdError) -> ItebdError {
    match error {
        ItebdError::NonSquare { field, rows, cols } => ItebdError::NonSquare {
            field: qualify_hamiltonian_field(field),
            rows,
            cols,
        },
        ItebdError::NonFiniteMatrix { field, row, column } => ItebdError::NonFiniteMatrix {
            field: qualify_hamiltonian_field(field),
            row,
            column,
        },
        ItebdError::NonHermitian {
            field,
            residual,
            tolerance,
        } => ItebdError::NonHermitian {
            field: qualify_hamiltonian_field(field),
            residual,
            tolerance,
        },
        other => other,
    }
}

fn qualify_observable_error(error: ItebdError) -> ItebdError {
    match error {
        ItebdError::NonSquare { rows, cols, .. } => ItebdError::NonSquare {
            field: "model.observable.matrix",
            rows,
            cols,
        },
        ItebdError::NonFiniteMatrix { row, column, .. } => ItebdError::NonFiniteMatrix {
            field: "model.observable.matrix",
            row,
            column,
        },
        ItebdError::NonHermitian {
            residual,
            tolerance,
            ..
        } => ItebdError::NonHermitian {
            field: "model.observable.matrix",
            residual,
            tolerance,
        },
        other => other,
    }
}

fn ensure_exactly_real(matrix: &DMatrix<Complex64>, field: &str) -> Result<(), ItebdError> {
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            if matrix[(row, column)].im != 0.0 {
                return Err(invalid(format!(
                    "{field}[{row}][{column}] has a nonzero imaginary entry"
                )));
            }
        }
    }
    Ok(())
}
