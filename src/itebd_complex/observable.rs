use super::tensor::{complex_from_fn, complex_relabel, complex_scalar, complex_scale_bond};
use super::{ComplexLocalHamiltonian, ComplexPurifiedMps, ComplexSite};
use crate::itebd_error::ItebdError;
use crate::tensor::{new_index, Tensor};
use crate::contraction_pairwise::{pairwise, pairwise_with_conjugation};
use nalgebra::DMatrix;
use num_complex::Complex64;

const OBSERVABLE_STAGE: &str = "complex_observable";

fn tensor_error(message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage: OBSERVABLE_STAGE,
        message: message.to_string(),
    }
}

fn contract(lhs: &Tensor, rhs: &Tensor, context: &'static str) -> Result<Tensor, ItebdError> {
    pairwise(lhs, rhs).map_err(|error| tensor_error(format!("{context}: {error}")))
}

fn contract_conjugated_bra(
    bra: &Tensor,
    ket_network: &Tensor,
    context: &'static str,
) -> Result<Tensor, ItebdError> {
    pairwise_with_conjugation(bra, ket_network, true, false)
        .map_err(|error| tensor_error(format!("{context}: {error}")))
}

fn validate_tolerance(tolerance: f64) -> Result<(), ItebdError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ItebdError::InvalidTolerance {
            name: "local observable Hermiticity tolerance",
            value: tolerance,
        });
    }
    Ok(())
}

pub(crate) fn validate_local_operator(
    operator: &DMatrix<Complex64>,
    dimension: usize,
    tolerance: f64,
) -> Result<(), ItebdError> {
    validate_tolerance(tolerance)?;
    if operator.nrows() != operator.ncols() {
        return Err(ItebdError::NonSquare {
            field: "local observable",
            rows: operator.nrows(),
            cols: operator.ncols(),
        });
    }
    if operator.nrows() != dimension {
        return Err(tensor_error(format!(
            "local observable must be {dimension}x{dimension}, got {}x{}",
            operator.nrows(),
            operator.ncols()
        )));
    }
    for column in 0..operator.ncols() {
        for row in 0..operator.nrows() {
            let value = operator[(row, column)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(ItebdError::NonFiniteMatrix {
                    field: "local observable",
                    row,
                    column,
                });
            }
        }
    }
    let residual = scaled_hermiticity_residual(operator);
    if !residual.is_finite() || residual > tolerance {
        return Err(ItebdError::NonHermitian {
            field: "local observable",
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
            let adjoint = matrix[(column, row)];
            scaled_anti_hermitian[(row, column)] = Complex64::new(
                value.re / scale - adjoint.re / scale,
                value.im / scale + adjoint.im / scale,
            );
        }
    }
    let matrix_norm = scaled_frobenius_norm(scaled_matrix.as_slice());
    let residual_norm = scaled_frobenius_norm(scaled_anti_hermitian.as_slice());
    if scale >= matrix_norm.recip() {
        residual_norm / matrix_norm
    } else {
        scale * residual_norm
    }
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

pub(crate) fn real_observable(
    observable: &'static str,
    value: Complex64,
    tolerance: f64,
) -> Result<f64, ItebdError> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(tensor_error(format!(
            "{observable} expectation is non-finite: {value:?}"
        )));
    }
    if value.im.abs() > tolerance * value.re.abs().max(1.0) {
        return Err(ItebdError::NonRealObservable {
            observable,
            imaginary: value.im,
            tolerance,
        });
    }
    Ok(value.re)
}

fn normalized_scalar(
    observable: &'static str,
    numerator: Tensor,
    denominator: Tensor,
) -> Result<Complex64, ItebdError> {
    let numerator = complex_scalar(&numerator)?;
    let denominator = complex_scalar(&denominator)?;
    if !denominator.re.is_finite() || !denominator.im.is_finite() || denominator.norm() == 0.0 {
        return Err(tensor_error(format!(
            "invalid {observable} normalization scalar {denominator:?}"
        )));
    }
    let value = numerator / denominator;
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(tensor_error(format!(
            "invalid normalized {observable} expectation {value:?}"
        )));
    }
    Ok(value)
}

fn two_site_expectation(
    operator: &DMatrix<Complex64>,
    left_site: &ComplexSite,
    right_site: &ComplexSite,
    outer_left: &[f64],
    middle: &[f64],
    outer_right: &[f64],
) -> Result<Complex64, ItebdError> {
    let right_environment = new_index(right_site.right.dim);
    let right_gamma = complex_relabel(&right_site.gamma, &right_site.right, &right_environment)?;
    let left_gamma = complex_scale_bond(
        &complex_scale_bond(&left_site.gamma, &left_site.left, outer_left)?,
        &left_site.right,
        middle,
    )?;
    let right_gamma = complex_scale_bond(&right_gamma, &right_environment, outer_right)?;
    let ket = contract(&left_gamma, &right_gamma, "two-site ket")?;

    let left_output = new_index(left_site.phys.dim);
    let right_output = new_index(right_site.phys.dim);
    let dimension = left_site.phys.dim;
    let operator_tensor = complex_from_fn(
        &[
            left_site.phys.clone(),
            right_site.phys.clone(),
            left_output.clone(),
            right_output.clone(),
        ],
        |index| {
            let input = index[0] + dimension * index[1];
            let output = index[2] + dimension * index[3];
            operator[(output, input)]
        },
    )?;
    let operated = contract(&ket, &operator_tensor, "two-site operator")?;
    let bra = complex_relabel(
        &complex_relabel(&ket, &left_site.phys, &left_output)?,
        &right_site.phys,
        &right_output,
    )?;
    let numerator = contract_conjugated_bra(&bra, &operated, "two-site bra")?;
    let denominator = contract_conjugated_bra(&ket, &ket, "two-site norm")?;
    normalized_scalar("complex energy density", numerator, denominator)
}

fn one_site_expectation(
    operator: &DMatrix<Complex64>,
    site: &ComplexSite,
    left: &[f64],
    right: &[f64],
) -> Result<Complex64, ItebdError> {
    let ket = complex_scale_bond(
        &complex_scale_bond(&site.gamma, &site.left, left)?,
        &site.right,
        right,
    )?;
    let physical_output = new_index(site.phys.dim);
    let operator_tensor =
        complex_from_fn(&[site.phys.clone(), physical_output.clone()], |index| {
            operator[(index[1], index[0])]
        })?;
    let operated = contract(&ket, &operator_tensor, "one-site operator")?;
    let bra = complex_relabel(&ket, &site.phys, &physical_output)?;
    let numerator = contract_conjugated_bra(&bra, &operated, "one-site bra")?;
    let denominator = contract_conjugated_bra(&ket, &ket, "one-site norm")?;
    normalized_scalar("complex local observable", numerator, denominator)
}

pub fn energy_density_complex(
    state: &ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
) -> Result<f64, ItebdError> {
    state.validate_topology()?;
    if state.a.phys.dim != hamiltonian.dim() {
        return Err(tensor_error(format!(
            "state physical dimension {} does not match Hamiltonian dimension {}",
            state.a.phys.dim,
            hamiltonian.dim()
        )));
    }
    let ab = two_site_expectation(
        hamiltonian.two_site_h(),
        &state.a,
        &state.b,
        &state.lambda_ba,
        &state.lambda_ab,
        &state.lambda_ba,
    )?;
    let ba = two_site_expectation(
        hamiltonian.two_site_h(),
        &state.b,
        &state.a,
        &state.lambda_ab,
        &state.lambda_ba,
        &state.lambda_ab,
    )?;
    real_observable(
        "complex energy density",
        0.5 * (ab + ba),
        hamiltonian.hermiticity_tolerance(),
    )
}

pub fn local_expectation_complex(
    state: &ComplexPurifiedMps,
    operator: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<f64, ItebdError> {
    state.validate_topology()?;
    validate_local_operator(operator, state.a.phys.dim, tolerance)?;
    let a = one_site_expectation(operator, &state.a, &state.lambda_ba, &state.lambda_ab)?;
    let b = one_site_expectation(operator, &state.b, &state.lambda_ab, &state.lambda_ba)?;
    real_observable("complex local observable", 0.5 * (a + b), tolerance)
}

#[cfg(test)]
mod tests {
    use super::real_observable;
    use crate::itebd_error::ItebdError;
    use num_complex::Complex64;

    #[test]
    fn material_imaginary_residual_is_not_projected_to_real() {
        // Mutation caught: deleting the relative imaginary-residual check would silently return
        // the real projection of a materially complex expectation value.
        assert!(matches!(
            real_observable(
                "synthetic observable",
                Complex64::new(2.0, 5e-9),
                1e-12,
            ),
            Err(ItebdError::NonRealObservable {
                observable: "synthetic observable",
                imaginary,
                tolerance: 1e-12,
            }) if imaginary.to_bits() == 5e-9_f64.to_bits()
        ));
    }
}
