use super::tensor::complex_from_fn;
use super::{
    close_env, dominant_fixed_point_left, dominant_fixed_point_right, identity_env, transfer_step,
    transfer_step_right, ComplexPurifiedMps, ComplexSite, Env, FixedPoint,
};
use crate::contraction_pairwise::pairwise;
use crate::itebd_error::ItebdError;
use crate::tensor::{new_index, Idx, Tensor};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

const CANONICALIZE_STAGE: &str = "complex_canonicalize";
const FIXED_POINT_TOLERANCE: f64 = 1e-13;
const FIXED_POINT_ITERATE_TOLERANCE: f64 = 1e-13;
const FIXED_POINT_MAX_ITERATIONS: usize = 4000;
const EIGENVALUE_FLOOR: f64 = 1e-12;
const CANONICAL_GRAM_TOLERANCE: f64 = 1e-10;

fn tensor_error(message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage: CANONICALIZE_STAGE,
        message: message.to_string(),
    }
}

fn to_dmatrix(tensor: &Tensor, dimension: usize) -> Result<DMatrix<Complex64>, ItebdError> {
    let dimensions = tensor.dims();
    if dimensions != [dimension, dimension] {
        return Err(tensor_error(format!(
            "fixed-point matrix must be {dimension}x{dimension}, got {dimensions:?}"
        )));
    }
    let values = tensor
        .to_vec::<Complex64>()
        .map_err(|error| tensor_error(error))?;
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(tensor_error(
            "fixed-point matrix contains a non-finite value",
        ));
    }
    if values
        .iter()
        .all(|value| *value == Complex64::new(0.0, 0.0))
    {
        return Err(tensor_error("fixed-point matrix is zero"));
    }
    // The transfer environment stores ket as its first index and bra as its second, so its
    // dense action is E' = A^T E A^*. Canonical gauge formulas use the conventional
    // A^H rho A orientation; conjugating the Hermitian fixed point converts between them.
    Ok(DMatrix::from_vec(
        dimension,
        dimension,
        values.into_iter().map(|value| value.conj()).collect(),
    ))
}

fn normalized_environment(environment: Env) -> Result<(Env, Vec<Complex64>), ItebdError> {
    let values = environment
        .tensor
        .to_vec::<Complex64>()
        .map_err(|error| tensor_error(error))?;
    let norm = values.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(tensor_error(format!(
            "cannot normalize fixed-point environment with Frobenius norm {norm}"
        )));
    }
    let normalized = values
        .into_iter()
        .map(|value| value / norm)
        .collect::<Vec<_>>();
    let tensor = Tensor::from_dense(
        vec![environment.ket.clone(), environment.bra.clone()],
        normalized.clone(),
    )
    .map_err(|error| tensor_error(error))?;
    Ok((
        Env {
            tensor,
            ket: environment.ket,
            bra: environment.bra,
        },
        normalized,
    ))
}

fn refine_fixed_point(
    initial: FixedPoint,
    step: impl Fn(&Env) -> Result<Env, ItebdError>,
) -> Result<Tensor, ItebdError> {
    let (mut environment, mut previous) = normalized_environment(Env {
        tensor: initial.matrix,
        ket: initial.row,
        bra: initial.col,
    })?;
    for _ in 0..FIXED_POINT_MAX_ITERATIONS {
        let (next, current) = normalized_environment(step(&environment)?)?;
        let residual = current
            .iter()
            .zip(&previous)
            .map(|(left, right)| (left - right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        if !residual.is_finite() {
            return Err(tensor_error("fixed-point iterate residual is non-finite"));
        }
        environment = next;
        previous = current;
        if residual <= FIXED_POINT_ITERATE_TOLERANCE {
            return Ok(environment.tensor);
        }
    }
    Err(tensor_error(format!(
        "fixed-point matrix did not converge within {FIXED_POINT_MAX_ITERATIONS} refinements at residual tolerance {FIXED_POINT_ITERATE_TOLERANCE}"
    )))
}

fn hermiticity_residual(matrix: &DMatrix<Complex64>) -> f64 {
    (matrix - matrix.adjoint()).norm() / matrix.norm().max(1.0)
}

fn positive_factors(
    field: &'static str,
    density: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>), ItebdError> {
    let residual = hermiticity_residual(density);
    if !residual.is_finite() || residual > tolerance {
        return Err(ItebdError::NonHermitian {
            field,
            residual,
            tolerance,
        });
    }

    let dimension = density.nrows();
    // Use the pinned tensor backend: weakly coupled, differently scaled blocks can give
    // inaccurate eigenvectors in the previous dense solver even for positive Hermitian input.
    let tensor = Tensor::from_dense(
        vec![new_index(dimension), new_index(dimension)],
        density.as_slice().to_vec(),
    )
    .map_err(tensor_error)?;
    let eigen = tensor
        .hermitian_eigendecomposition(tolerance)
        .map_err(tensor_error)?;
    let eigenvectors = DMatrix::from_vec(
        dimension,
        dimension,
        eigen
            .eigenvectors
            .to_vec::<Complex64>()
            .map_err(tensor_error)?,
    );
    let mut square_root = DMatrix::<Complex64>::zeros(dimension, dimension);
    let mut inverse_square_root = DMatrix::<Complex64>::zeros(dimension, dimension);
    for (index, eigenvalue) in eigen.eigenvalues.iter().copied().enumerate() {
        if !eigenvalue.is_finite() {
            return Err(tensor_error(format!(
                "{field} eigendecomposition produced a non-finite eigenvalue at index {index}"
            )));
        }
        let value = eigenvalue.max(0.0);
        if value > EIGENVALUE_FLOOR {
            let root = value.sqrt();
            square_root[(index, index)] = Complex64::new(root, 0.0);
            inverse_square_root[(index, index)] = Complex64::new(root.recip(), 0.0);
        }
    }
    let factor = &eigenvectors * square_root;
    let inverse = inverse_square_root * eigenvectors.adjoint();
    Ok((factor, inverse))
}

fn normalized_singular_values(values: &DVector<f64>) -> Result<(Vec<f64>, f64), ItebdError> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(ItebdError::ZeroSchmidtNorm {
            stage: CANONICALIZE_STAGE,
        });
    }
    Ok((values.iter().map(|value| value / norm).collect(), norm))
}

fn gauge_svd(
    matrix: &DMatrix<Complex64>,
) -> Result<(DMatrix<Complex64>, DVector<f64>, DMatrix<Complex64>), ItebdError> {
    use tensor4all_core::{factorize_full_rank, Canonical, FactorizeAlg};
    let dimension = matrix.nrows();
    let row = new_index(dimension);
    let column = new_index(dimension);
    let tensor = Tensor::from_dense(vec![row.clone(), column], matrix.as_slice().to_vec())
        .map_err(tensor_error)?;
    // Canonicalization is an exact gauge rewrite. Do not apply the global SVD cutoff,
    // discard tiny singular values, or change the number of zero-support slots.
    let factors = factorize_full_rank(&tensor, &[row], FactorizeAlg::SVD, Canonical::Left)
        .map_err(tensor_error)?;
    let values = factors
        .singular_values
        .ok_or_else(|| tensor_error("canonical gauge SVD did not return singular values"))?;
    let u = DMatrix::from_vec(
        dimension,
        dimension,
        factors.left.to_vec::<Complex64>().map_err(tensor_error)?,
    );
    let mut v_adjoint = DMatrix::from_vec(
        dimension,
        dimension,
        factors.right.to_vec::<Complex64>().map_err(tensor_error)?,
    );
    // The factors are U and S*V^H. Retain all U columns: even a zero-Schmidt outgoing
    // slot participates in site_gram_scalar. A zero V^H row is safe, because that row
    // becomes an incoming Gamma leg weighted by its zero Schmidt value.
    for (row, &singular) in values.iter().enumerate() {
        for column in 0..dimension {
            v_adjoint[(row, column)] = if singular > 0.0 {
                v_adjoint[(row, column)] / singular
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }
    Ok((u, DVector::from_vec(values), v_adjoint))
}

fn bond_gauge(
    lambda: &[f64],
    left_density: &DMatrix<Complex64>,
    right_density: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<(Vec<f64>, DMatrix<Complex64>, DMatrix<Complex64>, f64), ItebdError> {
    let (x, x_inverse) = positive_factors(
        "complex right transfer fixed point",
        right_density,
        tolerance,
    )?;
    let (left_factor, left_inverse_factor) =
        positive_factors("complex left transfer fixed point", left_density, tolerance)?;
    let y = left_factor.adjoint();
    let y_inverse = left_inverse_factor.adjoint();
    let lambda_matrix = DMatrix::from_diagonal(&DVector::from_iterator(
        lambda.len(),
        lambda
            .iter()
            .copied()
            .map(|value| Complex64::new(value, 0.0)),
    ));
    let (u, singular_values, v_adjoint) = gauge_svd(&(&y * &x))?;
    let (new_lambda, schmidt_norm) = normalized_singular_values(&singular_values)?;
    let left_gauge = y_inverse * u;
    let right_gauge = v_adjoint * x_inverse * lambda_matrix;
    Ok((new_lambda, right_gauge, left_gauge, schmidt_norm))
}

fn apply_left_matrix_to_leg(
    tensor: &Tensor,
    bond: &Idx,
    matrix: &DMatrix<Complex64>,
    new_bond: &Idx,
) -> Result<Tensor, ItebdError> {
    if matrix.nrows() != new_bond.dim || matrix.ncols() != bond.dim {
        return Err(tensor_error(
            "left gauge matrix dimensions do not match bond",
        ));
    }
    let matrix_tensor = complex_from_fn(&[bond.clone(), new_bond.clone()], |index| {
        matrix[(index[1], index[0])]
    })?;
    let transformed = pairwise(tensor, &matrix_tensor).map_err(|error| tensor_error(error))?;
    restore_replaced_index_order(&transformed, tensor, bond, new_bond)
}

fn apply_right_matrix_to_leg(
    tensor: &Tensor,
    bond: &Idx,
    matrix: &DMatrix<Complex64>,
    new_bond: &Idx,
) -> Result<Tensor, ItebdError> {
    if matrix.nrows() != bond.dim || matrix.ncols() != new_bond.dim {
        return Err(tensor_error(
            "right gauge matrix dimensions do not match bond",
        ));
    }
    let matrix_tensor = complex_from_fn(&[bond.clone(), new_bond.clone()], |index| {
        matrix[(index[0], index[1])]
    })?;
    let transformed = pairwise(tensor, &matrix_tensor).map_err(|error| tensor_error(error))?;
    restore_replaced_index_order(&transformed, tensor, bond, new_bond)
}

fn restore_replaced_index_order(
    transformed: &Tensor,
    original: &Tensor,
    bond: &Idx,
    new_bond: &Idx,
) -> Result<Tensor, ItebdError> {
    let requested = original
        .indices
        .iter()
        .map(|index| {
            if index.id == bond.id {
                new_bond.clone()
            } else {
                index.clone()
            }
        })
        .collect::<Vec<_>>();
    transformed
        .permute_indices(&requested)
        .map_err(|error| tensor_error(error))
}

fn canonicalize_reference_bond(
    state: &mut ComplexPurifiedMps,
    tolerance: f64,
) -> Result<f64, ItebdError> {
    let left_fixed_point =
        dominant_fixed_point_left(state, FIXED_POINT_TOLERANCE, FIXED_POINT_MAX_ITERATIONS)?;
    let right_fixed_point =
        dominant_fixed_point_right(state, FIXED_POINT_TOLERANCE, FIXED_POINT_MAX_ITERATIONS)?;
    let left_fixed_point = refine_fixed_point(left_fixed_point, |environment| {
        let after_a = transfer_step(environment, &state.a, &state.lambda_ba, None)?;
        transfer_step(&after_a, &state.b, &state.lambda_ab, None)
    })?;
    let right_fixed_point = refine_fixed_point(right_fixed_point, |environment| {
        let after_b = transfer_step_right(environment, &state.b, &state.lambda_ab)?;
        transfer_step_right(&after_b, &state.a, &state.lambda_ba)
    })?;
    let dimension = state.a.left.dim;
    let left_density = to_dmatrix(&left_fixed_point, dimension)?;
    let right_density = to_dmatrix(&right_fixed_point, dimension)?;
    let (new_lambda, right_gauge, left_gauge, schmidt_norm) =
        bond_gauge(&state.lambda_ba, &left_density, &right_density, tolerance)?;
    let new_bond = new_index(new_lambda.len());
    let new_a_gamma =
        apply_left_matrix_to_leg(&state.a.gamma, &state.a.left, &right_gauge, &new_bond)?;
    let new_b_gamma =
        apply_right_matrix_to_leg(&state.b.gamma, &state.b.right, &left_gauge, &new_bond)?;

    state.a = ComplexSite {
        gamma: new_a_gamma,
        left: new_bond.clone(),
        phys: state.a.phys.clone(),
        anc: state.a.anc.clone(),
        right: state.a.right.clone(),
    };
    state.b = ComplexSite {
        gamma: new_b_gamma,
        left: state.b.left.clone(),
        phys: state.b.phys.clone(),
        anc: state.b.anc.clone(),
        right: new_bond.clone(),
    };
    state.lambda_ba = new_lambda;
    state.lambda_bond_ba = new_bond;
    state.validate_topology()?;
    Ok(schmidt_norm)
}

fn rotate(state: ComplexPurifiedMps) -> ComplexPurifiedMps {
    ComplexPurifiedMps {
        a: state.b,
        b: state.a,
        lambda_ab: state.lambda_ba,
        lambda_bond_ab: state.lambda_bond_ba,
        lambda_ba: state.lambda_ab,
        lambda_bond_ba: state.lambda_bond_ab,
    }
}

fn scale_tensor(tensor: &Tensor, factor: f64) -> Result<Tensor, ItebdError> {
    let values = tensor
        .to_vec::<Complex64>()
        .map_err(|error| tensor_error(error))?
        .into_iter()
        .map(|value| value * factor)
        .collect::<Vec<_>>();
    Tensor::from_dense(tensor.indices.clone(), values).map_err(|error| tensor_error(error))
}

fn reference_cell_norm(state: &ComplexPurifiedMps, tolerance: f64) -> Result<f64, ItebdError> {
    let environment = identity_env(state.a.left.dim)?;
    let environment = transfer_step(&environment, &state.a, &state.lambda_ba, None)?;
    let environment = transfer_step(&environment, &state.b, &state.lambda_ab, None)?;
    let norm = close_env(&environment, &state.lambda_ba)?;
    if !norm.re.is_finite() || !norm.im.is_finite() || norm.re <= 0.0 {
        return Err(tensor_error(format!("invalid complex cell norm {norm:?}")));
    }
    if norm.im.abs() > tolerance * norm.re.abs().max(1.0) {
        return Err(ItebdError::NonRealObservable {
            observable: "complex canonicalization cell norm",
            imaginary: norm.im,
            tolerance,
        });
    }
    Ok(norm.re)
}

fn site_gram_scalar(site: &ComplexSite, lambda_left: &[f64]) -> Result<f64, ItebdError> {
    let environment = identity_env(site.left.dim)?;
    let gram = transfer_step(&environment, site, lambda_left, None)?;
    let matrix = gram
        .tensor
        .to_vec::<Complex64>()
        .map_err(|error| tensor_error(error))?;
    let dimension = site.right.dim;
    let scalar = (0..dimension)
        .map(|index| matrix[index * (dimension + 1)].re)
        .sum::<f64>()
        / dimension as f64;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(tensor_error(format!(
            "invalid one-site canonical Gram scalar {scalar}"
        )));
    }
    Ok(scalar)
}

pub fn canonicalize_complex(
    state: &mut ComplexPurifiedMps,
    tolerance: f64,
) -> Result<f64, ItebdError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ItebdError::InvalidTolerance {
            name: "complex canonicalization tolerance",
            value: tolerance,
        });
    }
    state.validate_topology()?;
    if state.a.left.dim == 1 && state.a.right.dim == 1 {
        return Ok(0.0);
    }

    let first_schmidt_norm = canonicalize_reference_bond(state, tolerance)?;
    let physical_dimension = state.a.phys.dim;
    let replacement = ComplexPurifiedMps::infinite_temperature(physical_dimension)?;
    let mut rotated = rotate(std::mem::replace(state, replacement));
    let second_schmidt_norm = canonicalize_reference_bond(&mut rotated, tolerance)?;
    *state = rotate(rotated);

    // Each SVD normalizes its new Schmidt vector by n, scaling the represented cell by 1/n.
    // Compensate the two known factors exactly instead of inferring them from another transfer
    // solve. This is pure gauge bookkeeping, so the returned log-norm contribution remains zero.
    let compensation = (first_schmidt_norm * second_schmidt_norm).sqrt();
    if !compensation.is_finite() || compensation <= 0.0 {
        return Err(tensor_error(format!(
            "invalid canonical Schmidt compensation {compensation}"
        )));
    }
    state.a.gamma = scale_tensor(&state.a.gamma, compensation)?;
    state.b.gamma = scale_tensor(&state.b.gamma, compensation)?;
    let _cell_norm = reference_cell_norm(state, tolerance)?;

    // Fixed-point eigenvectors have independent scalar normalizations on the two sublattices.
    // First balance their Gram scalars with reciprocal site factors, which is a pure gauge and
    // preserves the represented cell. A generic input may additionally carry a material common
    // transfer scale; normalize that scale and return the removed two-site-cell log amplitude.
    let a_gram_scalar = site_gram_scalar(&state.a, &state.lambda_ba)?;
    let b_gram_scalar = site_gram_scalar(&state.b, &state.lambda_ab)?;
    let common_log_scale = 0.5 * (a_gram_scalar.ln() + b_gram_scalar.ln());
    let common_gram_scalar = common_log_scale.exp();
    let reciprocal_gauge = (0.25 * (b_gram_scalar.ln() - a_gram_scalar.ln())).exp();
    if !common_log_scale.is_finite()
        || !common_gram_scalar.is_finite()
        || !reciprocal_gauge.is_finite()
        || reciprocal_gauge <= 0.0
    {
        return Err(tensor_error(format!(
            "invalid canonical Gram scales A={a_gram_scalar} B={b_gram_scalar}"
        )));
    }
    state.a.gamma = scale_tensor(&state.a.gamma, reciprocal_gauge)?;
    state.b.gamma = scale_tensor(&state.b.gamma, reciprocal_gauge.recip())?;

    let removed_log_norm = if (common_gram_scalar - 1.0).abs() <= CANONICAL_GRAM_TOLERANCE {
        0.0
    } else {
        let common_site_scale = (-0.5 * common_log_scale).exp();
        if !common_site_scale.is_finite() || common_site_scale <= 0.0 {
            return Err(tensor_error(format!(
                "invalid common canonical site scale {common_site_scale}"
            )));
        }
        state.a.gamma = scale_tensor(&state.a.gamma, common_site_scale)?;
        state.b.gamma = scale_tensor(&state.b.gamma, common_site_scale)?;
        common_log_scale
    };
    state.validate_topology()?;
    Ok(removed_log_norm)
}

#[cfg(test)]
mod tests {
    use super::{positive_factors, to_dmatrix, CANONICALIZE_STAGE};
    use crate::itebd_error::ItebdError;
    use crate::tensor::{new_index, Tensor};
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex64;

    #[test]
    fn zero_schmidt_support_does_not_introduce_a_normalization_change() {
        // The inactive outgoing B column has unit Gram norm but zero physical weight.
        // Dropping its U vector would spuriously change the canonical log normalization.
        use super::{canonicalize_complex, reference_cell_norm, ComplexPurifiedMps, ComplexSite};
        let ba = new_index(2);
        let ab = new_index(1);
        let a_phys = new_index(2);
        let b_phys = new_index(2);
        let a_anc = new_index(2);
        let b_anc = new_index(2);
        let mut a_data = vec![Complex64::new(0.0, 0.0); 8];
        a_data[0] = Complex64::new(1.0, 0.0);
        let mut b_data = vec![Complex64::new(0.0, 0.0); 8];
        b_data[0] = Complex64::new(1.0, 0.0);
        b_data[5] = Complex64::new(1.0, 0.0);
        let mut state = ComplexPurifiedMps {
            a: ComplexSite {
                gamma: Tensor::from_dense(
                    vec![ba.clone(), a_phys.clone(), a_anc.clone(), ab.clone()],
                    a_data,
                )
                .unwrap(),
                left: ba.clone(),
                phys: a_phys,
                anc: a_anc,
                right: ab.clone(),
            },
            b: ComplexSite {
                gamma: Tensor::from_dense(
                    vec![ab.clone(), b_phys.clone(), b_anc.clone(), ba.clone()],
                    b_data,
                )
                .unwrap(),
                left: ab.clone(),
                phys: b_phys,
                anc: b_anc,
                right: ba.clone(),
            },
            lambda_ab: vec![1.0],
            lambda_ba: vec![1.0, 0.0],
            lambda_bond_ab: ab,
            lambda_bond_ba: ba,
        };
        assert_eq!(reference_cell_norm(&state, 1e-12).unwrap(), 1.0);
        let log_norm = canonicalize_complex(&mut state, 1e-12).unwrap();
        assert!(
            log_norm.abs() < 1e-12,
            "spurious zero-support normalization: {log_norm}"
        );
        assert!((reference_cell_norm(&state, 1e-12).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(state.lambda_ab.len(), 1);
        assert_eq!(state.lambda_ba.len(), 2);
        let a = state.a.gamma.to_vec::<Complex64>().unwrap();
        let b = state.b.gamma.to_vec::<Complex64>().unwrap();
        let mut periodic_norm = 0.0;
        let mut both_phys_zero = 0.0;
        for physical_a in 0..2 {
            for physical_b in 0..2 {
                for ancilla_a in 0..2 {
                    for ancilla_b in 0..2 {
                        let amplitude: Complex64 = (0..2)
                            .map(|bond| {
                                state.lambda_ba[bond]
                                    * a[bond + 2 * (physical_a + 2 * ancilla_a)]
                                    * state.lambda_ab[0]
                                    * b[physical_b + 2 * ancilla_b + 4 * bond]
                            })
                            .sum();
                        periodic_norm += amplitude.norm_sqr();
                        if physical_a == 0 && physical_b == 0 {
                            both_phys_zero += amplitude.norm_sqr();
                        }
                    }
                }
            }
        }
        assert!((periodic_norm - 1.0).abs() < 1e-12);
        assert!((both_phys_zero - 1.0).abs() < 1e-12);
    }

    #[test]
    fn positive_factors_preserve_nonreal_hermitian_density() {
        let q = DMatrix::from_fn(3, 3, |row, column| {
            Complex64::from_polar(
                1.0 / 3.0_f64.sqrt(),
                2.0 * std::f64::consts::PI * (row * column) as f64 / 3.0,
            )
        });
        let spectrum = DVector::from_vec(vec![1.0.into(), 0.03.into(), 0.002.into()]);
        let density = &q * DMatrix::from_diagonal(&spectrum) * q.adjoint();
        assert!(density.iter().any(|value| value.im.abs() > 1e-3));
        let (factor, inverse) = positive_factors("nonreal test density", &density, 1e-12).unwrap();
        assert!((&factor * factor.adjoint() - &density).norm() < 1e-14);
        assert!((&inverse * density * inverse.adjoint() - DMatrix::identity(3, 3)).norm() < 1e-12);
    }

    #[test]
    fn gauge_svd_retains_tiny_values_and_zero_support_slots() {
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1e-16),
            Complex64::new(0.0, 0.0),
        ]));
        let (u, s, vh) = super::gauge_svd(&matrix).unwrap();
        assert_eq!(s.len(), 3);
        assert!((s[1] - 1e-16).abs() < 1e-30);
        assert_eq!(s[2], 0.0);
        let diagonal = DMatrix::from_diagonal(&s.map(|s| Complex64::new(s, 0.0)));
        assert!((&matrix - u * diagonal * vh).norm() < 1e-30);
    }

    #[test]
    fn gauge_svd_reconstructs_captured_twisted_xx_matrix() {
        let values: Vec<[f64; 2]> =
            serde_json::from_str(include_str!("../../tests/fixtures/canonical-svd-xx.json"))
                .unwrap();
        let matrix = DMatrix::from_iterator(
            10,
            10,
            values.into_iter().map(|[re, im]| Complex64::new(re, im)),
        );
        let (u, s, vh) = super::gauge_svd(&matrix).unwrap();
        let diagonal = DMatrix::from_diagonal(&s.map(|s| Complex64::new(s, 0.0)));
        let residual = (&matrix - &u * diagonal * &vh).norm();
        assert!(residual < 1e-14, "SVD reconstruction residual={residual:e}");
        assert!((u.adjoint() * &u - DMatrix::identity(10, 10)).norm() < 1e-12);
        assert!((&vh * vh.adjoint() - DMatrix::identity(10, 10)).norm() < 1e-12);
    }

    #[test]
    fn positive_factors_reconstruct_weakly_coupled_density() {
        for imaginary in [0.0, 1e-18] {
            let mut density = DMatrix::<Complex64>::zeros(3, 3);
            density[(0, 0)] = 1.0.into();
            density[(1, 1)] = 1e-6.into();
            density[(2, 2)] = 1e-3.into();
            density[(1, 2)] = Complex64::new(1e-18, imaginary);
            density[(2, 1)] = density[(1, 2)].conj();
            let (factor, inverse) = positive_factors("test density", &density, 1e-12).unwrap();
            let residual = (&factor * factor.adjoint() - &density).norm();
            assert!(
                residual < 1e-14,
                "imaginary={imaginary} factor residual={residual:e}"
            );
            let identity = &inverse * &density * inverse.adjoint();
            assert!((identity - DMatrix::identity(3, 3)).norm() < 1e-12);
        }
    }

    #[test]
    fn fixed_point_conversion_rejects_zero_data() {
        // Mutation caught: a zero density cannot define invertible canonical gauge factors.
        let row = new_index(2);
        let column = new_index(2);
        let tensor =
            Tensor::from_dense(vec![row, column], vec![Complex64::new(0.0, 0.0); 4]).unwrap();
        assert!(matches!(
            to_dmatrix(&tensor, 2),
            Err(ItebdError::TensorOperation {
                stage: CANONICALIZE_STAGE,
                ..
            })
        ));
    }

    #[test]
    fn fixed_point_conversion_rejects_nonfinite_data() {
        // Mutation caught: passing NaN to Hermitian eigendecomposition would make its failure
        // backend-dependent instead of preserving the typed canonicalization boundary.
        let row = new_index(2);
        let column = new_index(2);
        let tensor = Tensor::from_dense(
            vec![row, column],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();
        assert!(matches!(
            to_dmatrix(&tensor, 2),
            Err(ItebdError::TensorOperation {
                stage: CANONICALIZE_STAGE,
                ..
            })
        ));
    }

    #[test]
    fn positive_factors_reject_nonhermitian_density_without_symmetrizing() {
        // Mutation caught: replacing validation with (rho + rho^H)/2 would accept this material
        // anti-Hermitian component and hide a transfer-orientation defect.
        let density = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.2, 0.7),
                Complex64::new(0.2, 0.1),
                Complex64::new(0.5, 0.0),
            ],
        );
        assert!(matches!(
            positive_factors("test fixed point", &density, 1e-12),
            Err(ItebdError::NonHermitian {
                field: "test fixed point",
                tolerance,
                ..
            }) if tolerance == 1e-12
        ));
    }
}
