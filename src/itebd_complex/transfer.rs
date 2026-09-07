use super::tensor::{complex_from_fn, complex_relabel, complex_scalar, complex_scale_bond};
use super::{ComplexPurifiedMps, ComplexSite};
use crate::itebd_error::ItebdError;
use crate::tensor::{new_index, Idx, Tensor};
use crate::contraction_pairwise::{pairwise, pairwise_with_conjugation};
use nalgebra::DMatrix;
use num_complex::Complex64;

const TRANSFER_STAGE: &str = "complex_transfer";
const FIXED_POINT_STAGE: &str = "complex_transfer_fixed_point";

fn tensor_error(stage: &'static str, message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage,
        message: message.to_string(),
    }
}

fn transfer_contract(
    stage: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, ItebdError> {
    pairwise(lhs, rhs).map_err(|error| tensor_error(stage, error))
}

fn transfer_contract_conjugated_lhs(
    stage: &'static str,
    bra: &Tensor,
    ket_network: &Tensor,
) -> Result<Tensor, ItebdError> {
    pairwise_with_conjugation(bra, ket_network, true, false)
        .map_err(|error| tensor_error(stage, error))
}

fn restore_transfer_order(
    stage: &'static str,
    tensor: &Tensor,
    indices: &[Idx],
) -> Result<Tensor, ItebdError> {
    tensor
        .permute_indices(indices)
        .map_err(|error| tensor_error(stage, error))
}

#[derive(Clone)]
pub struct Env {
    pub tensor: Tensor,
    pub ket: Idx,
    pub bra: Idx,
}

pub fn identity_env(dimension: usize) -> Result<Env, ItebdError> {
    let ket = new_index(dimension);
    let bra = new_index(dimension);
    let tensor = complex_from_fn(&[ket.clone(), bra.clone()], |index| {
        if index[0] == index[1] {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })?;
    Ok(Env { tensor, ket, bra })
}

fn left_canonical(site: &ComplexSite, lambda_left: &[f64]) -> Result<Tensor, ItebdError> {
    complex_scale_bond(&site.gamma, &site.left, lambda_left)
}

fn validate_one_site_operator(
    site: &ComplexSite,
    operator: &DMatrix<Complex64>,
) -> Result<(), ItebdError> {
    let dimension = site.phys.dim;
    if operator.nrows() != dimension || operator.ncols() != dimension {
        return Err(tensor_error(
            TRANSFER_STAGE,
            format!(
                "one-site operator must be {dimension}x{dimension}, got {}x{}",
                operator.nrows(),
                operator.ncols()
            ),
        ));
    }
    Ok(())
}

pub fn transfer_step(
    env: &Env,
    site: &ComplexSite,
    lambda_left: &[f64],
    operator: Option<&DMatrix<Complex64>>,
) -> Result<Env, ItebdError> {
    if let Some(operator) = operator {
        validate_one_site_operator(site, operator)?;
    }
    let canonical = left_canonical(site, lambda_left)?;
    let ket_right = new_index(site.right.dim);
    let bra_right = new_index(site.right.dim);

    // The ket shares env.ket and retains ket_right. The bra shares env.bra and all traced
    // physical/ancilla identities, retains bra_right, and is conjugated by tensor4all at the
    // binary contraction boundary rather than materialized as a separate tensor.
    let ket = complex_relabel(
        &complex_relabel(&canonical, &site.left, &env.ket)?,
        &site.right,
        &ket_right,
    )?;
    let tensor = match operator {
        None => {
            let bra = complex_relabel(
                &complex_relabel(&canonical, &site.left, &env.bra)?,
                &site.right,
                &bra_right,
            )?;
            let environment_ket =
                transfer_contract("complex transfer step environment ket", &env.tensor, &ket)?;
            let output = transfer_contract_conjugated_lhs(
                "complex transfer step bra",
                &bra,
                &environment_ket,
            )?;
            restore_transfer_order(
                "complex transfer step output order",
                &output,
                &[ket_right.clone(), bra_right.clone()],
            )?
        }
        Some(operator) => {
            let physical_out = new_index(site.phys.dim);
            // [physical_in, physical_out], storing O[physical_out, physical_in].
            let operator_tensor =
                complex_from_fn(&[site.phys.clone(), physical_out.clone()], |index| {
                    operator[(index[1], index[0])]
                })?;
            let bra = complex_relabel(
                &complex_relabel(
                    &complex_relabel(&canonical, &site.left, &env.bra)?,
                    &site.right,
                    &bra_right,
                )?,
                &site.phys,
                &physical_out,
            )?;
            let environment_ket = transfer_contract(
                "complex transfer step operator environment ket",
                &env.tensor,
                &ket,
            )?;
            let operated = transfer_contract(
                "complex transfer step operator",
                &environment_ket,
                &operator_tensor,
            )?;
            let output = transfer_contract_conjugated_lhs(
                "complex transfer step operator bra",
                &bra,
                &operated,
            )?;
            restore_transfer_order(
                "complex transfer step operator output order",
                &output,
                &[ket_right.clone(), bra_right.clone()],
            )?
        }
    };
    Ok(Env {
        tensor,
        ket: ket_right,
        bra: bra_right,
    })
}

pub fn transfer_step_op2(
    env: &Env,
    site_i: &ComplexSite,
    lambda_i: &[f64],
    site_j: &ComplexSite,
    lambda_j: &[f64],
    operator: &DMatrix<Complex64>,
) -> Result<Env, ItebdError> {
    if site_i.phys.dim != site_j.phys.dim {
        return Err(tensor_error(
            TRANSFER_STAGE,
            "two-site transfer requires equal physical dimensions",
        ));
    }
    let dimension = site_i.phys.dim;
    let operator_dimension = dimension
        .checked_mul(dimension)
        .ok_or_else(|| tensor_error(TRANSFER_STAGE, "two-site operator dimension overflow"))?;
    if operator.nrows() != operator_dimension || operator.ncols() != operator_dimension {
        return Err(tensor_error(
            TRANSFER_STAGE,
            format!(
                "two-site operator must be {operator_dimension}x{operator_dimension}, got {}x{}",
                operator.nrows(),
                operator.ncols()
            ),
        ));
    }

    let canonical_i = left_canonical(site_i, lambda_i)?;
    let canonical_j = left_canonical(site_j, lambda_j)?;
    let ket_middle = new_index(site_i.right.dim);
    let ket_right = new_index(site_j.right.dim);
    let bra_middle = new_index(site_i.right.dim);
    let bra_right = new_index(site_j.right.dim);

    let ket_i = complex_relabel(
        &complex_relabel(&canonical_i, &site_i.left, &env.ket)?,
        &site_i.right,
        &ket_middle,
    )?;
    let ket_j = complex_relabel(
        &complex_relabel(&canonical_j, &site_j.left, &ket_middle)?,
        &site_j.right,
        &ket_right,
    )?;

    let physical_i_out = new_index(dimension);
    let physical_j_out = new_index(dimension);
    // The first physical index is fastest: flat = physical_i + d * physical_j.
    let operator_tensor = complex_from_fn(
        &[
            site_i.phys.clone(),
            site_j.phys.clone(),
            physical_i_out.clone(),
            physical_j_out.clone(),
        ],
        |index| {
            let input = index[0] + dimension * index[1];
            let output = index[2] + dimension * index[3];
            operator[(output, input)]
        },
    )?;

    let bra_i = complex_relabel(
        &complex_relabel(
            &complex_relabel(&canonical_i, &site_i.left, &env.bra)?,
            &site_i.right,
            &bra_middle,
        )?,
        &site_i.phys,
        &physical_i_out,
    )?;
    let bra_j = complex_relabel(
        &complex_relabel(
            &complex_relabel(&canonical_j, &site_j.left, &bra_middle)?,
            &site_j.right,
            &bra_right,
        )?,
        &site_j.phys,
        &physical_j_out,
    )?;

    let environment_ket_i = transfer_contract(
        "complex transfer op2 environment ket i",
        &env.tensor,
        &ket_i,
    )?;
    let environment_ket_pair =
        transfer_contract("complex transfer op2 ket j", &environment_ket_i, &ket_j)?;
    let operated = transfer_contract(
        "complex transfer op2 operator",
        &environment_ket_pair,
        &operator_tensor,
    )?;
    let bra_i_applied =
        transfer_contract_conjugated_lhs("complex transfer op2 bra i", &bra_i, &operated)?;
    let output =
        transfer_contract_conjugated_lhs("complex transfer op2 bra j", &bra_j, &bra_i_applied)?;
    let tensor = restore_transfer_order(
        "complex transfer op2 output order",
        &output,
        &[ket_right.clone(), bra_right.clone()],
    )?;
    Ok(Env {
        tensor,
        ket: ket_right,
        bra: bra_right,
    })
}

pub fn transfer_step_right(
    env: &Env,
    site: &ComplexSite,
    lambda_left: &[f64],
) -> Result<Env, ItebdError> {
    let canonical = left_canonical(site, lambda_left)?;
    let output_row = new_index(site.left.dim);
    let output_column = new_index(site.left.dim);

    // This is the Hilbert--Schmidt adjoint of transfer_step. The conjugated bra supplies the
    // retained row index and shares env.ket; the ket supplies the retained column and shares
    // env.bra. Swapping these identities would compute a transpose, not T^dagger.
    let bra = complex_relabel(
        &complex_relabel(&canonical, &site.right, &env.ket)?,
        &site.left,
        &output_row,
    )?;
    let ket = complex_relabel(
        &complex_relabel(&canonical, &site.right, &env.bra)?,
        &site.left,
        &output_column,
    )?;
    let bra_environment = transfer_contract_conjugated_lhs(
        "complex transfer step right bra environment",
        &bra,
        &env.tensor,
    )?;
    let output = transfer_contract("complex transfer step right ket", &bra_environment, &ket)?;
    let tensor = restore_transfer_order(
        "complex transfer step right output order",
        &output,
        &[output_row.clone(), output_column.clone()],
    )?;
    Ok(Env {
        tensor,
        ket: output_row,
        bra: output_column,
    })
}

/// Apply a Hermitian two-site operator while moving a complex environment from right to left.
/// The first physical index is fastest: `flat = physical_i + d * physical_j`.
pub(crate) fn transfer_step_op2_right(
    env: &Env,
    site_i: &ComplexSite,
    lambda_i: &[f64],
    site_j: &ComplexSite,
    lambda_j: &[f64],
    operator: &DMatrix<Complex64>,
) -> Result<Env, ItebdError> {
    if site_i.phys.dim != site_j.phys.dim {
        return Err(tensor_error(
            TRANSFER_STAGE,
            "two-site right transfer requires equal physical dimensions",
        ));
    }
    let dimension = site_i.phys.dim;
    let operator_dimension = dimension.checked_mul(dimension).ok_or_else(|| {
        tensor_error(TRANSFER_STAGE, "two-site right operator dimension overflow")
    })?;
    if operator.nrows() != operator_dimension || operator.ncols() != operator_dimension {
        return Err(tensor_error(
            TRANSFER_STAGE,
            format!(
                "two-site right operator must be {operator_dimension}x{operator_dimension}, got {}x{}",
                operator.nrows(),
                operator.ncols()
            ),
        ));
    }

    let canonical_i = left_canonical(site_i, lambda_i)?;
    let canonical_j = left_canonical(site_j, lambda_j)?;
    let ket_middle = new_index(site_i.right.dim);
    let ket_left = new_index(site_i.left.dim);
    let bra_middle = new_index(site_i.right.dim);
    let bra_left = new_index(site_i.left.dim);

    let physical_i_out = new_index(dimension);
    let physical_j_out = new_index(dimension);
    let operator_tensor = complex_from_fn(
        &[
            site_i.phys.clone(),
            site_j.phys.clone(),
            physical_i_out.clone(),
            physical_j_out.clone(),
        ],
        |index| {
            let input = index[0] + dimension * index[1];
            let output = index[2] + dimension * index[3];
            operator[(output, input)]
        },
    )?;

    // The unconjugated ket shares env.bra; the bra shares env.ket and is conjugated at the
    // pairwise boundary. This preserves the [ket_left, bra_left] retained-index convention.
    let ket_j = complex_relabel(
        &complex_relabel(&canonical_j, &site_j.right, &env.bra)?,
        &site_j.left,
        &bra_middle,
    )?;
    let ket_i = complex_relabel(
        &complex_relabel(&canonical_i, &site_i.right, &bra_middle)?,
        &site_i.left,
        &bra_left,
    )?;
    let bra_j = complex_relabel(
        &complex_relabel(
            &complex_relabel(&canonical_j, &site_j.right, &env.ket)?,
            &site_j.left,
            &ket_middle,
        )?,
        &site_j.phys,
        &physical_j_out,
    )?;
    let bra_i = complex_relabel(
        &complex_relabel(
            &complex_relabel(&canonical_i, &site_i.right, &ket_middle)?,
            &site_i.left,
            &ket_left,
        )?,
        &site_i.phys,
        &physical_i_out,
    )?;

    let environment_ket_j = transfer_contract(
        "complex transfer op2 right environment ket j",
        &env.tensor,
        &ket_j,
    )?;
    let environment_ket_pair = transfer_contract(
        "complex transfer op2 right ket i",
        &environment_ket_j,
        &ket_i,
    )?;
    let operated = transfer_contract(
        "complex transfer op2 right operator",
        &environment_ket_pair,
        &operator_tensor,
    )?;
    let bra_j_applied =
        transfer_contract_conjugated_lhs("complex transfer op2 right bra j", &bra_j, &operated)?;
    let output = transfer_contract_conjugated_lhs(
        "complex transfer op2 right bra i",
        &bra_i,
        &bra_j_applied,
    )?;
    let tensor = restore_transfer_order(
        "complex transfer op2 right output order",
        &output,
        &[ket_left.clone(), bra_left.clone()],
    )?;
    Ok(Env {
        tensor,
        ket: ket_left,
        bra: bra_left,
    })
}

pub struct FixedPoint {
    pub matrix: Tensor,
    pub row: Idx,
    pub col: Idx,
    pub eigenvalue: f64,
}

fn frobenius_norm(tensor: &Tensor) -> Result<f64, ItebdError> {
    let norm = tensor
        .to_vec::<Complex64>()
        .map_err(|error| tensor_error(FIXED_POINT_STAGE, error))?
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(tensor_error(
            FIXED_POINT_STAGE,
            format!("cannot normalize fixed-point iterate with Frobenius norm {norm}"),
        ));
    }
    Ok(norm)
}

fn power_iterate(
    mut env: Env,
    step: impl Fn(&Env) -> Result<Env, ItebdError>,
    tolerance: f64,
    max_iter: usize,
) -> Result<FixedPoint, ItebdError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ItebdError::InvalidTolerance {
            name: "fixed-point tolerance",
            value: tolerance,
        });
    }
    if max_iter == 0 {
        return Err(tensor_error(
            FIXED_POINT_STAGE,
            "fixed-point iteration requires max_iter > 0",
        ));
    }

    let mut previous_norm = 0.0;
    let mut eigenvalue = 1.0;
    let mut converged = false;
    for _ in 0..max_iter {
        let next = step(&env)?;
        let norm = frobenius_norm(&next.tensor)?;
        eigenvalue = norm;
        let normalized = next
            .tensor
            .to_vec::<Complex64>()
            .map_err(|error| tensor_error(FIXED_POINT_STAGE, error))?
            .into_iter()
            .map(|value| value / norm)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_dense(vec![next.ket.clone(), next.bra.clone()], normalized)
            .map_err(|error| tensor_error(FIXED_POINT_STAGE, error))?;
        env = Env {
            tensor,
            ket: next.ket,
            bra: next.bra,
        };
        if (eigenvalue - previous_norm).abs() < tolerance {
            converged = true;
            break;
        }
        previous_norm = eigenvalue;
    }
    if !converged {
        return Err(tensor_error(
            FIXED_POINT_STAGE,
            format!(
                "fixed-point iteration did not converge within {max_iter} iterations at tolerance {tolerance}"
            ),
        ));
    }

    let row = new_index(env.ket.dim);
    let col = new_index(env.bra.dim);
    let matrix = complex_relabel(
        &complex_relabel(&env.tensor, &env.ket, &row)?,
        &env.bra,
        &col,
    )?;
    Ok(FixedPoint {
        matrix,
        row,
        col,
        eigenvalue,
    })
}

pub fn dominant_fixed_point_left(
    state: &ComplexPurifiedMps,
    tolerance: f64,
    max_iter: usize,
) -> Result<FixedPoint, ItebdError> {
    state.validate_topology()?;
    let initial = identity_env(state.a.left.dim)?;
    power_iterate(
        initial,
        |env| {
            let after_a = transfer_step(env, &state.a, &state.lambda_ba, None)?;
            transfer_step(&after_a, &state.b, &state.lambda_ab, None)
        },
        tolerance,
        max_iter,
    )
}

pub fn dominant_fixed_point_right(
    state: &ComplexPurifiedMps,
    tolerance: f64,
    max_iter: usize,
) -> Result<FixedPoint, ItebdError> {
    state.validate_topology()?;
    let initial = identity_env(state.b.right.dim)?;
    power_iterate(
        initial,
        |env| {
            let after_b = transfer_step_right(env, &state.b, &state.lambda_ab)?;
            transfer_step_right(&after_b, &state.a, &state.lambda_ba)
        },
        tolerance,
        max_iter,
    )
}

pub fn close_env(env: &Env, lambda_right: &[f64]) -> Result<Complex64, ItebdError> {
    if lambda_right.len() != env.ket.dim || lambda_right.len() != env.bra.dim {
        return Err(tensor_error(
            TRANSFER_STAGE,
            format!(
                "closure Schmidt count {} does not match environment dimensions {}x{}",
                lambda_right.len(),
                env.ket.dim,
                env.bra.dim
            ),
        ));
    }
    let cap = complex_from_fn(&[env.ket.clone(), env.bra.clone()], |index| {
        if index[0] == index[1] {
            Complex64::new(lambda_right[index[0]] * lambda_right[index[0]], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })?;
    let closed = transfer_contract("complex transfer closure", &env.tensor, &cap)?;
    complex_scalar(&closed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itebd_complex::tensor::complex_from_fn;
    use crate::itebd_complex::ComplexSite;
    use crate::itebd_error::ItebdError;
    use crate::tensor::{new_index, Tensor};
    use crate::contraction_pairwise::{
        pairwise_call_count, pairwise_operand_conjugation_count, reset_pairwise_counts,
    };
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    fn asymmetric_site() -> ComplexSite {
        let left = new_index(2);
        let phys = new_index(2);
        let anc = new_index(2);
        let right = new_index(2);
        let gamma = complex_from_fn(
            &[left.clone(), phys.clone(), anc.clone(), right.clone()],
            |index| {
                let [left, physical, ancilla, right] = index else {
                    unreachable!()
                };
                Complex64::new(
                    0.17 + 0.11 * *left as f64 - 0.07 * *physical as f64
                        + 0.13 * *ancilla as f64
                        + 0.19 * *right as f64,
                    -0.23 + 0.05 * *left as f64 + 0.17 * *physical as f64 - 0.09 * *ancilla as f64
                        + 0.07 * *right as f64,
                )
            },
        )
        .unwrap();
        ComplexSite {
            gamma,
            left,
            phys,
            anc,
            right,
        }
    }

    fn asymmetric_pair() -> (ComplexSite, ComplexSite) {
        let left = new_index(2);
        let middle = new_index(2);
        let right = new_index(2);
        let phys_i = new_index(2);
        let anc_i = new_index(2);
        let phys_j = new_index(2);
        let anc_j = new_index(2);
        let gamma_i = complex_from_fn(
            &[left.clone(), phys_i.clone(), anc_i.clone(), middle.clone()],
            |index| {
                Complex64::new(
                    0.21 + 0.08 * index[0] as f64 - 0.12 * index[1] as f64
                        + 0.16 * index[2] as f64
                        + 0.05 * index[3] as f64,
                    -0.14 + 0.09 * index[0] as f64 + 0.13 * index[1] as f64
                        - 0.04 * index[2] as f64
                        + 0.18 * index[3] as f64,
                )
            },
        )
        .unwrap();
        let gamma_j = complex_from_fn(
            &[middle.clone(), phys_j.clone(), anc_j.clone(), right.clone()],
            |index| {
                Complex64::new(
                    -0.09 + 0.15 * index[0] as f64 + 0.06 * index[1] as f64
                        - 0.11 * index[2] as f64
                        + 0.20 * index[3] as f64,
                    0.24 - 0.07 * index[0] as f64 + 0.10 * index[1] as f64 + 0.03 * index[2] as f64
                        - 0.16 * index[3] as f64,
                )
            },
        )
        .unwrap();
        (
            ComplexSite {
                gamma: gamma_i,
                left,
                phys: phys_i,
                anc: anc_i,
                right: middle.clone(),
            },
            ComplexSite {
                gamma: gamma_j,
                left: middle,
                phys: phys_j,
                anc: anc_j,
                right,
            },
        )
    }

    fn matrix_env(values: [Complex64; 4]) -> Env {
        let ket = new_index(2);
        let bra = new_index(2);
        Env {
            tensor: Tensor::from_dense(vec![ket.clone(), bra.clone()], values.to_vec()).unwrap(),
            ket,
            bra,
        }
    }

    fn gamma(site: &ComplexSite, left: usize, phys: usize, anc: usize, right: usize) -> Complex64 {
        site.gamma.to_vec::<Complex64>().unwrap()[left + 2 * (phys + 2 * (anc + 2 * right))]
    }

    fn left_oracle(
        env: &[Complex64],
        site: &ComplexSite,
        lambda_left: &[f64],
        operator: Option<&DMatrix<Complex64>>,
    ) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); 4];
        for bra_right in 0..2 {
            for ket_right in 0..2 {
                let mut value = Complex64::new(0.0, 0.0);
                for bra_left in 0..2 {
                    for ket_left in 0..2 {
                        for ancilla in 0..2 {
                            for physical_in in 0..2 {
                                let ket = gamma(site, ket_left, physical_in, ancilla, ket_right)
                                    * lambda_left[ket_left];
                                if let Some(operator) = operator {
                                    for physical_out in 0..2 {
                                        let bra =
                                            gamma(site, bra_left, physical_out, ancilla, bra_right)
                                                * lambda_left[bra_left];
                                        value += env[ket_left + 2 * bra_left]
                                            * ket
                                            * operator[(physical_out, physical_in)]
                                            * bra.conj();
                                    }
                                } else {
                                    let bra =
                                        gamma(site, bra_left, physical_in, ancilla, bra_right)
                                            * lambda_left[bra_left];
                                    value += env[ket_left + 2 * bra_left] * ket * bra.conj();
                                }
                            }
                        }
                    }
                }
                output[ket_right + 2 * bra_right] = value;
            }
        }
        output
    }

    fn right_adjoint_oracle(
        env: &[Complex64],
        site: &ComplexSite,
        lambda_left: &[f64],
    ) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); 4];
        for ket_left in 0..2 {
            for bra_left in 0..2 {
                let mut value = Complex64::new(0.0, 0.0);
                for ket_right in 0..2 {
                    for bra_right in 0..2 {
                        for physical in 0..2 {
                            for ancilla in 0..2 {
                                let bra = gamma(site, ket_left, physical, ancilla, ket_right)
                                    * lambda_left[ket_left];
                                let ket = gamma(site, bra_left, physical, ancilla, bra_right)
                                    * lambda_left[bra_left];
                                value += env[ket_right + 2 * bra_right] * bra.conj() * ket;
                            }
                        }
                    }
                }
                output[ket_left + 2 * bra_left] = value;
            }
        }
        output
    }

    fn op2_oracle(
        env: &[Complex64],
        site_i: &ComplexSite,
        lambda_i: &[f64],
        site_j: &ComplexSite,
        lambda_j: &[f64],
        operator: &DMatrix<Complex64>,
    ) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); 4];
        for bra_right in 0..2 {
            for ket_right in 0..2 {
                let mut value = Complex64::new(0.0, 0.0);
                for bra_left in 0..2 {
                    for ket_left in 0..2 {
                        for bra_middle in 0..2 {
                            for ket_middle in 0..2 {
                                for ancilla_i in 0..2 {
                                    for ancilla_j in 0..2 {
                                        for physical_i_in in 0..2 {
                                            for physical_j_in in 0..2 {
                                                for physical_i_out in 0..2 {
                                                    for physical_j_out in 0..2 {
                                                        let ket_i = gamma(
                                                            site_i,
                                                            ket_left,
                                                            physical_i_in,
                                                            ancilla_i,
                                                            ket_middle,
                                                        ) * lambda_i[ket_left];
                                                        let ket_j = gamma(
                                                            site_j,
                                                            ket_middle,
                                                            physical_j_in,
                                                            ancilla_j,
                                                            ket_right,
                                                        ) * lambda_j[ket_middle];
                                                        let bra_i = gamma(
                                                            site_i,
                                                            bra_left,
                                                            physical_i_out,
                                                            ancilla_i,
                                                            bra_middle,
                                                        ) * lambda_i[bra_left];
                                                        let bra_j = gamma(
                                                            site_j,
                                                            bra_middle,
                                                            physical_j_out,
                                                            ancilla_j,
                                                            bra_right,
                                                        ) * lambda_j[bra_middle];
                                                        let input =
                                                            physical_i_in + 2 * physical_j_in;
                                                        let output_index =
                                                            physical_i_out + 2 * physical_j_out;
                                                        value += env[ket_left + 2 * bra_left]
                                                            * ket_i
                                                            * ket_j
                                                            * operator[(output_index, input)]
                                                            * bra_i.conj()
                                                            * bra_j.conj();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                output[ket_right + 2 * bra_right] = value;
            }
        }
        output
    }

    fn right_op2_oracle(
        env: &[Complex64],
        site_i: &ComplexSite,
        lambda_i: &[f64],
        site_j: &ComplexSite,
        lambda_j: &[f64],
        operator: &DMatrix<Complex64>,
        conjugate_bra: bool,
    ) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); 4];
        for bra_left in 0..2 {
            for ket_left in 0..2 {
                let mut value = Complex64::new(0.0, 0.0);
                for bra_right in 0..2 {
                    for ket_right in 0..2 {
                        for bra_middle in 0..2 {
                            for ket_middle in 0..2 {
                                for ancilla_i in 0..2 {
                                    for ancilla_j in 0..2 {
                                        for physical_i_in in 0..2 {
                                            for physical_j_in in 0..2 {
                                                for physical_i_out in 0..2 {
                                                    for physical_j_out in 0..2 {
                                                        let bra_i = gamma(
                                                            site_i,
                                                            ket_left,
                                                            physical_i_out,
                                                            ancilla_i,
                                                            ket_middle,
                                                        ) * lambda_i[ket_left];
                                                        let bra_j = gamma(
                                                            site_j,
                                                            ket_middle,
                                                            physical_j_out,
                                                            ancilla_j,
                                                            ket_right,
                                                        ) * lambda_j[ket_middle];
                                                        let ket_i = gamma(
                                                            site_i,
                                                            bra_left,
                                                            physical_i_in,
                                                            ancilla_i,
                                                            bra_middle,
                                                        ) * lambda_i[bra_left];
                                                        let ket_j = gamma(
                                                            site_j,
                                                            bra_middle,
                                                            physical_j_in,
                                                            ancilla_j,
                                                            bra_right,
                                                        ) * lambda_j[bra_middle];
                                                        let input =
                                                            physical_i_in + 2 * physical_j_in;
                                                        let output_index =
                                                            physical_i_out + 2 * physical_j_out;
                                                        let bra = bra_i * bra_j;
                                                        let bra = if conjugate_bra {
                                                            bra.conj()
                                                        } else {
                                                            bra
                                                        };
                                                        value += env[ket_right + 2 * bra_right]
                                                            * bra
                                                            * operator[(output_index, input)]
                                                            * ket_i
                                                            * ket_j;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // The retained indices are [ket_left, bra_left], with the ket index fastest.
                output[ket_left + 2 * bra_left] = value;
            }
        }
        output
    }

    fn assert_close(actual: &Tensor, expected: &[Complex64]) {
        assert_eq!(actual.dims(), vec![2, 2]);
        let actual = actual.to_vec::<Complex64>().unwrap();
        for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).norm() <= 2e-12,
                "entry {position}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn complex_transfer_matches_dense_oracles() {
        let site = asymmetric_site();
        let lambda = [0.8, 0.6];
        let env_values = [
            Complex64::new(0.7, -0.2),
            Complex64::new(-0.3, 0.5),
            Complex64::new(0.4, 0.1),
            Complex64::new(-0.6, 0.9),
        ];

        reset_pairwise_counts();
        let left = transfer_step(&matrix_env(env_values), &site, &lambda, None).unwrap();
        assert_close(
            &left.tensor,
            &left_oracle(&env_values, &site, &lambda, None),
        );
        assert_eq!(pairwise_call_count(), 2);
        assert_eq!(pairwise_operand_conjugation_count(), 1);

        let operator = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.2, 0.4),
                Complex64::new(-0.7, 0.3),
                Complex64::new(0.6, -0.1),
                Complex64::new(0.5, 0.8),
            ],
        );
        reset_pairwise_counts();
        let with_operator =
            transfer_step(&matrix_env(env_values), &site, &lambda, Some(&operator)).unwrap();
        assert_close(
            &with_operator.tensor,
            &left_oracle(&env_values, &site, &lambda, Some(&operator)),
        );
        assert_eq!(pairwise_call_count(), 3);
        assert_eq!(pairwise_operand_conjugation_count(), 1);

        reset_pairwise_counts();
        let right = transfer_step_right(&matrix_env(env_values), &site, &lambda).unwrap();
        assert_close(
            &right.tensor,
            &right_adjoint_oracle(&env_values, &site, &lambda),
        );
        assert_eq!(pairwise_call_count(), 2);
        assert_eq!(pairwise_operand_conjugation_count(), 1);

        let (site_i, site_j) = asymmetric_pair();
        let lambda_i = [0.9, 0.4];
        let lambda_j = [0.75, 0.55];
        let operator2 = DMatrix::from_fn(4, 4, |row, column| {
            Complex64::new(
                0.05 + 0.07 * row as f64 - 0.03 * column as f64,
                -0.08 + 0.02 * row as f64 + 0.06 * column as f64,
            )
        });
        reset_pairwise_counts();
        let two_site = transfer_step_op2(
            &matrix_env(env_values),
            &site_i,
            &lambda_i,
            &site_j,
            &lambda_j,
            &operator2,
        )
        .unwrap();
        assert_close(
            &two_site.tensor,
            &op2_oracle(
                &env_values,
                &site_i,
                &lambda_i,
                &site_j,
                &lambda_j,
                &operator2,
            ),
        );
        assert_eq!(pairwise_call_count(), 5);
        assert_eq!(pairwise_operand_conjugation_count(), 2);

        let closed = close_env(&matrix_env(env_values), &[0.8, 0.6]).unwrap();
        let expected = env_values[0] * 0.64 + env_values[3] * 0.36;
        assert!((closed - expected).norm() <= 1e-14);
    }

    #[test]
    fn right_two_site_transfer_matches_dense_complex_oracle() {
        let (site_i, site_j) = asymmetric_pair();
        let lambda_i = [0.9, 0.4];
        let lambda_j = [0.75, 0.55];
        let env_values = [
            Complex64::new(0.7, -0.2),
            Complex64::new(-0.3, 0.5),
            Complex64::new(0.4, 0.1),
            Complex64::new(-0.6, 0.9),
        ];
        let operator = DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(0.7, 0.0),
                Complex64::new(0.2, 0.3),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.1, 0.2),
                Complex64::new(0.2, -0.3),
                Complex64::new(-0.4, 0.0),
                Complex64::new(0.5, -0.1),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.5, 0.1),
                Complex64::new(0.9, 0.0),
                Complex64::new(-0.2, 0.4),
                Complex64::new(-0.1, -0.2),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.2, -0.4),
                Complex64::new(0.1, 0.0),
            ],
        );
        assert_eq!(operator, operator.adjoint());
        let expected = right_op2_oracle(
            &env_values,
            &site_i,
            &lambda_i,
            &site_j,
            &lambda_j,
            &operator,
            true,
        );
        let without_bra_conjugation = right_op2_oracle(
            &env_values,
            &site_i,
            &lambda_i,
            &site_j,
            &lambda_j,
            &operator,
            false,
        );
        assert!(
            expected
                .iter()
                .zip(&without_bra_conjugation)
                .map(|(expected, mutated)| (*expected - *mutated).norm())
                .fold(0.0_f64, f64::max)
                > 1e-5,
            "the fixture must detect omitted bra conjugation"
        );

        reset_pairwise_counts();
        let actual = transfer_step_op2_right(
            &matrix_env(env_values),
            &site_i,
            &lambda_i,
            &site_j,
            &lambda_j,
            &operator,
        )
        .unwrap();
        assert_eq!(
            actual.tensor.indices,
            vec![actual.ket.clone(), actual.bra.clone()]
        );
        assert_close(&actual.tensor, &expected);
        assert_eq!(pairwise_call_count(), 5);
        assert_eq!(pairwise_operand_conjugation_count(), 2);
    }

    #[test]
    fn complex_transfer_satisfies_adjoint_identity() {
        let site = asymmetric_site();
        let lambda = [0.8, 0.6];
        let x = [
            Complex64::new(0.9, -0.1),
            Complex64::new(-0.4, 0.7),
            Complex64::new(0.2, 0.6),
            Complex64::new(-0.5, -0.3),
        ];
        let y = [
            Complex64::new(-0.2, 0.8),
            Complex64::new(0.6, 0.1),
            Complex64::new(-0.7, 0.4),
            Complex64::new(0.3, -0.9),
        ];

        reset_pairwise_counts();
        let tx = transfer_step(&matrix_env(x), &site, &lambda, None).unwrap();
        let tdag_y = transfer_step_right(&matrix_env(y), &site, &lambda).unwrap();
        let tx = tx.tensor.to_vec::<Complex64>().unwrap();
        let tdag_y = tdag_y.tensor.to_vec::<Complex64>().unwrap();
        let lhs = y
            .iter()
            .zip(&tx)
            .map(|(y, tx)| y.conj() * tx)
            .sum::<Complex64>();
        let rhs = tdag_y
            .iter()
            .zip(x)
            .map(|(tdag_y, x)| tdag_y.conj() * x)
            .sum::<Complex64>();

        assert!((lhs - rhs).norm() <= 2e-12, "{lhs:?} != {rhs:?}");
        assert_eq!(pairwise_call_count(), 4);
        assert_eq!(pairwise_operand_conjugation_count(), 2);
    }

    #[test]
    fn complex_fixed_point_failures_return_typed_errors() {
        let zero = matrix_env([Complex64::new(0.0, 0.0); 4]);
        assert!(matches!(
            power_iterate(
                zero,
                |_| Ok(matrix_env([Complex64::new(0.0, 0.0); 4])),
                1e-12,
                4,
            ),
            Err(ItebdError::TensorOperation { .. })
        ));

        let initial = identity_env(2).unwrap();
        assert!(matches!(
            power_iterate(initial, |_| identity_env(2), 0.0, 1),
            Err(ItebdError::TensorOperation { .. })
        ));
    }
}
