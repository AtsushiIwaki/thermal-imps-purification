use super::tensor::{complex_from_fn, complex_relabel, complex_scalar, complex_scale_bond};
use super::transfer::transfer_step_op2_right;
use super::{
    close_env, identity_env, transfer_step, transfer_step_op2, transfer_step_right,
    ComplexLocalHamiltonian, ComplexPurifiedMps, ComplexSite, Env,
};
use crate::itebd_error::ItebdError;
use crate::specific_heat::{
    accumulate_specific_heat, ParityShell, SpecificHeatOptions, SpecificHeatReport,
};
use crate::tensor::{has_contractable_index, new_index, Tensor, LEGACY_DISCONNECTED_NETWORK_ERROR};
use crate::contraction_pairwise::{pairwise, pairwise_with_conjugation};
use nalgebra::DMatrix;
use num_complex::Complex64;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    static COMPLEX_VARIANCE_CONTRACT_RECORDS: RefCell<Option<Vec<ComplexVarianceContractRecord>>> = const { RefCell::new(None) };
    static COMPLEX_NEGATIVE_DIRECTION_PERTURBATION: RefCell<Option<ComplexNegativeDirectionPerturbation>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct ComplexNegativeDirectionPerturbation {
    parity: usize,
    distance: usize,
    delta: Complex64,
}

#[cfg(test)]
fn perturb_complex_negative_direction(
    value: Complex64,
    parity: usize,
    distance: usize,
) -> Complex64 {
    COMPLEX_NEGATIVE_DIRECTION_PERTURBATION.with(|configured| match *configured.borrow() {
        Some(perturbation)
            if perturbation.parity == parity && perturbation.distance == distance =>
        {
            value + perturbation.delta
        }
        _ => value,
    })
}

#[cfg(test)]
#[derive(Clone)]
struct ComplexVarianceContractRecord {
    stage: &'static str,
    output_indices: Vec<crate::tensor::Idx>,
}

#[cfg(test)]
impl ComplexVarianceContractRecord {
    fn output_element_count(&self) -> usize {
        self.output_indices
            .iter()
            .map(|index| index.dim)
            .product::<usize>()
            .max(1)
    }
}

#[cfg(test)]
fn reset_complex_variance_contract_records() {
    COMPLEX_VARIANCE_CONTRACT_RECORDS.with(|records| *records.borrow_mut() = Some(Vec::new()));
}

#[cfg(test)]
fn complex_variance_contract_records() -> Vec<ComplexVarianceContractRecord> {
    COMPLEX_VARIANCE_CONTRACT_RECORDS
        .with(|records| records.borrow_mut().take().unwrap_or_default())
}

#[cfg(test)]
fn record_complex_variance_contract(stage: &'static str, output: &Tensor) {
    COMPLEX_VARIANCE_CONTRACT_RECORDS.with(|records| {
        if let Some(records) = records.borrow_mut().as_mut() {
            records.push(ComplexVarianceContractRecord {
                stage,
                output_indices: output.indices.clone(),
            });
        }
    });
}

#[derive(Clone, Copy)]
enum Direction {
    Positive,
    Negative,
}

fn tensor_error(stage: &'static str, message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage,
        message: message.to_string(),
    }
}

fn variance_contract(
    stage: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, ItebdError> {
    if !has_contractable_index(lhs, rhs) {
        return Err(tensor_error(stage, LEGACY_DISCONNECTED_NETWORK_ERROR));
    }
    pairwise(lhs, rhs)
        .inspect(|_output| {
            #[cfg(test)]
            record_complex_variance_contract(stage, _output);
        })
        .map_err(|error| tensor_error(stage, error))
}

fn variance_contract_with_conjugation(
    stage: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
    conjugate_lhs: bool,
    conjugate_rhs: bool,
) -> Result<Tensor, ItebdError> {
    if !has_contractable_index(lhs, rhs) {
        return Err(tensor_error(stage, LEGACY_DISCONNECTED_NETWORK_ERROR));
    }
    pairwise_with_conjugation(lhs, rhs, conjugate_lhs, conjugate_rhs)
        .inspect(|_output| {
            #[cfg(test)]
            record_complex_variance_contract(stage, _output);
        })
        .map_err(|error| tensor_error(stage, error))
}

fn cell(state: &ComplexPurifiedMps, start: usize, site: usize) -> (&ComplexSite, &[f64], &[f64]) {
    if (start + site) % 2 == 0 {
        (&state.a, &state.lambda_ba, &state.lambda_ab)
    } else {
        (&state.b, &state.lambda_ab, &state.lambda_ba)
    }
}

fn checked_complex(
    value: Complex64,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<Complex64, ItebdError> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(ItebdError::NonFiniteSpecificHeatValue {
            stage,
            parity,
            distance,
            real: value.re,
            imaginary: value.im,
        });
    }
    Ok(value)
}

fn checked_ratio(
    numerator: Complex64,
    denominator: Complex64,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<Complex64, ItebdError> {
    checked_complex(numerator, stage, parity, distance)?;
    checked_complex(denominator, "correlation_denominator", parity, distance)?;
    if denominator.re == 0.0 && denominator.im == 0.0 {
        return Err(tensor_error(
            stage,
            format!("zero normalization denominator at parity {parity}, distance {distance}"),
        ));
    }
    checked_complex(numerator.fdiv(denominator), stage, parity, distance)
}

fn window_norm(
    state: &ComplexPurifiedMps,
    start: usize,
    length: usize,
) -> Result<Complex64, ItebdError> {
    let (site0, _, _) = cell(state, start, 0);
    let mut environment = identity_env(site0.left.dim)?;
    for site_index in 0..length {
        let (site, lambda_left, _) = cell(state, start, site_index);
        environment = transfer_step(&environment, site, lambda_left, None)?;
    }
    let (_, _, lambda_right) = cell(state, start, length - 1);
    checked_complex(
        close_env(&environment, lambda_right)?,
        "window_norm",
        start % 2,
        length.saturating_sub(1),
    )
}

fn corr_same_complex(
    state: &ComplexPurifiedMps,
    start: usize,
    squared_operator: &DMatrix<Complex64>,
) -> Result<Complex64, ItebdError> {
    let (site0, lambda0, _) = cell(state, start, 0);
    let (site1, lambda1, lambda_right) = cell(state, start, 1);
    let environment = identity_env(site0.left.dim)?;
    let operated = transfer_step_op2(
        &environment,
        site0,
        lambda0,
        site1,
        lambda1,
        squared_operator,
    )?;
    checked_ratio(
        close_env(&operated, lambda_right)?,
        window_norm(state, start, 2)?,
        "same_bond_correlation",
        start % 2,
        0,
    )
}

fn one_bond_correlation(
    state: &ComplexPurifiedMps,
    start: usize,
    operator: &DMatrix<Complex64>,
) -> Result<Complex64, ItebdError> {
    let (site0, lambda0, _) = cell(state, start, 0);
    let (site1, lambda1, lambda_right) = cell(state, start, 1);
    let environment = identity_env(site0.left.dim)?;
    let operated = transfer_step_op2(&environment, site0, lambda0, site1, lambda1, operator)?;
    checked_ratio(
        close_env(&operated, lambda_right)?,
        window_norm(state, start, 2)?,
        "local_energy",
        start % 2,
        0,
    )
}

fn relabel_site_tensor(
    tensor: &Tensor,
    site: &ComplexSite,
    left: &crate::tensor::Idx,
    right: &crate::tensor::Idx,
    physical: &crate::tensor::Idx,
    ancilla: &crate::tensor::Idx,
) -> Result<Tensor, ItebdError> {
    complex_relabel(
        &complex_relabel(
            &complex_relabel(
                &complex_relabel(tensor, &site.left, left)?,
                &site.right,
                right,
            )?,
            &site.phys,
            physical,
        )?,
        &site.anc,
        ancilla,
    )
}

fn two_site_operator_tensor(
    operator: &DMatrix<Complex64>,
    input0: &crate::tensor::Idx,
    input1: &crate::tensor::Idx,
    output0: &crate::tensor::Idx,
    output1: &crate::tensor::Idx,
) -> Result<Tensor, ItebdError> {
    let dimension = input0.dim;
    complex_from_fn(
        &[
            input0.clone(),
            input1.clone(),
            output0.clone(),
            output1.clone(),
        ],
        |index| {
            let input = index[0] + dimension * index[1];
            let output = index[2] + dimension * index[3];
            operator[(output, input)]
        },
    )
}

fn corr_overlap_ordered(
    state: &ComplexPurifiedMps,
    ordered_start: usize,
    operator: &DMatrix<Complex64>,
    direction: Direction,
) -> Result<Complex64, ItebdError> {
    let (site0, lambda0, _) = cell(state, ordered_start, 0);
    let (site1, lambda1, _) = cell(state, ordered_start, 1);
    let (site2, lambda2, lambda_right) = cell(state, ordered_start, 2);
    let canonical0 = complex_scale_bond(&site0.gamma, &site0.left, lambda0)?;
    let canonical1 = complex_scale_bond(&site1.gamma, &site1.left, lambda1)?;
    let canonical2 = complex_scale_bond(&site2.gamma, &site2.left, lambda2)?;

    let ket_left = new_index(site0.left.dim);
    let ket_middle0 = new_index(site0.right.dim);
    let ket_middle1 = new_index(site1.right.dim);
    let ket_right = new_index(site2.right.dim);
    let ket_physical0 = new_index(site0.phys.dim);
    let ket_physical1 = new_index(site1.phys.dim);
    let ket_physical2 = new_index(site2.phys.dim);
    let ancilla0 = new_index(site0.anc.dim);
    let ancilla1 = new_index(site1.anc.dim);
    let ancilla2 = new_index(site2.anc.dim);
    let ket0 = relabel_site_tensor(
        &canonical0,
        site0,
        &ket_left,
        &ket_middle0,
        &ket_physical0,
        &ancilla0,
    )?;
    let ket1 = relabel_site_tensor(
        &canonical1,
        site1,
        &ket_middle0,
        &ket_middle1,
        &ket_physical1,
        &ancilla1,
    )?;
    let ket2 = relabel_site_tensor(
        &canonical2,
        site2,
        &ket_middle1,
        &ket_right,
        &ket_physical2,
        &ancilla2,
    )?;
    let ket01 = variance_contract("complex variance overlap ket sites 0-1", &ket0, &ket1)?;
    let ket = variance_contract("complex variance overlap ket site 2", &ket01, &ket2)?;

    let output0 = new_index(site0.phys.dim);
    let middle_output = new_index(site1.phys.dim);
    let output1 = new_index(site1.phys.dim);
    let output2 = new_index(site2.phys.dim);
    let operated = match direction {
        Direction::Positive => {
            let right_operator = two_site_operator_tensor(
                operator,
                &ket_physical1,
                &ket_physical2,
                &middle_output,
                &output2,
            )?;
            let left_operator = two_site_operator_tensor(
                operator,
                &ket_physical0,
                &middle_output,
                &output0,
                &output1,
            )?;
            let after_right = variance_contract(
                "complex variance positive overlap operator 1",
                &ket,
                &right_operator,
            )?;
            variance_contract(
                "complex variance positive overlap operator 0",
                &after_right,
                &left_operator,
            )?
        }
        Direction::Negative => {
            let left_operator = two_site_operator_tensor(
                operator,
                &ket_physical0,
                &ket_physical1,
                &output0,
                &middle_output,
            )?;
            let right_operator = two_site_operator_tensor(
                operator,
                &middle_output,
                &ket_physical2,
                &output1,
                &output2,
            )?;
            let after_left = variance_contract(
                "complex variance negative overlap operator 0",
                &ket,
                &left_operator,
            )?;
            variance_contract(
                "complex variance negative overlap operator 1",
                &after_left,
                &right_operator,
            )?
        }
    };

    let bra_left = new_index(site0.left.dim);
    let bra_middle0 = new_index(site0.right.dim);
    let bra_middle1 = new_index(site1.right.dim);
    let bra_right = new_index(site2.right.dim);
    let bra0 = relabel_site_tensor(
        &canonical0,
        site0,
        &bra_left,
        &bra_middle0,
        &output0,
        &ancilla0,
    )?;
    let bra1 = relabel_site_tensor(
        &canonical1,
        site1,
        &bra_middle0,
        &bra_middle1,
        &output1,
        &ancilla1,
    )?;
    let bra2 = relabel_site_tensor(
        &canonical2,
        site2,
        &bra_middle1,
        &bra_right,
        &output2,
        &ancilla2,
    )?;
    let bra01 = variance_contract_with_conjugation(
        "complex variance overlap bra sites 0-1",
        &bra0,
        &bra1,
        true,
        true,
    )?;
    let bra = variance_contract_with_conjugation(
        "complex variance overlap bra site 2",
        &bra01,
        &bra2,
        false,
        true,
    )?;

    let left_cap = complex_from_fn(&[ket_left.clone(), bra_left.clone()], |index| {
        if index[0] == index[1] {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })?;
    let right_cap = complex_from_fn(&[ket_right.clone(), bra_right.clone()], |index| {
        if index[0] == index[1] {
            Complex64::new(lambda_right[index[0]] * lambda_right[index[0]], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })?;
    let left_closed = variance_contract("complex variance overlap left cap", &left_cap, &operated)?;
    let bra_closed = variance_contract(
        "complex variance overlap bra contraction",
        &bra,
        &left_closed,
    )?;
    let closed = variance_contract(
        "complex variance overlap right cap",
        &bra_closed,
        &right_cap,
    )?;
    checked_ratio(
        complex_scalar(&closed)?,
        window_norm(state, ordered_start, 3)?,
        match direction {
            Direction::Positive => "positive_overlap_correlation",
            Direction::Negative => "negative_overlap_correlation",
        },
        ordered_start % 2,
        1,
    )
}

fn corr_overlap_complex(
    state: &ComplexPurifiedMps,
    start: usize,
    operator: &DMatrix<Complex64>,
    direction: Direction,
) -> Result<Complex64, ItebdError> {
    match direction {
        Direction::Positive => corr_overlap_ordered(state, start, operator, direction),
        Direction::Negative => {
            let value = corr_overlap_ordered(state, start ^ 1, operator, direction)?;
            #[cfg(test)]
            let value = perturb_complex_negative_direction(value, start % 2, 1);
            Ok(value)
        }
    }
}

fn right_boundary_env(lambda_right: &[f64]) -> Result<Env, ItebdError> {
    let ket = new_index(lambda_right.len());
    let bra = new_index(lambda_right.len());
    let tensor = complex_from_fn(&[ket.clone(), bra.clone()], |index| {
        if index[0] == index[1] {
            Complex64::new(lambda_right[index[0]] * lambda_right[index[0]], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })?;
    Ok(Env { tensor, ket, bra })
}

fn close_left_identity(env: &Env) -> Result<Complex64, ItebdError> {
    close_env(env, &vec![1.0; env.ket.dim])
}

struct TailEnvironments {
    start: usize,
    positive_numerator: Env,
    positive_norm: Env,
    negative_numerator: Env,
    negative_norm: Env,
}

impl TailEnvironments {
    fn new(
        state: &ComplexPurifiedMps,
        start: usize,
        operator: &DMatrix<Complex64>,
    ) -> Result<Self, ItebdError> {
        let (site0, lambda0, _) = cell(state, start, 0);
        let (site1, lambda1, lambda_right) = cell(state, start, 1);
        let left_boundary = identity_env(site0.left.dim)?;
        let positive_numerator =
            transfer_step_op2(&left_boundary, site0, lambda0, site1, lambda1, operator)?;
        let positive_norm0 = transfer_step(&left_boundary, site0, lambda0, None)?;
        let positive_norm = transfer_step(&positive_norm0, site1, lambda1, None)?;

        let right_boundary = right_boundary_env(lambda_right)?;
        let negative_numerator =
            transfer_step_op2_right(&right_boundary, site0, lambda0, site1, lambda1, operator)?;
        let negative_norm1 = transfer_step_right(&right_boundary, site1, lambda1)?;
        let negative_norm = transfer_step_right(&negative_norm1, site0, lambda0)?;
        Ok(Self {
            start,
            positive_numerator,
            positive_norm,
            negative_numerator,
            negative_norm,
        })
    }

    fn shell(
        &mut self,
        state: &ComplexPurifiedMps,
        operator: &DMatrix<Complex64>,
        one_bond: &[Complex64; 2],
        distance: usize,
    ) -> Result<(Complex64, Complex64), ItebdError> {
        let parity = self.start % 2;
        let disconnected = one_bond[parity] * one_bond[(self.start + distance) % 2];

        let (positive_site0, positive_lambda0, _) = cell(state, self.start, distance);
        let (positive_site1, positive_lambda1, positive_right) =
            cell(state, self.start, distance + 1);
        let positive_with_operator = transfer_step_op2(
            &self.positive_numerator,
            positive_site0,
            positive_lambda0,
            positive_site1,
            positive_lambda1,
            operator,
        )?;
        let positive_norm0 =
            transfer_step(&self.positive_norm, positive_site0, positive_lambda0, None)?;
        let positive_norm1 =
            transfer_step(&positive_norm0, positive_site1, positive_lambda1, None)?;
        let positive = checked_complex(
            checked_ratio(
                close_env(&positive_with_operator, positive_right)?,
                close_env(&positive_norm1, positive_right)?,
                "positive_tail_correlation",
                parity,
                distance,
            )? - disconnected,
            "positive_connected_tail",
            parity,
            distance,
        )?;

        let (negative_site0, negative_lambda0, _) = cell(state, self.start, distance);
        let (negative_site1, negative_lambda1, _) = cell(state, self.start, distance - 1);
        let negative_with_operator = transfer_step_op2_right(
            &self.negative_numerator,
            negative_site0,
            negative_lambda0,
            negative_site1,
            negative_lambda1,
            operator,
        )?;
        let negative_norm1 =
            transfer_step_right(&self.negative_norm, negative_site1, negative_lambda1)?;
        let negative_norm0 =
            transfer_step_right(&negative_norm1, negative_site0, negative_lambda0)?;
        let negative = checked_complex(
            checked_ratio(
                close_left_identity(&negative_with_operator)?,
                close_left_identity(&negative_norm0)?,
                "negative_tail_correlation",
                parity,
                distance,
            )? - disconnected,
            "negative_connected_tail",
            parity,
            distance,
        )?;

        self.positive_numerator = transfer_step(
            &self.positive_numerator,
            positive_site0,
            positive_lambda0,
            None,
        )?;
        self.positive_norm =
            transfer_step(&self.positive_norm, positive_site0, positive_lambda0, None)?;
        self.negative_numerator =
            transfer_step_right(&self.negative_numerator, negative_site1, negative_lambda1)?;
        self.negative_norm =
            transfer_step_right(&self.negative_norm, negative_site1, negative_lambda1)?;
        #[cfg(test)]
        let negative = perturb_complex_negative_direction(negative, parity, distance);
        Ok((positive, negative))
    }
}

pub fn specific_heat_complex_with_options(
    state: &ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
    beta: f64,
    options: &SpecificHeatOptions,
) -> Result<SpecificHeatReport, ItebdError> {
    crate::specific_heat::validate_beta(beta)?;
    options.validate()?;
    state.validate_topology()?;
    if state.a.phys.dim != hamiltonian.dim() {
        return Err(tensor_error(
            "complex_specific_heat",
            format!(
                "state physical dimension {} does not match Hamiltonian dimension {}",
                state.a.phys.dim,
                hamiltonian.dim()
            ),
        ));
    }

    let operator = hamiltonian.site_energy();
    let squared_operator = operator * operator;
    let one_bond = [
        one_bond_correlation(state, 0, operator)?,
        one_bond_correlation(state, 1, operator)?,
    ];
    let onsite = [
        checked_complex(
            corr_same_complex(state, 0, &squared_operator)? - one_bond[0] * one_bond[0],
            "onsite_connected",
            0,
            0,
        )?,
        checked_complex(
            corr_same_complex(state, 1, &squared_operator)? - one_bond[1] * one_bond[1],
            "onsite_connected",
            1,
            0,
        )?,
    ];
    let first_positive = [
        corr_overlap_complex(state, 0, operator, Direction::Positive)? - one_bond[0] * one_bond[1],
        corr_overlap_complex(state, 1, operator, Direction::Positive)? - one_bond[1] * one_bond[0],
    ];
    let first_negative = [
        corr_overlap_complex(state, 0, operator, Direction::Negative)? - one_bond[0] * one_bond[1],
        corr_overlap_complex(state, 1, operator, Direction::Negative)? - one_bond[1] * one_bond[0],
    ];
    let first_shell = ParityShell::new(1, first_positive, first_negative);
    let mut tails = [
        TailEnvironments::new(state, 0, operator)?,
        TailEnvironments::new(state, 1, operator)?,
    ];
    accumulate_specific_heat(beta, options, onsite, first_shell, |distance| {
        let (positive_a, negative_a) = tails[0].shell(state, operator, &one_bond, distance)?;
        let (positive_b, negative_b) = tails[1].shell(state, operator, &one_bond, distance)?;
        Ok(ParityShell::new(
            distance,
            [positive_a, positive_b],
            [negative_a, negative_b],
        ))
    })
}

pub fn energy_variance_per_site_complex(
    state: &ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
) -> Result<f64, ItebdError> {
    Ok(specific_heat_complex_with_options(
        state,
        hamiltonian,
        1.0,
        &SpecificHeatOptions::default(),
    )?
    .energy_variance_per_site)
}

pub fn specific_heat_complex(
    state: &ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
    beta: f64,
) -> Result<f64, ItebdError> {
    Ok(specific_heat_complex_with_options(
        state,
        hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )?
    .specific_heat_per_site)
}

#[cfg(test)]
mod tests {
    use super::{
        checked_complex, checked_ratio, complex_variance_contract_records, corr_overlap_complex,
        corr_same_complex, reset_complex_variance_contract_records,
        specific_heat_complex_with_options, ComplexNegativeDirectionPerturbation, Direction,
        TailEnvironments, COMPLEX_NEGATIVE_DIRECTION_PERTURBATION,
    };
    use crate::itebd_complex::{
        canonicalize_complex, imaginary_time_step_second_order_complex, ComplexLocalHamiltonian,
        ComplexPurifiedMps, ComplexSite,
    };
    use crate::tensor::Truncation;
    use crate::contraction_pairwise::{
        pairwise_call_count, pairwise_operand_conjugation_count, reset_pairwise_counts,
    };
    use crate::{
        itebd_error::ItebdError,
        specific_heat::{accumulate_specific_heat, ParityShell, SpecificHeatOptions},
    };
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!(
            (actual - expected).norm() <= tolerance,
            "actual={actual:?}, expected={expected:?}, residual={}",
            (actual - expected).norm()
        );
    }

    fn with_complex_negative_direction_perturbation<T>(
        perturbation: ComplexNegativeDirectionPerturbation,
        run: impl FnOnce() -> T,
    ) -> T {
        let previous = COMPLEX_NEGATIVE_DIRECTION_PERTURBATION
            .with(|configured| configured.replace(Some(perturbation)));
        let result = run();
        COMPLEX_NEGATIVE_DIRECTION_PERTURBATION.with(|configured| configured.replace(previous));
        result
    }

    fn zero_complex_hamiltonian() -> ComplexLocalHamiltonian {
        let zero = DMatrix::zeros(4, 4);
        ComplexLocalHamiltonian::try_new(zero.clone(), zero).unwrap()
    }

    fn genuinely_complex_bond_two_state(operator: &DMatrix<Complex64>) -> ComplexPurifiedMps {
        let hamiltonian =
            ComplexLocalHamiltonian::try_new(operator.clone(), operator.clone()).unwrap();
        let truncation = Truncation {
            epsilon: 0.0,
            max_bond: Some(2),
        };
        let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
        imaginary_time_step_second_order_complex(&mut state, &hamiltonian, 0.08, &truncation)
            .unwrap();
        canonicalize_complex(&mut state, 1e-12).unwrap();
        assert_eq!(state.lambda_ab.len(), 2);
        assert_eq!(state.lambda_ba.len(), 2);
        assert!(state
            .a
            .gamma
            .to_vec::<Complex64>()
            .unwrap()
            .iter()
            .chain(state.b.gamma.to_vec::<Complex64>().unwrap().iter())
            .any(|value| value.im.abs() > 1e-8));
        state
    }

    fn complex_hermitian_bond_operator() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.2),
                Complex64::new(0.3, -0.7),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -0.2),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.3, 0.7),
                Complex64::new(0.3, 0.7),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.2),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.3, -0.7),
                Complex64::new(-1.0, -0.2),
                Complex64::new(0.0, 0.0),
            ],
        )
    }

    fn cell(
        state: &ComplexPurifiedMps,
        start: usize,
        site: usize,
    ) -> (&ComplexSite, &[f64], &[f64]) {
        if (start + site) % 2 == 0 {
            (&state.a, &state.lambda_ba, &state.lambda_ab)
        } else {
            (&state.b, &state.lambda_ab, &state.lambda_ba)
        }
    }

    fn gamma_value(
        site: &ComplexSite,
        left: usize,
        physical: usize,
        ancilla: usize,
        right: usize,
    ) -> Complex64 {
        let dimensions: Vec<_> = site.gamma.indices.iter().map(|index| index.dim).collect();
        let flat =
            left + dimensions[0] * (physical + dimensions[1] * (ancilla + dimensions[2] * right));
        site.gamma.to_vec::<Complex64>().unwrap()[flat]
    }

    fn dense_expectation_sites(
        state: &ComplexPurifiedMps,
        start: usize,
        length: usize,
        operator: &DMatrix<Complex64>,
    ) -> Complex64 {
        let physical_count = 1usize << length;
        let ancilla_count = 1usize << length;
        let internal_bond_dimensions: Vec<_> = (0..length - 1)
            .map(|site| cell(state, start, site).0.right.dim)
            .collect();
        let internal_bond_count = internal_bond_dimensions.iter().product::<usize>();
        let left_dimension = cell(state, start, 0).0.left.dim;
        let (_, _, lambda_right) = cell(state, start, length - 1);
        let mut numerator = Complex64::new(0.0, 0.0);
        let mut denominator = Complex64::new(0.0, 0.0);
        for left in 0..left_dimension {
            for right in 0..lambda_right.len() {
                for ancillas in 0..ancilla_count {
                    let mut wavefunction = vec![Complex64::new(0.0, 0.0); physical_count];
                    for physicals in 0..physical_count {
                        for internal_flat in 0..internal_bond_count {
                            let mut internal = internal_flat;
                            let mut bonds = Vec::with_capacity(length - 1);
                            for dimension in &internal_bond_dimensions {
                                bonds.push(internal % dimension);
                                internal /= dimension;
                            }
                            let mut amplitude = Complex64::new(1.0, 0.0);
                            for site_index in 0..length {
                                let (site, lambda_left, _) = cell(state, start, site_index);
                                let left_bond = if site_index == 0 {
                                    left
                                } else {
                                    bonds[site_index - 1]
                                };
                                let right_bond = if site_index + 1 == length {
                                    right
                                } else {
                                    bonds[site_index]
                                };
                                amplitude *= lambda_left[left_bond]
                                    * gamma_value(
                                        site,
                                        left_bond,
                                        (physicals >> site_index) & 1,
                                        (ancillas >> site_index) & 1,
                                        right_bond,
                                    );
                            }
                            wavefunction[physicals] += amplitude;
                        }
                    }
                    let boundary_weight = lambda_right[right] * lambda_right[right];
                    for input in 0..physical_count {
                        denominator +=
                            boundary_weight * wavefunction[input].conj() * wavefunction[input];
                        for output in 0..physical_count {
                            numerator += boundary_weight
                                * wavefunction[output].conj()
                                * operator[(output, input)]
                                * wavefunction[input];
                        }
                    }
                }
            }
        }
        numerator / denominator
    }

    fn embedded_bond_operator(
        operator: &DMatrix<Complex64>,
        length: usize,
        bond: usize,
    ) -> DMatrix<Complex64> {
        let dimension = 1usize << length;
        let mut embedded = DMatrix::zeros(dimension, dimension);
        for input in 0..dimension {
            for output in 0..dimension {
                let unchanged_elsewhere = (0..length).all(|site| {
                    site == bond
                        || site == bond + 1
                        || ((input >> site) & 1) == ((output >> site) & 1)
                });
                if unchanged_elsewhere {
                    let local_input = ((input >> bond) & 1) + 2 * ((input >> (bond + 1)) & 1);
                    let local_output = ((output >> bond) & 1) + 2 * ((output >> (bond + 1)) & 1);
                    embedded[(output, input)] = operator[(local_output, local_input)];
                }
            }
        }
        embedded
    }

    fn apply_diagonal_virtual_gauge(state: &mut ComplexPurifiedMps) {
        let ab = [Complex64::new(1.0, 0.0), Complex64::from_polar(1.0, 0.37)];
        let ba = [
            Complex64::from_polar(1.0, -0.22),
            Complex64::from_polar(1.0, 0.51),
        ];
        let gauge_site = |site: &mut ComplexSite,
                          left_phases: &[Complex64; 2],
                          right_phases: &[Complex64; 2]| {
            let dimensions = site.gamma.dims();
            let mut values = site.gamma.to_vec::<Complex64>().unwrap();
            for right in 0..dimensions[3] {
                for ancilla in 0..dimensions[2] {
                    for physical in 0..dimensions[1] {
                        for left in 0..dimensions[0] {
                            let flat = left
                                + dimensions[0]
                                    * (physical
                                        + dimensions[1] * (ancilla + dimensions[2] * right));
                            values[flat] *= left_phases[left].conj() * right_phases[right];
                        }
                    }
                }
            }
            site.gamma =
                crate::tensor::Tensor::from_dense(site.gamma.indices.clone(), values).unwrap();
        };
        gauge_site(&mut state.a, &ba, &ab);
        gauge_site(&mut state.b, &ab, &ba);
    }

    #[test]
    fn complex_correlations_match_independent_dense_oracles() {
        // Mutations caught: omitting bra conjugation, reversing either overlap order, swapping
        // parity, or replacing a right transfer by a transpose/conjugated positive result.
        let operator = complex_hermitian_bond_operator();
        let state = genuinely_complex_bond_two_state(&operator);
        let squared = &operator * &operator;
        let one_bond = [
            dense_expectation_sites(&state, 0, 2, &operator),
            dense_expectation_sites(&state, 1, 2, &operator),
        ];
        let mut tails = [
            TailEnvironments::new(&state, 0, &operator).unwrap(),
            TailEnvironments::new(&state, 1, &operator).unwrap(),
        ];
        let mut ordered_overlaps = Vec::new();

        for start in 0..2 {
            let same_oracle = dense_expectation_sites(&state, start, 2, &squared);
            assert_complex_close(
                corr_same_complex(&state, start, &squared).unwrap(),
                same_oracle,
                2e-12,
            );

            let left = embedded_bond_operator(&operator, 3, 0);
            let right = embedded_bond_operator(&operator, 3, 1);
            let positive_oracle = dense_expectation_sites(&state, start, 3, &(&left * &right));
            let negative_oracle = dense_expectation_sites(&state, start ^ 1, 3, &(&right * &left));
            ordered_overlaps.extend([positive_oracle, negative_oracle]);
            assert_complex_close(
                corr_overlap_complex(&state, start, &operator, Direction::Positive).unwrap(),
                positive_oracle,
                2e-12,
            );
            assert_complex_close(
                corr_overlap_complex(&state, start, &operator, Direction::Negative).unwrap(),
                negative_oracle,
                2e-12,
            );
        }

        assert!(ordered_overlaps.iter().any(|value| value.im.abs() > 1e-8));
        assert!(ordered_overlaps.iter().sum::<Complex64>().im.abs() < 2e-12);

        for distance in 2..=3 {
            let length = distance + 2;
            let left = embedded_bond_operator(&operator, length, 0);
            let right = embedded_bond_operator(&operator, length, distance);
            for start in 0..2 {
                let disconnected = one_bond[start] * one_bond[(start + distance) % 2];
                let positive_oracle =
                    dense_expectation_sites(&state, start, length, &(&left * &right))
                        - disconnected;
                let negative_oracle = dense_expectation_sites(
                    &state,
                    (start + distance) % 2,
                    length,
                    &(&right * &left),
                ) - disconnected;
                let (positive, negative) = tails[start]
                    .shell(&state, &operator, &one_bond, distance)
                    .unwrap();
                assert_complex_close(positive, positive_oracle, 2e-12);
                assert_complex_close(negative, negative_oracle, 2e-12);
            }
        }
    }

    #[test]
    fn complex_correlations_are_covariant_under_nonreal_virtual_gauge() {
        // Mutations caught: using transpose in either right transfer or omitting conjugation at
        // the overlap bra boundary makes at least one gauged ordered correlation change.
        let operator = complex_hermitian_bond_operator();
        let state = genuinely_complex_bond_two_state(&operator);
        let mut gauged = state.clone();
        apply_diagonal_virtual_gauge(&mut gauged);
        let squared = &operator * &operator;
        let one_bond = [
            dense_expectation_sites(&state, 0, 2, &operator),
            dense_expectation_sites(&state, 1, 2, &operator),
        ];
        let gauged_one_bond = [
            dense_expectation_sites(&gauged, 0, 2, &operator),
            dense_expectation_sites(&gauged, 1, 2, &operator),
        ];
        for parity in 0..2 {
            assert_complex_close(
                corr_same_complex(&gauged, parity, &squared).unwrap(),
                corr_same_complex(&state, parity, &squared).unwrap(),
                2e-12,
            );
            for direction in [Direction::Positive, Direction::Negative] {
                assert_complex_close(
                    corr_overlap_complex(&gauged, parity, &operator, direction).unwrap(),
                    corr_overlap_complex(&state, parity, &operator, direction).unwrap(),
                    2e-12,
                );
            }
        }
        let mut tails = [
            TailEnvironments::new(&state, 0, &operator).unwrap(),
            TailEnvironments::new(&state, 1, &operator).unwrap(),
        ];
        let mut gauged_tails = [
            TailEnvironments::new(&gauged, 0, &operator).unwrap(),
            TailEnvironments::new(&gauged, 1, &operator).unwrap(),
        ];
        for distance in 2..=3 {
            for parity in 0..2 {
                let actual = tails[parity]
                    .shell(&state, &operator, &one_bond, distance)
                    .unwrap();
                let gauged_actual = gauged_tails[parity]
                    .shell(&gauged, &operator, &gauged_one_bond, distance)
                    .unwrap();
                assert_complex_close(gauged_actual.0, actual.0, 3e-12);
                assert_complex_close(gauged_actual.1, actual.1, 3e-12);
            }
        }
    }

    #[test]
    fn complex_overlap_conjugates_bra_operands_at_pairwise_boundaries() {
        // Mutation caught: materializing or contracting the bra without tensor4all operand
        // conjugation reduces the count below the two overlap-bra plus three norm-bra stages.
        let operator = complex_hermitian_bond_operator();
        let state = genuinely_complex_bond_two_state(&operator);
        reset_pairwise_counts();
        corr_overlap_complex(&state, 0, &operator, Direction::Positive).unwrap();
        assert_eq!(pairwise_call_count(), 16);
        assert_eq!(pairwise_operand_conjugation_count(), 5);
    }

    #[test]
    fn complex_overlap_records_the_bounded_intermediate_sizes() {
        let operator = complex_hermitian_bond_operator();
        let state = genuinely_complex_bond_two_state(&operator);
        reset_complex_variance_contract_records();
        corr_overlap_complex(&state, 0, &operator, Direction::Positive).unwrap();
        let records = complex_variance_contract_records();
        assert_eq!(records.len(), 9);
        assert_eq!(
            records
                .iter()
                .map(|record| record.stage)
                .collect::<Vec<_>>(),
            [
                "complex variance overlap ket sites 0-1",
                "complex variance overlap ket site 2",
                "complex variance positive overlap operator 1",
                "complex variance positive overlap operator 0",
                "complex variance overlap bra sites 0-1",
                "complex variance overlap bra site 2",
                "complex variance overlap left cap",
                "complex variance overlap bra contraction",
                "complex variance overlap right cap",
            ]
        );
        let output_element_counts = records
            .iter()
            .map(|record| record.output_element_count())
            .collect::<Vec<_>>();
        assert_eq!(
            output_element_counts,
            [64, 256, 256, 256, 64, 256, 256, 4, 1]
        );
        assert_eq!(output_element_counts.iter().copied().max(), Some(256));
    }

    #[test]
    fn public_complex_specific_heat_consumes_the_independent_negative_overlap_stream() {
        // Mutation caught: replacing the complex r=-1 kernel result with the adjoint positive
        // value ignores this perturbation and incorrectly returns a public report.
        let state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
        let hamiltonian = zero_complex_hamiltonian();
        let error = with_complex_negative_direction_perturbation(
            ComplexNegativeDirectionPerturbation {
                parity: 0,
                distance: 1,
                delta: Complex64::new(1e-3, 0.0),
            },
            || {
                specific_heat_complex_with_options(
                    &state,
                    &hamiltonian,
                    0.7,
                    &SpecificHeatOptions::default(),
                )
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::SpecificHeatDirectionMismatch {
                parity: 0,
                distance: 1,
                residual,
                tolerance,
            } if residual > tolerance
        ));
    }

    #[test]
    fn public_complex_specific_heat_consumes_the_independent_negative_tail_stream() {
        // Mutation caught: synthesizing complex r<=-2 tails from positive transfers ignores the
        // independently evaluated right-transfer result perturbed here.
        let state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
        let hamiltonian = zero_complex_hamiltonian();
        let options = SpecificHeatOptions {
            max_distance: 4,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        let error = with_complex_negative_direction_perturbation(
            ComplexNegativeDirectionPerturbation {
                parity: 0,
                distance: 2,
                delta: Complex64::new(1e-3, 0.0),
            },
            || specific_heat_complex_with_options(&state, &hamiltonian, 0.7, &options),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::SpecificHeatDirectionMismatch {
                parity: 0,
                distance: 2,
                residual,
                tolerance,
            } if residual > tolerance
        ));
    }

    #[test]
    fn complex_specific_heat_failures_preserve_exact_typed_variants() {
        assert!(matches!(
            checked_complex(
                Complex64::new(f64::NAN, 0.0),
                "injected_complex_value",
                1,
                3,
            ),
            Err(ItebdError::NonFiniteSpecificHeatValue {
                stage: "injected_complex_value",
                parity: 1,
                distance: 3,
                real,
                imaginary: 0.0,
            }) if real.is_nan()
        ));

        let options = SpecificHeatOptions {
            max_distance: 4,
            consecutive_small_shells: 2,
            ..SpecificHeatOptions::default()
        };
        let zero_tail = |_| {
            Ok(ParityShell::new(
                0,
                [Complex64::new(0.0, 0.0); 2],
                [Complex64::new(0.0, 0.0); 2],
            ))
        };
        assert!(matches!(
            accumulate_specific_heat(
                1.0,
                &options,
                [Complex64::new(0.0, 0.0); 2],
                ParityShell::new(
                    1,
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0); 2],
                ),
                zero_tail,
            ),
            Err(ItebdError::SpecificHeatDirectionMismatch {
                parity: 1,
                distance: 1,
                ..
            })
        ));

        assert!(matches!(
            accumulate_specific_heat(
                1.0,
                &options,
                [Complex64::new(0.0, 0.0); 2],
                ParityShell::new(
                    1,
                    [Complex64::new(0.0, 1.0), Complex64::new(0.0, 2.0)],
                    [Complex64::new(0.0, -2.0), Complex64::new(0.0, -1.0)],
                ),
                zero_tail,
            ),
            Err(ItebdError::NonRealSpecificHeat {
                stage: "parity_a",
                ..
            })
        ));

        assert!(matches!(
            accumulate_specific_heat(
                1.0,
                &options,
                [Complex64::new(-2.0, 0.0); 2],
                ParityShell::new(
                    1,
                    [Complex64::new(0.0, 0.0); 2],
                    [Complex64::new(0.0, 0.0); 2],
                ),
                zero_tail,
            ),
            Err(ItebdError::NegativeEnergyVariance { value, .. }) if value == -2.0
        ));
    }

    #[test]
    fn complex_ratio_is_scale_safe_for_finite_nonzero_denominators() {
        // Mutations caught: norm_sqr-based zero detection underflows for the small denominator,
        // while generic complex division loses the representable large-denominator quotient.
        let small = checked_ratio(
            Complex64::new(2e-300, 3e-300),
            Complex64::new(1e-300, 1e-300),
            "small_scale_ratio",
            0,
            0,
        )
        .unwrap();
        assert_complex_close(small, Complex64::new(2.5, 0.5), 2e-15);

        let large = checked_ratio(
            Complex64::new(2.0, 3.0),
            Complex64::new(1e300, 1e300),
            "large_scale_ratio",
            0,
            0,
        )
        .unwrap();
        assert_complex_close(large, Complex64::new(2.5e-300, 5e-301), 1e-315);
    }
}
