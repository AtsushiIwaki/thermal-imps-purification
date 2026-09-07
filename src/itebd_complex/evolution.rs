use super::tensor::{complex_from_fn, complex_relabel, complex_scale_bond, complex_svd_bond};
use super::{ComplexLocalHamiltonian, ComplexPurifiedMps, ComplexSite};
use crate::itebd::StepInfo;
use crate::itebd_error::ItebdError;
use crate::tensor::{new_index, Idx, Tensor, Truncation};
use crate::contraction_pairwise::pairwise;
use nalgebra::DMatrix;
use num_complex::Complex64;

const APPLY_STAGE: &str = "complex_apply_gate_to_bond";
const SITE_CONTRACTION_STAGE: &str = "complex_apply_gate_sites";
const GATE_CONTRACTION_STAGE: &str = "complex_apply_gate_physical";
const RIGHT_SITE_PERMUTATION_STAGE: &str = "complex_apply_gate_right_site_permutation";

type ComplexBondUpdate = (ComplexSite, ComplexSite, Vec<f64>, Idx, f64);

fn tensor_error(stage: &'static str, message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage,
        message: message.to_string(),
    }
}

fn validate_step_inputs(
    state: &ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
    tau: f64,
) -> Result<(), ItebdError> {
    if !tau.is_finite() {
        return Err(ItebdError::InvalidTimeStep { value: tau });
    }
    state.validate_topology()?;
    if state.a.phys.dim != hamiltonian.dim() {
        return Err(tensor_error(
            APPLY_STAGE,
            format!(
                "state physical dimension {} does not match Hamiltonian dimension {}",
                state.a.phys.dim,
                hamiltonian.dim()
            ),
        ));
    }
    Ok(())
}

fn unchanged_step_info(state: &ComplexPurifiedMps) -> StepInfo {
    StepInfo {
        max_bond: state.lambda_ab.len().max(state.lambda_ba.len()),
        min_singular_value: state
            .lambda_ab
            .last()
            .copied()
            .unwrap_or(0.0)
            .min(state.lambda_ba.last().copied().unwrap_or(0.0)),
        log_norm: 0.0,
    }
}

fn complex_gate_tensor(
    gate: &DMatrix<Complex64>,
    p1: &Idx,
    p2: &Idx,
    p1_out: &Idx,
    p2_out: &Idx,
) -> Result<Tensor, ItebdError> {
    let dimension = p1.dim;
    complex_from_fn(
        &[p1.clone(), p2.clone(), p1_out.clone(), p2_out.clone()],
        |index| {
            let row = index[2] + dimension * index[3];
            let column = index[0] + dimension * index[1];
            gate[(row, column)]
        },
    )
}

fn normalized_schmidt_values(singular_values: &[f64]) -> Result<(Vec<f64>, f64), ItebdError> {
    let norm = singular_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(ItebdError::ZeroSchmidtNorm { stage: APPLY_STAGE });
    }
    Ok((
        singular_values.iter().map(|value| value / norm).collect(),
        norm,
    ))
}

fn inverse_schmidt_values(values: &[f64]) -> Vec<f64> {
    const INVERSE_CUTOFF: f64 = 1e-12;
    values
        .iter()
        .map(|&value| {
            if value > INVERSE_CUTOFF {
                1.0 / value
            } else {
                0.0
            }
        })
        .collect()
}

fn apply_complex_gate_to_bond(
    gate: &DMatrix<Complex64>,
    x: &ComplexSite,
    y: &ComplexSite,
    outer_left: &[f64],
    middle: &[f64],
    outer_right: &[f64],
    truncation: &Truncation,
) -> Result<ComplexBondUpdate, ItebdError> {
    let right_environment = new_index(y.right.dim);
    let y_gamma = complex_relabel(&y.gamma, &y.right, &right_environment)?;

    let x_gamma = complex_scale_bond(
        &complex_scale_bond(&x.gamma, &x.left, outer_left)?,
        &x.right,
        middle,
    )?;
    let y_gamma = complex_scale_bond(&y_gamma, &right_environment, outer_right)?;
    let theta = pairwise(&x_gamma, &y_gamma)
        .map_err(|error| tensor_error(SITE_CONTRACTION_STAGE, error))?;

    let p1_out = new_index(x.phys.dim);
    let p2_out = new_index(y.phys.dim);
    let gate_tensor = complex_gate_tensor(gate, &x.phys, &y.phys, &p1_out, &p2_out)?;
    let theta_gated = pairwise(&theta, &gate_tensor)
        .map_err(|error| tensor_error(GATE_CONTRACTION_STAGE, error))?;

    let result = complex_svd_bond(
        &theta_gated,
        &[x.left.clone(), p1_out.clone(), x.anc.clone()],
        truncation,
    )?;
    let (new_middle, norm) = normalized_schmidt_values(&result.s)?;

    let gamma_x = complex_scale_bond(&result.u, &x.left, &inverse_schmidt_values(outer_left))?;
    // tensor4all returns V, while the MPS right factor is V^H. With right indices stored
    // before the SVD bond, elementwise conjugation supplies the V^* payload that contracts
    // with U over the retained bond to reconstruct U S V^H.
    let gamma_y = complex_scale_bond(
        &result.v.conj(),
        &right_environment,
        &inverse_schmidt_values(outer_right),
    )?;
    let simulated_bond = gamma_y
        .indices
        .last()
        .cloned()
        .ok_or_else(|| tensor_error(APPLY_STAGE, "SVD right factor has no bond index"))?;
    let gamma_y = complex_relabel(&gamma_y, &simulated_bond, &result.bond)?;
    let gamma_y = complex_relabel(&gamma_y, &right_environment, &y.right)?;

    let new_phys_x = new_index(x.phys.dim);
    let new_phys_y = new_index(y.phys.dim);
    let gamma_x = complex_relabel(&gamma_x, &p1_out, &new_phys_x)?;
    let gamma_y = complex_relabel(&gamma_y, &p2_out, &new_phys_y)?;
    let gamma_y = gamma_y
        .permute_indices(&[
            result.bond.clone(),
            new_phys_y.clone(),
            y.anc.clone(),
            y.right.clone(),
        ])
        .map_err(|error| tensor_error(RIGHT_SITE_PERMUTATION_STAGE, error))?;

    let new_x = ComplexSite {
        gamma: gamma_x,
        left: x.left.clone(),
        phys: new_phys_x,
        anc: x.anc.clone(),
        right: result.bond.clone(),
    };
    let new_y = ComplexSite {
        gamma: gamma_y,
        left: result.bond.clone(),
        phys: new_phys_y,
        anc: y.anc.clone(),
        right: y.right.clone(),
    };
    Ok((new_x, new_y, new_middle, result.bond, norm.ln()))
}

pub fn imaginary_time_step_complex(
    state: &mut ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
    tau: f64,
    truncation: &Truncation,
) -> Result<StepInfo, ItebdError> {
    validate_step_inputs(state, hamiltonian, tau)?;
    if tau == 0.0 {
        return Ok(unchanged_step_info(state));
    }
    let gate = hamiltonian.trotter_gate(tau)?;

    let (a, b, lambda_ab, bond_ab, ln_norm_ab) = apply_complex_gate_to_bond(
        &gate,
        &state.a,
        &state.b,
        &state.lambda_ba,
        &state.lambda_ab,
        &state.lambda_ba,
        truncation,
    )?;
    state.a = a;
    state.b = b;
    state.lambda_ab = lambda_ab;
    state.lambda_bond_ab = bond_ab;

    let (b, a, lambda_ba, bond_ba, ln_norm_ba) = apply_complex_gate_to_bond(
        &gate,
        &state.b,
        &state.a,
        &state.lambda_ab,
        &state.lambda_ba,
        &state.lambda_ab,
        truncation,
    )?;
    state.b = b;
    state.a = a;
    state.lambda_ba = lambda_ba;
    state.lambda_bond_ba = bond_ba;
    state.validate_topology()?;

    Ok(StepInfo {
        max_bond: state.lambda_ab.len().max(state.lambda_ba.len()),
        min_singular_value: state
            .lambda_ab
            .last()
            .copied()
            .unwrap_or(0.0)
            .min(state.lambda_ba.last().copied().unwrap_or(0.0)),
        log_norm: 2.0 * (ln_norm_ab + ln_norm_ba),
    })
}

pub fn imaginary_time_step_second_order_complex(
    state: &mut ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
    tau: f64,
    truncation: &Truncation,
) -> Result<StepInfo, ItebdError> {
    validate_step_inputs(state, hamiltonian, tau)?;
    if tau == 0.0 {
        return Ok(unchanged_step_info(state));
    }
    let half_gate = hamiltonian.trotter_gate(0.5 * tau)?;
    let full_gate = hamiltonian.trotter_gate(tau)?;

    let (a, b, lambda_ab, bond_ab, ln_norm_ab_1) = apply_complex_gate_to_bond(
        &half_gate,
        &state.a,
        &state.b,
        &state.lambda_ba,
        &state.lambda_ab,
        &state.lambda_ba,
        truncation,
    )?;
    let ab1_dim = lambda_ab.len();
    let ab1_min = lambda_ab.last().copied().unwrap_or(0.0);
    state.a = a;
    state.b = b;
    state.lambda_ab = lambda_ab;
    state.lambda_bond_ab = bond_ab;

    let (b, a, lambda_ba, bond_ba, ln_norm_ba) = apply_complex_gate_to_bond(
        &full_gate,
        &state.b,
        &state.a,
        &state.lambda_ab,
        &state.lambda_ba,
        &state.lambda_ab,
        truncation,
    )?;
    let ba_dim = lambda_ba.len();
    let ba_min = lambda_ba.last().copied().unwrap_or(0.0);
    state.b = b;
    state.a = a;
    state.lambda_ba = lambda_ba;
    state.lambda_bond_ba = bond_ba;

    let (a, b, lambda_ab, bond_ab, ln_norm_ab_2) = apply_complex_gate_to_bond(
        &half_gate,
        &state.a,
        &state.b,
        &state.lambda_ba,
        &state.lambda_ab,
        &state.lambda_ba,
        truncation,
    )?;
    let ab2_dim = lambda_ab.len();
    let ab2_min = lambda_ab.last().copied().unwrap_or(0.0);
    state.a = a;
    state.b = b;
    state.lambda_ab = lambda_ab;
    state.lambda_bond_ab = bond_ab;
    state.validate_topology()?;

    Ok(StepInfo {
        max_bond: ab1_dim.max(ba_dim).max(ab2_dim),
        min_singular_value: ab1_min.min(ba_min).min(ab2_min),
        log_norm: 2.0 * (ln_norm_ab_1 + ln_norm_ba + ln_norm_ab_2),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_complex_gate_to_bond, imaginary_time_step_second_order_complex,
        normalized_schmidt_values,
    };
    use crate::itebd::StepInfo;
    use crate::itebd_complex::{ComplexLocalHamiltonian, ComplexPurifiedMps};
    use crate::itebd_error::ItebdError;
    use crate::model::Tfim;
    use crate::tensor::Truncation;
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex64;

    fn rotated_tfim() -> ComplexLocalHamiltonian {
        let real = Tfim { j: 1.0, g: 0.7 }.local();
        let angle = 0.37;
        let rotation = DMatrix::from_diagonal(&DVector::from_vec(vec![
            Complex64::from_polar(1.0, -0.5 * angle),
            Complex64::from_polar(1.0, 0.5 * angle),
        ]));
        let two_site_rotation = rotation.kronecker(&rotation);
        let rotate = |matrix: &DMatrix<f64>| {
            &two_site_rotation
                * matrix.map(|value| Complex64::new(value, 0.0))
                * two_site_rotation.adjoint()
        };
        ComplexLocalHamiltonian::try_new(rotate(&real.two_site_h), rotate(&real.site_energy))
            .unwrap()
    }

    fn explicit_strang_step(
        state: &mut ComplexPurifiedMps,
        hamiltonian: &ComplexLocalHamiltonian,
        tau: f64,
        truncation: &Truncation,
    ) -> StepInfo {
        let half_gate = hamiltonian.trotter_gate(0.5 * tau).unwrap();
        let full_gate = hamiltonian.trotter_gate(tau).unwrap();

        let (a, b, lambda_ab, bond_ab, ln_norm_ab_1) = apply_complex_gate_to_bond(
            &half_gate,
            &state.a,
            &state.b,
            &state.lambda_ba,
            &state.lambda_ab,
            &state.lambda_ba,
            truncation,
        )
        .unwrap();
        let ab1_dim = lambda_ab.len();
        let ab1_min = *lambda_ab.last().unwrap();
        state.a = a;
        state.b = b;
        state.lambda_ab = lambda_ab;
        state.lambda_bond_ab = bond_ab;

        let (b, a, lambda_ba, bond_ba, ln_norm_ba) = apply_complex_gate_to_bond(
            &full_gate,
            &state.b,
            &state.a,
            &state.lambda_ab,
            &state.lambda_ba,
            &state.lambda_ab,
            truncation,
        )
        .unwrap();
        let ba_dim = lambda_ba.len();
        let ba_min = *lambda_ba.last().unwrap();
        state.b = b;
        state.a = a;
        state.lambda_ba = lambda_ba;
        state.lambda_bond_ba = bond_ba;

        let (a, b, lambda_ab, bond_ab, ln_norm_ab_2) = apply_complex_gate_to_bond(
            &half_gate,
            &state.a,
            &state.b,
            &state.lambda_ba,
            &state.lambda_ab,
            &state.lambda_ba,
            truncation,
        )
        .unwrap();
        let ab2_dim = lambda_ab.len();
        let ab2_min = *lambda_ab.last().unwrap();
        state.a = a;
        state.b = b;
        state.lambda_ab = lambda_ab;
        state.lambda_bond_ab = bond_ab;

        StepInfo {
            max_bond: ab1_dim.max(ba_dim).max(ab2_dim),
            min_singular_value: ab1_min.min(ba_min).min(ab2_min),
            log_norm: 2.0 * (ln_norm_ab_1 + ln_norm_ba + ln_norm_ab_2),
        }
    }

    fn assert_state_entries_equal(actual: &ComplexPurifiedMps, expected: &ComplexPurifiedMps) {
        assert_eq!(actual.a.gamma.dims(), expected.a.gamma.dims());
        assert_eq!(actual.b.gamma.dims(), expected.b.gamma.dims());
        assert_eq!(actual.lambda_bond_ab.dim, expected.lambda_bond_ab.dim);
        assert_eq!(actual.lambda_bond_ba.dim, expected.lambda_bond_ba.dim);
        for (actual, expected) in actual
            .a
            .gamma
            .to_vec::<Complex64>()
            .unwrap()
            .iter()
            .zip(expected.a.gamma.to_vec::<Complex64>().unwrap())
            .chain(
                actual
                    .b
                    .gamma
                    .to_vec::<Complex64>()
                    .unwrap()
                    .iter()
                    .zip(expected.b.gamma.to_vec::<Complex64>().unwrap()),
            )
        {
            assert_eq!(*actual, expected);
        }
        assert_eq!(actual.lambda_ab, expected.lambda_ab);
        assert_eq!(actual.lambda_ba, expected.lambda_ba);
    }

    #[test]
    fn schmidt_normalization_rejects_zero_and_nonfinite_norm() {
        for singular_values in [vec![0.0, 0.0], vec![f64::INFINITY]] {
            assert!(matches!(
                normalized_schmidt_values(&singular_values),
                Err(ItebdError::ZeroSchmidtNorm {
                    stage: "complex_apply_gate_to_bond"
                })
            ));
        }
    }

    #[test]
    fn second_order_exactly_matches_internal_explicit_strang_sequence() {
        let hamiltonian = rotated_tfim();
        let truncation = Truncation {
            epsilon: 1e-13,
            max_bond: Some(32),
        };
        let mut actual = ComplexPurifiedMps::infinite_temperature(2).unwrap();
        let mut expected = actual.clone();

        let actual_info =
            imaginary_time_step_second_order_complex(&mut actual, &hamiltonian, 0.04, &truncation)
                .unwrap();
        let expected_info = explicit_strang_step(&mut expected, &hamiltonian, 0.04, &truncation);

        assert_state_entries_equal(&actual, &expected);
        assert_eq!(actual_info.max_bond, expected_info.max_bond);
        assert_eq!(
            actual_info.min_singular_value,
            expected_info.min_singular_value
        );
        assert_eq!(actual_info.log_norm, expected_info.log_norm);
    }
}
