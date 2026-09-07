use crate::itebd_error::ItebdError;
use crate::model::LocalHamiltonian;
use crate::observable::two_site_op_tensor;
use crate::purified_mps::{PurifiedMps, Site};
use crate::specific_heat::{
    accumulate_specific_heat, ParityShell, SpecificHeatOptions, SpecificHeatReport,
};
use crate::tensor::{dim, has_contractable_index, new_index, relabel, scale_bond};
use crate::transfer::{
    close_env_result, identity_env, transfer_step_op2_result, transfer_step_op2_right,
    transfer_step_result, transfer_step_right_result, Env,
};
use crate::contraction_pairwise::pairwise;
use nalgebra::DMatrix;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    static VARIANCE_CONTRACT_RECORDS: RefCell<Option<Vec<VarianceContractRecord>>> = const { RefCell::new(None) };
    static NEGATIVE_DIRECTION_PERTURBATION: RefCell<Option<NegativeDirectionPerturbation>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct NegativeDirectionPerturbation {
    parity: usize,
    distance: usize,
    delta: f64,
}

#[cfg(test)]
fn perturb_negative_direction(value: f64, parity: usize, distance: usize) -> f64 {
    NEGATIVE_DIRECTION_PERTURBATION.with(|configured| match *configured.borrow() {
        Some(perturbation)
            if perturbation.parity == parity && perturbation.distance == distance =>
        {
            value + perturbation.delta
        }
        _ => value,
    })
}

const REAL_SPECIFIC_HEAT_TOPOLOGY_STAGE: &str = "real_specific_heat_topology";

fn validate_real_specific_heat_inputs(
    state: &PurifiedMps,
    hamiltonian: &LocalHamiltonian,
) -> Result<(), ItebdError> {
    validate_real_state_topology(state)?;
    validate_real_matrix("two_site_h", &hamiltonian.two_site_h)?;
    validate_real_matrix("site_energy", &hamiltonian.site_energy)?;
    if hamiltonian.two_site_h.nrows() != hamiltonian.site_energy.nrows() {
        return Err(ItebdError::MatrixDimensionMismatch {
            two_site: hamiltonian.two_site_h.nrows(),
            site_energy: hamiltonian.site_energy.nrows(),
        });
    }
    let matrix_dim = hamiltonian.two_site_h.nrows();
    let physical_dim = (matrix_dim as f64).sqrt() as usize;
    if physical_dim == 0 || physical_dim.checked_mul(physical_dim) != Some(matrix_dim) {
        return Err(ItebdError::InvalidPhysicalDimension { matrix_dim });
    }
    if state.a.phys.dim != physical_dim {
        return Err(real_topology_error(format!(
            "state physical dimension {} does not match Hamiltonian dimension {physical_dim}",
            state.a.phys.dim
        )));
    }
    Ok(())
}

fn validate_real_matrix(field: &'static str, matrix: &DMatrix<f64>) -> Result<(), ItebdError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(ItebdError::NonSquare {
            field,
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            if !matrix[(row, column)].is_finite() {
                return Err(ItebdError::NonFiniteMatrix { field, row, column });
            }
        }
    }
    Ok(())
}

fn validate_real_state_topology(state: &PurifiedMps) -> Result<(), ItebdError> {
    validate_real_site("A", &state.a)?;
    validate_real_site("B", &state.b)?;
    require_real_index_match("A.right", &state.a.right, "B.left", &state.b.left)?;
    require_real_index_match(
        "A.right/B.left",
        &state.a.right,
        "lambda_bond_ab",
        &state.lambda_bond_ab,
    )?;
    require_real_index_match("B.right", &state.b.right, "A.left", &state.a.left)?;
    require_real_index_match(
        "B.right/A.left",
        &state.b.right,
        "lambda_bond_ba",
        &state.lambda_bond_ba,
    )?;
    if state.lambda_bond_ab == state.lambda_bond_ba {
        return Err(real_topology_error(
            "lambda_bond_ab and lambda_bond_ba must be distinct",
        ));
    }
    validate_real_schmidt_values("lambda_ab", &state.lambda_ab, &state.lambda_bond_ab)?;
    validate_real_schmidt_values("lambda_ba", &state.lambda_ba, &state.lambda_bond_ba)?;
    if state.a.phys.dim != state.b.phys.dim {
        return Err(real_topology_error(format!(
            "A.phys dimension {} does not match B.phys dimension {}",
            state.a.phys.dim, state.b.phys.dim
        )));
    }
    Ok(())
}

fn validate_real_site(label: &'static str, site: &Site) -> Result<(), ItebdError> {
    if site.gamma.is_complex() {
        return Err(real_topology_error(format!(
            "{label}.gamma must use f64 storage"
        )));
    }
    let expected = [&site.left, &site.phys, &site.anc, &site.right];
    if site.gamma.indices.len() != expected.len() {
        return Err(real_topology_error(format!(
            "{label}.gamma must have four indices, got {}",
            site.gamma.indices.len()
        )));
    }
    for (expected, field) in expected.into_iter().zip(["left", "phys", "anc", "right"]) {
        if let Some(actual) = site.gamma.indices.iter().find(|actual| *actual == expected) {
            if actual.dim != expected.dim {
                return Err(real_topology_error(format!(
                    "{label}.gamma {field} dimension {} does not match declared {label}.{field} dimension {}",
                    actual.dim, expected.dim
                )));
            }
            continue;
        }
        if let Some(actual) = site
            .gamma
            .indices
            .iter()
            .find(|actual| actual.id == expected.id)
        {
            return Err(real_topology_error(format!(
                "{label}.gamma {field} dimension {} does not match declared {label}.{field} dimension {}",
                actual.dim, expected.dim
            )));
        }
        return Err(real_topology_error(format!(
            "{label}.gamma does not contain declared {label}.{field}"
        )));
    }
    if site.phys.dim == 0 || site.anc.dim == 0 || site.phys.dim != site.anc.dim {
        return Err(real_topology_error(format!(
            "{label} physical/ancilla dimensions must be equal and positive, got {}/{}",
            site.phys.dim, site.anc.dim
        )));
    }
    let values = site.gamma.to_vec::<f64>().map_err(|error| {
        real_topology_error(format!("cannot materialize {label}.gamma: {error}"))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(real_topology_error(format!(
            "{label}.gamma contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_real_schmidt_values(
    label: &'static str,
    values: &[f64],
    bond: &crate::tensor::Idx,
) -> Result<(), ItebdError> {
    if values.is_empty() || values.len() != bond.dim {
        return Err(real_topology_error(format!(
            "{label} length {} does not match positive bond dimension {}",
            values.len(),
            bond.dim
        )));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        return Err(real_topology_error(format!(
            "{label}[{index}] must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

fn require_real_index_match(
    left_label: &'static str,
    left: &crate::tensor::Idx,
    right_label: &'static str,
    right: &crate::tensor::Idx,
) -> Result<(), ItebdError> {
    if left.dim != right.dim {
        return Err(real_topology_error(format!(
            "{left_label} dimension {} must match {right_label} dimension {}",
            left.dim, right.dim
        )));
    }
    if left != right {
        return Err(real_topology_error(format!(
            "{left_label} must fully match {right_label}"
        )));
    }
    Ok(())
}

fn real_topology_error(message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage: REAL_SPECIFIC_HEAT_TOPOLOGY_STAGE,
        message: message.to_string(),
    }
}

#[cfg(test)]
#[derive(Clone)]
struct VarianceContractRecord {
    stage: &'static str,
    lhs_indices: Vec<crate::tensor::Idx>,
    rhs_indices: Vec<crate::tensor::Idx>,
    output_indices: Vec<crate::tensor::Idx>,
}

#[cfg(test)]
impl VarianceContractRecord {
    fn output_element_count(&self) -> usize {
        self.output_indices
            .iter()
            .map(|index| index.dim)
            .product::<usize>()
            .max(1)
    }
}

#[cfg(test)]
fn reset_variance_contract_records() {
    VARIANCE_CONTRACT_RECORDS.with(|records| *records.borrow_mut() = Some(Vec::new()));
}

#[cfg(test)]
fn variance_contract_records() -> Vec<VarianceContractRecord> {
    VARIANCE_CONTRACT_RECORDS.with(|records| records.borrow_mut().take().unwrap_or_default())
}

pub(crate) fn variance_contract(
    stage: &'static str,
    lhs: &crate::tensor::Tensor,
    rhs: &crate::tensor::Tensor,
) -> Result<crate::tensor::Tensor, ItebdError> {
    if !has_contractable_index(lhs, rhs) {
        return Err(ItebdError::TensorOperation {
            stage,
            message: crate::tensor::LEGACY_DISCONNECTED_NETWORK_ERROR.to_owned(),
        });
    }
    pairwise(lhs, rhs)
        .inspect(|_output| {
            #[cfg(test)]
            VARIANCE_CONTRACT_RECORDS.with(|records| {
                if let Some(records) = records.borrow_mut().as_mut() {
                    records.push(VarianceContractRecord {
                        stage,
                        lhs_indices: lhs.indices.clone(),
                        rhs_indices: rhs.indices.clone(),
                        output_indices: _output.indices.clone(),
                    });
                }
            });
        })
        .map_err(|error| ItebdError::TensorOperation {
            stage,
            message: error.to_string(),
        })
}

fn variance_scalar(stage: &'static str, tensor: &crate::tensor::Tensor) -> Result<f64, ItebdError> {
    let values = tensor
        .to_vec::<f64>()
        .map_err(|error| ItebdError::TensorOperation {
            stage,
            message: error.to_string(),
        })?;
    if values.len() != 1 {
        return Err(ItebdError::TensorOperation {
            stage,
            message: format!("expected one value, got {}", values.len()),
        });
    }
    Ok(values[0])
}

/// サイト i の (Site, lambda_left, lambda_right) を A,B,A,B... の周期で返す。
/// `start` パリティ: 0 なら i=0 が A、1 なら i=0 が B。
fn cell<'a>(state: &'a PurifiedMps, start: usize, i: usize) -> (&'a Site, &'a [f64], &'a [f64]) {
    let is_a = (i + start) % 2 == 0;
    if is_a {
        (
            &state.a,
            state.lambda_ba.as_slice(),
            state.lambda_ab.as_slice(),
        )
    } else {
        (
            &state.b,
            state.lambda_ab.as_slice(),
            state.lambda_ba.as_slice(),
        )
    }
}

/// no-op で len サイト転送した左環境を λ_right² で閉じた正規化ノルム。
fn window_norm(state: &PurifiedMps, start: usize, len: usize) -> Result<f64, ItebdError> {
    let (s0, _l0, _r0) = cell(state, start, 0);
    let mut env = identity_env(dim(&s0.left));
    for i in 0..len {
        let (si, li, _ri) = cell(state, start, i);
        env = transfer_step_result(&env, si, li, None)?;
    }
    // 最後のサイト len-1 の right で閉じる
    let (_slast, _llast, rlast) = cell(state, start, len - 1);
    checked_real(
        close_env_result(&env, rlast)?,
        "window_norm",
        start % 2,
        len.saturating_sub(1),
    )
}

/// <e_0^2>: 同 2 サイト {0,1} 上で e2 = e·e を挿入。正規化は 2 サイトノルム。
fn corr_same(state: &PurifiedMps, start: usize, e2: &DMatrix<f64>) -> Result<f64, ItebdError> {
    let (s0, l0, _) = cell(state, start, 0);
    let (s1, l1, _) = cell(state, start, 1);
    let mut env = identity_env(dim(&s0.left));
    env = transfer_step_op2_result(&env, s0, l0, s1, l1, e2)?;
    let (_s, _l, rlast) = cell(state, start, 1);
    let num = close_env_result(&env, rlast)?;
    checked_ratio(
        num,
        window_norm(state, start, 2)?,
        "same_bond_correlation",
        start % 2,
        0,
    )
}

/// 単一ボンド <e>（e を {0,1} に挿入、2 サイトで close、正規化）。start=0 で AB ボンド、start=1 で BA ボンド。
fn e0_at(state: &PurifiedMps, start: usize, e: &DMatrix<f64>) -> Result<f64, ItebdError> {
    let (s0, l0, _) = cell(state, start, 0);
    let (s1, l1, _) = cell(state, start, 1);
    let mut env = identity_env(dim(&s0.left));
    env = transfer_step_op2_result(&env, s0, l0, s1, l1, e)?;
    let (_s, _l, rlast) = cell(state, start, 1);
    let num = close_env_result(&env, rlast)?;
    checked_ratio(
        num,
        window_norm(state, start, 2)?,
        "local_energy",
        start % 2,
        0,
    )
}

/// <e_0 e_1>（overlap, r=1）。e_0 on phys {0,1}, e_1 on phys {1,2}, 3 サイト直接縮約。
/// site1 の phys は両 op が作用する。
fn corr_overlap_ordered(
    state: &PurifiedMps,
    start: usize,
    e: &DMatrix<f64>,
    positive: bool,
) -> Result<f64, ItebdError> {
    let (s0, l0, _) = cell(state, start, 0);
    let (s1, l1, _) = cell(state, start, 1);
    let (s2, l2, _) = cell(state, start, 2);
    // 左 canonical A_i = λ_left·Γ。サイトが繰り返す（A,B,A）と Id 衝突するため、
    // 各サイトのボンドを fresh Id に付け替えて鎖を作る。phys/anc は op/トレースで使うので
    // ket と bra で別 Id にする必要がある（同一だと自己縮約）。
    let a0 = scale_bond(&s0.gamma, &s0.left, l0);
    let a1 = scale_bond(&s1.gamma, &s1.left, l1);
    let a2 = scale_bond(&s2.gamma, &s2.left, l2);
    // fresh ボンド Id
    let kl = new_index(dim(&s0.left)); // 左端 ket
    let km0 = new_index(dim(&s0.right)); // 0-1 mid ket
    let km1 = new_index(dim(&s1.right)); // 1-2 mid ket
    let kr = new_index(dim(&s2.right)); // 右端 ket
                                        // ket phys 脚（fresh）
    let kp0 = new_index(dim(&s0.phys));
    let kp1 = new_index(dim(&s1.phys));
    let kp2 = new_index(dim(&s2.phys));
    // ket anc 脚（fresh, ket/bra 共有 → トレース）
    let an0 = new_index(dim(&s0.anc));
    let an1 = new_index(dim(&s1.anc));
    let an2 = new_index(dim(&s2.anc));
    let relab = |t: &crate::tensor::Tensor,
                 s: &Site,
                 l: &crate::tensor::Idx,
                 ml: &crate::tensor::Idx,
                 p: &crate::tensor::Idx,
                 a: &crate::tensor::Idx| {
        relabel(
            &relabel(&relabel(&relabel(t, &s.left, l), &s.right, ml), &s.phys, p),
            &s.anc,
            a,
        )
    };
    let ket0 = relab(&a0, s0, &kl, &km0, &kp0, &an0);
    let ket1 = relab(&a1, s1, &km0, &km1, &kp1, &an1);
    let ket2 = relab(&a2, s2, &km1, &kr, &kp2, &an2);
    let ket01 = variance_contract("variance overlap ket sites 0-1", &ket0, &ket1)?;
    let theta_ket = variance_contract("variance overlap ket site 2", &ket01, &ket2)?;
    // op 出力脚
    let q0 = new_index(dim(&s0.phys));
    let q1m = new_index(dim(&s1.phys)); // e_0 出力の中間 phys
    let q1 = new_index(dim(&s1.phys)); // e_1 出力の最終 phys
    let q2 = new_index(dim(&s2.phys));
    let h_ket = if positive {
        // <e_0 e_1>: apply the right factor e_1 first, followed by e_0.
        let op1 = two_site_op_tensor(e, &kp1, &kp2, &q1m, &q2);
        let op0 = two_site_op_tensor(e, &kp0, &q1m, &q0, &q1);
        let after_op1 =
            variance_contract("variance positive overlap operator 1", &theta_ket, &op1)?;
        variance_contract("variance positive overlap operator 0", &after_op1, &op0)?
    } else {
        // <e_1 e_0>: apply the right factor e_0 first, followed by e_1.
        let op0 = two_site_op_tensor(e, &kp0, &kp1, &q0, &q1m);
        let op1 = two_site_op_tensor(e, &q1m, &kp2, &q1, &q2);
        let after_op0 =
            variance_contract("variance negative overlap operator 0", &theta_ket, &op0)?;
        variance_contract("variance negative overlap operator 1", &after_op0, &op1)?
    }; // phys -> q0,q1,q2
       // bra 鎖: 左端/右端は別 fresh Id、mid は bra 専用 fresh、phys = q0,q1,q2、anc = ket と共有
    let bl = new_index(dim(&s0.left));
    let bm0 = new_index(dim(&s0.right));
    let bm1 = new_index(dim(&s1.right));
    let br = new_index(dim(&s2.right));
    let bra0 = relab(&a0, s0, &bl, &bm0, &q0, &an0);
    let bra1 = relab(&a1, s1, &bm0, &bm1, &q1, &an1);
    let bra2 = relab(&a2, s2, &bm1, &br, &q2, &an2);
    let bra01 = variance_contract("variance overlap bra sites 0-1", &bra0, &bra1)?;
    let theta_bra = variance_contract("variance overlap bra site 2", &bra01, &bra2)?;
    // 左端 cap: identity δ(kl, bl)
    let lid = crate::tensor::from_fn(&[kl.clone(), bl.clone()], |ix| {
        if ix[0] == ix[1] {
            1.0
        } else {
            0.0
        }
    });
    // 右端 cap: λ_right²（site2.right）
    let (_s2c, _l2c, r2) = cell(state, start, 2);
    let w: Vec<f64> = r2.iter().map(|v| v * v).collect();
    let cap = crate::tensor::from_fn(&[kr.clone(), br.clone()], |ix| {
        if ix[0] == ix[1] {
            w[ix[0]]
        } else {
            0.0
        }
    });
    let left_closed = variance_contract("variance overlap left cap", &lid, &h_ket)?;
    let bra_closed =
        variance_contract("variance overlap bra contraction", &left_closed, &theta_bra)?;
    let num = variance_scalar(
        "variance overlap scalar",
        &variance_contract("variance overlap right cap", &bra_closed, &cap)?,
    )?;
    checked_ratio(
        num,
        window_norm(state, start, 3)?,
        if positive {
            "positive_overlap_correlation"
        } else {
            "negative_overlap_correlation"
        },
        start % 2,
        1,
    )
}

fn corr_overlap_positive(
    state: &PurifiedMps,
    start: usize,
    e: &DMatrix<f64>,
) -> Result<f64, ItebdError> {
    corr_overlap_ordered(state, start, e, true)
}

fn corr_overlap_negative(
    state: &PurifiedMps,
    start: usize,
    e: &DMatrix<f64>,
) -> Result<f64, ItebdError> {
    let value = corr_overlap_ordered(state, start ^ 1, e, false)?;
    #[cfg(test)]
    let value = perturb_negative_direction(value, start % 2, 1);
    Ok(value)
}

fn checked_real(
    value: f64,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<f64, ItebdError> {
    if !value.is_finite() {
        return Err(ItebdError::NonFiniteSpecificHeatValue {
            stage,
            parity,
            distance,
            real: value,
            imaginary: 0.0,
        });
    }
    Ok(value)
}

fn checked_ratio(
    numerator: f64,
    denominator: f64,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<f64, ItebdError> {
    checked_real(numerator, stage, parity, distance)?;
    checked_real(denominator, "correlation_denominator", parity, distance)?;
    if denominator == 0.0 {
        return Err(ItebdError::TensorOperation {
            stage,
            message: format!(
                "zero normalization denominator at parity {parity}, distance {distance}"
            ),
        });
    }
    checked_real(numerator / denominator, stage, parity, distance)
}

fn right_boundary_env(lambda_right: &[f64]) -> Env {
    let ket = new_index(lambda_right.len());
    let bra = new_index(lambda_right.len());
    let tensor = crate::tensor::from_fn(&[ket.clone(), bra.clone()], |index| {
        if index[0] == index[1] {
            lambda_right[index[0]] * lambda_right[index[0]]
        } else {
            0.0
        }
    });
    Env { tensor, ket, bra }
}

fn close_left_identity(env: &Env) -> Result<f64, ItebdError> {
    close_env_result(env, &vec![1.0; dim(&env.ket)])
}

struct TailEnvironments {
    start: usize,
    positive_numerator: Env,
    positive_norm: Env,
    negative_numerator: Env,
    negative_norm: Env,
}

impl TailEnvironments {
    fn new(state: &PurifiedMps, start: usize, operator: &DMatrix<f64>) -> Result<Self, ItebdError> {
        let (site0, lambda0, _) = cell(state, start, 0);
        let (site1, lambda1, lambda_right) = cell(state, start, 1);

        let left_boundary = identity_env(dim(&site0.left));
        let positive_numerator =
            transfer_step_op2_result(&left_boundary, site0, lambda0, site1, lambda1, operator)?;
        let positive_norm0 = transfer_step_result(&left_boundary, site0, lambda0, None)?;
        let positive_norm = transfer_step_result(&positive_norm0, site1, lambda1, None)?;

        let right_boundary = right_boundary_env(lambda_right);
        let negative_numerator =
            transfer_step_op2_right(&right_boundary, site0, lambda0, site1, lambda1, operator)?;
        let negative_norm1 = transfer_step_right_result(&right_boundary, site1, lambda1)?;
        let negative_norm = transfer_step_right_result(&negative_norm1, site0, lambda0)?;

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
        state: &PurifiedMps,
        operator: &DMatrix<f64>,
        one_bond: &[f64; 2],
        distance: usize,
    ) -> Result<(f64, f64), ItebdError> {
        let parity = self.start % 2;
        let disconnected = one_bond[parity] * one_bond[(self.start + distance) % 2];

        let (positive_site0, positive_lambda0, _) = cell(state, self.start, distance);
        let (positive_site1, positive_lambda1, positive_right) =
            cell(state, self.start, distance + 1);
        let positive_with_operator = transfer_step_op2_result(
            &self.positive_numerator,
            positive_site0,
            positive_lambda0,
            positive_site1,
            positive_lambda1,
            operator,
        )?;
        let positive_norm0 =
            transfer_step_result(&self.positive_norm, positive_site0, positive_lambda0, None)?;
        let positive_norm1 =
            transfer_step_result(&positive_norm0, positive_site1, positive_lambda1, None)?;
        let positive = checked_ratio(
            close_env_result(&positive_with_operator, positive_right)?,
            close_env_result(&positive_norm1, positive_right)?,
            "positive_tail_correlation",
            parity,
            distance,
        )? - disconnected;

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
            transfer_step_right_result(&self.negative_norm, negative_site1, negative_lambda1)?;
        let negative_norm0 =
            transfer_step_right_result(&negative_norm1, negative_site0, negative_lambda0)?;
        let negative = checked_ratio(
            close_left_identity(&negative_with_operator)?,
            close_left_identity(&negative_norm0)?,
            "negative_tail_correlation",
            parity,
            distance,
        )? - disconnected;

        self.positive_numerator = transfer_step_result(
            &self.positive_numerator,
            positive_site0,
            positive_lambda0,
            None,
        )?;
        self.positive_norm =
            transfer_step_result(&self.positive_norm, positive_site0, positive_lambda0, None)?;
        self.negative_numerator =
            transfer_step_right_result(&self.negative_numerator, negative_site1, negative_lambda1)?;
        self.negative_norm =
            transfer_step_right_result(&self.negative_norm, negative_site1, negative_lambda1)?;

        let positive = checked_real(positive, "positive_connected_tail", parity, distance)?;
        let negative = checked_real(negative, "negative_connected_tail", parity, distance)?;
        #[cfg(test)]
        let negative = perturb_negative_direction(negative, parity, distance);
        Ok((positive, negative))
    }
}

/// サイトあたり connected エネルギー variance var(H)/N = Σ_r <e_0 e_r>_c。
///
/// # 精度
/// 毎ステップカノニカル化（`imaginary_time_evolve` で `canonicalize_every=1`）と
/// δτ=0.005・χ=48 の設定では β ≲ 4 で定量的に信頼できる（β=2,3,4 で厳密解との
/// 相対誤差 < 0.1%）。カノニカル化なしの simple-update では β ≳ 2 で
/// ノイズ床（~1e-4〜-5）が `specific_heat` の β² 前因子に増幅されて過大評価する。
/// カノニカル化により局所推定子が真値に補正され、低温精度が大幅に改善する。
pub fn specific_heat_with_options(
    state: &PurifiedMps,
    ham: &LocalHamiltonian,
    beta: f64,
    options: &SpecificHeatOptions,
) -> Result<SpecificHeatReport, ItebdError> {
    crate::specific_heat::validate_beta(beta)?;
    options.validate()?;
    validate_real_specific_heat_inputs(state, ham)?;
    let e = &ham.site_energy;
    let e2 = e * e; // DMatrix * DMatrix
    let e0p = [e0_at(state, 0, e)?, e0_at(state, 1, e)?];
    let onsite = [
        checked_real(
            corr_same(state, 0, &e2)? - e0p[0] * e0p[0],
            "onsite_connected",
            0,
            0,
        )?,
        checked_real(
            corr_same(state, 1, &e2)? - e0p[1] * e0p[1],
            "onsite_connected",
            1,
            0,
        )?,
    ];
    let first_positive = [
        corr_overlap_positive(state, 0, e)? - e0p[0] * e0p[1],
        corr_overlap_positive(state, 1, e)? - e0p[1] * e0p[0],
    ];
    let first_negative = [
        corr_overlap_negative(state, 0, e)? - e0p[0] * e0p[1],
        corr_overlap_negative(state, 1, e)? - e0p[1] * e0p[0],
    ];
    let first_shell = ParityShell::new(1, first_positive, first_negative);
    let mut tails = [
        TailEnvironments::new(state, 0, e)?,
        TailEnvironments::new(state, 1, e)?,
    ];
    accumulate_specific_heat(beta, options, onsite, first_shell, |distance| {
        let (positive0, negative0) = tails[0].shell(state, e, &e0p, distance)?;
        let (positive1, negative1) = tails[1].shell(state, e, &e0p, distance)?;
        Ok(ParityShell::new(
            distance,
            [positive0, positive1],
            [negative0, negative1],
        ))
    })
}

pub fn energy_variance_per_site(
    state: &PurifiedMps,
    ham: &LocalHamiltonian,
) -> Result<f64, ItebdError> {
    Ok(
        specific_heat_with_options(state, ham, 1.0, &SpecificHeatOptions::default())?
            .energy_variance_per_site,
    )
}

/// 比熱 C(β) = β²·var(H)/N。精度の注意は [`energy_variance_per_site`] を参照
/// （毎ステップカノニカル化時は β ≲ 4 で信頼できる）。
pub fn specific_heat(
    state: &PurifiedMps,
    ham: &LocalHamiltonian,
    beta: f64,
) -> Result<f64, ItebdError> {
    Ok(
        specific_heat_with_options(state, ham, beta, &SpecificHeatOptions::default())?
            .specific_heat_per_site,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::exact_specific_heat;
    use crate::itebd::imaginary_time_step;
    use crate::itebd_error::ItebdError;
    use crate::model::Tfim;
    use crate::observable::energy_density;
    use crate::purified_mps::{infinite_temperature, PurifiedMps, Site};
    use crate::tensor::{from_fn, Truncation};
    use crate::contraction_pairwise::{
        pairwise_call_count, pairwise_operand_conjugation_count, reset_pairwise_counts,
    };
    use approx::assert_abs_diff_eq;

    fn with_negative_direction_perturbation<T>(
        perturbation: NegativeDirectionPerturbation,
        run: impl FnOnce() -> T,
    ) -> T {
        let previous = NEGATIVE_DIRECTION_PERTURBATION
            .with(|configured| configured.replace(Some(perturbation)));
        let result = run();
        NEGATIVE_DIRECTION_PERTURBATION.with(|configured| configured.replace(previous));
        result
    }

    fn zero_hamiltonian() -> LocalHamiltonian {
        LocalHamiltonian {
            two_site_h: DMatrix::zeros(4, 4),
            site_energy: DMatrix::zeros(4, 4),
        }
    }
    fn asymmetric_real_bond_two_state() -> PurifiedMps {
        let bond_ab = new_index(2);
        let bond_ba = new_index(2);
        let make_site =
            |left: &crate::tensor::Idx, right: &crate::tensor::Idx, offset: f64| -> Site {
                let phys = new_index(2);
                let anc = new_index(2);
                let gamma = from_fn(
                    &[left.clone(), phys.clone(), anc.clone(), right.clone()],
                    |index| {
                        offset + 0.11 * index[0] as f64 - 0.07 * index[1] as f64
                            + 0.13 * index[2] as f64
                            + 0.19 * index[3] as f64
                            + 0.05 * (index[0] * index[2]) as f64
                            - 0.09 * (index[1] * index[3]) as f64
                    },
                );
                Site {
                    gamma,
                    left: left.clone(),
                    phys,
                    anc,
                    right: right.clone(),
                }
            };
        PurifiedMps {
            a: make_site(&bond_ba, &bond_ab, 0.17),
            b: make_site(&bond_ab, &bond_ba, -0.08),
            lambda_ab: vec![0.8, 0.6],
            lambda_bond_ab: bond_ab,
            lambda_ba: vec![0.7, 0.5],
            lambda_bond_ba: bond_ba,
        }
    }

    fn gamma_value(site: &Site, left: usize, physical: usize, ancilla: usize, right: usize) -> f64 {
        site.gamma.to_vec::<f64>().unwrap()[left + 2 * (physical + 2 * (ancilla + 2 * right))]
    }

    fn dense_expectation_sites(
        state: &PurifiedMps,
        start: usize,
        length: usize,
        operator: &DMatrix<f64>,
    ) -> f64 {
        let physical_count = 1usize << length;
        let ancilla_count = 1usize << length;
        let internal_bond_count = 1usize << (length - 1);
        let (_, _, lambda_right) = cell(state, start, length - 1);
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for left in 0..2 {
            for right in 0..2 {
                for ancillas in 0..ancilla_count {
                    let mut wavefunction = vec![0.0; physical_count];
                    for physicals in 0..physical_count {
                        for internal_bonds in 0..internal_bond_count {
                            let mut amplitude = 1.0;
                            for site_index in 0..length {
                                let (site, lambda_left, _) = cell(state, start, site_index);
                                let left_bond = if site_index == 0 {
                                    left
                                } else {
                                    (internal_bonds >> (site_index - 1)) & 1
                                };
                                let right_bond = if site_index + 1 == length {
                                    right
                                } else {
                                    (internal_bonds >> site_index) & 1
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
                        denominator += boundary_weight * wavefunction[input] * wavefunction[input];
                        for output in 0..physical_count {
                            numerator += boundary_weight
                                * wavefunction[output]
                                * operator[(output, input)]
                                * wavefunction[input];
                        }
                    }
                }
            }
        }
        numerator / denominator
    }

    fn embedded_bond_operator(operator: &DMatrix<f64>, length: usize, bond: usize) -> DMatrix<f64> {
        let dimension = 1usize << length;
        let mut embedded = DMatrix::zeros(dimension, dimension);
        for input in 0..dimension {
            for output in 0..dimension {
                let mut unchanged_elsewhere = true;
                for site in 0..length {
                    if site != bond
                        && site != bond + 1
                        && ((input >> site) & 1) != ((output >> site) & 1)
                    {
                        unchanged_elsewhere = false;
                        break;
                    }
                }
                if unchanged_elsewhere {
                    let local_input = ((input >> bond) & 1) + 2 * ((input >> (bond + 1)) & 1);
                    let local_output = ((output >> bond) & 1) + 2 * ((output >> (bond + 1)) & 1);
                    embedded[(output, input)] = operator[(local_output, local_input)];
                }
            }
        }
        embedded
    }

    fn embedded_bond_operators(operator: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        (
            embedded_bond_operator(operator, 3, 0),
            embedded_bond_operator(operator, 3, 1),
        )
    }

    #[test]
    fn ordered_overlap_directions_match_dense_oracle() {
        let state = asymmetric_real_bond_two_state();
        let operator = DMatrix::from_row_slice(
            4,
            4,
            &[
                0.2, 0.3, -0.1, 0.4, //
                0.3, -0.5, 0.6, 0.2, //
                -0.1, 0.6, 0.7, -0.3, //
                0.4, 0.2, -0.3, -0.4,
            ],
        );
        let (e01, e12) = embedded_bond_operators(&operator);
        assert!((&e01 * &e12 - &e12 * &e01).norm() > 1e-8);

        for start in 0..2 {
            let positive = dense_expectation_sites(&state, start, 3, &(&e01 * &e12));
            let translated_start = start ^ 1;
            let negative = dense_expectation_sites(&state, translated_start, 3, &(&e12 * &e01));
            assert_abs_diff_eq!(
                corr_overlap_positive(&state, start, &operator).unwrap(),
                positive,
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                corr_overlap_negative(&state, start, &operator).unwrap(),
                negative,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn persistent_tail_directions_match_dense_oracles_at_distances_two_and_three() {
        let state = asymmetric_real_bond_two_state();
        let operator = DMatrix::from_row_slice(
            4,
            4,
            &[
                0.2, 0.3, -0.1, 0.4, //
                0.3, -0.5, 0.6, 0.2, //
                -0.1, 0.6, 0.7, -0.3, //
                0.4, 0.2, -0.3, -0.4,
            ],
        );
        let one_bond = [
            dense_expectation_sites(&state, 0, 2, &operator),
            dense_expectation_sites(&state, 1, 2, &operator),
        ];
        let mut tails = [
            TailEnvironments::new(&state, 0, &operator).unwrap(),
            TailEnvironments::new(&state, 1, &operator).unwrap(),
        ];

        for distance in 2..=3 {
            let length = distance + 2;
            let left_operator = embedded_bond_operator(&operator, length, 0);
            let right_operator = embedded_bond_operator(&operator, length, distance);
            for start in 0..2 {
                let disconnected = one_bond[start] * one_bond[(start + distance) % 2];
                let expected_positive = dense_expectation_sites(
                    &state,
                    start,
                    length,
                    &(&left_operator * &right_operator),
                ) - disconnected;
                let translated_start = (start + distance) % 2;
                let expected_negative = dense_expectation_sites(
                    &state,
                    translated_start,
                    length,
                    &(&right_operator * &left_operator),
                ) - disconnected;
                let (actual_positive, actual_negative) = tails[start]
                    .shell(&state, &operator, &one_bond, distance)
                    .unwrap();
                assert_abs_diff_eq!(actual_positive, expected_positive, epsilon = 2e-11);
                assert_abs_diff_eq!(actual_negative, expected_negative, epsilon = 2e-11);
            }
        }
    }

    #[test]
    fn overlap_variance_uses_exact_connected_binary_stage_sequence() {
        let state = infinite_temperature(2);
        let operator = Tfim { j: 0.8, g: 0.3 }.local().site_energy;
        reset_pairwise_counts();
        reset_variance_contract_records();
        let value = corr_overlap_positive(&state, 0, &operator).unwrap();
        assert!(value.is_finite());
        let records = variance_contract_records();
        assert_eq!(records.len(), 9);
        assert_eq!(
            records
                .iter()
                .map(|record| record.stage)
                .collect::<Vec<_>>(),
            [
                "variance overlap ket sites 0-1",
                "variance overlap ket site 2",
                "variance positive overlap operator 1",
                "variance positive overlap operator 0",
                "variance overlap bra sites 0-1",
                "variance overlap bra site 2",
                "variance overlap left cap",
                "variance overlap bra contraction",
                "variance overlap right cap",
            ]
        );
        for (first, second) in [(0usize, 1usize), (1, 2), (2, 3), (4, 5), (6, 7), (7, 8)] {
            assert_eq!(records[first].output_indices, records[second].lhs_indices);
        }
        assert_eq!(records[3].output_indices, records[6].rhs_indices);
        assert_eq!(records[5].output_indices, records[7].rhs_indices);
        assert!(records.iter().all(|record| !record.lhs_indices.is_empty()));
        assert!(records.iter().all(|record| !record.rhs_indices.is_empty()));
        assert!(records.last().unwrap().output_indices.is_empty());
        let output_element_counts = records
            .iter()
            .map(VarianceContractRecord::output_element_count)
            .collect::<Vec<_>>();
        assert_eq!(output_element_counts, [16, 64, 64, 64, 16, 64, 64, 1, 1]);
        assert_eq!(output_element_counts.iter().copied().max(), Some(64));
        // Nine recorded variance stages plus window_norm's three no-op transfer
        // steps (six pairwise calls) and its closing contraction.
        assert_eq!(pairwise_call_count(), 16);
        assert_eq!(pairwise_operand_conjugation_count(), 0);
    }

    #[test]
    fn negative_overlap_preserves_the_reverse_operator_stage_order() {
        let state = infinite_temperature(2);
        let operator = Tfim { j: 0.8, g: 0.3 }.local().site_energy;
        reset_pairwise_counts();
        reset_variance_contract_records();
        corr_overlap_negative(&state, 0, &operator).unwrap();
        let records = variance_contract_records();
        assert_eq!(records.len(), 9);
        assert_eq!(records[2].stage, "variance negative overlap operator 0");
        assert_eq!(records[3].stage, "variance negative overlap operator 1");
        assert_eq!(records[2].output_indices, records[3].lhs_indices);
        assert_eq!(pairwise_call_count(), 16);
        assert_eq!(pairwise_operand_conjugation_count(), 0);
    }

    #[test]
    fn public_specific_heat_consumes_the_independent_negative_overlap_stream() {
        // Mutation caught: replacing the r=-1 overlap with a translated/conjugated positive value
        // ignores this negative-kernel perturbation and incorrectly returns a valid public report.
        let state = infinite_temperature(2);
        let hamiltonian = zero_hamiltonian();
        let error = with_negative_direction_perturbation(
            NegativeDirectionPerturbation {
                parity: 0,
                distance: 1,
                delta: 1e-3,
            },
            || {
                specific_heat_with_options(
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
    fn public_specific_heat_consumes_the_independent_negative_tail_stream() {
        // Mutation caught: synthesizing r<=-2 tails from positive transfer results ignores this
        // negative right-transfer perturbation and incorrectly returns a valid public report.
        let state = infinite_temperature(2);
        let hamiltonian = zero_hamiltonian();
        let options = SpecificHeatOptions {
            max_distance: 4,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        let error = with_negative_direction_perturbation(
            NegativeDirectionPerturbation {
                parity: 0,
                distance: 2,
                delta: 1e-3,
            },
            || specific_heat_with_options(&state, &hamiltonian, 0.7, &options),
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
    fn variance_stage_failure_reports_the_exact_label() {
        let lhs = crate::tensor::from_fn(&[new_index(2)], |_| 1.0);
        let rhs = crate::tensor::from_fn(&[new_index(2)], |_| 1.0);
        let error = variance_contract("variance exact failure label", &lhs, &rhs).unwrap_err();
        assert!(matches!(
            error,
            ItebdError::TensorOperation { stage, message }
                if stage == "variance exact failure label"
                    && message == "TensorDynLen TensorDynLen failed: Disconnected tensor network: 2 components found"
        ));
    }

    #[test]
    fn site_energy_op_matrix_elements() {
        let m = Tfim { j: 1.0, g: 0.5 };
        let e = m.local().site_energy;
        assert_abs_diff_eq!(e[(0, 0)], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e[(0, 1)], -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(e[(0, 2)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn specific_heat_matches_exact_and_derivative() {
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let dtau = 0.01;
        let trunc = Truncation {
            epsilon: 1e-12,
            max_bond: Some(48),
        };
        let mut state = infinite_temperature(2);
        let nsteps = (1.0 / (2.0 * dtau)) as usize;
        for _ in 0..nsteps {
            imaginary_time_step(&mut state, &ham, dtau, &trunc);
        }
        crate::canonicalize::canonicalize(&mut state);
        let beta = 1.0;

        let c_var = specific_heat(&state, &ham, beta).unwrap();
        let c_exact = exact_specific_heat(m.j, m.g, beta, 4000);
        assert!(
            (c_var - c_exact).abs() < 1e-2,
            "variance vs exact: {c_var} vs {c_exact}"
        );
        assert!(c_var > 0.0, "C should be positive, got {c_var}");
    }

    #[test]
    fn specific_heat_vs_numerical_beta_derivative() {
        let m = Tfim { j: 1.0, g: 1.0 };
        let dtau = 0.01;
        let trunc = Truncation {
            epsilon: 1e-12,
            max_bond: Some(48),
        };
        let run_to = |steps: usize| -> f64 {
            let mut s = infinite_temperature(2);
            let ham = m.local();
            for _ in 0..steps {
                imaginary_time_step(&mut s, &ham, dtau, &trunc);
            }
            energy_density(&s, &ham)
        };
        let n = 50;
        let dn = 5;
        let u_plus = run_to(n + dn);
        let u_minus = run_to(n - dn);
        let dbeta = 2.0 * (dn as f64) * dtau;
        let c_deriv = -1.0 * 1.0 * (u_plus - u_minus) / (2.0 * dbeta);
        let mut state = infinite_temperature(2);
        let ham = m.local();
        for _ in 0..n {
            imaginary_time_step(&mut state, &ham, dtau, &trunc);
        }
        crate::canonicalize::canonicalize(&mut state);
        let c_var = specific_heat(&state, &ham, 1.0).unwrap();
        assert!(
            (c_var - c_deriv).abs() < 2e-2,
            "variance vs derivative: {c_var} vs {c_deriv}"
        );
    }

    /// Low-temperature guard for the connected-variance specific heat at the TFIM critical
    /// point. The existing tests only reach β=1; this evolves to β=3 (canonicalizing every
    /// step, as the production runner does) where C is small and the tail sum is most
    /// stressed. Regression guard: a stale/wrong artifact once shipped C≈0.067 here vs the
    /// exact ≈0.0925 (a ~60% miss) with no test to catch it.
    #[test]
    #[ignore = "heavy: evolves to β=3 canonicalizing every step; run in release: cargo test --release variance:: -- --ignored"]
    fn specific_heat_tfim_low_temperature() {
        use crate::canonicalize::canonicalize;
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let dtau = 0.01;
        let trunc = Truncation {
            epsilon: 1e-12,
            max_bond: Some(48),
        };
        let beta = 3.0_f64;
        let nsteps = (beta / (2.0 * dtau)).round() as usize;
        let mut state = infinite_temperature(2);
        for _ in 0..nsteps {
            imaginary_time_step(&mut state, &ham, dtau, &trunc);
            canonicalize(&mut state);
        }
        let c_var = specific_heat(&state, &ham, beta).unwrap();
        let c_exact = exact_specific_heat(m.j, m.g, beta, 4000);
        assert!(
            (c_var - c_exact).abs() < 5e-3,
            "TFIM C(beta=3): {c_var} vs exact {c_exact}"
        );
    }
}
