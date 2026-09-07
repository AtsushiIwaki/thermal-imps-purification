use crate::itebd_error::ItebdError;
use crate::purified_mps::{PurifiedMps, Site};
use crate::tensor::{
    dim, from_fn, has_contractable_index, new_index, relabel, scale_bond, Idx, Tensor,
    LEGACY_DISCONNECTED_NETWORK_ERROR,
};
use crate::contraction_pairwise::pairwise;
use nalgebra::DMatrix;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    static TRANSFER_CONTRACT_RECORDS: RefCell<Option<Vec<TransferContractRecord>>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Clone)]
struct TransferContractRecord {
    stage: &'static str,
    lhs_indices: Vec<Idx>,
    rhs_indices: Vec<Idx>,
    output_indices: Vec<Idx>,
}

#[cfg(test)]
fn reset_transfer_contract_records() {
    TRANSFER_CONTRACT_RECORDS.with(|records| *records.borrow_mut() = Some(Vec::new()));
}

#[cfg(test)]
fn transfer_contract_records() -> Vec<TransferContractRecord> {
    TRANSFER_CONTRACT_RECORDS.with(|records| records.borrow_mut().take().unwrap_or_default())
}

#[cfg(test)]
pub(crate) fn transfer_contract(stage: &'static str, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    transfer_contract_result(stage, lhs, rhs).unwrap_or_else(|error| match error {
        ItebdError::TensorOperation { message, .. } => panic!("{stage}: {message}"),
        other => panic!("{stage}: {other}"),
    })
}

fn transfer_contract_result(
    stage: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, ItebdError> {
    if !has_contractable_index(lhs, rhs) {
        return Err(ItebdError::TensorOperation {
            stage,
            message: LEGACY_DISCONNECTED_NETWORK_ERROR.to_owned(),
        });
    }
    pairwise(lhs, rhs)
        .inspect(|_output| {
            #[cfg(test)]
            TRANSFER_CONTRACT_RECORDS.with(|records| {
                if let Some(records) = records.borrow_mut().as_mut() {
                    records.push(TransferContractRecord {
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

fn restore_transfer_order_result(
    stage: &'static str,
    tensor: &Tensor,
    indices: &[Idx],
) -> Result<Tensor, ItebdError> {
    tensor
        .permute_indices(indices)
        .map_err(|error| ItebdError::TensorOperation {
            stage,
            message: error.to_string(),
        })
}

/// bond 空間の環境（ket bond × bra bond の行列）。
pub struct Env {
    pub tensor: Tensor,
    pub ket: Idx,
    pub bra: Idx,
}

/// 恒等環境 δ_{ket,bra}（dim χ）。
pub fn identity_env(chi: usize) -> Env {
    let ket = new_index(chi);
    let bra = new_index(chi);
    let tensor = from_fn(&[ket.clone(), bra.clone()], |ix| {
        if ix[0] == ix[1] {
            1.0
        } else {
            0.0
        }
    });
    Env { tensor, ket, bra }
}

/// 左 canonical テンソル A = λ_left · Γ。
fn left_canonical(site: &Site, lambda_left: &[f64]) -> Tensor {
    scale_bond(&site.gamma, &site.left, lambda_left)
}

/// 1 サイト分の転送を env(ket,bra) に左→右で適用。op が Some なら physical に挟む。
pub fn transfer_step(
    env: &Env,
    site: &Site,
    lambda_left: &[f64],
    op: Option<&DMatrix<f64>>,
) -> Env {
    transfer_step_result(env, site, lambda_left, op).unwrap_or_else(|error| panic!("{error}"))
}

pub(crate) fn transfer_step_result(
    env: &Env,
    site: &Site,
    lambda_left: &[f64],
    op: Option<&DMatrix<f64>>,
) -> Result<Env, ItebdError> {
    let a = left_canonical(site, lambda_left);
    let nk = new_index(dim(&site.right));
    let nb = new_index(dim(&site.right));
    // ket: site.left -> env.ket, site.right -> nk（phys/anc は a のまま）
    let ket = relabel(&relabel(&a, &site.left, &env.ket), &site.right, &nk);
    let tensor = match op {
        None => {
            // bra: site.left -> env.bra, site.right -> nb（phys/anc 共有 → トレース）
            let bra = relabel(&relabel(&a, &site.left, &env.bra), &site.right, &nb);
            let environment_ket =
                transfer_contract_result("transfer step environment ket", &env.tensor, &ket)?;
            let output = transfer_contract_result("transfer step bra", &environment_ket, &bra)?;
            restore_transfer_order_result(
                "transfer step output order",
                &output,
                &[nk.clone(), nb.clone()],
            )?
        }
        Some(o) => {
            // physical を op 経由で接続: ket.phys=pin -> op.in, op.out=pout -> bra.phys
            let pout = new_index(dim(&site.phys));
            // op tensor: [site.phys (in), pout (out)] with o[(out, in)]
            let opt = from_fn(&[site.phys.clone(), pout.clone()], |ix| o[(ix[1], ix[0])]);
            let bra = relabel(
                &relabel(&relabel(&a, &site.left, &env.bra), &site.right, &nb),
                &site.phys,
                &pout,
            );
            let environment_ket = transfer_contract_result(
                "transfer step operator environment ket",
                &env.tensor,
                &ket,
            )?;
            let operated =
                transfer_contract_result("transfer step operator", &environment_ket, &opt)?;
            let output = transfer_contract_result("transfer step operator bra", &operated, &bra)?;
            restore_transfer_order_result(
                "transfer step operator output order",
                &output,
                &[nk.clone(), nb.clone()],
            )?
        }
    };
    Ok(Env {
        tensor,
        ket: nk,
        bra: nb,
    })
}

/// 2 サイト op を連続する 2 サイト (i, j=i+1) の physical 脚に挟んで転送する。
/// `op` は d²×d² の DMatrix、flat = s_i + d*s_j、row=out, col=in（observable::two_site_op_tensor 規約。
/// stride d は phys index から読むので d=2/d=3 双方で正しい）。
/// env(ket,bra) から開始し、site_i・site_j を左→右で適用し、site_j.right を新 ket/bra Id とする。
/// anc 脚はトレース（ket=bra 共有）。戻り値の env は site_j を通過後の状態。
pub fn transfer_step_op2(
    env: &Env,
    site_i: &Site,
    lam_i: &[f64],
    site_j: &Site,
    lam_j: &[f64],
    op: &DMatrix<f64>,
) -> Env {
    transfer_step_op2_result(env, site_i, lam_i, site_j, lam_j, op)
        .unwrap_or_else(|error| panic!("{error}"))
}

pub(crate) fn transfer_step_op2_result(
    env: &Env,
    site_i: &Site,
    lam_i: &[f64],
    site_j: &Site,
    lam_j: &[f64],
    op: &DMatrix<f64>,
) -> Result<Env, ItebdError> {
    let ai = left_canonical(site_i, lam_i);
    let aj = left_canonical(site_j, lam_j);
    // 中間ボンド site_i.right == site_j.left は ket/bra で別 Id に分離する必要がある。
    let nk_mid = new_index(dim(&site_i.right));
    let nk_r = new_index(dim(&site_j.right));
    let nb_r = new_index(dim(&site_j.right));
    // ket 鎖: site_i.left -> env.ket, site_i.right(=site_j.left) -> nk_mid, site_j.right -> nk_r
    let ket_i = relabel(
        &relabel(&ai, &site_i.left, &env.ket),
        &site_i.right,
        &nk_mid,
    );
    let ket_j = relabel(&relabel(&aj, &site_j.left, &nk_mid), &site_j.right, &nk_r);
    // op: phys_i(in)=site_i.phys, phys_j(in)=site_j.phys -> pout_i, pout_j
    let pout_i = new_index(dim(&site_i.phys));
    let pout_j = new_index(dim(&site_j.phys));
    let opt =
        crate::observable::two_site_op_tensor(op, &site_i.phys, &site_j.phys, &pout_i, &pout_j);
    // bra 鎖: site_i.left -> env.bra, mid -> nb_mid, site_j.right -> nb_r,
    //         phys_i -> pout_i, phys_j -> pout_j（anc は共有でトレース）
    let nb_mid = new_index(dim(&site_i.right));
    let bra_i = relabel(
        &relabel(
            &relabel(&ai, &site_i.left, &env.bra),
            &site_i.right,
            &nb_mid,
        ),
        &site_i.phys,
        &pout_i,
    );
    let bra_j = relabel(
        &relabel(&relabel(&aj, &site_j.left, &nb_mid), &site_j.right, &nb_r),
        &site_j.phys,
        &pout_j,
    );
    let environment_ket_i =
        transfer_contract_result("transfer op2 environment ket i", &env.tensor, &ket_i)?;
    let environment_ket_pair =
        transfer_contract_result("transfer op2 ket j", &environment_ket_i, &ket_j)?;
    let operated = transfer_contract_result("transfer op2 operator", &environment_ket_pair, &opt)?;
    let bra_i_applied = transfer_contract_result("transfer op2 bra i", &operated, &bra_i)?;
    let output = transfer_contract_result("transfer op2 bra j", &bra_i_applied, &bra_j)?;
    let tensor = restore_transfer_order_result(
        "transfer op2 output order",
        &output,
        &[nk_r.clone(), nb_r.clone()],
    )?;
    Ok(Env {
        tensor,
        ket: nk_r,
        bra: nb_r,
    })
}

/// 右→左の 1 サイト転送（右不動点の冪乗法用）。op なし。
/// site.right 側の env を site.left 側へ移す（A=λ_left·Γ を使用）。
pub fn transfer_step_right(env: &Env, site: &Site, lambda_left: &[f64]) -> Env {
    transfer_step_right_result(env, site, lambda_left).unwrap_or_else(|error| panic!("{error}"))
}

pub(crate) fn transfer_step_right_result(
    env: &Env,
    site: &Site,
    lambda_left: &[f64],
) -> Result<Env, ItebdError> {
    let a = left_canonical(site, lambda_left);
    let nk = new_index(dim(&site.left));
    let nb = new_index(dim(&site.left));
    // ket: site.right -> env.ket, site.left -> nk（phys/anc は a のまま共有でトレース）
    let ket = relabel(&relabel(&a, &site.right, &env.ket), &site.left, &nk);
    // bra: site.right -> env.bra, site.left -> nb
    let bra = relabel(&relabel(&a, &site.right, &env.bra), &site.left, &nb);
    let environment_ket =
        transfer_contract_result("transfer step right environment ket", &env.tensor, &ket)?;
    let output = transfer_contract_result("transfer step right bra", &environment_ket, &bra)?;
    let tensor = restore_transfer_order_result(
        "transfer step right output order",
        &output,
        &[nk.clone(), nb.clone()],
    )?;
    Ok(Env {
        tensor,
        ket: nk,
        bra: nb,
    })
}

/// Apply a two-site operator while moving an environment from right to left.
/// The operator convention matches [`transfer_step_op2`]: the left physical index is fastest.
pub(crate) fn transfer_step_op2_right(
    env: &Env,
    site_i: &Site,
    lam_i: &[f64],
    site_j: &Site,
    lam_j: &[f64],
    op: &DMatrix<f64>,
) -> Result<Env, ItebdError> {
    let ai = left_canonical(site_i, lam_i);
    let aj = left_canonical(site_j, lam_j);
    let nk_mid = new_index(dim(&site_i.right));
    let nk_l = new_index(dim(&site_i.left));
    let nb_mid = new_index(dim(&site_i.right));
    let nb_l = new_index(dim(&site_i.left));

    let ket_j = relabel(
        &relabel(&aj, &site_j.right, &env.ket),
        &site_j.left,
        &nk_mid,
    );
    let ket_i = relabel(&relabel(&ai, &site_i.right, &nk_mid), &site_i.left, &nk_l);

    let pout_i = new_index(dim(&site_i.phys));
    let pout_j = new_index(dim(&site_j.phys));
    let opt =
        crate::observable::two_site_op_tensor(op, &site_i.phys, &site_j.phys, &pout_i, &pout_j);
    let bra_j = relabel(
        &relabel(
            &relabel(&aj, &site_j.right, &env.bra),
            &site_j.left,
            &nb_mid,
        ),
        &site_j.phys,
        &pout_j,
    );
    let bra_i = relabel(
        &relabel(&relabel(&ai, &site_i.right, &nb_mid), &site_i.left, &nb_l),
        &site_i.phys,
        &pout_i,
    );

    let environment_ket_j =
        transfer_contract_result("transfer op2 right environment ket j", &env.tensor, &ket_j)?;
    let environment_ket_pair =
        transfer_contract_result("transfer op2 right ket i", &environment_ket_j, &ket_i)?;
    let operated =
        transfer_contract_result("transfer op2 right operator", &environment_ket_pair, &opt)?;
    let bra_j_applied = transfer_contract_result("transfer op2 right bra j", &operated, &bra_j)?;
    let output = transfer_contract_result("transfer op2 right bra i", &bra_j_applied, &bra_i)?;
    let tensor = restore_transfer_order_result(
        "transfer op2 right output order",
        &output,
        &[nk_l.clone(), nb_l.clone()],
    )?;
    Ok(Env {
        tensor,
        ket: nk_l,
        bra: nb_l,
    })
}

/// 冪乗法で求まった不動点。
pub struct FixedPoint {
    pub matrix: Tensor,
    pub row: Idx,
    pub col: Idx,
    pub eigenvalue: f64,
}

fn frob_norm(t: &Tensor) -> f64 {
    t.to_vec::<f64>()
        .unwrap()
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// 環境テンソルの Frobenius ノルム比を固有値として冪乗法を実行する。
/// `step` は env（1 周セル転送）を適用する関数。
/// 収束後、matrix を対称化して返す。
fn power_iterate(
    mut env: Env,
    step: impl Fn(&Env) -> Env,
    tol: f64,
    max_iter: usize,
) -> FixedPoint {
    debug_assert!(max_iter > 0, "power_iterate requires max_iter > 0");
    let mut prev_norm = 0.0;
    let mut eigenvalue = 1.0;
    for _ in 0..max_iter {
        let next = step(&env);
        let nrm = frob_norm(&next.tensor);
        eigenvalue = nrm;
        // 正規化: 全要素を nrm で割る
        let data = next.tensor.to_vec::<f64>().unwrap();
        let norm_data: Vec<f64> = data.iter().map(|v| v / nrm).collect();
        let normalized =
            Tensor::from_dense(vec![next.ket.clone(), next.bra.clone()], norm_data).unwrap();
        env = Env {
            tensor: normalized,
            ket: next.ket,
            bra: next.bra,
        };
        if (eigenvalue - prev_norm).abs() < tol {
            break;
        }
        prev_norm = eigenvalue;
    }
    // 対称化: (M + M^T) / 2
    let n = dim(&env.ket);
    let data = env.tensor.to_vec::<f64>().unwrap();
    let row = new_index(n);
    let col = new_index(n);
    // col-major: element (a, b) at index a + b*n
    let sym = from_fn(&[row.clone(), col.clone()], |ix| {
        0.5 * (data[ix[0] + ix[1] * n] + data[ix[1] + ix[0] * n])
    });
    FixedPoint {
        matrix: sym,
        row,
        col,
        eigenvalue,
    }
}

/// 左→右セル転送（A then B）の主不動点。
/// A.left = ba-bond を基準ボンドとして identity_env から開始。
pub fn dominant_fixed_point_left(state: &PurifiedMps, tol: f64, max_iter: usize) -> FixedPoint {
    let chi = dim(&state.a.left);
    let env0 = identity_env(chi);
    power_iterate(
        env0,
        |e| {
            let e = transfer_step(e, &state.a, &state.lambda_ba, None);
            transfer_step(&e, &state.b, &state.lambda_ab, None)
        },
        tol,
        max_iter,
    )
}

/// 右→左セル転送（B then A）の主不動点。
/// B.right = ba-bond を基準ボンドとして identity_env から開始。
pub fn dominant_fixed_point_right(state: &PurifiedMps, tol: f64, max_iter: usize) -> FixedPoint {
    let chi = dim(&state.b.right);
    let env0 = identity_env(chi);
    power_iterate(
        env0,
        |e| {
            let e = transfer_step_right(e, &state.b, &state.lambda_ab);
            transfer_step_right(&e, &state.a, &state.lambda_ba)
        },
        tol,
        max_iter,
    )
}

/// env を λ_right² で閉じてスカラーにする（右端の正準密度行列で trace）。
pub fn close_env(env: &Env, lambda_right: &[f64]) -> f64 {
    close_env_result(env, lambda_right).unwrap_or_else(|error| panic!("{error}"))
}

pub(crate) fn close_env_result(env: &Env, lambda_right: &[f64]) -> Result<f64, ItebdError> {
    if lambda_right.len() != dim(&env.ket) || lambda_right.len() != dim(&env.bra) {
        return Err(ItebdError::TensorOperation {
            stage: "transfer environment closure",
            message: format!(
                "Schmidt count {} does not match environment dimensions {}x{}",
                lambda_right.len(),
                dim(&env.ket),
                dim(&env.bra)
            ),
        });
    }
    let w: Vec<f64> = lambda_right.iter().map(|v| v * v).collect();
    let cap = from_fn(&[env.ket.clone(), env.bra.clone()], |ix| {
        if ix[0] == ix[1] {
            w[ix[0]]
        } else {
            0.0
        }
    });
    let closed = transfer_contract_result("transfer environment closure", &env.tensor, &cap)?;
    let values = closed
        .to_vec::<f64>()
        .map_err(|error| ItebdError::TensorOperation {
            stage: "transfer environment scalar",
            message: error.to_string(),
        })?;
    if values.len() != 1 {
        return Err(ItebdError::TensorOperation {
            stage: "transfer environment scalar",
            message: format!("expected one value, got {}", values.len()),
        });
    }
    Ok(values[0])
}

/// <σx_0 σx_r>。サイト 0 = A に σx、r サイト先に σx を挿入し、右端 λ² で閉じる。
/// 2サイトユニットセル: サイト列 ...A B A B... を A から開始。
pub fn two_point_xx(state: &PurifiedMps, r: usize) -> f64 {
    let sx = crate::model::pauli_x();
    // サイトの (Site, lambda_left, lambda_right) を A,B,A,B... と並べる無限列を index で引く。
    let cell = |i: usize| -> (&Site, &[f64], &[f64]) {
        if i % 2 == 0 {
            (
                &state.a,
                state.lambda_ba.as_slice(),
                state.lambda_ab.as_slice(),
            ) // A: left=ba, right=ab
        } else {
            (
                &state.b,
                state.lambda_ab.as_slice(),
                state.lambda_ba.as_slice(),
            ) // B: left=ab, right=ba
        }
    };
    let (s0, l0, _r0) = cell(0);
    let mut env = identity_env(dim(&s0.left));
    // サイト 0 に σx 挿入（r=0 のときは同サイトに2回挿入＝σx² なので特別扱い）
    if r == 0 {
        // σx·σx = I を挿入: op = I （= 恒等）→ <I> = 1。明示的に I を挟む。
        let id = crate::model::id2();
        env = transfer_step(&env, s0, l0, Some(&id));
        let (_s0b, _l0b, r_last) = cell(0);
        return close_env(&env, r_last);
    }
    env = transfer_step(&env, s0, l0, Some(&sx));
    // サイト 1..r-1 を op 無しで転送
    for i in 1..r {
        let (si, li, _ri) = cell(i);
        env = transfer_step(&env, si, li, None);
    }
    // サイト r に σx 挿入
    let (sr, lr, rr) = cell(r);
    env = transfer_step(&env, sr, lr, Some(&sx));
    close_env(&env, rr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itebd::imaginary_time_step;
    use crate::model::Tfim;
    use crate::observable::magnetization_x;
    use crate::purified_mps::infinite_temperature;
    use crate::tensor::Truncation;
    use crate::contraction_pairwise::{
        pairwise_call_count, pairwise_operand_conjugation_count, reset_pairwise_counts,
    };
    use approx::assert_abs_diff_eq;

    fn asymmetric_real_pair() -> (Site, Site) {
        let left = new_index(2);
        let middle = new_index(2);
        let right = new_index(2);
        let phys_i = new_index(2);
        let anc_i = new_index(2);
        let phys_j = new_index(2);
        let anc_j = new_index(2);
        let gamma_i = from_fn(
            &[left.clone(), phys_i.clone(), anc_i.clone(), middle.clone()],
            |index| {
                0.21 + 0.08 * index[0] as f64 - 0.12 * index[1] as f64
                    + 0.16 * index[2] as f64
                    + 0.05 * index[3] as f64
            },
        );
        let gamma_j = from_fn(
            &[middle.clone(), phys_j.clone(), anc_j.clone(), right.clone()],
            |index| {
                -0.09 + 0.15 * index[0] as f64 + 0.06 * index[1] as f64 - 0.11 * index[2] as f64
                    + 0.20 * index[3] as f64
            },
        );
        (
            Site {
                gamma: gamma_i,
                left,
                phys: phys_i,
                anc: anc_i,
                right: middle.clone(),
            },
            Site {
                gamma: gamma_j,
                left: middle,
                phys: phys_j,
                anc: anc_j,
                right,
            },
        )
    }

    fn matrix_env(values: [f64; 4]) -> Env {
        let ket = new_index(2);
        let bra = new_index(2);
        Env {
            tensor: Tensor::from_dense(vec![ket.clone(), bra.clone()], values.to_vec()).unwrap(),
            ket,
            bra,
        }
    }

    fn gamma(site: &Site, left: usize, phys: usize, anc: usize, right: usize) -> f64 {
        site.gamma.to_vec::<f64>().unwrap()[left + 2 * (phys + 2 * (anc + 2 * right))]
    }

    fn right_op2_oracle(
        env: &[f64],
        site_i: &Site,
        lambda_i: &[f64],
        site_j: &Site,
        lambda_j: &[f64],
        operator: &DMatrix<f64>,
    ) -> Vec<f64> {
        let mut output = vec![0.0; 4];
        for bra_left in 0..2 {
            for ket_left in 0..2 {
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
                                                        let ket = lambda_i[ket_left]
                                                            * gamma(
                                                                site_i,
                                                                ket_left,
                                                                physical_i_in,
                                                                ancilla_i,
                                                                ket_middle,
                                                            )
                                                            * lambda_j[ket_middle]
                                                            * gamma(
                                                                site_j,
                                                                ket_middle,
                                                                physical_j_in,
                                                                ancilla_j,
                                                                ket_right,
                                                            );
                                                        let bra = lambda_i[bra_left]
                                                            * gamma(
                                                                site_i,
                                                                bra_left,
                                                                physical_i_out,
                                                                ancilla_i,
                                                                bra_middle,
                                                            )
                                                            * lambda_j[bra_middle]
                                                            * gamma(
                                                                site_j,
                                                                bra_middle,
                                                                physical_j_out,
                                                                ancilla_j,
                                                                bra_right,
                                                            );
                                                        let input =
                                                            physical_i_in + 2 * physical_j_in;
                                                        let physical_out =
                                                            physical_i_out + 2 * physical_j_out;
                                                        output[ket_left + 2 * bra_left] += env
                                                            [ket_right + 2 * bra_right]
                                                            * ket
                                                            * operator[(physical_out, input)]
                                                            * bra;
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
            }
        }
        output
    }

    #[test]
    fn right_to_left_two_site_operator_transfer_matches_dense_oracle() {
        let (site_i, site_j) = asymmetric_real_pair();
        let lambda_i = [0.8, 0.6];
        let lambda_j = [0.7, 0.5];
        let env_values = [0.9, -0.2, 0.4, 0.7];
        let env = matrix_env(env_values);
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
        let expected = right_op2_oracle(
            &env_values,
            &site_i,
            &lambda_i,
            &site_j,
            &lambda_j,
            &operator,
        );
        let actual =
            transfer_step_op2_right(&env, &site_i, &lambda_i, &site_j, &lambda_j, &operator)
                .unwrap()
                .tensor
                .to_vec::<f64>()
                .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_abs_diff_eq!(actual, &expected, epsilon = 1e-12);
        }
    }

    fn evolved_state() -> PurifiedMps {
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let mut s = infinite_temperature(2);
        let trunc = Truncation {
            epsilon: 1e-10,
            max_bond: Some(32),
        };
        // β≈1.2 (30 steps × 2 × 0.02 = 1.2)
        for _ in 0..30 {
            imaginary_time_step(&mut s, &ham, 0.02, &trunc);
        }
        s
    }

    #[test]
    fn transfer_topologies_use_exact_binary_stage_sequences_and_orders() {
        let state = infinite_temperature(2);
        let env = identity_env(dim(&state.a.left));

        reset_pairwise_counts();
        reset_transfer_contract_records();
        let no_operator = transfer_step(&env, &state.a, &state.lambda_ba, None);
        let records = transfer_contract_records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].stage, "transfer step environment ket");
        assert_eq!(records[1].stage, "transfer step bra");
        assert_eq!(records[0].lhs_indices, env.tensor.indices);
        assert_eq!(records[0].output_indices, records[1].lhs_indices);
        assert_eq!(records[1].output_indices, no_operator.tensor.indices);
        assert!(!records[0].rhs_indices.is_empty());
        assert!(!records[1].rhs_indices.is_empty());
        assert_eq!(pairwise_call_count(), 2);
        assert_eq!(pairwise_operand_conjugation_count(), 0);

        reset_pairwise_counts();
        reset_transfer_contract_records();
        let with_operator = transfer_step(
            &env,
            &state.a,
            &state.lambda_ba,
            Some(&crate::model::pauli_x()),
        );
        let records = transfer_contract_records();
        assert_eq!(
            records
                .iter()
                .map(|record| record.stage)
                .collect::<Vec<_>>(),
            [
                "transfer step operator environment ket",
                "transfer step operator",
                "transfer step operator bra",
            ]
        );
        assert_eq!(records[0].output_indices, records[1].lhs_indices);
        assert_eq!(records[1].output_indices, records[2].lhs_indices);
        assert_eq!(records[2].output_indices, with_operator.tensor.indices);
        assert_eq!(pairwise_call_count(), 3);
        assert_eq!(pairwise_operand_conjugation_count(), 0);

        reset_pairwise_counts();
        reset_transfer_contract_records();
        let op2 = crate::model::Tfim { j: 1.0, g: 0.4 }.local().site_energy;
        let two_site = transfer_step_op2(
            &env,
            &state.a,
            &state.lambda_ba,
            &state.b,
            &state.lambda_ab,
            &op2,
        );
        let records = transfer_contract_records();
        assert_eq!(records.len(), 5);
        assert_eq!(
            records
                .iter()
                .map(|record| record.stage)
                .collect::<Vec<_>>(),
            [
                "transfer op2 environment ket i",
                "transfer op2 ket j",
                "transfer op2 operator",
                "transfer op2 bra i",
                "transfer op2 bra j",
            ]
        );
        for pair in records.windows(2) {
            assert_eq!(pair[0].output_indices, pair[1].lhs_indices);
        }
        assert_eq!(
            records.last().unwrap().output_indices,
            two_site.tensor.indices
        );
        assert_eq!(pairwise_call_count(), 5);
        assert_eq!(pairwise_operand_conjugation_count(), 0);

        reset_pairwise_counts();
        reset_transfer_contract_records();
        let two_site_right = transfer_step_op2_right(
            &env,
            &state.a,
            &state.lambda_ba,
            &state.b,
            &state.lambda_ab,
            &op2,
        )
        .unwrap();
        let records = transfer_contract_records();
        assert_eq!(records.len(), 5);
        assert_eq!(
            records
                .iter()
                .map(|record| record.stage)
                .collect::<Vec<_>>(),
            [
                "transfer op2 right environment ket j",
                "transfer op2 right ket i",
                "transfer op2 right operator",
                "transfer op2 right bra j",
                "transfer op2 right bra i",
            ]
        );
        for pair in records.windows(2) {
            assert_eq!(pair[0].output_indices, pair[1].lhs_indices);
        }
        assert_eq!(
            records.last().unwrap().output_indices,
            two_site_right.tensor.indices
        );
        assert_eq!(pairwise_call_count(), 5);
        assert_eq!(pairwise_operand_conjugation_count(), 0);
    }

    #[test]
    fn transfer_stage_failure_reports_the_exact_label() {
        let lhs = from_fn(&[new_index(2)], |_| 1.0);
        let rhs = from_fn(&[new_index(2)], |_| 1.0);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transfer_contract("transfer exact failure label", &lhs, &rhs)
        }))
        .unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap();
        assert_eq!(
            message,
            "transfer exact failure label: TensorDynLen TensorDynLen failed: Disconnected tensor network: 2 components found"
        );
    }

    #[test]
    fn fallible_closure_rejects_mismatched_schmidt_count() {
        let env = identity_env(2);
        let error = close_env_result(&env, &[1.0]).unwrap_err();
        assert!(matches!(
            error,
            ItebdError::TensorOperation { stage, message }
                if stage == "transfer environment closure"
                    && message.contains("Schmidt count 1")
        ));
    }

    #[test]
    fn norm_closes_to_one() {
        // op 無しで 1 ユニットセル転送 → λ_right² で閉じると正規化により 1
        let s = evolved_state();
        let env = identity_env(crate::tensor::dim(&s.a.left));
        let env = transfer_step(&env, &s.a, &s.lambda_ba, None); // A (left=λ_ba)
        let env = transfer_step(&env, &s.b, &s.lambda_ab, None); // B (left=λ_ab)
        let norm = close_env(&env, &s.lambda_ba); // 右端は B.right=ba bond
                                                  // Relaxed tolerance: iTEBD state is only approximately Vidal-canonical (A is
                                                  // right-canonical after the trailing B-A update), so the gauge error is ~O(dtau)
                                                  // ≈1.4e-5 at dtau=0.02. A wrong gauge would be off by O(1), so 1e-4 still validates
                                                  // correctness; the clustering test is the real physics oracle.
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn two_point_clusters_to_magnetization_squared() {
        // <σx_0 σx_r> --(r→大)--> <σx>²（connected 部が有限温度で減衰）
        let s = evolved_state();
        let mx = magnetization_x(&s);
        let far = two_point_xx(&s, 20);
        // Relaxed tolerance: connected correlator at r=20 is contaminated by the same
        // ~O(dtau) gauge error; observed diff ≈6.1e-4 at dtau=0.02. The clustering toward
        // <σx>² (vs O(1) if the machinery were wrong) is the genuine correctness check.
        assert_abs_diff_eq!(far, mx * mx, epsilon = 1e-2);
    }

    #[test]
    fn two_point_same_site_is_one() {
        // 同一サイトに σx を2回 → σx² = I → <I> = 1
        let s = evolved_state();
        let same = two_point_xx(&s, 0);
        // Kept tight: σx²=I traced against the single-site canonical density (λ_ab²) is
        // exact regardless of gauge error — observed deviation ≈4e-16 at dtau=0.02.
        assert_abs_diff_eq!(same, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn dominant_eigenvalue_is_near_one_for_evolved_state() {
        // 近似カノニカルな発展状態では転送行列の主固有値 ≈ 1
        let s = evolved_state(); // β≈1.2
        let fl = dominant_fixed_point_left(&s, 1e-12, 500);
        let fr = dominant_fixed_point_right(&s, 1e-12, 500);
        assert!(
            (fl.eigenvalue - 1.0).abs() < 1e-2,
            "left eig {}",
            fl.eigenvalue
        );
        assert!(
            (fr.eigenvalue - 1.0).abs() < 1e-2,
            "right eig {}",
            fr.eigenvalue
        );
    }

    #[test]
    fn fixed_point_is_symmetric_psd() {
        // 不動点は対称 PSD（norm 転送なので）
        let s = evolved_state();
        let fr = dominant_fixed_point_right(&s, 1e-12, 500);
        let m = fr.matrix.to_vec::<f64>().unwrap();
        let n = crate::tensor::dim(&fr.row);
        // 対称性
        for a in 0..n {
            for b in 0..n {
                assert_abs_diff_eq!(m[a + b * n], m[b + a * n], epsilon = 1e-8);
            }
        }
        // 対角 ≥ 0（PSD の必要条件）
        for a in 0..n {
            assert!(m[a + a * n] >= -1e-10);
        }
    }
}
