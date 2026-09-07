use nalgebra::DMatrix;
use crate::tensor::{Idx, Tensor, new_index, from_fn, contract, scalar, relabel, scale_bond, dim};
use crate::purified_mps::{PurifiedMps, Site};
use crate::model::LocalHamiltonian;

/// physical 2 脚に作用する演算子テンソル。
/// in 脚 = (p1_in, p2_in)（state の physical Id に一致）、out 脚 = (p1_out, p2_out)（新規 Id）。
/// h_flat index = s1 + d*s2、row=out, col=in。
pub(crate) fn two_site_op_tensor(h: &DMatrix<f64>, p1_in: &Idx, p2_in: &Idx, p1_out: &Idx, p2_out: &Idx) -> Tensor {
    let d = dim(p1_in);
    from_fn(&[p1_in.clone(), p2_in.clone(), p1_out.clone(), p2_out.clone()], |ix| {
        let row = ix[2] + d * ix[3]; // out
        let col = ix[0] + d * ix[1]; // in
        h[(row, col)]
    })
}

/// 1 ボンドの <h>/<Θ|Θ> を、X(左)・Y(右) の Γ と λ から評価。
/// outer_l = X.left 上の λ、mid = X.right(==Y.left) 上の λ、outer_r = Y.right 上の λ。
fn bond_energy(h: &DMatrix<f64>, x: &Site, y: &Site, outer_l: &[f64], mid: &[f64], outer_r: &[f64]) -> f64 {
    debug_assert_eq!(outer_l.len(), dim(&x.left), "outer_l length must match x.left bond dim");
    debug_assert_eq!(outer_r.len(), dim(&y.right), "outer_r length must match y.right bond dim");
    debug_assert_eq!(mid.len(), dim(&x.right), "mid length must match x.right bond dim");

    // 無限 2 サイト窓: 左端 X.left と右端 Y.right は周期境界で同一 Id を共有しうるので、
    // Y.right を新 Id に逃がして「別の環境脚」として扱う（さもないと theta が同一 Id の脚を 2 本持つ）。
    let r_env = new_index(dim(&y.right));
    let y_gamma = relabel(&y.gamma, &y.right, &r_env);

    // Θ = λL · Γ_X · λmid · Γ_Y · λR （λ は scale_bond で畳み込む。mid は X 側に 1 回だけ適用）
    let gx = scale_bond(&scale_bond(&x.gamma, &x.left, outer_l), &x.right, mid);
    let gy = scale_bond(&y_gamma, &r_env, outer_r);
    let theta = contract(&gx, &gy); // mid ボンド (X.right==Y.left) を縮約
    // theta の脚: x.left, x.phys, x.anc, y.phys, y.anc, r_env（すべて異なる Id）

    // <Θ| (h⊗I_anc) |Θ>: physical 脚にのみ op を作用
    let p1o = new_index(dim(&x.phys));
    let p2o = new_index(dim(&y.phys));
    let op = two_site_op_tensor(h, &x.phys, &y.phys, &p1o, &p2o);
    let h_ket = contract(&theta, &op); // x.phys,y.phys -> p1o,p2o
    let bra = relabel(&relabel(&theta, &x.phys, &p1o), &y.phys, &p2o);
    let num = scalar(&contract(&h_ket, &bra));

    // 規格化 <Θ|Θ>（全脚自己縮約）
    let theta_bra = theta.clone();
    let den = scalar(&contract(&theta, &theta_bra));
    num / den
}

/// 単一サイト演算子テンソル: in=p_in、out=p_out。o[out][in]。
fn single_site_op_tensor(o: &DMatrix<f64>, p_in: &Idx, p_out: &Idx) -> Tensor {
    from_fn(&[p_in.clone(), p_out.clone()], |ix| o[(ix[1], ix[0])])
}

/// 単一サイトの <O> = <Θ|O|Θ>/<Θ|Θ>、Θ = λ_left·Γ·λ_right。
fn site_expectation(o: &DMatrix<f64>, site: &Site, left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), dim(&site.left));
    debug_assert_eq!(right.len(), dim(&site.right));
    let theta = scale_bond(&scale_bond(&site.gamma, &site.left, left), &site.right, right);
    // theta の脚: site.left, site.phys, site.anc, site.right（すべて別 Id）
    let po = new_index(dim(&site.phys));
    let op = single_site_op_tensor(o, &site.phys, &po);
    let o_ket = contract(&theta, &op); // phys -> po
    let bra = relabel(&theta, &site.phys, &po);
    let num = scalar(&contract(&o_ket, &bra));
    let theta_bra = theta.clone();
    let den = scalar(&contract(&theta, &theta_bra));
    num / den
}

/// Generic single-site magnetization: average of ⟨O⟩ over sites A and B.
pub fn magnetization(state: &PurifiedMps, op: &DMatrix<f64>) -> f64 {
    let ma = site_expectation(op, &state.a, &state.lambda_ba, &state.lambda_ab);
    let mb = site_expectation(op, &state.b, &state.lambda_ab, &state.lambda_ba);
    0.5 * (ma + mb)
}

pub fn magnetization_x(state: &PurifiedMps) -> f64 { magnetization(state, &crate::model::pauli_x()) }
pub fn magnetization_z(state: &PurifiedMps) -> f64 { magnetization(state, &crate::model::pauli_z()) }

pub fn energy_density(state: &PurifiedMps, ham: &LocalHamiltonian) -> f64 {
    let h = &ham.two_site_h;
    // A–B ボンド: X=A, Y=B; outer_l=λ_ba(A.left), mid=λ_ab, outer_r=λ_ba(B.right)
    let e_ab = bond_energy(h, &state.a, &state.b, &state.lambda_ba, &state.lambda_ab, &state.lambda_ba);
    // B–A ボンド: X=B, Y=A; outer_l=λ_ab(B.left), mid=λ_ba, outer_r=λ_ab(A.right)
    let e_ba = bond_energy(h, &state.b, &state.a, &state.lambda_ab, &state.lambda_ba, &state.lambda_ab);
    // サイトあたり = ボンドあたりエネルギーの平均（TFIM は 1 サイト 1 ボンド）
    0.5 * (e_ab + e_ba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::purified_mps::infinite_temperature;
    use crate::model::Tfim;
    use approx::assert_abs_diff_eq;

    #[test]
    fn magnetization_z_zero_at_infinite_temperature() {
        let s = infinite_temperature(2);
        assert_abs_diff_eq!(magnetization_z(&s), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn infinite_temperature_energy_is_zero() {
        // β=0（無限温度）では <sz sz>=0, <sx>=0 なので u=0
        let state = infinite_temperature(2);
        let m = Tfim { j: 1.0, g: 0.5 };
        let ham = m.local();
        let u = energy_density(&state, &ham);
        assert_abs_diff_eq!(u, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn infinite_temperature_magnetization_is_zero() {
        // β=0（無限温度）では <σx> = 0
        let state = infinite_temperature(2);
        let mx = magnetization_x(&state);
        assert_abs_diff_eq!(mx, 0.0, epsilon = 1e-12);
    }
}
