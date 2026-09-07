use nalgebra::DMatrix;
use crate::tensor::{Idx, Tensor, new_index, contract, relabel, scale_bond, svd_bond, dim, from_fn, Truncation};
use crate::purified_mps::{PurifiedMps, Site};
use crate::model::{LocalHamiltonian, trotter_gate};

/// 1 ステップで観測した結合次元の診断。
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    /// ユニットセルで到達した最大結合次元。
    pub max_bond: usize,
    /// 保持した最小 Schmidt 値（正規化後 λ の最小、打ち切り端の目安）。
    pub min_singular_value: f64,
    /// このステップで捨てた log-norm の合計（AB・BA 両ボンドの ln‖s‖）。
    /// 自由エネルギー帳簿（ln⟨ψ_u|ψ_u⟩ の累積）に使う。
    pub log_norm: f64,
}

/// gate テンソル: in=(p1,p2 の Id), out=(p1o,p2o 新規)。flat = s1 + d*s2、row=out, col=in。
fn gate_tensor(g: &DMatrix<f64>, p1: &Idx, p2: &Idx, p1o: &Idx, p2o: &Idx) -> Tensor {
    let d = dim(p1);
    from_fn(&[p1.clone(), p2.clone(), p1o.clone(), p2o.clone()], |ix| {
        let row = ix[2] + d * ix[3]; // out
        let col = ix[0] + d * ix[1]; // in
        g[(row, col)]
    })
}

/// X–Y ボンドを 1 回更新。bond Idx は Site から読む（outer_l=X.left, mid=X.right==Y.left, outer_r=Y.right）。
/// 戻り値: (新 X, 新 Y, 新 mid λ, 新 mid bond Idx, ln norm)。新 X.right == 新 Y.left == 新 mid bond。
/// `ln norm` = このボンド更新で捨てた log-norm（norm = ‖s‖、正規化前 Schmidt 値の L2 ノルム）。
fn apply_gate_to_bond(
    g: &DMatrix<f64>,
    x: &Site, y: &Site,
    outer_l: &[f64], mid: &[f64], outer_r: &[f64],
    trunc: &Truncation,
) -> (Site, Site, Vec<f64>, Idx, f64) {
    debug_assert_eq!(outer_l.len(), dim(&x.left), "outer_l length must match x.left bond dim");
    debug_assert_eq!(outer_r.len(), dim(&y.right), "outer_r length must match y.right bond dim");
    debug_assert_eq!(mid.len(), dim(&x.right), "mid length must match x.right bond dim");

    // 周期境界: 左端(x.left)と右端(y.right)が同一 Id を共有しうるので y.right を一旦逃がす。
    let r_env = new_index(dim(&y.right));
    let y_gamma = relabel(&y.gamma, &y.right, &r_env);

    // 1. Θ = λL · Γ_X · λmid · Γ_Y · λR （mid は X 側に 1 回だけ）
    let gx = scale_bond(&scale_bond(&x.gamma, &x.left, outer_l), &x.right, mid);
    let gy = scale_bond(&y_gamma, &r_env, outer_r);
    let theta = contract(&gx, &gy); // 脚: x.left, x.phys, x.anc, y.phys, y.anc, r_env

    // 2. ゲート適用（physical 脚を out に置換）
    let p1o = new_index(dim(&x.phys));
    let p2o = new_index(dim(&y.phys));
    let gate = gate_tensor(g, &x.phys, &y.phys, &p1o, &p2o);
    let theta_g = contract(&theta, &gate); // 脚: x.left, x.anc, y.anc, r_env, p1o, p2o

    // 3. SVD: left = {x.left, p1o, x.anc}
    let r = svd_bond(&theta_g, &[x.left.clone(), p1o.clone(), x.anc.clone()], trunc);
    let norm: f64 = r.s.iter().map(|v| v * v).sum::<f64>().sqrt();
    let new_mid: Vec<f64> = r.s.iter().map(|v| v / norm).collect();

    // 4. Γ_X = λL^{-1} · U（U の x.left をスケール）, Γ_Y = V · λR^{-1}（V の r_env をスケール）
    // Values at/below the SVD cutoff are treated as zero to avoid amplification of near-zero λ.
    let eps = 1e-12_f64;
    let inv_l: Vec<f64> = outer_l.iter().map(|&v| if v > eps { 1.0 / v } else { 0.0 }).collect();
    let inv_r: Vec<f64> = outer_r.iter().map(|&v| if v > eps { 1.0 / v } else { 0.0 }).collect();
    let gamma_x = scale_bond(&r.u, &x.left, &inv_l);
    let gamma_y = scale_bond(&r.v, &r_env, &inv_r);

    // V の bond 側 Id は bond_sim（= r.v の最終脚）。これを r.bond に揃えて X,Y が mid bond を共有。
    let bond_sim = r.v.indices.last().unwrap().clone();
    let gamma_y = relabel(&gamma_y, &bond_sim, &r.bond);
    // r_env を元の y.right Id へ戻す（更新後も x.left と y.right が同一 Id を共有＝圧縮表現維持）。
    let gamma_y = relabel(&gamma_y, &r_env, &y.right);

    // out physical を新しい phys Idx へ。
    let new_phys_x = new_index(dim(&x.phys));
    let new_phys_y = new_index(dim(&y.phys));
    let gamma_x = relabel(&gamma_x, &p1o, &new_phys_x);
    let gamma_y = relabel(&gamma_y, &p2o, &new_phys_y);

    let new_x = Site { gamma: gamma_x, left: x.left.clone(), phys: new_phys_x, anc: x.anc.clone(), right: r.bond.clone() };
    let new_y = Site { gamma: gamma_y, left: r.bond.clone(), phys: new_phys_y, anc: y.anc.clone(), right: y.right.clone() };
    (new_x, new_y, new_mid, r.bond, norm.ln())
}

/// `steps` 回の虚時間ステップを回し、`canonicalize_every` ステップごと＋最後に
/// カノニカル化する。`canonicalize_every == 0` はカノニカル化しない扱い。最後の StepInfo を返す。
pub fn imaginary_time_evolve(
    state: &mut PurifiedMps, ham: &LocalHamiltonian, dtau: f64, trunc: &Truncation,
    steps: usize, canonicalize_every: usize,
) -> StepInfo {
    let mut info = StepInfo { max_bond: 1, min_singular_value: 1.0, log_norm: 0.0 };
    for step in 1..=steps {
        info = imaginary_time_step(state, ham, dtau, trunc);
        if canonicalize_every != 0 && step % canonicalize_every == 0 {
            crate::canonicalize::canonicalize(state);
        }
    }
    // 末尾で一度カノニカル化（ループ最終ステップで既に行った場合は冪等なので省く）。
    if canonicalize_every != 0 && steps % canonicalize_every != 0 {
        crate::canonicalize::canonicalize(state);
    }
    info
}

pub fn imaginary_time_step(state: &mut PurifiedMps, ham: &LocalHamiltonian, dtau: f64, trunc: &Truncation) -> StepInfo {
    let g = trotter_gate(&ham.two_site_h, dtau);

    // --- A–B ボンド (mid = λ_ab): X=A,Y=B; outer_l=λ_ba, outer_r=λ_ba ---
    let (new_a, new_b, new_lab, bond_ab, ln_norm_ab) = apply_gate_to_bond(
        &g, &state.a, &state.b,
        &state.lambda_ba, &state.lambda_ab, &state.lambda_ba,
        trunc,
    );
    state.a = new_a;
    state.b = new_b;
    state.lambda_ab = new_lab;
    state.lambda_bond_ab = bond_ab;

    // --- B–A ボンド (mid = λ_ba): X=B,Y=A; outer_l=λ_ab(新), outer_r=λ_ab(新) ---
    let (new_b2, new_a2, new_lba, bond_ba, ln_norm_ba) = apply_gate_to_bond(
        &g, &state.b, &state.a,
        &state.lambda_ab, &state.lambda_ba, &state.lambda_ab,
        trunc,
    );
    state.b = new_b2;
    state.a = new_a2;
    state.lambda_ba = new_lba;
    state.lambda_bond_ba = bond_ba;

    let max_bond = state.lambda_ab.len().max(state.lambda_ba.len());
    debug_assert!(
        !state.lambda_ab.is_empty() && !state.lambda_ba.is_empty(),
        "lambda vectors must be non-empty after a valid SVD step"
    );
    // Safety: lambda vectors are non-empty after a valid SVD step; unwrap_or guards
    // the pathological case (zero tensor → all singular values truncated away).
    let min_singular_value = state
        .lambda_ab
        .last()
        .copied()
        .unwrap_or(0.0)
        .min(state.lambda_ba.last().copied().unwrap_or(0.0));
    // log_norm: このステップで捨てた per-site log-norm（自由エネルギー帳簿用）。
    // 1 ステップ（β を 2·dtau 進める）で 2 サイトセルの ket norm² が norm_AB²·norm_BA² 倍。
    // 1 サイトあたり Δln⟨ψ_u|ψ_u⟩ = (1/2)·2·(ln norm_AB + ln norm_BA) = ln norm_AB + ln norm_BA …
    // ではなく、norm² 倍率は両ボンド合わせて per-site で 2·(ln norm_AB + ln norm_BA) になる
    // （二重ゲートおよび u = −∂log_norm/∂β の厳密一致で確定: 係数 2 を落とすと u が半分・f がずれる）。
    StepInfo { max_bond, min_singular_value, log_norm: 2.0 * (ln_norm_ab + ln_norm_ba) }
}

/// 二次 Strang 分解 `AB(dtau/2) → BA(dtau) → AB(dtau/2)` による虚時間ステップ。
pub fn imaginary_time_step_second_order(
    state: &mut PurifiedMps,
    ham: &LocalHamiltonian,
    dtau: f64,
    trunc: &Truncation,
) -> StepInfo {
    let g_half = trotter_gate(&ham.two_site_h, 0.5 * dtau);
    let g_full = trotter_gate(&ham.two_site_h, dtau);

    let (new_a, new_b, new_lab, bond_ab, ln_norm_ab_1) = apply_gate_to_bond(
        &g_half, &state.a, &state.b,
        &state.lambda_ba, &state.lambda_ab, &state.lambda_ba, trunc,
    );
    let ab1_dim = new_lab.len();
    let ab1_min = new_lab.last().copied().unwrap_or(0.0);
    state.a = new_a;
    state.b = new_b;
    state.lambda_ab = new_lab;
    state.lambda_bond_ab = bond_ab;

    let (new_b, new_a, new_lba, bond_ba, ln_norm_ba) = apply_gate_to_bond(
        &g_full, &state.b, &state.a,
        &state.lambda_ab, &state.lambda_ba, &state.lambda_ab, trunc,
    );
    let ba_dim = new_lba.len();
    let ba_min = new_lba.last().copied().unwrap_or(0.0);
    state.b = new_b;
    state.a = new_a;
    state.lambda_ba = new_lba;
    state.lambda_bond_ba = bond_ba;

    let (new_a, new_b, new_lab, bond_ab, ln_norm_ab_2) = apply_gate_to_bond(
        &g_half, &state.a, &state.b,
        &state.lambda_ba, &state.lambda_ab, &state.lambda_ba, trunc,
    );
    let ab2_dim = new_lab.len();
    let ab2_min = new_lab.last().copied().unwrap_or(0.0);
    state.a = new_a;
    state.b = new_b;
    state.lambda_ab = new_lab;
    state.lambda_bond_ab = bond_ab;

    StepInfo {
        max_bond: ab1_dim.max(ba_dim).max(ab2_dim),
        min_singular_value: ab1_min.min(ba_min).min(ab2_min),
        log_norm: 2.0 * (ln_norm_ab_1 + ln_norm_ba + ln_norm_ab_2),
    }
}

/// 累積した per-site log-norm から自由エネルギー密度を返す。
/// f(β) = −(1/β)[ ln d + accum_log_norm_per_site ]、`accum_log_norm_per_site`
/// = (全ステップ＋全 canonicalize で捨てた log-norm の総和) / セルのサイト数(=2)。
pub fn free_energy_from_log_norm(accum_log_norm_per_site: f64, beta: f64, d: usize) -> f64 {
    -(1.0 / beta) * ((d as f64).ln() + accum_log_norm_per_site)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::purified_mps::infinite_temperature;
    use crate::observable::energy_density;
    use crate::model::Tfim;
    use approx::assert_abs_diff_eq;

    fn manual_second_order_step(
        state: &mut PurifiedMps,
        ham: &LocalHamiltonian,
        dtau: f64,
        trunc: &Truncation,
    ) -> StepInfo {
        let g_half = trotter_gate(&ham.two_site_h, 0.5 * dtau);
        let g_full = trotter_gate(&ham.two_site_h, dtau);

        let (new_a, new_b, new_lab, bond_ab, ab1_log) = apply_gate_to_bond(
            &g_half, &state.a, &state.b,
            &state.lambda_ba, &state.lambda_ab, &state.lambda_ba, trunc,
        );
        let ab1_dim = new_lab.len();
        let ab1_min = new_lab.last().copied().unwrap_or(0.0);
        state.a = new_a;
        state.b = new_b;
        state.lambda_ab = new_lab;
        state.lambda_bond_ab = bond_ab;

        let (new_b, new_a, new_lba, bond_ba, ba_log) = apply_gate_to_bond(
            &g_full, &state.b, &state.a,
            &state.lambda_ab, &state.lambda_ba, &state.lambda_ab, trunc,
        );
        let ba_dim = new_lba.len();
        let ba_min = new_lba.last().copied().unwrap_or(0.0);
        state.b = new_b;
        state.a = new_a;
        state.lambda_ba = new_lba;
        state.lambda_bond_ba = bond_ba;

        let (new_a, new_b, new_lab, bond_ab, ab2_log) = apply_gate_to_bond(
            &g_half, &state.a, &state.b,
            &state.lambda_ba, &state.lambda_ab, &state.lambda_ba, trunc,
        );
        let ab2_dim = new_lab.len();
        let ab2_min = new_lab.last().copied().unwrap_or(0.0);
        state.a = new_a;
        state.b = new_b;
        state.lambda_ab = new_lab;
        state.lambda_bond_ab = bond_ab;

        StepInfo {
            max_bond: ab1_dim.max(ba_dim).max(ab2_dim),
            min_singular_value: ab1_min.min(ba_min).min(ab2_min),
            log_norm: 2.0 * (ab1_log + ba_log + ab2_log),
        }
    }

    #[test]
    fn second_order_matches_explicit_strang_sequence() {
        let ham = Tfim { j: 1.0, g: 0.7 }.local();
        let trunc = Truncation { epsilon: 1e-13, max_bond: Some(32) };
        let mut actual = infinite_temperature(2);
        let mut expected = infinite_temperature(2);

        let actual_info = imaginary_time_step_second_order(&mut actual, &ham, 0.04, &trunc);
        let expected_info = manual_second_order_step(&mut expected, &ham, 0.04, &trunc);

        assert_eq!(actual_info.max_bond, expected_info.max_bond);
        assert_abs_diff_eq!(
            actual_info.min_singular_value,
            expected_info.min_singular_value,
            epsilon = 1e-13
        );
        assert_abs_diff_eq!(actual_info.log_norm, expected_info.log_norm, epsilon = 1e-13);
        assert_eq!(actual.lambda_ab.len(), expected.lambda_ab.len());
        assert_eq!(actual.lambda_ba.len(), expected.lambda_ba.len());
        for (&actual_lambda, &expected_lambda) in actual.lambda_ab.iter().zip(&expected.lambda_ab) {
            assert_abs_diff_eq!(actual_lambda, expected_lambda, epsilon = 1e-13);
        }
        for (&actual_lambda, &expected_lambda) in actual.lambda_ba.iter().zip(&expected.lambda_ba) {
            assert_abs_diff_eq!(actual_lambda, expected_lambda, epsilon = 1e-13);
        }
        assert_abs_diff_eq!(
            energy_density(&actual, &ham),
            energy_density(&expected, &ham),
            epsilon = 1e-12
        );
    }

    #[test]
    fn second_order_zero_time_preserves_infinite_temperature_observables() {
        let ham = Tfim { j: 1.0, g: 0.7 }.local();
        let mut state = infinite_temperature(2);
        let info = imaginary_time_step_second_order(
            &mut state,
            &ham,
            0.0,
            &Truncation { epsilon: 1e-13, max_bond: Some(32) },
        );

        assert_abs_diff_eq!(info.log_norm, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(energy_density(&state, &ham), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn second_order_stays_finite_over_twenty_five_steps() {
        let ham = Tfim { j: 1.0, g: 0.7 }.local();
        let trunc = Truncation { epsilon: 1e-13, max_bond: Some(32) };
        let mut state = infinite_temperature(2);
        let mut info = StepInfo { max_bond: 1, min_singular_value: 1.0, log_norm: 0.0 };

        for _ in 0..25 {
            info = imaginary_time_step_second_order(&mut state, &ham, 0.02, &trunc);
        }

        assert!(info.log_norm.is_finite(), "log norm must be finite: {}", info.log_norm);
        assert!(energy_density(&state, &ham).is_finite());
        assert!(!state.lambda_ab.is_empty());
        assert!(!state.lambda_ba.is_empty());
        assert!(info.min_singular_value > 0.0);
        assert!(state.lambda_ab.len() <= 32);
        assert!(state.lambda_ba.len() <= 32);
        assert!(info.max_bond <= 32);
    }

    #[test]
    fn one_step_lowers_energy_below_zero() {
        let mut state = infinite_temperature(2);
        let m = Tfim { j: 1.0, g: 0.5 };
        let ham = m.local();
        let u0 = energy_density(&state, &ham);
        assert_abs_diff_eq!(u0, 0.0, epsilon = 1e-12);
        let info = imaginary_time_step(&mut state, &ham, 0.05, &Truncation { epsilon: 1e-12, max_bond: Some(16) });
        assert!(info.max_bond >= 1);
        // min retained Schmidt value is a valid probability amplitude in (0, 1]
        assert!(
            info.min_singular_value > 0.0 && info.min_singular_value <= 1.0,
            "min_singular_value out of range: {}",
            info.min_singular_value
        );
        let u1 = energy_density(&state, &ham);
        assert!(u1 < -1e-6, "energy should decrease, got {u1}");
    }

    #[test]
    fn norm_stays_finite_over_many_steps() {
        let mut state = infinite_temperature(2);
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        for _ in 0..200 {
            imaginary_time_step(&mut state, &ham, 0.02, &Truncation { epsilon: 1e-10, max_bond: Some(16) });
        }
        let u = energy_density(&state, &ham);
        assert!(u.is_finite() && u < 0.0);
    }

    #[test]
    fn smaller_epsilon_keeps_more_bonds() {
        let m = Tfim { j: 1.0, g: 1.0 };
        let run = |eps: f64| -> usize {
            let mut s = infinite_temperature(2);
            let ham = m.local();
            let mut last = StepInfo { max_bond: 1, min_singular_value: 1.0, log_norm: 0.0 };
            for _ in 0..40 {
                last = imaginary_time_step(&mut s, &ham, 0.05, &Truncation { epsilon: eps, max_bond: Some(128) });
            }
            last.max_bond
        };
        let coarse = run(1e-4);
        let fine = run(1e-10);
        assert!(fine > coarse, "smaller epsilon should keep strictly more bonds: fine={fine}, coarse={coarse}");
        assert!(fine > 1, "fine run should grow beyond bond dim 1");
    }

    #[test]
    fn free_energy_high_temperature_limit() {
        // β→0 で βf → -ln 2
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let dtau = 0.005;
        let trunc = Truncation { epsilon: 1e-12, max_bond: Some(32) };
        let mut s = infinite_temperature(2);
        let mut accum = 0.0;
        let steps = 5; // 小さい β
        for _ in 0..steps {
            let info = imaginary_time_step(&mut s, &ham, dtau, &trunc);
            accum += info.log_norm;
            accum += crate::canonicalize::canonicalize(&mut s);
        }
        let beta = 2.0 * (steps as f64) * dtau; // = 0.05
        let f = free_energy_from_log_norm(accum / 2.0, beta, 2);
        assert_abs_diff_eq!(beta * f, -(2.0_f64).ln(), epsilon = 1e-2);
    }

    #[test]
    fn free_energy_matches_exact_tfim() {
        use crate::exact::free_energy_density;
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let dtau = 0.01;
        let trunc = Truncation { epsilon: 1e-12, max_bond: Some(48) };
        let mut s = infinite_temperature(2);
        let mut accum = 0.0;
        let beta = 1.0;
        let steps = (beta / (2.0 * dtau)) as usize; // 50
        for _ in 0..steps {
            let info = imaginary_time_step(&mut s, &ham, dtau, &trunc);
            accum += info.log_norm;
            accum += crate::canonicalize::canonicalize(&mut s);
        }
        let f = free_energy_from_log_norm(accum / 2.0, beta, 2);
        let f_exact = free_energy_density(1.0, 1.0, beta, 4000);
        assert!((f - f_exact).abs() < 5e-3, "f={f} exact={f_exact}");
    }
}
