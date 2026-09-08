use crate::purified_mps::{PurifiedMps, Site};
use crate::transfer::{dominant_fixed_point_left, dominant_fixed_point_right};
use crate::tensor::{Idx, Tensor, new_index, from_fn, contract, dim};
use nalgebra::DMatrix;

/// χ×χ Tensor(row,col) を nalgebra DMatrix へ（どちらも col-major: (a,b) at a + b*n）。
fn to_dmatrix(t: &Tensor, n: usize) -> DMatrix<f64> {
    DMatrix::from_vec(n, n, t.to_vec::<f64>().unwrap())
}

/// 対称 PSD 行列の平方根因子。floor 以下（または負）の固有値はクランプして 0。
/// 返り値 (F, Fi): rho ≈ F Fᵀ かつ Fi F = I（support 上）。F = U sqrt(Λ), Fi = sqrt(Λ)⁻¹ Uᵀ。
fn psd_factor(rho: &DMatrix<f64>, floor: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = rho.nrows();
    // Match the complex canonicalizer's stable decomposition for weakly coupled blocks.
    let tensor = Tensor::from_dense(vec![new_index(n), new_index(n)], rho.as_slice().to_vec()).unwrap();
    let eig = tensor.hermitian_eigendecomposition(1e-12).unwrap();
    let eigenvectors = DMatrix::from_vec(n, n, eig.eigenvectors.to_vec::<f64>().unwrap());
    let mut sqrt_d = DMatrix::<f64>::zeros(n, n);
    let mut inv_sqrt_d = DMatrix::<f64>::zeros(n, n);
    for k in 0..n {
        let v = eig.eigenvalues[k].max(0.0);
        if v > floor {
            let s = v.sqrt();
            sqrt_d[(k, k)] = s;
            inv_sqrt_d[(k, k)] = 1.0 / s;
        }
    }
    let u = &eigenvectors;
    let f = u * &sqrt_d;
    let fi = &inv_sqrt_d * u.transpose();
    (f, fi)
}

/// Γ の `bond` 脚（dim n_old）に行列 G[new, old] を左から作用させ、`bond`→`new_bond` に置換。
/// 結果[new] = Σ_old Γ[old] · G[new, old]（= G·Γ_leg）。
fn apply_matrix_to_leg(gamma: &Tensor, bond: &Idx, g: &DMatrix<f64>, new_bond: &Idx) -> Tensor {
    debug_assert_eq!(g.nrows(), dim(new_bond));
    debug_assert_eq!(g.ncols(), dim(bond));
    let gt = from_fn(&[bond.clone(), new_bond.clone()], |ix| g[(ix[1], ix[0])]);
    contract(gamma, &gt)
}

/// 1 本のボンド（左テンソル X の右脚 == 右テンソル Y の左脚、bond weight = `lambda`）を
/// Orús–Vidal で正準化する。L=左不動点(bare 左脚側), R=右不動点(λ 畳み込み側)。
///
/// この codebase の transfer は A = λ_left·Γ。左不動点 L は bare な脚（X の右脚側）上に、
/// 右不動点 R は λ を畳んだ脚（Y の左脚側）上にある。標準式 SVD(Y λ X_bare) は、
/// X_folded = diag(λ) X_bare なので SVD(Y X_folded) と等しい。
///
/// 連鎖 Γ_X λ Γ_Y = Γ_X (Y⁻¹U) S (Vᵀ X_bare⁻¹) Γ_Y、X_bare⁻¹ = X_folded⁻¹ diag(λ)。
/// 返り: (新 λ', GR=Vᵀ X_folded⁻¹ diag(λ)[右テンソルの左脚へ], GL=Y⁻¹U[左テンソルの右脚へ])。
fn bond_gauge(
    lambda: &[f64],
    l_mat: &DMatrix<f64>,
    r_mat: &DMatrix<f64>,
    floor: f64,
) -> (Vec<f64>, DMatrix<f64>, DMatrix<f64>) {
    let n = lambda.len();
    let (x, xi) = psd_factor(r_mat, floor); // R = X Xᵀ
    let (fy, fyi) = psd_factor(l_mat, floor); // L = Fy Fyᵀ → Y = Fyᵀ
    let y = fy.transpose();
    let yi = fyi.transpose();
    let lam = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(n, lambda.iter().cloned()));
    let theta = &y * &x; // = Y λ X_bare
    let svd = theta.clone().svd(true, true);
    let u = svd.u.clone().expect("svd u");
    let v_t = svd.v_t.clone().expect("svd v_t");
    let s = &svd.singular_values;
    let nrm: f64 = s.iter().map(|v| v * v).sum::<f64>().sqrt();
    let new_lambda: Vec<f64> = s.iter().map(|v| v / nrm).collect();
    let gl = &yi * &u; // Y⁻¹ U
    let gr = &v_t * &xi * &lam; // Vᵀ X_folded⁻¹ diag(λ)
    (new_lambda, gr, gl)
}

/// `a.left == b.right` のボンド（"ba タイプ"。左テンソル=B の右脚, 右テンソル=A の左脚）を正準化。
/// 不動点は dominant_fixed_point_left/right からこのボンド上で得られる。
fn canonicalize_reference_bond(state: &mut PurifiedMps, floor: f64) {
    let fl = dominant_fixed_point_left(state, 1e-13, 4000);
    let fr = dominant_fixed_point_right(state, 1e-13, 4000);
    let n = dim(&state.a.left);
    let l = to_dmatrix(&fl.matrix, n);
    let r = to_dmatrix(&fr.matrix, n);
    let (new_lambda, gr, gl) = bond_gauge(&state.lambda_ba, &l, &r, floor);
    let new_bond = new_index(new_lambda.len());

    // GR は右テンソル A の左脚へ左から（G·Γ）。GL は左テンソル B の右脚へ右から（Γ·G）なので転置。
    let new_a_gamma = apply_matrix_to_leg(&state.a.gamma, &state.a.left, &gr, &new_bond);
    let new_b_gamma = apply_matrix_to_leg(&state.b.gamma, &state.b.right, &gl.transpose(), &new_bond);

    state.a = Site {
        gamma: new_a_gamma,
        left: new_bond.clone(),
        phys: state.a.phys.clone(),
        anc: state.a.anc.clone(),
        right: state.a.right.clone(),
    };
    state.b = Site {
        gamma: new_b_gamma,
        left: state.b.left.clone(),
        phys: state.b.phys.clone(),
        anc: state.b.anc.clone(),
        right: new_bond.clone(),
    };
    state.lambda_ba = new_lambda;
    state.lambda_bond_ba = new_bond;
}

/// セルを 1 サイト回転した view を作る（所有権を移す）。
/// 回転後: a' = 旧 b, b' = 旧 a, lambda_ba' = 旧 lambda_ab, lambda_ab' = 旧 lambda_ba。
/// これにより「a'.left == b'.right」= 旧 ab ボンドとなり、ba 用機構をそのまま適用できる。
fn rotate(state: PurifiedMps) -> PurifiedMps {
    PurifiedMps {
        a: state.b,
        b: state.a,
        lambda_ab: state.lambda_ba,
        lambda_bond_ab: state.lambda_bond_ba,
        lambda_ba: state.lambda_ab,
        lambda_bond_ba: state.lambda_bond_ab,
    }
}

/// Orús–Vidal 正準化。最後にセル norm を 1 にする再スケールを行い、自由エネルギー帳簿へ
/// 加算すべき「状態ノルム(log)への寄与」を返す。
///
/// 重要（二重ゲートで確定）: この再スケールは**純ゲージ変換**であり、状態の物理ノルムを
/// 変えない（cell norm = 転送主固有値は、Vidal-canonical な λ 規約のもとでは恒等的に 1 で、
/// reference_cell_norm が返す ~d の値は ancilla トレースに由来する規約上の定数。再スケールで
/// それを 1 に揃えても、imaginary_time_step が拾う per-bond log-norm が既に全物理ノルム変化を
/// 捕捉しているため、帳簿への寄与は 0）。よって 0.0 を返す。
/// （β→0 で βf→−ln2、有限 β で exact f 一致の二重ゲートで検証済み。-0.5·ln(nrm) を返すと
/// ~d 由来の定数が毎ステップ重複計上され f が発散する。）
pub fn canonicalize(state: &mut PurifiedMps) -> f64 {
    if dim(&state.a.left) == 1 && dim(&state.a.right) == 1 {
        return 0.0;
    }
    let floor = 1e-12;

    // 1) ba ボンド（a.left == b.right）を正準化。
    canonicalize_reference_bond(state, floor);

    // 2) ab ボンドを正準化: セルを回転して ab を "ba タイプ" にし、同じ機構を適用、戻す。
    let rotated = rotate(std::mem::replace(state, dummy()));
    let mut rotated = rotated;
    canonicalize_reference_bond(&mut rotated, floor);
    *state = rotate(rotated); // 回転は対合（2 回で元に戻る）。

    // 3) 全体スケールを正規化: セル norm（転送固有値）を 1 にする。
    // cell_norm は Γ_A, Γ_B に対し ∝ a²b²（各 ket+bra）。各 Γ を N^{-1/4} 倍。
    let nrm = reference_cell_norm(state);
    if nrm > 0.0 {
        let s = nrm.powf(-0.25);
        state.a.gamma = scale_tensor(&state.a.gamma, s);
        state.b.gamma = scale_tensor(&state.b.gamma, s);
    }
    // 純ゲージ（上記参照）: 自由エネルギー帳簿への寄与は 0。
    0.0
}

/// 全要素を s 倍した Tensor。
fn scale_tensor(t: &Tensor, s: f64) -> Tensor {
    let data: Vec<f64> = t.to_vec::<f64>().unwrap().iter().map(|v| v * s).collect();
    Tensor::from_dense(t.indices.clone(), data).unwrap()
}

/// セル norm（A then B 転送 → λ_ba² で閉じる）。canonicalize 内部の正規化用。
fn reference_cell_norm(s: &PurifiedMps) -> f64 {
    use crate::transfer::{identity_env, transfer_step, close_env};
    let env = identity_env(dim(&s.a.left));
    let env = transfer_step(&env, &s.a, &s.lambda_ba, None);
    let env = transfer_step(&env, &s.b, &s.lambda_ab, None);
    close_env(&env, &s.lambda_ba)
}

/// std::mem::replace 用の一時ダミー（即座に上書きされる）。
fn dummy() -> PurifiedMps {
    crate::purified_mps::infinite_temperature(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::purified_mps::{infinite_temperature, PurifiedMps};
    use crate::itebd::imaginary_time_step;
    use crate::observable::{energy_density, magnetization_x};
    use crate::transfer::{identity_env, transfer_step, close_env};
    use crate::model::Tfim;
    use crate::tensor::{Truncation, dim};
    use approx::assert_abs_diff_eq;

    #[test]
    fn psd_factor_reconstructs_weakly_coupled_density() {
        let mut density = DMatrix::<f64>::zeros(3, 3);
        density[(0, 0)] = 1.0;
        density[(1, 1)] = 1e-6;
        density[(2, 2)] = 1e-3;
        density[(1, 2)] = 1e-18;
        density[(2, 1)] = 1e-18;
        let (factor, inverse) = psd_factor(&density, 1e-12);
        let residual = (&factor * factor.transpose() - &density).norm();
        assert!(residual < 1e-14, "factor residual={residual:e}");
        assert!((&inverse * &density * inverse.transpose() - DMatrix::identity(3, 3)).norm() < 1e-12);
    }

    fn evolved(beta: f64) -> PurifiedMps {
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let mut s = infinite_temperature(2);
        let trunc = Truncation { epsilon: 1e-12, max_bond: Some(48) };
        let n = (beta / (2.0 * 0.02)) as usize;
        for _ in 0..n { imaginary_time_step(&mut s, &ham, 0.02, &trunc); }
        s
    }

    fn cell_norm(s: &PurifiedMps) -> f64 {
        let env = identity_env(dim(&s.a.left));
        let env = transfer_step(&env, &s.a, &s.lambda_ba, None);
        let env = transfer_step(&env, &s.b, &s.lambda_ab, None);
        close_env(&env, &s.lambda_ba)
    }

    // canonicalize は純ゲージ変換: 結合次元を変えない（隠れた打ち切りなし）。
    #[test]
    fn canonicalize_preserves_bond_dimension() {
        let mut s = evolved(2.0);
        let chi_ab = s.lambda_ab.len();
        let chi_ba = s.lambda_ba.len();
        canonicalize(&mut s);
        assert_eq!(s.lambda_ab.len(), chi_ab, "ab bond dim changed (hidden truncation?)");
        assert_eq!(s.lambda_ba.len(), chi_ba, "ba bond dim changed (hidden truncation?)");
    }

    // 正しい canonicalize により局所推定子 energy_density / magnetization_x が真値（canonical 値）
    // になる。非 canonical の simple-update 入力では λ²-proxy がずれているのを canonicalize が直す。
    // energy_density は gauge 不変ではないため「前後で不変」ではなく「真値に近づく/悪化しない」で判定。
    #[test]
    fn canonicalize_makes_local_observable_correct() {
        use crate::exact::{exact_energy_density, exact_magnetization_x};
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let beta = 2.0;
        let mut s = evolved(beta);
        let u_before = energy_density(&s, &ham);
        canonicalize(&mut s);
        let u_after = energy_density(&s, &ham);
        let mx_after = magnetization_x(&s);
        let u_exact = exact_energy_density(m.j, m.g, beta, 4000);
        let mx_exact = exact_magnetization_x(m.j, m.g, beta, 4000);
        // canonicalize は局所推定子の精度を悪化させない（実際は改善する）
        assert!((u_after - u_exact).abs() <= (u_before - u_exact).abs() + 1e-12,
            "accuracy worsened: before={u_before} after={u_after} exact={u_exact}");
        // canonical 後は局所推定子が真値（厳密解に近い canonical 値、残差は Trotter/χ 誤差のみ）
        assert!((u_after - u_exact).abs() < 1e-3, "post-canon energy off: {u_after} vs {u_exact}");
        assert!((mx_after - mx_exact).abs() < 1e-3, "post-canon mx off: {mx_after} vs {mx_exact}");
    }

    #[test]
    fn canonicalize_makes_norm_machine_precision() {
        let mut s = evolved(2.0);
        canonicalize(&mut s);
        let n = cell_norm(&s);
        assert_abs_diff_eq!(n, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let mut s = evolved(2.0);
        canonicalize(&mut s);
        let u1 = energy_density(&s, &ham);
        canonicalize(&mut s);
        let u2 = energy_density(&s, &ham);
        assert_abs_diff_eq!(u2, u1, epsilon = 1e-10);
    }

    #[test]
    fn canonicalize_trivial_at_bond_dim_one() {
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let mut s = infinite_temperature(2);
        let u_before = energy_density(&s, &ham);
        canonicalize(&mut s);
        assert_abs_diff_eq!(energy_density(&s, &ham), u_before, epsilon = 1e-12);
    }

    #[test]
    fn canonicalization_improves_low_t_specific_heat() {
        use crate::itebd::imaginary_time_evolve;
        use crate::variance::specific_heat;
        use crate::exact::exact_specific_heat;
        let m = Tfim { j: 1.0, g: 1.0 };
        let ham = m.local();
        let dtau = 0.02;
        let trunc = Truncation { epsilon: 1e-12, max_bond: Some(48) };
        let beta = 3.0;
        let steps = (beta / (2.0 * dtau)) as usize;

        // カノニカル化あり（毎ステップ）
        let mut s_can = infinite_temperature(2);
        imaginary_time_evolve(&mut s_can, &ham, dtau, &trunc, steps, 1);
        let c_can = specific_heat(&s_can, &ham, beta).unwrap();

        // カノニカル化なし（simple update のみ）
        let mut s_raw = infinite_temperature(2);
        for _ in 0..steps { imaginary_time_step(&mut s_raw, &ham, dtau, &trunc); }
        let raw_error = specific_heat(&s_raw, &ham, beta).unwrap_err();
        assert!(matches!(
            raw_error,
            crate::itebd_error::ItebdError::SpecificHeatTailNonConvergence { .. }
        ));

        let c_exact = exact_specific_heat(m.j, m.g, beta, 4000);
        // The canonicalized estimate converges and remains quantitatively close to the exact
        // result; the raw simple-update state now reports its unresolved tail instead of silently
        // truncating at the old gauge-noise heuristic.
        assert!(
            (c_can - c_exact).abs() < 2e-2,
            "canonical specific heat should remain accurate: c_can={c_can}, exact={c_exact}"
        );
    }
}
