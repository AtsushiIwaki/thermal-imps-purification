use std::f64::consts::PI;

pub fn dispersion(j: f64, g: f64, k: f64) -> f64 {
    2.0 * (j * j + g * g - 2.0 * j * g * k.cos()).max(0.0).sqrt()
}

/// f(β) = -(1/β)(1/2π)∫_0^{2π} ln(2 cosh(β ε_k /2)) dk
/// (uniform left-endpoint Riemann sum over [0,2π); exact in the limit for a periodic integrand)
pub fn free_energy_density(j: f64, g: f64, beta: f64, nk: usize) -> f64 {
    let mut integral = 0.0;
    for n in 0..nk {
        let k = 2.0 * PI * (n as f64) / (nk as f64);
        let e = dispersion(j, g, k);
        integral += (2.0 * (0.5 * beta * e).cosh()).ln();
    }
    integral *= 2.0 * PI / (nk as f64); // dk
    -(1.0 / beta) * integral / (2.0 * PI)
}

/// <σx>(β) = -∂f/∂g （H = -J ΣZZ - g ΣX なので ∂f/∂g = -<σx>）
pub fn exact_magnetization_x(j: f64, g: f64, beta: f64, nk: usize) -> f64 {
    let dg = 1e-4;
    -(free_energy_density(j, g + dg, beta, nk) - free_energy_density(j, g - dg, beta, nk)) / (2.0 * dg)
}

/// u(β) = ∂(β f)/∂β via central difference
pub fn exact_energy_density(j: f64, g: f64, beta: f64, nk: usize) -> f64 {
    let db = (1e-4 * beta).max(1e-6);
    let bf = |b: f64| b * free_energy_density(j, g, b, nk);
    (bf(beta + db) - bf(beta - db)) / (2.0 * db)
}

/// XY dispersion: ε_k = 2 sqrt((cos k − h)² + γ² sin² k)
/// At γ=1 this equals the TFIM dispersion(1, h, k).
pub fn xy_dispersion(gamma: f64, h: f64, k: f64) -> f64 {
    let c = k.cos();
    let s = k.sin();
    2.0 * ((c - h).powi(2) + gamma * gamma * s * s).max(0.0).sqrt()
}

/// f(β) for the XY model — same integral form as free_energy_density but with xy_dispersion.
pub fn xy_free_energy_density(gamma: f64, h: f64, beta: f64, nk: usize) -> f64 {
    let mut integral = 0.0;
    for n in 0..nk {
        let k = 2.0 * PI * (n as f64) / (nk as f64);
        let e = xy_dispersion(gamma, h, k);
        integral += (2.0 * (0.5 * beta * e).cosh()).ln();
    }
    integral *= 2.0 * PI / (nk as f64);
    -(1.0 / beta) * integral / (2.0 * PI)
}

/// u(β) = ∂(β f)/∂β for the XY model via central difference.
pub fn xy_energy_density(gamma: f64, h: f64, beta: f64, nk: usize) -> f64 {
    let db = (1e-4 * beta).max(1e-6);
    let bf = |b: f64| b * xy_free_energy_density(gamma, h, b, nk);
    (bf(beta + db) - bf(beta - db)) / (2.0 * db)
}

/// <σz>(β) = −∂f/∂h for the XY model (transverse field polarization).
pub fn xy_magnetization_z(gamma: f64, h: f64, beta: f64, nk: usize) -> f64 {
    let dh = 1e-4;
    -(xy_free_energy_density(gamma, h + dh, beta, nk)
        - xy_free_energy_density(gamma, h - dh, beta, nk))
        / (2.0 * dh)
}

/// C(β) = −β² ∂u/∂β for the XY model.
pub fn xy_specific_heat(gamma: f64, h: f64, beta: f64, nk: usize) -> f64 {
    let db = (1e-3 * beta).max(1e-4);
    let u = |b: f64| xy_energy_density(gamma, h, b, nk);
    let dudb = (u(beta + db) - u(beta - db)) / (2.0 * db);
    -beta * beta * dudb
}

/// C(β) = -β² ∂u/∂β （u = exact_energy_density）
pub fn exact_specific_heat(j: f64, g: f64, beta: f64, nk: usize) -> f64 {
    let db = (1e-3 * beta).max(1e-4);
    let u = |b: f64| exact_energy_density(j, g, b, nk);
    let dudb = (u(beta + db) - u(beta - db)) / (2.0 * db);
    -beta * beta * dudb
}

/// AKLT ground-state energy per site (= per bond) for H = Σ[S·S + (1/3)(S·S)²].
pub const AKLT_GROUND_ENERGY_PER_SITE: f64 = -2.0 / 3.0;

/// AKLT infinite-temperature energy per site (= per bond) = Tr[h]/9 = 4/9.
/// The biquadratic term (1/3)(S·S)² is NOT traceless (Tr[(S·S)²] = 12), unlike the
/// bilinear S·S (Tr = 0), so u(T=∞) = 4/9, not 0.
pub const AKLT_INFINITE_T_ENERGY_PER_SITE: f64 = 4.0 / 9.0;

/// Ground-state energy per site for H = Σ P₂(i,i+1).
pub const AKLT_PROJECTOR_GROUND_ENERGY_PER_SITE: f64 = 0.0;

/// Infinite-temperature energy per site for H = Σ P₂(i,i+1), equal to Tr[P₂]/9.
pub const AKLT_PROJECTOR_INFINITE_T_ENERGY_PER_SITE: f64 = 5.0 / 9.0;

/// Spin-1 infinite-temperature entropy per site = ln 3 (β f → −ln 3 as β → 0).
pub fn aklt_infinite_t_entropy() -> f64 { (3.0_f64).ln() }

#[cfg(test)]
mod aklt_checkpoints {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn checkpoints_have_expected_values() {
        assert_abs_diff_eq!(AKLT_GROUND_ENERGY_PER_SITE, -0.666_666_666_7, epsilon = 1e-9);
        assert_abs_diff_eq!(AKLT_INFINITE_T_ENERGY_PER_SITE, 0.444_444_444_4, epsilon = 1e-9);
        assert_abs_diff_eq!(aklt_infinite_t_entropy(), 1.098_612_288_7, epsilon = 1e-9);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn xy_dispersion_reduces_to_tfim_at_gamma_one() {
        // γ=1: ε_k = 2 sqrt((cos k - h)² + sin² k) = TFIM dispersion(1, h, k)
        for &k in &[0.3, 1.1, 2.7] {
            assert_abs_diff_eq!(xy_dispersion(1.0, 0.7, k), dispersion(1.0, 0.7, k), epsilon = 1e-12);
        }
    }
    #[test]
    fn xy_free_energy_matches_tfim_at_gamma_one() {
        let beta = 1.3;
        assert_abs_diff_eq!(
            xy_free_energy_density(1.0, 0.7, beta, 4000),
            free_energy_density(1.0, 0.7, beta, 4000),
            epsilon = 1e-9
        );
    }
    #[test]
    fn xy_magnetization_z_positive_in_field() {
        // 横磁場 h>0 で <σz> > 0
        let mz = xy_magnetization_z(0.5, 0.8, 5.0, 4000);
        assert!(mz > 0.0, "mz={mz}");
    }

    #[test]
    fn exact_magnetization_x_infinite_temperature_is_zero() {
        // β→0 で <σx> → 0
        let mx = exact_magnetization_x(1.0, 1.0, 1e-4, 2000);
        assert_abs_diff_eq!(mx, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn exact_magnetization_x_is_positive_at_low_t() {
        // 低温では横磁場方向に分極して <σx> > 0（g>0）
        let mx = exact_magnetization_x(1.0, 1.0, 20.0, 4000);
        assert!(mx > 0.5, "expected sizeable transverse magnetization, got {mx}");
    }

    #[test]
    fn infinite_temperature_limit_is_zero() {
        // β→0 gives u→0
        let u = exact_energy_density(1.0, 0.5, 1e-4, 2000);
        assert_abs_diff_eq!(u, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn ground_state_energy_matches_known() {
        // β→∞ gives the ground-state energy density
        // At the critical point J=g=1, the exact value is -4/π ≈ -1.2732
        // (for H = -Σ σzσz - Σ σx normalization)
        let u = exact_energy_density(1.0, 1.0, 50.0, 4000);
        // numerical residual at β=50, nk=4000 is ~5e-5, so 1e-3 is a discriminating bound
        assert_abs_diff_eq!(u, -4.0 / std::f64::consts::PI, epsilon = 1e-3);
    }

    #[test]
    fn exact_specific_heat_vanishes_at_high_t() {
        // β→0 で C→0
        let c = exact_specific_heat(1.0, 1.0, 1e-3, 4000);
        assert!(c.abs() < 1e-2, "C should vanish at high T, got {c}");
    }

    #[test]
    fn exact_specific_heat_is_positive_intermediate() {
        // 中間温度で C > 0
        let c = exact_specific_heat(1.0, 1.0, 1.0, 4000);
        assert!(c > 0.0, "C should be positive at intermediate T, got {c}");
    }
}
