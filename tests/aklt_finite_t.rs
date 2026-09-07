//! Finite-temperature validation for the spin-1 AKLT chain. No closed-form finite-T
//! exact reference exists, so we check endpoint limits and internal consistency.
//!
//! Two tests evolve to large β (hundreds of steps, d=3) and are `#[ignore]`d so the
//! default `cargo test` stays fast. Run them in release:
//!   cargo test --release --test aklt_finite_t -- --ignored

use thermal_imps_purification::purified_mps::{infinite_temperature, PurifiedMps};
use thermal_imps_purification::itebd::{imaginary_time_step, free_energy_from_log_norm};
use thermal_imps_purification::canonicalize::canonicalize;
use thermal_imps_purification::observable::{energy_density, magnetization};
use thermal_imps_purification::variance::specific_heat;
use thermal_imps_purification::model::{BilinearBiquadratic, sz1};
use thermal_imps_purification::exact::{aklt_infinite_t_entropy, AKLT_GROUND_ENERGY_PER_SITE, AKLT_INFINITE_T_ENERGY_PER_SITE};
use thermal_imps_purification::tensor::Truncation;

/// Evolve from infinite temperature to inverse temperature `beta` (= 2·steps·dtau),
/// canonicalizing every step. Returns the state and the accumulated per-cell log-norm.
fn evolve_to(beta: f64, dtau: f64, max_bond: usize) -> (PurifiedMps, f64) {
    let ham = BilinearBiquadratic::aklt().local();
    let trunc = Truncation { epsilon: 1e-12, max_bond: Some(max_bond) };
    let steps = (beta / (2.0 * dtau)).round() as usize;
    let mut s = infinite_temperature(3);
    let mut accum = 0.0;
    for _ in 0..steps {
        let info = imaginary_time_step(&mut s, &ham, dtau, &trunc);
        accum += info.log_norm;
        accum += canonicalize(&mut s);
    }
    (s, accum)
}

#[test]
fn infinite_temperature_checkpoints() {
    // β=0: maximally mixed ⇒ u = Tr[h]/9 = 4/9 (NOT 0; the biquadratic term is not
    // traceless), and ⟨Sz⟩ = 0 (Sz is traceless). Instant — no evolution.
    let s = infinite_temperature(3);
    let ham = BilinearBiquadratic::aklt().local();
    assert!((energy_density(&s, &ham) - AKLT_INFINITE_T_ENERGY_PER_SITE).abs() < 1e-12);
    assert!((magnetization(&s, &sz1())).abs() < 1e-12);
}

#[test]
fn high_temperature_free_energy_approaches_minus_ln3() {
    // β→0 ⇒ β f → −ln 3 (spin-1 infinite-T entropy). At small finite β the leading
    // correction is β·u(∞) = β·4/9, so β must be small enough that it is within tol.
    let dtau = 0.005;
    let beta = 0.01; // correction β·4/9 ≈ 0.0044 < 1e-2
    let (_s, accum) = evolve_to(beta, dtau, 16);
    let f = free_energy_from_log_norm(accum / 2.0, beta, 3);
    assert!((beta * f - -aklt_infinite_t_entropy()).abs() < 1e-2, "βf = {}", beta * f);
}

#[ignore = "heavy: evolves to β=10 (d=3); run in release: cargo test --release --test aklt_finite_t -- --ignored"]
#[test]
fn low_temperature_energy_approaches_ground_state() {
    // β large ⇒ u → −2/3 (finite Haldane gap ⇒ exponential convergence).
    let (s, _accum) = evolve_to(10.0, 0.02, 32);
    let ham = BilinearBiquadratic::aklt().local();
    let u = energy_density(&s, &ham);
    assert!((u - AKLT_GROUND_ENERGY_PER_SITE).abs() < 3e-2, "u = {u}");
}

#[ignore = "heavy: multiple evolutions (d=3); run in release: cargo test --release --test aklt_finite_t -- --ignored"]
#[test]
fn specific_heat_matches_numerical_beta_derivative() {
    // C(β) = β² var(H)/N must agree with −β² du/dβ at moderate β.
    let dtau: f64 = 0.01;
    let beta: f64 = 1.0;
    let max_bond = 32;
    let dn = 5usize;
    let n = (beta / (2.0 * dtau)).round() as usize;
    let u_at = |steps: usize| -> f64 {
        let ham = BilinearBiquadratic::aklt().local();
        let trunc = Truncation { epsilon: 1e-12, max_bond: Some(max_bond) };
        let mut s = infinite_temperature(3);
        for _ in 0..steps {
            imaginary_time_step(&mut s, &ham, dtau, &trunc);
            canonicalize(&mut s);
        }
        energy_density(&s, &ham)
    };
    let u_plus = u_at(n + dn);
    let u_minus = u_at(n - dn);
    let dbeta = 2.0 * (dn as f64) * dtau;
    let c_deriv = -(u_plus - u_minus) / (2.0 * dbeta) * beta * beta;

    let (s, _accum) = evolve_to(beta, dtau, max_bond);
    let ham = BilinearBiquadratic::aklt().local();
    let c_var = specific_heat(&s, &ham, beta).unwrap();
    assert!(c_var > 0.0, "C should be positive, got {c_var}");
    assert!((c_var - c_deriv).abs() < 2e-2, "variance {c_var} vs derivative {c_deriv}");
}
