/// Integration test: iMPS purification energy density vs exact free-fermion reference (TFIM).
///
/// Physics derivation of β = 2τ:
///   Purification state: |ψ(τ)⟩ = e^{-τH} |I⟩
///   where |I⟩ = Σ_s |s⟩_phys ⊗ |s⟩_anc / √d is the maximally entangled state.
///   The thermal density matrix: ρ(β) ∝ Tr_anc[|ψ(τ)⟩⟨ψ(τ)|] = e^{-τH} · ρ_I · e^{-τH} ∝ e^{-βH}
///   so β = 2τ (bra and ket each contribute one factor e^{-τH}).
///
///   Each call to `imaginary_time_step(dtau)` advances τ by dtau (applies e^{-dtau H} once).
///   After n steps: τ = n·dtau, β = 2·n·dtau = 2τ.
///
/// Confirmed empirically: comparing u_imps vs u_exact(β=2τ) gives |diff| ~ 3×10⁻⁴,
/// while u_exact(β=τ) gives |diff| ~ 0.1 — decisively confirming β=2τ.

use thermal_imps_purification::purified_mps::infinite_temperature;
use thermal_imps_purification::itebd::imaginary_time_step;
use thermal_imps_purification::observable::energy_density;
use thermal_imps_purification::exact::exact_energy_density;
use thermal_imps_purification::model::Tfim;
use thermal_imps_purification::tensor::Truncation;

#[test]
fn imps_matches_exact_energy_density() {
    let m = Tfim { j: 1.0, g: 1.0 };
    let ham = m.local();
    let dtau = 0.01;
    let trunc = Truncation { epsilon: 1e-10, max_bond: Some(32) };
    let mut state = infinite_temperature(2);

    // Tolerance justification:
    //   - Trotter error per step: O(dtau²) for 2-site exact gates → O(dtau) accumulated.
    //     At dtau=0.01 over ~100 steps (β=2): accumulated Trotter error ~ 1e-3.
    //   - Truncation error at chi=32 for critical TFIM: ~ 1e-4.
    //   - Empirically measured diff_2tau ~ 3-4×10⁻⁴ across β∈[0.1,4.0].
    //   - Tolerance 5e-3 gives ≥10× safety margin over the actual ~4×10⁻⁴ error.
    let tol = 5e-3;

    // Checkpoints chosen as multiples of 2*dtau = 0.02 to hit exactly:
    //   β=0.26 → step 13,  β=0.50 → step 25,  β=1.00 → step 50,  β=2.00 → step 100
    // β = 2 * n * dtau: n = β / (2*dtau)
    let beta_targets = [0.26_f64, 0.50, 1.00, 2.00];
    let step_targets: Vec<usize> = beta_targets.iter()
        .map(|&b| (b / (2.0 * dtau)).round() as usize)
        .collect();

    let nsteps = *step_targets.last().unwrap();
    let mut ci = 0;

    for step in 1..=nsteps {
        imaginary_time_step(&mut state, &ham, dtau, &trunc);
        let tau = step as f64 * dtau;
        let beta = 2.0 * tau; // β = 2τ from purification convention

        if ci < step_targets.len() && step == step_targets[ci] {
            let u_imps = energy_density(&state, &ham);
            let u_exact = exact_energy_density(m.j, m.g, beta, 4000);
            let err = (u_imps - u_exact).abs();
            assert!(
                err < tol,
                "beta={:.4} (tau={:.4}, step={}): imps={:.6}, exact={:.6}, |diff|={:.2e} >= tol={:.2e}",
                beta, tau, step, u_imps, u_exact, err, tol
            );
            ci += 1;
        }
    }
    assert_eq!(ci, beta_targets.len(), "Not all checkpoints were reached");
}

#[test]
fn imps_matches_exact_offcritical() {
    // Disordered phase: g=2.0 > J=1.0. Single checkpoint at β=1.0 (step 50).
    // Exercises the inverse-λ path with a different λ-spectrum than the critical point.
    let m = Tfim { j: 1.0, g: 2.0 };
    let ham = m.local();
    let dtau = 0.01;
    let trunc = Truncation { epsilon: 1e-10, max_bond: Some(32) };
    let tol = 5e-3;

    let mut state = infinite_temperature(2);

    // β=1.0 → τ=0.5 → step = β/(2*dtau) = 1.0/(2*0.01) = 50
    let target_step: usize = (1.0_f64 / (2.0 * dtau)).round() as usize;

    for _step in 1..=target_step {
        imaginary_time_step(&mut state, &ham, dtau, &trunc);
    }

    let tau = target_step as f64 * dtau;
    let beta = 2.0 * tau; // β = 2τ from purification convention
    let u_imps = energy_density(&state, &ham);
    let u_exact = exact_energy_density(m.j, m.g, beta, 4000);
    let err = (u_imps - u_exact).abs();
    println!("offcritical (g=2.0, beta={:.4}): u_imps={:.6}, u_exact={:.6}, |diff|={:.2e}", beta, u_imps, u_exact, err);
    assert!(
        err < tol,
        "offcritical g=2.0, beta={:.4} (tau={:.4}, step={}): imps={:.6}, exact={:.6}, |diff|={:.2e} >= tol={:.2e}",
        beta, tau, target_step, u_imps, u_exact, err, tol
    );
}
