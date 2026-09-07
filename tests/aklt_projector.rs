#[path = "support/aklt_projector_oracle.rs"]
mod oracle;

use thermal_imps_purification::exact::{
    AKLT_PROJECTOR_GROUND_ENERGY_PER_SITE, AKLT_PROJECTOR_INFINITE_T_ENERGY_PER_SITE,
};
use thermal_imps_purification::model::{sz1, trotter_gate, AkltProjector, BilinearBiquadratic};
use thermal_imps_purification::observable::{energy_density, magnetization};
use thermal_imps_purification::purified_mps::infinite_temperature;
use nalgebra::{DMatrix, SymmetricEigen};

#[test]
fn projector_matches_independent_spin_two_subspace() {
    let h = AkltProjector.local();
    assert_eq!(h.dim(), 3);
    assert_eq!(h.two_site_h, h.site_energy);
    assert!((&h.two_site_h - oracle::projector_from_cg()).norm() <= 1e-12);
    assert!((&h.two_site_h * &h.two_site_h - &h.two_site_h).norm() <= 1e-12);
    assert!((&h.two_site_h - h.two_site_h.transpose()).norm() <= 1e-12);
    assert!((h.two_site_h.trace() - 5.0).abs() <= 1e-12);
}

#[test]
fn projector_spectrum_has_four_zeros_and_five_ones() {
    let h = AkltProjector.local().two_site_h;
    let mut eigenvalues: Vec<f64> = SymmetricEigen::new(h).eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(f64::total_cmp);

    assert!(eigenvalues.iter().all(|&eigenvalue| eigenvalue >= -1e-12));
    assert!(eigenvalues[..4]
        .iter()
        .all(|eigenvalue| eigenvalue.abs() <= 1e-12));
    assert!(eigenvalues[4..]
        .iter()
        .all(|eigenvalue| (eigenvalue - 1.0).abs() <= 1e-12));
}

#[test]
fn projector_has_the_expected_relation_to_the_existing_aklt_model() {
    let old_h = BilinearBiquadratic::aklt().local().two_site_h;
    let projector = AkltProjector.local().two_site_h;
    let expected = 2.0 * projector - (2.0 / 3.0) * DMatrix::<f64>::identity(9, 9);

    assert!((old_h - expected).norm() <= 1e-12);
}

#[test]
fn projector_gate_matches_the_analytic_exponential() {
    let h = AkltProjector.local().two_site_h;
    let projector = oracle::projector_from_cg();
    let identity = DMatrix::<f64>::identity(9, 9);

    for tau in [0.0_f64, 0.025, 0.1] {
        let expected = &identity + ((-tau).exp() - 1.0) * &projector;
        assert!((trotter_gate(&h, tau) - expected).norm() <= 1e-12);
    }
}

#[test]
fn projector_exact_endpoints_and_infinite_temperature_observables_agree() {
    assert_eq!(AKLT_PROJECTOR_GROUND_ENERGY_PER_SITE, 0.0);
    assert_eq!(AKLT_PROJECTOR_INFINITE_T_ENERGY_PER_SITE, 5.0 / 9.0);

    let state = infinite_temperature(3);
    let h = AkltProjector.local();
    assert!(
        (energy_density(&state, &h) - AKLT_PROJECTOR_INFINITE_T_ENERGY_PER_SITE).abs() <= 1e-12
    );
    assert!(magnetization(&state, &sz1()).abs() <= 1e-12);
}

#[test]
fn periodic_four_site_ground_vector_has_zero_production_energy() {
    let oracle_ring = oracle::periodic_four_site_h(&oracle::projector_from_cg());
    let eig = SymmetricEigen::new(oracle_ring.clone());
    let ground_index = (0..eig.eigenvalues.len())
        .min_by(|&left, &right| eig.eigenvalues[left].total_cmp(&eig.eigenvalues[right]))
        .unwrap();
    let ground_energy = eig.eigenvalues[ground_index];
    let ground = eig.eigenvectors.column(ground_index).into_owned();

    assert!(ground_energy.abs() <= 1e-12);
    assert!((&oracle_ring * &ground - ground_energy * &ground).norm() <= 1e-12);

    let production_ring = oracle::periodic_four_site_h(&AkltProjector.local().two_site_h);
    let production_ground = &production_ring * &ground;
    assert!((ground.dot(&production_ground) / 4.0).abs() <= 1e-12);
    assert!(production_ground.norm() <= 1e-12);
}
