use thermal_imps_purification::ising_reference::*;
use thermal_imps_purification::itebd_auto::ItebdState;
fn request(variant: &str) -> ReferenceRequest {
    serde_json::from_value(serde_json::json!({"schema_version":1,"model":{"name":"ising_mixed","g":1.05,"h":0.5},"variant":variant,"source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})).unwrap()
}
#[test]
fn reference_schedule_has_nested_exact_targets() {
    let coarse = resolve_request(&request("time-step")).expect("checked coarse settings");
    let fine = resolve_request(&request("fine")).unwrap();
    let bond = resolve_request(&request("bond-cap")).unwrap();
    assert_eq!(coarse.segments().iter().map(|s| s.steps).sum::<u64>(), 507);
    assert_eq!(fine.segments().iter().map(|s| s.steps).sum::<u64>(), 1014);
    assert_eq!(bond.segments().iter().map(|s| s.steps).sum::<u64>(), 1014);
    assert_eq!(
        fine.targets()
            .iter()
            .map(|t| t.temperature_tick)
            .collect::<Vec<_>>(),
        (1u32..=20).rev().collect::<Vec<_>>()
    );
    assert_eq!(fine.truncation().max_bond, Some(128));
    assert_eq!(bond.truncation().max_bond, Some(64));
    assert_eq!(fine.truncation().epsilon, 1e-13);
    let mut beta = 0.;
    for ((c, f), target) in coarse
        .segments()
        .iter()
        .zip(fine.segments())
        .zip(fine.targets())
    {
        assert_eq!(f.steps, 2 * c.steps);
        assert_eq!(f.dtau * 2., c.dtau);
        beta += 2. * f.steps as f64 * f.dtau;
        assert!((beta - target.beta).abs() < 1e-13);
        assert_eq!(target.beta, 10. / target.temperature_tick as f64);
    }
    let pre = preflight(&request("fine")).unwrap();
    assert_eq!(pre["total_steps"], 1014);
    assert_eq!(pre["basis_order"], "first_site_fastest");
}
#[test]
fn reference_strict_request_rejects_unsupported_conditions() {
    let original = serde_json::to_value(request("fine")).unwrap();
    for (key, value) in [
        ("extra", serde_json::json!(1)),
        ("variant", serde_json::json!("unknown")),
    ] {
        let mut v = original.clone();
        v[key] = value;
        assert!(serde_json::from_value::<ReferenceRequest>(v).is_err());
    }
    for (field, value) in [
        ("g", serde_json::json!(1.04)),
        ("h", serde_json::json!(0.)),
        ("name", serde_json::json!("tfim")),
    ] {
        let mut v = original.clone();
        v["model"][field] = value;
        assert!(resolve_request(&serde_json::from_value(v).unwrap()).is_err());
    }
    let mut req = request("fine");
    req.schema_version = 2;
    assert!(resolve_request(&req).is_err());
    req = request("fine");
    req.worker_sha256 = "not-a-digest".into();
    assert!(resolve_request(&req).is_err());
}
#[test]
fn reference_mixed_matrix_and_rdm_energy_identity() {
    let settings = resolve_request(&request("fine")).unwrap();
    let thermal_imps_purification::itebd_auto::ItebdHamiltonian::Real(h) = settings.hamiltonian()
    else {
        panic!("real")
    };
    let expected = nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            -1.5, -0.525, -0.525, 0., -0.525, 1., 0., -0.525, -0.525, 0., 1., -0.525, 0., -0.525,
            -0.525, -0.5,
        ],
    );
    assert!((&h.two_site_h - expected).norm() < 1e-15);
    let mut state = ItebdState::infinite_temperature(settings.hamiltonian()).unwrap();
    let step = thermal_imps_purification::itebd_auto::imaginary_time_step_second_order_auto(
        &mut state,
        settings.hamiltonian(),
        0.01,
        &thermal_imps_purification::tensor::Truncation {
            epsilon: 1e-13,
            max_bond: Some(4),
        },
    )
    .unwrap();
    let cn = thermal_imps_purification::itebd_auto::canonicalize_auto(
        &mut state,
        settings.hamiltonian(),
    )
    .unwrap();
    let v = measure_reference(&state, settings.hamiltonian(), 0.02, step.log_norm + cn).unwrap();
    // Independent scalar reference contraction checks the reported parity average/difference.
    use thermal_imps_purification::itebd_rdm::{
        reduced_density_matrix_auto, AutoRdmResult, RdmOptions, RdmParity,
    };
    let mut energies = Vec::new();
    for parity in [RdmParity::A, RdmParity::B] {
        let AutoRdmResult::Real(rdm) =
            reduced_density_matrix_auto(&state, parity, 2, &RdmOptions::default()).unwrap()
        else {
            panic!("real RDM")
        };
        let mut e = 0.;
        for row in 0..4 {
            for col in 0..4 {
                e += rdm.density_matrix[(row, col)] * h.two_site_h[(col, row)];
            }
        }
        energies.push(e);
    }
    assert!((v.energy - (energies[0] + energies[1]) / 2.).abs() < 1e-14);
    assert!((v.parity_delta_energy - (energies[0] - energies[1])).abs() < 1e-14);
    assert!((v.energy + v.zz + 1.05 * v.mx + 0.5 * v.mz).abs() < 1e-9);
    assert!(v.energy < -0.01);
    assert!(v.direct_energy_difference.abs() < 1e-9);
    assert!(v.parity_delta_energy.abs() < 1e-5);
    assert!((v.entropy - (v.energy - v.free_energy) * 0.02).abs() < 1e-14);
}
