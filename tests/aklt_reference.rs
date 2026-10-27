use serde_json::json;
use thermal_imps_purification::{
    aklt_reference::*,
    itebd_auto::{ItebdHamiltonian, ItebdState},
    model::{sx1, sz1},
};
// Reuse the CG projector oracle; its finite-ring helper is outside this test scope.
#[allow(dead_code)]
#[path = "support/aklt_projector_oracle.rs"]
mod oracle;
fn request(variant: &str) -> AkltReferenceRequest {
    serde_json::from_value(
        json!({"schema_version":1,"contract":"aklt-projector-reference-v1",
        "model":{"name":"aklt_projector"},"variant":variant,"max_seconds":3600.,
        "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)}),
    )
    .unwrap()
}
#[test]
fn aklt_reference_nested_schedule_and_preflight() {
    let coarse = resolve_request(&request("time-step")).unwrap();
    let fine = resolve_request(&request("fine")).unwrap();
    assert_eq!(coarse.segments().iter().map(|s| s.steps).sum::<u64>(), 507);
    assert_eq!(fine.segments().iter().map(|s| s.steps).sum::<u64>(), 1014);
    assert_eq!(fine.truncation().max_bond, Some(128));
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
    let p = preflight(&request("fine")).unwrap();
    for (key, value) in [
        ("contract", json!("aklt-projector-reference-v1")),
        ("physical_dimension", json!(3)),
        ("cell_sites", json!(2)),
        ("backend", json!("real")),
        ("basis_order", json!("first_site_fastest")),
        ("temperature_tick_scale", json!(10)),
        ("initial_beta", json!(0.)),
        ("max_dtau", json!(0.005)),
        ("max_bond", json!(128)),
        ("svd_cutoff", json!(1e-13)),
        ("total_steps", json!(1014)),
        ("trotter_order", json!("second")),
        ("canonicalize_every", json!(1)),
        ("max_seconds", json!(3600.)),
    ] {
        assert_eq!(p[key], value, "{key}");
    }
    assert_eq!(
        p["canonicalizer"],
        json!({"tolerance":1e-13,"max_iterations":4000,"support_floor":1e-12})
    );
    assert_eq!(
        p["rdm"],
        json!({"fixed_point_tolerance":1e-12,"max_iterations":10000,"trace_tolerance":1e-10,
        "hermiticity_tolerance":1e-10,"positivity_tolerance":1e-10,"max_output_elements":1048576,"max_intermediate_elements":16777216})
    );
    let h = oracle::projector_from_cg();
    for i in 0..9 {
        for j in 0..9 {
            assert!((p["two_site_h"][i][j].as_f64().unwrap() - h[(i, j)]).abs() < 1e-14);
        }
    }
    let mut v2 = serde_json::to_value(request("fine")).unwrap();
    v2["schema_version"] = json!(2);
    v2["contract"] = json!("aklt-projector-reference-v2");
    v2["max_seconds"] = serde_json::Value::Null;
    let r2: AkltReferenceRequest = serde_json::from_value(v2).unwrap();
    let mut unlimited = preflight(&r2).unwrap();
    assert_eq!(unlimited["schema_version"], 2);
    assert!(unlimited["max_seconds"].is_null());
    for key in ["schema_version", "contract", "request", "max_seconds"] {
        unlimited[key] = p[key].clone();
    }
    assert_eq!(unlimited, p);
    assert_eq!(p["site_energy"], p["two_site_h"]);
    assert!(p["log_norm_convention"].as_str().unwrap().contains("ln(3)"));
}
#[test]
fn aklt_reference_rejects_nonstudy_requests() {
    let original = serde_json::to_value(request("fine")).unwrap();
    for (key, value) in [
        ("extra", json!(1)),
        ("variant", json!("bond-cap")),
        ("model", json!({"name":"aklt_projector","g":1.})),
        ("model", json!({"name":"aklt"})),
        ("max_seconds", json!(3601.)),
        ("source_revision", json!("bad")),
        ("worker_sha256", json!("g".repeat(64))),
        ("schema_version", json!(2)),
        ("contract", json!("other")),
    ] {
        let mut v = original.clone();
        v[key] = value;
        if let Ok(r) = serde_json::from_value::<AkltReferenceRequest>(v) {
            assert!(resolve_request(&r).is_err(), "{key}");
        }
    }
}
#[test]
fn aklt_reference_projector_and_spin_one_values() {
    let settings = resolve_request(&request("fine")).unwrap();
    let ItebdHamiltonian::Real(h) = settings.hamiltonian() else {
        panic!("real model")
    };
    let projector = oracle::projector_from_cg();
    assert!((&h.two_site_h - &projector).norm() < 1e-12);
    assert!((&h.two_site_h * &h.two_site_h - &h.two_site_h).norm() < 1e-12);
    assert!((h.two_site_h.trace() - 5.).abs() < 1e-12);
    assert!(((sx1() * sx1()).trace() - 2.).abs() < 1e-12);
    assert_eq!((sz1() * sz1()).trace(), 2.);
    let mut state = ItebdState::infinite_temperature(settings.hamiltonian()).unwrap();
    let infinite = measure_reference(&state, settings.hamiltonian(), 1e-8, 0.).unwrap();
    assert!((infinite.energy - 5. / 9.).abs() < 1e-12);
    assert!(infinite.mx.abs() < 1e-12 && infinite.mz.abs() < 1e-12);
    assert!(!infinite.cap_reached_now);
    let step = thermal_imps_purification::itebd_auto::imaginary_time_step_second_order_auto(
        &mut state,
        settings.hamiltonian(),
        0.001,
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
    let v = measure_reference(&state, settings.hamiltonian(), 0.002, step.log_norm + cn).unwrap();
    use thermal_imps_purification::itebd_rdm::{
        reduced_density_matrix_auto, AutoRdmResult, RdmOptions, RdmParity,
    };
    let mut energies = Vec::new();
    for parity in [RdmParity::A, RdmParity::B] {
        let AutoRdmResult::Real(r) =
            reduced_density_matrix_auto(&state, parity, 2, &RdmOptions::default()).unwrap()
        else {
            panic!("real RDM")
        };
        let mut e = 0.;
        for i in 0..9 {
            for j in 0..9 {
                e += r.density_matrix[(i, j)] * projector[(j, i)];
            }
        }
        energies.push(e);
    }
    assert!((v.energy - (energies[0] + energies[1]) / 2.).abs() < 1e-12);
    assert!((v.parity_delta_energy - (energies[0] - energies[1])).abs() < 1e-12);
    assert!(v.direct_energy_difference.abs() < 1e-9);
}
#[test]
fn aklt_reference_norm_ledger_uses_two_sites() {
    let settings = resolve_request(&request("fine")).unwrap();
    let state = ItebdState::infinite_temperature(settings.hamiltonian()).unwrap();
    let beta = 0.4;
    let norm = 0.7;
    let v = measure_reference(&state, settings.hamiltonian(), beta, norm).unwrap();
    assert!((v.free_energy + (3f64.ln() + norm / 2.) / beta).abs() < 1e-12);
    assert!((v.entropy - beta * (v.energy - v.free_energy)).abs() < 1e-12);
    let tiny = measure_reference(&state, settings.hamiltonian(), 1e-8, 0.).unwrap();
    assert!((1e-8 * tiny.free_energy + 3f64.ln()).abs() < 1e-12);
    for (b, n) in [(0., 0.), (-1., 0.), (1., f64::NAN)] {
        assert!(measure_reference(&state, settings.hamiltonian(), b, n).is_err());
    }
}

#[test]
fn aklt_v2_requires_explicit_null_deadline() {
    let mut base = serde_json::to_value(request("fine")).unwrap();
    base["schema_version"] = json!(2);
    base["contract"] = json!("aklt-projector-reference-v2");
    base["max_seconds"] = serde_json::Value::Null;
    resolve_request(&serde_json::from_value::<AkltReferenceRequest>(base.clone()).unwrap())
        .unwrap();
    let mut missing = base.clone();
    missing.as_object_mut().unwrap().remove("max_seconds");
    let error = serde_json::from_value::<AkltReferenceRequest>(missing).unwrap_err();
    assert!(error.to_string().contains("max_seconds"), "{error}");
    for (key, value) in [
        ("max_seconds", json!(0.)),
        ("max_seconds", json!(3600.)),
        ("max_seconds", json!("null")),
        ("schema_version", json!(1)),
        ("contract", json!("aklt-projector-reference-v1")),
    ] {
        let mut bad = base.clone();
        bad[key] = value;
        if let Ok(request) = serde_json::from_value::<AkltReferenceRequest>(bad) {
            assert!(resolve_request(&request).is_err(), "{key}");
        }
    }
}
