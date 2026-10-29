use serde_json::{json, Value};
use thermal_imps_purification::aklt_reference::{
    resolve_request,
    v3::{
        capture_runtime, measure_v3_reference, preflight_v3, resolve_v3_request,
        AkltReferenceRequestV3,
    },
    AkltReferenceRequest,
};
use thermal_imps_purification::itebd_auto::ItebdState;

fn request_value() -> Value {
    json!({"schema_version":3,"contract":"aklt-projector-reference-v3",
        "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,
        "max_seconds":null,"recording_policy":"target-snapshots-v1",
        "heat_capacity_policy":"variance-default-v1",
        "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})
}
fn request_v3() -> AkltReferenceRequestV3 {
    serde_json::from_value(request_value()).unwrap()
}

// Wrong schedule/cap dispatch would invalidate the scientific comparison.
#[test]
fn v3_reuses_exact_fine_grid() {
    let legacy: AkltReferenceRequest = serde_json::from_value(json!({
        "schema_version":2,"contract":"aklt-projector-reference-v2",
        "model":{"name":"aklt_projector"},"variant":"fine","max_seconds":null,
        "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)}))
    .unwrap();
    let old = resolve_request(&legacy).unwrap();
    let new = resolve_v3_request(&request_v3()).unwrap();
    assert_eq!(new.truncation().max_bond, Some(256));
    assert_eq!(new.truncation().epsilon, 1e-13);
    assert_eq!(new.segments(), old.segments());
    assert_eq!(new.segments().iter().map(|s| s.steps).sum::<u64>(), 1014);
    assert_eq!(new.targets().len(), 20);
    let p = preflight_v3(&request_v3()).unwrap();
    assert_eq!(p["schema_version"], 3);
    assert_eq!(p["max_bond"], 256);
    assert!(p["max_seconds"].is_null());
    assert_eq!(
        p["heat_capacity"],
        json!({"kind":"energy_variance",
        "max_distance":200,"relative_tolerance":1e-6,"absolute_tolerance":1e-12,
        "consecutive_small_shells":3,"reality_tolerance":1e-10})
    );
    assert_eq!(
        p["snapshot_policy"],
        json!({"kind":"target_snapshots","file":"states.h5","maximum_count":20})
    );
}

// Unsupported controls must not silently select a different experiment.
#[test]
fn v3_rejects_control_drift() {
    let mut cases = Vec::new();
    for (key, value) in [
        ("max_seconds", json!(3600)),
        ("max_seconds", json!("null")),
        ("max_bond", json!(128)),
        ("max_bond", json!(512)),
        ("variant", json!("time-step")),
        ("schema_version", json!(2)),
        ("recording_policy", json!("every-step")),
        ("heat_capacity_policy", json!("difference")),
        ("extra", json!(1)),
        ("model", json!({"name":"aklt"})),
        ("source_revision", json!("bad")),
        ("worker_sha256", json!("g".repeat(64))),
    ] {
        let mut v = request_value();
        v[key] = value;
        cases.push(v);
    }
    let mut missing = request_value();
    missing.as_object_mut().unwrap().remove("max_seconds");
    cases.push(missing);
    for v in cases {
        if let Ok(r) = serde_json::from_value::<AkltReferenceRequestV3>(v.clone()) {
            assert!(resolve_v3_request(&r).is_err(), "accepted {v}");
        }
    }
}

// Independent projector traces and product-state correlations fix var/site=20/81.
#[test]
fn v3_variance_uses_beta_squared() {
    let settings = resolve_v3_request(&request_v3()).unwrap();
    let state = ItebdState::infinite_temperature(settings.hamiltonian()).unwrap();
    let v = measure_v3_reference(&state, settings.hamiltonian(), 0.5, 0.).unwrap();
    assert!((v.physical.energy - 5.0 / 9.0).abs() < 1e-12);
    assert!(v.heat.available);
    assert!(v.heat.reason.is_none());
    let h = v.heat.report.unwrap();
    assert!((h.energy_variance_per_site - 20.0 / 81.0).abs() < 1e-12);
    assert!((h.specific_heat_per_site - 0.25 * 20.0 / 81.0).abs() < 1e-12);
    assert_eq!(h.stop_reason, "consecutive_small_shells");
}

#[test]
fn runtime_evidence_matches_native_context() {
    let evidence = capture_runtime().unwrap();
    let backend = tenferro_cpu::CpuBackend::new();
    assert_eq!(evidence.thread_hint, backend.num_threads());
    assert_eq!(evidence.provider, format!("{:?}", backend.kind()));
    assert_eq!(
        evidence.visible_cpu_count,
        tenferro_cpu::available_parallelism()
    );
}

// A classical mixture of two constant-energy sectors has a persistent energy tail.
#[test]
fn variance_tail_failure_keeps_independent_physical_values() {
    use thermal_imps_purification::purified_mps::{PurifiedMps, Site};
    use thermal_imps_purification::tensor::{from_fn, new_index, Idx};
    let ab = new_index(2);
    let ba = new_index(2);
    let site = |left: &Idx, right: &Idx| {
        let phys = new_index(3);
        let anc = new_index(3);
        let gamma = from_fn(
            &[left.clone(), phys.clone(), anc.clone(), right.clone()],
            |ix| {
                if ix[0] == ix[3] && ix[1] == ix[0] && ix[2] == ix[0] {
                    2f64.sqrt()
                } else {
                    0.
                }
            },
        );
        Site {
            gamma,
            left: left.clone(),
            phys,
            anc,
            right: right.clone(),
        }
    };
    let state = ItebdState::Real(PurifiedMps {
        a: site(&ba, &ab),
        b: site(&ab, &ba),
        lambda_ab: vec![1. / 2f64.sqrt(); 2],
        lambda_bond_ab: ab,
        lambda_ba: vec![1. / 2f64.sqrt(); 2],
        lambda_bond_ba: ba,
    });
    let settings = resolve_v3_request(&request_v3()).unwrap();
    let v = measure_v3_reference(&state, settings.hamiltonian(), 0.5, 0.).unwrap();
    assert!((v.physical.energy - 5. / 6.).abs() < 1e-12);
    assert!(!v.heat.available);
    assert!(v.heat.report.is_none());
    assert!(v.heat.reason.as_deref().unwrap().contains("tail"));
}
