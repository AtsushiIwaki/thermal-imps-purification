#[path = "support/itebd_checkpoint_rdm.rs"]
#[allow(dead_code)]
mod checkpoint_support;
#[path = "support/solve_matrix.rs"]
mod support;

use thermal_imps_purification::itebd::free_energy_from_log_norm;
use thermal_imps_purification::itebd_auto::energy_density_auto;
use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions,
    ItebdTrajectoryWriter, LoadedItebdCheckpoint,
};
use thermal_imps_purification::itebd_rdm::RdmParity;
use thermal_imps_purification::itebd_state_view::ItebdStateRef;
use thermal_imps_purification::runner::{read_result, Record};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Default)]
struct Maxima(BTreeMap<&'static str, (f64, f64)>);
impl Maxima {
    fn close(&mut self, field: &'static str, a: f64, b: f64, label: &str) {
        self.difference(field, (a - b).abs(), a.abs().max(b.abs()).max(1.0), label);
    }
    fn difference(&mut self, field: &'static str, absolute: f64, scale: f64, label: &str) {
        let scaled = absolute / scale;
        let max = self.0.entry(field).or_default();
        max.0 = max.0.max(absolute);
        max.1 = max.1.max(scaled);
        assert!(
            scaled <= 1e-10,
            "{label} {field}: absolute={absolute:e} scaled={scaled:e}"
        );
    }
    fn records(&mut self, a: &Record, b: &Record, label: &str) {
        assert_eq!(a.beta.to_bits(), b.beta.to_bits(), "{label} beta");
        assert_eq!(a.max_bond, b.max_bond, "{label} max bond");
        for (field, a, b) in [
            ("u", a.u, b.u),
            ("c", a.c, b.c),
            ("f", a.f, b.f),
            ("local", a.magnetization, b.magnetization),
        ] {
            self.close(field, a, b, label);
        }
    }
}

#[test]
fn matrix_cli_continuation() {
    // Catches resetting global schedules/progress, mutating checkpoint states, changing storage,
    // or omitting the per-site factor of two when resuming a complex matrix Hamiltonian.
    for format in ["toml", "json"] {
        for order in [1, 2] {
            for canon in [1, 3] {
                exercise(format, format, order, canon);
            }
        }
    }
    exercise("toml", "json", 2, 3);
}

fn exercise(format: &str, restart_format: &str, order: usize, canon: usize) {
    let dir = tempfile::tempdir().unwrap();
    let label = format!("{format}->{restart_format}-order{order}-canon{canon}");
    let stride = if canon == 1 { 3 } else { 6 };
    for (name, steps, checkpoint, source) in [
        ("whole", 10, true, None),
        ("split", 5, true, None),
        ("restart", 10, true, Some(dir.path().join("split.h5"))),
        ("plain", 10, false, None),
    ] {
        let mut cfg = support::phase_config();
        cfg["evolution"] = json!({"dtau":0.01,"beta_max":0.02 * steps as f64,
            "record_every_beta":0.02 * stride as f64,"trotter_order":order});
        cfg["run"]["canonicalize_every"] = json!(canon);
        cfg["output"]["path"] = json!(dir.path().join(format!("{name}.result.json")));
        if checkpoint {
            cfg["checkpoint"] =
                json!({"path":dir.path().join(format!("{name}.h5")),"every_steps":4});
        }
        if let Some(source) = source {
            cfg["restart"] = json!({"path":source});
        }
        let fmt = if name == "restart" {
            restart_format
        } else {
            format
        };
        let path = dir.path().join(format!("{name}.{fmt}"));
        support::write_input(&cfg, &path, fmt);
        support::run_cli(&path);
    }
    for (name, expected) in [
        ("whole", vec![0, 4, 8, 10]),
        ("split", vec![0, 4, 5]),
        ("restart", vec![5, 8, 10]),
    ] {
        assert_eq!(
            list_itebd_checkpoints(&dir.path().join(format!("{name}.h5")))
                .unwrap()
                .iter()
                .map(|entry| entry.completed_steps)
                .collect::<Vec<_>>(),
            expected,
            "{label} {name}"
        );
    }
    let read = |name| read_result(&dir.path().join(format!("{name}.result.json"))).unwrap();
    let whole = read("whole");
    let restart = read("restart");
    let plain = read("plain");
    let steps = |r: &Record| (r.beta / 0.02).round() as usize;
    let expected: Vec<_> = (stride..=10).step_by(stride).filter(|s| *s > 5).collect();
    assert_eq!(
        restart.records.iter().map(steps).collect::<Vec<_>>(),
        expected,
        "{label}"
    );
    assert_eq!(
        whole.records.iter().map(steps).collect::<Vec<_>>(),
        (stride..=10).step_by(stride).collect::<Vec<_>>()
    );
    let suffix: Vec<_> = whole.records.iter().filter(|r| steps(r) > 5).collect();
    assert_eq!(suffix.len(), restart.records.len());
    assert_eq!(whole.records.len(), plain.records.len());
    let mut maxima = Maxima::default();
    for (a, b) in suffix.into_iter().zip(&restart.records) {
        maxima.records(a, b, &label);
    }
    for (a, b) in whole.records.iter().zip(&plain.records) {
        maxima.records(a, b, &label);
    }
    let a = latest(&dir.path().join("whole.h5"));
    let b = latest(&dir.path().join("restart.h5"));
    assert!(matches!(
        b.state,
        thermal_imps_purification::itebd_auto::ItebdState::Complex(_)
    ));
    assert_eq!(a.progress.completed_steps, 10);
    assert_eq!(b.progress.completed_steps, 10);
    assert_eq!(a.progress.beta.to_bits(), b.progress.beta.to_bits());
    maxima.close(
        "log_norm",
        a.progress.accumulated_log_norm,
        b.progress.accumulated_log_norm,
        &label,
    );
    let av = ItebdStateRef::from(&a.state);
    let bv = ItebdStateRef::from(&b.state);
    for (al, bl) in [
        (av.lambda_ab(), bv.lambda_ab()),
        (av.lambda_ba(), bv.lambda_ba()),
    ] {
        assert_eq!(al.len(), bl.len());
        for (&a, &b) in al.iter().zip(bl) {
            maxima.close("spectrum", a, b, &label);
        }
    }
    maxima.close(
        "energy",
        energy_density_auto(&a.state, &a.hamiltonian).unwrap(),
        energy_density_auto(&b.state, &b.hamiltonian).unwrap(),
        &label,
    );
    let free = |s: &LoadedItebdCheckpoint| {
        free_energy_from_log_norm(s.progress.accumulated_log_norm / 2.0, s.progress.beta, 2)
    };
    maxima.close("final_free", free(&a), free(&b), &label);
    for parity in [RdmParity::A, RdmParity::B] {
        for length in [1, 2] {
            // Default RDM APIs enforce raw trace, Hermiticity and positivity before returning;
            // no symmetrization or normalization of their output is applied here.
            let ar = checkpoint_support::rdm(&a.state, parity, length);
            let br = checkpoint_support::rdm(&b.state, parity, length);
            maxima.difference(
                "rdm",
                (&ar - &br).norm(),
                ar.norm().max(br.norm()).max(1.0),
                &label,
            );
        }
    }
    // Compare IDs only across one exact serialization, never unrelated CLI processes.
    let before = checkpoint_support::snapshot(&b.state, &b.progress);
    let path = dir.path().join("roundtrip.h5");
    let mut writer =
        ItebdTrajectoryWriter::create(&path, b.metadata.clone(), &b.hamiltonian).unwrap();
    writer.append((&b.state).into(), &b.progress).unwrap();
    writer.finish().unwrap();
    assert_eq!(before, checkpoint_support::snapshot(&b.state, &b.progress));
    let loaded = latest(&path);
    assert_eq!(
        before,
        checkpoint_support::snapshot(&loaded.state, &loaded.progress)
    );
    let source = latest(&dir.path().join("split.h5"));
    let initial = load_itebd_checkpoint(
        &dir.path().join("restart.h5"),
        0,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap();
    assert_eq!(
        checkpoint_support::snapshot(&source.state, &source.progress),
        checkpoint_support::snapshot(&initial.state, &initial.progress)
    );
    for (field, (absolute, scaled)) in maxima.0 {
        println!("CONTINUATION {label} {field} absolute={absolute:.16e} scaled={scaled:.16e}");
    }
}

fn latest(path: &Path) -> LoadedItebdCheckpoint {
    let index = list_itebd_checkpoints(path).unwrap().last().unwrap().index;
    load_itebd_checkpoint(path, index, &ItebdCheckpointLoadOptions::default()).unwrap()
}
