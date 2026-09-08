#[path = "support/itebd_checkpoint_rdm.rs"]
#[allow(dead_code)]
mod support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use thermal_imps_purification::config::TrotterOrder;
use thermal_imps_purification::itebd::free_energy_from_log_norm;
use thermal_imps_purification::itebd_auto::{energy_density_auto, ItebdState};
use thermal_imps_purification::itebd_checkpoint::{
    list_itebd_checkpoints, load_itebd_checkpoint, ItebdCheckpointLoadOptions,
};
use thermal_imps_purification::itebd_rdm::RdmParity;
use thermal_imps_purification::runner::{read_result, Record, SweepResult};

const TOLERANCE: f64 = 1.0e-10;

#[derive(Default)]
struct Maxima {
    suffix_observable: f64,
    legacy_observable: f64,
    schmidt: f64,
    accumulated_log_norm: f64,
    energy: f64,
    free_energy: f64,
    rdm: f64,
}

#[test]
fn cli_continuation_preserves_physics_for_tfim_and_xy_both_orders_and_cadences() {
    let mut maxima = Maxima::default();
    for model in ["tfim", "xy"] {
        for order in [TrotterOrder::First, TrotterOrder::Second] {
            for canonicalize_every in [1, 3] {
                exercise_case(model, order, canonicalize_every, &mut maxima);
            }
        }
    }
    eprintln!(
        "continuation maxima: suffix_observable={:.3e} legacy_observable={:.3e} schmidt={:.3e} accumulated_log_norm={:.3e} energy={:.3e} free_energy={:.3e} rdm={:.3e}",
        maxima.suffix_observable,
        maxima.legacy_observable,
        maxima.schmidt,
        maxima.accumulated_log_norm,
        maxima.energy,
        maxima.free_energy,
        maxima.rdm,
    );
}

fn exercise_case(model: &str, order: TrotterOrder, canonicalize_every: usize, maxima: &mut Maxima) {
    let dir = tempfile::tempdir().unwrap();
    let stride = if canonicalize_every == 3 { 6 } else { 3 };
    let order_number = match order {
        TrotterOrder::First => 1,
        TrotterOrder::Second => 2,
    };
    let label = format!("{model}-o{order_number}-c{canonicalize_every}");

    let whole = write_config(
        dir.path(),
        &format!("{label}-whole"),
        model,
        order_number,
        canonicalize_every,
        stride,
        10,
        true,
        None,
    );
    success(&run(&whole));
    let split = write_config(
        dir.path(),
        &format!("{label}-split"),
        model,
        order_number,
        canonicalize_every,
        stride,
        5,
        true,
        None,
    );
    success(&run(&split));
    let split_h5 = output_path(&split, "h5");
    let resumed = write_config(
        dir.path(),
        &format!("{label}-resumed"),
        model,
        order_number,
        canonicalize_every,
        stride,
        10,
        true,
        Some(&split_h5),
    );
    success(&run(&resumed));
    let legacy = write_config(
        dir.path(),
        &format!("{label}-legacy"),
        model,
        order_number,
        canonicalize_every,
        stride,
        10,
        false,
        None,
    );
    success(&run(&legacy));

    assert_eq!(
        checkpoint_steps(&output_path(&whole, "h5")),
        vec![0, 4, 8, 10]
    );
    assert_eq!(checkpoint_steps(&split_h5), vec![0, 4, 5]);
    assert_eq!(
        checkpoint_steps(&output_path(&resumed, "h5")),
        vec![5, 8, 10]
    );

    let whole_result = read_result(&output_path(&whole, "json")).unwrap();
    let resumed_result = read_result(&output_path(&resumed, "json")).unwrap();
    let legacy_result = read_result(&output_path(&legacy, "json")).unwrap();
    let expected_steps: Vec<u64> = ((stride as u64)..=10)
        .step_by(stride)
        .filter(|step| *step > 5)
        .collect();
    assert_eq!(
        record_steps(&resumed_result),
        expected_steps,
        "{label} resumed schedule"
    );
    let whole_suffix: Vec<&Record> = whole_result
        .records
        .iter()
        .filter(|record| beta_step(record.beta) > 5)
        .collect();
    assert_eq!(whole_suffix.len(), resumed_result.records.len());
    for (whole_record, resumed_record) in whole_suffix.into_iter().zip(&resumed_result.records) {
        maxima.suffix_observable = maxima.suffix_observable.max(compare_records(
            whole_record,
            resumed_record,
            &format!("{label} suffix"),
        ));
    }
    assert_eq!(whole_result.records.len(), legacy_result.records.len());
    for (checkpointed, legacy) in whole_result.records.iter().zip(&legacy_result.records) {
        maxima.legacy_observable = maxima.legacy_observable.max(compare_records(
            checkpointed,
            legacy,
            &format!("{label} legacy"),
        ));
    }

    let whole_final = load_final(&output_path(&whole, "h5"));
    let resumed_final = load_final(&output_path(&resumed, "h5"));
    let resumed_again = load_final(&output_path(&resumed, "h5"));
    assert_eq!(
        support::snapshot(&resumed_final.state, &resumed_final.progress),
        support::snapshot(&resumed_again.state, &resumed_again.progress),
        "{label} repeated load"
    );
    assert_eq!(whole_final.progress.completed_steps, 10);
    assert_eq!(resumed_final.progress.completed_steps, 10);
    assert_eq!(
        whole_final.progress.beta.to_bits(),
        resumed_final.progress.beta.to_bits()
    );
    maxima.accumulated_log_norm = maxima.accumulated_log_norm.max(assert_close(
        whole_final.progress.accumulated_log_norm,
        resumed_final.progress.accumulated_log_norm,
        &format!("{label} accumulated log norm"),
    ));
    maxima.schmidt = maxima.schmidt.max(compare_schmidt(
        &whole_final.state,
        &resumed_final.state,
        &label,
    ));
    let whole_energy = energy_density_auto(&whole_final.state, &whole_final.hamiltonian).unwrap();
    let resumed_energy =
        energy_density_auto(&resumed_final.state, &resumed_final.hamiltonian).unwrap();
    maxima.energy = maxima.energy.max(assert_close(
        whole_energy,
        resumed_energy,
        &format!("{label} final energy"),
    ));
    let whole_free = final_free_energy(&whole_final.progress, whole_final.hamiltonian.dim());
    let resumed_free = final_free_energy(&resumed_final.progress, resumed_final.hamiltonian.dim());
    maxima.free_energy = maxima.free_energy.max(assert_close(
        whole_free,
        resumed_free,
        &format!("{label} final free energy"),
    ));
    for parity in [RdmParity::A, RdmParity::B] {
        for length in 1..=2 {
            let left = support::rdm(&whole_final.state, parity, length);
            let right = support::rdm(&resumed_final.state, parity, length);
            let difference = (&left - &right).norm();
            let scale = 1.0_f64.max(left.norm()).max(right.norm());
            assert!(
                difference <= TOLERANCE * scale,
                "{label} RDM {parity:?} length {length}: difference={difference:e} scale={scale:e}"
            );
            maxima.rdm = maxima.rdm.max(difference);
        }
    }
}

fn write_config(
    dir: &Path,
    name: &str,
    model: &str,
    order: u8,
    canonicalize_every: usize,
    record_stride: usize,
    target_steps: u64,
    checkpoint: bool,
    restart: Option<&Path>,
) -> PathBuf {
    let path = dir.join(format!("{name}.toml"));
    let model_config = match model {
        "tfim" => "type='tfim'\nj=1.0\ng=0.7",
        "xy" => "type='xy'\ngamma=0.5\nh=0.7",
        _ => unreachable!(),
    };
    let output = dir.join(format!("{name}.json"));
    let checkpoint_text = if checkpoint {
        format!(
            "\n[checkpoint]\npath='{}'\nevery_steps=4\n",
            dir.join(format!("{name}.h5")).display()
        )
    } else {
        String::new()
    };
    let restart_text = restart
        .map(|source| format!("\n[restart]\npath='{}'\n", source.display()))
        .unwrap_or_default();
    let text = format!(
        "[model]\n{model_config}\n[evolution]\ndtau=0.01\nbeta_max={}\nrecord_every_beta={}\ntrotter_order={order}\n[truncation]\nepsilon=1e-12\nmax_bond=32\n[run]\ncanonicalize_every={canonicalize_every}\n[output]\npath='{}'\ninclude_exact=false\n{checkpoint_text}{restart_text}",
        0.02 * target_steps as f64,
        0.02 * record_stride as f64,
        output.display(),
    );
    std::fs::write(&path, text).unwrap();
    path
}

fn output_path(config: &Path, extension: &str) -> PathBuf {
    config.with_extension(extension)
}

fn run(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(path)
        .output()
        .unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn checkpoint_steps(path: &Path) -> Vec<u64> {
    list_itebd_checkpoints(path)
        .unwrap()
        .into_iter()
        .map(|entry| entry.completed_steps)
        .collect()
}

fn record_steps(result: &SweepResult) -> Vec<u64> {
    result
        .records
        .iter()
        .map(|record| beta_step(record.beta))
        .collect()
}

fn beta_step(beta: f64) -> u64 {
    (beta / 0.02).round() as u64
}

fn compare_records(actual: &Record, expected: &Record, context: &str) -> f64 {
    assert_eq!(
        actual.beta.to_bits(),
        expected.beta.to_bits(),
        "{context} beta"
    );
    assert_eq!(actual.max_bond, expected.max_bond, "{context} max bond");
    let mut maximum = 0.0_f64;
    assert_eq!(actual.f.is_some(), expected.f.is_some());
    if let (Some(a), Some(b)) = (actual.f, expected.f) {
        maximum = maximum.max(assert_close(a, b, &format!("{context} free energy")));
    }
    for (field, actual, expected) in [
        ("energy", actual.u, expected.u),
        ("specific heat", actual.c, expected.c),
        ("beta f", actual.beta_f, expected.beta_f),
        (
            "magnetization",
            actual.magnetization,
            expected.magnetization,
        ),
    ] {
        maximum = maximum.max(assert_close(
            actual,
            expected,
            &format!("{context} {field}"),
        ));
    }
    maximum
}

fn assert_close(actual: f64, expected: f64, context: &str) -> f64 {
    let difference = (actual - expected).abs();
    let scale = 1.0_f64.max(actual.abs()).max(expected.abs());
    assert!(
        difference <= TOLERANCE * scale,
        "{context}: actual={actual:e} expected={expected:e} difference={difference:e} scale={scale:e}"
    );
    difference
}

fn compare_schmidt(actual: &ItebdState, expected: &ItebdState, context: &str) -> f64 {
    let mut maximum = 0.0_f64;
    for (bond, (actual, expected)) in schmidt_spectra(actual)
        .into_iter()
        .zip(schmidt_spectra(expected))
        .enumerate()
    {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context} Schmidt bond {bond}"
        );
        for (position, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            maximum = maximum.max(assert_close(
                actual,
                expected,
                &format!("{context} Schmidt bond {bond} value {position}"),
            ));
        }
    }
    maximum
}

fn schmidt_spectra(state: &ItebdState) -> [&[f64]; 2] {
    match state {
        ItebdState::Real(state) => [&state.lambda_ab, &state.lambda_ba],
        ItebdState::Complex(state) => [&state.lambda_ab, &state.lambda_ba],
    }
}

fn load_final(path: &Path) -> thermal_imps_purification::itebd_checkpoint::LoadedItebdCheckpoint {
    let entries = list_itebd_checkpoints(path).unwrap();
    load_itebd_checkpoint(
        path,
        (entries.len() - 1) as u64,
        &ItebdCheckpointLoadOptions::default(),
    )
    .unwrap()
}

fn final_free_energy(
    progress: &thermal_imps_purification::itebd_checkpoint::ItebdCheckpointProgress,
    local_dim: usize,
) -> f64 {
    free_energy_from_log_norm(
        progress.accumulated_log_norm / 2.0,
        progress.beta,
        local_dim,
    )
}
