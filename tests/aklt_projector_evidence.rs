#[path = "support/aklt_projector_solve.rs"]
mod support;

use nalgebra::DMatrix;
use serde_json::json;
use std::path::Path;
use thermal_imps_purification::canonicalize::canonicalize;
use thermal_imps_purification::itebd::{
    free_energy_from_log_norm, imaginary_time_step, imaginary_time_step_second_order,
};
use thermal_imps_purification::itebd_rdm::{reduced_density_matrix, RdmOptions, RdmParity};
use thermal_imps_purification::model::{sz1, AkltProjector, BilinearBiquadratic, LocalHamiltonian};
use thermal_imps_purification::observable::{energy_density, magnetization};
use thermal_imps_purification::purified_mps::infinite_temperature;
use thermal_imps_purification::runner::{read_result, Record};
use thermal_imps_purification::tensor::Truncation;
use thermal_imps_purification::variance::specific_heat;

// HDF5 native descriptors can be inherited by concurrently launched solve processes.
// Every CLI or HDF5 operation in this target must share this lock.
static PROCESS_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

const REFERENCE_EPSILON: f64 = 1e-12;
const REFERENCE_CAP: usize = 32;

#[derive(Clone)]
struct Sample {
    beta: f64,
    u: f64,
    c: f64,
    f: f64,
    sz: f64,
    bond: usize,
    trajectory_max_bond: usize,
    log_norm: f64,
    a: DMatrix<f64>,
    b: DMatrix<f64>,
}

fn evolve(
    ham: &LocalHamiltonian,
    order: u8,
    dtau: f64,
    epsilon: f64,
    cap: usize,
    endpoints: &[f64],
) -> Vec<Sample> {
    let trunc = Truncation {
        epsilon,
        max_bond: Some(cap),
    };
    let mut state = infinite_temperature(3);
    let mut log_norm = 0.0;
    let mut samples = Vec::new();
    let mut completed = 0_usize;
    let mut trajectory_max_bond = 1_usize;
    for &beta in endpoints {
        let target = (beta / (2.0 * dtau)).round() as usize;
        assert!((2.0 * dtau * target as f64 - beta).abs() <= 1e-14);
        while completed < target {
            let info = match order {
                1 => imaginary_time_step(&mut state, ham, dtau, &trunc),
                2 => imaginary_time_step_second_order(&mut state, ham, dtau, &trunc),
                _ => panic!("invalid test Trotter order"),
            };
            log_norm += info.log_norm;
            trajectory_max_bond = trajectory_max_bond.max(info.max_bond);
            log_norm += canonicalize(&mut state);
            completed += 1;
        }
        let u = energy_density(&state, ham);
        let c = specific_heat(&state, ham, beta).unwrap();
        let f = free_energy_from_log_norm(log_norm / 2.0, beta, 3);
        let sz = magnetization(&state, &sz1());
        assert!(
            [u, c, f, sz, log_norm].into_iter().all(f64::is_finite),
            "non-finite direct sample at beta={beta:.16e}"
        );
        let rdm = |parity| {
            reduced_density_matrix(&state, parity, 1, &RdmOptions::default())
                .unwrap()
                .density_matrix
        };
        samples.push(Sample {
            beta,
            u,
            c,
            f,
            sz,
            bond: state.lambda_ab.len().max(state.lambda_ba.len()),
            trajectory_max_bond,
            log_norm,
            a: rdm(RdmParity::A),
            b: rdm(RdmParity::B),
        });
    }
    samples
}

fn absolute_scaled(a: f64, b: f64) -> (f64, f64) {
    let absolute = (a - b).abs();
    (absolute, absolute / a.abs().max(b.abs()).max(1.0))
}

fn print_sample(model: &str, order: u8, dtau: f64, epsilon: f64, cap: usize, sample: &Sample) {
    println!(
        "SAMPLE model={model} order={order} dtau={dtau:.16e} cutoff={epsilon:.16e} cap={cap} beta={:.16e} u={:.16e} c={:.16e} f={:.16e} sz={:.16e} endpoint_bond={} trajectory_max_bond={} log_norm={:.16e}",
        sample.beta,
        sample.u,
        sample.c,
        sample.f,
        sample.sz,
        sample.bond,
        sample.trajectory_max_bond,
        sample.log_norm,
    );
}

fn print_mapping_field(field: &str, actual: f64, expected: f64) {
    let (absolute, scaled) = absolute_scaled(actual, expected);
    println!(
        "MAPPING field={field} actual={actual:.16e} expected={expected:.16e} absolute={absolute:.16e} scaled={scaled:.16e}"
    );
}

fn check_mapping(p: &Sample, old: &Sample) {
    let expected_u = old.u / 2.0 + 1.0 / 3.0;
    let expected_f = old.f / 2.0 + 1.0 / 3.0;
    let a_rdm = (&p.a - &old.a).norm();
    let b_rdm = (&p.b - &old.b).norm();

    print_mapping_field("beta", p.beta / 2.0, old.beta);
    print_mapping_field("u", p.u, expected_u);
    print_mapping_field("f", p.f, expected_f);
    print_mapping_field("c", p.c, old.c);
    print_mapping_field("sz", p.sz, old.sz);
    println!("MAPPING field=rdm_a frobenius={a_rdm:.16e}");
    println!("MAPPING field=rdm_b frobenius={b_rdm:.16e}");

    support::close(p.beta / 2.0, old.beta, 1e-14);
    support::close(p.u, expected_u, 1e-8);
    support::close(p.f, expected_f, 1e-8);
    support::close(p.c, old.c, 1e-7);
    support::close(p.sz, old.sz, 1e-8);
    assert!(a_rdm <= 1e-8, "A RDM Frobenius difference={a_rdm:.16e}");
    assert!(b_rdm <= 1e-8, "B RDM Frobenius difference={b_rdm:.16e}");
}

fn run_cli_success(path: &Path) {
    let output = support::run_cli(path);
    assert!(
        output.status.success(),
        "config={} status={} stdout={} stderr={}",
        path.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn compare_direct_cli(order: u8, direct: &Sample, cli: &Record) {
    for (field, actual, expected) in [
        ("u", cli.u, direct.u),
        ("c", cli.c, direct.c),
        ("f", cli.f.unwrap(), direct.f),
        ("sz", cli.magnetization, direct.sz),
    ] {
        let (absolute, scaled) = absolute_scaled(actual, expected);
        println!(
            "DIRECT_CLI order={order} field={field} cli={actual:.16e} direct={expected:.16e} absolute={absolute:.16e} scaled={scaled:.16e}"
        );
        support::close(actual, expected, 1e-10);
    }
}

#[test]
fn projector_high_temperature_control() {
    let beta = 0.01;
    let sample = evolve(&AkltProjector.local(), 2, 0.005, 1e-12, 16, &[beta])
        .into_iter()
        .next()
        .unwrap();
    print_sample("projector", 2, 0.005, 1e-12, 16, &sample);
    let entropy_residual = (beta * sample.f + 3.0_f64.ln()).abs();
    println!(
        "HIGH_T beta_f={:.16e} negative_ln3={:.16e} residual={entropy_residual:.16e} leading_beta_u_infinite={:.16e}",
        beta * sample.f,
        -3.0_f64.ln(),
        beta * 5.0 / 9.0,
    );
    assert!(sample.u.is_finite() && sample.c.is_finite() && sample.f.is_finite());
    assert!(entropy_residual <= 1e-2);
    assert!(sample.sz.abs() <= 1e-8);
}

#[test]
fn projector_direct_matches_json_cli_for_both_orders() {
    let _process_io = PROCESS_IO.lock().unwrap_or_else(|error| error.into_inner());
    let root = tempfile::tempdir().unwrap();

    for order in [1, 2] {
        let direct = evolve(
            &AkltProjector.local(),
            order,
            0.05,
            REFERENCE_EPSILON,
            REFERENCE_CAP,
            &[0.2],
        )
        .into_iter()
        .next()
        .unwrap();
        let mut value = support::config(
            root.path(),
            &format!("projector-order-{order}"),
            "aklt_projector",
            order,
            0.2,
        );
        value["evolution"]["record_every_beta"] = json!(0.2);
        let input = root.path().join(format!("projector-order-{order}.json"));
        support::write_input(&value, &input, "json");
        run_cli_success(&input);
        let result = read_result(Path::new(value["output"]["path"].as_str().unwrap())).unwrap();
        assert_eq!(result.records.len(), 2);
        let cli = &result.records[1];
        support::close(cli.beta, direct.beta, 1e-10);
        compare_direct_cli(order, &direct, cli);
        println!(
            "DIRECT_CLI_BOND order={order} cli={} direct_endpoint={} direct_trajectory_max={}",
            cli.max_bond, direct.bond, direct.trajectory_max_bond
        );
    }
}

#[derive(Clone)]
struct GridRow {
    dtau: f64,
    epsilon: f64,
    cap: usize,
    sample: Sample,
}

fn row<'a>(rows: &'a [GridRow], dtau: f64, epsilon: f64, cap: usize, beta: f64) -> &'a Sample {
    &rows
        .iter()
        .find(|row| {
            row.dtau == dtau && row.epsilon == epsilon && row.cap == cap && row.sample.beta == beta
        })
        .unwrap()
        .sample
}

fn print_change(kind: &str, beta: f64, field: &str, reference: f64, comparison: f64, detail: &str) {
    println!(
        "SENSITIVITY kind={kind} beta={beta:.16e} field={field} absolute={:.16e} reference={reference:.16e} comparison={comparison:.16e} {detail}",
        (reference - comparison).abs(),
    );
}

fn print_sensitivities(rows: &[GridRow]) {
    for dtau in [0.05, 0.025] {
        for cap in [16, 32] {
            for beta in [0.2, 0.4] {
                let reference = row(rows, dtau, REFERENCE_EPSILON, cap, beta);
                let comparison = row(rows, dtau, 1e-10, cap, beta);
                for (field, a, b) in [
                    ("u", reference.u, comparison.u),
                    ("c", reference.c, comparison.c),
                    ("f", reference.f, comparison.f),
                ] {
                    print_change(
                        "cutoff",
                        beta,
                        field,
                        a,
                        b,
                        &format!(
                            "dtau={dtau:.16e} fixed_cap={cap} reference_cutoff={REFERENCE_EPSILON:.16e} comparison_cutoff={:.16e}",
                            1e-10
                        ),
                    );
                }
            }
        }
    }

    for dtau in [0.05, 0.025] {
        for epsilon in [1e-10, REFERENCE_EPSILON] {
            for beta in [0.2, 0.4] {
                let reference = row(rows, dtau, epsilon, REFERENCE_CAP, beta);
                let comparison = row(rows, dtau, epsilon, 16, beta);
                for (field, a, b) in [
                    ("u", reference.u, comparison.u),
                    ("c", reference.c, comparison.c),
                    ("f", reference.f, comparison.f),
                ] {
                    print_change(
                        "cap",
                        beta,
                        field,
                        a,
                        b,
                        &format!(
                            "dtau={dtau:.16e} fixed_cutoff={epsilon:.16e} reference_cap={REFERENCE_CAP} comparison_cap=16"
                        ),
                    );
                }
            }
        }
    }

    for beta in [0.2, 0.4] {
        let reference = row(rows, 0.025, REFERENCE_EPSILON, REFERENCE_CAP, beta);
        let comparison = row(rows, 0.05, REFERENCE_EPSILON, REFERENCE_CAP, beta);
        for (field, a, b) in [
            ("u", reference.u, comparison.u),
            ("c", reference.c, comparison.c),
            ("f", reference.f, comparison.f),
        ] {
            print_change(
                "step",
                beta,
                field,
                a,
                b,
                &format!(
                    "fixed_cutoff={REFERENCE_EPSILON:.16e} fixed_cap={REFERENCE_CAP} reference_dtau={:.16e} comparison_dtau={:.16e}",
                    0.025, 0.05
                ),
            );
        }
    }
}

#[test]
#[ignore = "release scientific qualification: eight matched second-order trajectory pairs"]
fn projector_old_normalization_grid() {
    assert!(
        !cfg!(debug_assertions),
        "run scientific qualification with --release"
    );
    println!(
        "REFERENCE_SELECTION order=2 cutoff={REFERENCE_EPSILON:.16e} cap={REFERENCE_CAP} dtau_values={:.16e},{:.16e} selected_before_execution=true",
        0.05, 0.025
    );
    let mut rows = Vec::new();
    let mut pair_count = 0_usize;
    let mut endpoint_sample_count = 0_usize;

    for dtau in [0.05, 0.025] {
        for epsilon in [1e-10, 1e-12] {
            for cap in [16, 32] {
                let p = evolve(&AkltProjector.local(), 2, dtau, epsilon, cap, &[0.2, 0.4]);
                let old = evolve(
                    &BilinearBiquadratic::aklt().local(),
                    2,
                    dtau / 2.0,
                    epsilon,
                    cap,
                    &[0.1, 0.2],
                );
                pair_count += 1;
                endpoint_sample_count += p.len() + old.len();
                for (projector, old_sample) in p.iter().zip(&old) {
                    print_sample("projector", 2, dtau, epsilon, cap, projector);
                    print_sample("old", 2, dtau / 2.0, epsilon, cap, old_sample);
                    println!(
                        "PAIR order=2 projector_dtau={dtau:.16e} old_dtau={:.16e} cutoff={epsilon:.16e} cap={cap} projector_beta={:.16e} old_beta={:.16e}",
                        dtau / 2.0,
                        projector.beta,
                        old_sample.beta,
                    );
                    check_mapping(projector, old_sample);
                    rows.push(GridRow {
                        dtau,
                        epsilon,
                        cap,
                        sample: projector.clone(),
                    });
                }
            }
        }
    }

    println!(
        "COUNT order=2 trajectory_pairs={pair_count} trajectories={} endpoint_samples={endpoint_sample_count}",
        pair_count * 2
    );
    assert_eq!(pair_count, 8);
    assert_eq!(endpoint_sample_count, 32);
    assert_eq!(rows.len(), 16);
    print_sensitivities(&rows);
}

#[test]
#[ignore = "release scientific qualification: one matched first-order trajectory pair"]
fn projector_old_first_order_control() {
    assert!(
        !cfg!(debug_assertions),
        "run scientific qualification with --release"
    );
    let dtau = 0.05;
    let epsilon = 1e-12;
    let cap = 32;
    let p = evolve(&AkltProjector.local(), 1, dtau, epsilon, cap, &[0.2]);
    let old = evolve(
        &BilinearBiquadratic::aklt().local(),
        1,
        dtau / 2.0,
        epsilon,
        cap,
        &[0.1],
    );
    print_sample("projector", 1, dtau, epsilon, cap, &p[0]);
    print_sample("old", 1, dtau / 2.0, epsilon, cap, &old[0]);
    println!(
        "PAIR order=1 projector_dtau={dtau:.16e} old_dtau={:.16e} cutoff={epsilon:.16e} cap={cap} projector_beta={:.16e} old_beta={:.16e}",
        dtau / 2.0,
        p[0].beta,
        old[0].beta,
    );
    check_mapping(&p[0], &old[0]);
    println!("COUNT order=1 trajectory_pairs=1 trajectories=2 endpoint_samples=2");
}
