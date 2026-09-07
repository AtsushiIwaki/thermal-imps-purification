#[path = "support/solve_matrix.rs"]
mod support;
use thermal_imps_purification::config::TrotterOrder;
use thermal_imps_purification::exact::{
    exact_energy_density, exact_magnetization_x, exact_specific_heat, free_energy_density,
};
use thermal_imps_purification::runner::{read_result, Record};
use serde_json::json;

fn values(r: &Record) -> [f64; 4] {
    [r.u, r.c, r.f, r.magnetization]
}
fn oracle_values(r: support::OracleRecord) -> [f64; 4] {
    [r.u, r.c, r.f, r.local]
}
const FIELDS: [&str; 4] = ["u", "c", "f", "local"];
fn close(a: [f64; 4], b: [f64; 4], label: &str) -> (f64, f64) {
    let mut absolute: f64 = 0.0;
    let mut scaled: f64 = 0.0;
    for i in 0..4 {
        let diff = (a[i] - b[i]).abs();
        let residual = diff / a[i].abs().max(b[i].abs()).max(1.0);
        assert!(
            residual <= 1e-10,
            "{label} {} abs={diff:e} scaled={residual:e}",
            FIELDS[i]
        );
        absolute = absolute.max(diff);
        scaled = scaled.max(residual);
    }
    (absolute, scaled)
}
fn cli(mut cfg: serde_json::Value, format: &str) -> Record {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("result.json");
    cfg["output"]["path"] = json!(output);
    let input = dir.path().join(format!("input.{format}"));
    support::write_input(&cfg, &input, format);
    support::run_cli(&input);
    let result = read_result(&output).unwrap();
    assert_eq!(result.records.len(), 1);
    assert_eq!(
        serde_json::to_value(&result.metadata.model).unwrap(),
        cfg["model"]
    );
    result.records.into_iter().next().unwrap()
}

#[test]
fn matrix_cli_direct_oracles_both_orders() {
    // Catches wrong fresh backend/observable dispatch or matrix leg/sign conventions at the CLI.
    for order in [TrotterOrder::First, TrotterOrder::Second] {
        let real = support::real_oracle(order);
        let phase = support::phase_oracle(order);
        // Cross-backend equality is a retained diagnostic, separate from the required
        // comparisons of each CLI mode to its corresponding independent direct API.
        for (i, (a, b)) in oracle_values(real)
            .into_iter()
            .zip(oracle_values(phase))
            .enumerate()
        {
            println!("CROSS_BACKEND {order:?} {} real={a:.16e} phase={b:.16e} absolute={:.16e} scaled={:.16e}",
                FIELDS[i], (a-b).abs(), (a-b).abs()/a.abs().max(b.abs()).max(1.0));
        }
        for model in ["builtin", "real_matrix", "complex_matrix"] {
            let mut cfg = if model == "complex_matrix" {
                support::phase_config()
            } else {
                support::real_tfim_config()
            };
            if model == "builtin" {
                cfg["model"] = json!({"type":"tfim","j":1.0,"g":0.7});
            }
            cfg["evolution"]["trotter_order"] =
                json!(if order == TrotterOrder::First { 1 } else { 2 });
            let record = cli(cfg, "json");
            let oracle = if model == "complex_matrix" {
                phase
            } else {
                real
            };
            let (abs, scaled) = close(values(&record), oracle_values(oracle), model);
            assert_eq!(record.max_bond, oracle.max_bond);
            println!("ENDPOINT {model} {order:?} absolute={abs:.16e} scaled={scaled:.16e}");
        }
    }
}

#[derive(Clone)]
struct Row {
    dtau: f64,
    epsilon: f64,
    cap: usize,
    format: &'static str,
    values: [f64; 4],
}

#[test]
#[ignore = "release scientific qualification: 16 CLI processes and independent direct rotations"]
fn matrix_cli_refinement() {
    // Fixed gates catch a broken Trotter order, normalization convention, complex rotation,
    // or cutoff/cap sensitivity that invalidates the prespecified numerical window.
    assert!(
        !cfg!(debug_assertions),
        "run scientific qualification with --release"
    );
    let exact = |nk| {
        [
            exact_energy_density(1., 0.7, 1., nk),
            exact_specific_heat(1., 0.7, 1., nk),
            free_energy_density(1., 0.7, 1., nk),
            exact_magnetization_x(1., 0.7, 1., nk),
        ]
    };
    let reference = exact(16_384);
    let refined_reference = exact(32_768);
    for i in 0..4 {
        println!(
            "REFERENCE {} nk16384={:.16e} nk32768={:.16e} absolute_change={:.16e}",
            FIELDS[i],
            reference[i],
            refined_reference[i],
            (reference[i] - refined_reference[i]).abs()
        );
    }
    println!("ROW dtau epsilon cap format u c f local max_bond direct_abs direct_scaled");
    let mut rows = Vec::new();
    for dtau in [0.1, 0.05] {
        for epsilon in [1e-13, 1e-14] {
            for cap in [64, 128] {
                let direct =
                    support::phase_oracle_window(TrotterOrder::Second, dtau, 1., epsilon, cap);
                for format in ["toml", "json"] {
                    let mut cfg = support::phase_config();
                    cfg["evolution"] = json!({"dtau":dtau,"beta_max":1.0,"record_every_beta":1.0,"trotter_order":2});
                    cfg["truncation"] = json!({"epsilon":epsilon,"max_bond":cap});
                    let record = cli(cfg, format);
                    let v = values(&record);
                    // Preserve completed raw rows before applying any numerical gate.
                    let diff = v
                        .iter()
                        .zip(oracle_values(direct))
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f64::max);
                    let scaled = v
                        .iter()
                        .zip(oracle_values(direct))
                        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.))
                        .fold(0., f64::max);
                    println!("ROW {dtau:.2} {epsilon:.1e} {cap} {format} {:.16e} {:.16e} {:.16e} {:.16e} {} {diff:.16e} {scaled:.16e}",v[0],v[1],v[2],v[3],record.max_bond);
                    rows.push(Row {
                        dtau,
                        epsilon,
                        cap,
                        format,
                        values: v,
                    });
                    close(v, oracle_values(direct), "direct/CLI");
                    assert_eq!(record.max_bond, direct.max_bond);
                }
            }
        }
    }
    // All scientific gates run after retaining the full Cartesian table.
    for row in &rows {
        let json = rows
            .iter()
            .find(|r| {
                r.dtau == row.dtau
                    && r.epsilon == row.epsilon
                    && r.cap == row.cap
                    && r.format == "json"
            })
            .unwrap();
        close(row.values, json.values, "TOML/JSON");
    }
    let mut failures = Vec::new();
    for format in ["toml", "json"] {
        for epsilon in [1e-13, 1e-14] {
            for cap in [64, 128] {
                let lookup = |dtau| {
                    rows.iter()
                        .find(|r| {
                            r.dtau == dtau
                                && r.epsilon == epsilon
                                && r.cap == cap
                                && r.format == format
                        })
                        .unwrap()
                };
                for i in 0..4 {
                    let coarse_error = (lookup(0.1).values[i] - reference[i]).abs();
                    let fine_error = (lookup(0.05).values[i] - reference[i]).abs();
                    let order = (coarse_error / fine_error).log2();
                    println!("REFINEMENT {format} {epsilon:.1e} {cap} {} coarse_error={coarse_error:.16e} fine_error={fine_error:.16e} order={order:.16e}",FIELDS[i]);
                    if !(coarse_error > 0. && fine_error > 0. && fine_error < coarse_error) {
                        failures.push(format!(
                            "decreasing errors {format} {epsilon} {cap} {}",
                            FIELDS[i]
                        ));
                    }
                    if epsilon == 1e-14 && cap == 128 && !(order >= 1.6) {
                        failures.push(format!(
                            "primary refinement order {order} {} {format}",
                            FIELDS[i]
                        ));
                    }
                }
            }
        }
    }
    for row in &rows {
        let primary = rows
            .iter()
            .find(|r| {
                r.dtau == row.dtau && r.epsilon == 1e-14 && r.cap == 128 && r.format == row.format
            })
            .unwrap();
        for i in 0..4 {
            let primary_exact_error = (primary.values[i] - reference[i]).abs();
            let difference = (row.values[i] - primary.values[i]).abs();
            let fraction = difference / primary_exact_error;
            println!("SENSITIVITY {} {:.2} {:.1e} {} {} absolute={difference:.16e} fraction={fraction:.16e}",row.format,row.dtau,row.epsilon,row.cap,FIELDS[i]);
            if !(difference <= 0.05 * primary_exact_error) {
                failures.push(format!(
                    "sensitivity {} {} {} {}: {fraction}",
                    row.format, row.dtau, row.epsilon, FIELDS[i]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fixed scientific gates failed: {failures:#?}"
    );
}
