use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};
use thermal_imps_purification::canonicalize::canonicalize;
use thermal_imps_purification::config::{ModelSpec, TrotterOrder};
use thermal_imps_purification::itebd::{
    free_energy_from_log_norm, imaginary_time_step, imaginary_time_step_second_order,
};
use thermal_imps_purification::observable::{energy_density, magnetization};
use thermal_imps_purification::purified_mps::infinite_temperature;
use thermal_imps_purification::tensor::Truncation;
use thermal_imps_purification::variance::specific_heat;

#[derive(Debug, Clone, Copy)]
struct EvolutionSample {
    dtau: f64,
    energy: f64,
    free_energy: f64,
    specific_heat: f64,
    magnetization: f64,
    max_bond: usize,
}

fn evolve(
    model: &ModelSpec,
    order: TrotterOrder,
    beta: f64,
    dtau: f64,
    trunc: Truncation,
) -> EvolutionSample {
    let ham = model.hamiltonian().unwrap();
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!((2.0 * steps as f64 * dtau - beta).abs() <= 1e-12);

    let mut state = infinite_temperature(ham.dim());
    let mut accum = 0.0;
    let mut max_bond = 1;
    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => imaginary_time_step(&mut state, &ham, dtau, &trunc),
            TrotterOrder::Second => {
                imaginary_time_step_second_order(&mut state, &ham, dtau, &trunc)
            }
        };
        max_bond = max_bond.max(info.max_bond);
        accum += info.log_norm + canonicalize(&mut state);
    }

    EvolutionSample {
        dtau,
        energy: energy_density(&state, &ham),
        free_energy: free_energy_from_log_norm(accum / 2.0, beta, ham.dim()),
        specific_heat: specific_heat(&state, &ham, beta).unwrap(),
        magnetization: magnetization(&state, &model.magnetization_op().unwrap()),
        max_bond,
    }
}

fn refinement_order(coarse: f64, fine: f64) -> f64 {
    (coarse / fine).ln() / 2.0_f64.ln()
}

fn has_quadratic_window(errors: &[f64]) -> bool {
    errors
        .windows(2)
        .any(|p| p[0] > p[1] && refinement_order(p[0], p[1]) >= 1.6)
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn timed_evolution(
    model: &ModelSpec,
    order: TrotterOrder,
    beta: f64,
    dtau: f64,
    trunc: Truncation,
) -> (Duration, usize, usize) {
    let ham = model.hamiltonian().unwrap();
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!((2.0 * steps as f64 * dtau - beta).abs() <= 1e-12);

    let mut state = infinite_temperature(ham.dim());
    let mut accum = 0.0;
    let mut max_bond = 1;
    let start = Instant::now();
    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => imaginary_time_step(&mut state, &ham, dtau, &trunc),
            TrotterOrder::Second => {
                imaginary_time_step_second_order(&mut state, &ham, dtau, &trunc)
            }
        };
        max_bond = max_bond.max(info.max_bond);
        accum += info.log_norm + canonicalize(&mut state);
    }
    let elapsed = start.elapsed();

    let final_energy = black_box(energy_density(&state, &ham));
    black_box((final_energy, accum));
    (elapsed, steps, max_bond)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TimingKey {
    second_order: bool,
    dtau_bits: u64,
}

impl TimingKey {
    fn new(order: TrotterOrder, dtau: f64) -> Self {
        Self {
            second_order: matches!(order, TrotterOrder::Second),
            dtau_bits: dtau.to_bits(),
        }
    }

    fn order(self) -> TrotterOrder {
        if self.second_order {
            TrotterOrder::Second
        } else {
            TrotterOrder::First
        }
    }

    fn dtau(self) -> f64 {
        f64::from_bits(self.dtau_bits)
    }
}

#[derive(Debug)]
struct TimingAccumulator {
    durations: Vec<Duration>,
    steps: Option<usize>,
    max_bond: usize,
}

#[derive(Debug)]
struct TimingSummary {
    raw_ns: Vec<u128>,
    median: Duration,
    beta: f64,
    steps: usize,
    max_bond: usize,
}

fn measure_timing_workloads(
    model: &ModelSpec,
    beta: f64,
    trunc: Truncation,
    workloads: &[TimingKey],
    warmups: usize,
    samples: usize,
) -> HashMap<TimingKey, TimingSummary> {
    let mut unique_workloads = Vec::with_capacity(workloads.len());
    for key in workloads {
        if !unique_workloads.contains(key) {
            unique_workloads.push(*key);
        }
    }

    for warmup in 0..warmups {
        for second_order in if warmup % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            for key in unique_workloads
                .iter()
                .copied()
                .filter(|key| key.second_order == second_order)
            {
                let _ = timed_evolution(model, key.order(), beta, key.dtau(), trunc.clone());
            }
        }
    }

    let mut accumulators: HashMap<_, _> = unique_workloads
        .iter()
        .copied()
        .map(|key| {
            (
                key,
                TimingAccumulator {
                    durations: Vec::with_capacity(samples),
                    steps: None,
                    max_bond: 1,
                },
            )
        })
        .collect();

    for sample in 0..samples {
        for second_order in if sample % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            for key in unique_workloads
                .iter()
                .copied()
                .filter(|key| key.second_order == second_order)
            {
                let (duration, steps, max_bond) =
                    timed_evolution(model, key.order(), beta, key.dtau(), trunc.clone());
                assert!(
                    duration > Duration::ZERO,
                    "timed workload returned a zero duration: {key:?}"
                );
                let accumulator = accumulators
                    .get_mut(&key)
                    .expect("timing accumulator exists for every unique workload");
                if let Some(expected_steps) = accumulator.steps {
                    assert_eq!(steps, expected_steps, "step count changed for {key:?}");
                } else {
                    accumulator.steps = Some(steps);
                }
                accumulator.max_bond = accumulator.max_bond.max(max_bond);
                accumulator.durations.push(duration);
            }
        }
    }

    accumulators
        .into_iter()
        .map(|(key, accumulator)| {
            let mut sorted = accumulator.durations.clone();
            let median = median_duration(&mut sorted);
            (
                key,
                TimingSummary {
                    raw_ns: accumulator
                        .durations
                        .iter()
                        .map(Duration::as_nanos)
                        .collect(),
                    median,
                    beta,
                    steps: accumulator.steps.expect("at least one timing sample"),
                    max_bond: accumulator.max_bond,
                },
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct EqualAccuracySelection {
    regime: &'static str,
    observable: &'static str,
    first_dtau: f64,
    second_dtau: f64,
    first_error: f64,
    second_error: f64,
}

#[derive(Debug, Clone, Copy)]
struct EqualAccuracyCandidate {
    dtau: f64,
    errors: [f64; 4],
}

fn select_equal_accuracy_pair(
    regime: &'static str,
    observable: &'static str,
    observable_index: usize,
    first: &[EqualAccuracyCandidate],
    second: &[EqualAccuracyCandidate],
) -> Option<EqualAccuracySelection> {
    let mut best: Option<(f64, EqualAccuracySelection)> = None;
    for first_sample in first {
        let first_error = first_sample.errors[observable_index];
        for second_sample in second {
            let second_error = second_sample.errors[observable_index];
            let ratio = first_error / second_error;
            if first_error > 0.0
                && second_error > 0.0
                && ratio.is_finite()
                && (0.5..=2.0).contains(&ratio)
            {
                let score = ratio.ln().abs();
                let selection = EqualAccuracySelection {
                    regime,
                    observable,
                    first_dtau: first_sample.dtau,
                    second_dtau: second_sample.dtau,
                    first_error,
                    second_error,
                };
                if best
                    .as_ref()
                    .map_or(true, |(best_score, _)| score < *best_score)
                {
                    best = Some((score, selection));
                }
            }
        }
    }
    best.map(|(_, selection)| selection)
}

fn select_required_equal_accuracy_pair(
    fine_regime: &'static str,
    observable: &'static str,
    observable_index: usize,
    fine_first: &[EqualAccuracyCandidate],
    fine_second: &[EqualAccuracyCandidate],
    coarse: Option<(&[EqualAccuracyCandidate], &[EqualAccuracyCandidate])>,
) -> Option<EqualAccuracySelection> {
    let fine = select_equal_accuracy_pair(
        fine_regime,
        observable,
        observable_index,
        fine_first,
        fine_second,
    );
    if fine.is_some() || observable_index != 0 {
        return fine;
    }

    coarse.and_then(|(coarse_first, coarse_second)| {
        select_equal_accuracy_pair(
            "xy_coarse_trotter_regime",
            observable,
            observable_index,
            coarse_first,
            coarse_second,
        )
    })
}

fn format_equal_accuracy_timing_record(
    model_name: &str,
    selection: &EqualAccuracySelection,
    beta: f64,
    epsilon: f64,
    cap: usize,
    warmups: usize,
    samples: usize,
    first: &TimingSummary,
    second: &TimingSummary,
) -> String {
    let error_ratio = selection.first_error / selection.second_error;
    let first_per_step_ns = first.median.as_nanos() as f64 / first.steps as f64;
    let second_per_step_ns = second.median.as_nanos() as f64 / second.steps as f64;
    let timing_ratio = second.median.as_secs_f64() / first.median.as_secs_f64();
    format!(
        "ITEBD_TROTTER_TIMING_EQUAL_ACCURACY regime={} model={model_name} observable={} selection=available beta={beta:.16e} epsilon={epsilon:.16e} cap={cap} warmups={warmups} samples={samples} first_dtau={:.16e} second_dtau={:.16e} first_error={:.16e} second_error={:.16e} error_ratio={error_ratio:.16e} first_raw_ns={:?} second_raw_ns={:?} first_median_ns={} second_median_ns={} first_steps={} second_steps={} first_bond_updates={} second_bond_updates={} first_max_bond={} second_max_bond={} first_per_step_ns={first_per_step_ns:.16e} second_per_step_ns={second_per_step_ns:.16e} second_over_first={timing_ratio:.16e}",
        selection.regime,
        selection.observable,
        selection.first_dtau,
        selection.second_dtau,
        selection.first_error,
        selection.second_error,
        first.raw_ns,
        second.raw_ns,
        first.median.as_nanos(),
        second.median.as_nanos(),
        first.steps,
        second.steps,
        2 * first.steps,
        3 * second.steps,
        first.max_bond,
        second.max_bond,
    )
}

fn assert_second_order_matches_exact_and_improves(
    model_name: &str,
    model: ModelSpec,
    exact_energy: f64,
    exact_free_energy: f64,
    exact_specific_heat: f64,
    exact_magnetization: f64,
) {
    let beta = 1.0;
    let dtau = 0.05;
    let max_bond = 64;
    let trunc = Truncation {
        epsilon: 1e-13,
        max_bond: Some(max_bond),
    };
    let first = evolve(&model, TrotterOrder::First, beta, dtau, trunc.clone());
    let second = evolve(&model, TrotterOrder::Second, beta, dtau, trunc);

    for (name, value) in [
        ("first energy", first.energy),
        ("first free energy", first.free_energy),
        ("first specific heat", first.specific_heat),
        ("first magnetization", first.magnetization),
        ("second energy", second.energy),
        ("second free energy", second.free_energy),
        ("second specific heat", second.specific_heat),
        ("second magnetization", second.magnetization),
    ] {
        assert!(
            value.is_finite(),
            "{model_name} {name} is not finite: {value}"
        );
    }
    assert!(
        first.max_bond <= max_bond,
        "{model_name} first-order max bond {} exceeded cap {max_bond}",
        first.max_bond,
    );
    assert!(
        second.max_bond <= max_bond,
        "{model_name} second-order max bond {} exceeded cap {max_bond}",
        second.max_bond,
    );

    let first_energy_error = (first.energy - exact_energy).abs();
    let first_free_energy_error = (first.free_energy - exact_free_energy).abs();
    let second_energy_error = (second.energy - exact_energy).abs();
    let second_free_energy_error = (second.free_energy - exact_free_energy).abs();
    let second_specific_heat_error = (second.specific_heat - exact_specific_heat).abs();
    let second_magnetization_error = (second.magnetization - exact_magnetization).abs();

    println!(
        "{model_name}: dtau={:.3e}; first={{u={:.16e}, f={:.16e}, c={:.16e}, m={:.16e}, chi={}}}; second={{u={:.16e}, f={:.16e}, c={:.16e}, m={:.16e}, chi={}}}; exact={{u={:.16e}, f={:.16e}, c={:.16e}, m={:.16e}}}; errors={{first_u={:.3e}, first_f={:.3e}, second_u={:.3e}, second_f={:.3e}, second_c={:.3e}, second_m={:.3e}}}",
        second.dtau,
        first.energy,
        first.free_energy,
        first.specific_heat,
        first.magnetization,
        first.max_bond,
        second.energy,
        second.free_energy,
        second.specific_heat,
        second.magnetization,
        second.max_bond,
        exact_energy,
        exact_free_energy,
        exact_specific_heat,
        exact_magnetization,
        first_energy_error,
        first_free_energy_error,
        second_energy_error,
        second_free_energy_error,
        second_specific_heat_error,
        second_magnetization_error,
    );

    assert!(
        second_energy_error < first_energy_error,
        "{model_name} second-order energy was not more accurate: second={second_energy_error:.3e}, first={first_energy_error:.3e}",
    );
    assert!(
        second_free_energy_error < first_free_energy_error,
        "{model_name} second-order free energy was not more accurate: second={second_free_energy_error:.3e}, first={first_free_energy_error:.3e}",
    );
    assert!(
        second_energy_error < 5e-3,
        "{model_name} second-order energy error {second_energy_error:.3e} exceeds 5e-3",
    );
    assert!(
        second_free_energy_error < 5e-3,
        "{model_name} second-order free-energy error {second_free_energy_error:.3e} exceeds 5e-3",
    );
    assert!(
        second_magnetization_error < 5e-3,
        "{model_name} second-order magnetization error {second_magnetization_error:.3e} exceeds 5e-3",
    );
    assert!(
        second_specific_heat_error < 3e-2,
        "{model_name} second-order specific-heat error {second_specific_heat_error:.3e} exceeds 3e-2",
    );
}

#[test]
fn second_order_tfim_matches_exact_observables_and_improves_energy_and_free_energy() {
    let model = ModelSpec::Tfim { j: 1.0, g: 0.7 };
    let exact = model.exact(1.0).expect("TFIM has an exact reference");
    assert_second_order_matches_exact_and_improves(
        "TFIM (J=1, g=0.7)",
        model,
        exact.u,
        exact.f.unwrap(),
        exact.c,
        exact.magnetization,
    );
}

#[test]
fn second_order_xy_matches_exact_observables_and_improves_energy_and_free_energy() {
    let model = ModelSpec::Xy { gamma: 0.5, h: 0.7 };
    let exact = model.exact(1.0).expect("XY has an exact reference");
    assert_second_order_matches_exact_and_improves(
        "XY (gamma=0.5, h=0.7)",
        model,
        exact.u,
        exact.f.unwrap(),
        exact.c,
        exact.magnetization,
    );
}

#[test]
fn second_order_log_norm_is_thermodynamically_consistent() {
    let model = ModelSpec::Tfim { j: 1.0, g: 0.7 };
    let trunc = Truncation {
        epsilon: 1e-13,
        max_bond: Some(64),
    };
    let at_minus = evolve(&model, TrotterOrder::Second, 0.75, 0.0125, trunc.clone());
    let at_center = evolve(&model, TrotterOrder::Second, 0.80, 0.0125, trunc.clone());
    let at_plus = evolve(&model, TrotterOrder::Second, 0.85, 0.0125, trunc);

    let derivative = (0.85 * at_plus.free_energy - 0.75 * at_minus.free_energy) / 0.10;
    println!(
        "TFIM log-norm thermodynamics: derivative={derivative:.16e}, u(beta=0.80)={:.16e}, difference={:.3e}",
        at_center.energy,
        (derivative - at_center.energy).abs(),
    );
    assert!(
        (derivative - at_center.energy).abs() < 5e-3,
        "d(beta*f)/d beta={derivative:.16e} differs from u={:.16e}",
        at_center.energy,
    );
}

#[test]
#[ignore = "release-mode accuracy and refinement benchmark"]
fn benchmark_first_vs_second_order_accuracy() {
    let beta = 1.0;
    let fine_dtaus = [0.1, 0.05, 0.025, 0.0125];
    let xy_coarse_dtaus = [0.25, 0.125, 0.0625, 0.03125];
    let models = [
        ("tfim_j1_g0.7", ModelSpec::Tfim { j: 1.0, g: 0.7 }),
        ("xy_gamma0.5_h0.7", ModelSpec::Xy { gamma: 0.5, h: 0.7 }),
    ];

    for (model_name, model) in models {
        let exact = model.exact(beta).expect("benchmark model has exact data");
        let regimes = if model_name == "xy_gamma0.5_h0.7" {
            vec![
                ("fine_canonicalization_floor", fine_dtaus, false),
                ("xy_coarse_trotter_regime", xy_coarse_dtaus, true),
            ]
        } else {
            vec![("fine_trotter_regime", fine_dtaus, true)]
        };

        for (regime_name, dtaus, require_quadratic_window) in regimes {
            let matrices = if regime_name == "xy_coarse_trotter_regime" {
                [
                    ("xy_coarse_trotter_primary", 1.0e-14, 128),
                    ("xy_coarse_trotter_cutoff_sensitivity", 1.0e-13, 128),
                    ("xy_coarse_trotter_cap_sensitivity", 1.0e-14, 64),
                ]
            } else {
                [
                    ("primary", 1.0e-14, 128),
                    ("cutoff_sensitivity", 1.0e-13, 128),
                    ("cap_sensitivity", 1.0e-14, 64),
                ]
            };
            let mut primary_first_errors = None;
            let mut second_errors_by_matrix: [Option<Vec<[f64; 4]>>; 3] = [None, None, None];
            let mut second_bonds_by_matrix: [Option<Vec<usize>>; 3] = [None, None, None];

            for (matrix_index, (matrix_name, epsilon, cap)) in matrices.iter().enumerate() {
                for order in [TrotterOrder::First, TrotterOrder::Second] {
                    let order_name = match order {
                        TrotterOrder::First => "first",
                        TrotterOrder::Second => "second",
                    };
                    let trunc = Truncation {
                        epsilon: *epsilon,
                        max_bond: Some(*cap),
                    };
                    let mut errors = Vec::with_capacity(dtaus.len());
                    let mut bonds = Vec::with_capacity(dtaus.len());

                    for dtau in dtaus {
                        let sample = evolve(&model, order, beta, dtau, trunc.clone());
                        let sample_errors = [
                            (sample.energy - exact.u).abs(),
                            (sample.free_energy - exact.f.unwrap()).abs(),
                            (sample.specific_heat - exact.c).abs(),
                            (sample.magnetization - exact.magnetization).abs(),
                        ];
                        println!(
                            "ITEBD_TROTTER_ACCURACY matrix={matrix_name} model={model_name} order={order_name} beta={beta:.16e} dtau={dtau:.16e} epsilon={epsilon:.16e} cap={cap} energy_error={:.16e} free_energy_error={:.16e} specific_heat_error={:.16e} magnetization_error={:.16e} max_bond={}",
                            sample_errors[0],
                            sample_errors[1],
                            sample_errors[2],
                            sample_errors[3],
                            sample.max_bond,
                        );
                        if matrix_index == 0 {
                            assert!(
                                sample.max_bond < *cap,
                                "{regime_name} {model_name} {order_name} dtau={dtau:.4e} hit primary cap {cap}",
                            );
                        }
                        errors.push(sample_errors);
                        bonds.push(sample.max_bond);
                    }

                    for (window, pair) in errors.windows(2).enumerate() {
                        for (observable, error_index) in [
                            ("energy", 0),
                            ("free_energy", 1),
                            ("specific_heat", 2),
                            ("magnetization", 3),
                        ] {
                            println!(
                                "ITEBD_TROTTER_REFINEMENT matrix={matrix_name} model={model_name} order={order_name} observable={observable} dtau_coarse={:.16e} dtau_fine={:.16e} refinement_order={:.16e}",
                                dtaus[window],
                                dtaus[window + 1],
                                refinement_order(pair[0][error_index], pair[1][error_index]),
                            );
                        }
                    }

                    match order {
                        TrotterOrder::First if matrix_index == 0 => {
                            primary_first_errors = Some(errors)
                        }
                        TrotterOrder::Second => {
                            second_errors_by_matrix[matrix_index] = Some(errors);
                            second_bonds_by_matrix[matrix_index] = Some(bonds);
                        }
                        TrotterOrder::First => {}
                    }
                }
            }

            let first_errors = primary_first_errors
                .as_ref()
                .expect("primary first-order matrix was evaluated");
            let second_errors = second_errors_by_matrix[0]
                .as_ref()
                .expect("primary second-order matrix was evaluated");
            let first_finest = first_errors.last().expect("nonempty first-order matrix");
            let second_finest = second_errors.last().expect("nonempty second-order matrix");
            assert!(
                second_finest[0] < first_finest[0],
                "{regime_name} {model_name} finest second-order energy error {:.3e} was not below first-order {:.3e}",
                second_finest[0],
                first_finest[0],
            );
            assert!(
                second_finest[1] < first_finest[1],
                "{regime_name} {model_name} finest second-order free-energy error {:.3e} was not below first-order {:.3e}",
                second_finest[1],
                first_finest[1],
            );
            let second_energy_errors: Vec<_> =
                second_errors.iter().map(|errors| errors[0]).collect();
            let second_free_energy_errors: Vec<_> =
                second_errors.iter().map(|errors| errors[1]).collect();

            if !require_quadratic_window {
                println!(
                    "ITEBD_TROTTER_LIMITATION regime={regime_name} model={model_name} cause=every_step_canonicalization_floor energy_errors={second_energy_errors:?} free_energy_errors={second_free_energy_errors:?}",
                );
                continue;
            }

            assert!(
                has_quadratic_window(&second_energy_errors),
                "{regime_name} {model_name} second-order energy errors have no refinement window at order >= 1.6: {second_energy_errors:?}",
            );
            assert!(
                has_quadratic_window(&second_free_energy_errors),
                "{regime_name} {model_name} second-order free-energy errors have no refinement window at order >= 1.6: {second_free_energy_errors:?}",
            );

            if regime_name == "xy_coarse_trotter_regime" {
                let selected_window = second_errors
                    .windows(2)
                    .position(|pair| {
                        pair[0][0] > pair[1][0]
                            && refinement_order(pair[0][0], pair[1][0]) >= 1.6
                            && pair[0][1] > pair[1][1]
                            && refinement_order(pair[0][1], pair[1][1]) >= 1.6
                    })
                    .expect("XY coarse matrix has no common energy/free-energy quadratic window");
                println!(
                    "ITEBD_TROTTER_SELECTED_WINDOW regime={regime_name} model={model_name} dtau_coarse={:.16e} dtau_fine={:.16e} energy_order={:.16e} free_energy_order={:.16e}",
                    dtaus[selected_window],
                    dtaus[selected_window + 1],
                    refinement_order(
                        second_errors[selected_window][0],
                        second_errors[selected_window + 1][0],
                    ),
                    refinement_order(
                        second_errors[selected_window][1],
                        second_errors[selected_window + 1][1],
                    ),
                );

                for matrix_index in 1..matrices.len() {
                    let (matrix_name, _, cap) = matrices[matrix_index];
                    let sensitivity_errors = second_errors_by_matrix[matrix_index]
                        .as_ref()
                        .expect("second-order sensitivity matrix was evaluated");
                    let sensitivity_bonds = second_bonds_by_matrix[matrix_index]
                        .as_ref()
                        .expect("second-order sensitivity bonds were recorded");
                    for endpoint in selected_window..=selected_window + 1 {
                        assert!(
                            sensitivity_bonds[endpoint] < cap,
                            "{matrix_name} selected XY coarse window hit cap: dtau={:.16e} max_bond={} cap={cap}",
                            dtaus[endpoint],
                            sensitivity_bonds[endpoint],
                        );
                        for (observable, error_index) in [("energy", 0), ("free_energy", 1)] {
                            let primary_error = second_errors[endpoint][error_index];
                            let sensitivity_error = sensitivity_errors[endpoint][error_index];
                            let relative_change =
                                (sensitivity_error - primary_error).abs() / primary_error;
                            println!(
                                "ITEBD_TROTTER_SENSITIVITY matrix={matrix_name} model={model_name} observable={observable} dtau={:.16e} primary_error={primary_error:.16e} sensitivity_error={sensitivity_error:.16e} relative_change={relative_change:.16e} max_bond={} cap={cap}",
                                dtaus[endpoint],
                                sensitivity_bonds[endpoint],
                            );
                            assert!(
                                relative_change < 0.20,
                                "{matrix_name} selected XY coarse {observable} endpoint changed by >=20%: dtau={:.16e} primary_error={primary_error:.16e} sensitivity_error={sensitivity_error:.16e} relative_change={relative_change:.16e} max_bond={} cap={cap}",
                                dtaus[endpoint],
                                sensitivity_bonds[endpoint],
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "release-mode equal-step and equal-accuracy timing benchmark"]
fn benchmark_first_vs_second_order_timing() {
    let beta = 1.0;
    let equal_step_dtau = 0.025;
    let dtaus = [0.1, 0.05, 0.025, 0.0125];
    let epsilon = 1.0e-13;
    let cap = 64;
    let warmups = 2;
    let samples = 7;
    let trunc = Truncation {
        epsilon,
        max_bond: Some(cap),
    };
    let models = [
        ("tfim_j1_g0.7", ModelSpec::Tfim { j: 1.0, g: 0.7 }),
        ("xy_gamma0.5_h0.7", ModelSpec::Xy { gamma: 0.5, h: 0.7 }),
    ];
    let observables = ["energy", "free_energy", "specific_heat", "magnetization"];

    for (model_name, model) in models {
        let exact = model.exact(beta).expect("benchmark model has exact data");
        let exact_values = [exact.u, exact.f.unwrap(), exact.c, exact.magnetization];
        let evaluate_candidates = |regime: &'static str, grid: &[f64]| {
            let mut candidates: [Vec<EqualAccuracyCandidate>; 2] = [
                Vec::with_capacity(grid.len()),
                Vec::with_capacity(grid.len()),
            ];
            for (order_index, order) in [TrotterOrder::First, TrotterOrder::Second]
                .into_iter()
                .enumerate()
            {
                let order_name = if matches!(order, TrotterOrder::First) {
                    "first"
                } else {
                    "second"
                };
                for &dtau in grid {
                    let sample = evolve(&model, order, beta, dtau, trunc.clone());
                    let errors = [
                        (sample.energy - exact_values[0]).abs(),
                        (sample.free_energy - exact_values[1]).abs(),
                        (sample.specific_heat - exact_values[2]).abs(),
                        (sample.magnetization - exact_values[3]).abs(),
                    ];
                    assert!(
                        sample.max_bond <= cap,
                        "{regime} {model_name} accuracy candidate exceeded cap: order={order_name} dtau={dtau:.16e} max_bond={} cap={cap}",
                        sample.max_bond,
                    );
                    println!(
                        "ITEBD_TROTTER_TIMING_EQUAL_ACCURACY_CANDIDATE regime={regime} model={model_name} order={order_name} beta={beta:.16e} dtau={dtau:.16e} epsilon={epsilon:.16e} cap={cap} energy_error={:.16e} free_energy_error={:.16e} specific_heat_error={:.16e} magnetization_error={:.16e} max_bond={}",
                        errors[0], errors[1], errors[2], errors[3], sample.max_bond,
                    );
                    candidates[order_index].push(EqualAccuracyCandidate { dtau, errors });
                }
            }
            candidates
        };

        let fine_regime = if model_name == "xy_gamma0.5_h0.7" {
            "fine_canonicalization_floor"
        } else {
            "fine_trotter_regime"
        };
        let accuracy_candidates = evaluate_candidates(fine_regime, &dtaus);
        let coarse_accuracy_candidates = if model_name == "xy_gamma0.5_h0.7" {
            Some(evaluate_candidates(
                "xy_coarse_trotter_regime",
                &[0.25, 0.125, 0.0625, 0.03125],
            ))
        } else {
            None
        };

        let selections: Vec<_> = observables
            .iter()
            .enumerate()
            .map(|(observable_index, observable)| {
                let selection = select_required_equal_accuracy_pair(
                    fine_regime,
                    observable,
                    observable_index,
                    &accuracy_candidates[0],
                    &accuracy_candidates[1],
                    coarse_accuracy_candidates.as_ref().map(|coarse_candidates| {
                        (
                            coarse_candidates[0].as_slice(),
                            coarse_candidates[1].as_slice(),
                        )
                    }),
                );
                if observable_index < 2 && selection.is_none() {
                    println!(
                        "ITEBD_TROTTER_TIMING_EQUAL_ACCURACY model={model_name} observable={observable} selection=unavailable regime={fine_regime} beta={beta:.16e} epsilon={epsilon:.16e} cap={cap} first_candidates={:?} second_candidates={:?} coarse_candidates={coarse_accuracy_candidates:?}",
                        accuracy_candidates[0],
                        accuracy_candidates[1],
                    );
                    panic!(
                        "{model_name} {observable} has no equal-accuracy cross-order pair with error ratio in [0.5, 2.0]; fine_first_candidates={:?}; fine_second_candidates={:?}; coarse_candidates={coarse_accuracy_candidates:?}",
                        accuracy_candidates[0],
                        accuracy_candidates[1],
                    );
                }
                selection
            })
            .collect();

        let equal_step_first = TimingKey::new(TrotterOrder::First, equal_step_dtau);
        let equal_step_second = TimingKey::new(TrotterOrder::Second, equal_step_dtau);
        let mut workloads = vec![equal_step_first, equal_step_second];
        for selection in selections.iter().flatten() {
            workloads.push(TimingKey::new(TrotterOrder::First, selection.first_dtau));
            workloads.push(TimingKey::new(TrotterOrder::Second, selection.second_dtau));
        }
        let timing_cache =
            measure_timing_workloads(&model, beta, trunc.clone(), &workloads, warmups, samples);

        let first = timing_cache
            .get(&equal_step_first)
            .expect("equal-step first-order timing was cached");
        let second = timing_cache
            .get(&equal_step_second)
            .expect("equal-step second-order timing was cached");
        assert_eq!(first.beta.to_bits(), second.beta.to_bits());
        assert_eq!(first.beta.to_bits(), beta.to_bits());
        assert!(first.median > Duration::ZERO);
        assert!(second.median > Duration::ZERO);
        assert!(first.max_bond <= cap);
        assert!(second.max_bond <= cap);
        let first_per_step_ns = first.median.as_nanos() as f64 / first.steps as f64;
        let second_per_step_ns = second.median.as_nanos() as f64 / second.steps as f64;
        let equal_step_ratio = second.median.as_secs_f64() / first.median.as_secs_f64();
        println!(
            "ITEBD_TROTTER_TIMING_EQUAL_STEP model={model_name} beta={beta:.16e} dtau={equal_step_dtau:.16e} epsilon={epsilon:.16e} cap={cap} warmups={warmups} samples={samples} first_raw_ns={:?} second_raw_ns={:?} first_median_ns={} second_median_ns={} first_steps={} second_steps={} first_bond_updates={} second_bond_updates={} first_max_bond={} second_max_bond={} first_per_step_ns={first_per_step_ns:.16e} second_per_step_ns={second_per_step_ns:.16e} second_over_first={equal_step_ratio:.16e}",
            first.raw_ns,
            second.raw_ns,
            first.median.as_nanos(),
            second.median.as_nanos(),
            first.steps,
            second.steps,
            2 * first.steps,
            3 * second.steps,
            first.max_bond,
            second.max_bond,
        );

        for (observable, selection) in observables.iter().zip(selections.iter()) {
            let Some(selection) = selection else {
                println!(
                    "ITEBD_TROTTER_TIMING_EQUAL_ACCURACY model={model_name} observable={observable} selection=unavailable regime={fine_regime} beta={beta:.16e} epsilon={epsilon:.16e} cap={cap} first_candidates={:?} second_candidates={:?}",
                    accuracy_candidates[0],
                    accuracy_candidates[1],
                );
                continue;
            };
            assert_eq!(selection.observable, *observable);
            let error_ratio = selection.first_error / selection.second_error;
            assert!(
                (0.5..=2.0).contains(&error_ratio),
                "{model_name} {observable} invalid equal-accuracy ratio {error_ratio:.16e}"
            );

            let first_key = TimingKey::new(TrotterOrder::First, selection.first_dtau);
            let second_key = TimingKey::new(TrotterOrder::Second, selection.second_dtau);
            let first_timing = timing_cache
                .get(&first_key)
                .expect("selected first-order timing was cached");
            let second_timing = timing_cache
                .get(&second_key)
                .expect("selected second-order timing was cached");
            assert_eq!(first_timing.beta.to_bits(), second_timing.beta.to_bits());
            assert_eq!(first_timing.beta.to_bits(), beta.to_bits());
            assert!(first_timing.median > Duration::ZERO);
            assert!(second_timing.median > Duration::ZERO);
            assert!(first_timing.max_bond <= cap);
            assert!(second_timing.max_bond <= cap);
            println!(
                "{}",
                format_equal_accuracy_timing_record(
                    model_name,
                    selection,
                    beta,
                    epsilon,
                    cap,
                    warmups,
                    samples,
                    first_timing,
                    second_timing,
                )
            );
        }
    }
}

#[test]
fn refinement_order_reports_halving_convergence_rate() {
    assert!((refinement_order(4.0e-4, 1.0e-4) - 2.0).abs() < 1.0e-12);
}

#[test]
fn quadratic_window_requires_decreasing_errors_at_order_at_least_one_point_six() {
    assert!(has_quadratic_window(&[4.0e-4, 1.0e-4, 8.0e-5]));
    assert!(!has_quadratic_window(&[4.0e-4, 1.5e-4, 8.0e-5]));
    assert!(!has_quadratic_window(&[1.0e-4, 1.0e-4]));
}

#[test]
fn median_duration_sorts_samples_and_returns_the_middle_value() {
    let mut samples = [
        Duration::from_nanos(70),
        Duration::from_nanos(10),
        Duration::from_nanos(30),
        Duration::from_nanos(50),
        Duration::from_nanos(20),
    ];

    assert_eq!(median_duration(&mut samples), Duration::from_nanos(30));
}

#[test]
fn required_equal_accuracy_coarse_fallback_is_energy_only() {
    let fine_first = [EqualAccuracyCandidate {
        dtau: 0.1,
        errors: [4.0, 4.0, 4.0, 4.0],
    }];
    let fine_second = [EqualAccuracyCandidate {
        dtau: 0.1,
        errors: [1.0, 1.0, 1.0, 1.0],
    }];
    let coarse_first = [EqualAccuracyCandidate {
        dtau: 0.25,
        errors: [1.0, 1.0, 1.0, 1.0],
    }];
    let coarse_second = [EqualAccuracyCandidate {
        dtau: 0.25,
        errors: [1.0, 1.0, 1.0, 1.0],
    }];

    let energy = select_required_equal_accuracy_pair(
        "fine_canonicalization_floor",
        "energy",
        0,
        &fine_first,
        &fine_second,
        Some((&coarse_first, &coarse_second)),
    )
    .expect("energy may use the accepted coarse fallback");
    assert_eq!(energy.regime, "xy_coarse_trotter_regime");

    let free_energy = select_required_equal_accuracy_pair(
        "fine_canonicalization_floor",
        "free_energy",
        1,
        &fine_first,
        &fine_second,
        Some((&coarse_first, &coarse_second)),
    );
    assert!(
        free_energy.is_none(),
        "free energy must not use the coarse fallback"
    );
}

#[test]
fn equal_accuracy_record_includes_per_step_timings() {
    let selection = EqualAccuracySelection {
        regime: "fine_trotter_regime",
        observable: "energy",
        first_dtau: 0.05,
        second_dtau: 0.1,
        first_error: 1.0e-4,
        second_error: 1.5e-4,
    };
    let first = TimingSummary {
        raw_ns: vec![30],
        median: Duration::from_nanos(30),
        beta: 1.0,
        steps: 3,
        max_bond: 8,
    };
    let second = TimingSummary {
        raw_ns: vec![44],
        median: Duration::from_nanos(44),
        beta: 1.0,
        steps: 4,
        max_bond: 9,
    };

    let record = format_equal_accuracy_timing_record(
        "tfim_j1_g0.7",
        &selection,
        1.0,
        1.0e-13,
        64,
        2,
        7,
        &first,
        &second,
    );

    assert!(record.contains("first_per_step_ns=1.0000000000000000e1"));
    assert!(record.contains("second_per_step_ns=1.1000000000000000e1"));
}
