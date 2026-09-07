use thermal_imps_purification::exact::{
    exact_energy_density, exact_specific_heat, xy_energy_density, xy_specific_heat,
};
use thermal_imps_purification::itebd_complex::{
    canonicalize_complex, energy_density_complex, imaginary_time_step_second_order_complex,
    specific_heat_complex_with_options, ComplexLocalHamiltonian, ComplexPurifiedMps,
};
use thermal_imps_purification::model::{Tfim, Xy};
use thermal_imps_purification::specific_heat::{SpecificHeatOptions, SpecificHeatReport};
use thermal_imps_purification::tensor::Truncation;
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Instant;

const EXACT_NK: usize = 32_000;
const DIRECTION_TOTAL_TOLERANCE: f64 = 2e-10;

#[derive(Clone, Copy)]
struct Model {
    name: &'static str,
    beta: f64,
    hamiltonian: fn() -> ComplexLocalHamiltonian,
    exact_heat: fn(f64, usize) -> f64,
    exact_energy: fn(f64, usize) -> f64,
}

#[derive(Clone, Copy)]
struct TailVariant {
    name: &'static str,
    options: SpecificHeatOptions,
}

#[derive(Clone)]
struct Sample {
    model: &'static str,
    matrix: &'static str,
    tail: TailVariant,
    beta: f64,
    dtau: f64,
    cutoff: f64,
    cap: usize,
    heat: f64,
    variance: f64,
    energy: f64,
    exact_heat: f64,
    exact_energy: f64,
    derivative_one: f64,
    derivative_two: f64,
    report: SpecificHeatReport,
    final_bond: usize,
    max_bond: usize,
    elapsed_ns: u128,
}

struct BlockedSample {
    model: &'static str,
    matrix: &'static str,
    tail: TailVariant,
    beta: f64,
    dtau: f64,
    cutoff: f64,
    cap: usize,
    energy: f64,
    exact_heat: f64,
    exact_energy: f64,
    derivative_one: f64,
    derivative_two: f64,
    final_bond: usize,
    max_bond: usize,
    elapsed_ns: u128,
    error: String,
}

fn phase_rotated_tfim() -> ComplexLocalHamiltonian {
    let real = Tfim { j: 1.0, g: 0.7 }.local();
    let phase = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let rotate = |matrix: &DMatrix<f64>| {
        let mut rotated = DMatrix::<Complex64>::zeros(4, 4);
        for s2 in 0..2 {
            for s1 in 0..2 {
                for t2 in 0..2 {
                    for t1 in 0..2 {
                        let row = s1 + 2 * s2;
                        let column = t1 + 2 * t2;
                        rotated[(row, column)] = phase[s1]
                            * phase[s2]
                            * matrix[(row, column)]
                            * phase[t1].conj()
                            * phase[t2].conj();
                    }
                }
            }
        }
        rotated
    };
    ComplexLocalHamiltonian::try_new(rotate(&real.two_site_h), rotate(&real.site_energy)).unwrap()
}

fn phase_twisted_xx() -> ComplexLocalHamiltonian {
    let real = Xy { gamma: 0.0, h: 0.4 }.local();
    let twist = |matrix: &DMatrix<f64>| {
        let mut twisted = matrix.map(|value| Complex64::new(value, 0.0));
        twisted[(1, 2)] = -Complex64::from_polar(1.0, -PI / 5.0);
        twisted[(2, 1)] = twisted[(1, 2)].conj();
        twisted
    };
    ComplexLocalHamiltonian::try_new(twist(&real.two_site_h), twist(&real.site_energy)).unwrap()
}

fn tfim_exact_heat(beta: f64, nk: usize) -> f64 {
    exact_specific_heat(1.0, 0.7, beta, nk)
}

fn tfim_exact_energy(beta: f64, nk: usize) -> f64 {
    exact_energy_density(1.0, 0.7, beta, nk)
}

fn xx_exact_heat(beta: f64, nk: usize) -> f64 {
    xy_specific_heat(0.0, 0.4, beta, nk)
}

fn xx_exact_energy(beta: f64, nk: usize) -> f64 {
    xy_energy_density(0.0, 0.4, beta, nk)
}

fn evolve(
    hamiltonian: &ComplexLocalHamiltonian,
    beta: f64,
    dtau: f64,
    truncation: &Truncation,
) -> (ComplexPurifiedMps, usize, usize) {
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!(
        (2.0 * steps as f64 * dtau - beta).abs() <= 1e-12,
        "beta={beta} is not an integer number of complete second-order steps at dtau={dtau}"
    );
    let mut state = ComplexPurifiedMps::infinite_temperature(hamiltonian.dim()).unwrap();
    let mut max_bond = 1;
    for _ in 0..steps {
        let info =
            imaginary_time_step_second_order_complex(&mut state, hamiltonian, dtau, truncation)
                .unwrap();
        assert!(info.log_norm.is_finite());
        assert!(info.min_singular_value.is_finite());
        max_bond = max_bond.max(info.max_bond);
        canonicalize_complex(&mut state, 1e-12).unwrap();
    }
    let final_bond = state.lambda_ab.len().max(state.lambda_ba.len());
    (state, final_bond, max_bond)
}

fn derivative_heat(
    hamiltonian: &ComplexLocalHamiltonian,
    beta: f64,
    dtau: f64,
    truncation: &Truncation,
    steps_per_side: usize,
) -> f64 {
    let delta_beta = 2.0 * steps_per_side as f64 * dtau;
    let energy = |sample_beta| {
        let (state, _, _) = evolve(hamiltonian, sample_beta, dtau, truncation);
        energy_density_complex(&state, hamiltonian).unwrap()
    };
    -beta * beta * (energy(beta + delta_beta) - energy(beta - delta_beta)) / (2.0 * delta_beta)
}

fn assert_report_invariants(sample: &Sample) {
    for value in [
        sample.heat,
        sample.variance,
        sample.energy,
        sample.exact_heat,
        sample.exact_energy,
        sample.derivative_one,
        sample.derivative_two,
        sample.report.raw_energy_variance_per_site,
        sample.report.last_positive_shell_magnitude,
        sample.report.last_negative_shell_magnitude,
        sample.report.max_imaginary_residual,
    ] {
        assert!(
            value.is_finite(),
            "non-finite benchmark result for {}",
            sample.model
        );
    }
    let by_directions = sample.report.onsite_contribution
        + sample.report.positive_direction_contribution
        + sample.report.negative_direction_contribution;
    let by_parity = sample.report.parity_a_contribution + sample.report.parity_b_contribution;
    let expected = Complex64::new(sample.report.raw_energy_variance_per_site, 0.0);
    let scale = 1.0 + expected.norm();
    assert!((by_directions - expected).norm() <= 1e-10 * scale);
    assert!((by_parity - expected).norm() <= 1e-10 * scale);
    assert!(sample.variance >= 0.0);
    assert!(sample.report.max_imaginary_residual <= 1e-10 * scale);
}

fn evaluate(
    model: Model,
    matrix: &'static str,
    beta: f64,
    dtau: f64,
    cutoff: f64,
    cap: usize,
    tail: TailVariant,
    derivatives: Option<(f64, f64)>,
) -> Result<Sample, BlockedSample> {
    let hamiltonian = (model.hamiltonian)();
    let truncation = Truncation {
        epsilon: cutoff,
        max_bond: Some(cap),
    };
    let (state, final_bond, max_bond) = evolve(&hamiltonian, beta, dtau, &truncation);
    let energy = energy_density_complex(&state, &hamiltonian).unwrap();
    let start = Instant::now();
    let estimate = black_box(specific_heat_complex_with_options(
        &state,
        &hamiltonian,
        beta,
        &tail.options,
    ));
    let elapsed_ns = start.elapsed().as_nanos();
    let (derivative_one, derivative_two) = derivatives.unwrap_or_else(|| {
        (
            derivative_heat(&hamiltonian, beta, dtau, &truncation, 1),
            derivative_heat(&hamiltonian, beta, dtau, &truncation, 2),
        )
    });
    let exact_heat = (model.exact_heat)(beta, EXACT_NK);
    let exact_energy = (model.exact_energy)(beta, EXACT_NK);
    let report = match estimate {
        Ok(report) => report,
        Err(error) => {
            return Err(BlockedSample {
                model: model.name,
                matrix,
                tail,
                beta,
                dtau,
                cutoff,
                cap,
                energy,
                exact_heat,
                exact_energy,
                derivative_one,
                derivative_two,
                final_bond,
                max_bond,
                elapsed_ns,
                error: error.to_string(),
            });
        }
    };
    let sample = Sample {
        model: model.name,
        matrix,
        tail,
        beta,
        dtau,
        cutoff,
        cap,
        heat: report.specific_heat_per_site,
        variance: report.energy_variance_per_site,
        energy,
        exact_heat,
        exact_energy,
        derivative_one,
        derivative_two,
        report,
        final_bond,
        max_bond,
        elapsed_ns,
    };
    assert_report_invariants(&sample);
    Ok(sample)
}

fn print_sample(sample: &Sample) {
    let report = &sample.report;
    println!(
        "ITEBD_COMPLEX_SPECIFIC_HEAT model={} matrix={} tail={} beta={:.16e} dtau={:.16e} cutoff={:.16e} cap={} tail_relative={:.16e} tail_absolute={:.16e} tail_max={} tail_consecutive={} tail_reality={:.16e} heat={:.16e} variance={:.16e} energy={:.16e} exact_heat={:.16e} heat_error={:.16e} exact_energy={:.16e} energy_error={:.16e} derivative_one_heat={:.16e} derivative_one_error={:.16e} derivative_two_heat={:.16e} derivative_two_error={:.16e} onsite_re={:.16e} onsite_im={:.16e} positive_re={:.16e} positive_im={:.16e} negative_re={:.16e} negative_im={:.16e} parity_a_re={:.16e} parity_a_im={:.16e} parity_b_re={:.16e} parity_b_im={:.16e} direction_total_difference={:.16e} max_distance={} last_positive_shell={:.16e} last_negative_shell={:.16e} max_imaginary_residual={:.16e} final_bond={} max_bond={} elapsed_ns={}",
        sample.model,
        sample.matrix,
        sample.tail.name,
        sample.beta,
        sample.dtau,
        sample.cutoff,
        sample.cap,
        sample.tail.options.relative_tolerance,
        sample.tail.options.absolute_tolerance,
        sample.tail.options.max_distance,
        sample.tail.options.consecutive_small_shells,
        sample.tail.options.reality_tolerance,
        sample.heat,
        sample.variance,
        sample.energy,
        sample.exact_heat,
        (sample.heat - sample.exact_heat).abs(),
        sample.exact_energy,
        (sample.energy - sample.exact_energy).abs(),
        sample.derivative_one,
        (sample.derivative_one - sample.exact_heat).abs(),
        sample.derivative_two,
        (sample.derivative_two - sample.exact_heat).abs(),
        report.onsite_contribution.re,
        report.onsite_contribution.im,
        report.positive_direction_contribution.re,
        report.positive_direction_contribution.im,
        report.negative_direction_contribution.re,
        report.negative_direction_contribution.im,
        report.parity_a_contribution.re,
        report.parity_a_contribution.im,
        report.parity_b_contribution.re,
        report.parity_b_contribution.im,
        (report.positive_direction_contribution - report.negative_direction_contribution).norm(),
        report.max_distance,
        report.last_positive_shell_magnitude,
        report.last_negative_shell_magnitude,
        report.max_imaginary_residual,
        sample.final_bond,
        sample.max_bond,
        sample.elapsed_ns,
    );
}

fn print_blocked(sample: &BlockedSample) {
    println!(
        "ITEBD_COMPLEX_SPECIFIC_HEAT model={} matrix={} tail={} estimator_status=BLOCKED beta={:.16e} dtau={:.16e} cutoff={:.16e} cap={} tail_relative={:.16e} tail_absolute={:.16e} tail_max={} tail_consecutive={} tail_reality={:.16e} energy={:.16e} exact_heat={:.16e} exact_energy={:.16e} energy_error={:.16e} derivative_one_heat={:.16e} derivative_one_error={:.16e} derivative_two_heat={:.16e} derivative_two_error={:.16e} final_bond={} max_bond={} elapsed_ns={} error={}",
        sample.model,
        sample.matrix,
        sample.tail.name,
        sample.beta,
        sample.dtau,
        sample.cutoff,
        sample.cap,
        sample.tail.options.relative_tolerance,
        sample.tail.options.absolute_tolerance,
        sample.tail.options.max_distance,
        sample.tail.options.consecutive_small_shells,
        sample.tail.options.reality_tolerance,
        sample.energy,
        sample.exact_heat,
        sample.exact_energy,
        (sample.energy - sample.exact_energy).abs(),
        sample.derivative_one,
        (sample.derivative_one - sample.exact_heat).abs(),
        sample.derivative_two,
        (sample.derivative_two - sample.exact_heat).abs(),
        sample.final_bond,
        sample.max_bond,
        sample.elapsed_ns,
        sample.error,
    );
}

fn find_sample<'a>(
    samples: &'a [Sample],
    model: &str,
    matrix: &str,
    dtau: f64,
) -> Option<&'a Sample> {
    samples.iter().find(|sample| {
        sample.model == model
            && sample.matrix == matrix
            && sample.tail.name == "default"
            && sample.dtau == dtau
    })
}

fn assert_selected_window(model: &str, selected_dtau: Option<f64>, tail_passes: bool) {
    assert!(
        selected_dtau.is_some(),
        "{model} has no primary window satisfying the fixed pre-tail acceptance criteria"
    );
    assert!(
        tail_passes,
        "{model} selected endpoint failed the fixed tail acceptance criterion"
    );
}

#[test]
fn selected_window_guard_rejects_missing_or_tail_failed_selection() {
    // Mutations caught: removing either required fixed-criterion assertion lets an ignored
    // scientific matrix report BLOCKED while still passing its Rust test.
    assert!(std::panic::catch_unwind(|| {
        assert_selected_window("control", None, true);
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| {
        assert_selected_window("control", Some(0.05), false);
    })
    .is_err());
    assert_selected_window("control", Some(0.05), true);
}

#[test]
#[ignore = "release-mode complex specific-heat refinement and sensitivity matrix"]
fn benchmark_complex_specific_heat_validation() {
    let dtaus = [0.1, 0.05, 0.025, 0.0125];
    let truncations = [
        ("primary", 1e-14, 128),
        ("cutoff_sensitivity", 1e-13, 128),
        ("cap_sensitivity", 1e-14, 64),
    ];
    assert_eq!(dtaus.len(), 4);
    assert_eq!(truncations.len(), 3);
    let default_tail = TailVariant {
        name: "default",
        options: SpecificHeatOptions::default(),
    };
    let tail_variants = [
        default_tail,
        TailVariant {
            name: "strict",
            options: SpecificHeatOptions {
                relative_tolerance: 3e-7,
                absolute_tolerance: 3e-13,
                max_distance: 300,
                ..SpecificHeatOptions::default()
            },
        },
        TailVariant {
            name: "relaxed",
            options: SpecificHeatOptions {
                relative_tolerance: 3e-6,
                absolute_tolerance: 3e-12,
                max_distance: 120,
                ..SpecificHeatOptions::default()
            },
        },
    ];
    let models = [
        Model {
            name: "phase_rotated_tfim",
            beta: 1.0,
            hamiltonian: phase_rotated_tfim,
            exact_heat: tfim_exact_heat,
            exact_energy: tfim_exact_energy,
        },
        Model {
            name: "phase_twisted_xx",
            beta: 0.6,
            hamiltonian: phase_twisted_xx,
            exact_heat: xx_exact_heat,
            exact_energy: xx_exact_energy,
        },
    ];

    for variant in tail_variants {
        println!(
            "ITEBD_COMPLEX_SPECIFIC_HEAT_TAIL_OPTION tail={} relative={:.16e} absolute={:.16e} max_distance={} consecutive_small_shells={} reality_tolerance={:.16e}",
            variant.name,
            variant.options.relative_tolerance,
            variant.options.absolute_tolerance,
            variant.options.max_distance,
            variant.options.consecutive_small_shells,
            variant.options.reality_tolerance,
        );
    }

    let mut samples = Vec::new();
    for model in models {
        println!(
            "ITEBD_COMPLEX_SPECIFIC_HEAT_EXACT model={} beta={:.16e} nk={} exact_heat={:.16e} exact_energy={:.16e}",
            model.name,
            model.beta,
            EXACT_NK,
            (model.exact_heat)(model.beta, EXACT_NK),
            (model.exact_energy)(model.beta, EXACT_NK),
        );
        for (matrix, cutoff, cap) in truncations {
            for dtau in dtaus {
                match evaluate(
                    model,
                    matrix,
                    model.beta,
                    dtau,
                    cutoff,
                    cap,
                    default_tail,
                    None,
                ) {
                    Ok(sample) => {
                        print_sample(&sample);
                        samples.push(sample);
                    }
                    Err(blocked) => print_blocked(&blocked),
                }
            }
        }
    }

    for model in models {
        let mut selected = None;
        for pair in dtaus.windows(2) {
            let (Some(coarse), Some(fine), Some(cutoff), Some(cap)) = (
                find_sample(&samples, model.name, "primary", pair[0]),
                find_sample(&samples, model.name, "primary", pair[1]),
                find_sample(&samples, model.name, "cutoff_sensitivity", pair[1]),
                find_sample(&samples, model.name, "cap_sensitivity", pair[1]),
            ) else {
                println!(
                    "ITEBD_COMPLEX_SPECIFIC_HEAT_WINDOW model={} dtau_coarse={:.16e} dtau_fine={:.16e} status=BLOCKED reason=estimator_did_not_return_a_finite_reality_validated_report",
                    model.name,
                    pair[0],
                    pair[1],
                );
                break;
            };
            let coarse_error = (coarse.heat - coarse.exact_heat).abs();
            let fine_error = (fine.heat - fine.exact_heat).abs();
            let p = (coarse_error / fine_error).ln() / 2.0_f64.ln();
            let cutoff_change = (cutoff.heat - fine.heat).abs() / fine_error;
            let cap_change = (cap.heat - fine.heat).abs() / fine_error;
            let derivative_agreement = (fine.derivative_one - fine.derivative_two).abs();
            let derivative_limit = (fine.derivative_one - fine.exact_heat)
                .abs()
                .max((fine.derivative_two - fine.exact_heat).abs());
            let candidate = fine_error < coarse_error
                && p >= 1.6
                && cutoff_change <= 0.05
                && cap_change <= 0.05
                && derivative_agreement <= derivative_limit;
            println!(
                "ITEBD_COMPLEX_SPECIFIC_HEAT_WINDOW model={} dtau_coarse={:.16e} dtau_fine={:.16e} coarse_error={:.16e} fine_error={:.16e} refinement_order={:.16e} cutoff_endpoint_change_fraction={:.16e} cap_endpoint_change_fraction={:.16e} derivative_difference={:.16e} derivative_limit={:.16e} pre_tail_status={}",
                model.name,
                pair[0],
                pair[1],
                coarse_error,
                fine_error,
                p,
                cutoff_change,
                cap_change,
                derivative_agreement,
                derivative_limit,
                if candidate { "CANDIDATE" } else { "REJECTED" },
            );
            if selected.is_none() && candidate {
                selected = Some(pair[1]);
            }
        }

        let Some(selected_dtau) = selected else {
            println!(
                "ITEBD_COMPLEX_SPECIFIC_HEAT_SELECTION model={} status=BLOCKED reason=no_primary_window_satisfies_fixed_pre_tail_criteria",
                model.name
            );
            assert_selected_window(model.name, None, false);
            unreachable!("the selection guard must panic when no window is available");
        };
        let primary = find_sample(&samples, model.name, "primary", selected_dtau)
            .expect("selected primary endpoint remains available")
            .clone();
        let derivative = (primary.derivative_one, primary.derivative_two);
        let mut tail_passes = true;
        for tail in tail_variants.into_iter().skip(1) {
            match evaluate(
                model,
                "primary",
                model.beta,
                selected_dtau,
                1e-14,
                128,
                tail,
                Some(derivative),
            ) {
                Ok(tail_sample) => {
                    let tail_change = (tail_sample.heat - primary.heat).abs();
                    tail_passes &= tail_change < (primary.heat - primary.exact_heat).abs();
                    println!(
                        "ITEBD_COMPLEX_SPECIFIC_HEAT_TAIL_CHANGE model={} dtau={:.16e} tail={} change={:.16e} selected_trotter_error={:.16e} status={}",
                        model.name,
                        selected_dtau,
                        tail.name,
                        tail_change,
                        (primary.heat - primary.exact_heat).abs(),
                        if tail_change < (primary.heat - primary.exact_heat).abs() {
                            "PASS"
                        } else {
                            "FAIL"
                        },
                    );
                    print_sample(&tail_sample);
                }
                Err(blocked) => {
                    tail_passes = false;
                    println!(
                        "ITEBD_COMPLEX_SPECIFIC_HEAT_TAIL_CHANGE model={} dtau={:.16e} tail={} status=BLOCKED reason=estimator_did_not_return_a_finite_reality_validated_report",
                        model.name,
                        selected_dtau,
                        tail.name,
                    );
                    print_blocked(&blocked);
                }
            }
        }
        println!(
            "ITEBD_COMPLEX_SPECIFIC_HEAT_SELECTION model={} status={} dtau_fine={:.16e} tail_criterion={}",
            model.name,
            if tail_passes { "SELECTED" } else { "BLOCKED" },
            selected_dtau,
            if tail_passes { "PASS" } else { "FAIL" },
        );
        assert_selected_window(model.name, Some(selected_dtau), tail_passes);
    }

    for sample in samples
        .iter()
        .filter(|sample| sample.model == "phase_twisted_xx")
    {
        let difference = (sample.report.positive_direction_contribution
            - sample.report.negative_direction_contribution)
            .norm();
        assert!(
            difference <= DIRECTION_TOTAL_TOLERANCE,
            "phase-twisted XX pure-gauge direction control exceeded its bound: {difference}"
        );
    }
}
