use thermal_imps_purification::canonicalize::canonicalize;
use thermal_imps_purification::config::TrotterOrder;
use thermal_imps_purification::exact::{exact_energy_density, exact_magnetization_x, free_energy_density};
use thermal_imps_purification::itebd::{
    free_energy_from_log_norm, imaginary_time_step, imaginary_time_step_second_order,
};
use thermal_imps_purification::itebd_auto::{
    canonicalize_auto, energy_density_auto, imaginary_time_step_auto,
    imaginary_time_step_second_order_auto, ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::itebd_complex::{
    canonicalize_complex, energy_density_complex, imaginary_time_step_complex,
    imaginary_time_step_second_order_complex, local_expectation_complex, ComplexLocalHamiltonian,
    ComplexPurifiedMps,
};
use thermal_imps_purification::model::{pauli_x, LocalHamiltonian, Tfim};
use thermal_imps_purification::observable::energy_density;
use thermal_imps_purification::purified_mps::infinite_temperature;
use thermal_imps_purification::tensor::Truncation;
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

const J: f64 = 1.0;
const G: f64 = 0.7;
const BETA: f64 = 1.0;
const EXACT_NK: usize = 16_000;

fn phase_rotated_tfim() -> ComplexLocalHamiltonian {
    let real = Tfim { j: J, g: G }.local();
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

fn phase_rotated_pauli_x() -> DMatrix<Complex64> {
    let real = pauli_x();
    let phase = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    DMatrix::from_fn(2, 2, |row, column| {
        phase[row] * real[(row, column)] * phase[column].conj()
    })
}

fn exact_real_facade_hamiltonian(real: &LocalHamiltonian) -> ItebdHamiltonian {
    let hamiltonian = ItebdHamiltonian::try_from_complex(
        real.two_site_h.map(|value| Complex64::new(value, 0.0)),
        real.site_energy.map(|value| Complex64::new(value, 0.0)),
    )
    .unwrap();
    assert!(matches!(hamiltonian, ItebdHamiltonian::Real(_)));
    hamiltonian
}

#[derive(Debug, Clone, Copy)]
struct ComplexSample {
    energy: f64,
    free_energy: f64,
    transformed_observable: f64,
    log_norm_per_site: f64,
    final_bond: usize,
    max_bond: usize,
}

fn evolve_complex(
    order: TrotterOrder,
    beta: f64,
    dtau: f64,
    truncation: Truncation,
) -> ComplexSample {
    let hamiltonian = phase_rotated_tfim();
    let operator = phase_rotated_pauli_x();
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!((2.0 * steps as f64 * dtau - beta).abs() <= 1e-12);
    let mut state = ComplexPurifiedMps::infinite_temperature(hamiltonian.dim()).unwrap();
    let mut accumulated_log_norm = 0.0;
    let mut max_bond = 1;

    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => {
                imaginary_time_step_complex(&mut state, &hamiltonian, dtau, &truncation)
            }
            TrotterOrder::Second => imaginary_time_step_second_order_complex(
                &mut state,
                &hamiltonian,
                dtau,
                &truncation,
            ),
        }
        .unwrap();
        assert!(info.log_norm.is_finite());
        assert!(info.min_singular_value.is_finite());
        max_bond = max_bond.max(info.max_bond);
        accumulated_log_norm += info.log_norm + canonicalize_complex(&mut state, 1e-12).unwrap();
    }

    let log_norm_per_site = accumulated_log_norm / 2.0;
    let sample = ComplexSample {
        energy: energy_density_complex(&state, &hamiltonian).unwrap(),
        free_energy: free_energy_from_log_norm(log_norm_per_site, beta, hamiltonian.dim()),
        transformed_observable: local_expectation_complex(&state, &operator, 1e-12).unwrap(),
        log_norm_per_site,
        final_bond: state.lambda_ab.len().max(state.lambda_ba.len()),
        max_bond,
    };
    for value in [
        sample.energy,
        sample.free_energy,
        sample.transformed_observable,
        sample.log_norm_per_site,
    ] {
        assert!(value.is_finite());
    }
    sample
}

fn order_name(order: TrotterOrder) -> &'static str {
    match order {
        TrotterOrder::First => "first",
        TrotterOrder::Second => "second",
    }
}

fn refinement_order(coarse: f64, fine: f64) -> f64 {
    (coarse / fine).ln() / 2.0_f64.ln()
}

#[derive(Debug, Clone, Copy)]
struct AccuracyRow {
    errors: [f64; 3],
    final_bond: usize,
    max_bond: usize,
}

#[test]
#[ignore = "release-mode complex accuracy, refinement, and sensitivity benchmark"]
fn benchmark_complex_accuracy() {
    let dtaus = [0.1, 0.05, 0.025, 0.0125];
    let matrices = [
        ("primary", 1.0e-14, 128),
        ("cutoff_sensitivity", 1.0e-13, 128),
        ("cap_sensitivity", 1.0e-14, 64),
    ];
    let exact = [
        exact_energy_density(J, G, BETA, EXACT_NK),
        free_energy_density(J, G, BETA, EXACT_NK),
        exact_magnetization_x(J, G, BETA, EXACT_NK),
    ];
    println!(
        "ITEBD_COMPLEX_EXACT model=phase_rotated_tfim j={J:.16e} g={G:.16e} beta={BETA:.16e} nk={EXACT_NK} energy={:.16e} free_energy={:.16e} transformed_observable={:.16e}",
        exact[0], exact[1], exact[2],
    );

    let mut second_order_rows: [Option<Vec<AccuracyRow>>; 3] = [None, None, None];
    for (matrix_index, (matrix_name, epsilon, cap)) in matrices.iter().enumerate() {
        for order in [TrotterOrder::First, TrotterOrder::Second] {
            let mut rows = Vec::with_capacity(dtaus.len());
            for dtau in dtaus {
                let sample = evolve_complex(
                    order,
                    BETA,
                    dtau,
                    Truncation {
                        epsilon: *epsilon,
                        max_bond: Some(*cap),
                    },
                );
                let row = AccuracyRow {
                    errors: [
                        (sample.energy - exact[0]).abs(),
                        (sample.free_energy - exact[1]).abs(),
                        (sample.transformed_observable - exact[2]).abs(),
                    ],
                    final_bond: sample.final_bond,
                    max_bond: sample.max_bond,
                };
                println!(
                    "ITEBD_COMPLEX_ACCURACY matrix={matrix_name} order={} beta={BETA:.16e} dtau={dtau:.16e} epsilon={epsilon:.16e} cap={cap} energy={:.16e} energy_error={:.16e} free_energy={:.16e} free_energy_error={:.16e} transformed_observable={:.16e} transformed_observable_error={:.16e} log_norm_per_site={:.16e} final_bond={} max_bond={}",
                    order_name(order),
                    sample.energy,
                    row.errors[0],
                    sample.free_energy,
                    row.errors[1],
                    sample.transformed_observable,
                    row.errors[2],
                    sample.log_norm_per_site,
                    row.final_bond,
                    row.max_bond,
                );
                rows.push(row);
            }
            for (window, pair) in rows.windows(2).enumerate() {
                for (observable, index) in [
                    ("energy", 0),
                    ("free_energy", 1),
                    ("transformed_observable", 2),
                ] {
                    println!(
                        "ITEBD_COMPLEX_REFINEMENT matrix={matrix_name} order={} observable={observable} dtau_coarse={:.16e} dtau_fine={:.16e} refinement_order={:.16e}",
                        order_name(order),
                        dtaus[window],
                        dtaus[window + 1],
                        refinement_order(pair[0].errors[index], pair[1].errors[index]),
                    );
                }
            }
            if matches!(order, TrotterOrder::Second) {
                second_order_rows[matrix_index] = Some(rows);
            }
        }
    }

    let primary = second_order_rows[0]
        .as_ref()
        .expect("primary second-order matrix was evaluated");
    let selected_window = primary
        .windows(2)
        .position(|pair| {
            (0..=1).all(|index| {
                pair[0].errors[index] > pair[1].errors[index]
                    && refinement_order(pair[0].errors[index], pair[1].errors[index]) >= 1.6
            })
        })
        .expect("no common second-order energy/free-energy refinement window at order >= 1.6");
    println!(
        "ITEBD_COMPLEX_SELECTED_WINDOW dtau_coarse={:.16e} dtau_fine={:.16e} energy_order={:.16e} free_energy_order={:.16e}",
        dtaus[selected_window],
        dtaus[selected_window + 1],
        refinement_order(
            primary[selected_window].errors[0],
            primary[selected_window + 1].errors[0],
        ),
        refinement_order(
            primary[selected_window].errors[1],
            primary[selected_window + 1].errors[1],
        ),
    );

    for matrix_index in 0..matrices.len() {
        let (matrix_name, _, cap) = matrices[matrix_index];
        let rows = second_order_rows[matrix_index]
            .as_ref()
            .expect("every second-order matrix was evaluated");
        for endpoint in selected_window..=selected_window + 1 {
            assert!(
                rows[endpoint].final_bond < cap && rows[endpoint].max_bond < cap,
                "{matrix_name} selected endpoint dtau={} hit cap {cap}: final_bond={} max_bond={}",
                dtaus[endpoint],
                rows[endpoint].final_bond,
                rows[endpoint].max_bond,
            );
            if matrix_index == 0 {
                continue;
            }
            for (observable, index) in [("energy", 0), ("free_energy", 1)] {
                let primary_error = primary[endpoint].errors[index];
                let sensitivity_error = rows[endpoint].errors[index];
                let relative_change = (sensitivity_error - primary_error).abs() / primary_error;
                println!(
                    "ITEBD_COMPLEX_SENSITIVITY matrix={matrix_name} observable={observable} dtau={:.16e} primary_error={primary_error:.16e} sensitivity_error={sensitivity_error:.16e} relative_change={relative_change:.16e} final_bond={} max_bond={} cap={cap}",
                    dtaus[endpoint], rows[endpoint].final_bond, rows[endpoint].max_bond,
                );
                assert!(
                    relative_change < 0.20,
                    "{matrix_name} selected {observable} endpoint changed by >=20% at dtau={}: {relative_change:.3e}",
                    dtaus[endpoint],
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TimingBackend {
    RealDirect,
    ExactRealFacade,
    ComplexDirect,
}

impl TimingBackend {
    fn name(self) -> &'static str {
        match self {
            Self::RealDirect => "real_direct",
            Self::ExactRealFacade => "exact_real_facade",
            Self::ComplexDirect => "complex_direct",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TimedRun {
    duration: Duration,
    steps: usize,
    final_bond: usize,
    max_bond: usize,
}

fn timed_real(
    order: TrotterOrder,
    hamiltonian: &LocalHamiltonian,
    dtau: f64,
    truncation: Truncation,
) -> TimedRun {
    let steps = (BETA / (2.0 * dtau)).round() as usize;
    let mut state = infinite_temperature(hamiltonian.dim());
    let mut log_norm = 0.0;
    let mut max_bond = 1;
    let start = Instant::now();
    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => imaginary_time_step(&mut state, hamiltonian, dtau, &truncation),
            TrotterOrder::Second => {
                imaginary_time_step_second_order(&mut state, hamiltonian, dtau, &truncation)
            }
        };
        max_bond = max_bond.max(info.max_bond);
        log_norm += info.log_norm + canonicalize(&mut state);
    }
    let duration = start.elapsed();
    black_box((energy_density(&state, hamiltonian), log_norm));
    TimedRun {
        duration,
        steps,
        final_bond: state.lambda_ab.len().max(state.lambda_ba.len()),
        max_bond,
    }
}

fn timed_auto(
    order: TrotterOrder,
    hamiltonian: &ItebdHamiltonian,
    dtau: f64,
    truncation: Truncation,
) -> TimedRun {
    let steps = (BETA / (2.0 * dtau)).round() as usize;
    let mut state = ItebdState::infinite_temperature(hamiltonian).unwrap();
    let mut log_norm = 0.0;
    let mut max_bond = 1;
    let start = Instant::now();
    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => {
                imaginary_time_step_auto(&mut state, hamiltonian, dtau, &truncation)
            }
            TrotterOrder::Second => {
                imaginary_time_step_second_order_auto(&mut state, hamiltonian, dtau, &truncation)
            }
        }
        .unwrap();
        max_bond = max_bond.max(info.max_bond);
        log_norm += info.log_norm + canonicalize_auto(&mut state, hamiltonian).unwrap();
    }
    let duration = start.elapsed();
    black_box((energy_density_auto(&state, hamiltonian).unwrap(), log_norm));
    let final_bond = match &state {
        ItebdState::Real(state) => state.lambda_ab.len().max(state.lambda_ba.len()),
        ItebdState::Complex(state) => state.lambda_ab.len().max(state.lambda_ba.len()),
    };
    TimedRun {
        duration,
        steps,
        final_bond,
        max_bond,
    }
}

fn timed_complex(
    order: TrotterOrder,
    hamiltonian: &ComplexLocalHamiltonian,
    dtau: f64,
    truncation: Truncation,
) -> TimedRun {
    let steps = (BETA / (2.0 * dtau)).round() as usize;
    let mut state = ComplexPurifiedMps::infinite_temperature(hamiltonian.dim()).unwrap();
    let mut log_norm = 0.0;
    let mut max_bond = 1;
    let start = Instant::now();
    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => {
                imaginary_time_step_complex(&mut state, hamiltonian, dtau, &truncation)
            }
            TrotterOrder::Second => {
                imaginary_time_step_second_order_complex(&mut state, hamiltonian, dtau, &truncation)
            }
        }
        .unwrap();
        max_bond = max_bond.max(info.max_bond);
        log_norm += info.log_norm + canonicalize_complex(&mut state, 1e-12).unwrap();
    }
    let duration = start.elapsed();
    black_box((
        energy_density_complex(&state, hamiltonian).unwrap(),
        log_norm,
    ));
    TimedRun {
        duration,
        steps,
        final_bond: state.lambda_ab.len().max(state.lambda_ba.len()),
        max_bond,
    }
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

#[test]
#[ignore = "release-mode informational timing benchmark"]
fn benchmark_complex_timing() {
    let dtau = 0.025;
    let truncation = Truncation {
        epsilon: 1e-13,
        max_bond: Some(64),
    };
    let warmups = 2;
    let samples = 7;
    let real_hamiltonian = Tfim { j: J, g: G }.local();
    let facade_hamiltonian = exact_real_facade_hamiltonian(&real_hamiltonian);
    let complex_hamiltonian = phase_rotated_tfim();
    let backends = [
        TimingBackend::RealDirect,
        TimingBackend::ExactRealFacade,
        TimingBackend::ComplexDirect,
    ];
    let run = |backend, order| match backend {
        TimingBackend::RealDirect => timed_real(order, &real_hamiltonian, dtau, truncation),
        TimingBackend::ExactRealFacade => timed_auto(order, &facade_hamiltonian, dtau, truncation),
        TimingBackend::ComplexDirect => {
            timed_complex(order, &complex_hamiltonian, dtau, truncation)
        }
    };

    for warmup in 0..warmups {
        let orders = if warmup % 2 == 0 {
            [TrotterOrder::First, TrotterOrder::Second]
        } else {
            [TrotterOrder::Second, TrotterOrder::First]
        };
        for order in orders {
            for backend in backends {
                black_box(run(backend, order));
            }
        }
    }

    let mut records: BTreeMap<(TimingBackend, bool), (Vec<Duration>, usize, usize, usize)> =
        BTreeMap::new();
    for sample_index in 0..samples {
        let orders = if sample_index % 2 == 0 {
            [TrotterOrder::First, TrotterOrder::Second]
        } else {
            [TrotterOrder::Second, TrotterOrder::First]
        };
        let backend_order = if sample_index % 2 == 0 {
            backends
        } else {
            [backends[2], backends[1], backends[0]]
        };
        for order in orders {
            for backend in backend_order {
                let timed = run(backend, order);
                assert!(timed.duration > Duration::ZERO);
                let entry = records
                    .entry((backend, matches!(order, TrotterOrder::Second)))
                    .or_insert_with(|| (Vec::with_capacity(samples), timed.steps, 1, 1));
                assert_eq!(entry.1, timed.steps);
                entry.2 = entry.2.max(timed.final_bond);
                entry.3 = entry.3.max(timed.max_bond);
                entry.0.push(timed.duration);
            }
        }
    }

    for backend in backends {
        for order in [TrotterOrder::First, TrotterOrder::Second] {
            let (durations, steps, final_bond, max_bond) = records
                .get(&(backend, matches!(order, TrotterOrder::Second)))
                .expect("every timing workload was sampled");
            assert_eq!(durations.len(), samples);
            let raw_ns: Vec<_> = durations.iter().map(Duration::as_nanos).collect();
            let median = median(durations);
            println!(
                "ITEBD_COMPLEX_TIMING backend={} order={} beta={BETA:.16e} dtau={dtau:.16e} epsilon={:.16e} cap={} warmups={warmups} samples={samples} raw_ns={raw_ns:?} median_ns={} steps={steps} bond_updates={} final_bond={final_bond} max_bond={max_bond}",
                backend.name(),
                order_name(order),
                truncation.epsilon,
                truncation.max_bond.unwrap(),
                median.as_nanos(),
                match order {
                    TrotterOrder::First => 2 * steps,
                    TrotterOrder::Second => 3 * steps,
                },
            );
        }
    }
}
