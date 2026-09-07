use thermal_imps_purification::canonicalize::canonicalize;
use thermal_imps_purification::exact::{exact_specific_heat, xy_specific_heat};
use thermal_imps_purification::itebd::imaginary_time_step_second_order;
use thermal_imps_purification::itebd_auto::{
    specific_heat_auto, specific_heat_with_options_auto, ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::itebd_complex::{
    canonicalize_complex, energy_variance_per_site_complex,
    imaginary_time_step_second_order_complex, specific_heat_complex,
    specific_heat_complex_with_options, ComplexLocalHamiltonian, ComplexPurifiedMps, ComplexSite,
};
use thermal_imps_purification::itebd_error::ItebdError;
use thermal_imps_purification::model::{LocalHamiltonian, Tfim, Xy};
use thermal_imps_purification::purified_mps::{infinite_temperature, PurifiedMps};
use thermal_imps_purification::specific_heat::{SpecificHeatOptions, SpecificHeatReport};
use thermal_imps_purification::tensor::{new_index, Tensor, Truncation};
use thermal_imps_purification::variance::{
    energy_variance_per_site, specific_heat, specific_heat_with_options,
};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::f64::consts::PI;

const ROUTINE_TFIM_TROTTER_HEAT_TOLERANCE: f64 = 5e-5;
const ROUTINE_TWISTED_XX_TROTTER_HEAT_TOLERANCE: f64 = 5e-5;
const UNITARY_COVARIANCE_TOLERANCE: f64 = 2e-10;
const ROUTINE_XX_BACKEND_COVARIANCE_TOLERANCE: f64 = 1e-8;
const TWISTED_XX_DIRECTION_EQUALITY_TOLERANCE: f64 = 2e-10;

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

fn twisted_xx_matrices(
    phi: f64,
    longitudinal_field: f64,
) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let real = Xy {
        gamma: 0.0,
        h: longitudinal_field,
    }
    .local();
    let twist = |matrix: &DMatrix<f64>| {
        let mut twisted = matrix.map(|value| Complex64::new(value, 0.0));
        twisted[(1, 2)] = -Complex64::from_polar(1.0, -phi);
        twisted[(2, 1)] = twisted[(1, 2)].conj();
        twisted
    };
    (twist(&real.two_site_h), twist(&real.site_energy))
}

fn twisted_xx(phi: f64) -> ComplexLocalHamiltonian {
    let (two_site_h, site_energy) = twisted_xx_matrices(phi, 0.4);
    ComplexLocalHamiltonian::try_new(two_site_h, site_energy).unwrap()
}

fn evolve_real_second_order(
    hamiltonian: &LocalHamiltonian,
    beta: f64,
    dtau: f64,
    truncation: &Truncation,
) -> PurifiedMps {
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!((2.0 * steps as f64 * dtau - beta).abs() <= 1e-12);
    let mut state = infinite_temperature(hamiltonian.dim());
    for _ in 0..steps {
        imaginary_time_step_second_order(&mut state, hamiltonian, dtau, truncation);
        canonicalize(&mut state);
    }
    state
}

fn evolve_complex_second_order(
    hamiltonian: &ComplexLocalHamiltonian,
    beta: f64,
    dtau: f64,
    truncation: &Truncation,
) -> ComplexPurifiedMps {
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!((2.0 * steps as f64 * dtau - beta).abs() <= 1e-12);
    let mut state = ComplexPurifiedMps::infinite_temperature(hamiltonian.dim()).unwrap();
    for _ in 0..steps {
        imaginary_time_step_second_order_complex(&mut state, hamiltonian, dtau, truncation)
            .unwrap();
        canonicalize_complex(&mut state, 1e-12).unwrap();
    }
    state
}

fn assert_hermitian(matrix: &DMatrix<Complex64>) {
    let residual = (matrix - matrix.adjoint()).norm();
    assert!(residual <= 1e-15, "Hermiticity residual {residual}");
}

fn complex_hermitian_bond_operator() -> DMatrix<Complex64> {
    DMatrix::from_row_slice(
        4,
        4,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.2),
            Complex64::new(0.3, -0.7),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -0.2),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.3, 0.7),
            Complex64::new(0.3, 0.7),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.2),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.3, -0.7),
            Complex64::new(-1.0, -0.2),
            Complex64::new(0.0, 0.0),
        ],
    )
}

fn complex_product_fixture() -> (ComplexPurifiedMps, ComplexLocalHamiltonian) {
    let operator = complex_hermitian_bond_operator();
    let hamiltonian = ComplexLocalHamiltonian::try_new(operator.clone(), operator).unwrap();
    let bond_ab = new_index(1);
    let bond_ba = new_index(1);
    let make_site = |left: &thermal_imps_purification::tensor::Idx,
                     right: &thermal_imps_purification::tensor::Idx,
                     second_amplitude: Complex64| {
        let phys = new_index(2);
        let anc = new_index(2);
        let inverse_sqrt_two = 2.0_f64.sqrt().recip();
        let gamma = Tensor::from_dense(
            vec![left.clone(), phys.clone(), anc.clone(), right.clone()],
            vec![
                Complex64::new(inverse_sqrt_two, 0.0),
                second_amplitude * inverse_sqrt_two,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        ComplexSite {
            gamma,
            left: left.clone(),
            phys,
            anc,
            right: right.clone(),
        }
    };
    let state = ComplexPurifiedMps {
        a: make_site(&bond_ba, &bond_ab, Complex64::new(0.0, 1.0)),
        b: make_site(&bond_ab, &bond_ba, Complex64::new(0.6, 0.8)),
        lambda_ab: vec![1.0],
        lambda_bond_ab: bond_ab,
        lambda_ba: vec![1.0],
        lambda_bond_ba: bond_ba,
    };
    (state, hamiltonian)
}

fn assert_report_close(actual: &SpecificHeatReport, expected: &SpecificHeatReport, tolerance: f64) {
    for (name, actual, expected) in [
        (
            "raw energy variance",
            actual.raw_energy_variance_per_site,
            expected.raw_energy_variance_per_site,
        ),
        (
            "energy variance",
            actual.energy_variance_per_site,
            expected.energy_variance_per_site,
        ),
        (
            "specific heat",
            actual.specific_heat_per_site,
            expected.specific_heat_per_site,
        ),
        (
            "last positive shell magnitude",
            actual.last_positive_shell_magnitude,
            expected.last_positive_shell_magnitude,
        ),
        (
            "last negative shell magnitude",
            actual.last_negative_shell_magnitude,
            expected.last_negative_shell_magnitude,
        ),
        (
            "maximum imaginary residual",
            actual.max_imaginary_residual,
            expected.max_imaginary_residual,
        ),
    ] {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
    for (name, actual, expected) in [
        (
            "onsite contribution",
            actual.onsite_contribution,
            expected.onsite_contribution,
        ),
        (
            "positive-direction contribution",
            actual.positive_direction_contribution,
            expected.positive_direction_contribution,
        ),
        (
            "negative-direction contribution",
            actual.negative_direction_contribution,
            expected.negative_direction_contribution,
        ),
        (
            "parity-A contribution",
            actual.parity_a_contribution,
            expected.parity_a_contribution,
        ),
        (
            "parity-B contribution",
            actual.parity_b_contribution,
            expected.parity_b_contribution,
        ),
    ] {
        assert!(
            (actual - expected).norm() <= tolerance,
            "{name}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
    assert_eq!(actual.max_distance, expected.max_distance);
    assert_eq!(actual.stop_reason, expected.stop_reason);
}

#[test]
fn phase_rotated_tfim_preserves_specific_heat_report() {
    // Mutations caught: a transpose in the complex contractions, a phase convention mismatch,
    // or applying a different beta factor breaks covariance with the direct real report.
    let beta = 0.4;
    let dtau = 0.05;
    let truncation = Truncation {
        epsilon: 1e-12,
        max_bond: Some(32),
    };
    let real_hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let complex_hamiltonian = phase_rotated_tfim();
    let real_state = evolve_real_second_order(&real_hamiltonian, beta, dtau, &truncation);
    let complex_state = evolve_complex_second_order(&complex_hamiltonian, beta, dtau, &truncation);
    let real_report = specific_heat_with_options(
        &real_state,
        &real_hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    let complex_report = specific_heat_complex_with_options(
        &complex_state,
        &complex_hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )
    .unwrap();

    assert_report_close(&complex_report, &real_report, UNITARY_COVARIANCE_TOLERANCE);
    assert_eq!(
        specific_heat(&real_state, &real_hamiltonian, beta).unwrap(),
        real_report.specific_heat_per_site
    );
    assert_eq!(
        energy_variance_per_site(&real_state, &real_hamiltonian).unwrap(),
        real_report.energy_variance_per_site
    );
    assert_eq!(
        specific_heat_complex(&complex_state, &complex_hamiltonian, beta).unwrap(),
        complex_report.specific_heat_per_site
    );
    assert_eq!(
        energy_variance_per_site_complex(&complex_state, &complex_hamiltonian).unwrap(),
        complex_report.energy_variance_per_site
    );

    let exact = exact_specific_heat(1.0, 0.7, beta, 16_000);
    let decomposition_residual = [
        (complex_report.onsite_contribution - real_report.onsite_contribution).norm(),
        (complex_report.positive_direction_contribution
            - real_report.positive_direction_contribution)
            .norm(),
        (complex_report.negative_direction_contribution
            - real_report.negative_direction_contribution)
            .norm(),
        (complex_report.parity_a_contribution - real_report.parity_a_contribution).norm(),
        (complex_report.parity_b_contribution - real_report.parity_b_contribution).norm(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    println!(
        "phase-rotated TFIM diagnostics: real_heat={:.16e} complex_heat={:.16e} exact_heat={exact:.16e} real_error={:.16e} complex_error={:.16e} heat_covariance_residual={:.16e} variance_covariance_residual={:.16e} decomposition_residual={decomposition_residual:.16e} complex_imaginary_residual={:.16e}",
        real_report.specific_heat_per_site,
        complex_report.specific_heat_per_site,
        (real_report.specific_heat_per_site - exact).abs(),
        (complex_report.specific_heat_per_site - exact).abs(),
        (complex_report.specific_heat_per_site - real_report.specific_heat_per_site).abs(),
        (complex_report.energy_variance_per_site - real_report.energy_variance_per_site).abs(),
        complex_report.max_imaginary_residual,
    );
    for (backend, actual) in [
        ("real", real_report.specific_heat_per_site),
        ("phase-rotated", complex_report.specific_heat_per_site),
    ] {
        assert!(
            (actual - exact).abs() <= ROUTINE_TFIM_TROTTER_HEAT_TOLERANCE,
            "{backend} TFIM heat {actual} differs from exact {exact}"
        );
    }
}

#[test]
fn twisted_xx_matches_untwisted_report_and_exact_heat() {
    // Mutations caught: dropping the twist entirely removes the required nonreal entry, while
    // twisting only one Hamiltonian matrix or losing Hermiticity breaks unitary covariance or the
    // exact XX reference. Equality of the physical direction totals is a pure-gauge control, not
    // a shortcut: Task 4 independently checks both complex overlap orders and both tail streams
    // against dense oracles.
    let beta = 0.6;
    let dtau = 0.05;
    let truncation = Truncation {
        epsilon: 1e-12,
        max_bond: Some(32),
    };
    let real_hamiltonian = Xy { gamma: 0.0, h: 0.4 }.local();
    let untwisted_hamiltonian = twisted_xx(0.0);
    let twisted_hamiltonian = twisted_xx(PI / 5.0);
    assert_hermitian(untwisted_hamiltonian.two_site_h());
    assert_hermitian(untwisted_hamiltonian.site_energy());
    assert_hermitian(twisted_hamiltonian.two_site_h());
    assert_hermitian(twisted_hamiltonian.site_energy());
    assert!(twisted_hamiltonian
        .two_site_h()
        .iter()
        .any(|value| value.im.abs() > 1e-8));

    let real_state = evolve_real_second_order(&real_hamiltonian, beta, dtau, &truncation);
    let untwisted_state =
        evolve_complex_second_order(&untwisted_hamiltonian, beta, dtau, &truncation);
    let twisted_state = evolve_complex_second_order(&twisted_hamiltonian, beta, dtau, &truncation);
    let real_report = specific_heat_with_options(
        &real_state,
        &real_hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    let untwisted_report = specific_heat_complex_with_options(
        &untwisted_state,
        &untwisted_hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    let twisted_report = specific_heat_complex_with_options(
        &twisted_state,
        &twisted_hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    assert_report_close(
        &untwisted_report,
        &real_report,
        ROUTINE_XX_BACKEND_COVARIANCE_TOLERANCE,
    );
    assert_report_close(
        &twisted_report,
        &real_report,
        ROUTINE_XX_BACKEND_COVARIANCE_TOLERANCE,
    );

    let exact = xy_specific_heat(0.0, 0.4, beta, 16_000);
    println!(
        "twisted XX heat diagnostics: real_heat={:.16e} untwisted_complex_heat={:.16e} twisted_complex_heat={:.16e} exact_heat={exact:.16e} real_error={:.16e} untwisted_error={:.16e} twisted_error={:.16e} twisted_real_residual={:.16e}",
        real_report.specific_heat_per_site,
        untwisted_report.specific_heat_per_site,
        twisted_report.specific_heat_per_site,
        (real_report.specific_heat_per_site - exact).abs(),
        (untwisted_report.specific_heat_per_site - exact).abs(),
        (twisted_report.specific_heat_per_site - exact).abs(),
        (twisted_report.specific_heat_per_site - real_report.specific_heat_per_site).abs(),
    );
    for (backend, actual) in [
        ("real", real_report.specific_heat_per_site),
        ("untwisted complex", untwisted_report.specific_heat_per_site),
        ("twisted complex", twisted_report.specific_heat_per_site),
    ] {
        assert!(
            (actual - exact).abs() <= ROUTINE_TWISTED_XX_TROTTER_HEAT_TOLERANCE,
            "{backend} XX heat {actual} differs from exact {exact}"
        );
    }

    let direction_difference = (twisted_report.positive_direction_contribution
        - twisted_report.negative_direction_contribution)
        .norm();
    println!(
        "twisted XX diagnostics: positive={} negative={} difference={direction_difference:.16e} max_imaginary_residual={}",
        twisted_report.positive_direction_contribution,
        twisted_report.negative_direction_contribution,
        twisted_report.max_imaginary_residual,
    );
    assert!(
        direction_difference <= TWISTED_XX_DIRECTION_EQUALITY_TOLERANCE,
        "twisted XX positive and negative direction totals differ: {direction_difference}"
    );
}

#[test]
fn limiting_beta_zero_has_finite_variance_and_exactly_zero_heat() {
    // Mutations caught: applying beta only in a convenience wrapper, or multiplying variance by
    // beta fewer than two times, makes the beta-zero report and scalar disagree or become nonzero.
    let hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let state = infinite_temperature(hamiltonian.dim());
    let report =
        specific_heat_with_options(&state, &hamiltonian, 0.0, &SpecificHeatOptions::default())
            .unwrap();
    assert_eq!(report.specific_heat_per_site, 0.0);
    assert!(report.energy_variance_per_site.is_finite());
    assert!(report.energy_variance_per_site >= 0.0);
    println!(
        "beta-zero diagnostics: heat={:.16e} variance={:.16e} raw_variance={:.16e} max_imaginary_residual={:.16e}",
        report.specific_heat_per_site,
        report.energy_variance_per_site,
        report.raw_energy_variance_per_site,
        report.max_imaginary_residual,
    );
    assert_eq!(
        specific_heat(&state, &hamiltonian, 0.0).unwrap(),
        report.specific_heat_per_site
    );
    assert_eq!(
        energy_variance_per_site(&state, &hamiltonian).unwrap(),
        report.energy_variance_per_site
    );
}

#[test]
fn limiting_zero_hamiltonian_report_is_exactly_zero() {
    // Mutations caught: a disconnected subtraction, tail accumulator, or beta prefactor that
    // injects a spurious value makes at least one independently exposed report field nonzero.
    let hamiltonian = LocalHamiltonian {
        two_site_h: DMatrix::zeros(4, 4),
        site_energy: DMatrix::zeros(4, 4),
    };
    let state = infinite_temperature(hamiltonian.dim());
    let options = SpecificHeatOptions {
        max_distance: 4,
        consecutive_small_shells: 3,
        ..SpecificHeatOptions::default()
    };
    let report = specific_heat_with_options(&state, &hamiltonian, 0.7, &options).unwrap();
    println!(
        "zero-Hamiltonian diagnostics: heat={:.16e} variance={:.16e} max_distance={} last_positive={:.16e} last_negative={:.16e}",
        report.specific_heat_per_site,
        report.energy_variance_per_site,
        report.max_distance,
        report.last_positive_shell_magnitude,
        report.last_negative_shell_magnitude,
    );
    for (name, value) in [
        ("raw variance", report.raw_energy_variance_per_site),
        ("variance", report.energy_variance_per_site),
        ("specific heat", report.specific_heat_per_site),
        ("last positive shell", report.last_positive_shell_magnitude),
        ("last negative shell", report.last_negative_shell_magnitude),
        ("imaginary residual", report.max_imaginary_residual),
    ] {
        assert_eq!(value, 0.0, "{name}");
    }
    for (name, value) in [
        ("onsite", report.onsite_contribution),
        ("positive", report.positive_direction_contribution),
        ("negative", report.negative_direction_contribution),
        ("parity A", report.parity_a_contribution),
        ("parity B", report.parity_b_contribution),
    ] {
        assert_eq!(value, Complex64::new(0.0, 0.0), "{name}");
    }
    assert_eq!(report.max_distance, 4);
}

#[test]
fn complex_direct_scalar_apis_match_the_default_report() {
    // Mutations caught: selecting the wrong report field or applying beta outside the shared
    // report path changes one of these exact default-options comparisons.
    let (state, hamiltonian) = complex_product_fixture();
    let beta = 0.7;
    let report = specific_heat_complex_with_options(
        &state,
        &hamiltonian,
        beta,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    assert_eq!(
        specific_heat_complex(&state, &hamiltonian, beta).unwrap(),
        report.specific_heat_per_site
    );
    let variance_report = specific_heat_complex_with_options(
        &state,
        &hamiltonian,
        1.0,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    assert_eq!(
        energy_variance_per_site_complex(&state, &hamiltonian).unwrap(),
        variance_report.energy_variance_per_site
    );
}

#[test]
fn automatic_dispatch_matches_direct_complex_specific_heat() {
    // Mutations caught: retaining the unsupported complex arm, sending a complex state through
    // the real backend, or returning a report/scalar from a different complex calculation.
    let (state, hamiltonian) = complex_product_fixture();
    let auto_state = ItebdState::Complex(state.clone());
    let auto_hamiltonian = ItebdHamiltonian::Complex(hamiltonian.clone());
    let beta = 0.7;
    let options = SpecificHeatOptions::default();
    let direct = specific_heat_complex_with_options(&state, &hamiltonian, beta, &options).unwrap();
    let automatic =
        specific_heat_with_options_auto(&auto_state, &auto_hamiltonian, beta, &options).unwrap();
    assert_report_close(&automatic, &direct, 2e-12);
    assert!(
        (specific_heat_auto(&auto_state, &auto_hamiltonian, beta).unwrap()
            - specific_heat_complex(&state, &hamiltonian, beta).unwrap())
        .abs()
            <= 2e-12
    );
}

#[test]
fn complex_report_reconstructs_real_variance_from_direction_and_parity_totals() {
    // Mutations caught: swapped parity, omitted direction, or premature real projection breaks
    // one of the two independently exposed report reconstructions.
    let (state, hamiltonian) = complex_product_fixture();
    let report = specific_heat_complex_with_options(
        &state,
        &hamiltonian,
        0.7,
        &SpecificHeatOptions::default(),
    )
    .unwrap();
    let directions = report.onsite_contribution
        + report.positive_direction_contribution
        + report.negative_direction_contribution;
    let parities = report.parity_a_contribution + report.parity_b_contribution;
    assert!((directions - Complex64::new(report.raw_energy_variance_per_site, 0.0)).norm() < 1e-10);
    assert!((parities - directions).norm() < 1e-10);
    assert!(report.positive_direction_contribution.im.abs() > 1e-8);
    assert!(
        (report.positive_direction_contribution.im + report.negative_direction_contribution.im)
            .abs()
            < 1e-10
    );
}

#[test]
fn complex_nonfinite_tensor_is_rejected_before_any_division() {
    // Mutation caught: deleting topology/finiteness validation would pass NaN into transfer
    // normalization and obscure the source behind a later non-real or convergence failure.
    let (mut state, hamiltonian) = complex_product_fixture();
    let mut values = state.a.gamma.to_vec::<Complex64>().unwrap();
    values[3] = Complex64::new(f64::NAN, 0.0);
    state.a.gamma = Tensor::from_dense(state.a.gamma.indices.clone(), values).unwrap();
    assert!(matches!(
        specific_heat_complex(&state, &hamiltonian, 0.7),
        Err(ItebdError::TensorOperation {
            stage: "complex_state_topology",
            message,
        }) if message == "A.gamma contains a non-finite value"
    ));
}

#[test]
fn direct_real_specific_heat_rejects_malformed_hamiltonian_shapes() {
    let state = infinite_temperature(2);
    let rectangular = LocalHamiltonian {
        two_site_h: DMatrix::zeros(4, 4),
        site_energy: DMatrix::zeros(3, 4),
    };
    assert!(matches!(
        specific_heat(&state, &rectangular, 0.7),
        Err(ItebdError::NonSquare {
            field: "site_energy",
            rows: 3,
            cols: 4,
        })
    ));

    let wrong_size = LocalHamiltonian {
        two_site_h: DMatrix::zeros(4, 4),
        site_energy: DMatrix::zeros(9, 9),
    };
    assert!(matches!(
        specific_heat(&state, &wrong_size, 0.7),
        Err(ItebdError::MatrixDimensionMismatch {
            two_site: 4,
            site_energy: 9,
        })
    ));
}

#[test]
fn direct_real_specific_heat_rejects_nonfinite_hamiltonian_entries() {
    let state = infinite_temperature(2);
    let mut hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    hamiltonian.site_energy[(1, 2)] = f64::NAN;
    assert!(matches!(
        specific_heat(&state, &hamiltonian, 0.7),
        Err(ItebdError::NonFiniteMatrix {
            field: "site_energy",
            row: 1,
            column: 2,
        })
    ));
}

#[test]
fn direct_real_specific_heat_rejects_inconsistent_schmidt_data() {
    let hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let mut state = infinite_temperature(2);
    state.lambda_ab.push(0.0);

    assert!(matches!(
        specific_heat(&state, &hamiltonian, 0.7),
        Err(ItebdError::TensorOperation {
            stage: "real_specific_heat_topology",
            message,
        }) if message == "lambda_ab length 2 does not match positive bond dimension 1"
    ));
}

fn real_state_with_mismatched_gamma_index_dimension() -> PurifiedMps {
    let mut state = infinite_temperature(2);
    let mut indices = state.a.gamma.indices.clone();
    indices[0].dim = 2;
    let element_count = indices.iter().map(|index| index.dim).product();
    state.a.gamma = Tensor::from_dense(indices, vec![0.0; element_count]).unwrap();
    state
}

#[test]
fn direct_real_specific_heat_rejects_gamma_index_dimension_mismatch() {
    // Mutation caught: comparing tensor/declaration index identities before their dimensions lets
    // the malformed tensor reach scale_bond and panic below this public Result boundary.
    let state = real_state_with_mismatched_gamma_index_dimension();
    let hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();

    assert!(matches!(
        specific_heat(&state, &hamiltonian, 0.7),
        Err(ItebdError::TensorOperation {
            stage: "real_specific_heat_topology",
            message,
        }) if message == "A.gamma left dimension 2 does not match declared A.left dimension 1"
    ));
}

#[test]
fn automatic_real_specific_heat_rejects_gamma_index_dimension_mismatch() {
    // Mutation caught: automatic real dispatch must preserve the direct backend's typed topology
    // rejection instead of exposing the same scale_bond panic through either public facade.
    let state = ItebdState::Real(real_state_with_mismatched_gamma_index_dimension());
    let hamiltonian = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());

    for result in [
        specific_heat_auto(&state, &hamiltonian, 0.7).map(|_| ()),
        specific_heat_with_options_auto(&state, &hamiltonian, 0.7, &SpecificHeatOptions::default())
            .map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(ItebdError::TensorOperation {
                stage: "real_specific_heat_topology",
                message,
            }) if message == "A.gamma left dimension 2 does not match declared A.left dimension 1"
        ));
    }
}

#[test]
fn readme_documents_fallible_specific_heat_api() {
    // Mutation caught: removing the public scalar, options/report, or typed-tail-error guidance
    // leaves users without the documented fallible API contract.
    let readme = include_str!("../README.md");
    for required in [
        "specific_heat(&state, &hamiltonian, beta)?",
        "SpecificHeatOptions",
        "specific_heat_with_options",
        "SpecificHeatTailNonConvergence",
        "reconstruct `raw_energy_variance_per_site`",
    ] {
        assert!(readme.contains(required), "README missing {required}");
    }
}
