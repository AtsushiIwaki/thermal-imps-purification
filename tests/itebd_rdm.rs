use thermal_imps_purification::itebd_rdm::{reduced_density_matrix, RdmOptions, RdmParity};
use thermal_imps_purification::purified_mps::infinite_temperature;

#[test]
fn physical_interval_at_infinite_temperature_is_maximally_mixed() {
    for d in [1_usize, 2, 3] {
        let s = infinite_temperature(d);
        for n in 1_u32..=4 {
            for start in [RdmParity::A, RdmParity::B] {
                let result =
                    reduced_density_matrix(&s, start, n as usize, &RdmOptions::default()).unwrap();
                let m = d.pow(n);
                assert_eq!(result.density_matrix.shape(), (m, m));
                assert_eq!(result.report.largest_intermediate_elements, m * m);
                let expected = nalgebra::DMatrix::<f64>::identity(m, m) / m as f64;
                assert!((result.density_matrix - expected).norm() <= 1e-12);
                assert!(result.report.left.relative_residual <= 1e-12);
                assert!(result.report.minimum_eigenvalue >= -1e-10);
            }
        }
    }
}

#[path = "support/itebd_rdm.rs"]
mod support;
use thermal_imps_purification::itebd_auto::ItebdState;
use thermal_imps_purification::itebd_rdm::{
    reduced_density_matrix_auto, reduced_density_matrix_complex, AutoRdmResult, RdmError,
};

#[test]
fn alternating_probabilities_use_first_site_fastest_for_both_backends() {
    let real = support::alternating_product_state();
    let complex = support::complex_ancilla_rotated(&real);
    assert!(complex
        .a
        .gamma
        .to_vec::<num_complex::Complex64>()
        .unwrap()
        .iter()
        .any(|z| z.im != 0.0));
    for length in 1..=4 {
        for start in [RdmParity::A, RdmParity::B] {
            let r = reduced_density_matrix(&real, start, length, &RdmOptions::default()).unwrap();
            let c = reduced_density_matrix_complex(&complex, start, length, &RdmOptions::default())
                .unwrap();
            let dim = 1_usize << length;
            let expected =
                nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_fn(dim, |mut p, _| {
                    (0..length)
                        .map(|site| {
                            let probabilities =
                                if (site + usize::from(start == RdmParity::B)) % 2 == 0 {
                                    [0.2, 0.8]
                                } else {
                                    [0.7, 0.3]
                                };
                            let value = probabilities[p % 2];
                            p /= 2;
                            value
                        })
                        .product::<f64>()
                }));
            assert!((&r.density_matrix - &expected).norm() < 1e-12);
            assert!(
                (&c.density_matrix - expected.map(num_complex::Complex64::from)).norm() < 1e-12
            );
        }
    }
    assert!(matches!(
        reduced_density_matrix_auto(
            &ItebdState::Real(real),
            RdmParity::A,
            1,
            &RdmOptions::default()
        ),
        Ok(AutoRdmResult::Real(_))
    ));
    assert!(matches!(
        reduced_density_matrix_auto(
            &ItebdState::Complex(complex),
            RdmParity::B,
            1,
            &RdmOptions::default()
        ),
        Ok(AutoRdmResult::Complex(_))
    ));
}

#[test]
fn interval_preflight_rejects_invalid_lengths_and_caps_before_solving() {
    for d in [1, 2] {
        let state = infinite_temperature(d);
        assert!(matches!(
            reduced_density_matrix(&state, RdmParity::A, 0, &RdmOptions::default()),
            Err(RdmError::InvalidOption { .. })
        ));
        for length in [usize::MAX, usize::MAX / 2] {
            assert!(matches!(
                reduced_density_matrix(&state, RdmParity::A, length, &RdmOptions::default()),
                Err(RdmError::DimensionOverflow {
                    stage: "interval index metadata"
                })
            ));
        }
    }
    let state = infinite_temperature(2);
    let output_cap = RdmOptions {
        max_output_elements: 15,
        max_iterations: 1,
        ..RdmOptions::default()
    };
    assert!(matches!(
        reduced_density_matrix(&state, RdmParity::A, 2, &output_cap),
        Err(RdmError::ResourceLimit { .. })
    ));
    let intermediate_cap = RdmOptions {
        max_intermediate_elements: 15,
        ..RdmOptions::default()
    };
    assert!(matches!(
        reduced_density_matrix(&state, RdmParity::A, 2, &intermediate_cap),
        Err(RdmError::ResourceLimit { .. })
    ));
}

// Reuse the independent test-only oracle through public state views.
use thermal_imps_purification::{itebd_auto, itebd_complex, itebd_state_view, purified_mps, tensor};
#[allow(dead_code)]
#[path = "support/itebd_checkpoint_rdm.rs"]
mod checkpoint_support;
#[allow(dead_code)]
#[path = "support/itebd_rdm_invariants.rs"]
mod invariants;
#[allow(dead_code)]
#[path = "../src/itebd_rdm/oracles.rs"]
mod oracles;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn partial_traces_preserve_start_and_shift_start_on_correlated_states() {
    for complex in [false, true] {
        let state = oracles::fixture(complex, 2, 3);
        for start in [RdmParity::A, RdmParity::B] {
            let opposite = if start == RdmParity::A {
                RdmParity::B
            } else {
                RdmParity::A
            };
            for length in 2..=4 {
                let rho = checkpoint_support::rdm(&state, start, length);
                let last = (invariants::trace_last(&rho, 2)
                    - checkpoint_support::rdm(&state, start, length - 1))
                .norm();
                let first = (invariants::trace_first(&rho, 2)
                    - checkpoint_support::rdm(&state, opposite, length - 1))
                .norm();
                eprintln!("marginal complex={complex} start={start:?} n={length} last={last:e} first={first:e}");
                assert!(last <= 1e-10 && first <= 1e-10);
            }
        }
    }
}

#[test]
fn virtual_gauges_ancilla_rotations_and_physical_covariance() {
    for complex in [false, true] {
        let state = oracles::fixture(complex, 2, 3);
        for transform in [
            invariants::Transformation::PositiveGauge,
            invariants::Transformation::UnitaryGauge,
            invariants::Transformation::Ancilla,
            invariants::Transformation::Physical,
        ] {
            let transformed = invariants::transform_fixture(&state, transform);
            let mut largest_error = 0.0_f64;
            let mut sensitivity = 0.0_f64;
            for start in [RdmParity::A, RdmParity::B] {
                for length in 1..=4 {
                    let rho = checkpoint_support::rdm(&state, start, length);
                    let got = checkpoint_support::rdm(&transformed, start, length);
                    let expected = if matches!(transform, invariants::Transformation::Physical) {
                        let phases: Vec<_> = (0..1_usize << length)
                            .map(|p| Complex64::new(0.0, 1.0).powu(p.count_ones()))
                            .collect();
                        DMatrix::from_fn(rho.nrows(), rho.ncols(), |p, q| {
                            phases[p] * rho[(p, q)] * phases[q].conj()
                        })
                    } else {
                        rho.clone()
                    };
                    sensitivity = sensitivity.max((&got - &rho).norm());
                    let error = (got - expected).norm();
                    largest_error = largest_error.max(error);
                    assert!(
                        error <= 1e-10,
                        "{transform:?} complex={complex} start={start:?} n={length}: {error:e}"
                    );
                }
            }
            eprintln!("transform={transform:?} complex={complex} max_error={largest_error:e} change={sensitivity:e}");
            if matches!(transform, invariants::Transformation::Physical) {
                assert!(sensitivity > 1e-3);
            }
        }
    }
}

#[test]
fn auto_real_payload_and_all_report_fields_are_bit_exact_and_complex_stays_complex() {
    let state = oracles::fixture(false, 2, 3);
    let ItebdState::Real(real) = &state else {
        panic!("real fixture")
    };
    for start in [RdmParity::A, RdmParity::B] {
        for length in 1..=4 {
            let direct =
                reduced_density_matrix(real, start, length, &RdmOptions::default()).unwrap();
            let AutoRdmResult::Real(auto) =
                reduced_density_matrix_auto(&state, start, length, &RdmOptions::default()).unwrap()
            else {
                panic!("real dispatch changed backend")
            };
            assert_eq!(direct.density_matrix.shape(), auto.density_matrix.shape());
            assert_eq!(
                direct
                    .density_matrix
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                auto.density_matrix
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            checkpoint_support::assert_report_bits(&direct.report, &auto.report);
        }
    }
    let zero_imaginary = checkpoint_support::promote_real_state(real);
    for site in [&zero_imaginary.a, &zero_imaginary.b] {
        assert!(site
            .gamma
            .to_vec::<Complex64>()
            .unwrap()
            .iter()
            .all(|z| z.im == 0.0));
    }
    assert!(matches!(
        reduced_density_matrix_auto(
            &ItebdState::Complex(zero_imaginary),
            RdmParity::B,
            2,
            &RdmOptions::default()
        )
        .unwrap(),
        AutoRdmResult::Complex(_)
    ));
}

#[test]
fn invalid_options_and_residual_exhaustion_return_specific_errors() {
    for complex in [false, true] {
        let state = oracles::fixture(complex, 2, 3);
        for field in 0..4 {
            for value in [-1.0, f64::NAN, f64::INFINITY] {
                let mut options = RdmOptions::default();
                let (name, slot) = match field {
                    0 => ("fixed_point_tolerance", &mut options.fixed_point_tolerance),
                    1 => ("hermiticity_tolerance", &mut options.hermiticity_tolerance),
                    2 => ("trace_tolerance", &mut options.trace_tolerance),
                    _ => ("positivity_tolerance", &mut options.positivity_tolerance),
                };
                *slot = value;
                assert!(
                    matches!(reduced_density_matrix_auto(&state, RdmParity::A, 1, &options), Err(RdmError::InvalidOption {name: got, ..}) if got == name)
                );
            }
        }
        for field in 0..3 {
            let mut options = RdmOptions::default();
            let (name, slot) = match field {
                0 => ("max_iterations", &mut options.max_iterations),
                1 => ("max_output_elements", &mut options.max_output_elements),
                _ => (
                    "max_intermediate_elements",
                    &mut options.max_intermediate_elements,
                ),
            };
            *slot = 0;
            assert!(
                matches!(reduced_density_matrix_auto(&state, RdmParity::A, 1, &options), Err(RdmError::InvalidOption {name: got, ..}) if got == name)
            );
        }
        assert!(matches!(
            reduced_density_matrix_auto(&state, RdmParity::A, 0, &RdmOptions::default()),
            Err(RdmError::InvalidOption { name: "length", .. })
        ));
        let output_cap = RdmOptions {
            max_output_elements: 15,
            ..RdmOptions::default()
        };
        assert!(matches!(
            reduced_density_matrix_auto(&state, RdmParity::A, 2, &output_cap),
            Err(RdmError::ResourceLimit {
                stage: "output matrix",
                requested: 16,
                limit: 15
            })
        ));
        let intermediate_cap = RdmOptions {
            max_intermediate_elements: 2000,
            ..RdmOptions::default()
        };
        assert!(matches!(
            reduced_density_matrix_auto(&state, RdmParity::A, 4, &intermediate_cap),
            Err(RdmError::ResourceLimit {
                stage: "interval bra",
                requested: 2304,
                limit: 2000
            })
        ));
        let options = RdmOptions {
            max_iterations: 1,
            ..RdmOptions::default()
        };
        assert!(
            matches!(reduced_density_matrix_auto(&state, RdmParity::A, 1, &options), Err(RdmError::FixedPointNonConvergence {iterations: 1, residual, ..}) if residual > 1e-12)
        );
        let exact_options = RdmOptions {
            fixed_point_tolerance: 0.0,
            ..options
        };
        assert!(
            matches!(reduced_density_matrix_auto(&state, RdmParity::A, 1, &exact_options), Err(RdmError::FixedPointNonConvergence {iterations: 1, residual, ..}) if residual > 0.0)
        );
        let mut zero = oracles::fixture(complex, 2, 3);
        match &mut zero {
            ItebdState::Real(s) => {
                s.a.gamma =
                    tensor::Tensor::from_dense(s.a.gamma.indices.clone(), vec![0.0; 24]).unwrap()
            }
            ItebdState::Complex(s) => {
                s.a.gamma = tensor::Tensor::from_dense(
                    s.a.gamma.indices.clone(),
                    vec![Complex64::default(); 24],
                )
                .unwrap()
            }
        }
        assert!(matches!(
            reduced_density_matrix_auto(&zero, RdmParity::A, 1, &RdmOptions::default()),
            Err(RdmError::InvalidEnvironment { .. })
        ));
        let options = RdmOptions {
            max_output_elements: usize::MAX,
            max_intermediate_elements: usize::MAX,
            ..RdmOptions::default()
        };
        assert!(matches!(
            reduced_density_matrix_auto(&state, RdmParity::A, usize::BITS as usize, &options),
            Err(RdmError::DimensionOverflow {
                stage: "physical dimension"
            })
        ));
        assert!(matches!(
            reduced_density_matrix_auto(&state, RdmParity::A, usize::BITS as usize / 2, &options),
            Err(RdmError::DimensionOverflow {
                stage: "output matrix"
            })
        ));
    }
}

#[test]
fn exact_zero_tolerances_are_valid_for_exactly_representable_state() {
    let state = infinite_temperature(1);
    for zero in [0.0_f64, -0.0_f64] {
        let options = RdmOptions {
            fixed_point_tolerance: zero,
            hermiticity_tolerance: zero,
            trace_tolerance: zero,
            positivity_tolerance: zero,
            ..RdmOptions::default()
        };
        let real = reduced_density_matrix(&state, RdmParity::A, 2, &options).unwrap();
        assert_eq!(real.density_matrix[(0, 0)].to_bits(), 1.0_f64.to_bits());
        let complex = checkpoint_support::promote_real_state(&state);
        let result = reduced_density_matrix_complex(&complex, RdmParity::B, 2, &options).unwrap();
        assert_eq!(result.density_matrix[(0, 0)], Complex64::from(1.0));
    }
}
