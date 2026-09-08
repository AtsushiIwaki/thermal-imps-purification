use thermal_imps_purification::config::{RunConfig, TrotterOrder};
use thermal_imps_purification::itebd::{imaginary_time_step, imaginary_time_step_second_order, StepInfo};
use thermal_imps_purification::itebd_auto::{
    canonicalize_auto, energy_density_auto, imaginary_time_step_auto,
    imaginary_time_step_second_order_auto, local_expectation_auto, specific_heat_auto,
    specific_heat_with_options_auto, ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::itebd_complex::{
    canonicalize_complex, complex_trotter_gate, dominant_fixed_point_left,
    dominant_fixed_point_right, energy_density_complex, imaginary_time_step_complex,
    imaginary_time_step_second_order_complex, local_expectation_complex, ComplexLocalHamiltonian,
    ComplexPurifiedMps, ComplexSite,
};
use thermal_imps_purification::itebd_error::ItebdError;
use thermal_imps_purification::model::{pauli_x, LocalHamiltonian, Tfim};
use thermal_imps_purification::observable::{energy_density, magnetization};
use thermal_imps_purification::purified_mps::{infinite_temperature, PurifiedMps, Site};
use thermal_imps_purification::specific_heat::{SpecificHeatOptions, SpecificHeatReport};
use thermal_imps_purification::tensor::{new_index, Idx, Tensor, Truncation};
use thermal_imps_purification::variance::{specific_heat, specific_heat_with_options};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::{fs, path::Path};
use tensor4all_core::svd::{svd_with, SvdOptions};
use tensor4all_core::SvdTruncationPolicy;

fn genuinely_complex_hermitian() -> DMatrix<Complex64> {
    DMatrix::from_row_slice(
        4,
        4,
        &[
            Complex64::new(0.7, 0.0),
            Complex64::new(0.2, 0.3),
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.1, 0.2),
            Complex64::new(0.2, -0.3),
            Complex64::new(-0.4, 0.0),
            Complex64::new(0.5, -0.1),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.9, 0.0),
            Complex64::new(-0.2, 0.4),
            Complex64::new(-0.1, -0.2),
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.2, -0.4),
            Complex64::new(0.1, 0.0),
        ],
    )
}

fn max_entrywise_norm(matrix: &DMatrix<Complex64>) -> f64 {
    matrix.iter().map(|value| value.norm()).fold(0.0, f64::max)
}

fn rotated_tfim() -> ComplexLocalHamiltonian {
    let real = Tfim { j: 1.0, g: 0.7 }.local();
    let angle = 0.37;
    let rotation = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
        Complex64::from_polar(1.0, -0.5 * angle),
        Complex64::from_polar(1.0, 0.5 * angle),
    ]));
    let two_site_rotation = rotation.kronecker(&rotation);
    let rotate = |matrix: &DMatrix<f64>| {
        &two_site_rotation
            * matrix.map(|value| Complex64::new(value, 0.0))
            * two_site_rotation.adjoint()
    };
    let two_site_h = rotate(&real.two_site_h);
    assert!(two_site_h.iter().any(|value| value.im.abs() > 1e-8));
    ComplexLocalHamiltonian::try_new(two_site_h, rotate(&real.site_energy)).unwrap()
}

fn phase_rotated_tfim() -> ComplexLocalHamiltonian {
    let real = Tfim { j: 1.0, g: 0.7 }.local();
    let phase = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let rotate_two_site = |matrix: &DMatrix<f64>| {
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
    ComplexLocalHamiltonian::try_new(
        rotate_two_site(&real.two_site_h),
        rotate_two_site(&real.site_energy),
    )
    .unwrap()
}

fn phase_rotated_pauli_x() -> DMatrix<Complex64> {
    let real = pauli_x();
    let phase = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let mut rotated = DMatrix::<Complex64>::zeros(2, 2);
    for row in 0..2 {
        for column in 0..2 {
            rotated[(row, column)] = phase[row] * real[(row, column)] * phase[column].conj();
        }
    }
    rotated
}

#[derive(Debug, Clone, Copy)]
struct ComplexThermodynamicSample {
    energy: f64,
    free_energy: f64,
    log_norm_per_site: f64,
    min_singular_value: f64,
    max_bond: usize,
}

fn evolve_phase_rotated_tfim_second_order(
    beta: f64,
    dtau: f64,
    truncation: Truncation,
) -> ComplexThermodynamicSample {
    let hamiltonian = phase_rotated_tfim();
    let steps = (beta / (2.0 * dtau)).round() as usize;
    assert!((2.0 * steps as f64 * dtau - beta).abs() <= 1e-12);
    let mut state = ComplexPurifiedMps::infinite_temperature(hamiltonian.dim()).unwrap();
    let mut accumulated_log_norm = 0.0;
    let mut min_singular_value = f64::INFINITY;
    let mut max_bond = 1;

    for _ in 0..steps {
        let info =
            imaginary_time_step_second_order_complex(&mut state, &hamiltonian, dtau, &truncation)
                .unwrap();
        assert!(info.log_norm.is_finite());
        assert!(info.min_singular_value.is_finite());
        accumulated_log_norm += info.log_norm + canonicalize_complex(&mut state, 1e-12).unwrap();
        min_singular_value = min_singular_value.min(info.min_singular_value);
        max_bond = max_bond.max(info.max_bond);
    }

    let log_norm_per_site = accumulated_log_norm / 2.0;
    ComplexThermodynamicSample {
        energy: energy_density_complex(&state, &hamiltonian).unwrap(),
        free_energy: thermal_imps_purification::itebd::free_energy_from_log_norm(
            log_norm_per_site,
            beta,
            hamiltonian.dim(),
        ),
        log_norm_per_site,
        min_singular_value,
        max_bond,
    }
}

fn nondegenerate_complex_fixture() -> (ComplexPurifiedMps, ComplexLocalHamiltonian) {
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let normalized = |mut values: Vec<Complex64>| {
        let norm = values.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    };
    state.a.gamma = Tensor::from_dense(
        state.a.gamma.indices.clone(),
        normalized(vec![
            Complex64::new(0.73, 0.11),
            Complex64::new(-0.29, 0.37),
            Complex64::new(0.41, -0.53),
            Complex64::new(0.19, 0.61),
        ]),
    )
    .unwrap();
    state.b.gamma = Tensor::from_dense(
        state.b.gamma.indices.clone(),
        normalized(vec![
            Complex64::new(-0.17, 0.67),
            Complex64::new(0.59, 0.23),
            Complex64::new(0.31, -0.47),
            Complex64::new(0.79, -0.07),
        ]),
    )
    .unwrap();
    state.validate_topology().unwrap();

    let two_site_h = genuinely_complex_hermitian();
    let hamiltonian = ComplexLocalHamiltonian::try_new(two_site_h.clone(), two_site_h).unwrap();
    (state, hamiltonian)
}

fn dense_complex(tensor: &Tensor) -> Vec<Complex64> {
    assert!(tensor.is_complex());
    tensor.to_vec::<Complex64>().unwrap()
}

fn test_relabel(tensor: &Tensor, from: &Idx, to: &Idx) -> Tensor {
    assert_eq!(from.dim, to.dim);
    let indices = tensor
        .indices
        .iter()
        .map(|index| {
            if index.id == from.id {
                to.clone()
            } else {
                index.clone()
            }
        })
        .collect();
    Tensor::from_dense(indices, dense_complex(tensor)).unwrap()
}

fn test_scale_bond(tensor: &Tensor, bond: &Idx, factors: &[f64]) -> Tensor {
    let dimensions = tensor.dims();
    let position = tensor
        .indices
        .iter()
        .position(|index| index.id == bond.id)
        .unwrap();
    assert_eq!(dimensions[position], factors.len());
    let mut values = dense_complex(tensor);
    let mut multi = vec![0usize; dimensions.len()];
    for value in &mut values {
        *value *= factors[multi[position]];
        for (coordinate, &dimension) in multi.iter_mut().zip(&dimensions) {
            *coordinate += 1;
            if *coordinate < dimension {
                break;
            }
            *coordinate = 0;
        }
    }
    Tensor::from_dense(tensor.indices.clone(), values).unwrap()
}

fn test_binary_contract(left: &Tensor, right: &Tensor) -> Tensor {
    let shared = left
        .indices
        .iter()
        .enumerate()
        .filter_map(|(left_position, left_index)| {
            right
                .indices
                .iter()
                .position(|right_index| right_index.id == left_index.id)
                .map(|right_position| (left_position, right_position, left_index.dim))
        })
        .collect::<Vec<_>>();
    assert!(!shared.is_empty());
    let left_open = (0..left.indices.len())
        .filter(|position| !shared.iter().any(|(left, _, _)| left == position))
        .collect::<Vec<_>>();
    let right_open = (0..right.indices.len())
        .filter(|position| !shared.iter().any(|(_, right, _)| right == position))
        .collect::<Vec<_>>();
    let output_indices = left_open
        .iter()
        .map(|&position| left.indices[position].clone())
        .chain(
            right_open
                .iter()
                .map(|&position| right.indices[position].clone()),
        )
        .collect::<Vec<_>>();
    let output_dimensions = output_indices
        .iter()
        .map(|index| index.dim)
        .collect::<Vec<_>>();
    let left_dimensions = left.dims();
    let right_dimensions = right.dims();
    let left_values = dense_complex(left);
    let right_values = dense_complex(right);
    let mut output = Vec::with_capacity(output_dimensions.iter().product());

    for output_flat in 0..output_dimensions.iter().product::<usize>() {
        let mut residual = output_flat;
        let output_coordinates = output_dimensions
            .iter()
            .map(|&dimension| {
                let coordinate = residual % dimension;
                residual /= dimension;
                coordinate
            })
            .collect::<Vec<_>>();
        let mut left_coordinates = vec![0; left_dimensions.len()];
        let mut right_coordinates = vec![0; right_dimensions.len()];
        for (coordinate, &position) in output_coordinates.iter().zip(&left_open) {
            left_coordinates[position] = *coordinate;
        }
        for (coordinate, &position) in output_coordinates
            .iter()
            .skip(left_open.len())
            .zip(&right_open)
        {
            right_coordinates[position] = *coordinate;
        }

        let shared_dimensions = shared
            .iter()
            .map(|(_, _, dimension)| *dimension)
            .collect::<Vec<_>>();
        let mut value = Complex64::new(0.0, 0.0);
        for shared_flat in 0..shared_dimensions.iter().product::<usize>() {
            let mut residual = shared_flat;
            for ((left_position, right_position, _), &dimension) in
                shared.iter().zip(&shared_dimensions)
            {
                let coordinate = residual % dimension;
                residual /= dimension;
                left_coordinates[*left_position] = coordinate;
                right_coordinates[*right_position] = coordinate;
            }
            let left_flat = left_coordinates
                .iter()
                .zip(&left_dimensions)
                .fold(
                    (0usize, 1usize),
                    |(flat, stride), (&coordinate, &dimension)| {
                        (flat + stride * coordinate, stride * dimension)
                    },
                )
                .0;
            let right_flat = right_coordinates
                .iter()
                .zip(&right_dimensions)
                .fold(
                    (0usize, 1usize),
                    |(flat, stride), (&coordinate, &dimension)| {
                        (flat + stride * coordinate, stride * dimension)
                    },
                )
                .0;
            value += left_values[left_flat] * right_values[right_flat];
        }
        output.push(value);
    }

    Tensor::from_dense(output_indices, output).unwrap()
}

fn test_gate_tensor(
    gate: &DMatrix<Complex64>,
    p1: &Idx,
    p2: &Idx,
    p1_out: &Idx,
    p2_out: &Idx,
) -> Tensor {
    let dimension = p1.dim;
    let indices = vec![p1.clone(), p2.clone(), p1_out.clone(), p2_out.clone()];
    let mut values = Vec::with_capacity(dimension.pow(4));
    for p2_out_value in 0..dimension {
        for p1_out_value in 0..dimension {
            for p2_value in 0..dimension {
                for p1_value in 0..dimension {
                    values.push(
                        gate[(
                            p1_out_value + dimension * p2_out_value,
                            p1_value + dimension * p2_value,
                        )],
                    );
                }
            }
        }
    }
    Tensor::from_dense(indices, values).unwrap()
}

struct TestSvd {
    u: Tensor,
    singular_values: Vec<f64>,
    bond: Idx,
    v: Tensor,
}

fn test_svd_bond(tensor: &Tensor, left: &[Idx], truncation: &Truncation) -> TestSvd {
    let policy = SvdTruncationPolicy::new(truncation.epsilon)
        .with_squared_values()
        .with_discarded_tail_sum();
    let mut options = SvdOptions::new().with_policy(policy);
    if let Some(max_bond) = truncation.max_bond {
        options = options.with_max_bond_dim(max_bond);
    }
    let (u, singular, v_raw) = svd_with::<Complex64>(tensor, left, &options).unwrap();
    let rank = singular.dims()[0];
    let singular_dense = dense_complex(&singular);
    let singular_values = (0..rank)
        .map(|index| singular_dense[index * (rank + 1)].re)
        .collect();
    let left_ids: Vec<_> = left.iter().map(|index| index.id).collect();
    let bond = u
        .indices
        .iter()
        .find(|index| !left_ids.contains(&index.id))
        .unwrap()
        .clone();
    let v = test_relabel(&v_raw, &bond, &singular.indices[1]);
    TestSvd {
        u,
        singular_values,
        bond,
        v,
    }
}

fn test_apply_gate_to_bond(
    gate: &DMatrix<Complex64>,
    x: &ComplexSite,
    y: &ComplexSite,
    outer_left: &[f64],
    middle: &[f64],
    outer_right: &[f64],
    truncation: &Truncation,
) -> (ComplexSite, ComplexSite, Vec<f64>, Idx, f64) {
    let right_environment = new_index(y.right.dim);
    let y_gamma = test_relabel(&y.gamma, &y.right, &right_environment);
    let x_gamma = test_scale_bond(
        &test_scale_bond(&x.gamma, &x.left, outer_left),
        &x.right,
        middle,
    );
    let y_gamma = test_scale_bond(&y_gamma, &right_environment, outer_right);
    let theta = test_binary_contract(&x_gamma, &y_gamma);

    let p1_out = new_index(x.phys.dim);
    let p2_out = new_index(y.phys.dim);
    let gate_tensor = test_gate_tensor(gate, &x.phys, &y.phys, &p1_out, &p2_out);
    let theta_gated = test_binary_contract(&theta, &gate_tensor);
    let result = test_svd_bond(
        &theta_gated,
        &[x.left.clone(), p1_out.clone(), x.anc.clone()],
        truncation,
    );
    let norm = result
        .singular_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(norm.is_finite() && norm > 0.0);
    let new_middle = result
        .singular_values
        .iter()
        .map(|value| value / norm)
        .collect();
    let inverse = |values: &[f64]| {
        values
            .iter()
            .map(|&value| if value > 1e-12 { 1.0 / value } else { 0.0 })
            .collect::<Vec<_>>()
    };
    let gamma_x = test_scale_bond(&result.u, &x.left, &inverse(outer_left));
    let gamma_y = test_scale_bond(&result.v.conj(), &right_environment, &inverse(outer_right));
    let simulated_bond = gamma_y.indices.last().unwrap().clone();
    let gamma_y = test_relabel(&gamma_y, &simulated_bond, &result.bond);
    let gamma_y = test_relabel(&gamma_y, &right_environment, &y.right);
    let new_phys_x = new_index(x.phys.dim);
    let new_phys_y = new_index(y.phys.dim);
    let gamma_x = test_relabel(&gamma_x, &p1_out, &new_phys_x);
    let gamma_y = test_relabel(&gamma_y, &p2_out, &new_phys_y);
    let gamma_y = gamma_y
        .permute_indices(&[
            result.bond.clone(),
            new_phys_y.clone(),
            y.anc.clone(),
            y.right.clone(),
        ])
        .unwrap();
    let new_x = ComplexSite {
        gamma: gamma_x,
        left: x.left.clone(),
        phys: new_phys_x,
        anc: x.anc.clone(),
        right: result.bond.clone(),
    };
    let new_y = ComplexSite {
        gamma: gamma_y,
        left: result.bond.clone(),
        phys: new_phys_y,
        anc: y.anc.clone(),
        right: y.right.clone(),
    };
    (new_x, new_y, new_middle, result.bond, norm.ln())
}

fn explicit_strang_step(
    state: &mut ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
    tau: f64,
    truncation: &Truncation,
) -> StepInfo {
    let half_gate = hamiltonian.trotter_gate(0.5 * tau).unwrap();
    let full_gate = hamiltonian.trotter_gate(tau).unwrap();
    let (a, b, lambda_ab, bond_ab, log_ab_1) = test_apply_gate_to_bond(
        &half_gate,
        &state.a,
        &state.b,
        &state.lambda_ba,
        &state.lambda_ab,
        &state.lambda_ba,
        truncation,
    );
    let ab1_dim = lambda_ab.len();
    let ab1_min = *lambda_ab.last().unwrap();
    state.a = a;
    state.b = b;
    state.lambda_ab = lambda_ab;
    state.lambda_bond_ab = bond_ab;

    let (b, a, lambda_ba, bond_ba, log_ba) = test_apply_gate_to_bond(
        &full_gate,
        &state.b,
        &state.a,
        &state.lambda_ab,
        &state.lambda_ba,
        &state.lambda_ab,
        truncation,
    );
    let ba_dim = lambda_ba.len();
    let ba_min = *lambda_ba.last().unwrap();
    state.b = b;
    state.a = a;
    state.lambda_ba = lambda_ba;
    state.lambda_bond_ba = bond_ba;

    let (a, b, lambda_ab, bond_ab, log_ab_2) = test_apply_gate_to_bond(
        &half_gate,
        &state.a,
        &state.b,
        &state.lambda_ba,
        &state.lambda_ab,
        &state.lambda_ba,
        truncation,
    );
    let ab2_dim = lambda_ab.len();
    let ab2_min = *lambda_ab.last().unwrap();
    state.a = a;
    state.b = b;
    state.lambda_ab = lambda_ab;
    state.lambda_bond_ab = bond_ab;

    StepInfo {
        max_bond: ab1_dim.max(ba_dim).max(ab2_dim),
        min_singular_value: ab1_min.min(ba_min).min(ab2_min),
        log_norm: 2.0 * (log_ab_1 + log_ba + log_ab_2),
    }
}

fn assert_complex_slice_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tolerance,
            "complex entry {position} differs: {actual:?} != {expected:?}"
        );
    }
}

fn assert_real_slice_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "real entry {position} differs: {actual} != {expected}"
        );
    }
}

fn assert_complex_states_close(
    actual: &ComplexPurifiedMps,
    expected: &ComplexPurifiedMps,
    tolerance: f64,
) {
    assert_eq!(actual.a.gamma.dims(), expected.a.gamma.dims());
    assert_eq!(actual.b.gamma.dims(), expected.b.gamma.dims());
    assert_eq!(actual.lambda_bond_ab.dim, expected.lambda_bond_ab.dim);
    assert_eq!(actual.lambda_bond_ba.dim, expected.lambda_bond_ba.dim);
    assert_complex_slice_close(
        &dense_complex(&actual.a.gamma),
        &dense_complex(&expected.a.gamma),
        tolerance,
    );
    assert_complex_slice_close(
        &dense_complex(&actual.b.gamma),
        &dense_complex(&expected.b.gamma),
        tolerance,
    );
    assert_real_slice_close(&actual.lambda_ab, &expected.lambda_ab, tolerance);
    assert_real_slice_close(&actual.lambda_ba, &expected.lambda_ba, tolerance);
}

fn flat_index(dimensions: &[usize], coordinates: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(dimensions)
        .fold(
            (0usize, 1usize),
            |(flat, stride), (&coordinate, &dimension)| {
                (flat + stride * coordinate, stride * dimension)
            },
        )
        .0
}

fn unit_phase(value: Complex64, context: &str) -> Complex64 {
    let magnitude = value.norm();
    assert!(
        magnitude.is_finite() && magnitude > 1e-24,
        "cannot resolve {context} gauge phase from overlap {value:?}"
    );
    value / magnitude
}

fn assert_nondegenerate_schmidt_spectrum(values: &[f64], bond: &str) {
    assert!(!values.is_empty());
    let scale = values[0];
    assert!(scale.is_finite() && scale > 0.0);
    for (position, pair) in values.windows(2).enumerate() {
        let relative_gap = (pair[0] - pair[1]).abs() / scale;
        assert!(
            relative_gap > 1e-8,
            "{bond} Schmidt values {position} and {} are not separated: {values:?}",
            position + 1
        );
    }
}

fn assert_gauge_aligned_complex_states_close(
    actual: &ComplexPurifiedMps,
    expected: &ComplexPurifiedMps,
    tolerance: f64,
) {
    assert_eq!(actual.a.gamma.dims(), expected.a.gamma.dims());
    assert_eq!(actual.b.gamma.dims(), expected.b.gamma.dims());
    assert_eq!(actual.lambda_bond_ab.dim, expected.lambda_bond_ab.dim);
    assert_eq!(actual.lambda_bond_ba.dim, expected.lambda_bond_ba.dim);
    assert_real_slice_close(&actual.lambda_ab, &expected.lambda_ab, tolerance);
    assert_real_slice_close(&actual.lambda_ba, &expected.lambda_ba, tolerance);

    // The fixture has separated Schmidt values, so each SVD vector is unique up to phase. The
    // only allowed gauge freedom is therefore a diagonal unitary on each virtual bond. Fit those
    // AB/BA phases from all gamma entries, fix the common phase by q_ba[0] = 1, and then compare
    // every aligned payload entry. This cannot hide a non-gauge tensor mutation.
    assert_nondegenerate_schmidt_spectrum(&actual.lambda_ab, "AB");
    assert_nondegenerate_schmidt_spectrum(&actual.lambda_ba, "BA");

    let a_dimensions = actual.a.gamma.dims();
    let b_dimensions = actual.b.gamma.dims();
    let actual_a = dense_complex(&actual.a.gamma);
    let actual_b = dense_complex(&actual.b.gamma);
    let expected_a = dense_complex(&expected.a.gamma);
    let expected_b = dense_complex(&expected.b.gamma);
    let ab_dimension = actual.lambda_ab.len();
    let ba_dimension = actual.lambda_ba.len();
    let mut q_ab = vec![Complex64::new(1.0, 0.0); ab_dimension];
    let mut q_ba = vec![Complex64::new(1.0, 0.0); ba_dimension];

    for _ in 0..12 {
        for right_ab in 0..ab_dimension {
            let mut overlap = Complex64::new(0.0, 0.0);
            for left_ba in 0..ba_dimension {
                for physical in 0..a_dimensions[1] {
                    for ancilla in 0..a_dimensions[2] {
                        let position =
                            flat_index(&a_dimensions, &[left_ba, physical, ancilla, right_ab]);
                        let base = q_ba[left_ba].conj() * expected_a[position];
                        overlap += base.conj() * actual_a[position];
                    }
                }
            }
            q_ab[right_ab] = unit_phase(overlap, "AB");
        }

        for right_ba in 0..ba_dimension {
            let mut overlap = Complex64::new(0.0, 0.0);
            for left_ab in 0..ab_dimension {
                for physical in 0..b_dimensions[1] {
                    for ancilla in 0..b_dimensions[2] {
                        let position =
                            flat_index(&b_dimensions, &[left_ab, physical, ancilla, right_ba]);
                        let base = q_ab[left_ab].conj() * expected_b[position];
                        overlap += base.conj() * actual_b[position];
                    }
                }
            }
            q_ba[right_ba] = unit_phase(overlap, "BA");
        }

        let anchor = q_ba[0].conj();
        for phase in q_ab.iter_mut().chain(&mut q_ba) {
            *phase *= anchor;
        }
    }

    let mut aligned_a = expected_a;
    for left_ba in 0..ba_dimension {
        for physical in 0..a_dimensions[1] {
            for ancilla in 0..a_dimensions[2] {
                for right_ab in 0..ab_dimension {
                    let position =
                        flat_index(&a_dimensions, &[left_ba, physical, ancilla, right_ab]);
                    aligned_a[position] *= q_ba[left_ba].conj() * q_ab[right_ab];
                }
            }
        }
    }
    let mut aligned_b = expected_b;
    for left_ab in 0..ab_dimension {
        for physical in 0..b_dimensions[1] {
            for ancilla in 0..b_dimensions[2] {
                for right_ba in 0..ba_dimension {
                    let position =
                        flat_index(&b_dimensions, &[left_ab, physical, ancilla, right_ba]);
                    aligned_b[position] *= q_ab[left_ab].conj() * q_ba[right_ba];
                }
            }
        }
    }

    assert_complex_slice_close(&actual_a, &aligned_a, tolerance);
    assert_complex_slice_close(&actual_b, &aligned_b, tolerance);
}

fn evolved_complex_bond_two_fixture() -> (ComplexPurifiedMps, ComplexLocalHamiltonian) {
    let hamiltonian = rotated_tfim();
    let truncation = Truncation {
        epsilon: 0.0,
        max_bond: Some(2),
    };
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    imaginary_time_step_second_order_complex(&mut state, &hamiltonian, 0.02, &truncation).unwrap();
    assert_eq!(state.lambda_ab.len(), 2);
    assert_eq!(state.lambda_ba.len(), 2);
    assert!(dense_complex(&state.a.gamma)
        .iter()
        .chain(dense_complex(&state.b.gamma).iter())
        .any(|value| value.im.abs() > 1e-8));
    (state, hamiltonian)
}

fn generic_complex_bond_two_fixture() -> ComplexPurifiedMps {
    let bond_ab = new_index(2);
    let bond_ba = new_index(2);
    let make_site = |left: &Idx, right: &Idx, phase: f64| {
        let phys = new_index(2);
        let anc = new_index(2);
        let values = (0..16)
            .map(|flat| {
                let x = flat as f64 + 1.0;
                Complex64::new(
                    0.07 * (0.31 * x + phase).cos() + 0.11,
                    0.09 * (0.47 * x - phase).sin() - 0.03,
                )
            })
            .collect::<Vec<_>>();
        ComplexSite {
            gamma: Tensor::from_dense(
                vec![left.clone(), phys.clone(), anc.clone(), right.clone()],
                values,
            )
            .unwrap(),
            left: left.clone(),
            phys,
            anc,
            right: right.clone(),
        }
    };
    let state = ComplexPurifiedMps {
        a: make_site(&bond_ba, &bond_ab, 0.23),
        b: make_site(&bond_ab, &bond_ba, -0.41),
        lambda_ab: vec![0.8, 0.6],
        lambda_bond_ab: bond_ab,
        lambda_ba: vec![0.5, 3.0_f64.sqrt() / 2.0],
        lambda_bond_ba: bond_ba,
    };
    state.validate_topology().unwrap();
    state
}

fn direct_two_site_norm_and_energy(
    state: &ComplexPurifiedMps,
    hamiltonian: &ComplexLocalHamiltonian,
) -> (f64, f64) {
    let a = dense_complex(&state.a.gamma);
    let b = dense_complex(&state.b.gamma);
    let a_dimensions = state.a.gamma.dims();
    let b_dimensions = state.b.gamma.dims();
    let d = state.a.phys.dim;
    let mut norm = 0.0;
    let mut energy = Complex64::new(0.0, 0.0);

    // Close the two-site cell periodically as
    // Tr[Gamma_A lambda_ab Gamma_B lambda_ba]. Both AB and BA gauge matrices therefore meet
    // their inverse across the trace. Unlike an open segment with lambda weights at both ends,
    // this direct finite-cell oracle is invariant under either virtual-bond gauge.
    for ancilla_a in 0..d {
        for ancilla_b in 0..d {
            let mut wavefunction = vec![Complex64::new(0.0, 0.0); d * d];
            for physical_b in 0..d {
                for physical_a in 0..d {
                    let physical = physical_a + d * physical_b;
                    for left in 0..state.lambda_ba.len() {
                        for middle in 0..state.lambda_ab.len() {
                            let a_position =
                                flat_index(&a_dimensions, &[left, physical_a, ancilla_a, middle]);
                            let b_position =
                                flat_index(&b_dimensions, &[middle, physical_b, ancilla_b, left]);
                            wavefunction[physical] += a[a_position]
                                * state.lambda_ab[middle]
                                * b[b_position]
                                * state.lambda_ba[left];
                        }
                    }
                }
            }
            norm += wavefunction.iter().map(Complex64::norm_sqr).sum::<f64>();
            for input in 0..d * d {
                for output in 0..d * d {
                    energy += wavefunction[output].conj()
                        * hamiltonian.two_site_h()[(output, input)]
                        * wavefunction[input];
                }
            }
        }
    }
    assert!(energy.im.abs() <= 1e-12 * energy.re.abs().max(1.0));
    (norm, energy.re / norm)
}

fn canonical_gram_residual(state: &ComplexPurifiedMps) -> (f64, f64) {
    let mut left_residual: f64 = 0.0;
    let mut right_residual: f64 = 0.0;
    for (site, lambda_left, lambda_right) in [
        (&state.a, &state.lambda_ba, &state.lambda_ab),
        (&state.b, &state.lambda_ab, &state.lambda_ba),
    ] {
        let values = dense_complex(&site.gamma);
        let dimensions = site.gamma.dims();
        for right in 0..site.right.dim {
            for right_prime in 0..site.right.dim {
                let mut gram = Complex64::new(0.0, 0.0);
                for left in 0..site.left.dim {
                    for physical in 0..site.phys.dim {
                        for ancilla in 0..site.anc.dim {
                            let value = lambda_left[left]
                                * values
                                    [flat_index(&dimensions, &[left, physical, ancilla, right])];
                            let value_prime = lambda_left[left]
                                * values[flat_index(
                                    &dimensions,
                                    &[left, physical, ancilla, right_prime],
                                )];
                            gram += value.conj() * value_prime;
                        }
                    }
                }
                let expected = if right == right_prime { 1.0 } else { 0.0 };
                left_residual = left_residual.max((gram - expected).norm());
            }
        }
        for left in 0..site.left.dim {
            for left_prime in 0..site.left.dim {
                let mut gram = Complex64::new(0.0, 0.0);
                for physical in 0..site.phys.dim {
                    for ancilla in 0..site.anc.dim {
                        for right in 0..site.right.dim {
                            let value = values
                                [flat_index(&dimensions, &[left, physical, ancilla, right])]
                                * lambda_right[right];
                            let value_prime = values
                                [flat_index(&dimensions, &[left_prime, physical, ancilla, right])]
                                * lambda_right[right];
                            gram += value * value_prime.conj();
                        }
                    }
                }
                let expected = if left == left_prime { 1.0 } else { 0.0 };
                right_residual = right_residual.max((gram - expected).norm());
            }
        }
    }
    (left_residual, right_residual)
}

fn apply_matrix_to_test_leg(
    tensor: &Tensor,
    bond: &Idx,
    matrix: &DMatrix<Complex64>,
    matrix_on_left: bool,
) -> Tensor {
    let dimensions = tensor.dims();
    let position = tensor
        .indices
        .iter()
        .position(|index| index.id == bond.id)
        .unwrap();
    let old_values = dense_complex(tensor);
    let mut new_values = vec![Complex64::new(0.0, 0.0); old_values.len()];
    let mut coordinates = vec![0usize; dimensions.len()];
    for output_flat in 0..new_values.len() {
        let mut residual = output_flat;
        for (coordinate, &dimension) in coordinates.iter_mut().zip(&dimensions) {
            *coordinate = residual % dimension;
            residual /= dimension;
        }
        let output = coordinates[position];
        for input in 0..bond.dim {
            coordinates[position] = input;
            let input_flat = flat_index(&dimensions, &coordinates);
            let factor = if matrix_on_left {
                matrix[(output, input)]
            } else {
                matrix[(input, output)]
            };
            new_values[output_flat] += factor * old_values[input_flat];
        }
        coordinates[position] = output;
    }
    Tensor::from_dense(tensor.indices.clone(), new_values).unwrap()
}

fn apply_nonreal_ab_gauge(state: &mut ComplexPurifiedMps) {
    // A nondegenerate diagonal Schmidt spectrum permits phase rotations, but not mixing, without
    // changing the fixed diagonal lambda. Distinct non-real phases make U^T = U differ
    // materially from U^H while preserving Gamma_A lambda Gamma_B exactly.
    let unitary = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
        Complex64::from_polar(1.0, 0.43),
        Complex64::from_polar(1.0, -0.71),
    ]));
    assert!(max_entrywise_norm(&(&unitary.adjoint() - unitary.transpose())) > 0.5);
    state.a.gamma = apply_matrix_to_test_leg(&state.a.gamma, &state.a.right, &unitary, false);
    state.b.gamma =
        apply_matrix_to_test_leg(&state.b.gamma, &state.b.left, &unitary.adjoint(), true);
    state.validate_topology().unwrap();
}

fn assert_f64_bits_equal(actual: f64, expected: f64, context: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context}: {actual:?} != {expected:?}"
    );
}

fn assert_report_bits_equal(actual: &SpecificHeatReport, expected: &SpecificHeatReport) {
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
            "onsite contribution real",
            actual.onsite_contribution.re,
            expected.onsite_contribution.re,
        ),
        (
            "onsite contribution imaginary",
            actual.onsite_contribution.im,
            expected.onsite_contribution.im,
        ),
        (
            "positive-direction contribution real",
            actual.positive_direction_contribution.re,
            expected.positive_direction_contribution.re,
        ),
        (
            "positive-direction contribution imaginary",
            actual.positive_direction_contribution.im,
            expected.positive_direction_contribution.im,
        ),
        (
            "negative-direction contribution real",
            actual.negative_direction_contribution.re,
            expected.negative_direction_contribution.re,
        ),
        (
            "negative-direction contribution imaginary",
            actual.negative_direction_contribution.im,
            expected.negative_direction_contribution.im,
        ),
        (
            "parity-A contribution real",
            actual.parity_a_contribution.re,
            expected.parity_a_contribution.re,
        ),
        (
            "parity-A contribution imaginary",
            actual.parity_a_contribution.im,
            expected.parity_a_contribution.im,
        ),
        (
            "parity-B contribution real",
            actual.parity_b_contribution.re,
            expected.parity_b_contribution.re,
        ),
        (
            "parity-B contribution imaginary",
            actual.parity_b_contribution.im,
            expected.parity_b_contribution.im,
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
        assert_f64_bits_equal(actual, expected, name);
    }
    assert_eq!(actual.max_distance, expected.max_distance);
    assert_eq!(actual.stop_reason, expected.stop_reason);
}

fn assert_real_site_bitwise_equal(actual: &Site, expected: &Site, context: &str) {
    assert_eq!(actual.gamma.dims(), expected.gamma.dims(), "{context} dims");
    let actual_values = actual.gamma.to_vec::<f64>().unwrap();
    let expected_values = expected.gamma.to_vec::<f64>().unwrap();
    assert_eq!(actual_values.len(), expected_values.len());
    for (position, (&actual, &expected)) in actual_values.iter().zip(&expected_values).enumerate() {
        assert_f64_bits_equal(actual, expected, &format!("{context}[{position}]"));
    }
}

fn assert_real_state_bitwise_equal(actual: &PurifiedMps, expected: &PurifiedMps) {
    assert_real_site_bitwise_equal(&actual.a, &expected.a, "A.gamma");
    assert_real_site_bitwise_equal(&actual.b, &expected.b, "B.gamma");
    assert_eq!(actual.lambda_ab.len(), expected.lambda_ab.len());
    assert_eq!(actual.lambda_ba.len(), expected.lambda_ba.len());
    for (position, (&actual, &expected)) in
        actual.lambda_ab.iter().zip(&expected.lambda_ab).enumerate()
    {
        assert_f64_bits_equal(actual, expected, &format!("lambda_ab[{position}]"));
    }
    for (position, (&actual, &expected)) in
        actual.lambda_ba.iter().zip(&expected.lambda_ba).enumerate()
    {
        assert_f64_bits_equal(actual, expected, &format!("lambda_ba[{position}]"));
    }
}

fn assert_complex_state_bitwise_equal(actual: &ComplexPurifiedMps, expected: &ComplexPurifiedMps) {
    assert_eq!(actual.a.gamma.dims(), expected.a.gamma.dims());
    assert_eq!(actual.b.gamma.dims(), expected.b.gamma.dims());
    for (context, actual, expected) in [
        (
            "A.gamma",
            dense_complex(&actual.a.gamma),
            dense_complex(&expected.a.gamma),
        ),
        (
            "B.gamma",
            dense_complex(&actual.b.gamma),
            dense_complex(&expected.b.gamma),
        ),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (position, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_f64_bits_equal(actual.re, expected.re, &format!("{context}[{position}].re"));
            assert_f64_bits_equal(actual.im, expected.im, &format!("{context}[{position}].im"));
        }
    }
    for (context, actual, expected) in [
        ("lambda_ab", &actual.lambda_ab, &expected.lambda_ab),
        ("lambda_ba", &actual.lambda_ba, &expected.lambda_ba),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (position, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_f64_bits_equal(actual, expected, &format!("{context}[{position}]"));
        }
    }
}

fn assert_step_info_bitwise_equal(actual: StepInfo, expected: StepInfo) {
    assert_eq!(actual.max_bond, expected.max_bond);
    assert_f64_bits_equal(
        actual.min_singular_value,
        expected.min_singular_value,
        "StepInfo.min_singular_value",
    );
    assert_f64_bits_equal(actual.log_norm, expected.log_norm, "StepInfo.log_norm");
}

fn exact_real_auto_hamiltonian(real: &LocalHamiltonian) -> ItebdHamiltonian {
    ItebdHamiltonian::try_from_complex(
        real.two_site_h.map(|value| Complex64::new(value, -0.0)),
        real.site_energy.map(|value| Complex64::new(value, 0.0)),
    )
    .unwrap()
}

#[test]
fn auto_real_dispatch_is_bitwise_identical() {
    // Mutations caught: routing either automatic real step through Complex64, changing a real
    // facade observable implementation, or rebuilding a direct result through complex storage.
    let real_hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let automatic_hamiltonian = exact_real_auto_hamiltonian(&real_hamiltonian);
    assert!(matches!(automatic_hamiltonian, ItebdHamiltonian::Real(_)));
    let truncation = Truncation {
        epsilon: 1e-13,
        max_bond: Some(16),
    };
    let operator = pauli_x().map(|value| Complex64::new(value, -0.0));

    for second_order in [false, true] {
        let mut direct = infinite_temperature(2);
        let mut automatic = ItebdState::infinite_temperature(&automatic_hamiltonian).unwrap();
        let direct_info = if second_order {
            imaginary_time_step_second_order(&mut direct, &real_hamiltonian, 0.04, &truncation)
        } else {
            imaginary_time_step(&mut direct, &real_hamiltonian, 0.04, &truncation)
        };
        let automatic_info = if second_order {
            imaginary_time_step_second_order_auto(
                &mut automatic,
                &automatic_hamiltonian,
                0.04,
                &truncation,
            )
            .unwrap()
        } else {
            imaginary_time_step_auto(&mut automatic, &automatic_hamiltonian, 0.04, &truncation)
                .unwrap()
        };
        assert_step_info_bitwise_equal(automatic_info, direct_info);
        let automatic_real = match &automatic {
            ItebdState::Real(state) => state,
            ItebdState::Complex(_) => panic!("exact-real Hamiltonian selected complex state"),
        };
        assert_real_state_bitwise_equal(automatic_real, &direct);
        assert_f64_bits_equal(
            energy_density_auto(&automatic, &automatic_hamiltonian).unwrap(),
            energy_density(&direct, &real_hamiltonian),
            "energy density",
        );
        assert_f64_bits_equal(
            local_expectation_auto(&automatic, &automatic_hamiltonian, &operator, 1e-12).unwrap(),
            magnetization(&direct, &pauli_x()),
            "local expectation",
        );
    }
}

#[test]
fn auto_real_local_expectation_checks_the_complex_scalar_residual() {
    // Mutation caught: projecting a tolerance-admissible complex operator to its real matrix
    // part before evaluating the scalar would return 0.0 for the rejected counterexample.
    let real_hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let automatic_hamiltonian = exact_real_auto_hamiltonian(&real_hamiltonian);
    let state = ItebdState::infinite_temperature(&automatic_hamiltonian).unwrap();
    let tolerance = 1e-12;

    let rejected = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
        Complex64::new(1e12, 0.4),
        Complex64::new(-1e12, 0.4),
    ]));
    assert!(matches!(
        local_expectation_auto(
            &state,
            &automatic_hamiltonian,
            &rejected,
            tolerance,
        ),
        Err(ItebdError::NonRealObservable {
            observable: "automatic local observable",
            imaginary,
            tolerance: 1e-12,
        }) if (imaginary - 0.4).abs() < 1e-15
    ));

    let accepted = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
        Complex64::new(1e12, 0.4),
        Complex64::new(1e12, 0.4),
    ]));
    let accepted_value =
        local_expectation_auto(&state, &automatic_hamiltonian, &accepted, tolerance).unwrap();
    assert!(
        (accepted_value - 1e12).abs() < 2e-4,
        "relative scalar acceptance returned {accepted_value}"
    );
}

#[test]
fn auto_complex_success_arms_match_direct_complex_apis() {
    // Mutations caught: replacing any Complex/Complex facade arm with a default success value,
    // the wrong step order, or a real-backend call changes a direct-vs-facade result.
    let hamiltonian = phase_rotated_tfim();
    let automatic_hamiltonian = ItebdHamiltonian::Complex(hamiltonian.clone());
    let truncation = Truncation {
        epsilon: 1e-13,
        max_bond: Some(16),
    };
    let operator = phase_rotated_pauli_x();

    for second_order in [false, true] {
        let mut direct = ComplexPurifiedMps::infinite_temperature(2).unwrap();
        let mut automatic = ItebdState::infinite_temperature(&automatic_hamiltonian).unwrap();
        let direct_info = if second_order {
            imaginary_time_step_second_order_complex(&mut direct, &hamiltonian, 0.04, &truncation)
                .unwrap()
        } else {
            imaginary_time_step_complex(&mut direct, &hamiltonian, 0.04, &truncation).unwrap()
        };
        let automatic_info = if second_order {
            imaginary_time_step_second_order_auto(
                &mut automatic,
                &automatic_hamiltonian,
                0.04,
                &truncation,
            )
            .unwrap()
        } else {
            imaginary_time_step_auto(&mut automatic, &automatic_hamiltonian, 0.04, &truncation)
                .unwrap()
        };
        assert_step_info_bitwise_equal(automatic_info, direct_info);
        let automatic_complex = match &automatic {
            ItebdState::Complex(state) => state,
            ItebdState::Real(_) => panic!("complex Hamiltonian selected real state"),
        };
        assert_complex_state_bitwise_equal(automatic_complex, &direct);
        assert_f64_bits_equal(
            energy_density_auto(&automatic, &automatic_hamiltonian).unwrap(),
            energy_density_complex(&direct, &hamiltonian).unwrap(),
            "complex energy facade",
        );
        assert_f64_bits_equal(
            local_expectation_auto(&automatic, &automatic_hamiltonian, &operator, 1e-12).unwrap(),
            local_expectation_complex(&direct, &operator, 1e-12).unwrap(),
            "complex local facade",
        );

        let mut direct_canonical = direct.clone();
        let mut automatic_canonical = automatic;
        let direct_log = canonicalize_complex(&mut direct_canonical, 1e-12).unwrap();
        let automatic_log =
            canonicalize_auto(&mut automatic_canonical, &automatic_hamiltonian).unwrap();
        assert_f64_bits_equal(automatic_log, direct_log, "complex canonicalization facade");
        let automatic_complex = match &automatic_canonical {
            ItebdState::Complex(state) => state,
            ItebdState::Real(_) => unreachable!(),
        };
        assert_complex_state_bitwise_equal(automatic_complex, &direct_canonical);
    }
}

#[test]
fn complex_unitary_rotation_preserves_energy_and_magnetization() {
    // Mutations caught: transpose in place of adjoint, either omitted bra conjugation, or the
    // opposite two-site first-index-fastest operator convention.
    let real_hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let complex_hamiltonian = phase_rotated_tfim();
    let truncation = Truncation {
        epsilon: 1e-13,
        max_bond: Some(24),
    };
    let mut real_state = infinite_temperature(2);
    let mut complex_state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    for _ in 0..8 {
        imaginary_time_step_second_order(&mut real_state, &real_hamiltonian, 0.025, &truncation);
        imaginary_time_step_second_order_complex(
            &mut complex_state,
            &complex_hamiltonian,
            0.025,
            &truncation,
        )
        .unwrap();
    }

    let real_energy = energy_density(&real_state, &real_hamiltonian);
    let complex_energy = energy_density_complex(&complex_state, &complex_hamiltonian).unwrap();
    assert!(
        (real_energy - complex_energy).abs() < 2e-11,
        "rotated energy differs: real={real_energy} complex={complex_energy}"
    );
    let real_magnetization = magnetization(&real_state, &pauli_x());
    let complex_magnetization =
        local_expectation_complex(&complex_state, &phase_rotated_pauli_x(), 1e-12).unwrap();
    assert!(
        (real_magnetization - complex_magnetization).abs() < 2e-11,
        "rotated magnetization differs: real={real_magnetization} complex={complex_magnetization}"
    );
}

#[test]
fn complex_rotated_tfim_matches_exact_thermodynamics() {
    let beta = 1.0;
    let dtau = 0.05;
    let cap = 64;
    let sample = evolve_phase_rotated_tfim_second_order(
        beta,
        dtau,
        Truncation {
            epsilon: 1e-13,
            max_bond: Some(cap),
        },
    );
    let exact_energy = thermal_imps_purification::exact::exact_energy_density(1.0, 0.7, beta, 16_000);
    let exact_free_energy = thermal_imps_purification::exact::free_energy_density(1.0, 0.7, beta, 16_000);
    let energy_error = (sample.energy - exact_energy).abs();
    let free_energy_error = (sample.free_energy - exact_free_energy).abs();

    println!(
        "complex rotated TFIM: beta={beta:.3e} dtau={dtau:.3e} energy={:.16e} exact_energy={exact_energy:.16e} energy_error={energy_error:.3e} free_energy={:.16e} exact_free_energy={exact_free_energy:.16e} free_energy_error={free_energy_error:.3e} log_norm={:.16e} max_bond={}",
        sample.energy, sample.free_energy, sample.log_norm_per_site, sample.max_bond,
    );

    for (name, value) in [
        ("energy", sample.energy),
        ("free energy", sample.free_energy),
        ("log norm per site", sample.log_norm_per_site),
        ("minimum Schmidt value", sample.min_singular_value),
    ] {
        assert!(value.is_finite(), "{name} is non-finite: {value}");
    }
    assert!(sample.max_bond < cap, "routine run hit bond cap {cap}");
    // The established real TFIM routine at the same beta, dtau, cutoff, cap, and
    // once-per-complete-step canonicalization accepts both thermodynamic errors below 5e-3.
    assert!(
        energy_error < 5e-3,
        "complex rotated TFIM energy error {energy_error:.3e} exceeds the qualified real-case tolerance",
    );
    assert!(
        free_energy_error < 5e-3,
        "complex rotated TFIM free-energy error {free_energy_error:.3e} exceeds the qualified real-case tolerance",
    );
}

#[test]
fn complex_local_observable_rejects_nonhermitian_input() {
    // Mutation caught: skipping observable Hermiticity validation would contract a changed,
    // non-physical operator and return its real projection.
    let state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let mut operator = phase_rotated_pauli_x();
    operator[(0, 1)] += Complex64::new(0.25, 0.0);
    assert!(matches!(
        local_expectation_complex(&state, &operator, 1e-12),
        Err(ItebdError::NonHermitian {
            field: "local observable",
            residual,
            tolerance: 1e-12,
        }) if residual.is_finite() && residual > 1e-12
    ));
}

#[test]
fn complex_local_observable_validates_shape_finiteness_and_tolerance() {
    // Mutations caught: allowing an incompatible shape reaches tensor construction/indexing;
    // allowing NaN reaches contraction; accepting NaN tolerance bypasses ordered validation.
    let state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let nonsquare = DMatrix::<Complex64>::zeros(2, 3);
    assert!(matches!(
        local_expectation_complex(&state, &nonsquare, 1e-12),
        Err(ItebdError::NonSquare {
            field: "local observable",
            rows: 2,
            cols: 3,
        })
    ));
    let wrong_dimension = DMatrix::<Complex64>::identity(3, 3);
    assert!(matches!(
        local_expectation_complex(&state, &wrong_dimension, 1e-12),
        Err(ItebdError::TensorOperation {
            stage: "complex_observable",
            ..
        })
    ));
    let mut nonfinite = DMatrix::<Complex64>::identity(2, 2);
    nonfinite[(1, 0)].im = f64::NAN;
    assert!(matches!(
        local_expectation_complex(&state, &nonfinite, 1e-12),
        Err(ItebdError::NonFiniteMatrix {
            field: "local observable",
            row: 1,
            column: 0,
        })
    ));
    assert!(matches!(
        local_expectation_complex(&state, &DMatrix::identity(2, 2), f64::NAN),
        Err(ItebdError::InvalidTolerance {
            name: "local observable Hermiticity tolerance",
            value,
        }) if value.is_nan()
    ));
}

#[test]
fn automatic_facade_preserves_exact_real_specific_heat_reports() {
    // Mutations caught: converting exact-real data through the complex backend, selecting a
    // different report field, or applying different default options in either facade method.
    let real_hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let auto_real_hamiltonian = exact_real_auto_hamiltonian(&real_hamiltonian);
    let real_state = ItebdState::infinite_temperature(&auto_real_hamiltonian).unwrap();
    let options = SpecificHeatOptions::default();
    let (direct, direct_specific_heat) = match &real_state {
        ItebdState::Real(state) => (
            specific_heat_with_options(state, &real_hamiltonian, 0.7, &options).unwrap(),
            specific_heat(state, &real_hamiltonian, 0.7).unwrap(),
        ),
        ItebdState::Complex(_) => unreachable!(),
    };
    let automatic =
        specific_heat_with_options_auto(&real_state, &auto_real_hamiltonian, 0.7, &options)
            .unwrap();
    assert_report_bits_equal(&automatic, &direct);
    assert_f64_bits_equal(
        specific_heat_auto(&real_state, &auto_real_hamiltonian, 0.7).unwrap(),
        direct_specific_heat,
        "specific heat",
    );
}

#[test]
fn automatic_facade_reports_specific_heat_backend_errors() {
    // Mutations caught: a report facade that bypasses either cross-backend mismatch arm.
    let real_hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let auto_real_hamiltonian = exact_real_auto_hamiltonian(&real_hamiltonian);
    let complex_hamiltonian = ItebdHamiltonian::Complex(phase_rotated_tfim());
    let mut real_state = ItebdState::infinite_temperature(&auto_real_hamiltonian).unwrap();
    let mut complex_state = ItebdState::infinite_temperature(&complex_hamiltonian).unwrap();
    let options = SpecificHeatOptions::default();

    let truncation = Truncation::default();
    let operator = phase_rotated_pauli_x();
    for error in [
        imaginary_time_step_auto(&mut real_state, &complex_hamiltonian, 0.01, &truncation)
            .unwrap_err(),
        imaginary_time_step_second_order_auto(
            &mut real_state,
            &complex_hamiltonian,
            0.01,
            &truncation,
        )
        .unwrap_err(),
        canonicalize_auto(&mut real_state, &complex_hamiltonian).unwrap_err(),
        energy_density_auto(&real_state, &complex_hamiltonian).unwrap_err(),
        local_expectation_auto(&real_state, &complex_hamiltonian, &operator, 1e-12).unwrap_err(),
        specific_heat_auto(&real_state, &complex_hamiltonian, 0.7).unwrap_err(),
        specific_heat_with_options_auto(&real_state, &complex_hamiltonian, 0.7, &options)
            .unwrap_err(),
    ] {
        assert!(matches!(
            error,
            ItebdError::BackendMismatch {
                state: "real",
                hamiltonian: "complex"
            }
        ));
    }
    for error in [
        imaginary_time_step_auto(
            &mut complex_state,
            &auto_real_hamiltonian,
            0.01,
            &truncation,
        )
        .unwrap_err(),
        imaginary_time_step_second_order_auto(
            &mut complex_state,
            &auto_real_hamiltonian,
            0.01,
            &truncation,
        )
        .unwrap_err(),
        canonicalize_auto(&mut complex_state, &auto_real_hamiltonian).unwrap_err(),
        energy_density_auto(&complex_state, &auto_real_hamiltonian).unwrap_err(),
        local_expectation_auto(&complex_state, &auto_real_hamiltonian, &operator, 1e-12)
            .unwrap_err(),
        specific_heat_auto(&complex_state, &auto_real_hamiltonian, 0.7).unwrap_err(),
        specific_heat_with_options_auto(&complex_state, &auto_real_hamiltonian, 0.7, &options)
            .unwrap_err(),
    ] {
        assert!(matches!(
            error,
            ItebdError::BackendMismatch {
                state: "complex",
                hamiltonian: "real"
            }
        ));
    }
}

#[test]
fn complex_canonicalization_preserves_genuinely_complex_state() {
    // Mutations caught: deleting canonicalization leaves the Gram residuals large, while
    // replacing any gauge adjoint with transpose breaks the non-real virtual-gauge comparison.
    let (mut state, hamiltonian) = evolved_complex_bond_two_fixture();
    let mut gauged = state.clone();
    apply_nonreal_ab_gauge(&mut gauged);
    let dimensions = (state.lambda_ab.len(), state.lambda_ba.len());
    let before = direct_two_site_norm_and_energy(&state, &hamiltonian);
    let gauged_before = direct_two_site_norm_and_energy(&gauged, &hamiltonian);
    assert!((before.0 - gauged_before.0).abs() < 1e-12);
    assert!((before.1 - gauged_before.1).abs() < 1e-12);

    assert_eq!(canonicalize_complex(&mut state, 1e-12).unwrap(), 0.0);
    assert_eq!(canonicalize_complex(&mut gauged, 1e-12).unwrap(), 0.0);
    assert_eq!((state.lambda_ab.len(), state.lambda_ba.len()), dimensions);
    assert_eq!((gauged.lambda_ab.len(), gauged.lambda_ba.len()), dimensions);
    let residual = canonical_gram_residual(&state);
    let gauged_residual = canonical_gram_residual(&gauged);
    assert!(residual.0 < 1e-10, "left Gram residual {}", residual.0);
    assert!(residual.1 < 1e-10, "right Gram residual {}", residual.1);
    assert!(
        gauged_residual.0 < 1e-10,
        "gauged left Gram residual {}",
        gauged_residual.0
    );
    assert!(
        gauged_residual.1 < 1e-10,
        "gauged right Gram residual {}",
        gauged_residual.1
    );
    let after = direct_two_site_norm_and_energy(&state, &hamiltonian);
    let gauged_after = direct_two_site_norm_and_energy(&gauged, &hamiltonian);
    assert!(
        (before.0 - after.0).abs() < 1e-10,
        "periodic norm changed: before={} after={}",
        before.0,
        after.0
    );
    assert!(
        (before.1 - after.1).abs() < 1e-10,
        "periodic energy changed: before={} after={}",
        before.1,
        after.1
    );
    assert!((after.0 - gauged_after.0).abs() < 1e-10);
    assert!((after.1 - gauged_after.1).abs() < 1e-10);
    assert_gauge_aligned_complex_states_close(&state, &gauged, 1e-10);

    let once = state.clone();
    assert_eq!(canonicalize_complex(&mut state, 1e-12).unwrap(), 0.0);
    assert_gauge_aligned_complex_states_close(&state, &once, 1e-11);
}

#[test]
fn complex_canonicalization_accounts_for_generic_state_scalar() {
    // A generic valid state need not already have unit transfer scale. Canonicalization may
    // normalize that scalar, but the returned log norm must account for the removed two-site-cell
    // amplitude: norm_before = norm_after * exp(2 * log_norm).
    let hamiltonian = rotated_tfim();
    let mut state = generic_complex_bond_two_fixture();
    let before = direct_two_site_norm_and_energy(&state, &hamiltonian);
    let log_norm = canonicalize_complex(&mut state, 1e-12).unwrap();
    let after = direct_two_site_norm_and_energy(&state, &hamiltonian);
    let accounted_norm = after.0 * (2.0 * log_norm).exp();
    assert!(
        (before.0 - accounted_norm).abs() <= 1e-10 * before.0.max(1.0),
        "generic periodic norm is not accounted: before={} after={} log_norm={} accounted={}",
        before.0,
        after.0,
        log_norm,
        accounted_norm
    );
    assert!(
        (before.1 - after.1).abs() < 1e-10,
        "generic normalized energy changed: before={} after={}",
        before.1,
        after.1
    );
    let residual = canonical_gram_residual(&state);
    assert!(residual.0 < 1e-10, "left Gram residual {}", residual.0);
    assert!(residual.1 < 1e-10, "right Gram residual {}", residual.1);
    assert!(log_norm.abs() > 1e-3, "fixture must have material scalar");
}

#[test]
fn complex_canonicalization_rejects_invalid_tolerance() {
    // Mutation caught: accepting NaN would let a non-orderable Hermiticity threshold reach the
    // eigendecomposition boundary.
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    assert!(matches!(
        canonicalize_complex(&mut state, f64::NAN),
        Err(ItebdError::InvalidTolerance {
            name: "complex canonicalization tolerance",
            value
        }) if value.is_nan()
    ));
}

#[test]
fn complex_canonicalization_rejects_zero_fixed_point() {
    // Mutation caught: allowing a zero transfer iterate through factorization would create a
    // zero Schmidt norm or non-finite inverse instead of a typed fixed-point failure.
    let (mut state, _) = evolved_complex_bond_two_fixture();
    state.a.gamma = Tensor::from_dense(
        state.a.gamma.indices.clone(),
        vec![Complex64::new(0.0, 0.0); dense_complex(&state.a.gamma).len()],
    )
    .unwrap();
    let error = canonicalize_complex(&mut state, 1e-12).unwrap_err();
    assert!(matches!(error, ItebdError::TensorOperation { .. }));
}

#[test]
fn complex_zero_time_step_preserves_state() {
    let hamiltonian = rotated_tfim();
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let expected = state.clone();

    let info = imaginary_time_step_complex(
        &mut state,
        &hamiltonian,
        0.0,
        &Truncation {
            epsilon: 1e-13,
            max_bond: Some(32),
        },
    )
    .unwrap();

    assert_complex_states_close(&state, &expected, 1e-13);
    assert_eq!(info.max_bond, 1);
    assert!((info.min_singular_value - 1.0).abs() <= 1e-13);
    assert!(info.log_norm.abs() <= 1e-13);
}

#[test]
fn complex_second_order_matches_explicit_strang() {
    let (state, hamiltonian) = nondegenerate_complex_fixture();
    let truncation = Truncation {
        epsilon: 1e-10,
        max_bond: Some(16),
    };
    let mut actual = state;
    let mut expected = actual.clone();

    let actual_info =
        imaginary_time_step_second_order_complex(&mut actual, &hamiltonian, 0.04, &truncation)
            .unwrap();
    let expected_info = explicit_strang_step(&mut expected, &hamiltonian, 0.04, &truncation);

    assert_gauge_aligned_complex_states_close(&actual, &expected, 1e-11);
    assert_eq!(actual_info.max_bond, expected_info.max_bond);
    assert!((actual_info.min_singular_value - expected_info.min_singular_value).abs() <= 1e-13);
    assert!((actual_info.log_norm - expected_info.log_norm).abs() <= 1e-13);
}

#[test]
fn complex_rotated_tfim_stays_finite_over_twenty_five_steps() {
    let hamiltonian = rotated_tfim();
    let truncation = Truncation {
        epsilon: 1e-13,
        max_bond: Some(32),
    };
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();

    for _ in 0..25 {
        let info =
            imaginary_time_step_second_order_complex(&mut state, &hamiltonian, 0.02, &truncation)
                .unwrap();
        assert!(info.log_norm.is_finite());
        assert!(info.min_singular_value.is_finite());
        assert!(info.max_bond <= 32);
    }

    state.validate_topology().unwrap();
    assert!(!state.lambda_ab.is_empty());
    assert!(!state.lambda_ba.is_empty());
    assert!(state
        .lambda_ab
        .iter()
        .chain(&state.lambda_ba)
        .all(|value| value.is_finite()));
    assert!(dense_complex(&state.a.gamma)
        .iter()
        .chain(dense_complex(&state.b.gamma).iter())
        .all(|value| value.re.is_finite() && value.im.is_finite()));
    assert!(state.lambda_ab.len() <= 32);
    assert!(state.lambda_ba.len() <= 32);
}

#[test]
fn complex_transfer_fixed_points_close_to_unit_norm() {
    let hamiltonian = rotated_tfim();
    let truncation = Truncation {
        epsilon: 1e-13,
        max_bond: Some(24),
    };
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    for _ in 0..12 {
        imaginary_time_step_second_order_complex(&mut state, &hamiltonian, 0.02, &truncation)
            .unwrap();
    }

    let left = dominant_fixed_point_left(&state, 1e-12, 500).unwrap();
    let right = dominant_fixed_point_right(&state, 1e-12, 500).unwrap();
    for fixed_point in [left, right] {
        let norm = fixed_point
            .matrix
            .to_vec::<Complex64>()
            .unwrap()
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 2e-12, "fixed-point norm {norm:e}");
        assert!(fixed_point.eigenvalue.is_finite());
        assert!(fixed_point.eigenvalue > 0.0);
    }
}

#[test]
fn complex_infinite_temperature_has_complex_storage_and_unit_norm() {
    let state = ComplexPurifiedMps::infinite_temperature(3).unwrap();

    assert!(state.a.gamma.is_complex());
    assert!(state.b.gamma.is_complex());
    state.validate_topology().unwrap();

    let a_norm: f64 = state
        .a
        .gamma
        .to_vec::<Complex64>()
        .unwrap()
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    let b_norm: f64 = state
        .b
        .gamma
        .to_vec::<Complex64>()
        .unwrap()
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    let finite_cell_norm = a_norm
        * b_norm
        * state
            .lambda_ab
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
        * state
            .lambda_ba
            .iter()
            .map(|value| value * value)
            .sum::<f64>();

    assert!((finite_cell_norm - 1.0).abs() < 1e-12);
}

#[test]
fn complex_infinite_temperature_rejects_zero_dimension() {
    assert!(matches!(
        ComplexPurifiedMps::infinite_temperature(0),
        Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 0 })
    ));
}

#[test]
fn complex_state_topology_validation_returns_typed_error() {
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    state.b.left = new_index(1);

    assert!(matches!(
        state.validate_topology(),
        Err(ItebdError::TensorOperation {
            stage: "complex_state_topology",
            message,
        }) if message.contains("B.left")
    ));
}

#[test]
fn complex_state_topology_rejects_declared_index_dimension_mismatch() {
    // Mutation caught: relying on Idx equality alone accepts the same id/tags/prime with a
    // different public dimension than the corresponding gamma leg.
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    state.b.left.dim = 2;

    assert!(matches!(
        state.validate_topology(),
        Err(ItebdError::TensorOperation {
            stage: "complex_state_topology",
            message,
        }) if message.contains("B.gamma index 0") && message.contains("dimension")
    ));
}

#[test]
fn complex_state_topology_rejects_shared_bond_dimension_mismatch() {
    // Mutation caught: removing the explicit dimension comparison from the cross-site bond check
    // accepts A.right and B.left when identity metadata matches but dimensions differ.
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    state.b.left.dim = 2;
    state.b.gamma.indices[0].dim = 2;

    assert!(matches!(
        state.validate_topology(),
        Err(ItebdError::TensorOperation {
            stage: "complex_state_topology",
            message,
        }) if message.contains("A.right")
            && message.contains("B.left")
            && message.contains("dimension")
    ));
}

#[test]
fn complex_state_validation_rejects_nonfinite_tensor_values() {
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let mut data = state.a.gamma.to_vec::<Complex64>().unwrap();
    data[1] = Complex64::new(0.0, f64::NAN);
    state.a.gamma =
        thermal_imps_purification::tensor::Tensor::from_dense(state.a.gamma.indices.clone(), data).unwrap();

    assert!(matches!(
        state.validate_topology(),
        Err(ItebdError::TensorOperation {
            stage: "complex_state_topology",
            message,
        }) if message.contains("non-finite")
    ));
}

#[test]
fn exact_zero_imaginary_parts_dispatch_to_real() {
    let real = Tfim { j: 1.0, g: 0.7 }.local();
    let mut two = real.two_site_h.map(|x| Complex64::new(x, 0.0));
    two[(0, 0)].im = -0.0;
    let site = real.site_energy.map(|x| Complex64::new(x, -0.0));

    let classified = ItebdHamiltonian::try_from_complex(two, site).unwrap();

    assert!(matches!(classified, ItebdHamiltonian::Real(_)));
    assert_eq!(classified.dim(), 2);
}

#[test]
fn one_nonzero_imaginary_component_selects_complex() {
    let mut h = DMatrix::<Complex64>::identity(4, 4);
    h[(0, 1)] = Complex64::new(0.0, 1e-300);
    h[(1, 0)] = h[(0, 1)].conj();

    let classified = ItebdHamiltonian::try_from_complex(h.clone(), h).unwrap();

    assert!(matches!(classified, ItebdHamiltonian::Complex(_)));
    assert_eq!(classified.dim(), 2);
}

#[test]
fn exact_real_facade_input_still_passes_shared_validation() {
    let malformed = DMatrix::<Complex64>::identity(2, 2);

    assert!(matches!(
        ItebdHamiltonian::try_from_complex(malformed.clone(), malformed),
        Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 2 })
    ));
}

#[test]
fn complex_hamiltonian_rejects_nonsquare_matrix() {
    let two = DMatrix::<Complex64>::zeros(4, 3);
    let site = DMatrix::<Complex64>::zeros(4, 4);

    assert!(matches!(
        ComplexLocalHamiltonian::try_new(two, site),
        Err(ItebdError::NonSquare {
            field: "two_site_h",
            rows: 4,
            cols: 3,
        })
    ));
}

#[test]
fn complex_hamiltonian_rejects_non_square_physical_dimension() {
    let matrix = DMatrix::<Complex64>::identity(2, 2);

    assert!(matches!(
        ComplexLocalHamiltonian::try_new(matrix.clone(), matrix),
        Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 2 })
    ));
}

#[test]
fn complex_hamiltonian_rejects_mismatched_matrix_dimensions() {
    let two = DMatrix::<Complex64>::identity(4, 4);
    let site = DMatrix::<Complex64>::identity(9, 9);

    assert!(matches!(
        ComplexLocalHamiltonian::try_new(two, site),
        Err(ItebdError::MatrixDimensionMismatch {
            two_site: 4,
            site_energy: 9,
        })
    ));
}

#[test]
fn complex_hamiltonian_rejects_nonfinite_entry_with_location() {
    let two = DMatrix::<Complex64>::identity(4, 4);
    let mut site = DMatrix::<Complex64>::identity(4, 4);
    site[(2, 1)] = Complex64::new(f64::NAN, 0.0);

    assert!(matches!(
        ComplexLocalHamiltonian::try_new(two, site),
        Err(ItebdError::NonFiniteMatrix {
            field: "site_energy",
            row: 2,
            column: 1,
        })
    ));
}

#[test]
fn complex_hamiltonian_rejects_invalid_hermiticity_tolerance() {
    let matrix = DMatrix::<Complex64>::identity(4, 4);

    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ComplexLocalHamiltonian::try_new_with_tolerance(
                matrix.clone(),
                matrix.clone(),
                invalid,
            ),
            Err(ItebdError::InvalidTolerance {
                name: "hermiticity tolerance",
                value,
            }) if value.to_bits() == invalid.to_bits()
        ));
    }
}

#[test]
fn complex_hamiltonian_reports_finite_nonhermitian_residual() {
    let mut two = DMatrix::<Complex64>::identity(4, 4);
    two[(0, 1)] = Complex64::new(0.25, 0.5);
    let site = DMatrix::<Complex64>::identity(4, 4);

    let error = ComplexLocalHamiltonian::try_new(two, site).unwrap_err();

    match error {
        ItebdError::NonHermitian {
            field,
            residual,
            tolerance,
        } => {
            assert_eq!(field, "two_site_h");
            assert!(residual.is_finite());
            assert!(tolerance.is_finite());
            assert!(residual > tolerance);
        }
        other => panic!("expected NonHermitian, got {other:?}"),
    }
}

#[test]
fn huge_finite_nonhermitian_entries_report_finite_residual() {
    let mut two = DMatrix::<Complex64>::identity(4, 4);
    two[(0, 1)] = Complex64::new(f64::MAX, 0.0);
    two[(1, 0)] = Complex64::new(-f64::MAX, 0.0);
    let site = DMatrix::<Complex64>::identity(4, 4);

    let error = ComplexLocalHamiltonian::try_new(two, site).unwrap_err();

    match error {
        ItebdError::NonHermitian {
            residual,
            tolerance,
            ..
        } => {
            assert!(residual.is_finite());
            assert!(tolerance.is_finite());
            assert!(residual > tolerance);
        }
        other => panic!("expected NonHermitian, got {other:?}"),
    }
}

#[test]
fn complex_gate_matches_small_tau_expansion() {
    let h = genuinely_complex_hermitian();
    let zero_gate = complex_trotter_gate(&h, 0.0).unwrap();
    let identity = DMatrix::<Complex64>::identity(4, 4);
    assert!(max_entrywise_norm(&(zero_gate - &identity)) < 2e-14);

    let tau = 1e-6;
    let gate = complex_trotter_gate(&h, tau).unwrap();
    let first_order = identity - h * Complex64::new(tau, 0.0);
    assert!(max_entrywise_norm(&(gate - first_order)) < 2e-12);
}

#[test]
fn complex_gate_rejects_nonfinite_time_step_with_typed_error() {
    let h = genuinely_complex_hermitian();

    assert!(matches!(
        complex_trotter_gate(&h, f64::NAN),
        Err(ItebdError::InvalidTimeStep { value }) if value.is_nan()
    ));
}

#[test]
fn complex_gate_rejects_non_square_physical_dimension() {
    let h = DMatrix::<Complex64>::identity(2, 2);

    assert!(matches!(
        complex_trotter_gate(&h, 0.1),
        Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 2 })
    ));
}

#[test]
fn complex_gate_rejects_zero_dimension_before_eigendecomposition() {
    let h = DMatrix::<Complex64>::zeros(0, 0);

    assert!(matches!(
        complex_trotter_gate(&h, 0.1),
        Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 0 })
    ));
}

#[test]
fn custom_tolerance_validated_hamiltonian_is_gate_usable() {
    let mut two = DMatrix::<Complex64>::identity(4, 4);
    two[(0, 1)] = Complex64::new(5e-9, 0.0);
    let site = DMatrix::<Complex64>::identity(4, 4);
    let hamiltonian =
        ComplexLocalHamiltonian::try_new_with_tolerance(two.clone(), site, 1e-8).unwrap();

    assert!(matches!(
        complex_trotter_gate(&two, 0.1),
        Err(ItebdError::NonHermitian { .. })
    ));
    let gate = hamiltonian.trotter_gate(0.1).unwrap();
    assert!(gate
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite()));
}

#[test]
fn documented_configs_resolve_expected_trotter_order() {
    let config_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
    let mut config_paths = fs::read_dir(&config_directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "toml")
        })
        .collect::<Vec<_>>();
    config_paths.sort();
    assert!(
        !config_paths.is_empty(),
        "no tracked TOML configurations found"
    );

    for path in config_paths {
        let text = fs::read_to_string(&path).unwrap();
        let resolved = RunConfig::from_toml_str(&text).unwrap_or_else(|error| {
            panic!("{} did not parse: {error}", path.display());
        });
        let documents_historical_first_order = text.contains("historical first-order reproduction");
        let expected = if documents_historical_first_order {
            assert!(
                text.lines()
                    .any(|line| line.trim() == "trotter_order = 1"),
                "{} documents historical first-order reproduction without setting trotter_order = 1",
                path.display()
            );
            TrotterOrder::First
        } else {
            TrotterOrder::Second
        };
        assert_eq!(
            resolved.evolution.trotter_order,
            expected,
            "{} resolves to an order that does not match its documented purpose",
            path.display()
        );
    }
}

#[test]
fn library_api_complex_example_compiles_and_steps() {
    let library_api = include_str!("../docs/library-api.md");
    for required_fragment in [
        "ItebdHamiltonian::try_from_complex",
        "ItebdState::infinite_temperature",
        "imaginary_time_step_second_order_auto",
        "use nalgebra::DMatrix;",
        "use num_complex::Complex64;",
        "let mut two_site_h = DMatrix::<Complex64>::identity(4, 4);",
        "two_site_h[(0, 1)] = Complex64::new(0.0, 0.2);",
        "two_site_h[(1, 0)] = Complex64::new(0.0, -0.2);",
        "max_bond: Some(64)",
    ] {
        assert!(
            library_api.contains(required_fragment),
            "library API complex example is missing {required_fragment}"
        );
    }

    let mut two_site_h = DMatrix::<Complex64>::identity(4, 4);
    two_site_h[(0, 1)] = Complex64::new(0.0, 0.2);
    two_site_h[(1, 0)] = Complex64::new(0.0, -0.2);
    let site_energy = two_site_h.clone();
    let hamiltonian = ItebdHamiltonian::try_from_complex(two_site_h, site_energy).unwrap();
    assert!(matches!(hamiltonian, ItebdHamiltonian::Complex(_)));
    let mut state = ItebdState::infinite_temperature(&hamiltonian).unwrap();
    let step = imaginary_time_step_second_order_auto(
        &mut state,
        &hamiltonian,
        0.01,
        &Truncation {
            epsilon: 1e-12,
            max_bond: Some(64),
        },
    )
    .unwrap();
    assert!(step.log_norm.is_finite());
    assert!(step.min_singular_value.is_finite());
    assert!(step.max_bond <= 64);
}
