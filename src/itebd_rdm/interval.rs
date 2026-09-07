use super::environment::{prepare, PreparedRdm};
use super::tensor::{checked_product, failure, RdmContractions};
use super::{AutoRdmResult, RdmError, RdmOptions, RdmParity, RdmReport, RdmResult};
use crate::itebd_auto::ItebdState;
use crate::itebd_complex::ComplexPurifiedMps;
use crate::itebd_state_view::ItebdStateRef;
use crate::purified_mps::PurifiedMps;
use crate::tensor::{new_index, Tensor};
use nalgebra::{ComplexField, DMatrix, SymmetricEigen};
use num_complex::Complex64;

/// Reconstruct an interval in first-site-fastest physical basis order.
/// Boundaries are identity-seeded fixed points; the report does not imply a unique sector.
pub fn reduced_density_matrix(
    state: &PurifiedMps,
    start: RdmParity,
    length: usize,
    options: &RdmOptions,
) -> Result<RdmResult<f64>, RdmError> {
    let dim = preflight(state.into(), start, length, options)?;
    let mut prepared = prepare(state.into(), options)?;
    let (tensor, report) = contract_interval(&mut prepared, start, length, options)?;
    let values = tensor
        .to_vec::<f64>()
        .map_err(|e| failure("real matrix", e))?;
    Ok(RdmResult {
        density_matrix: DMatrix::from_column_slice(dim, dim, &values),
        report,
    })
}

/// Complex counterpart preserving the raw normalized matrix without Hermitian projection.
pub fn reduced_density_matrix_complex(
    state: &ComplexPurifiedMps,
    start: RdmParity,
    length: usize,
    options: &RdmOptions,
) -> Result<RdmResult<Complex64>, RdmError> {
    let dim = preflight(state.into(), start, length, options)?;
    let mut prepared = prepare(state.into(), options)?;
    let (tensor, report) = contract_interval(&mut prepared, start, length, options)?;
    let values = tensor
        .to_vec::<Complex64>()
        .map_err(|e| failure("complex matrix", e))?;
    Ok(RdmResult {
        density_matrix: DMatrix::from_column_slice(dim, dim, &values),
        report,
    })
}

/// Dispatch according to the stored backend, including complex tensors with real values.
pub fn reduced_density_matrix_auto(
    state: &ItebdState,
    start: RdmParity,
    length: usize,
    options: &RdmOptions,
) -> Result<AutoRdmResult, RdmError> {
    match state {
        ItebdState::Real(state) => {
            reduced_density_matrix(state, start, length, options).map(AutoRdmResult::Real)
        }
        ItebdState::Complex(state) => reduced_density_matrix_complex(state, start, length, options)
            .map(AutoRdmResult::Complex),
    }
}

// Call only after interval metadata has been bounded by public preflight.
fn dimension(base: usize, exponent: usize) -> Result<usize, RdmError> {
    (0..exponent).try_fold(1, |size, _| {
        checked_product("physical dimension", &[size, base])
    })
}

fn preflight(
    state: ItebdStateRef<'_>,
    start: RdmParity,
    length: usize,
    options: &RdmOptions,
) -> Result<usize, RdmError> {
    options.validate()?;
    if length == 0 {
        return Err(RdmError::InvalidOption {
            name: "length",
            reason: "must be positive".into(),
        });
    }
    // Bound occurrence metadata as well as dense elements. In particular d=1 cannot
    // evade resource checks with a constant output size and an arbitrary site count.
    let metadata = checked_product("interval index metadata", &[length, 2])?
        .checked_add(4)
        .ok_or(RdmError::DimensionOverflow {
            stage: "interval index metadata",
        })?;
    if metadata > options.max_intermediate_elements {
        return Err(RdmError::ResourceLimit {
            stage: "interval index metadata",
            requested: metadata,
            limit: options.max_intermediate_elements,
        });
    }
    let mut bounds = RdmContractions::new(options.max_intermediate_elements);
    for site in [state.a(), state.b()] {
        bounds.check("input gamma", &site.gamma.dims())?;
        bounds.check(
            "declared site roles",
            &[
                site.left.dim,
                site.physical.dim,
                site.ancilla.dim,
                site.right.dim,
            ],
        )?;
        bounds.check("left environment", &[site.left.dim, site.left.dim])?;
        bounds.check("right environment", &[site.right.dim, site.right.dim])?;
    }
    // The pinned tensor API can materialize malformed underreported payloads during
    // validation; caps cover declared/planned tensors, not backend scratch or that case.
    state.validate()?;
    let d = state.a().physical.dim;
    let dim = dimension(d, length)?;
    let count = checked_product("output matrix", &[dim, dim])?;
    if count > options.max_output_elements {
        return Err(RdmError::ResourceLimit {
            stage: "output matrix",
            requested: count,
            limit: options.max_output_elements,
        });
    }
    bounds.check("output matrix", &[dim, dim])?;
    let mut previous = 1;
    for k in 0..length {
        let site = if (k % 2) == usize::from(start == RdmParity::B) {
            state.a()
        } else {
            state.b()
        };
        bounds.check(
            "interval open",
            &[previous, previous, site.left.dim, site.left.dim],
        )?;
        bounds.check(
            "interval ket",
            &[
                previous,
                previous,
                site.left.dim,
                d,
                site.ancilla.dim,
                site.right.dim,
            ],
        )?;
        bounds.check(
            "interval bra",
            &[previous, previous, d, d, site.right.dim, site.right.dim],
        )?;
        previous = checked_product("physical dimension", &[previous, d])?;
    }
    Ok(dim)
}

/// Consume a prepared state and interval arguments already checked by preflight.
/// Cached callers must preserve that precondition and the preparation options.
pub(super) fn contract_interval(
    prepared: &mut PreparedRdm<'_>,
    start: RdmParity,
    length: usize,
    options: &RdmOptions,
) -> Result<(Tensor, RdmReport), RdmError> {
    let d = prepared.state.a().physical.dim;
    let dim = dimension(d, length)?;
    let mut cut = usize::from(start == RdmParity::B);
    let c = &mut prepared.contractions;
    c.reset_largest(prepared.preparation_largest);
    let left = &prepared.environments.left[cut];
    let mut bra_left = new_index(left.dims()[0]);
    let mut ket_left = new_index(left.dims()[1]);
    let mut open = c.relabel_scale(
        "interval left",
        left,
        &left.indices,
        &[bra_left.clone(), ket_left.clone()],
        None,
    )?;
    let mut ket_roles = Vec::with_capacity(length);
    let mut bra_roles = Vec::with_capacity(length);
    let mut previous = 1;
    for _ in 0..length {
        let w = &prepared.weighted_sites[cut];
        let p = new_index(d);
        let q = new_index(d);
        let ancilla = new_index(w.dims()[2]);
        let ket_right = new_index(w.dims()[3]);
        let bra_right = new_index(w.dims()[3]);
        let ket = c.relabel_scale(
            "interval ket occurrence",
            w,
            &w.indices,
            &[
                ket_left.clone(),
                p.clone(),
                ancilla.clone(),
                ket_right.clone(),
            ],
            None,
        )?;
        let bra = c.relabel_scale(
            "interval bra occurrence",
            w,
            &w.indices,
            &[bra_left.clone(), q.clone(), ancilla, bra_right.clone()],
            None,
        )?;
        check_shape(
            c,
            "interval open",
            &open,
            &[previous, previous, ket_left.dim, ket_left.dim],
        )?;
        let first = c.pair("interval ket", &open, &ket, false, false)?;
        check_shape(
            c,
            "interval ket",
            &first,
            &[
                previous,
                previous,
                ket_left.dim,
                d,
                w.dims()[2],
                ket_right.dim,
            ],
        )?;
        open = c.pair("interval bra", &first, &bra, false, true)?;
        check_shape(
            c,
            "interval bra",
            &open,
            &[previous, previous, d, d, ket_right.dim, ket_right.dim],
        )?;
        ket_roles.push(p);
        bra_roles.push(q);
        ket_left = ket_right;
        bra_left = bra_right;
        previous = checked_product("physical dimension", &[previous, d])?;
        cut = 1 - cut;
    }
    let right = &prepared.environments.right[cut];
    let right = c.relabel_scale(
        "interval right",
        right,
        &right.indices,
        &[ket_left, bra_left],
        None,
    )?;
    let closed = c.pair("interval closure", &open, &right, false, false)?;
    ket_roles.extend(bra_roles);
    let ordered = c.relabel_scale("physical order", &closed, &ket_roles, &ket_roles, None)?;
    let mut report = RdmReport {
        start,
        length,
        left: prepared.environments.left_diagnostics.clone(),
        right: prepared.environments.right_diagnostics.clone(),
        raw_trace: Complex64::default(),
        trace_residual: 0.0,
        hermiticity_residual: 0.0,
        minimum_eigenvalue: 0.0,
        largest_intermediate_elements: c.largest(),
    };
    let output = if ordered.is_complex() {
        let values = normalize_density(
            ordered
                .to_vec::<Complex64>()
                .map_err(|e| failure("density payload", e))?,
            dim,
            options,
            &mut report,
        )?;
        Tensor::from_dense(ordered.indices, values)
    } else {
        let values = normalize_density(
            ordered
                .to_vec::<f64>()
                .map_err(|e| failure("density payload", e))?,
            dim,
            options,
            &mut report,
        )?;
        Tensor::from_dense(ordered.indices, values)
    }
    .map_err(|e| failure("normalized density", e))?;
    Ok((output, report))
}

fn check_shape(
    c: &mut RdmContractions,
    stage: &'static str,
    tensor: &Tensor,
    expected: &[usize],
) -> Result<(), RdmError> {
    let planned = c.check(stage, expected)?;
    let actual = c.check(stage, &tensor.dims())?;
    if actual != planned {
        return Err(failure(
            stage,
            "actual retained size disagrees with interval plan",
        ));
    }
    Ok(())
}

fn invalid(invariant: &'static str, value: f64, tolerance: f64) -> RdmError {
    RdmError::InvalidDensityMatrix {
        invariant,
        value,
        tolerance,
    }
}

fn normalize_density<T: ComplexField<RealField = f64> + Copy>(
    values: Vec<T>,
    dim: usize,
    options: &RdmOptions,
    report: &mut RdmReport,
) -> Result<Vec<T>, RdmError> {
    if values
        .iter()
        .any(|v| !v.real().is_finite() || !v.imaginary().is_finite())
    {
        return Err(invalid("finite components", f64::NAN, 0.0));
    }
    let mut rho = DMatrix::from_column_slice(dim, dim, &values);
    let trace = rho.trace();
    report.raw_trace = Complex64::new(trace.real(), trace.imaginary());
    if !trace.real().is_finite()
        || trace.real() <= 0.0
        || !trace.imaginary().is_finite()
        || trace.imaginary().abs() > options.trace_tolerance * trace.real().abs()
    {
        return Err(invalid(
            "positive real raw trace",
            trace.real(),
            options.trace_tolerance,
        ));
    }
    rho /= T::from_real(trace.real());
    if rho
        .iter()
        .any(|v| !v.real().is_finite() || !v.imaginary().is_finite())
    {
        return Err(invalid("finite normalized components", f64::NAN, 0.0));
    }
    report.trace_residual = (rho.trace() - T::one()).modulus();
    if !report.trace_residual.is_finite() || report.trace_residual > options.trace_tolerance {
        return Err(invalid(
            "normalized trace",
            report.trace_residual,
            options.trace_tolerance,
        ));
    }
    report.hermiticity_residual = (&rho - rho.adjoint()).norm() / rho.norm().max(1.0);
    if !report.hermiticity_residual.is_finite()
        || report.hermiticity_residual > options.hermiticity_tolerance
    {
        return Err(invalid(
            "Hermiticity",
            report.hermiticity_residual,
            options.hermiticity_tolerance,
        ));
    }
    let hermitian = (&rho + rho.adjoint()) * T::from_real(0.5);
    let eigen = SymmetricEigen::try_new(hermitian, f64::EPSILON, 10_000)
        .ok_or_else(|| failure("density spectrum", "eigensolver did not converge"))?;
    if eigen.eigenvalues.iter().any(|v| !v.is_finite())
        || eigen
            .eigenvectors
            .iter()
            .any(|v| !v.real().is_finite() || !v.imaginary().is_finite())
    {
        return Err(invalid("finite eigensolver results", f64::NAN, 0.0));
    }
    report.minimum_eigenvalue = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if report.minimum_eigenvalue < -options.positivity_tolerance {
        return Err(invalid(
            "positivity",
            report.minimum_eigenvalue,
            options.positivity_tolerance,
        ));
    }
    Ok(rho.as_slice().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itebd_rdm::oracles::{
        dense_interval_rdm, dense_left, dense_right, fixture, weighted_site_matrices,
    };

    #[test]
    fn cached_short_interval_reports_its_own_peak() {
        for complex in [false, true] {
            let state = fixture(complex, 2, 3);
            let options = RdmOptions::default();
            let mut prepared = prepare((&state).into(), &options).unwrap();
            let (_, long) = contract_interval(&mut prepared, RdmParity::A, 4, &options).unwrap();
            assert_eq!(long.largest_intermediate_elements, 2304);
            let (short, report) =
                contract_interval(&mut prepared, RdmParity::A, 1, &options).unwrap();
            assert_eq!(report.largest_intermediate_elements, 24);
            assert_eq!(short.is_complex(), complex);
        }
    }

    #[test]
    fn full_intervals_match_independent_infinite_boundary_oracle() {
        for complex in [false, true] {
            let state = fixture(complex, 2, 3);
            let view = ItebdStateRef::from(&state);
            let wa = weighted_site_matrices(view.a(), view.lambda_ba());
            let wb = weighted_site_matrices(view.b(), view.lambda_ab());
            let mut l = DMatrix::<Complex64>::identity(3, 3);
            let mut r = l.clone();
            for _ in 0..512 {
                l = dense_left(&wb, &dense_left(&wa, &l));
                r = dense_right(&wa, &dense_right(&wb, &r));
                let ln = l.norm();
                let rn = r.norm();
                assert!(ln.is_finite() && ln > 0.0 && rn.is_finite() && rn > 0.0);
                l /= Complex64::from(ln);
                r /= Complex64::from(rn);
            }
            for (side, x, image) in [
                ("left", &l, dense_left(&wb, &dense_left(&wa, &l))),
                ("right", &r, dense_right(&wa, &dense_right(&wb, &r))),
            ] {
                let eta = x.dotc(&image).re / x.norm_squared();
                let residual = (&image - x * Complex64::from(eta)).norm()
                    / image.norm().max(eta.abs() * x.norm());
                eprintln!("oracle boundary complex={complex} side={side} residual={residual:e}");
                assert!(residual <= 1e-13, "invalid independent boundary oracle");
            }
            let ls = [l.clone(), dense_left(&wa, &l)];
            let rs = [r.clone(), dense_right(&wb, &r)];
            for start in [RdmParity::A, RdmParity::B] {
                let cut = usize::from(start == RdmParity::B);
                let largest = if cut == 0 {
                    [24, 144, 384, 2304]
                } else {
                    [36, 96, 576, 1536]
                };
                for length in 1..=4 {
                    let sites: Vec<_> = (0..length)
                        .map(|k| {
                            if (cut + k) % 2 == 0 {
                                wa.clone()
                            } else {
                                wb.clone()
                            }
                        })
                        .collect();
                    let expected = dense_interval_rdm(&sites, &ls[cut], &rs[(cut + length) % 2], 2);
                    let (got, report) = match reduced_density_matrix_auto(
                        &state,
                        start,
                        length,
                        &RdmOptions::default(),
                    )
                    .unwrap()
                    {
                        AutoRdmResult::Real(result) => {
                            (result.density_matrix.map(Complex64::from), result.report)
                        }
                        AutoRdmResult::Complex(result) => (result.density_matrix, result.report),
                    };
                    let disagreement = (&got - expected).norm();
                    eprintln!("interval complex={complex} start={start:?} n={length}: error={disagreement:e} largest={} min={:e}", report.largest_intermediate_elements, report.minimum_eigenvalue);
                    assert!(disagreement <= 1e-10);
                    assert_eq!(report.largest_intermediate_elements, largest[length - 1]);
                    assert!(
                        report.trace_residual <= 1e-10
                            && report.hermiticity_residual <= 1e-10
                            && report.minimum_eigenvalue >= -1e-10
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_raw_output_is_rejected_without_projection_or_clipping() {
        // Injection is private to this child module; public tolerances are unchanged.
        let options = RdmOptions::default();
        let state = crate::purified_mps::infinite_temperature(1);
        let report = reduced_density_matrix(&state, RdmParity::A, 1, &options)
            .unwrap()
            .report;
        for (values, invariant) in [
            (vec![0.0, 0.0, 0.0, 0.0], "positive real raw trace"),
            (vec![-1.0, 0.0, 0.0, -1.0], "positive real raw trace"),
            (vec![1.0, 1.0, 0.0, 1.0], "Hermiticity"),
            (vec![1.1, 0.0, 0.0, -0.1], "positivity"),
            (vec![f64::NAN, 0.0, 0.0, 1.0], "finite components"),
        ] {
            for complex in [false, true] {
                let mut report = report.clone();
                let error = if complex {
                    normalize_density(
                        values.iter().copied().map(Complex64::from).collect(),
                        2,
                        &options,
                        &mut report,
                    )
                    .unwrap_err()
                } else {
                    normalize_density(values.clone(), 2, &options, &mut report).unwrap_err()
                };
                assert!(
                    matches!(error, RdmError::InvalidDensityMatrix {invariant: got, ..} if got == invariant)
                );
            }
        }
        let mut report = report.clone();
        assert!(matches!(
            normalize_density(vec![Complex64::new(1.0, 0.1)], 1, &options, &mut report),
            Err(RdmError::InvalidDensityMatrix {
                invariant: "positive real raw trace",
                ..
            })
        ));
        // A tiny accepted anti-Hermitian component and negative eigenvalue must
        // survive verbatim: using the diagnostic Hermitian part or clipping fails.
        let raw = vec![
            Complex64::from(1.0 + 1e-12),
            Complex64::new(0.0, 1e-12),
            Complex64::default(),
            Complex64::from(-1e-12),
        ];
        let result = normalize_density(raw.clone(), 2, &options, &mut report).unwrap();
        let trace = raw[0].re + raw[3].re;
        assert_eq!(result, raw.iter().map(|v| *v / trace).collect::<Vec<_>>());
        assert!(result[3].re < 0.0 && result[1].im != 0.0 && result[2].im == 0.0);
    }
}
