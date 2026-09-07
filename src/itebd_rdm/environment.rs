use super::tensor::{failure, RdmContractions};
use super::{RdmEnvironmentDiagnostics, RdmError, RdmOptions};
use crate::itebd_state_view::{ItebdScalarType, ItebdStateRef};
use crate::tensor::Tensor;
use crate::tensor::{new_index, Idx};
use nalgebra::{ComplexField, DMatrix, SymmetricEigen};
use num_complex::Complex64;

fn invalid(side: &'static str, reason: impl Into<String>) -> RdmError {
    RdmError::InvalidEnvironment {
        side,
        reason: reason.into(),
    }
}

// Stable scalar/vector bookkeeping: these loops do not contract tensor networks.
fn stable_norm<T: ComplexField<RealField = f64> + Copy>(values: &[T]) -> f64 {
    let scale = values.iter().map(|v| v.modulus()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    scale
        * values
            .iter()
            .map(|v| (v.modulus() / scale).powi(2))
            .sum::<f64>()
            .sqrt()
}

fn normalized_values<T: ComplexField<RealField = f64> + Copy>(
    mut values: Vec<T>,
    side: &'static str,
) -> Result<Vec<T>, RdmError> {
    let norm = stable_norm(&values);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid(side, "nonfinite or zero Frobenius norm"));
    }
    values.iter_mut().for_each(|v| *v /= T::from_real(norm));
    Ok(values)
}

fn normalize(tensor: &Tensor, side: &'static str) -> Result<Tensor, RdmError> {
    if tensor.is_complex() {
        Tensor::from_dense(
            tensor.indices.clone(),
            normalized_values(
                tensor
                    .to_vec::<Complex64>()
                    .map_err(|e| failure("normalize", e))?,
                side,
            )?,
        )
    } else {
        Tensor::from_dense(
            tensor.indices.clone(),
            normalized_values(
                tensor
                    .to_vec::<f64>()
                    .map_err(|e| failure("normalize", e))?,
                side,
            )?,
        )
    }
    .map_err(|e| failure("normalize", e))
}

fn measure_values<T: ComplexField<RealField = f64> + Copy>(
    x: &[T],
    y: &[T],
    options: &RdmOptions,
    side: &'static str,
    iterations: usize,
) -> Result<RdmEnvironmentDiagnostics, RdmError> {
    let nx = stable_norm(x);
    let ny = stable_norm(y);
    if !nx.is_finite() || !ny.is_finite() || nx <= 0.0 || ny <= 0.0 {
        return Err(invalid(side, "nonfinite or zero residual denominator"));
    }
    let q = x
        .iter()
        .zip(y)
        .map(|(x, y)| (*x / T::from_real(nx)).conjugate() * (*y / T::from_real(ny)))
        .fold(T::zero(), |a, b| a + b);
    let xx = x
        .iter()
        .map(|v| (*v / T::from_real(nx)).modulus_squared())
        .sum::<f64>();
    let q = q / T::from_real(xx);
    let eta = q.real() * (ny / nx);
    if !eta.is_finite()
        || eta <= 0.0
        || !q.imaginary().is_finite()
        || q.imaginary().abs() > options.trace_tolerance * q.real().abs()
    {
        return Err(invalid(
            side,
            "Rayleigh eigenvalue must be finite, positive and real",
        ));
    }
    let differences: Vec<T> = x
        .iter()
        .zip(y)
        .map(|(x, y)| *y / T::from_real(ny) - (*x / T::from_real(nx)) * T::from_real(q.real()))
        .collect();
    let relative_residual = stable_norm(&differences) / q.real().abs().max(1.0);
    if !relative_residual.is_finite() {
        return Err(invalid(side, "nonfinite residual"));
    }
    Ok(RdmEnvironmentDiagnostics {
        eigenvalue: eta,
        iterations,
        relative_residual,
    })
}

fn measure(
    x: &Tensor,
    y: &Tensor,
    options: &RdmOptions,
    side: &'static str,
    iterations: usize,
) -> Result<RdmEnvironmentDiagnostics, RdmError> {
    if x.indices != y.indices || x.dims() != y.dims() || x.is_complex() != y.is_complex() {
        return Err(invalid(
            side,
            "transfer changed environment roles, shape, or backend",
        ));
    }
    if x.is_complex() {
        measure_values(
            &x.to_vec::<Complex64>()
                .map_err(|e| failure("residual", e))?,
            &y.to_vec::<Complex64>()
                .map_err(|e| failure("residual", e))?,
            options,
            side,
            iterations,
        )
    } else {
        measure_values(
            &x.to_vec::<f64>().map_err(|e| failure("residual", e))?,
            &y.to_vec::<f64>().map_err(|e| failure("residual", e))?,
            options,
            side,
            iterations,
        )
    }
}

pub(super) fn power_iterate(
    initial: Tensor,
    mut apply: impl FnMut(&Tensor) -> Result<Tensor, RdmError>,
    options: &RdmOptions,
    side: &'static str,
) -> Result<(Tensor, RdmEnvironmentDiagnostics), RdmError> {
    options.validate()?;
    let mut bounds = RdmContractions::new(options.max_intermediate_elements);
    bounds.check("power seed", &initial.dims())?;
    let mut x = normalize(&initial, side)?;
    for iterations in 1..=options.max_iterations {
        let y = apply(&x)?;
        bounds.check("power image", &y.dims())?;
        let diagnostics = measure(&x, &y, options, side, iterations)?;
        if diagnostics.relative_residual <= options.fixed_point_tolerance {
            return Ok((x, diagnostics));
        }
        if iterations == options.max_iterations {
            return Err(RdmError::FixedPointNonConvergence {
                side,
                iterations,
                residual: diagnostics.relative_residual,
            });
        }
        x = normalize(&y, side)?;
    }
    unreachable!("validated positive budget")
}

pub(super) struct RdmEnvironmentSet {
    pub left: [Tensor; 2],  // BA, AB: [bra, ket]
    pub right: [Tensor; 2], // BA, AB: [ket, bra]
    pub left_diagnostics: RdmEnvironmentDiagnostics,
    pub right_diagnostics: RdmEnvironmentDiagnostics,
}
pub(super) struct PreparedRdm<'a> {
    pub state: ItebdStateRef<'a>,
    pub weighted_sites: [Tensor; 2], // Each ordered [ket_left,physical,ancilla,ket_right].
    pub environments: RdmEnvironmentSet,
    pub contractions: RdmContractions,
    pub preparation_largest: usize,
}

pub(super) fn prepare<'a>(
    state: ItebdStateRef<'a>,
    options: &RdmOptions,
) -> Result<PreparedRdm<'a>, RdmError> {
    options.validate()?;
    let mut contractions = RdmContractions::new(options.max_intermediate_elements);
    // These metadata-only checks precede validate(), which copies each gamma payload.
    for site in [state.a(), state.b()] {
        contractions.check("input gamma", &site.gamma.dims())?;
        contractions.check(
            "declared site roles",
            &[
                site.left.dim,
                site.physical.dim,
                site.ancilla.dim,
                site.right.dim,
            ],
        )?;
        contractions.check("left environment", &[site.left.dim, site.left.dim])?;
        contractions.check("right environment", &[site.right.dim, site.right.dim])?;
    }
    state.validate()?;
    let ba = new_index(state.a().left.dim);
    let ab = new_index(state.a().right.dim);
    let make = |site: crate::itebd_state_view::ItebdSiteRef<'_>,
                lambda: &[f64],
                left: &Idx,
                right: &Idx,
                c: &mut RdmContractions| {
        c.relabel_scale(
            "weighted site",
            site.gamma,
            &[
                site.left.clone(),
                site.physical.clone(),
                site.ancilla.clone(),
                site.right.clone(),
            ],
            &[
                left.clone(),
                new_index(site.physical.dim),
                new_index(site.ancilla.dim),
                right.clone(),
            ],
            Some((site.left, lambda)),
        )
    };
    let weighted_sites = [
        make(state.a(), state.lambda_ba(), &ba, &ab, &mut contractions)?,
        make(state.b(), state.lambda_ab(), &ab, &ba, &mut contractions)?,
    ];
    let bras = cache_bras(&weighted_sites, &mut contractions)?;
    let complex = state.scalar_type() == ItebdScalarType::Complex;
    let initial_left = identity(
        &[bras[0].indices[0].clone(), ba.clone()],
        complex,
        &mut contractions,
    )?;
    let initial_right = identity(
        &[ba, bras[0].indices[0].clone()],
        complex,
        &mut contractions,
    )?;
    let (mut left, mut ld) = power_iterate(
        initial_left,
        |x| left_cell(x, &weighted_sites, &bras, &mut contractions),
        options,
        "left",
    )?;
    let (mut right, mut rd) = power_iterate(
        initial_right,
        |x| right_cell(x, &weighted_sites, &bras, &mut contractions),
        options,
        "right",
    )?;
    // Independent first crossings need not have sufficiently matching eigenvalues. Continue
    // both solves using their remaining original budgets, never a tightened public tolerance.
    if eigenvalue_difference(&ld, &rd) > options.fixed_point_tolerance {
        let mut ly = remeasure(
            &left,
            &mut ld,
            |x| left_cell(x, &weighted_sites, &bras, &mut contractions),
            options,
            "left",
        )?;
        let mut ry = remeasure(
            &right,
            &mut rd,
            |x| right_cell(x, &weighted_sites, &bras, &mut contractions),
            options,
            "right",
        )?;
        while ld.relative_residual > options.fixed_point_tolerance
            || rd.relative_residual > options.fixed_point_tolerance
            || eigenvalue_difference(&ld, &rd) > options.fixed_point_tolerance
        {
            left = normalize(&ly, "left")?;
            right = normalize(&ry, "right")?;
            ly = remeasure(
                &left,
                &mut ld,
                |x| left_cell(x, &weighted_sites, &bras, &mut contractions),
                options,
                "left",
            )?;
            ry = remeasure(
                &right,
                &mut rd,
                |x| right_cell(x, &weighted_sites, &bras, &mut contractions),
                options,
                "right",
            )?;
        }
    }
    let left_ab = normalize(
        &left_transfer(&left, &weighted_sites[0], &bras[0], &mut contractions)?,
        "left AB",
    )?;
    let right_ab = normalize(
        &right_transfer(&right, &weighted_sites[1], &bras[1], &mut contractions)?,
        "right AB",
    )?;
    let environments = RdmEnvironmentSet {
        left: [left, left_ab],
        right: [right, right_ab],
        left_diagnostics: ld,
        right_diagnostics: rd,
    };
    for cut in 0..2 {
        validate_environment(&environments.left[cut], options, "left")?;
        validate_environment(&environments.right[cut], options, "right")?;
        let overlap = contractions.pair(
            "environment overlap",
            &environments.left[cut],
            &environments.right[cut],
            false,
            false,
        )?;
        let value = if complex {
            overlap
                .to_vec::<Complex64>()
                .map_err(|e| failure("overlap", e))?[0]
        } else {
            Complex64::from(overlap.to_vec::<f64>().map_err(|e| failure("overlap", e))?[0])
        };
        if !value.re.is_finite()
            || !value.im.is_finite()
            || value.re <= 0.0
            || value.im.abs() > options.trace_tolerance * value.re.abs()
        {
            return Err(invalid(
                "overlap",
                "Tr(L R) must be finite, positive and real at both cuts",
            ));
        }
    }
    let preparation_largest = contractions.largest();
    Ok(PreparedRdm {
        state,
        weighted_sites,
        environments,
        contractions,
        preparation_largest,
    })
}
fn eigenvalue_difference(
    left: &RdmEnvironmentDiagnostics,
    right: &RdmEnvironmentDiagnostics,
) -> f64 {
    (left.eigenvalue - right.eigenvalue).abs() / left.eigenvalue.abs().max(right.eigenvalue.abs())
}
fn remeasure(
    x: &Tensor,
    diagnostics: &mut RdmEnvironmentDiagnostics,
    mut apply: impl FnMut(&Tensor) -> Result<Tensor, RdmError>,
    options: &RdmOptions,
    side: &'static str,
) -> Result<Tensor, RdmError> {
    if diagnostics.iterations >= options.max_iterations {
        return Err(RdmError::FixedPointNonConvergence {
            side,
            iterations: diagnostics.iterations,
            residual: diagnostics.relative_residual,
        });
    }
    let y = apply(x)?;
    *diagnostics = measure(x, &y, options, side, diagnostics.iterations + 1)?;
    Ok(y)
}
fn cache_bras(sites: &[Tensor; 2], c: &mut RdmContractions) -> Result<[Tensor; 2], RdmError> {
    let ba = new_index(sites[0].indices[0].dim);
    let ab = new_index(sites[0].indices[3].dim);
    let make = |w: &Tensor, left: &Idx, right: &Idx, c: &mut RdmContractions| {
        c.relabel_scale(
            "bra layout cache",
            w,
            &w.indices,
            &[
                left.clone(),
                w.indices[1].clone(),
                w.indices[2].clone(),
                right.clone(),
            ],
            None,
        )
    };
    Ok([make(&sites[0], &ba, &ab, c)?, make(&sites[1], &ab, &ba, c)?])
}
fn identity(
    indices: &[Idx; 2],
    complex: bool,
    c: &mut RdmContractions,
) -> Result<Tensor, RdmError> {
    if indices[0].dim != indices[1].dim {
        return Err(failure("identity seed", "nonsquare roles"));
    }
    let n = c.check("identity seed", &[indices[0].dim, indices[1].dim])?;
    let dim = indices[0].dim;
    if complex {
        Tensor::from_dense(
            indices.to_vec(),
            (0..n)
                .map(|i| Complex64::from(if i % dim == i / dim { 1.0 } else { 0.0 }))
                .collect(),
        )
    } else {
        Tensor::from_dense(
            indices.to_vec(),
            (0..n)
                .map(|i| if i % dim == i / dim { 1.0 } else { 0.0 })
                .collect(),
        )
    }
    .map_err(|e| failure("identity seed", e))
}
fn left_transfer(
    x: &Tensor,
    w: &Tensor,
    bra: &Tensor,
    c: &mut RdmContractions,
) -> Result<Tensor, RdmError> {
    let first = c.pair("left transfer ket", x, w, false, false)?;
    let result = c.pair("left transfer bra", &first, bra, false, true)?;
    let ordered = [bra.indices[3].clone(), w.indices[3].clone()];
    c.relabel_scale("left transfer order", &result, &ordered, &ordered, None)
}
fn right_transfer(
    x: &Tensor,
    w: &Tensor,
    bra: &Tensor,
    c: &mut RdmContractions,
) -> Result<Tensor, RdmError> {
    let first = c.pair("right transfer ket", w, x, false, false)?;
    let result = c.pair("right transfer bra", &first, bra, false, true)?;
    let ordered = [w.indices[0].clone(), bra.indices[0].clone()];
    c.relabel_scale("right transfer order", &result, &ordered, &ordered, None)
}
fn left_cell(
    x: &Tensor,
    w: &[Tensor; 2],
    bra: &[Tensor; 2],
    c: &mut RdmContractions,
) -> Result<Tensor, RdmError> {
    let ab = left_transfer(x, &w[0], &bra[0], c)?;
    left_transfer(&ab, &w[1], &bra[1], c)
}
fn right_cell(
    x: &Tensor,
    w: &[Tensor; 2],
    bra: &[Tensor; 2],
    c: &mut RdmContractions,
) -> Result<Tensor, RdmError> {
    let ab = right_transfer(x, &w[1], &bra[1], c)?;
    right_transfer(&ab, &w[0], &bra[0], c)
}
fn validate_values<T: ComplexField<RealField = f64> + Copy>(
    values: Vec<T>,
    dim: usize,
    options: &RdmOptions,
    side: &'static str,
) -> Result<(), RdmError> {
    let matrix = DMatrix::from_column_slice(dim, dim, &values);
    let anti = &matrix - matrix.adjoint();
    let residual = stable_norm(anti.as_slice()) / stable_norm(matrix.as_slice());
    if !residual.is_finite() || residual > options.hermiticity_tolerance {
        return Err(invalid(side, format!("Hermiticity residual {residual:e}")));
    }
    // The Hermitian part is used for diagnostics only, after the raw check; no projection.
    let hermitian = (&matrix + matrix.adjoint()) * T::from_real(0.5);
    let spectrum = SymmetricEigen::try_new(hermitian, f64::EPSILON, 10_000)
        .ok_or_else(|| invalid(side, "Hermitian eigensolver did not converge"))?
        .eigenvalues;
    if spectrum.iter().any(|value| !value.is_finite()) {
        return Err(invalid(side, "nonfinite environment eigenvalue"));
    }
    let minimum = spectrum.iter().copied().fold(f64::INFINITY, f64::min);
    if !minimum.is_finite() || minimum < -options.positivity_tolerance {
        return Err(invalid(side, format!("minimum eigenvalue {minimum:e}")));
    }
    Ok(())
}
fn validate_environment(
    x: &Tensor,
    options: &RdmOptions,
    side: &'static str,
) -> Result<(), RdmError> {
    if x.indices.len() != 2 || x.indices[0].dim != x.indices[1].dim {
        return Err(invalid(side, "environment must be square"));
    }
    let dim = x.indices[0].dim;
    if x.is_complex() {
        validate_values(
            x.to_vec::<Complex64>()
                .map_err(|e| failure("environment PSD", e))?,
            dim,
            options,
            side,
        )
    } else {
        validate_values(
            x.to_vec::<f64>()
                .map_err(|e| failure("environment PSD", e))?,
            dim,
            options,
            side,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itebd_rdm::{RdmError, RdmOptions};
    use crate::tensor::{new_index, Tensor};

    fn seed() -> Tensor {
        Tensor::from_dense(vec![new_index(2), new_index(2)], vec![2.0, 0.0, 0.0, 1.0]).unwrap()
    }

    #[test]
    fn constant_norm_cycle_does_not_pass_as_a_fixed_point() {
        let options = RdmOptions {
            max_iterations: 4,
            ..RdmOptions::default()
        };
        let result = power_iterate(
            seed(),
            |x| {
                let v = x.to_vec::<f64>().unwrap();
                Ok(Tensor::from_dense(x.indices.clone(), vec![v[3], v[2], v[1], v[0]]).unwrap())
            },
            &options,
            "left",
        );
        assert!(
            matches!(result, Err(RdmError::FixedPointNonConvergence { iterations: 4, residual, .. }) if residual > 0.5)
        );
    }

    #[test]
    fn zero_map_is_invalid() {
        let result = power_iterate(
            seed(),
            |x| Ok(Tensor::from_dense(x.indices.clone(), vec![0.0; 4]).unwrap()),
            &RdmOptions::default(),
            "left",
        );
        assert!(matches!(result, Err(RdmError::InvalidEnvironment { .. })));
    }

    #[test]
    fn options_reject_zero_iterations_and_nonfinite_tolerance() {
        for options in [
            RdmOptions {
                max_iterations: 0,
                ..RdmOptions::default()
            },
            RdmOptions {
                fixed_point_tolerance: f64::NAN,
                ..RdmOptions::default()
            },
        ] {
            assert!(matches!(
                power_iterate(seed(), |x| Ok(x.clone()), &options, "left"),
                Err(RdmError::InvalidOption { .. })
            ));
        }
    }

    #[test]
    fn identity_map_returns_residual_checked_seed() {
        let initial =
            Tensor::from_dense(vec![new_index(2), new_index(2)], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let (fixed, report) =
            power_iterate(initial, |x| Ok(x.clone()), &RdmOptions::default(), "left").unwrap();
        assert_eq!(report.relative_residual, 0.0);
        assert_eq!(report.eigenvalue, 1.0);
        assert_eq!(report.iterations, 1);
        assert!(!fixed.is_complex());
    }
    #[test]
    fn primitive_fixture_transfers_and_environments_match_independent_oracles() {
        use super::super::oracles::*;
        use crate::itebd_state_view::ItebdStateRef;
        use nalgebra::DMatrix;
        for complex in [false, true] {
            let state = fixture(complex, 2, 3);
            let view = ItebdStateRef::from(&state);
            let wa = weighted_site_matrices(view.a(), view.lambda_ba());
            let wb = weighted_site_matrices(view.b(), view.lambda_ab());
            let kraus: Vec<_> = wa
                .iter()
                .flat_map(|a| wb.iter().map(move |b| a * b))
                .collect();
            let span = DMatrix::from_fn(9, kraus.len(), |i, j| kraus[j].as_slice()[i]);
            let spectrum = span.svd(false, false).singular_values;
            let rank = spectrum
                .iter()
                .filter(|v| **v > 1e-10 * spectrum[0])
                .count();
            eprintln!(
                "primitive complex={complex}: rank={rank}, singular_values={:?}",
                spectrum.as_slice()
            );
            assert_eq!(rank, 9);
            let options = RdmOptions::default();
            let mut prepared = prepare(view, &options).unwrap();
            assert_eq!(prepared.state.scalar_type(), view.scalar_type());
            let bras = cache_bras(&prepared.weighted_sites, &mut prepared.contractions).unwrap();
            for (site, matrices) in [(0, &wa), (1, &wb)] {
                let w = &prepared.weighted_sites[site];
                let left = positive_seed(
                    &[bras[site].indices[0].clone(), w.indices[0].clone()],
                    complex,
                );
                let right = positive_seed(
                    &[w.indices[3].clone(), bras[site].indices[3].clone()],
                    complex,
                );
                let got = left_transfer(&left, w, &bras[site], &mut prepared.contractions).unwrap();
                assert!((matrix(&got) - dense_left(matrices, &matrix(&left))).norm() < 1e-12);
                let got =
                    right_transfer(&right, w, &bras[site], &mut prepared.contractions).unwrap();
                assert!((matrix(&got) - dense_right(matrices, &matrix(&right))).norm() < 1e-12);
            }
            let initial_left = positive_seed(
                &[
                    bras[0].indices[0].clone(),
                    prepared.weighted_sites[0].indices[0].clone(),
                ],
                complex,
            );
            let initial_right = positive_seed(
                &[
                    prepared.weighted_sites[0].indices[0].clone(),
                    bras[0].indices[0].clone(),
                ],
                complex,
            );
            let cell_l = left_cell(
                &initial_left,
                &prepared.weighted_sites,
                &bras,
                &mut prepared.contractions,
            )
            .unwrap();
            let cell_r = right_cell(
                &initial_right,
                &prepared.weighted_sites,
                &bras,
                &mut prepared.contractions,
            )
            .unwrap();
            assert!(
                (matrix(&cell_l) - dense_left(&wb, &dense_left(&wa, &matrix(&initial_left))))
                    .norm()
                    < 1e-12
            );
            assert!(
                (matrix(&cell_r) - dense_right(&wa, &dense_right(&wb, &matrix(&initial_right))))
                    .norm()
                    < 1e-12
            );
            let mut l = DMatrix::identity(3, 3);
            let mut r = l.clone();
            for _ in 0..512 {
                l = dense_left(&wb, &dense_left(&wa, &l));
                l /= Complex64::from(l.norm());
                r = dense_right(&wa, &dense_right(&wb, &r));
                r /= Complex64::from(r.norm());
            }
            for (got, expected) in [
                (&prepared.environments.left[0], l),
                (&prepared.environments.right[0], r),
            ] {
                let agreement = (matrix(got) - expected).norm();
                eprintln!("fixed-point oracle agreement complex={complex}: {agreement:e}");
                assert!(agreement < 1e-10);
                assert_eq!(got.is_complex(), complex);
            }
            let left = &prepared.environments.left_diagnostics;
            let right = &prepared.environments.right_diagnostics;
            assert!(left.relative_residual <= 1e-12 && right.relative_residual <= 1e-12);
            let l = matrix(&prepared.environments.left[0]);
            let r = matrix(&prepared.environments.right[0]);
            for (x, y, diagnostics) in [
                (&l, dense_left(&wb, &dense_left(&wa, &l)), left),
                (&r, dense_right(&wa, &dense_right(&wb, &r)), right),
            ] {
                let residual = (&y - x * Complex64::from(diagnostics.eigenvalue)).norm()
                    / y.norm().max(diagnostics.eigenvalue * x.norm());
                assert!(residual <= 1e-12);
                assert!((residual - diagnostics.relative_residual).abs() < 1e-14);
                eprintln!("independent residual complex={complex}: {residual:e}");
            }
            let mut l_ab = dense_left(&wa, &l);
            let mut r_ab = dense_right(&wb, &r);
            l_ab /= Complex64::from(l_ab.norm());
            r_ab /= Complex64::from(r_ab.norm());
            assert!((matrix(&prepared.environments.left[1]) - l_ab).norm() < 1e-12);
            assert!((matrix(&prepared.environments.right[1]) - r_ab).norm() < 1e-12);
            assert!(
                (left.eigenvalue - right.eigenvalue).abs()
                    / left.eigenvalue.abs().max(right.eigenvalue.abs())
                    <= 1e-12
            );
            assert_eq!(prepared.contractions.largest(), 24);
            eprintln!(
                "environment complex={complex}: left={left:?}, right={right:?}, largest={}",
                prepared.contractions.largest()
            );
        }
    }
    #[test]
    fn preparation_prechecks_payload_and_environment_caps() {
        use super::super::oracles::fixture;
        use crate::itebd_state_view::ItebdStateRef;
        let state = fixture(false, 2, 3);
        for limit in [8, 23] {
            let options = RdmOptions {
                max_intermediate_elements: limit,
                ..RdmOptions::default()
            };
            assert!(matches!(
                prepare(ItebdStateRef::from(&state), &options),
                Err(RdmError::ResourceLimit { .. })
            ));
        }
    }

    #[test]
    fn preparation_rejects_equal_product_storage_shape_mismatch() {
        use super::super::oracles::fixture;
        use crate::itebd_auto::ItebdState;
        use crate::itebd_state_view::ItebdStateRef;

        for complex in [false, true] {
            let mut state = fixture(complex, 2, 2);
            match &mut state {
                ItebdState::Real(state) => {
                    state.lambda_bond_ab.dim = 1;
                    state.a.right.dim = 1;
                    state.b.left.dim = 1;
                    state.a.gamma.indices[1].dim = 1;
                    state.b.gamma.indices[2].dim = 1;
                    state.lambda_ab = vec![1.0];
                    state.lambda_bond_ba.dim = 4;
                    state.a.left.dim = 4;
                    state.b.right.dim = 4;
                    state.a.gamma.indices[2].dim = 4;
                    state.b.gamma.indices[1].dim = 4;
                    state.lambda_ba = vec![1.0; 4];
                }
                ItebdState::Complex(state) => {
                    state.lambda_bond_ab.dim = 1;
                    state.a.right.dim = 1;
                    state.b.left.dim = 1;
                    state.a.gamma.indices[1].dim = 1;
                    state.b.gamma.indices[2].dim = 1;
                    state.lambda_ab = vec![1.0];
                    state.lambda_bond_ba.dim = 4;
                    state.a.left.dim = 4;
                    state.b.right.dim = 4;
                    state.a.gamma.indices[2].dim = 4;
                    state.b.gamma.indices[1].dim = 4;
                    state.lambda_ba = vec![1.0; 4];
                }
            }
            let a = ItebdStateRef::from(&state).a();
            assert_eq!(a.gamma.dims(), vec![2, 1, 4, 2]);
            assert_eq!(a.gamma.storage().unwrap().logical_dims(), &[2, 2, 2, 2]);
            assert_eq!(a.gamma.dims().iter().product::<usize>(), 16);
            assert!(matches!(
                prepare(ItebdStateRef::from(&state), &RdmOptions::default()),
                Err(RdmError::State(
                    crate::itebd_state_view::ItebdStateValidationError::InvalidField {
                        field,
                        reason,
                    }
                )) if field == "a.gamma"
                    && reason
                        == "storage shape [2, 2, 2, 2] does not match gamma index dimensions [2, 1, 4, 2]"
            ));
        }
    }

    fn positive_seed(indices: &[Idx; 2], complex: bool) -> Tensor {
        let n = indices[0].dim;
        let matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            Complex64::new(
                (1 + i + 2 * j) as f64,
                if complex {
                    0.2 * (1 + 3 * i + j) as f64
                } else {
                    0.0
                },
            )
        });
        let psd = &matrix * matrix.adjoint() + nalgebra::DMatrix::<Complex64>::identity(n, n);
        if complex {
            Tensor::from_dense(indices.to_vec(), psd.as_slice().to_vec()).unwrap()
        } else {
            Tensor::from_dense(
                indices.to_vec(),
                psd.as_slice().iter().map(|v| v.re).collect(),
            )
            .unwrap()
        }
    }
    #[test]
    fn power_iteration_checks_seed_and_returned_payload_caps() {
        let options = RdmOptions {
            max_intermediate_elements: 3,
            ..RdmOptions::default()
        };
        assert!(matches!(
            power_iterate(seed(), |x| Ok(x.clone()), &options, "left"),
            Err(RdmError::ResourceLimit { .. })
        ));
        let options = RdmOptions {
            max_intermediate_elements: 4,
            ..RdmOptions::default()
        };
        assert!(matches!(
            power_iterate(
                seed(),
                |_| Ok(Tensor::from_dense(vec![new_index(3), new_index(3)], vec![1.0; 9]).unwrap()),
                &options,
                "left"
            ),
            Err(RdmError::ResourceLimit {
                requested: 9,
                limit: 4,
                ..
            })
        ));
    }
    #[test]
    fn stable_kernel_handles_extreme_finite_scales_and_rejects_invalid_rayleigh() {
        for scale in [1e-200, 1e200] {
            let (x, d) = power_iterate(
                seed(),
                |x| {
                    Ok(Tensor::from_dense(
                        x.indices.clone(),
                        x.to_vec::<f64>()
                            .unwrap()
                            .iter()
                            .map(|v| v * scale)
                            .collect(),
                    )
                    .unwrap())
                },
                &RdmOptions::default(),
                "left",
            )
            .unwrap();
            assert!(d.relative_residual < 1e-12);
            assert!((d.eigenvalue / scale - 1.0).abs() < 1e-14);
            assert!(!x.is_complex());
        }
        for scale in [-1.0, f64::INFINITY, f64::NAN] {
            assert!(matches!(
                power_iterate(
                    seed(),
                    |x| Ok(Tensor::from_dense(
                        x.indices.clone(),
                        x.to_vec::<f64>()
                            .unwrap()
                            .iter()
                            .map(|v| v * scale)
                            .collect()
                    )
                    .unwrap()),
                    &RdmOptions::default(),
                    "left"
                ),
                Err(RdmError::InvalidEnvironment { .. })
            ));
        }
        let initial =
            Tensor::from_dense(vec![new_index(1), new_index(1)], vec![Complex64::from(1.0)])
                .unwrap();
        assert!(matches!(
            power_iterate(
                initial,
                |x| Ok(
                    Tensor::from_dense(x.indices.clone(), vec![Complex64::new(1.0, 0.1)]).unwrap()
                ),
                &RdmOptions::default(),
                "right"
            ),
            Err(RdmError::InvalidEnvironment { .. })
        ));
    }
    #[test]
    fn raw_environment_invariants_reject_nonhermitian_and_negative_inputs() {
        for values in [
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, -0.1],
            vec![1.0, 0.0, 0.0, f64::NAN],
        ] {
            let tensor = Tensor::from_dense(vec![new_index(2), new_index(2)], values).unwrap();
            assert!(matches!(
                validate_environment(&tensor, &RdmOptions::default(), "left"),
                Err(RdmError::InvalidEnvironment { .. })
            ));
        }
    }
    #[test]
    fn joint_refinement_keeps_original_tolerance_and_iteration_budgets() {
        use super::super::oracles::fixture;
        let state = fixture(false, 4, 4);
        let options = RdmOptions::default();
        let prepared = prepare(ItebdStateRef::from(&state), &options).unwrap();
        let l = &prepared.environments.left_diagnostics;
        let r = &prepared.environments.right_diagnostics;
        eprintln!(
            "joint real chi4: left={l:?}, right={r:?}, eta_difference={:e}",
            eigenvalue_difference(l, r)
        );
        assert!(
            l.relative_residual <= options.fixed_point_tolerance
                && r.relative_residual <= options.fixed_point_tolerance
        );
        assert!(eigenvalue_difference(l, r) <= options.fixed_point_tolerance);
        assert!(l.iterations <= options.max_iterations && r.iterations <= options.max_iterations);
        // The original fixture needs refinement after separate first crossings (~4945/4796).
        assert!(l.iterations > 4945 && r.iterations > 4796);
        let short = RdmOptions {
            max_iterations: 4945,
            ..options
        };
        assert!(matches!(
            prepare(ItebdStateRef::from(&state), &short),
            Err(RdmError::FixedPointNonConvergence {
                iterations: 4945,
                ..
            })
        ));
    }
}
