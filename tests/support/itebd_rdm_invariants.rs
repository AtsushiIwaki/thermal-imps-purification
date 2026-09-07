//! Independent small test-only invariants and fixture transformations.
use crate::oracles::{dense_interval_rdm, dense_left, dense_right, weighted_site_matrices};
use thermal_imps_purification::itebd_state_view::ItebdSiteRef;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// First-site-fastest partial trace over the slowest (last) site.
#[cfg(test)]
pub(super) fn trace_last(rho: &DMatrix<Complex64>, d: usize) -> DMatrix<Complex64> {
    let m = rho.nrows() / d;
    let mut marginal = DMatrix::zeros(m, m);
    for q in 0..m {
        for p in 0..m {
            for a in 0..d {
                marginal[(p, q)] += rho[(p + m * a, q + m * a)];
            }
        }
    }
    marginal
}

#[cfg(test)]
pub(super) fn trace_first(rho: &DMatrix<Complex64>, d: usize) -> DMatrix<Complex64> {
    let m = rho.nrows() / d;
    let mut marginal = DMatrix::zeros(m, m);
    for q in 0..m {
        for p in 0..m {
            for a in 0..d {
                marginal[(p, q)] += rho[(a + d * p, a + d * q)];
            }
        }
    }
    marginal
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(super) enum Transformation {
    PositiveGauge,
    UnitaryGauge,
    Ancilla,
    Physical,
}

#[cfg(test)]
fn block_unitary(dim: usize) -> DMatrix<Complex64> {
    let mut u = DMatrix::identity(dim, dim);
    for j in (0..dim - 1).step_by(2) {
        u[(j, j)] = Complex64::from(1.0 / 2.0_f64.sqrt());
        u[(j + 1, j + 1)] = u[(j, j)];
        u[(j, j + 1)] = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());
        u[(j + 1, j)] = u[(j, j + 1)];
    }
    assert!((&u * u.adjoint() - DMatrix::identity(dim, dim)).norm() < 1e-14);
    u
}

/// Pack matrices using explicit role-axis strides, independent of runtime relabeling.
#[cfg(test)]
fn pack_site(
    site: ItebdSiteRef<'_>,
    matrices: &[DMatrix<Complex64>],
    complex: bool,
) -> crate::tensor::Tensor {
    let roles = [site.left, site.physical, site.ancilla, site.right];
    let axes: Vec<_> = roles
        .iter()
        .map(|r| site.gamma.indices.iter().position(|i| i == *r).unwrap())
        .collect();
    let dims = site.gamma.dims();
    let strides: Vec<usize> = (0..4).map(|i| dims[..i].iter().product()).collect();
    let mut values = vec![Complex64::default(); dims.iter().product()];
    for l in 0..site.left.dim {
        for p in 0..site.physical.dim {
            for a in 0..site.ancilla.dim {
                for r in 0..site.right.dim {
                    let coordinates = [l, p, a, r];
                    let offset: usize = (0..4).map(|k| coordinates[k] * strides[axes[k]]).sum();
                    values[offset] = matrices[p + site.physical.dim * a][(l, r)];
                }
            }
        }
    }
    if complex {
        crate::tensor::Tensor::from_dense(site.gamma.indices.clone(), values).unwrap()
    } else {
        assert!(values.iter().all(|z| z.im == 0.0));
        crate::tensor::Tensor::from_dense(
            site.gamma.indices.clone(),
            values.iter().map(|z| z.re).collect(),
        )
        .unwrap()
    }
}

#[cfg(test)]
pub(super) fn transform_fixture(
    state: &crate::itebd_auto::ItebdState,
    transform: Transformation,
) -> crate::itebd_auto::ItebdState {
    use crate::itebd_auto::ItebdState;
    use crate::itebd_state_view::ItebdStateRef;
    let view = ItebdStateRef::from(state);
    assert!(view
        .lambda_ab()
        .iter()
        .chain(view.lambda_ba())
        .all(|x| *x == 1.0));
    let gauge = |dim, slope| {
        if matches!(transform, Transformation::UnitaryGauge) {
            block_unitary(dim)
        } else {
            DMatrix::from_diagonal(&nalgebra::DVector::from_fn(dim, |j, _| {
                Complex64::from(1.0 + slope * j as f64)
            }))
        }
    };
    let ab = gauge(view.a().right.dim, 0.1);
    let ba = gauge(view.a().left.dim, 0.07);
    let ancilla = block_unitary(2);
    let complex =
        view.a().gamma.is_complex() || !matches!(transform, Transformation::PositiveGauge);
    let mut sites = Vec::new();
    for (site, left, right) in [(view.a(), &ba, &ab), (view.b(), &ab, &ba)] {
        let w = weighted_site_matrices(site, &vec![1.0; site.left.dim]);
        let changed: Vec<_> = match transform {
            Transformation::PositiveGauge | Transformation::UnitaryGauge => {
                let inverse = left.clone().try_inverse().unwrap();
                w.iter().map(|w| &inverse * w * right).collect()
            }
            Transformation::Ancilla => (0..4)
                .map(|pa| {
                    let (p, a) = (pa % 2, pa / 2);
                    &w[p] * ancilla[(a, 0)] + &w[p + 2] * ancilla[(a, 1)]
                })
                .collect(),
            Transformation::Physical => (0..4)
                .map(|pa| {
                    &w[pa]
                        * if pa % 2 == 0 {
                            Complex64::from(1.0)
                        } else {
                            Complex64::new(0.0, 1.0)
                        }
                })
                .collect(),
        };
        assert!(
            w.iter()
                .zip(&changed)
                .map(|(before, after)| (before - after).norm())
                .sum::<f64>()
                > 1e-6,
            "transformation must change the fixture payload"
        );
        sites.push(crate::itebd_complex::ComplexSite {
            gamma: pack_site(site, &changed, complex),
            left: site.left.clone(),
            right: site.right.clone(),
            phys: site.physical.clone(),
            anc: site.ancilla.clone(),
        });
    }
    let b = sites.pop().unwrap();
    let a = sites.pop().unwrap();
    if complex {
        ItebdState::Complex(crate::itebd_complex::ComplexPurifiedMps {
            a,
            b,
            lambda_ab: view.lambda_ab().to_vec(),
            lambda_ba: view.lambda_ba().to_vec(),
            lambda_bond_ab: view.a().right.clone(),
            lambda_bond_ba: view.a().left.clone(),
        })
    } else {
        let real = |s: crate::itebd_complex::ComplexSite| crate::purified_mps::Site {
            gamma: s.gamma,
            left: s.left,
            right: s.right,
            phys: s.phys,
            anc: s.anc,
        };
        ItebdState::Real(crate::purified_mps::PurifiedMps {
            a: real(a),
            b: real(b),
            lambda_ab: view.lambda_ab().to_vec(),
            lambda_ba: view.lambda_ba().to_vec(),
            lambda_bond_ab: view.a().right.clone(),
            lambda_bond_ba: view.a().left.clone(),
        })
    }
}

#[cfg(test)]
pub(super) fn oracle_rdm(
    state: &crate::itebd_auto::ItebdState,
    start: usize,
    length: usize,
) -> DMatrix<Complex64> {
    let view = crate::itebd_state_view::ItebdStateRef::from(state);
    let wa = weighted_site_matrices(view.a(), view.lambda_ba());
    let wb = weighted_site_matrices(view.b(), view.lambda_ab());
    let mut l = DMatrix::identity(view.a().left.dim, view.a().left.dim);
    let mut r = l.clone();
    for _ in 0..512 {
        l = dense_left(&wb, &dense_left(&wa, &l));
        r = dense_right(&wa, &dense_right(&wb, &r));
        assert!(l.norm().is_finite() && l.norm() > 0.0 && r.norm().is_finite() && r.norm() > 0.0);
        l /= Complex64::from(l.norm());
        r /= Complex64::from(r.norm());
    }
    for (x, image) in [
        (&l, dense_left(&wb, &dense_left(&wa, &l))),
        (&r, dense_right(&wa, &dense_right(&wb, &r))),
    ] {
        let eta = x.dotc(&image).re / x.norm_squared();
        assert!(
            (&image - x * Complex64::from(eta)).norm() / image.norm().max(eta.abs() * x.norm())
                <= 1e-13
        );
    }
    let ls = [l.clone(), dense_left(&wa, &l)];
    let rs = [r.clone(), dense_right(&wb, &r)];
    let sites: Vec<_> = (0..length)
        .map(|k| {
            if (start + k) % 2 == 0 {
                wa.clone()
            } else {
                wb.clone()
            }
        })
        .collect();
    dense_interval_rdm(&sites, &ls[start], &rs[(start + length) % 2], 2)
}
