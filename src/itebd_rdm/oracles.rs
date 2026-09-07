//! Small independent dense numerical oracles. The explicit cfg keeps standalone AST audits aware
//! that matrix network sums in this file are test-only.
#[cfg(test)]
use crate::itebd_state_view::ItebdSiteRef;
#[cfg(test)]
use nalgebra::DMatrix;
#[cfg(test)]
use num_complex::Complex64;

#[cfg(test)]
pub(super) fn weighted_site_matrices(
    site: ItebdSiteRef<'_>,
    lambda_left: &[f64],
) -> Vec<DMatrix<Complex64>> {
    let roles = [site.left, site.physical, site.ancilla, site.right];
    let axes: Vec<usize> = roles
        .iter()
        .map(|r| site.gamma.indices.iter().position(|i| i == *r).unwrap())
        .collect();
    let dims = site.gamma.dims();
    let strides: Vec<usize> = (0..dims.len())
        .map(|i| dims[..i].iter().product())
        .collect();
    let values: Vec<Complex64> = if site.gamma.is_complex() {
        site.gamma.to_vec().unwrap()
    } else {
        site.gamma
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(Complex64::from)
            .collect()
    };
    (0..site.physical.dim * site.ancilla.dim)
        .map(|pa| {
            DMatrix::from_fn(site.left.dim, site.right.dim, |l, r| {
                let coordinates = [l, pa % site.physical.dim, pa / site.physical.dim, r];
                let offset: usize = (0..4).map(|k| coordinates[k] * strides[axes[k]]).sum();
                values[offset] * lambda_left[l]
            })
        })
        .collect()
}
#[cfg(test)]
pub(super) fn dense_left(
    matrices: &[DMatrix<Complex64>],
    l: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    let mut output = DMatrix::zeros(matrices[0].ncols(), matrices[0].ncols());
    for w in matrices {
        output += w.adjoint() * l * w;
    }
    output
}
#[cfg(test)]
pub(super) fn dense_right(
    matrices: &[DMatrix<Complex64>],
    r: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    let mut output = DMatrix::zeros(matrices[0].nrows(), matrices[0].nrows());
    for w in matrices {
        output += w * r * w.adjoint();
    }
    output
}
#[cfg(test)]
pub(super) fn fixture(complex: bool, ab: usize, ba: usize) -> crate::itebd_auto::ItebdState {
    use crate::itebd_auto::ItebdState;
    use crate::itebd_complex::{ComplexPurifiedMps, ComplexSite};
    use crate::purified_mps::{PurifiedMps, Site};
    use crate::tensor::new_index;
    use crate::tensor::Tensor;
    let bond_ab = new_index(ab);
    let bond_ba = new_index(ba);
    let make = |left: &crate::tensor::Idx, right: &crate::tensor::Idx| {
        let p = new_index(2);
        let a = new_index(2);
        // Deliberately permuted input axes exercise role lookup independently of storage order.
        let indices = vec![a.clone(), right.clone(), left.clone(), p.clone()];
        let data: Vec<Complex64> = (0..4 * left.dim * right.dim)
            .map(|mut i| {
                let av = i % 2;
                i /= 2;
                let r = i % right.dim;
                i /= right.dim;
                let l = i % left.dim;
                i /= left.dim;
                let pv = i;
                Complex64::new(
                    if l == r { 1.0 } else { 0.0 }
                        + 0.07 * ((1 + l + 3 * r + 5 * pv + 7 * av) as f64).sin(),
                    if complex {
                        0.05 * ((2 + 2 * l + r + 3 * pv + 5 * av) as f64).cos()
                    } else {
                        0.0
                    },
                ) / (4.0 * left.dim.max(right.dim) as f64).sqrt()
            })
            .collect();
        let gamma = if complex {
            Tensor::from_dense(indices, data).unwrap()
        } else {
            Tensor::from_dense(indices, data.iter().map(|z| z.re).collect()).unwrap()
        };
        ComplexSite {
            gamma,
            left: left.clone(),
            right: right.clone(),
            phys: p,
            anc: a,
        }
    };
    let a = make(&bond_ba, &bond_ab);
    let b = make(&bond_ab, &bond_ba);
    if complex {
        ItebdState::Complex(ComplexPurifiedMps {
            a,
            b,
            lambda_ab: vec![1.0; ab],
            lambda_ba: vec![1.0; ba],
            lambda_bond_ab: bond_ab,
            lambda_bond_ba: bond_ba,
        })
    } else {
        let real = |site: ComplexSite| Site {
            gamma: site.gamma,
            left: site.left,
            right: site.right,
            phys: site.phys,
            anc: site.anc,
        };
        ItebdState::Real(PurifiedMps {
            a: real(a),
            b: real(b),
            lambda_ab: vec![1.0; ab],
            lambda_ba: vec![1.0; ba],
            lambda_bond_ab: bond_ab,
            lambda_bond_ba: bond_ba,
        })
    }
}
#[cfg(test)]
pub(super) fn matrix(tensor: &crate::tensor::Tensor) -> DMatrix<Complex64> {
    let values: Vec<Complex64> = if tensor.is_complex() {
        tensor.to_vec().unwrap()
    } else {
        tensor
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(Complex64::from)
            .collect()
    };
    DMatrix::from_column_slice(tensor.dims()[0], tensor.dims()[1], &values)
}

/// Full small-interval enumeration, independent of the production tensor recurrence.
#[cfg(test)]
pub(super) fn dense_interval_rdm(
    sites: &[Vec<DMatrix<Complex64>>],
    left: &DMatrix<Complex64>,
    right: &DMatrix<Complex64>,
    d: usize,
) -> DMatrix<Complex64> {
    let dim = d.pow(sites.len() as u32);
    let ancilla_dims: Vec<usize> = sites.iter().map(|site| site.len() / d).collect();
    let ancilla_count: usize = ancilla_dims.iter().product();
    let mut rho = DMatrix::<Complex64>::zeros(dim, dim);
    for p in 0..dim {
        for q in 0..dim {
            for ancillas in 0..ancilla_count {
                let (mut pp, mut qq, mut aa) = (p, q, ancillas);
                let mut ket_product = DMatrix::identity(left.nrows(), left.ncols());
                let mut bra_product = ket_product.clone();
                for (site, a_dim) in sites.iter().zip(&ancilla_dims) {
                    let ancilla = aa % a_dim;
                    aa /= a_dim;
                    ket_product *= &site[pp % d + d * ancilla];
                    bra_product *= &site[qq % d + d * ancilla];
                    pp /= d;
                    qq /= d;
                }
                rho[(p, q)] += (left * &ket_product * right * bra_product.adjoint()).trace();
            }
        }
    }
    let trace = rho.trace();
    assert!(trace.re.is_finite() && trace.re > 0.0 && trace.im.abs() <= 1e-13 * trace.re);
    rho / trace
}
