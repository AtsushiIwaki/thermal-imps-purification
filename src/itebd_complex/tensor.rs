use crate::itebd_error::ItebdError;
use crate::tensor::{Idx, Tensor, Truncation};
use num_complex::Complex64;
use tensor4all_core::svd::{svd_with, SvdOptions};
use tensor4all_core::SvdTruncationPolicy;

const COMPLEX_FROM_FN_STAGE: &str = "complex_from_fn";
const COMPLEX_RELABEL_STAGE: &str = "complex_relabel";
const COMPLEX_SCALE_BOND_STAGE: &str = "complex_scale_bond";
const COMPLEX_SCALAR_STAGE: &str = "complex_scalar";
const COMPLEX_SVD_STAGE: &str = "complex_svd_bond";

fn tensor_error(stage: &'static str, message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage,
        message: message.to_string(),
    }
}

fn require_complex(stage: &'static str, tensor: &Tensor) -> Result<(), ItebdError> {
    if tensor.is_complex() {
        Ok(())
    } else {
        Err(tensor_error(stage, "expected Complex64 tensor storage"))
    }
}

fn complex_dense(stage: &'static str, tensor: &Tensor) -> Result<Vec<Complex64>, ItebdError> {
    require_complex(stage, tensor)?;
    tensor
        .to_vec::<Complex64>()
        .map_err(|error| tensor_error(stage, error))
}

pub(crate) fn complex_from_fn(
    indices: &[Idx],
    f: impl Fn(&[usize]) -> Complex64,
) -> Result<Tensor, ItebdError> {
    let dims: Vec<usize> = indices.iter().map(|index| index.dim).collect();
    if let Some(position) = dims.iter().position(|&dimension| dimension == 0) {
        return Err(tensor_error(
            COMPLEX_FROM_FN_STAGE,
            format!("index {position} has zero dimension"),
        ));
    }
    let total = dims
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| tensor_error(COMPLEX_FROM_FN_STAGE, "tensor element count overflow"))?;
    let mut data = Vec::with_capacity(total);
    let mut multi = vec![0usize; dims.len()];
    for _ in 0..total {
        data.push(f(&multi));
        for (coordinate, &dimension) in multi.iter_mut().zip(&dims) {
            *coordinate += 1;
            if *coordinate < dimension {
                break;
            }
            *coordinate = 0;
        }
    }
    Tensor::from_dense(indices.to_vec(), data)
        .map_err(|error| tensor_error(COMPLEX_FROM_FN_STAGE, error))
}

pub(crate) fn complex_relabel(tensor: &Tensor, from: &Idx, to: &Idx) -> Result<Tensor, ItebdError> {
    require_complex(COMPLEX_RELABEL_STAGE, tensor)?;
    if from.dim != to.dim {
        return Err(tensor_error(
            COMPLEX_RELABEL_STAGE,
            format!("index dimension mismatch: {} != {}", from.dim, to.dim),
        ));
    }
    if !tensor.indices.iter().any(|index| index.id == from.id) {
        return Err(tensor_error(
            COMPLEX_RELABEL_STAGE,
            "source index is not present in tensor",
        ));
    }
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
    Tensor::from_dense(indices, complex_dense(COMPLEX_RELABEL_STAGE, tensor)?)
        .map_err(|error| tensor_error(COMPLEX_RELABEL_STAGE, error))
}

pub(crate) fn complex_scale_bond(
    tensor: &Tensor,
    bond: &Idx,
    factors: &[f64],
) -> Result<Tensor, ItebdError> {
    let dimensions: Vec<usize> = tensor.indices.iter().map(|index| index.dim).collect();
    let position = tensor
        .indices
        .iter()
        .position(|index| index.id == bond.id)
        .ok_or_else(|| {
            tensor_error(
                COMPLEX_SCALE_BOND_STAGE,
                "bond index is not present in tensor",
            )
        })?;
    if dimensions[position] != factors.len() {
        return Err(tensor_error(
            COMPLEX_SCALE_BOND_STAGE,
            format!(
                "factor count {} does not match bond dimension {}",
                factors.len(),
                dimensions[position]
            ),
        ));
    }

    let mut data = complex_dense(COMPLEX_SCALE_BOND_STAGE, tensor)?;
    let mut multi = vec![0usize; dimensions.len()];
    for value in &mut data {
        *value *= factors[multi[position]];
        for (coordinate, &dimension) in multi.iter_mut().zip(&dimensions) {
            *coordinate += 1;
            if *coordinate < dimension {
                break;
            }
            *coordinate = 0;
        }
    }
    Tensor::from_dense(tensor.indices.clone(), data)
        .map_err(|error| tensor_error(COMPLEX_SCALE_BOND_STAGE, error))
}

pub(crate) fn complex_scalar(tensor: &Tensor) -> Result<Complex64, ItebdError> {
    let values = complex_dense(COMPLEX_SCALAR_STAGE, tensor)?;
    if values.len() != 1 {
        return Err(tensor_error(
            COMPLEX_SCALAR_STAGE,
            format!("expected one scalar value, got {}", values.len()),
        ));
    }
    Ok(values[0])
}

pub(crate) struct ComplexSvdResult {
    /// Carries indices `[left..., bond]`.
    pub(crate) u: Tensor,
    /// Retained real singular values in descending order.
    pub(crate) s: Vec<f64>,
    /// The U-side bond index.
    pub(crate) bond: Idx,
    /// Carries indices `[right..., bond_sim]` and stores V. Contract this factor with
    /// conjugation to reconstruct `U S V^H`.
    pub(crate) v: Tensor,
}

pub(crate) fn complex_svd_bond(
    tensor: &Tensor,
    left: &[Idx],
    truncation: &Truncation,
) -> Result<ComplexSvdResult, ItebdError> {
    require_complex(COMPLEX_SVD_STAGE, tensor)?;
    let policy = SvdTruncationPolicy::new(truncation.epsilon)
        .with_squared_values()
        .with_discarded_tail_sum();
    let mut options = SvdOptions::new().with_policy(policy);
    if let Some(max_bond) = truncation.max_bond {
        options = options.with_max_bond_dim(max_bond);
    }
    let (u, singular_tensor, v_raw) = svd_with::<Complex64>(tensor, left, &options)
        .map_err(|error| tensor_error(COMPLEX_SVD_STAGE, error))?;
    let dimensions = singular_tensor.dims();
    if dimensions.len() != 2 || dimensions[0] != dimensions[1] {
        return Err(tensor_error(
            COMPLEX_SVD_STAGE,
            format!("expected square rank-2 singular tensor, got {dimensions:?}"),
        ));
    }
    let rank = dimensions[0];
    let singular_dense = complex_dense(COMPLEX_SVD_STAGE, &singular_tensor)?;
    let mut singular_values = Vec::with_capacity(rank);
    for index in 0..rank {
        let singular = singular_dense[index * (rank + 1)];
        if !singular.re.is_finite() || !singular.im.is_finite() || singular.im != 0.0 {
            return Err(tensor_error(
                COMPLEX_SVD_STAGE,
                format!("invalid singular value {singular:?} at index {index}"),
            ));
        }
        singular_values.push(singular.re);
    }

    let left_ids: Vec<_> = left.iter().map(|index| index.id).collect();
    let bond = u
        .indices
        .iter()
        .find(|index| !left_ids.contains(&index.id))
        .cloned()
        .ok_or_else(|| tensor_error(COMPLEX_SVD_STAGE, "SVD U factor has no new bond index"))?;
    let bond_sim = singular_tensor.indices.get(1).cloned().ok_or_else(|| {
        tensor_error(
            COMPLEX_SVD_STAGE,
            "singular tensor has no simulated bond index",
        )
    })?;
    let v = complex_relabel(&v_raw, &bond, &bond_sim)?;

    Ok(ComplexSvdResult {
        u,
        s: singular_values,
        bond,
        v,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        complex_from_fn, complex_relabel, complex_scalar, complex_scale_bond, complex_svd_bond,
    };
    use crate::tensor::{new_index, Tensor, Truncation};
    use crate::contraction_pairwise::{pairwise, pairwise_with_conjugation};
    use num_complex::Complex64;

    fn genuinely_complex_unequal_matrix() -> (Vec<Complex64>, crate::tensor::Tensor) {
        let row = new_index(2);
        let column = new_index(3);
        let data = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.75, -2.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(-1.0, 1.0),
        ];
        let tensor =
            complex_from_fn(&[row, column], |index| data[index[0] + 2 * index[1]]).unwrap();
        (data, tensor)
    }

    #[test]
    fn complex_svd_reconstructs_genuinely_complex_unequal_matrix() {
        // Mutation caught: removing conjugation from the returned right singular factor changes
        // at least one genuinely complex reconstructed entry.
        let (expected, matrix) = genuinely_complex_unequal_matrix();
        let left = matrix.indices[0].clone();
        let result = complex_svd_bond(&matrix, &[left], &Truncation::default()).unwrap();
        let right_bond = result.v.indices.last().unwrap().clone();
        let singular = complex_from_fn(&[result.bond.clone(), right_bond], |index| {
            if index[0] == index[1] {
                Complex64::new(result.s[index[0]], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
        let us = pairwise(&result.u, &singular).unwrap();
        let reconstructed = pairwise_with_conjugation(&us, &result.v, false, true).unwrap();
        let actual = reconstructed.to_vec::<Complex64>().unwrap();

        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).norm() < 1e-12,
                "{actual:?} != {expected:?}"
            );
        }

        let without_conjugation = pairwise(&us, &result.v)
            .unwrap()
            .to_vec::<Complex64>()
            .unwrap();
        assert!(
            without_conjugation
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| (actual - expected).norm() > 1e-6),
            "the fixture must detect removal of right-factor conjugation"
        );
    }

    #[test]
    fn complex_svd_uses_relative_discarded_weight_cutoff() {
        let row = new_index(2);
        let column = new_index(2);
        let matrix = complex_from_fn(&[row.clone(), column], |index| match index {
            [0, 0] => Complex64::new(3.0, 0.0),
            [1, 1] => Complex64::new(1e-8, 0.0),
            _ => Complex64::new(0.0, 0.0),
        })
        .unwrap();

        let result = complex_svd_bond(
            &matrix,
            &[row],
            &Truncation {
                epsilon: 1e-12,
                max_bond: None,
            },
        )
        .unwrap();

        assert_eq!(result.s.len(), 1);
        assert!((result.s[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn complex_svd_respects_max_bond() {
        let row = new_index(3);
        let column = new_index(3);
        let matrix = complex_from_fn(&[row.clone(), column], |index| {
            if index[0] == index[1] {
                Complex64::new((3 - index[0]) as f64, 0.25 * index[0] as f64)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();

        let result = complex_svd_bond(
            &matrix,
            &[row],
            &Truncation {
                epsilon: 0.0,
                max_bond: Some(2),
            },
        )
        .unwrap();

        assert_eq!(result.s.len(), 2);
        assert_eq!(result.u.dims().last(), Some(&2));
        assert_eq!(result.v.dims().last(), Some(&2));
    }

    #[test]
    fn complex_relabel_rejects_missing_index() {
        // Mutation caught: silently accepting an absent source index leaves malformed topology
        // undetected instead of returning the required typed tensor error.
        let present = new_index(2);
        let missing = new_index(2);
        let replacement = new_index(2);
        let tensor = complex_from_fn(&[present], |_| Complex64::new(1.0, 0.5)).unwrap();

        assert!(matches!(
            complex_relabel(&tensor, &missing, &replacement),
            Err(crate::itebd_error::ItebdError::TensorOperation {
                stage: "complex_relabel",
                ..
            })
        ));
    }

    #[test]
    fn complex_scale_bond_scales_selected_complex_axis() {
        // Mutation caught: selecting the wrong tensor coordinate scales the wrong column.
        let row = new_index(2);
        let bond = new_index(2);
        let tensor = complex_from_fn(&[row, bond.clone()], |index| {
            Complex64::new((1 + index[0] + 2 * index[1]) as f64, index[1] as f64)
        })
        .unwrap();

        let scaled = complex_scale_bond(&tensor, &bond, &[2.0, -1.0]).unwrap();

        assert_eq!(
            scaled.to_vec::<Complex64>().unwrap(),
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(-3.0, -1.0),
                Complex64::new(-4.0, -1.0),
            ]
        );
    }

    #[test]
    fn complex_scale_bond_rejects_malformed_factors() {
        let bond = new_index(2);
        let tensor = complex_from_fn(&[bond.clone()], |_| Complex64::new(1.0, 0.0)).unwrap();

        assert!(matches!(
            complex_scale_bond(&tensor, &bond, &[1.0]),
            Err(crate::itebd_error::ItebdError::TensorOperation {
                stage: "complex_scale_bond",
                ..
            })
        ));
    }

    #[test]
    fn complex_scalar_reads_genuinely_complex_value() {
        let expected = Complex64::new(-0.75, 2.5);
        let scalar = complex_from_fn(&[], |_| expected).unwrap();

        assert_eq!(complex_scalar(&scalar).unwrap(), expected);
    }

    #[test]
    fn complex_scalar_rejects_nonscalar_and_real_storage() {
        let nonscalar = complex_from_fn(&[new_index(2)], |_| Complex64::new(1.0, 0.0)).unwrap();
        let real = Tensor::from_dense(Vec::new(), vec![1.0]).unwrap();

        for tensor in [&nonscalar, &real] {
            assert!(matches!(
                complex_scalar(tensor),
                Err(crate::itebd_error::ItebdError::TensorOperation {
                    stage: "complex_scalar",
                    ..
                })
            ));
        }
    }
}
