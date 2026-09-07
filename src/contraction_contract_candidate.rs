use crate::tensor::Tensor;
use tensor4all_core::TensorDynLenError;

#[cfg(test)]
pub(crate) fn contract_candidate(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorDynLenError> {
    tensor4all_core::contract(&[lhs, rhs])
}

#[cfg(test)]
mod tests {
    use super::contract_candidate;
    use crate::tensor::{new_tagged_index, Tensor};
    use crate::contraction_pairwise::pairwise;
    use crate::test_tensor_support::dense_c64;
    use num_complex::Complex64;

    fn asymmetric_pair(complex: bool) -> (Tensor, Tensor) {
        let left = new_tagged_index(2, "candidate-left").unwrap();
        let shared = new_tagged_index(2, "candidate-shared").unwrap();
        let right = new_tagged_index(2, "candidate-right").unwrap();
        if complex {
            (
                Tensor::from_dense(
                    vec![left, shared.clone()],
                    vec![
                        Complex64::new(0.2, 0.7),
                        Complex64::new(-0.4, 0.1),
                        Complex64::new(0.8, -0.3),
                        Complex64::new(0.5, 0.9),
                    ],
                )
                .unwrap(),
                Tensor::from_dense(
                    vec![shared, right],
                    vec![
                        Complex64::new(-0.1, 0.6),
                        Complex64::new(0.3, -0.2),
                        Complex64::new(0.9, 0.4),
                        Complex64::new(-0.7, 0.5),
                    ],
                )
                .unwrap(),
            )
        } else {
            (
                Tensor::from_dense(vec![left, shared.clone()], vec![0.2, -0.4, 0.8, 0.5]).unwrap(),
                Tensor::from_dense(vec![shared, right], vec![-0.1, 0.3, 0.9, -0.7]).unwrap(),
            )
        }
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.indices, expected.indices);
        let actual = dense_c64("candidate actual", actual).unwrap();
        let expected = dense_c64("candidate expected", expected).unwrap();
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(maximum <= 1e-13, "maximum distance {maximum:e}");
    }

    #[test]
    fn contract_candidate_contracts_real_and_complex_pairs() {
        for (lhs, rhs) in [asymmetric_pair(false), asymmetric_pair(true)] {
            let candidate = contract_candidate(&lhs, &rhs).unwrap();
            let pairwise = pairwise(&lhs, &rhs).unwrap();
            assert_close(&candidate, &pairwise);
        }
    }
}
