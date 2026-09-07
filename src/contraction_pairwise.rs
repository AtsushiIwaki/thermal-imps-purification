use crate::tensor::Tensor;
use tensor4all_core::{
    contract_pair, contract_pair_with_operand_options, PairwiseContractionOptions,
    TensorDynLenError,
};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static PAIRWISE_CALL_COUNT: Cell<usize> = const { Cell::new(0) };
    static PAIRWISE_OPERAND_CONJUGATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn pairwise(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorDynLenError> {
    contract_pair(lhs, rhs).inspect(|_| record_pairwise_success(false))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn pairwise_with_conjugation(
    lhs: &Tensor,
    rhs: &Tensor,
    lhs_conj: bool,
    rhs_conj: bool,
) -> Result<Tensor, TensorDynLenError> {
    contract_pair_with_operand_options(
        lhs,
        rhs,
        PairwiseContractionOptions::new()
            .with_lhs_conj(lhs_conj)
            .with_rhs_conj(rhs_conj),
    )
    .inspect(|_| record_pairwise_success(lhs_conj || rhs_conj))
}

#[cfg(not(test))]
#[cfg_attr(not(test), allow(dead_code))]
fn record_pairwise_success(_operand_conjugated: bool) {}

#[cfg(test)]
fn record_pairwise_success(operand_conjugated: bool) {
    PAIRWISE_CALL_COUNT.with(|count| {
        count.set(count.get() + 1);
    });
    if operand_conjugated {
        PAIRWISE_OPERAND_CONJUGATION_COUNT.with(|count| {
            count.set(count.get() + 1);
        });
    }
}

#[cfg(test)]
pub(crate) fn reset_pairwise_counts() {
    PAIRWISE_CALL_COUNT.with(|count| count.set(0));
    PAIRWISE_OPERAND_CONJUGATION_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn pairwise_call_count() -> usize {
    PAIRWISE_CALL_COUNT.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn pairwise_operand_conjugation_count() -> usize {
    PAIRWISE_OPERAND_CONJUGATION_COUNT.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::{
        pairwise, pairwise_call_count, pairwise_operand_conjugation_count,
        pairwise_with_conjugation, reset_pairwise_counts,
    };
    use crate::tensor::{new_tagged_index, Tensor};
    use crate::contraction_contract_candidate::contract_candidate;
    use crate::test_tensor_support::dense_c64;
    use num_complex::Complex64;
    use tensor4all_core::IndexLike;

    fn asymmetric_pair(complex: bool) -> (Tensor, Tensor) {
        let left = new_tagged_index(2, "pairwise-left").unwrap();
        let shared = new_tagged_index(2, "pairwise-shared").unwrap();
        let right = new_tagged_index(2, "pairwise-right").unwrap();
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

    fn mismatched_common_dimension_pair() -> (Tensor, Tensor) {
        let shared_left = new_tagged_index(2, "pairwise-mismatch-shared").unwrap();
        let shared_right = tensor4all_core::DynIndex::new_with_tags(
            *shared_left.id(),
            3,
            shared_left.tags().clone(),
        );
        let left = Tensor::from_dense(
            vec![shared_left],
            vec![Complex64::new(0.1, 0.2), Complex64::new(0.3, -0.4)],
        )
        .unwrap();
        let right = Tensor::from_dense(
            vec![shared_right],
            vec![
                Complex64::new(0.9, -0.1),
                Complex64::new(-0.2, 0.3),
                Complex64::new(0.4, 0.5),
            ],
        )
        .unwrap();
        (left, right)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.indices, expected.indices);
        let actual = dense_c64("pairwise actual", actual).unwrap();
        let expected = dense_c64("pairwise expected", expected).unwrap();
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(maximum <= 1e-13, "maximum distance {maximum:e}");
    }

    #[test]
    fn pairwise_adapter_matches_frozen_contract_and_materialized_conjugation() {
        let (lhs_real, rhs_real) = asymmetric_pair(false);
        let (lhs_complex, rhs_complex) = asymmetric_pair(true);
        for (lhs, rhs) in [(lhs_real, rhs_real), (lhs_complex, rhs_complex)] {
            let ordinary = pairwise(&lhs, &rhs).unwrap();
            let frozen = contract_candidate(&lhs, &rhs).unwrap();
            assert_close(&ordinary, &frozen);

            let flagged = pairwise_with_conjugation(&lhs, &rhs, true, false).unwrap();
            let materialized = contract_candidate(&lhs.conj(), &rhs).unwrap();
            assert_close(&flagged, &materialized);
            assert_eq!(flagged.indices, materialized.indices);
        }
    }

    #[test]
    fn pairwise_counts_observe_success_without_explicit_reset() {
        let (lhs, rhs) = asymmetric_pair(true);

        pairwise(&lhs, &rhs).unwrap();

        assert_eq!(pairwise_call_count(), 1);
        assert_eq!(pairwise_call_count(), 1);
        assert_eq!(pairwise_operand_conjugation_count(), 0);
        assert_eq!(pairwise_operand_conjugation_count(), 0);

        reset_pairwise_counts();
    }

    #[test]
    fn pairwise_counts_track_only_successful_calls() {
        let (lhs, rhs) = asymmetric_pair(true);

        reset_pairwise_counts();
        pairwise(&lhs, &rhs).unwrap();
        assert_eq!(pairwise_call_count(), 1);
        assert_eq!(pairwise_operand_conjugation_count(), 0);

        reset_pairwise_counts();
        pairwise_with_conjugation(&lhs, &rhs, true, false).unwrap();
        assert_eq!(pairwise_call_count(), 1);
        assert_eq!(pairwise_operand_conjugation_count(), 1);

        let (duplicate_lhs, duplicate_rhs) = mismatched_common_dimension_pair();
        reset_pairwise_counts();
        assert!(pairwise(&duplicate_lhs, &duplicate_rhs).is_err());
        assert_eq!(pairwise_call_count(), 0);
        assert_eq!(pairwise_operand_conjugation_count(), 0);

        reset_pairwise_counts();
        assert!(pairwise_with_conjugation(&duplicate_lhs, &duplicate_rhs, true, true).is_err());
        assert_eq!(pairwise_call_count(), 0);
        assert_eq!(pairwise_operand_conjugation_count(), 0);
    }
}
