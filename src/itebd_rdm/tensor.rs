use super::RdmError;
use crate::tensor::{Idx, Tensor};
use crate::contraction_pairwise::{pairwise, pairwise_with_conjugation};
use num_complex::Complex64;
use tensor4all_core::{ConjState, IndexLike};
pub(super) fn failure(stage: &'static str, error: impl std::fmt::Display) -> RdmError {
    RdmError::Tensor {
        stage,
        reason: error.to_string(),
    }
}

pub(super) fn checked_product(stage: &'static str, dims: &[usize]) -> Result<usize, RdmError> {
    dims.iter().try_fold(1_usize, |n, d| {
        n.checked_mul(*d)
            .ok_or(RdmError::DimensionOverflow { stage })
    })
}

pub(super) struct RdmContractions {
    limit: usize,
    largest: usize,
}
impl RdmContractions {
    pub fn new(limit: usize) -> Self {
        Self { limit, largest: 0 }
    }
    /// Restart per-interval accounting from the cached preparation peak.
    pub fn reset_largest(&mut self, preparation_largest: usize) {
        self.largest = preparation_largest;
    }
    pub fn largest(&self) -> usize {
        self.largest
    }
    pub fn check(&mut self, stage: &'static str, dims: &[usize]) -> Result<usize, RdmError> {
        let requested = checked_product(stage, dims)?;
        if requested > self.limit {
            return Err(RdmError::ResourceLimit {
                stage,
                requested,
                limit: self.limit,
            });
        }
        self.largest = self.largest.max(requested);
        Ok(requested)
    }
    pub fn pair(
        &mut self,
        stage: &'static str,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> Result<Tensor, RdmError> {
        self.check(stage, &lhs.dims())?;
        self.check(stage, &rhs.dims())?;
        let left: Vec<Idx> = lhs
            .indices
            .iter()
            .map(|i| if lhs_conj { i.conj() } else { i.clone() })
            .collect();
        let right: Vec<Idx> = rhs
            .indices
            .iter()
            .map(|i| if rhs_conj { i.conj() } else { i.clone() })
            .collect();
        require_unique(stage, &left)?;
        require_unique(stage, &right)?;
        let mut matched_left = vec![false; left.len()];
        let mut matched_right = vec![false; right.len()];
        for (i, l) in left.iter().enumerate() {
            for (j, r) in right.iter().enumerate() {
                if compatible_identity(l, r) && l.dim != r.dim {
                    return Err(failure(
                        stage,
                        "compatible index identity has mismatched dimensions",
                    ));
                }
                if l.is_contractable(r) {
                    if matched_left[i] || matched_right[j] {
                        return Err(failure(stage, "ambiguous contraction index match"));
                    }
                    matched_left[i] = true;
                    matched_right[j] = true;
                }
            }
        }
        if !matched_left.contains(&true) {
            return Err(failure(
                stage,
                "disconnected operands require an explicit outer product",
            ));
        }
        let retained: Vec<Idx> = left
            .iter()
            .zip(&matched_left)
            .chain(right.iter().zip(&matched_right))
            .filter_map(|(i, m)| (!m).then_some(i.clone()))
            .collect();
        require_unique(stage, &retained)?;
        let dims: Vec<usize> = retained.iter().map(|i| i.dim).collect();
        self.check(stage, &dims)?;
        let output = if lhs_conj || rhs_conj {
            pairwise_with_conjugation(lhs, rhs, lhs_conj, rhs_conj)
        } else {
            pairwise(lhs, rhs)
        }
        .map_err(|e| failure(stage, e))?;
        self.check(stage, &output.dims())?;
        if output.indices != retained || output.dims() != dims {
            return Err(failure(
                stage,
                "pairwise output disagrees with retained-index plan",
            ));
        }
        Ok(output)
    }

    /// Put explicit source roles into the supplied destination order, optionally scaling one role.
    /// Only elementwise layout/scaling is performed; there are no network sums here.
    pub fn relabel_scale(
        &mut self,
        stage: &'static str,
        tensor: &Tensor,
        source_roles: &[Idx],
        destination_roles: &[Idx],
        scale: Option<(&Idx, &[f64])>,
    ) -> Result<Tensor, RdmError> {
        let count = self.check(stage, &tensor.dims())?;
        if source_roles.len() != tensor.indices.len()
            || destination_roles.len() != source_roles.len()
        {
            return Err(failure(stage, "role count mismatch"));
        }
        require_unique(stage, &tensor.indices)?;
        require_unique(stage, source_roles)?;
        require_unique(stage, destination_roles)?;
        let mut axes = Vec::with_capacity(source_roles.len());
        for (source, destination) in source_roles.iter().zip(destination_roles) {
            let axis = tensor
                .indices
                .iter()
                .position(|i| i == source)
                .ok_or_else(|| failure(stage, "source role missing"))?;
            if tensor.indices[axis].dim != source.dim || source.dim != destination.dim {
                return Err(failure(stage, "role dimension mismatch"));
            }
            axes.push(axis);
        }
        let scale_axis = if let Some((role, values)) = scale {
            let axis = source_roles
                .iter()
                .position(|i| i == role)
                .ok_or_else(|| failure(stage, "scale role missing"))?;
            if role.dim != source_roles[axis].dim
                || values.len() != role.dim
                || values.iter().any(|v| !v.is_finite())
            {
                return Err(failure(stage, "invalid axis scale"));
            }
            Some(axis)
        } else {
            None
        };
        let destination_dims: Vec<usize> = destination_roles.iter().map(|i| i.dim).collect();
        self.check(stage, &destination_dims)?;
        let mut strides = Vec::with_capacity(axes.len());
        let mut stride = 1_usize;
        for dim in tensor.dims() {
            strides.push(stride);
            stride = stride
                .checked_mul(dim)
                .ok_or(RdmError::DimensionOverflow { stage })?;
        }
        // Every coordinate addresses an existing, checked dense payload element.
        let position_and_scale = |mut position: usize| {
            let mut old_position = 0;
            let mut factor = 1.0;
            for (axis, dim) in destination_dims.iter().enumerate() {
                let coordinate = position % dim;
                position /= dim;
                old_position += coordinate * strides[axes[axis]];
                if scale_axis == Some(axis) {
                    factor = scale.unwrap().1[coordinate];
                }
            }
            (old_position, factor)
        };
        if tensor.is_complex() {
            let values = tensor
                .to_vec::<Complex64>()
                .map_err(|e| failure(stage, e))?;
            if values.len() != count {
                return Err(failure(
                    stage,
                    "payload length disagrees with role dimensions",
                ));
            }
            let output: Vec<Complex64> = (0..count)
                .map(|i| {
                    let (old, factor) = position_and_scale(i);
                    values[old] * factor
                })
                .collect();
            Tensor::from_dense(destination_roles.to_vec(), output)
        } else {
            let values = tensor.to_vec::<f64>().map_err(|e| failure(stage, e))?;
            if values.len() != count {
                return Err(failure(
                    stage,
                    "payload length disagrees with role dimensions",
                ));
            }
            let output: Vec<f64> = (0..count)
                .map(|i| {
                    let (old, factor) = position_and_scale(i);
                    values[old] * factor
                })
                .collect();
            Tensor::from_dense(destination_roles.to_vec(), output)
        }
        .map_err(|e| failure(stage, e))
    }
}
fn compatible_identity(left: &Idx, right: &Idx) -> bool {
    match (left.conj_state(), right.conj_state()) {
        (ConjState::Ket, ConjState::Bra) | (ConjState::Bra, ConjState::Ket) => {
            left.conj() == *right
        }
        (ConjState::Undirected, ConjState::Undirected) => left == right,
        _ => false,
    }
}
fn require_unique(stage: &'static str, indices: &[Idx]) -> Result<(), RdmError> {
    for (position, index) in indices.iter().enumerate() {
        if index.dim == 0
            || indices[position + 1..]
                .iter()
                .any(|other| index == other || compatible_identity(index, other))
        {
            return Err(failure(
                stage,
                "zero dimension or duplicate/ambiguous role identity",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{new_index, Tensor};
    use num_complex::Complex64;
    use tensor4all_core::IndexLike;

    #[test]
    fn checked_product_rejects_overflow() {
        assert!(matches!(
            checked_product("overflow", &[usize::MAX, 2]),
            Err(RdmError::DimensionOverflow { .. })
        ));
    }
    #[test]
    fn planner_rejects_compatible_identity_dimension_mismatch_before_call() {
        let shared = new_index(2);
        let wrong =
            tensor4all_core::DynIndex::new_with_tags(*shared.id(), 3, shared.tags().clone());
        let x = Tensor::from_dense(vec![shared], vec![1.0; 2]).unwrap();
        let y = Tensor::from_dense(vec![wrong], vec![1.0; 3]).unwrap();
        assert!(matches!(
            RdmContractions::new(100).pair("bad", &x, &y, false, false),
            Err(RdmError::Tensor { .. })
        ));
    }
    #[test]
    fn planner_checks_input_and_output_limits_and_counts_actual_elements() {
        let shared = new_index(2);
        let x = Tensor::from_dense(vec![new_index(3), shared.clone()], vec![1.0; 6]).unwrap();
        let y = Tensor::from_dense(vec![shared, new_index(4)], vec![1.0; 8]).unwrap();
        for limit in [5, 10] {
            assert!(matches!(
                RdmContractions::new(limit).pair("cap", &x, &y, false, false),
                Err(RdmError::ResourceLimit { .. })
            ));
        }
        let mut contractions = RdmContractions::new(12);
        let result = contractions.pair("good", &x, &y, false, false).unwrap();
        assert_eq!(result.dims(), vec![3, 4]);
        assert_eq!(contractions.largest(), 12);
    }
    #[test]
    fn role_scale_handles_permuted_axes_without_backend_promotion() {
        for complex in [false, true] {
            let x = new_index(2);
            let y = new_index(3);
            let values: Vec<f64> = (1..=6).map(|i| i as f64).collect();
            let tensor = if complex {
                Tensor::from_dense(
                    vec![y.clone(), x.clone()],
                    values.iter().map(|v| Complex64::new(*v, -*v)).collect(),
                )
                .unwrap()
            } else {
                Tensor::from_dense(vec![y.clone(), x.clone()], values).unwrap()
            };
            let mut contractions = RdmContractions::new(6);
            let output = contractions
                .relabel_scale(
                    "roles",
                    &tensor,
                    &[x.clone(), y.clone()],
                    &[new_index(2), new_index(3)],
                    Some((&x, &[2.0, 3.0])),
                )
                .unwrap();
            assert_eq!(output.is_complex(), complex);
            let expected = vec![2.0, 12.0, 4.0, 15.0, 6.0, 18.0];
            if complex {
                assert_eq!(
                    output.to_vec::<Complex64>().unwrap(),
                    expected
                        .iter()
                        .map(|v| Complex64::new(*v, -*v))
                        .collect::<Vec<_>>()
                );
            } else {
                assert_eq!(output.to_vec::<f64>().unwrap(), expected);
            }
            assert!(matches!(
                RdmContractions::new(5).relabel_scale(
                    "cap",
                    &tensor,
                    &[x.clone(), y.clone()],
                    &[new_index(2), new_index(3)],
                    None
                ),
                Err(RdmError::ResourceLimit { .. })
            ));
        }
    }
    #[test]
    fn planner_rejects_ambiguous_duplicate_matches_and_disconnected_pairs() {
        let shared = new_index(2);
        let mut lhs = Tensor::from_dense(vec![shared.clone(), new_index(2)], vec![1.0; 4]).unwrap();
        lhs.indices[1] = shared.clone();
        let rhs = Tensor::from_dense(vec![shared], vec![1.0; 2]).unwrap();
        assert!(matches!(
            RdmContractions::new(100).pair("duplicate", &lhs, &rhs, false, false),
            Err(RdmError::Tensor { .. })
        ));
        let other = Tensor::from_dense(vec![new_index(2)], vec![1.0; 2]).unwrap();
        assert!(matches!(
            RdmContractions::new(100).pair("disconnected", &other, &rhs, false, false),
            Err(RdmError::Tensor { .. })
        ));
    }
    #[test]
    fn pairwise_conjugation_flags_preserve_complex_values() {
        let shared = new_index(2);
        let a = [Complex64::new(1.0, 0.5), Complex64::new(-0.7, 0.2)];
        let b = [Complex64::new(0.3, -0.4), Complex64::new(1.2, 0.8)];
        let lhs = Tensor::from_dense(vec![shared.clone()], a.to_vec()).unwrap();
        let rhs = Tensor::from_dense(vec![shared], b.to_vec()).unwrap();
        for lc in [false, true] {
            for rc in [false, true] {
                let result = RdmContractions::new(2)
                    .pair("conjugation", &lhs, &rhs, lc, rc)
                    .unwrap();
                let expected: Complex64 = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| {
                        (if lc { a.conj() } else { *a }) * (if rc { b.conj() } else { b })
                    })
                    .sum();
                assert!((result.to_vec::<Complex64>().unwrap()[0] - expected).norm() < 1e-14);
            }
        }
    }
    #[test]
    fn relabel_scale_rejects_payload_metadata_mismatch_without_panicking() {
        for complex in [false, true] {
            let indices = vec![new_index(2), new_index(2)];
            let mut tensor = if complex {
                Tensor::from_dense(indices, vec![Complex64::from(1.0); 4]).unwrap()
            } else {
                Tensor::from_dense(indices, vec![1.0; 4]).unwrap()
            };
            tensor.indices[0].dim = 3;
            assert!(matches!(
                RdmContractions::new(10).relabel_scale(
                    "corrupt metadata",
                    &tensor,
                    &tensor.indices,
                    &tensor.indices,
                    None
                ),
                Err(RdmError::Tensor { .. })
            ));
        }
    }
}
