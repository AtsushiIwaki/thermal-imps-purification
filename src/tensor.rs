use tensor4all_core::svd::{svd_with, SvdOptions};
use tensor4all_core::{IndexLike, SvdTruncationPolicy};

pub type Idx = tensor4all_core::DynIndex;
pub type Tensor = tensor4all_core::TensorDynLen;

/// Exact Tensor4all N-ary `contract` display text for two disconnected inputs.
pub(crate) const LEGACY_DISCONNECTED_NETWORK_ERROR: &str =
    "TensorDynLen TensorDynLen failed: Disconnected tensor network: 2 components found";

pub fn new_index(d: usize) -> Idx {
    tensor4all_core::DynIndex::new_dyn(d)
}

pub fn new_tagged_index(d: usize, tag: &str) -> Result<Idx, tensor4all_core::TagSetError> {
    tensor4all_core::DynIndex::new_dyn_with_tag(d, tag)
}

pub fn dim(ix: &Idx) -> usize {
    ix.dim
}

/// col-major（最初の index が最速）でデータを充填してテンソルを作る。
pub fn from_fn(indices: &[Idx], f: impl Fn(&[usize]) -> f64) -> Tensor {
    let dims: Vec<usize> = indices.iter().map(|ix| ix.dim).collect();
    let total: usize = dims.iter().product::<usize>().max(1);
    let mut data = vec![0.0_f64; total];
    let mut multi = vec![0usize; dims.len()];
    for slot in data.iter_mut() {
        *slot = f(&multi);
        for k in 0..dims.len() {
            multi[k] += 1;
            if multi[k] < dims[k] {
                break;
            }
            multi[k] = 0;
        }
    }
    Tensor::from_dense(indices.to_vec(), data).unwrap()
}

pub fn contract(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    if !has_contractable_index(lhs, rhs) {
        panic!("binary contract failed: {LEGACY_DISCONNECTED_NETWORK_ERROR}");
    }
    crate::contraction_pairwise::pairwise(lhs, rhs).expect("binary contract failed")
}

pub(crate) fn has_contractable_index(lhs: &Tensor, rhs: &Tensor) -> bool {
    lhs.indices
        .iter()
        .any(|left| rhs.indices.iter().any(|right| left.is_contractable(right)))
}

pub fn scalar(t: &Tensor) -> f64 {
    let v = t.to_vec::<f64>().unwrap();
    assert_eq!(v.len(), 1, "scalar() expects a single-element tensor, got {}", v.len());
    v[0]
}

/// index の Id を付け替える（同 dim）。
pub fn relabel(t: &Tensor, from: &Idx, to: &Idx) -> Tensor {
    assert_eq!(from.dim, to.dim, "relabel dim mismatch");
    let new_inds: Vec<Idx> = t
        .indices
        .iter()
        .map(|ix| if ix.id == from.id { to.clone() } else { ix.clone() })
        .collect();
    Tensor::from_dense(new_inds, t.to_vec::<f64>().unwrap()).unwrap()
}

/// `t` の index `bond`（Id で照合）に沿って、bond 座標 k のスライスを `factors[k]` 倍する。
pub fn scale_bond(t: &Tensor, bond: &Idx, factors: &[f64]) -> Tensor {
    let dims: Vec<usize> = t.indices.iter().map(|ix| ix.dim).collect();
    let pos = t
        .indices
        .iter()
        .position(|ix| ix.id == bond.id)
        .expect("scale_bond: bond index not found in tensor");
    assert_eq!(dims[pos], factors.len(), "scale_bond: factors length mismatch");
    let mut data = t.to_vec::<f64>().unwrap(); // col-major
    let mut multi = vec![0usize; dims.len()];
    for slot in data.iter_mut() {
        *slot *= factors[multi[pos]];
        for k in 0..dims.len() {
            multi[k] += 1;
            if multi[k] < dims[k] {
                break;
            }
            multi[k] = 0;
        }
    }
    Tensor::from_dense(t.indices.clone(), data).unwrap()
}

/// SVD 打ち切りの制御パラメタ。`epsilon` を主、`max_bond` を任意の安全上限とする。
#[derive(Debug, Clone, Copy)]
pub struct Truncation {
    /// 相対 discarded weight の上限: Σ_{i≥r} σ_i² / Σ σ_i² ≤ epsilon。
    pub epsilon: f64,
    /// 安全上限の結合次元（None で無制限）。
    pub max_bond: Option<usize>,
}

impl Default for Truncation {
    fn default() -> Self {
        Self { epsilon: 1e-12, max_bond: None }
    }
}

pub struct SvdResult {
    /// `u` carries indices `[left..., bond]`.
    pub u: Tensor,
    /// Singular values (descending), length = bond dim.
    pub s: Vec<f64>,
    /// The u-side bond index. NOTE: `v`'s bond-side index is NOT this `bond`
    /// but its sim-partner `bond_sim`; see the `v` field.
    pub bond: Idx,
    /// `v` carries indices `[right..., bond_sim]`. Its bond-side index is
    /// `bond_sim` (a distinct `DynId` from `bond`), retrievable as
    /// `v.indices.last().unwrap()`. To make two MPS neighbors share one bond,
    /// relabel `bond_sim` -> the desired shared bond id (or use `scale_bond`
    /// to fold singular values in without a separate diagonal tensor).
    pub v: Tensor,
}

/// SVD of `t` splitting at `left` indices.
///
/// Returns `u` with indices `[left..., bond]`, singular values `s` (descending),
/// and `v` with indices `[right..., bond_sim]` where `bond_sim` is the partner
/// index of `bond` in the core's diagonal S matrix. To reconstruct the original
/// tensor: build `sdiag = from_fn(&[r.bond.clone(), v_bond], ...)` where
/// `v_bond = r.v.indices.last().unwrap().clone()`, then contract in two binary stages.
pub fn svd_bond(t: &Tensor, left: &[Idx], trunc: &Truncation) -> SvdResult {
    // 相対・二乗和の discarded weight 打ち切りを core に委譲。
    let policy = SvdTruncationPolicy::new(trunc.epsilon)
        .with_squared_values()
        .with_discarded_tail_sum();
    let mut opts = SvdOptions::new().with_policy(policy);
    if let Some(chi) = trunc.max_bond {
        opts = opts.with_max_bond_dim(chi);
    }
    let (u, s_tensor, v_raw) = svd_with::<f64>(t, left, &opts).unwrap();
    // s is a 2D diagonal matrix [bond, bond_sim]; extract diagonal via stride k*(r+1)
    let s_dense = s_tensor.to_vec::<f64>().unwrap();
    let r = s_tensor.dims()[0];
    let svals: Vec<f64> = (0..r).map(|k| s_dense[k * (r + 1)]).collect();
    // bond = u's last index (the fresh bond created by svd_with, not among the left indices)
    let left_ids: Vec<_> = left.iter().map(|x| x.id).collect();
    let bond = u
        .indices
        .iter()
        .find(|ix| !left_ids.contains(&ix.id))
        .unwrap()
        .clone();
    // The core returns s with indices [bond, bond_sim] and v_raw with [right..., bond].
    // Relabel v's bond to bond_sim so that [u, sdiag, v] contracts unambiguously:
    //   u contracts with sdiag on bond; v contracts with sdiag on bond_sim.
    let bond_sim = s_tensor.indices[1].clone();
    let v = relabel(&v_raw, &bond, &bond_sim);
    SvdResult { u, s: svals, bond, v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contraction_pairwise::{pairwise_call_count, reset_pairwise_counts};
    use approx::assert_abs_diff_eq;

    #[test]
    fn legacy_binary_adapter_delegates_to_tensor4all_pairwise() {
        let left = new_index(2);
        let shared = new_index(2);
        let right = new_index(3);
        let lhs = from_fn(&[left.clone(), shared.clone()], |x| (x[0] + 2 * x[1]) as f64);
        let rhs = from_fn(&[shared, right.clone()], |x| (1 + x[0] + 3 * x[1]) as f64);

        reset_pairwise_counts();
        let output = contract(&lhs, &rhs);

        assert_eq!(pairwise_call_count(), 1);
        assert_eq!(output.indices, vec![left, right]);
    }

    #[test]
    fn legacy_binary_adapter_preserves_disconnected_network_error() {
        let lhs = from_fn(&[new_index(2)], |_| 1.0);
        let rhs = from_fn(&[new_index(2)], |_| 1.0);

        reset_pairwise_counts();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| contract(&lhs, &rhs)))
            .unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap();

        assert_eq!(
            message,
            "binary contract failed: TensorDynLen TensorDynLen failed: Disconnected tensor network: 2 components found"
        );
        assert_eq!(pairwise_call_count(), 0);
    }

    #[test]
    fn from_fn_is_col_major_and_contracts() {
        // A_{i j} = i + 2*j  (i,j in 0..2) -> col-major [0,1,2,3]
        let i = new_index(2);
        let j = new_index(2);
        let a = from_fn(&[i.clone(), j.clone()], |idx| (idx[0] + 2 * idx[1]) as f64);
        // contract A with zero tensor gives 0
        let zero = from_fn(&[i.clone(), j.clone()], |_| 0.0);
        let zero_contraction = scalar(&contract(&a, &zero));
        assert_abs_diff_eq!(zero_contraction, 0.0);
        // 単位ベクトルとの縮約で列和を確認
        let ones_j = from_fn(&[j.clone()], |_| 1.0);
        let col_sum = contract(&a, &ones_j); // over j -> dims [i]
        assert_eq!(col_sum.dims(), vec![2]);
        // A_{i0}+A_{i1} = (i+0) + (i+2) = 2i+2 -> [2, 4]
        let v = col_sum.to_vec::<f64>().unwrap();
        assert_abs_diff_eq!(v[0], 2.0);
        assert_abs_diff_eq!(v[1], 4.0);
    }

    #[test]
    fn svd_bond_reconstructs() {
        let i = new_index(3);
        let j = new_index(3);
        // 一般の行列（rank-3 フルランク: 1 + 3*i_val + j_val）
        let m = from_fn(&[i.clone(), j.clone()], |x| (1 + x[0] * 3 + x[1]) as f64);
        let r = svd_bond(&m, &[i.clone()], &Truncation::default());
        // v の bond 側 index（bond_sim）を取得して対角行列を構成
        // svd_bond relabels v's bond to bond_sim, so v.indices.last() is bond_sim
        let bond_sim = r.v.indices.last().unwrap().clone();
        let sdiag = from_fn(&[r.bond.clone(), bond_sim], |x| {
            if x[0] == x[1] { r.s[x[0]] } else { 0.0 }
        });
        // U * diag(s) * V を再構成して元と一致
        let us = contract(&r.u, &sdiag);
        let recon = contract(&us, &r.v);
        let orig = m.to_vec::<f64>().unwrap();
        let got = recon.to_vec::<f64>().unwrap();
        for (a, b) in orig.iter().zip(got.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn scale_bond_scales_along_axis() {
        // T_{i j}, i,j in 0..2, T = i + 2*j（col-major [0,1,2,3]）
        let i = new_index(2);
        let j = new_index(2);
        let t = from_fn(&[i.clone(), j.clone()], |x| (x[0] + 2 * x[1]) as f64);
        // j 軸を [10, 100] でスケール: 列 j=0 を ×10, j=1 を ×100
        let scaled = scale_bond(&t, &j, &[10.0, 100.0]);
        // 行 i=0 と全列の和 = T[0,0]*10 + T[0,1]*100 = 0*10 + 2*100 = 200
        // 行 i=1 = 1*10 + 3*100 = 310
        let ones_j = from_fn(&[j.clone()], |_| 1.0);
        let row = contract(&scaled, &ones_j).to_vec::<f64>().unwrap();
        assert_abs_diff_eq!(row[0], 200.0);
        assert_abs_diff_eq!(row[1], 310.0);
    }

    #[test]
    fn svd_bond_truncates_by_discarded_weight() {
        // 既知特異値 [1.0, 0.1, 0.01, 1e-4] を持つ対角行列
        let i = new_index(4);
        let j = new_index(4);
        let d = [1.0_f64, 0.1, 0.01, 1e-4];
        let m = from_fn(&[i.clone(), j.clone()], |x| if x[0] == x[1] { d[x[0]] } else { 0.0 });

        // epsilon 極小 → 全 4 本保持
        let r_all = svd_bond(&m, &[i.clone()], &Truncation { epsilon: 1e-14, max_bond: None });
        assert_eq!(r_all.s.len(), 4);
        assert_abs_diff_eq!(*r_all.s.last().unwrap(), 1e-4, epsilon = 1e-9);

        // epsilon = 5e-3 → 最小 2 本 (σ=1e-4, 0.01) を捨て rank 2、σ_min=0.1
        let r_trunc = svd_bond(&m, &[i.clone()], &Truncation { epsilon: 5e-3, max_bond: None });
        assert_eq!(r_trunc.s.len(), 2);
        assert_abs_diff_eq!(*r_trunc.s.last().unwrap(), 0.1, epsilon = 1e-9);

        // max_bond の上限が epsilon より優先（極小 epsilon でも 2 本に cap）
        let r_cap = svd_bond(&m, &[i.clone()], &Truncation { epsilon: 1e-14, max_bond: Some(2) });
        assert_eq!(r_cap.s.len(), 2);
    }

    #[test]
    fn relabel_changes_contraction_partner() {
        let i = new_index(2);
        let i2 = new_index(2);
        let t = from_fn(&[i.clone()], |x| (x[0] + 1) as f64);
        let t2 = relabel(&t, &i, &i2);
        // t2 has i2 (not i), so outer-product with an i-indexed vector gives 4 elements
        let other = from_fn(&[i.clone()], |_| 1.0);
        let outer = tensor4all_core::outer_product(&t2, &other).expect("outer_product failed");
        assert_eq!(outer.dims().iter().product::<usize>(), 4);
        // also verify t2 contracts normally with an i2-indexed vector (inner product)
        let partner = from_fn(&[i2.clone()], |_| 1.0);
        let inner = contract(&t2, &partner);
        assert_abs_diff_eq!(scalar(&inner), 3.0); // 1+2 = 3
    }
}
