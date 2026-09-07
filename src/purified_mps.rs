use crate::tensor::{Idx, Tensor, new_index, from_fn};

pub struct Site {
    pub gamma: Tensor,
    pub left: Idx,
    pub phys: Idx,
    pub anc: Idx,
    pub right: Idx,
}

pub struct PurifiedMps {
    pub a: Site,
    pub b: Site,
    pub lambda_ab: Vec<f64>,
    pub lambda_bond_ab: Idx,
    pub lambda_ba: Vec<f64>,
    pub lambda_bond_ba: Idx,
}

pub fn infinite_temperature(d: usize) -> PurifiedMps {
    let inv_sqrt = 1.0 / (d as f64).sqrt();
    // 2 つの共有ボンド（A-B 間 = bond_ab, B-A 間 = bond_ba）
    let bond_ab = new_index(1);
    let bond_ba = new_index(1);
    let make = |left: &Idx, right: &Idx| -> Site {
        let phys = new_index(d);
        let anc = new_index(d);
        let gamma = from_fn(
            &[left.clone(), phys.clone(), anc.clone(), right.clone()],
            |ix| if ix[1] == ix[2] { inv_sqrt } else { 0.0 },
        );
        Site { gamma, left: left.clone(), phys, anc, right: right.clone() }
    };
    // A: left = bond_ba, right = bond_ab ; B: left = bond_ab, right = bond_ba
    let a = make(&bond_ba, &bond_ab);
    let b = make(&bond_ab, &bond_ba);
    PurifiedMps {
        a, b,
        lambda_ab: vec![1.0], lambda_bond_ab: bond_ab,
        lambda_ba: vec![1.0], lambda_bond_ba: bond_ba,
    }
}

/// テスト補助: bond-dim 1 の Site の norm 用に gamma の bra コピーを返す（全脚同 Id のまま自己縮約）。
/// bond-dim 1 専用: bond 次元が 1 のとき、gamma をそのままコピーすると全脚が同 Id を共有し、
/// contract で自己縮約が norm^2 を与える。bond-dim > 1 では脚が衝突するため使用不可。
#[cfg(test)]
pub fn self_overlap_bra_dim1(site: &Site) -> Tensor {
    // 実数なので conj 不要。全 index 共有のまま自己縮約すると全脚和になる。
    site.gamma.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn infinite_temperature_structure() {
        let s = infinite_temperature(2);
        assert_eq!(dim(&s.a.left), 1);
        assert_eq!(dim(&s.a.phys), 2);
        assert_eq!(dim(&s.a.anc), 2);
        assert_eq!(dim(&s.a.right), 1);
        assert_eq!(s.lambda_ab, vec![1.0]);
        assert_eq!(s.lambda_ba, vec![1.0]);
    }

    #[test]
    fn infinite_temperature_is_normalized() {
        // <psi|psi> over one site = sum_{p,a} |gamma|^2 = sum (1/2) = 1
        let s = infinite_temperature(2);
        // gamma と自身を全脚縮約（bond=1 なので単純）
        let bra = self_overlap_bra_dim1(&s.a); // 下の実装で提供
        let n = crate::tensor::contract(&s.a.gamma, &bra);
        assert_abs_diff_eq!(scalar(&n), 1.0, epsilon = 1e-12);
    }
}
