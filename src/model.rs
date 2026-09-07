use nalgebra::{DMatrix, SymmetricEigen};

#[derive(Debug, Clone, Copy)]
pub struct Tfim { pub j: f64, pub g: f64 }

pub fn pauli_z() -> DMatrix<f64> { DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]) }
pub fn pauli_x() -> DMatrix<f64> { DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]) }
pub fn id2() -> DMatrix<f64> { DMatrix::identity(2, 2) }

/// 2サイト局所ハミルトニアン h = -J sz⊗sz - (g/2)(sx⊗I + I⊗sx)。flat = s1 + 2*s2。
pub fn two_site_h(m: &Tfim) -> DMatrix<f64> {
    let sz = pauli_z();
    let sx = pauli_x();
    let id = id2();
    let idx = |s1: usize, s2: usize| s1 + 2 * s2;
    let mut h = DMatrix::<f64>::zeros(4, 4);
    for s1 in 0..2 { for s2 in 0..2 { for t1 in 0..2 { for t2 in 0..2 {
        let r = idx(s1, s2);
        let c = idx(t1, t2);
        let zz = sz[(s1, t1)] * sz[(s2, t2)];
        let xi = sx[(s1, t1)] * id[(s2, t2)];
        let ix = id[(s1, t1)] * sx[(s2, t2)];
        h[(r, c)] += -m.j * zz - 0.5 * m.g * (xi + ix);
    }}}}
    h
}

#[derive(Debug, Clone)]
pub struct LocalHamiltonian {
    pub two_site_h: DMatrix<f64>,
    pub site_energy: DMatrix<f64>,
}

impl LocalHamiltonian {
    /// Local physical dimension `d`, derived from the `d²×d²` two-site matrix.
    pub fn dim(&self) -> usize {
        let n = self.two_site_h.nrows();
        let d = (n as f64).sqrt().round() as usize;
        debug_assert_eq!(d * d, n, "two_site_h must be d²×d² (got {n} rows)");
        d
    }
}

fn tfim_site_energy(m: &Tfim) -> DMatrix<f64> {
    let sz = pauli_z();
    let sx = pauli_x();
    let id = id2();
    let idx = |s1: usize, s2: usize| s1 + 2 * s2;
    let mut e = DMatrix::<f64>::zeros(4, 4);
    for s1 in 0..2 { for s2 in 0..2 { for t1 in 0..2 { for t2 in 0..2 {
        let r = idx(s1, s2);
        let c = idx(t1, t2);
        let zz = sz[(s1, t1)] * sz[(s2, t2)];
        let xi = sx[(s1, t1)] * id[(s2, t2)];
        e[(r, c)] += -m.j * zz - m.g * xi;
    }}}}
    e
}

impl Tfim {
    pub fn local(&self) -> LocalHamiltonian {
        LocalHamiltonian { two_site_h: two_site_h(self), site_energy: tfim_site_energy(self) }
    }
}

/// e^{-tau h}（h は実対称）を対称固有分解で計算。任意サイズ対応。
pub fn trotter_gate(h: &DMatrix<f64>, tau: f64) -> DMatrix<f64> {
    let eig = SymmetricEigen::new(h.clone());
    let expd = DMatrix::from_diagonal(&eig.eigenvalues.map(|l| (-tau * l).exp()));
    &eig.eigenvectors * expd * eig.eigenvectors.transpose()
}

#[derive(Debug, Clone, Copy)]
pub struct Xy { pub gamma: f64, pub h: f64 }

/// σ^y = i·yc, yc = [[0,-1],[1,0]]。σ^y⊗σ^y = -(yc⊗yc)（実値）。
fn yc() -> DMatrix<f64> { DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]) }

impl Xy {
    pub fn local(&self) -> LocalHamiltonian {
        let sx = pauli_x();
        let sz = pauli_z();
        let id = id2();
        let y = yc();
        let cx = 0.5 * (1.0 + self.gamma);
        let cy = 0.5 * (1.0 - self.gamma);
        let idx = |s1: usize, s2: usize| s1 + 2 * s2;
        let mut two = DMatrix::<f64>::zeros(4, 4);
        let mut site = DMatrix::<f64>::zeros(4, 4);
        for s1 in 0..2 { for s2 in 0..2 { for t1 in 0..2 { for t2 in 0..2 {
            let r = idx(s1, s2);
            let c = idx(t1, t2);
            let xx = sx[(s1, t1)] * sx[(s2, t2)];
            let yy = -(y[(s1, t1)] * y[(s2, t2)]);
            let zi = sz[(s1, t1)] * id[(s2, t2)];
            let iz = id[(s1, t1)] * sz[(s2, t2)];
            two[(r, c)] += -cx * xx - cy * yy - 0.5 * self.h * (zi + iz);
            site[(r, c)] += -cx * xx - cy * yy - self.h * zi;
        }}}}
        LocalHamiltonian { two_site_h: two, site_energy: site }
    }
}

// --- Spin-1 (d=3) operators. Basis |m⟩, m = +1,0,−1 → index 0,1,2. ---

/// S^z = diag(1, 0, −1).
pub fn sz1() -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
    ])
}

/// S^x = (1/√2)[[0,1,0],[1,0,1],[0,1,0]].
pub fn sx1() -> DMatrix<f64> {
    let a = std::f64::consts::FRAC_1_SQRT_2;
    DMatrix::from_row_slice(3, 3, &[
        0.0, a, 0.0,
        a, 0.0, a,
        0.0, a, 0.0,
    ])
}

/// Real antisymmetric "yc" with S^y = i·sy1c = (1/√2)[[0,−i,0],[i,0,−i],[0,i,0]].
/// Hence S^y⊗S^y = −(sy1c⊗sy1c) is real.
pub fn sy1c() -> DMatrix<f64> {
    let a = std::f64::consts::FRAC_1_SQRT_2;
    DMatrix::from_row_slice(3, 3, &[
        0.0, -a, 0.0,
        a, 0.0, -a,
        0.0, a, 0.0,
    ])
}

/// Two-site S·S = S^x⊗S^x + S^y⊗S^y + S^z⊗S^z as a 9×9 real symmetric matrix.
/// flat index = s1 + 3*s2 (row=out, col=in convention shared with the tensor builders).
fn spin1_dot() -> DMatrix<f64> {
    let sx = sx1();
    let sz = sz1();
    let yc = sy1c();
    let idx = |s1: usize, s2: usize| s1 + 3 * s2;
    let mut ss = DMatrix::<f64>::zeros(9, 9);
    for s1 in 0..3 { for s2 in 0..3 { for t1 in 0..3 { for t2 in 0..3 {
        let r = idx(s1, s2);
        let c = idx(t1, t2);
        let xx = sx[(s1, t1)] * sx[(s2, t2)];
        let yy = -(yc[(s1, t1)] * yc[(s2, t2)]); // = (S^y⊗S^y)
        let zz = sz[(s1, t1)] * sz[(s2, t2)];
        ss[(r, c)] += xx + yy + zz;
    }}}}
    ss
}

/// Spin-1 AKLT Hamiltonian with coefficient-one projectors onto total spin two.
#[derive(Debug, Clone, Copy)]
pub struct AkltProjector;

impl AkltProjector {
    pub fn local(&self) -> LocalHamiltonian {
        let x = spin1_dot();
        let p = (&x * &x + 3.0 * &x + 2.0 * DMatrix::<f64>::identity(9, 9)) / 6.0;
        LocalHamiltonian { site_energy: p.clone(), two_site_h: p }
    }
}

/// Spin-1 bilinear-biquadratic chain: H = Σ_i [ j1 (S_i·S_{i+1}) + j2 (S_i·S_{i+1})² ].
#[derive(Debug, Clone, Copy)]
pub struct BilinearBiquadratic { pub j1: f64, pub j2: f64 }

impl BilinearBiquadratic {
    /// AKLT point (j1, j2) = (1, 1/3).
    pub fn aklt() -> Self { Self { j1: 1.0, j2: 1.0 / 3.0 } }
    /// Spin-1 Heisenberg (j1, j2) = (1, 0).
    pub fn heisenberg() -> Self { Self { j1: 1.0, j2: 0.0 } }

    pub fn local(&self) -> LocalHamiltonian {
        let ss = spin1_dot();
        let ss2 = &ss * &ss;
        let two = self.j1 * &ss + self.j2 * &ss2;
        LocalHamiltonian { site_energy: two.clone(), two_site_h: two }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn h_is_symmetric_and_correct() {
        let m = Tfim { j: 1.0, g: 0.5 };
        let h = two_site_h(&m);
        // 対称性
        for a in 0..4 { for b in 0..4 { assert_abs_diff_eq!(h[(a, b)], h[(b, a)], epsilon = 1e-12); } }
        // 対角は -J*sz*sz 部分: 状態 (s1,s2), sz=+1 for s=0, -1 for s=1
        // flat = s1 + 2*s2
        // (0,0): sz1*sz2=+1 -> -J = -1.0
        assert_abs_diff_eq!(h[(0, 0)], -1.0, epsilon = 1e-12);
        // (1,0): s1=1->sz=-1, s2=0->sz=+1 -> sz*sz=-1 -> +J = +1.0
        assert_abs_diff_eq!(h[(1, 1)], 1.0, epsilon = 1e-12);
        // 横磁場 off-diagonal: -g/2 で s1 をフリップ (0,0)<->(1,0): flat 0<->1
        assert_abs_diff_eq!(h[(0, 1)], -0.25, epsilon = 1e-12);
    }

    #[test]
    fn gate_at_tau_zero_is_identity() {
        let m = Tfim { j: 1.0, g: 0.5 };
        let h = two_site_h(&m);
        let g0 = trotter_gate(&h, 0.0);
        for a in 0..4 { for b in 0..4 {
            assert_abs_diff_eq!(g0[(a, b)], if a == b { 1.0 } else { 0.0 }, epsilon = 1e-12);
        }}
    }

    #[test]
    fn gate_small_tau_first_order() {
        let m = Tfim { j: 1.0, g: 0.5 };
        let h = two_site_h(&m);
        let tau = 1e-4;
        let g = trotter_gate(&h, tau);
        // e^{-tau h} ≈ I - tau h
        for a in 0..4 { for b in 0..4 {
            let expected = (if a == b { 1.0 } else { 0.0 }) - tau * h[(a, b)];
            assert_abs_diff_eq!(g[(a, b)], expected, epsilon = 1e-7);
        }}
    }

    #[test]
    fn xy_local_reduces_to_tfim_at_gamma_one() {
        // γ=1 で XY の two_site_h は TFIM(J=1, x-coupling) と同型のはず（軸は異なるが固有値=エネルギーは一致）。
        // ここでは XY の two_site_h が対称・正しい σx σx 係数を持つことを確認。
        let xy = Xy { gamma: 1.0, h: 0.5 };
        let lh = xy.local();
        let hh = lh.two_site_h;
        for a in 0..4 { for b in 0..4 { assert_abs_diff_eq!(hh[(a, b)], hh[(b, a)], epsilon = 1e-12); } }
        // γ=1: 係数 (1+γ)/2 = 1 の σxσx、(1-γ)/2 = 0 の σyσy。
        // σxσx は |s1 s2> を両方フリップ: (0,0)<->(1,1) = flat 0<->3、係数 -1。
        assert_abs_diff_eq!(hh[(0, 3)], -1.0, epsilon = 1e-12);
        // 横磁場 -h/2 (σz⊗I + I⊗σz): 対角 (0,0) 両 up(sz=+1) -> -h/2*(1+1) = -h = -0.5
        assert_abs_diff_eq!(hh[(0, 0)], -0.5, epsilon = 1e-12);
    }

    #[test]
    fn xy_sigma_yy_sign_and_symmetry() {
        // γ=0（XX）: (1+γ)/2 = (1-γ)/2 = 1/2。σxσx と σyσy が両方 1/2 係数。
        // σxσx[0][3] = -1 を 1/2 倍 = -0.5。σyσy[0][3] = +1 を ... 全体で h[0][3] を確認。
        // σx⊗σx: (0,0)->(1,1) は +1（両フリップ）。σy⊗σy: (0,0)->(1,1) は -1。
        // h[0][3] = -0.5*( σxσx[0][3] ) - 0.5*( σyσy[0][3] ) = -0.5*(1) -0.5*(-1) = 0。
        let xy = Xy { gamma: 0.0, h: 0.0 };
        let hh = xy.local().two_site_h;
        assert_abs_diff_eq!(hh[(0, 3)], 0.0, epsilon = 1e-12);
        // (0,1)<->(1,0): σxσx と σyσy がともに寄与。σxσx[(1,0)->(0,1)]=+1, σyσy[(1,0)->(0,1)]=+1。
        // flat (1,0)=1, (0,1)=2。h[1][2] = -0.5*(1) -0.5*(1) = -1.0。
        assert_abs_diff_eq!(hh[(1, 2)], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn spin1_commutator_and_casimir() {
        // Sy = i·sy1c ⇒ [Sx,Sy] = i Sz reduces to Sx·sy1c − sy1c·Sx = Sz (real).
        let sx = sx1(); let yc = sy1c(); let sz = sz1();
        let comm = &sx * &yc - &yc * &sx;
        for a in 0..3 { for b in 0..3 { assert_abs_diff_eq!(comm[(a, b)], sz[(a, b)], epsilon = 1e-12); } }
        // S² = Sx² + Sy² + Sz² = 2 I, with Sy² = −sy1c².
        let s2 = &sx * &sx - &yc * &yc + &sz * &sz;
        for a in 0..3 { for b in 0..3 {
            assert_abs_diff_eq!(s2[(a, b)], if a == b { 2.0 } else { 0.0 }, epsilon = 1e-12);
        }}
    }

    #[test]
    fn aklt_two_site_spectrum() {
        use nalgebra::SymmetricEigen;
        let h = BilinearBiquadratic::aklt().local().two_site_h;
        // Real symmetric 9×9.
        for a in 0..9 { for b in 0..9 { assert_abs_diff_eq!(h[(a, b)], h[(b, a)], epsilon = 1e-12); } }
        let mut ev: Vec<f64> = SymmetricEigen::new(h).eigenvalues.iter().copied().collect();
        ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
        // h = SS + (1/3)SS²; eigenvalues −2/3 (×4: S_tot=0,1) and +4/3 (×5: S_tot=2).
        for k in 0..4 { assert_abs_diff_eq!(ev[k], -2.0 / 3.0, epsilon = 1e-10); }
        for k in 4..9 { assert_abs_diff_eq!(ev[k], 4.0 / 3.0, epsilon = 1e-10); }
    }

    #[test]
    fn heisenberg_two_site_spectrum() {
        use nalgebra::SymmetricEigen;
        // j2=0 ⇒ h = S·S, eigenvalues −2 (×1), −1 (×3), +1 (×5).
        let h = BilinearBiquadratic::heisenberg().local().two_site_h;
        let mut ev: Vec<f64> = SymmetricEigen::new(h).eigenvalues.iter().copied().collect();
        ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_abs_diff_eq!(ev[0], -2.0, epsilon = 1e-10);
        for k in 1..4 { assert_abs_diff_eq!(ev[k], -1.0, epsilon = 1e-10); }
        for k in 4..9 { assert_abs_diff_eq!(ev[k], 1.0, epsilon = 1e-10); }
    }

    #[test]
    fn aklt_site_energy_equals_two_site_h() {
        let lh = BilinearBiquadratic::aklt().local();
        assert_eq!(lh.site_energy, lh.two_site_h);
    }

    #[test]
    fn local_hamiltonian_dim() {
        assert_eq!(Tfim { j: 1.0, g: 0.5 }.local().dim(), 2);
        assert_eq!(Xy { gamma: 0.5, h: 0.5 }.local().dim(), 2);
        assert_eq!(BilinearBiquadratic::aklt().local().dim(), 3);
        assert_eq!(BilinearBiquadratic::heisenberg().local().dim(), 3);
    }
}
