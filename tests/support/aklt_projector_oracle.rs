use nalgebra::DMatrix;

pub fn projector_from_cg() -> DMatrix<f64> {
    let mut v = DMatrix::<f64>::zeros(9, 5);
    let q2 = 1.0 / 2.0_f64.sqrt();
    let q6 = 1.0 / 6.0_f64.sqrt();
    v[(0, 0)] = 1.0;
    v[(1, 1)] = q2;
    v[(3, 1)] = q2;
    v[(2, 2)] = q6;
    v[(4, 2)] = 2.0 * q6;
    v[(6, 2)] = q6;
    v[(5, 3)] = q2;
    v[(7, 3)] = q2;
    v[(8, 4)] = 1.0;
    &v * v.transpose()
}

pub fn periodic_four_site_h(h: &DMatrix<f64>) -> DMatrix<f64> {
    let mut full = DMatrix::<f64>::zeros(81, 81);
    for col in 0..81_usize {
        let digits = [col % 3, (col / 3) % 3, (col / 9) % 3, (col / 27) % 3];
        for i in 0..4 {
            let j = (i + 1) % 4;
            let input = digits[i] + 3 * digits[j];
            for a in 0..3 {
                for b in 0..3 {
                    let mut out = digits;
                    out[i] = a;
                    out[j] = b;
                    let row = out[0] + 3 * out[1] + 9 * out[2] + 27 * out[3];
                    full[(row, col)] += h[(a + 3 * b, input)];
                }
            }
        }
    }
    full
}
