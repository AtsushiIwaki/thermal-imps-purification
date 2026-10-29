//! Protect the finite clustered matrix that failed in the deployed Faer backend.
use thermal_imps_purification::tensor::{self, Truncation};

#[test]
fn clustered_finite_svd_reconstructs_without_iteration_failure() {
    let a = f64::from_bits(0x3fed_b6e9_5cf8_3a86);
    let b = f64::from_bits(0x3fed_b6e9_5cf8_3a87);
    let i = tensor::new_index(3);
    let j = tensor::new_index(3);
    let input = tensor::from_fn(&[i.clone(), j.clone()], |x| match (x[0], x[1]) {
        (0, 0) => 1.,
        (1, 1) => a,
        (1, 2) => f64::EPSILON,
        (2, 2) => b,
        _ => 0.,
    });
    let r = tensor::svd_bond(
        &input,
        &[i],
        &Truncation {
            epsilon: 1e-13,
            max_bond: Some(256),
        },
    );
    // Independent scalar reconstruction oracle, confined to this 3x3 test.
    let u = r.u.to_vec::<f64>().unwrap();
    let v = r.v.to_vec::<f64>().unwrap();
    let input = input.to_vec::<f64>().unwrap();
    for j in 0..3 {
        for i in 0..3 {
            let value: f64 = (0..3).map(|k| u[i + 3 * k] * r.s[k] * v[j + 3 * k]).sum();
            assert!(
                (value - input[i + 3 * j]).abs() < 1e-14,
                "SVD reconstruction error"
            );
        }
    }
    assert_eq!(r.s.len(), 3);
    for (x, y) in r.s.iter().zip([1., b, a]) {
        assert!((x - y).abs() < 1e-14);
    }
}
