use thermal_imps_purification::itebd_complex::{ComplexPurifiedMps, ComplexSite};
use thermal_imps_purification::purified_mps::{infinite_temperature, PurifiedMps};
use thermal_imps_purification::tensor::Tensor;
use num_complex::Complex64;

pub fn alternating_product_state() -> PurifiedMps {
    let mut state = infinite_temperature(2);
    for (site, probabilities) in [
        (&mut state.a, [0.2_f64, 0.8]),
        (&mut state.b, [0.7_f64, 0.3]),
    ] {
        site.gamma = Tensor::from_dense(
            site.gamma.indices.clone(),
            vec![probabilities[0].sqrt(), 0.0, 0.0, probabilities[1].sqrt()],
        )
        .unwrap();
    }
    state
}

pub fn complex_ancilla_rotated(state: &PurifiedMps) -> ComplexPurifiedMps {
    let promote = |site: &thermal_imps_purification::purified_mps::Site| {
        let values = site.gamma.to_vec::<f64>().unwrap();
        // U = [[1,i],[i,1]] / sqrt(2) is unitary. Apply it on the ancilla.
        let values = vec![
            Complex64::from(values[0]),
            Complex64::new(0.0, values[3]),
            Complex64::new(0.0, values[0]),
            Complex64::from(values[3]),
        ]
        .into_iter()
        .map(|z| z / 2.0_f64.sqrt())
        .collect();
        ComplexSite {
            gamma: Tensor::from_dense(site.gamma.indices.clone(), values).unwrap(),
            left: site.left.clone(),
            right: site.right.clone(),
            phys: site.phys.clone(),
            anc: site.anc.clone(),
        }
    };
    ComplexPurifiedMps {
        a: promote(&state.a),
        b: promote(&state.b),
        lambda_ab: state.lambda_ab.clone(),
        lambda_ba: state.lambda_ba.clone(),
        lambda_bond_ab: state.lambda_bond_ab.clone(),
        lambda_bond_ba: state.lambda_bond_ba.clone(),
    }
}
