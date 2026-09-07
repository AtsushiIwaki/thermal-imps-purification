use thermal_imps_purification::purified_mps::infinite_temperature;
use thermal_imps_purification::itebd::{imaginary_time_step, free_energy_from_log_norm};
use thermal_imps_purification::canonicalize::canonicalize;
use thermal_imps_purification::observable::{energy_density, magnetization_z};
use thermal_imps_purification::variance::specific_heat;
use thermal_imps_purification::exact::{xy_energy_density, xy_magnetization_z, xy_specific_heat, xy_free_energy_density};
use thermal_imps_purification::model::Xy;
use thermal_imps_purification::tensor::Truncation;

#[test]
fn xy_matches_exact_at_representative_point() {
    let xy = Xy { gamma: 0.5, h: 0.5 };
    let ham = xy.local();
    let dtau = 0.01;
    let trunc = Truncation { epsilon: 1e-12, max_bond: Some(48) };
    let beta = 1.5;
    let steps = (beta / (2.0 * dtau)) as usize;
    let mut s = infinite_temperature(2);
    let mut accum = 0.0;
    for _ in 0..steps {
        let info = imaginary_time_step(&mut s, &ham, dtau, &trunc);
        accum += info.log_norm;
        accum += canonicalize(&mut s);
    }
    let u = energy_density(&s, &ham);
    let mz = magnetization_z(&s);
    let c = specific_heat(&s, &ham, beta).unwrap();
    let f = free_energy_from_log_norm(accum / 2.0, beta, 2);
    let u_ex = xy_energy_density(0.5, 0.5, beta, 4000);
    let mz_ex = xy_magnetization_z(0.5, 0.5, beta, 4000);
    let c_ex = xy_specific_heat(0.5, 0.5, beta, 4000);
    let f_ex = xy_free_energy_density(0.5, 0.5, beta, 4000);
    assert!((u - u_ex).abs() < 5e-3, "u {u} vs {u_ex}");
    assert!((mz - mz_ex).abs() < 5e-3, "mz {mz} vs {mz_ex}");
    assert!((c - c_ex).abs() < 2e-2, "c {c} vs {c_ex}");
    assert!((f - f_ex).abs() < 5e-3, "f {f} vs {f_ex}");
}
