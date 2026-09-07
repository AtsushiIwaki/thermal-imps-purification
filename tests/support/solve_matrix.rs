use thermal_imps_purification::config::TrotterOrder;
use thermal_imps_purification::itebd::free_energy_from_log_norm;
use thermal_imps_purification::itebd_complex::{
    canonicalize_complex, energy_density_complex, imaginary_time_step_complex,
    imaginary_time_step_second_order_complex, local_expectation_complex, specific_heat_complex,
    ComplexLocalHamiltonian, ComplexPurifiedMps,
};
use thermal_imps_purification::model::{pauli_x, Tfim};
use thermal_imps_purification::tensor::Truncation;
use nalgebra::DMatrix;
use num_complex::Complex64;

pub fn phase_config() -> serde_json::Value {
    let model = serde_json::json!({
        "type": "matrix", "version": 1, "local_dim": 2,
        "basis_order": "first_site_fastest", "label": "phase TFIM",
        "two_site_h": {
            "real": [[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],
            "imag": [[0.,0.35,0.35,0.],[-0.35,0.,0.,0.35],[-0.35,0.,0.,0.35],[0.,-0.35,-0.35,0.]]
        },
        "site_energy": {
            "real": [[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],
            "imag": [[0.,0.7,0.,0.],[-0.7,0.,0.,0.],[0.,0.,0.,0.7],[0.,0.,-0.7,0.]]
        },
        "observable": {"name": "sigma_y", "matrix": {
            "real": [[0.,0.],[0.,0.]], "imag": [[0.,-1.],[1.,0.]]
        }}
    });
    serde_json::json!({
        "model": model,
        "evolution": {
            "dtau": 0.05, "beta_max": 0.2, "record_every_beta": 0.2,
            "trotter_order": 2
        },
        "truncation": {"epsilon": 1e-12, "max_bond": 32},
        "run": {"canonicalize_every": 1},
        "output": {"path": "unused.json", "include_exact": false}
    })
}

#[allow(dead_code)]
pub fn real_tfim_config() -> serde_json::Value {
    let hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    serde_json::json!({
        "model": {
            "type": "matrix", "version": 1, "local_dim": 2,
            "basis_order": "first_site_fastest", "label": "exact real TFIM",
            "two_site_h": {"real": real_rows(&hamiltonian.two_site_h)},
            "site_energy": {"real": real_rows(&hamiltonian.site_energy)},
            "observable": {"name": "sigma_x", "matrix": {"real": real_rows(&pauli_x())}}
        },
        "evolution": {
            "dtau": 0.05, "beta_max": 0.2, "record_every_beta": 0.2,
            "trotter_order": 2
        },
        "truncation": {"epsilon": 1e-12, "max_bond": 32},
        "run": {"canonicalize_every": 1},
        "output": {"path": "unused.json", "include_exact": false}
    })
}

#[allow(dead_code)]
fn real_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}

#[allow(dead_code)]
fn phase_rotated_tfim() -> ComplexLocalHamiltonian {
    let real = Tfim { j: 1.0, g: 0.7 }.local();
    let phase = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let rotate = |matrix: &DMatrix<f64>| {
        DMatrix::from_fn(4, 4, |row, column| {
            let (s1, s2) = (row % 2, row / 2);
            let (t1, t2) = (column % 2, column / 2);
            phase[s1]
                * phase[s2]
                * Complex64::new(matrix[(row, column)], 0.0)
                * phase[t1].conj()
                * phase[t2].conj()
        })
    };
    ComplexLocalHamiltonian::try_new(rotate(&real.two_site_h), rotate(&real.site_energy)).unwrap()
}

#[allow(dead_code)]
fn phase_rotated_pauli_x() -> DMatrix<Complex64> {
    let real = pauli_x();
    let phase = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    DMatrix::from_fn(2, 2, |row, column| {
        phase[row] * Complex64::new(real[(row, column)], 0.0) * phase[column].conj()
    })
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OracleRecord {
    pub u: f64,
    pub c: f64,
    pub f: f64,
    pub local: f64,
    pub max_bond: usize,
}

#[allow(dead_code)]
pub fn phase_oracle(order: TrotterOrder) -> OracleRecord {
    phase_oracle_window(order, 0.05, 0.2, 1e-12, 32)
}

#[allow(dead_code)]
pub fn phase_oracle_window(
    order: TrotterOrder,
    dtau: f64,
    beta: f64,
    epsilon: f64,
    cap: usize,
) -> OracleRecord {
    let hamiltonian = phase_rotated_tfim();
    let observable = phase_rotated_pauli_x();
    let mut state = ComplexPurifiedMps::infinite_temperature(2).unwrap();
    let truncation = Truncation {
        epsilon,
        max_bond: Some(cap),
    };
    let steps = (beta / (2.0 * dtau)).round() as usize;
    let mut accumulated_log_norm = 0.0;
    let mut max_bond = 1;
    for _ in 0..steps {
        let info = match order {
            TrotterOrder::First => {
                imaginary_time_step_complex(&mut state, &hamiltonian, dtau, &truncation).unwrap()
            }
            TrotterOrder::Second => imaginary_time_step_second_order_complex(
                &mut state,
                &hamiltonian,
                dtau,
                &truncation,
            )
            .unwrap(),
        };
        accumulated_log_norm += info.log_norm;
        accumulated_log_norm += canonicalize_complex(&mut state, 1e-12).unwrap();
        max_bond = info.max_bond;
    }
    OracleRecord {
        u: energy_density_complex(&state, &hamiltonian).unwrap(),
        c: specific_heat_complex(&state, &hamiltonian, beta).unwrap(),
        f: free_energy_from_log_norm(accumulated_log_norm / 2.0, beta, 2),
        local: local_expectation_complex(&state, &observable, 1e-12).unwrap(),
        max_bond,
    }
}

#[allow(dead_code)]
pub fn real_oracle(order: TrotterOrder) -> OracleRecord {
    use thermal_imps_purification::canonicalize::canonicalize;
    use thermal_imps_purification::itebd::{imaginary_time_step, imaginary_time_step_second_order};
    use thermal_imps_purification::observable::{energy_density, magnetization};
    use thermal_imps_purification::purified_mps::infinite_temperature;
    use thermal_imps_purification::variance::specific_heat;

    let hamiltonian = Tfim { j: 1.0, g: 0.7 }.local();
    let observable = pauli_x();
    let mut state = infinite_temperature(2);
    let truncation = Truncation {
        epsilon: 1e-12,
        max_bond: Some(32),
    };
    let dtau = 0.05;
    let beta = 0.2;
    let mut accumulated_log_norm = 0.0;
    let mut max_bond = 1;
    for _ in 0..2 {
        let info = match order {
            TrotterOrder::First => imaginary_time_step(&mut state, &hamiltonian, dtau, &truncation),
            TrotterOrder::Second => {
                imaginary_time_step_second_order(&mut state, &hamiltonian, dtau, &truncation)
            }
        };
        accumulated_log_norm += info.log_norm;
        accumulated_log_norm += canonicalize(&mut state);
        max_bond = info.max_bond;
    }
    OracleRecord {
        u: energy_density(&state, &hamiltonian),
        c: specific_heat(&state, &hamiltonian, beta).unwrap(),
        f: free_energy_from_log_norm(accumulated_log_norm / 2.0, beta, 2),
        local: magnetization(&state, &observable),
        max_bond,
    }
}

/// Serialize literal input through the public schema; the independent oracle never uses this.
#[allow(dead_code)]
pub fn write_input(value: &serde_json::Value, path: &std::path::Path, format: &str) {
    let text = match format {
        "json" => serde_json::to_string_pretty(value).unwrap(),
        "toml" => {
            let cfg =
                thermal_imps_purification::config::RunConfig::from_json_str(&value.to_string()).unwrap();
            toml::to_string(&cfg).unwrap()
        }
        _ => panic!("unknown format {format}"),
    };
    std::fs::write(path, text).unwrap();
}

#[allow(dead_code)]
pub fn run_cli(path: &std::path::Path) {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_solve"))
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "config={} status={} stdout={} stderr={}",
        path.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
