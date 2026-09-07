//! Retained release evidence: checkpoint persistence, physical RDMs and TFIM refinement.
#[path = "support/itebd_evidence.rs"]
mod evidence;
#[path = "support/itebd_checkpoint_rdm.rs"]
mod support;
use thermal_imps_purification::exact::{exact_energy_density, exact_magnetization_x, free_energy_density};
use thermal_imps_purification::itebd::free_energy_from_log_norm;
use thermal_imps_purification::itebd_auto::{
    energy_density_auto, local_expectation_auto, ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::itebd_checkpoint::*;
use thermal_imps_purification::itebd_rdm::*;
use thermal_imps_purification::itebd_state_view::ItebdStateRef;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde_json::{json, Value};

fn physical(
    state: &ItebdState,
    start: RdmParity,
    length: usize,
    options: &RdmOptions,
) -> Result<(DMatrix<Complex64>, RdmReport), RdmError> {
    Ok(
        match reduced_density_matrix_auto(state, start, length, options)? {
            AutoRdmResult::Real(r) => (r.density_matrix.map(Complex64::from), r.report),
            AutoRdmResult::Complex(r) => (r.density_matrix, r.report),
        },
    )
}
fn diagnostic(r: &RdmReport) -> Value {
    let env = |d: &RdmEnvironmentDiagnostics| {
        json!({"eigenvalue":d.eigenvalue,
        "iterations":d.iterations,"residual":d.relative_residual})
    };
    json!({"start":format!("{:?}",r.start),"length":r.length,"left":env(&r.left),"right":env(&r.right),
        "raw_trace":[r.raw_trace.re,r.raw_trace.im],"trace_residual":r.trace_residual,
        "hermiticity_residual":r.hermiticity_residual,"minimum_eigenvalue":r.minimum_eigenvalue,
        "largest_intermediate_elements":r.largest_intermediate_elements})
}
fn run_row(
    complex: bool,
    dtau: f64,
    epsilon: f64,
    cap: usize,
    tolerance: f64,
) -> Result<Value, Box<dyn std::error::Error>> {
    let h = if complex {
        support::phase_tfim()
    } else {
        ItebdHamiltonian::Real(thermal_imps_purification::model::Tfim { j: 1.0, g: 0.7 }.local())
    };
    let mut metadata = support::real_tfim_metadata();
    metadata.dtau = dtau;
    metadata.truncation.epsilon = epsilon;
    metadata.truncation.max_bond = Some(cap);
    metadata.canonicalize_every = 1;
    let options = RdmOptions {
        fixed_point_tolerance: tolerance,
        ..RdmOptions::default()
    };
    let mut state = ItebdState::infinite_temperature(&h)?;
    let mut progress = ItebdCheckpointProgress {
        beta: 0.0,
        completed_steps: 0,
        accumulated_log_norm: 0.0,
        last_step: None,
    };
    let count = (1.0 / (2.0 * dtau)).round() as u64;
    let split = count / 2;
    let mut bonds = [1, 1];
    for _ in 0..split {
        support::advance(&mut state, &h, &metadata, &mut progress, 1)?;
        let v = ItebdStateRef::from(&state);
        bonds[0] = bonds[0].max(v.lambda_ab().len());
        bonds[1] = bonds[1].max(v.lambda_ba().len());
    }
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("continuation.h5");
    let mut writer = ItebdTrajectoryWriter::create(&path, metadata.clone(), &h)?;
    writer.append((&state).into(), &progress)?;
    writer.finish()?;
    let mut loaded = load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default())?;
    let roundtrip_bits =
        support::snapshot(&state, &progress) == support::snapshot(&loaded.state, &loaded.progress);
    let mut roundtrip_rdm = 0.0_f64;
    for start in [RdmParity::A, RdmParity::B] {
        for length in 1..=2 {
            roundtrip_rdm = roundtrip_rdm.max(
                (physical(&state, start, length, &options)?.0
                    - physical(&loaded.state, start, length, &options)?.0)
                    .norm(),
            );
        }
    }
    for _ in split..count {
        support::advance(&mut state, &h, &metadata, &mut progress, 1)?;
        support::advance(
            &mut loaded.state,
            &loaded.hamiltonian,
            &loaded.metadata,
            &mut loaded.progress,
            1,
        )?;
        let v = ItebdStateRef::from(&state);
        bonds[0] = bonds[0].max(v.lambda_ab().len());
        bonds[1] = bonds[1].max(v.lambda_ba().len());
    }
    let h_matrix = match &h {
        ItebdHamiltonian::Real(h) => h.two_site_h.map(Complex64::from),
        ItebdHamiltonian::Complex(h) => h.two_site_h().clone(),
    };
    let mut op = DMatrix::from_row_slice(2, 2, &[0., 1., 1., 0.]).map(Complex64::from);
    if complex {
        let u = DMatrix::from_diagonal(&DVector::from_vec(vec![
            Complex64::from(1.),
            Complex64::new(0., 1.),
        ]));
        op = &u * op * u.adjoint();
    }
    let mut diagnostics = Vec::new();
    let mut e_rdm = 0.;
    let mut m_rdm = 0.;
    let mut continuation_rdm = 0.0_f64;
    let mut imaginary = 0.0_f64;
    for start in [RdmParity::A, RdmParity::B] {
        for length in 1..=2 {
            let (rho, report) = physical(&state, start, length, &options)?;
            continuation_rdm = continuation_rdm
                .max((&rho - physical(&loaded.state, start, length, &options)?.0).norm());
            let estimate = (&rho * if length == 1 { &op } else { &h_matrix }).trace();
            imaginary = imaginary.max(estimate.im.abs());
            if length == 1 {
                m_rdm += 0.5 * estimate.re;
            } else {
                e_rdm += 0.5 * estimate.re;
            }
            evidence::finite(
                "RDM diagnostics",
                &[
                    report.left.eigenvalue,
                    report.right.eigenvalue,
                    report.left.relative_residual,
                    report.right.relative_residual,
                    report.raw_trace.re,
                    report.raw_trace.im,
                    report.trace_residual,
                    report.hermiticity_residual,
                    report.minimum_eigenvalue,
                ],
            )?;
            diagnostics.push(diagnostic(&report));
        }
    }
    let direct_e = energy_density_auto(&state, &h)?;
    let direct_m = local_expectation_auto(&state, &h, &op, 1e-12)?;
    let exact_e = exact_energy_density(1., 0.7, 1., 16_384);
    let exact_m = exact_magnetization_x(1., 0.7, 1., 16_384);
    let v = ItebdStateRef::from(&state);
    let w = ItebdStateRef::from(&loaded.state);
    if v.lambda_ab().len() != w.lambda_ab().len() || v.lambda_ba().len() != w.lambda_ba().len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "continued checkpoint Schmidt spectrum lengths differ",
        )
        .into());
    }
    let spectrum_residual = v
        .lambda_ab()
        .iter()
        .zip(w.lambda_ab())
        .chain(v.lambda_ba().iter().zip(w.lambda_ba()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let free = |p: &ItebdCheckpointProgress| {
        free_energy_from_log_norm(p.accumulated_log_norm / 2., p.beta, 2)
    };
    evidence::finite(
        "scientific observables and continuation",
        &[
            progress.beta,
            roundtrip_rdm,
            progress.accumulated_log_norm,
            loaded.progress.accumulated_log_norm,
            spectrum_residual,
            direct_e,
            energy_density_auto(&loaded.state, &loaded.hamiltonian)?,
            direct_m,
            e_rdm,
            m_rdm,
            exact_e,
            exact_m,
            free(&progress),
            free(&loaded.progress),
            continuation_rdm,
            imaginary,
        ],
    )?;
    if imaginary > options.trace_tolerance {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "RDM observable imaginary residual {imaginary:e} exceeds {}",
                options.trace_tolerance
            ),
        )
        .into());
    }
    Ok(
        json!({"status":"valid","beta":progress.beta,"max_bonds_ab_ba":bonds,
        "checkpoint_roundtrip":{"component_bits_equal":roundtrip_bits,"rdm_frobenius":roundtrip_rdm},
        "continuation":{"log_norm_abs":(progress.accumulated_log_norm-loaded.progress.accumulated_log_norm).abs(),
        "schmidt_max_abs":spectrum_residual,"energy_abs":(direct_e-energy_density_auto(&loaded.state,&loaded.hamiltonian)?).abs(),
        "free_energy_abs":(free(&progress)-free(&loaded.progress)).abs(),"rdm_frobenius":continuation_rdm},
        "rdms":diagnostics,"observable_imaginary_max":imaginary,
        "energy":{"direct":direct_e,"rdm":e_rdm,"exact":exact_e,"direct_error":(direct_e-exact_e).abs(),
        "rdm_error":(e_rdm-exact_e).abs(),"disagreement":(direct_e-e_rdm).abs()},
        "local_observable":{"direct":direct_m,"rdm":m_rdm,"exact":exact_m,"direct_error":(direct_m-exact_m).abs(),
        "rdm_error":(m_rdm-exact_m).abs(),"disagreement":(direct_m-m_rdm).abs()},
        "free_energy":free(&progress)}),
    )
}

#[test]
#[ignore]
fn checkpoint_rdm_refinement_matrix() {
    let destination = evidence::destination().expect("exclusive evidence destination");
    let mut rows = Vec::new();
    for complex in [false, true] {
        for dtau in [0.1, 0.05] {
            for epsilon in [1e-13, 1e-14] {
                for cap in [64, 128] {
                    for tolerance in [1e-10, 1e-12] {
                        let settings = json!({"backend":if complex {"complex"} else {"real"},"dtau":dtau,
                "epsilon":epsilon,"cap":cap,"fixed_point_tolerance":tolerance,
                "trotter_order":2,"canonicalize_every":1,"max_iterations":10_000,
                "hermiticity_tolerance":1e-10,"trace_tolerance":1e-10,"positivity_tolerance":1e-10,
                "max_output_elements":1_048_576,"max_intermediate_elements":16_777_216});
                        let mut row = match run_row(complex, dtau, epsilon, cap, tolerance) {
                            Ok(row) => row,
                            Err(error) => {
                                json!({"status":"failed","typed_error":format!("{error:?}")})
                            }
                        };
                        row["settings"] = settings;
                        eprintln!("science row {}: {}", rows.len() + 1, row["status"]);
                        rows.push(row);
                    }
                }
            }
        }
    }
    // Diagnostic derivative-step comparison, not a rigorous bound on the exact reference.
    let half_e = ((1.00005 * free_energy_density(1., 0.7, 1.00005, 16_384))
        - (0.99995 * free_energy_density(1., 0.7, 0.99995, 16_384)))
        / 0.0001;
    let half_m = -(free_energy_density(1., 0.70005, 1., 16_384)
        - free_energy_density(1., 0.69995, 1., 16_384))
        / 0.0001;
    let floors = [
        (half_e - exact_energy_density(1., 0.7, 1., 16_384)).abs(),
        (half_m - exact_magnetization_x(1., 0.7, 1., 16_384)).abs(),
    ];
    let mut failures = Vec::new();
    let mut refinements = Vec::new();
    let mut sensitivities = Vec::new();
    let number = |r: &Value, key: &str| r[key].as_f64().unwrap();
    for (i, row) in rows.iter().enumerate() {
        if row["status"] != "valid" {
            failures.push(format!("row {i} failed"));
            continue;
        }
        if row["checkpoint_roundtrip"]["component_bits_equal"] != true
            || number(&row["checkpoint_roundtrip"], "rdm_frobenius") != 0.
        {
            failures.push(format!("row {i} roundtrip"));
        }
        for (key, value) in row["continuation"].as_object().unwrap() {
            if value.as_f64().unwrap() > 1e-10 {
                failures.push(format!("row {i} continuation {key}"));
            }
        }
        for key in ["energy", "local_observable"] {
            if number(&row[key], "disagreement")
                > 1e-10_f64.max(0.1 * number(&row[key], "direct_error"))
            {
                failures.push(format!("row {i} {key} direct/RDM disagreement"));
            }
        }
    }
    for coarse in rows
        .iter()
        .filter(|r| r["settings"]["dtau"] == 0.1 && r["status"] == "valid")
    {
        let matched = |r: &&Value| {
            ["backend", "epsilon", "cap", "fixed_point_tolerance"]
                .iter()
                .all(|k| r["settings"][k] == coarse["settings"][k])
                && r["settings"]["dtau"] == 0.05
                && r["status"] == "valid"
        };
        if let Some(fine) = rows.iter().find(matched) {
            for (j, key) in ["energy", "local_observable"].iter().enumerate() {
                let a = number(&coarse[key], "rdm_error");
                let b = number(&fine[key], "rdm_error");
                let p = if a.is_finite()
                    && b.is_finite()
                    && a > floors[j]
                    && b > floors[j]
                    && a > 0.
                    && b > 0.
                {
                    Some((a / b).log2())
                } else {
                    None
                };
                if b >= a {
                    failures.push(format!("nondecreasing {key}: {}", coarse["settings"]));
                }
                let primary = coarse["settings"]["epsilon"] == 1e-13
                    && coarse["settings"]["cap"] == 128
                    && coarse["settings"]["fixed_point_tolerance"] == 1e-12;
                if primary && p.is_none_or(|p| p < 1.6) {
                    failures.push(format!("primary order {key}"));
                }
                refinements.push(json!({"settings":coarse["settings"],"observable":key,"errors":[a,b],"p":p,"primary":primary}));
            }
        }
    }
    for primary in rows.iter().filter(|r| {
        r["status"] == "valid"
            && r["settings"]["epsilon"] == 1e-13
            && r["settings"]["cap"] == 128
            && r["settings"]["fixed_point_tolerance"] == 1e-12
    }) {
        for key in ["energy", "local_observable"] {
            let shift = rows
                .iter()
                .filter(|r| {
                    r["status"] == "valid"
                        && r["settings"]["backend"] == primary["settings"]["backend"]
                        && r["settings"]["dtau"] == primary["settings"]["dtau"]
                })
                .map(|r| (number(&r[key], "rdm") - number(&primary[key], "rdm")).abs())
                .fold(0.0_f64, f64::max);
            let fraction = shift / number(&primary[key], "rdm_error");
            if fraction >= 0.05 {
                failures.push(format!("sensitivity {key}: {}", primary["settings"]));
            }
            sensitivities.push(json!({"settings":primary["settings"],"observable":key,"max_shift":shift,"fraction_of_primary_error":fraction}));
        }
    }
    evidence::emit(
        destination,
        &json!({"kind":"checkpoint_rdm_refinement","context":evidence::context(),
        "model":{"J":1.0,"g":0.7,"beta":1.0,"nk":16_384,"complex_unitary":"diag(1,i)"},
        "reference_diagnostic":{"central_difference_step":1e-4,"half_step":5e-5,"energy_shift":floors[0],"local_shift":floors[1],"caveat":"finite-difference diagnostic, not rigorous error bound"},
        "rows":rows,"refinements":refinements,"sensitivities":sensitivities,"acceptance_failures":failures}),
    );
    assert_eq!(rows.len(), 32);
    assert!(failures.is_empty(), "acceptance failures: {failures:?}");
}
