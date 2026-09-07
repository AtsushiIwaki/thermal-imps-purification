//! Host-specific timing evidence only; equal-bond fixtures are not uniqueness oracles.
#[path = "../../tests/support/itebd_evidence.rs"]
mod evidence;
use super::environment::prepare;
use super::interval::contract_interval;
use super::oracles::fixture;
use super::{RdmOptions, RdmParity};
use crate::itebd_state_view::ItebdStateRef;
use serde_json::{json, Value};
use std::time::Instant;

fn matrix(tensor: &crate::tensor::Tensor) -> nalgebra::DMatrix<num_complex::Complex64> {
    let values = if tensor.is_complex() {
        tensor.to_vec::<num_complex::Complex64>().unwrap()
    } else {
        tensor
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(num_complex::Complex64::from)
            .collect()
    };
    let dim = tensor.dims()[..tensor.dims().len() / 2].iter().product();
    nalgebra::DMatrix::from_column_slice(dim, dim, &values)
}

fn timed_row(
    complex: bool,
    chi: usize,
    start: RdmParity,
    length: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let state = fixture(complex, chi, chi);
    let view = ItebdStateRef::from(&state);
    let options = RdmOptions::default();
    // One untimed prepare + contraction warmup supplies a valid reference.
    let mut cache = prepare(view, &options)?;
    let (reference, reference_report) = contract_interval(&mut cache, start, length, &options)?;
    let reference = matrix(&reference);
    let mut preparation_seconds = Vec::new();
    let mut contraction_seconds = Vec::new();
    let mut samples = Vec::new();
    for _ in 0..7 {
        let now = Instant::now();
        let mut prepared = prepare(view, &options)?;
        preparation_seconds.push(now.elapsed().as_secs_f64());
        // Validate the timed preparation against the untimed reference outside its timer.
        let (prepared_output, prepared_report) =
            contract_interval(&mut prepared, start, length, &options)?;
        let preparation_difference = (matrix(&prepared_output) - &reference).norm();
        let now = Instant::now();
        // Reuse W and environments; contract_interval resets per-call accounting itself.
        let (output, report) = contract_interval(&mut cache, start, length, &options)?;
        contraction_seconds.push(now.elapsed().as_secs_f64());
        let contraction_difference = (matrix(&output) - &reference).norm();
        evidence::finite(
            "timing sample",
            &[
                preparation_difference,
                contraction_difference,
                *preparation_seconds.last().unwrap(),
                *contraction_seconds.last().unwrap(),
                prepared_report.left.relative_residual,
                prepared_report.right.relative_residual,
                prepared_report.left.eigenvalue,
                prepared_report.right.eigenvalue,
                report.trace_residual,
                report.hermiticity_residual,
                report.minimum_eigenvalue,
            ],
        )?;
        samples.push(json!({"preparation_difference":preparation_difference,
            "contraction_difference":contraction_difference,
            "left_iterations":prepared_report.left.iterations,"right_iterations":prepared_report.right.iterations,
            "left_residual":prepared_report.left.relative_residual,"right_residual":prepared_report.right.relative_residual,
            "left_eigenvalue":prepared_report.left.eigenvalue,"right_eigenvalue":prepared_report.right.eigenvalue,
            "preparation_largest_elements":prepared.preparation_largest,
            "largest_intermediate_elements":report.largest_intermediate_elements,
            "trace_residual":report.trace_residual,"hermiticity_residual":report.hermiticity_residual,
            "minimum_eigenvalue":report.minimum_eigenvalue}));
    }
    Ok(
        json!({"status":"valid","preparation_seconds":preparation_seconds,"contraction_seconds":contraction_seconds,
        "samples":samples,"output_dimensions":[1usize << length,1usize << length],
        "largest_intermediate_elements":reference_report.largest_intermediate_elements}),
    )
}

#[test]
#[ignore]
fn checkpoint_rdm_timing_matrix() {
    let destination = evidence::destination().expect("exclusive evidence destination");
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for complex in [false, true] {
        for chi in [4, 8] {
            for start in [RdmParity::A, RdmParity::B] {
                for length in 1..=4 {
                    let mut row = match timed_row(complex, chi, start, length) {
                        Ok(row) => row,
                        Err(error) => json!({"status":"failed","typed_error":format!("{error:?}")}),
                    };
                    row["settings"] = json!({"backend":if complex {"complex"} else {"real"},"chi_ab":chi,"chi_ba":chi,
            "physical_dim":2,"ancilla_dim":2,"start":format!("{start:?}"),"length":length,
            "fixed_point_tolerance":1e-12,"max_iterations":10_000,"trace_tolerance":1e-10,
            "hermiticity_tolerance":1e-10,"positivity_tolerance":1e-10,
            "max_output_elements":1_048_576,"max_intermediate_elements":16_777_216});
                    if row["status"] != "valid" {
                        failures.push(format!("row {} failed", rows.len()));
                    } else {
                        for sample in row["samples"].as_array().unwrap() {
                            for key in ["preparation_difference", "contraction_difference"] {
                                let difference = sample[key].as_f64().unwrap();
                                if !difference.is_finite() || difference > 1e-10 {
                                    failures.push(format!("row {} {key}", rows.len()));
                                }
                            }
                        }
                    }
                    eprintln!("timing row {}: {}", rows.len() + 1, row["status"]);
                    rows.push(row);
                }
            }
        }
    }
    evidence::emit(
        destination,
        &json!({"kind":"checkpoint_rdm_timing","context":evidence::context(),
        "fixture":"original deterministic Task 4 formula; equal bonds, timing only, no primitivity claim",
        "warmups":1,"samples_per_row":7,"rows":rows,"acceptance_failures":failures}),
    );
    assert!(
        failures.is_empty(),
        "timing validity failures: {failures:?}"
    );
}
