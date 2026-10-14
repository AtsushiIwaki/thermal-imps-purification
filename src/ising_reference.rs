//! Fixed mixed-field Ising purification reference and ordinary thermodynamics.
pub mod output;
use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
use crate::itebd_checkpoint::{ItebdCheckpointProgress, ItebdStepSegment};
use crate::tensor::Truncation;
use anyhow::{ensure, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReferenceVariant {
    Fine,
    TimeStep,
    BondCap,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceModel {
    pub name: String,
    pub g: f64,
    pub h: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceRequest {
    pub schema_version: u32,
    pub model: ReferenceModel,
    pub variant: ReferenceVariant,
    pub source_revision: String,
    pub worker_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceTarget {
    pub temperature_tick: u32,
    pub temperature: f64,
    pub beta: f64,
    pub completed_steps: u64,
}
/// Construct only with resolve_request; production numerical controls cannot be changed.
pub struct ReferenceSettings {
    request: ReferenceRequest,
    hamiltonian: ItebdHamiltonian,
    truncation: Truncation,
    max_dtau: f64,
    segments: Vec<ItebdStepSegment>,
    targets: Vec<ReferenceTarget>,
}
impl ReferenceSettings {
    pub fn request(&self) -> &ReferenceRequest {
        &self.request
    }
    pub fn hamiltonian(&self) -> &ItebdHamiltonian {
        &self.hamiltonian
    }
    pub fn truncation(&self) -> &Truncation {
        &self.truncation
    }
    pub fn max_dtau(&self) -> f64 {
        self.max_dtau
    }
    pub fn segments(&self) -> &[ItebdStepSegment] {
        &self.segments
    }
    pub fn targets(&self) -> &[ReferenceTarget] {
        &self.targets
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceSampleValues {
    pub energy: f64,
    pub mx: f64,
    pub mz: f64,
    pub zz: f64,
    pub free_energy: f64,
    pub entropy: f64,
    pub parity_delta_energy: f64,
    pub parity_delta_mx: f64,
    pub parity_delta_mz: f64,
    pub parity_delta_zz: f64,
    pub direct_energy_difference: f64,
    pub maximum_rdm_fixed_point_residual: f64,
    pub maximum_rdm_trace_residual: f64,
    pub maximum_rdm_hermiticity_residual: f64,
    pub maximum_rdm_iterations: usize,
    pub minimum_rdm_eigenvalue: f64,
    pub bond_dimension_ab: usize,
    pub bond_dimension_ba: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSample {
    pub temperature_tick: u32,
    pub temperature: f64,
    pub beta_target: f64,
    pub beta_actual: f64,
    pub completed_steps: u64,
    pub available: bool,
    pub reason: Option<String>,
    pub accumulated_log_norm: f64,
    #[serde(flatten)]
    pub values: Option<ReferenceSampleValues>,
}
pub struct ReferenceOutcome {
    pub samples: Vec<ReferenceSample>,
    pub terminal_state: Option<ItebdState>,
    pub progress: ItebdCheckpointProgress,
    pub status: String,
    pub reason: String,
    pub elapsed_seconds: f64,
}
pub fn resolve_request(request: &ReferenceRequest) -> Result<ReferenceSettings> {
    ensure!(
        request.schema_version == 1,
        "request.schema_version must be 1"
    );
    ensure!(
        request.model.name == "ising_mixed" && request.model.g == 1.05 && request.model.h == 0.5,
        "request.model requires ising_mixed with g=1.05 and h=0.5"
    );
    for (name, value, length) in [
        ("source_revision", &request.source_revision, 40),
        ("worker_sha256", &request.worker_sha256, 64),
    ] {
        ensure!(
            value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit()),
            "request.{name} requires {length} hexadecimal characters"
        );
    }
    let split = if request.variant == ReferenceVariant::TimeStep {
        1
    } else {
        2
    };
    let mut segments = Vec::with_capacity(20);
    let mut targets = Vec::with_capacity(20);
    let mut completed_steps = 0;
    for j in (1_u32..=20).rev() {
        let denominator = u64::from(j) * u64::from(j + 1);
        let coarse_steps = if j == 20 {
            25
        } else {
            500_u64.div_ceil(denominator)
        };
        let delta_beta = if j == 20 {
            0.5
        } else {
            10.0 / denominator as f64
        };
        let steps = coarse_steps * split;
        segments.push(ItebdStepSegment {
            steps,
            dtau: delta_beta / (2.0 * steps as f64),
        });
        completed_steps += steps;
        targets.push(ReferenceTarget {
            temperature_tick: j,
            temperature: j as f64 / 10.0,
            beta: 10.0 / j as f64,
            completed_steps,
        });
    }
    let (x, z, i) = (
        crate::model::pauli_x(),
        crate::model::pauli_z(),
        crate::model::id2(),
    );
    // Kronecker's right operand is the fastest physical site.
    let zz = z.kronecker(&z);
    let xi = i.kronecker(&x);
    let ix = x.kronecker(&i);
    let zi = i.kronecker(&z);
    let iz = z.kronecker(&i);
    let h2 = -&zz - (&xi + ix) * 0.525 - (&zi + iz) * 0.25;
    let site = h2.clone();
    let hamiltonian =
        ItebdHamiltonian::try_from_complex(h2.map(Complex64::from), site.map(Complex64::from))
            .context("validate mixed-field reference Hamiltonian")?;
    Ok(ReferenceSettings {
        request: request.clone(),
        hamiltonian,
        truncation: Truncation {
            epsilon: 1e-13,
            max_bond: Some(if request.variant == ReferenceVariant::BondCap {
                64
            } else {
                128
            }),
        },
        max_dtau: if split == 1 { 0.01 } else { 0.005 },
        segments,
        targets,
    })
}
/// Resolve every production control without evolution or filesystem effects.
pub fn preflight(request: &ReferenceRequest) -> Result<serde_json::Value> {
    let settings = resolve_request(request)?;
    let ItebdHamiltonian::Real(ham) = &settings.hamiltonian else {
        unreachable!("validated real model")
    };
    let rows = |m: &DMatrix<f64>| {
        (0..m.nrows())
            .map(|r| (0..m.ncols()).map(|c| m[(r, c)]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let rdm = crate::itebd_rdm::RdmOptions::default();
    Ok(serde_json::json!({
        "schema_version":1,"request":request,"basis_order":"first_site_fastest","physical_dimension":2,
        "cell_sites":2,"backend":"real","initial_beta":0.0,"temperature_tick_scale":10,
        "log_norm_convention":"sum step and canonicalization log norms per two-site cell; f=-(ln(2)+sum/2)/beta",
        "max_dtau":settings.max_dtau,"max_bond":settings.truncation.max_bond,"svd_cutoff":settings.truncation.epsilon,
        "trotter_order":"second","canonicalize_every":1,"canonicalizer":{"tolerance":1e-13,"max_iterations":4000,"support_floor":1e-12},
        "rdm":{"fixed_point_tolerance":rdm.fixed_point_tolerance,"max_iterations":rdm.max_iterations,
            "trace_tolerance":rdm.trace_tolerance,"hermiticity_tolerance":rdm.hermiticity_tolerance,"positivity_tolerance":rdm.positivity_tolerance,
            "max_output_elements":rdm.max_output_elements,"max_intermediate_elements":rdm.max_intermediate_elements},
        "two_site_h":rows(&ham.two_site_h),"site_energy":rows(&ham.site_energy),
        "targets":settings.targets,"time_grid":{"kind":"segmented","segments":settings.segments},
        "total_steps":settings.targets.last().unwrap().completed_steps,
        "file_membership":["conditions.json","summary.json","samples.csv"],"optional_terminal_checkpoint":"state.h5"
    }))
}
/// Ordinary physical thermodynamics from normalized physical RDMs and the norm ledger.
pub fn measure_reference(
    state: &ItebdState,
    ham: &ItebdHamiltonian,
    beta: f64,
    accumulated_log_norm: f64,
) -> Result<ReferenceSampleValues> {
    use crate::itebd_rdm::{reduced_density_matrix_auto, AutoRdmResult, RdmOptions, RdmParity};
    ensure!(
        beta.is_finite() && beta > 0.,
        "reference beta must be finite and positive"
    );
    ensure!(
        accumulated_log_norm.is_finite(),
        "reference accumulated_log_norm must be finite"
    );
    ensure!(
        ham.dim() == 2,
        "reference measurements require Pauli physical dimension 2"
    );
    let summary = crate::itebd_state_view::ItebdStateRef::from(state)
        .validate()
        .context("reference measurement state validation")?;
    let h2 = match ham {
        ItebdHamiltonian::Real(h) => h.two_site_h.map(Complex64::from),
        ItebdHamiltonian::Complex(h) => h.two_site_h().clone(),
    };
    let x = crate::model::pauli_x().map(Complex64::from);
    let z = crate::model::pauli_z().map(Complex64::from);
    let zz = z.kronecker(&z);
    let mut values = ReferenceSampleValues {
        minimum_rdm_eigenvalue: f64::INFINITY,
        bond_dimension_ab: summary.bond_dimensions[0],
        bond_dimension_ba: summary.bond_dimensions[1],
        ..Default::default()
    };
    let mut parity_values = [[0.; 4]; 2];
    for (p, parity) in [RdmParity::A, RdmParity::B].into_iter().enumerate() {
        for length in [1, 2] {
            let rdm = reduced_density_matrix_auto(state, parity, length, &RdmOptions::default())
                .with_context(|| format!("reference RDM parity {parity:?}, length {length}"))?;
            let (rho, report) = match rdm {
                AutoRdmResult::Real(r) => (r.density_matrix.map(Complex64::from), r.report),
                AutoRdmResult::Complex(r) => (r.density_matrix, r.report),
            };
            values.maximum_rdm_fixed_point_residual = values
                .maximum_rdm_fixed_point_residual
                .max(report.left.relative_residual)
                .max(report.right.relative_residual);
            values.maximum_rdm_trace_residual =
                values.maximum_rdm_trace_residual.max(report.trace_residual);
            values.maximum_rdm_hermiticity_residual = values
                .maximum_rdm_hermiticity_residual
                .max(report.hermiticity_residual);
            values.maximum_rdm_iterations = values
                .maximum_rdm_iterations
                .max(report.left.iterations)
                .max(report.right.iterations);
            values.minimum_rdm_eigenvalue =
                values.minimum_rdm_eigenvalue.min(report.minimum_eigenvalue);
            if length == 1 {
                parity_values[p][1] = expectation(&rho, &x)?;
                parity_values[p][2] = expectation(&rho, &z)?;
            } else {
                parity_values[p][0] = expectation(&rho, &h2)?;
                parity_values[p][3] = expectation(&rho, &zz)?;
            }
        }
    }
    let [a, b] = parity_values;
    values.energy = (a[0] + b[0]) / 2.;
    values.mx = (a[1] + b[1]) / 2.;
    values.mz = (a[2] + b[2]) / 2.;
    values.zz = (a[3] + b[3]) / 2.;
    values.parity_delta_energy = a[0] - b[0];
    values.parity_delta_mx = a[1] - b[1];
    values.parity_delta_mz = a[2] - b[2];
    values.parity_delta_zz = a[3] - b[3];
    values.direct_energy_difference = values.energy
        - crate::itebd_auto::energy_density_auto(state, ham).context("reference direct energy")?;
    values.free_energy =
        crate::itebd::free_energy_from_log_norm(accumulated_log_norm / 2., beta, 2);
    values.entropy = (values.energy - values.free_energy) * beta;
    values.validate()?;
    Ok(values)
}
impl ReferenceSampleValues {
    fn validate(&self) -> Result<()> {
        for value in [
            self.energy,
            self.mx,
            self.mz,
            self.zz,
            self.free_energy,
            self.entropy,
            self.parity_delta_energy,
            self.parity_delta_mx,
            self.parity_delta_mz,
            self.parity_delta_zz,
            self.direct_energy_difference,
            self.maximum_rdm_fixed_point_residual,
            self.maximum_rdm_trace_residual,
            self.maximum_rdm_hermiticity_residual,
            self.minimum_rdm_eigenvalue,
        ] {
            ensure!(
                value.is_finite(),
                "reference measurement contains a nonfinite scalar"
            );
        }
        ensure!(
            self.bond_dimension_ab > 0 && self.bond_dimension_ba > 0,
            "reference measurement has an empty bond"
        );
        Ok(())
    }
}
pub fn run_reference(
    settings: &ReferenceSettings,
    callback: &mut dyn FnMut(&ReferenceSample) -> Result<()>,
) -> Result<ReferenceOutcome> {
    run_engine(settings, callback, &mut NativeEngine)
}
trait Engine {
    fn setup(&mut self, h: &ItebdHamiltonian) -> Result<ItebdState> {
        Ok(ItebdState::infinite_temperature(h)?)
    }
    fn step(
        &mut self,
        s: &mut ItebdState,
        h: &ItebdHamiltonian,
        dt: f64,
        t: &Truncation,
    ) -> Result<(crate::itebd::StepInfo, f64)>;
    fn measure(
        &mut self,
        s: &ItebdState,
        h: &ItebdHamiltonian,
        beta: f64,
        norm: f64,
    ) -> Result<ReferenceSampleValues> {
        measure_reference(s, h, beta, norm)
    }
}
struct NativeEngine;
impl Engine for NativeEngine {
    fn step(
        &mut self,
        state: &mut ItebdState,
        ham: &ItebdHamiltonian,
        dt: f64,
        truncation: &Truncation,
    ) -> Result<(crate::itebd::StepInfo, f64)> {
        let step =
            crate::itebd_auto::imaginary_time_step_second_order_auto(state, ham, dt, truncation)
                .context("Strang evolution step")?;
        let canonical = crate::itebd_auto::canonicalize_auto(state, ham)
            .context("canonicalization after complete Strang step")?;
        Ok((step, canonical))
    }
}
fn run_engine(
    settings: &ReferenceSettings,
    callback: &mut dyn FnMut(&ReferenceSample) -> Result<()>,
    engine: &mut dyn Engine,
) -> Result<ReferenceOutcome> {
    let start = std::time::Instant::now();
    let mut state = engine
        .setup(&settings.hamiltonian)
        .context("reference initial state setup")?;
    crate::itebd_state_view::ItebdStateRef::from(&state)
        .validate()
        .context("reference initial state validation")?;
    let mut progress = ItebdCheckpointProgress {
        beta: 0.,
        completed_steps: 0,
        accumulated_log_norm: 0.,
        last_step: None,
    };
    let mut samples = Vec::with_capacity(settings.targets.len());
    let mut stop: Option<String> = None;
    let mut reasons = Vec::new();
    for (segment, target) in settings.segments.iter().zip(&settings.targets) {
        if stop.is_none() {
            for _ in 0..segment.steps {
                let attempt = (|| -> Result<_> {
                    let (step, canonical) = engine.step(
                        &mut state,
                        &settings.hamiltonian,
                        segment.dtau,
                        &settings.truncation,
                    )?;
                    let norm = progress.accumulated_log_norm + step.log_norm + canonical;
                    ensure!(
                        step.log_norm.is_finite() && canonical.is_finite() && norm.is_finite(),
                        "nonfinite evolution normalization"
                    );
                    ensure!(
                        step.min_singular_value.is_finite() && step.max_bond > 0,
                        "invalid evolution step diagnostics"
                    );
                    Ok((step, norm))
                })();
                match attempt {
                    Ok((step, norm)) => {
                        progress.beta += 2. * segment.dtau;
                        progress.completed_steps += 1;
                        progress.accumulated_log_norm = norm;
                        progress.last_step = Some(step);
                    }
                    Err(error) => {
                        let reason = format!(
                            "evolution failed at step {}: {error:#}",
                            progress.completed_steps + 1
                        );
                        reasons.push(reason.clone());
                        stop = Some(reason);
                        break;
                    }
                }
            }
        }
        let measured = if let Some(reason) = &stop {
            Err(anyhow::anyhow!(reason.clone()))
        } else {
            (|| {
                ensure!(
                    progress.completed_steps == target.completed_steps
                        && (progress.beta - target.beta).abs() <= 1e-12 * target.beta.max(1.),
                    "target beta/step roundoff agreement failed"
                );
                let values = engine.measure(
                    &state,
                    &settings.hamiltonian,
                    progress.beta,
                    progress.accumulated_log_norm,
                )?;
                values.validate()?;
                Ok(values)
            })()
        };
        let (values, reason) = match measured {
            Ok(v) => (Some(v), None),
            Err(e) => {
                let reason = format!("target {} unavailable: {e:#}", target.temperature_tick);
                if stop.is_none() {
                    reasons.push(reason.clone());
                }
                (None, Some(reason))
            }
        };
        let sample = ReferenceSample {
            temperature_tick: target.temperature_tick,
            temperature: target.temperature,
            beta_target: target.beta,
            beta_actual: progress.beta,
            completed_steps: progress.completed_steps,
            available: values.is_some(),
            reason,
            accumulated_log_norm: progress.accumulated_log_norm,
            values,
        };
        callback(&sample).with_context(|| {
            format!(
                "reference output callback at target {}",
                target.temperature_tick
            )
        })?;
        samples.push(sample);
    }
    // A failed step may have mutated the state beyond this ledger. Do not label it with
    // the previous beta, even if its tensors still pass structural validation.
    let terminal_state = if stop.is_some() {
        None
    } else {
        match crate::itebd_state_view::ItebdStateRef::from(&state).validate() {
            Ok(_) => Some(state),
            Err(e) => {
                reasons.push(format!("terminal state validation: {e}"));
                None
            }
        }
    };
    Ok(ReferenceOutcome {
        samples,
        terminal_state,
        progress,
        status: if reasons.is_empty() {
            "complete"
        } else {
            "partial"
        }
        .into(),
        reason: if reasons.is_empty() {
            "completed all requested targets".into()
        } else {
            reasons.join("; ")
        },
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}
fn expectation(rho: &DMatrix<Complex64>, operator: &DMatrix<Complex64>) -> Result<f64> {
    ensure!(
        rho.is_square() && rho.shape() == operator.shape(),
        "RDM expectation shape mismatch"
    );
    // Tr(rho O), with bra/ket orientation supplied by the matrix product.
    let value = (rho * operator).trace();
    ensure!(
        value.re.is_finite()
            && value.im.is_finite()
            && value.im.abs() <= 1e-10 * value.re.abs().max(1.),
        "RDM expectation is not finite and real: {value}"
    );
    Ok(value.re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{id2, pauli_z};
    use anyhow::bail;
    fn request() -> ReferenceRequest {
        serde_json::from_value(serde_json::json!({"schema_version":1,"model":{"name":"ising_mixed","g":1.05,"h":0.5},"variant":"fine","source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})).unwrap()
    }
    fn tiny() -> ReferenceSettings {
        let z = pauli_z();
        let i = id2();
        let h = -(i.kronecker(&z) + z.kronecker(&i)) * 0.25;
        let ham = ItebdHamiltonian::try_from_complex(
            h.map(|x| Complex64::new(x, 0.)),
            (-i.kronecker(&z) * 0.5).map(|x| Complex64::new(x, 0.)),
        )
        .unwrap();
        ReferenceSettings {
            request: request(),
            hamiltonian: ham,
            truncation: Truncation {
                epsilon: 1e-13,
                max_bond: Some(2),
            },
            max_dtau: 0.05,
            segments: vec![
                ItebdStepSegment {
                    steps: 2,
                    dtau: 0.05
                };
                2
            ],
            targets: vec![
                ReferenceTarget {
                    temperature_tick: 50,
                    temperature: 5.,
                    beta: 0.2,
                    completed_steps: 2,
                },
                ReferenceTarget {
                    temperature_tick: 25,
                    temperature: 2.5,
                    beta: 0.4,
                    completed_steps: 4,
                },
            ],
        }
    }
    #[test]
    fn reference_pauli_energy_and_entropy_oracle() {
        let out = run_reference(&tiny(), &mut |_| Ok(())).unwrap();
        assert_eq!(out.samples.len(), 2);
        let v = out.samples[1].values.as_ref().unwrap();
        let x: f64 = 0.2;
        let logz = (2. * x.cosh()).ln();
        for (got, want) in [
            (v.energy, -0.5 * x.tanh()),
            (v.free_energy, -logz / 0.4),
            (v.entropy, logz - x * x.tanh()),
        ] {
            assert!((got - want).abs() < 1e-9, "{got} != {want}");
        }
        assert_eq!(out.progress.completed_steps, 4);
        assert!(out.terminal_state.is_some());
    }
    struct Seam {
        steps: usize,
        measures: usize,
        fail_step: Option<usize>,
        gap: Option<usize>,
    }
    impl Engine for Seam {
        fn step(
            &mut self,
            state: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &Truncation,
        ) -> Result<(crate::itebd::StepInfo, f64)> {
            self.steps += 1;
            if self.fail_step == Some(self.steps) {
                if let ItebdState::Real(s) = state {
                    let values =
                        s.a.gamma
                            .to_vec::<f64>()
                            .unwrap()
                            .into_iter()
                            .map(|v| 2.0 * v)
                            .collect();
                    s.a.gamma =
                        crate::tensor::Tensor::from_dense(s.a.gamma.indices.clone(), values)
                            .unwrap();
                }
                crate::itebd_state_view::ItebdStateRef::from(&*state)
                    .validate()
                    .unwrap();
                bail!("native step failure after mutation")
            }
            Ok((
                crate::itebd::StepInfo {
                    max_bond: 1,
                    min_singular_value: 1.,
                    log_norm: 0.2,
                },
                0.3,
            ))
        }
        fn measure(
            &mut self,
            s: &ItebdState,
            h: &ItebdHamiltonian,
            b: f64,
            n: f64,
        ) -> Result<ReferenceSampleValues> {
            self.measures += 1;
            if self.gap == Some(self.measures) {
                bail!("target RDM unavailable")
            }
            measure_reference(s, h, b, n)
        }
    }
    fn seam() -> Seam {
        Seam {
            steps: 0,
            measures: 0,
            fail_step: None,
            gap: None,
        }
    }
    #[test]
    fn reference_failed_measurement_preserves_prior_targets() {
        let mut engine = seam();
        engine.gap = Some(1);
        let mut emitted = vec![];
        let out = run_engine(
            &tiny(),
            &mut |s| {
                emitted.push(s.clone());
                Ok(())
            },
            &mut engine,
        )
        .unwrap();
        assert_eq!(out.samples.len(), 2);
        assert!(!out.samples[0].available);
        assert!(out.samples[0]
            .reason
            .as_ref()
            .unwrap()
            .contains("target RDM unavailable"));
        assert!(out.samples[1].available);
        assert_eq!(emitted.len(), 2);
        assert_eq!(engine.measures, 2);
        assert!(out.terminal_state.is_some());
        assert_eq!(out.status, "partial");
        assert_eq!(out.progress.accumulated_log_norm, 2.);
        let f = out.samples[1].values.as_ref().unwrap().free_energy;
        assert!((f + (2f64.ln() + 1.) / 0.4).abs() < 1e-12);
    }
    #[test]
    fn reference_later_step_failure_has_no_invented_checkpoint_or_retry() {
        let mut engine = seam();
        engine.fail_step = Some(3);
        let mut emitted = vec![];
        let out = run_engine(
            &tiny(),
            &mut |s| {
                emitted.push(s.clone());
                Ok(())
            },
            &mut engine,
        )
        .unwrap();
        assert_eq!(out.samples.len(), 2);
        assert!(out.samples[0].available);
        assert!(!out.samples[1].available);
        assert!(out.reason.contains("native step failure after mutation"));
        assert_eq!(out.progress.completed_steps, 2);
        assert_eq!(out.progress.beta, 0.2);
        assert!(out.terminal_state.is_none());
        assert_eq!(engine.steps, 3);
        assert_eq!(engine.measures, 1);
        assert_eq!(emitted.len(), 2);
    }
    #[test]
    fn reference_callback_and_setup_errors_preserve_cause() {
        let error = run_engine(
            &tiny(),
            &mut |_| Err(std::io::Error::other("disk callback cause").into()),
            &mut seam(),
        )
        .err()
        .expect("callback must fail");
        assert!(error.downcast_ref::<std::io::Error>().is_some());
        assert!(format!("{error:#}").contains("disk callback cause"));
        struct BadSetup;
        impl Engine for BadSetup {
            fn setup(&mut self, _: &ItebdHamiltonian) -> Result<ItebdState> {
                Err(std::io::Error::other("native setup cause").into())
            }
            fn step(
                &mut self,
                _: &mut ItebdState,
                _: &ItebdHamiltonian,
                _: f64,
                _: &Truncation,
            ) -> Result<(crate::itebd::StepInfo, f64)> {
                unreachable!()
            }
        }
        let error = run_engine(&tiny(), &mut |_| Ok(()), &mut BadSetup)
            .err()
            .expect("setup must fail");
        assert!(error.downcast_ref::<std::io::Error>().is_some());
        assert!(format!("{error:#}").contains("native setup cause"));
    }
    #[test]
    fn reference_rdm_trace_real_complex_basis_oracle() {
        // Independent component sum uses rho_ij O_ji, including complex conjugate off-diagonals.
        for phase in [0., 0.2] {
            let rho = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(0.7, 0.),
                    Complex64::new(0.1, phase),
                    Complex64::new(0.1, -phase),
                    Complex64::new(0.3, 0.),
                ],
            );
            let op = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(2., 0.),
                    Complex64::new(0.4, if phase == 0. { 0. } else { -0.3 }),
                    Complex64::new(0.4, if phase == 0. { 0. } else { 0.3 }),
                    Complex64::new(-1., 0.),
                ],
            );
            let mut oracle = Complex64::new(0., 0.);
            for a in 0..2 {
                for b in 0..2 {
                    oracle += rho[(a, b)] * op[(b, a)];
                }
            }
            assert!((expectation(&rho, &op).unwrap() - oracle.re).abs() < 1e-14);
            let joint = rho.kronecker(&DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
            ])));
            let first = id2()
                .map(|x| Complex64::new(x, 0.))
                .kronecker(&pauli_z().map(|x| Complex64::new(x, 0.)));
            assert!((expectation(&joint, &first).unwrap() - 1.).abs() < 1e-14);
        }
    }
}
