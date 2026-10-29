//! Fixed coefficient-one AKLT projector reference, with ordinary spin-one thermodynamics.
pub mod output;
pub mod v3;
use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
use crate::itebd_checkpoint::ItebdStepSegment;
use crate::reference_trajectory::{
    expectation, run_scheduled, Measurement, NativeEngine, ReferenceTarget, RunPolicy,
    TrajectorySettings,
};
pub use crate::reference_trajectory::{CapCoverage, CapEvidence};
use crate::tensor::Truncation;
use anyhow::{ensure, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
pub const CONTRACT: &str = "aklt-projector-reference-v1";
pub const CONTRACT_V2: &str = "aklt-projector-reference-v2";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AkltVariant {
    Fine,
    TimeStep,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkltModel {
    pub name: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkltReferenceRequest {
    pub schema_version: u32,
    pub contract: String,
    pub model: AkltModel,
    pub variant: AkltVariant,
    #[serde(deserialize_with = "required_deadline")]
    pub max_seconds: Option<f64>,
    pub source_revision: String,
    pub worker_sha256: String,
}
// A missing key must remain an error; explicit JSON null selects v2 only.
fn required_deadline<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<f64>, D::Error> {
    Option::<f64>::deserialize(deserializer)
}
/// Construct only with resolve_request; production numerical controls cannot be changed.
pub struct AkltReferenceSettings {
    request: AkltReferenceRequest,
    hamiltonian: ItebdHamiltonian,
    truncation: Truncation,
    max_dtau: f64,
    segments: Vec<ItebdStepSegment>,
    targets: Vec<ReferenceTarget>,
}
impl AkltReferenceSettings {
    pub fn request(&self) -> &AkltReferenceRequest {
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
pub struct AkltSampleValues {
    pub energy: f64,
    pub mx: f64,
    pub mz: f64,
    pub free_energy: f64,
    pub entropy: f64,
    pub parity_delta_energy: f64,
    pub parity_delta_mx: f64,
    pub parity_delta_mz: f64,
    pub direct_energy_difference: f64,
    pub maximum_rdm_fixed_point_residual: f64,
    pub maximum_rdm_trace_residual: f64,
    pub maximum_rdm_hermiticity_residual: f64,
    pub maximum_rdm_iterations: usize,
    pub minimum_rdm_eigenvalue: f64,
    pub bond_dimension_ab: usize,
    pub bond_dimension_ba: usize,
    pub cap_reached_now: bool,
}

pub type AkltSample = crate::reference_trajectory::ScheduledSample<AkltSampleValues>;
pub type AkltOutcome = crate::reference_trajectory::ScheduledOutcome<AkltSampleValues>;
pub fn resolve_request(request: &AkltReferenceRequest) -> Result<AkltReferenceSettings> {
    let valid_contract = (request.schema_version == 1
        && request.contract == CONTRACT
        && request.max_seconds == Some(3600.))
        || (request.schema_version == 2
            && request.contract == CONTRACT_V2
            && request.max_seconds.is_none());
    ensure!(
        valid_contract,
        "request schema/contract/max_seconds mismatch: v1 requires3600; v2 requires explicit null"
    );
    ensure!(
        request.model.name == "aklt_projector",
        "request.model requires aklt_projector"
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
    let split = if request.variant == AkltVariant::TimeStep {
        1
    } else {
        2
    };
    let (segments, targets) = crate::reference_trajectory::nested_schedule(split)?;
    let h = crate::model::AkltProjector.local();
    let hamiltonian = ItebdHamiltonian::try_from_complex(
        h.two_site_h.map(Complex64::from),
        h.site_energy.map(Complex64::from),
    )
    .context("validate AKLT projector reference Hamiltonian")?;
    Ok(AkltReferenceSettings {
        request: request.clone(),
        hamiltonian,
        truncation: Truncation {
            epsilon: 1e-13,
            max_bond: Some(128),
        },
        max_dtau: if split == 1 { 0.01 } else { 0.005 },
        segments,
        targets,
    })
}
/// Resolve every production control without evolution or filesystem effects.
pub fn preflight(request: &AkltReferenceRequest) -> Result<serde_json::Value> {
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
    let contract_name = &request.contract;
    Ok(serde_json::json!({
        "schema_version":request.schema_version,"contract":contract_name,"max_seconds":request.max_seconds,"request":request,"basis_order":"first_site_fastest","physical_dimension":3,
        "cell_sites":2,"backend":"real","initial_beta":0.0,"temperature_tick_scale":10,
        "log_norm_convention":"sum step and canonicalization log norms per two-site cell; f=-(ln(3)+sum/2)/beta",
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
pub fn measure_reference(
    state: &ItebdState,
    ham: &ItebdHamiltonian,
    beta: f64,
    accumulated_log_norm: f64,
) -> Result<AkltSampleValues> {
    measure_reference_with_cap(state, ham, beta, accumulated_log_norm, 128)
}
pub(crate) fn measure_reference_with_cap(
    state: &ItebdState,
    ham: &ItebdHamiltonian,
    beta: f64,
    accumulated_log_norm: f64,
    cap: usize,
) -> Result<AkltSampleValues> {
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
        ham.dim() == 3,
        "reference measurements require spin-one physical dimension 3"
    );
    let summary = crate::itebd_state_view::ItebdStateRef::from(state)
        .validate()
        .context("reference measurement state validation")?;
    let h2 = match ham {
        ItebdHamiltonian::Real(h) => h.two_site_h.map(Complex64::from),
        ItebdHamiltonian::Complex(h) => h.two_site_h().clone(),
    };
    let x = crate::model::sx1().map(Complex64::from);
    let z = crate::model::sz1().map(Complex64::from);
    let mut values = AkltSampleValues {
        minimum_rdm_eigenvalue: f64::INFINITY,
        cap_reached_now: summary.bond_dimensions.iter().any(|d| *d >= cap),
        bond_dimension_ab: summary.bond_dimensions[0],
        bond_dimension_ba: summary.bond_dimensions[1],
        ..Default::default()
    };
    let mut parity_values = [[0.; 3]; 2];
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
            }
        }
    }
    let [a, b] = parity_values;
    values.energy = (a[0] + b[0]) / 2.;
    values.mx = (a[1] + b[1]) / 2.;
    values.mz = (a[2] + b[2]) / 2.;
    values.parity_delta_energy = a[0] - b[0];
    values.parity_delta_mx = a[1] - b[1];
    values.parity_delta_mz = a[2] - b[2];
    values.direct_energy_difference = values.energy
        - crate::itebd_auto::energy_density_auto(state, ham).context("reference direct energy")?;
    values.free_energy =
        crate::itebd::free_energy_from_log_norm(accumulated_log_norm / 2., beta, 3);
    values.entropy = (values.energy - values.free_energy) * beta;
    values.validate()?;
    Ok(values)
}
impl AkltSampleValues {
    fn validate(&self) -> Result<()> {
        for value in [
            self.energy,
            self.mx,
            self.mz,
            self.free_energy,
            self.entropy,
            self.parity_delta_energy,
            self.parity_delta_mx,
            self.parity_delta_mz,
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

impl Measurement for AkltSampleValues {
    fn validate(&self) -> Result<()> {
        AkltSampleValues::validate(self)
    }
}
pub fn run_reference(
    settings: &AkltReferenceSettings,
    callback: &mut dyn FnMut(&AkltSample) -> Result<()>,
) -> Result<AkltOutcome> {
    let started = std::time::Instant::now();
    run_with_elapsed(settings, callback, &mut || started.elapsed().as_secs_f64())
}
pub(crate) fn run_with_elapsed(
    settings: &AkltReferenceSettings,
    callback: &mut dyn FnMut(&AkltSample) -> Result<()>,
    elapsed: &mut dyn FnMut() -> f64,
) -> Result<AkltOutcome> {
    run_scheduled(
        TrajectorySettings {
            hamiltonian: &settings.hamiltonian,
            truncation: &settings.truncation,
            segments: &settings.segments,
            targets: &settings.targets,
            policy: RunPolicy::Aklt {
                max_seconds: settings.request.max_seconds,
                bond_cap: 128,
            },
        },
        callback,
        &mut NativeEngine(measure_reference),
        elapsed,
    )
}
