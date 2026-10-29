//! Cap256 controls and measurements; legacy AKLT requests retain their meaning.
use super::*;
use crate::specific_heat::{SpecificHeatOptions, SpecificHeatReport, TailStopReason};
use std::collections::BTreeMap;

pub const CONTRACT_V3: &str = "aklt-projector-reference-v3";
pub const RECORDING_POLICY: &str = "target-snapshots-v1";
pub const HEAT_POLICY: &str = "variance-default-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkltReferenceRequestV3 {
    pub schema_version: u32,
    pub contract: String,
    pub model: AkltModel,
    pub variant: AkltVariant,
    pub max_bond: usize,
    #[serde(deserialize_with = "required_deadline")]
    pub max_seconds: Option<f64>,
    pub recording_policy: String,
    pub heat_capacity_policy: String,
    pub source_revision: String,
    pub worker_sha256: String,
}

pub struct AkltReferenceSettingsV3 {
    request: AkltReferenceRequestV3,
    base: AkltReferenceSettings,
    pub(super) resume: Option<super::v4::AkltReferenceResumeRequest>,
}
impl AkltReferenceSettingsV3 {
    pub fn request(&self) -> &AkltReferenceRequestV3 {
        &self.request
    }
    pub fn hamiltonian(&self) -> &ItebdHamiltonian {
        self.base.hamiltonian()
    }
    pub fn truncation(&self) -> &Truncation {
        self.base.truncation()
    }
    pub fn segments(&self) -> &[ItebdStepSegment] {
        &self.base.segments()[if self.resume.is_some() { 14 } else { 0 }..]
    }
    pub fn targets(&self) -> &[ReferenceTarget] {
        &self.base.targets()[if self.resume.is_some() { 14 } else { 0 }..]
    }
    pub(super) fn full_segments(&self) -> &[ItebdStepSegment] {
        self.base.segments()
    }
    pub(super) fn with_resume(mut self, r: super::v4::AkltReferenceResumeRequest) -> Self {
        self.resume = Some(r);
        self
    }
    pub(super) fn schema_version(&self) -> u32 {
        if self.resume.is_some() {
            4
        } else {
            3
        }
    }
    pub(super) fn contract(&self) -> &str {
        if self.resume.is_some() {
            super::v4::CONTRACT_V4
        } else {
            CONTRACT_V3
        }
    }
    pub(super) fn recorded_request(&self) -> Result<serde_json::Value> {
        match &self.resume {
            Some(r) => Ok(serde_json::to_value(r)?),
            None => Ok(serde_json::to_value(&self.request)?),
        }
    }
    pub(super) fn preflight(&self) -> Result<serde_json::Value> {
        match &self.resume {
            Some(r) => super::v4::preflight_resume(r),
            None => preflight_v3(&self.request),
        }
    }
}

pub fn resolve_v3_request(request: &AkltReferenceRequestV3) -> Result<AkltReferenceSettingsV3> {
    let valid_contract = request.schema_version == 3
        && request.contract == CONTRACT_V3
        && request.max_seconds.is_none();
    ensure!(
        valid_contract,
        "v3 requires schema3 and explicit max_seconds:null"
    );
    ensure!(
        request.max_bond == 256 && request.variant == AkltVariant::Fine,
        "v3 requires fine variant and max_bond256"
    );
    ensure!(
        request.recording_policy == RECORDING_POLICY && request.heat_capacity_policy == HEAT_POLICY,
        "unsupported v3 recording/heat policy"
    );
    let mut base = resolve_request(&AkltReferenceRequest {
        schema_version: 2,
        contract: CONTRACT_V2.into(),
        model: request.model.clone(),
        variant: AkltVariant::Fine,
        max_seconds: None,
        source_revision: request.source_revision.clone(),
        worker_sha256: request.worker_sha256.clone(),
    })?;
    base.truncation.max_bond = Some(256);
    Ok(AkltReferenceSettingsV3 {
        request: request.clone(),
        base,
        resume: None,
    })
}

pub fn preflight_v3(request: &AkltReferenceRequestV3) -> Result<serde_json::Value> {
    let settings = resolve_v3_request(request)?;
    let mut value = preflight(settings.base.request())?;
    value["schema_version"] = 3.into();
    value["contract"] = CONTRACT_V3.into();
    value["request"] = serde_json::to_value(request)?;
    value["max_bond"] = 256.into();
    value
        .as_object_mut()
        .unwrap()
        .remove("optional_terminal_checkpoint");
    value["optional_snapshot_checkpoint"] = "states.h5".into();
    value["snapshot_policy"] =
        serde_json::json!({"kind":"target_snapshots","file":"states.h5","maximum_count":20});
    let options = SpecificHeatOptions::default();
    value["heat_capacity"] = serde_json::json!({"kind":"energy_variance",
        "max_distance":options.max_distance,"relative_tolerance":options.relative_tolerance,
        "absolute_tolerance":options.absolute_tolerance,"consecutive_small_shells":options.consecutive_small_shells,
        "reality_tolerance":options.reality_tolerance});
    Ok(value)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeEvidence {
    pub environment: BTreeMap<String, Option<String>>,
    pub visible_cpu_count: usize,
    pub thread_hint: usize,
    pub provider: String,
}

pub fn capture_runtime() -> Result<RuntimeEvidence> {
    let mut environment = BTreeMap::new();
    for name in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        let value = match std::env::var(name) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(error).with_context(|| format!("read runtime environment {name}"))
            }
        };
        environment.insert(name.into(), value);
    }
    let backend = tenferro_cpu::CpuBackend::new();
    Ok(RuntimeEvidence {
        environment,
        visible_cpu_count: tenferro_cpu::available_parallelism(),
        thread_hint: backend.num_threads(),
        provider: format!("{:?}", backend.kind()),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VarianceHeatRecord {
    pub raw_energy_variance_per_site: f64,
    pub energy_variance_per_site: f64,
    pub specific_heat_per_site: f64,
    pub onsite_contribution: [f64; 2],
    pub positive_direction_contribution: [f64; 2],
    pub negative_direction_contribution: [f64; 2],
    pub parity_a_contribution: [f64; 2],
    pub parity_b_contribution: [f64; 2],
    pub max_distance: usize,
    pub stop_reason: String,
    pub last_positive_shell_magnitude: f64,
    pub last_negative_shell_magnitude: f64,
    pub max_imaginary_residual: f64,
}
impl From<SpecificHeatReport> for VarianceHeatRecord {
    fn from(r: SpecificHeatReport) -> Self {
        let pair = |z: Complex64| [z.re, z.im];
        Self {
            raw_energy_variance_per_site: r.raw_energy_variance_per_site,
            energy_variance_per_site: r.energy_variance_per_site,
            specific_heat_per_site: r.specific_heat_per_site,
            onsite_contribution: pair(r.onsite_contribution),
            positive_direction_contribution: pair(r.positive_direction_contribution),
            negative_direction_contribution: pair(r.negative_direction_contribution),
            parity_a_contribution: pair(r.parity_a_contribution),
            parity_b_contribution: pair(r.parity_b_contribution),
            max_distance: r.max_distance,
            stop_reason: match r.stop_reason {
                TailStopReason::ConsecutiveSmallShells => "consecutive_small_shells".into(),
            },
            last_positive_shell_magnitude: r.last_positive_shell_magnitude,
            last_negative_shell_magnitude: r.last_negative_shell_magnitude,
            max_imaginary_residual: r.max_imaginary_residual,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VarianceHeatEvidence {
    pub available: bool,
    pub reason: Option<String>,
    pub report: Option<VarianceHeatRecord>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkltValuesV3 {
    pub physical: AkltSampleValues,
    pub heat: VarianceHeatEvidence,
}
impl Measurement for AkltValuesV3 {
    fn validate(&self) -> Result<()> {
        self.physical.validate()?;
        ensure!(
            self.heat.available == self.heat.report.is_some(),
            "variance availability/report mismatch"
        );
        ensure!(
            self.heat.available == self.heat.reason.is_none(),
            "variance availability/reason mismatch"
        );
        if let Some(reason) = &self.heat.reason {
            ensure!(!reason.trim().is_empty(), "empty variance failure");
        }
        Ok(())
    }
    fn stop_reason(&self) -> Option<&str> {
        self.heat.reason.as_deref()
    }
}
pub fn measure_v3_reference(
    state: &ItebdState,
    ham: &ItebdHamiltonian,
    beta: f64,
    norm: f64,
) -> Result<AkltValuesV3> {
    let physical = measure_reference_with_cap(state, ham, beta, norm, 256)?;
    let heat = match crate::itebd_auto::specific_heat_with_options_auto(
        state,
        ham,
        beta,
        &SpecificHeatOptions::default(),
    ) {
        Ok(report) => VarianceHeatEvidence {
            available: true,
            reason: None,
            report: Some(report.into()),
        },
        Err(error) => VarianceHeatEvidence {
            available: false,
            reason: Some(format!("variance heat failed: {error:#}")),
            report: None,
        },
    };
    Ok(AkltValuesV3 { physical, heat })
}
