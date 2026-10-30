//! Fixed, scalar-only observations on the unchanged cap256 evolution grid.
use super::{v3, v4, AkltModel, AkltVariant, CapCoverage};
use crate::{
    itebd_auto::ItebdHamiltonian, itebd_checkpoint::ItebdStepSegment,
    reference_trajectory::step_sampled::StepMeasurementPoint, tensor::Truncation,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

pub const CONTRACT_V5: &str = "aklt-projector-reference-step-sampled-v5";
pub const MEASUREMENT_POLICY: &str = "every-two-complete-steps-v1";
pub const RECORDING_POLICY: &str = "observables-only-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkltStepSampledRequest {
    pub schema_version: u32,
    pub contract: String,
    pub model: AkltModel,
    pub variant: AkltVariant,
    pub max_bond: usize,
    #[serde(deserialize_with = "super::required_deadline")]
    pub max_seconds: Option<f64>,
    pub measurement_policy: String,
    pub recording_policy: String,
    pub heat_capacity_policy: String,
    pub source_revision: String,
    pub worker_sha256: String,
    pub restart: v4::RestartSpec,
}

pub(crate) struct AkltStepSampledSettings {
    request: AkltStepSampledRequest,
    base: v3::AkltReferenceSettingsV3,
    points: Vec<StepMeasurementPoint>,
}
impl AkltStepSampledSettings {
    pub(crate) fn request(&self) -> &AkltStepSampledRequest {
        &self.request
    }
    pub(crate) fn hamiltonian(&self) -> &ItebdHamiltonian {
        self.base.hamiltonian()
    }
    pub(crate) fn truncation(&self) -> &Truncation {
        self.base.truncation()
    }
    pub(crate) fn full_segments(&self) -> &[ItebdStepSegment] {
        self.base.full_segments()
    }
    pub(crate) fn remaining_segments(&self) -> &[ItebdStepSegment] {
        &self.full_segments()[1..]
    }
    pub(crate) fn points(&self) -> &[StepMeasurementPoint] {
        &self.points
    }
    pub(crate) fn new_points(&self) -> &[StepMeasurementPoint] {
        &self.points[1..]
    }
}

pub(crate) fn resolve_request(r: &AkltStepSampledRequest) -> Result<AkltStepSampledSettings> {
    ensure!(
        r.schema_version == 5 && r.contract == CONTRACT_V5 && r.max_seconds.is_none(),
        "step-sampled v5 requires schema5 and explicit max_seconds:null"
    );
    ensure!(
        r.max_bond == 256
            && r.variant == AkltVariant::Fine
            && r.measurement_policy == MEASUREMENT_POLICY
            && r.recording_policy == RECORDING_POLICY
            && r.heat_capacity_policy == v3::HEAT_POLICY,
        "unsupported step-sampled controls/policies"
    );
    let start = &r.restart;
    ensure!(
        !start.directory.is_empty() && !Path::new(&start.directory).is_absolute(),
        "step-sampled parent directory must be relative"
    );
    ensure!(
        start.snapshot_index == 0
            && start.progress.completed_steps == 50
            && start.progress.beta.is_finite()
            && (start.progress.beta - 0.5).abs() < 1e-12
            && start.progress.accumulated_log_norm.is_finite(),
        "step-sampled requires T2/step50/snapshot0"
    );
    ensure!(
        start.files.len() == 4
            && [
                "conditions.json",
                "summary.json",
                "samples.csv",
                "states.h5"
            ]
            .iter()
            .all(|name| start.files.contains_key(*name)),
        "step-sampled requires four parent pins"
    );
    for pin in start.files.values() {
        ensure!(
            pin.bytes > 0
                && pin.sha256.len() == 64
                && pin
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid step-sampled parent pin"
        );
    }
    let cap = &start.cap;
    ensure!(
        cap.coverage == CapCoverage::CompleteThroughStop
            && cap.observed_strang_steps == 50
            && cap
                .maximum_dimension_seen
                .is_some_and(|n| (1..=256).contains(&n))
            && cap.ever_reached == cap.maximum_dimension_seen.map(|n| n >= 256),
        "step-sampled requires complete boundary-row cap evidence"
    );
    let base = v3::resolve_v3_request(&v3::AkltReferenceRequestV3 {
        schema_version: 3,
        contract: v3::CONTRACT_V3.into(),
        model: r.model.clone(),
        variant: r.variant,
        max_bond: 256,
        max_seconds: None,
        recording_policy: v3::RECORDING_POLICY.into(),
        heat_capacity_policy: v3::HEAT_POLICY.into(),
        source_revision: r.source_revision.clone(),
        worker_sha256: r.worker_sha256.clone(),
    })?;
    let mut beta = start.progress.beta;
    let mut step = 50;
    let mut points = vec![StepMeasurementPoint {
        sample_index: 0,
        completed_steps: step,
        beta_expected: beta,
    }];
    for segment in &base.full_segments()[1..] {
        for _ in 0..segment.steps {
            beta += 2. * segment.dtau;
            step += 1;
            if (step - 50) % 2 == 0 {
                points.push(StepMeasurementPoint {
                    sample_index: points.len() as u32,
                    completed_steps: step,
                    beta_expected: beta,
                });
            }
        }
    }
    ensure!(
        step == 1014 && points.len() == 483,
        "step-sampled schedule mismatch"
    );
    Ok(AkltStepSampledSettings {
        request: r.clone(),
        base,
        points,
    })
}

pub fn preflight(r: &AkltStepSampledRequest) -> Result<Value> {
    let s = resolve_request(r)?;
    let mut value = v3::preflight_v3(s.base.request())?;
    for key in [
        "targets",
        "temperature_tick_scale",
        "optional_snapshot_checkpoint",
        "snapshot_policy",
    ] {
        value.as_object_mut().unwrap().remove(key);
    }
    value["schema_version"] = 5.into();
    value["contract"] = CONTRACT_V5.into();
    value["request"] = serde_json::to_value(r)?;
    value["initial_beta"] = r.restart.progress.beta.into();
    value["initial_progress"] = serde_json::to_value(&r.restart.progress)?;
    value["initial_cap"] = serde_json::to_value(&r.restart.cap)?;
    value["remaining_steps"] = 964.into();
    value["sample_points"] = serde_json::to_value(s.points())?;
    value["planned_samples"] = 483.into();
    value["new_measurements"] = 482.into();
    value["cached_samples"] = 1.into();
    value["measurement_policy"] = MEASUREMENT_POLICY.into();
    value["recording_policy"] = RECORDING_POLICY.into();
    value["failure_recording"] = json!({"kind":"none"});
    value["output_state_retained"] = false.into();
    Ok(value)
}
