//! Explicit continuation from supported retained cap256 target snapshots.
use super::v3::{
    preflight_v3, resolve_v3_request, AkltReferenceRequestV3, AkltReferenceSettingsV3,
};
use super::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

pub const CONTRACT_V4: &str = "aklt-projector-reference-resume-v4";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilePin {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeProgress {
    pub completed_steps: u64,
    pub beta: f64,
    pub accumulated_log_norm: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestartSpec {
    /// Relative to the new reference directory; no duplicated parent state.
    pub directory: String,
    pub snapshot_index: u64,
    pub files: BTreeMap<String, FilePin>,
    pub progress: ResumeProgress,
    pub cap: CapEvidence,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkltReferenceResumeRequest {
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
    pub restart: RestartSpec,
}
pub(super) fn resolve_resume_request(
    r: &AkltReferenceResumeRequest,
) -> Result<AkltReferenceSettingsV3> {
    let contract_name = r.contract.as_str();
    ensure!(
        r.schema_version == 4 && contract_name == CONTRACT_V4,
        "resume requires explicit schema4 contract"
    );
    let s = &r.restart;
    ensure!(
        !s.directory.is_empty() && !Path::new(&s.directory).is_absolute(),
        "restart directory must be relative to child"
    );
    let beta_target = match (s.progress.completed_steps, s.snapshot_index) {
        (154, 13) => 10. / 7.,
        (346, 3) => 10. / 3.,
        _ => anyhow::bail!("restart must identify the retained T0.7 or T0.3 boundary"),
    };
    ensure!(
        s.progress.beta.is_finite()
            && (s.progress.beta - beta_target).abs() < 1e-12
            && s.progress.accumulated_log_norm.is_finite(),
        "restart progress differs from supported boundary"
    );
    ensure!(
        s.files.len() == 4
            && [
                "conditions.json",
                "samples.csv",
                "states.h5",
                "summary.json"
            ]
            .iter()
            .all(|k| s.files.contains_key(*k)),
        "restart requires all four parent file pins"
    );
    for pin in s.files.values() {
        ensure!(
            pin.bytes > 0
                && pin.sha256.len() == 64
                && pin
                    .sha256
                    .bytes()
                    .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v)),
            "invalid parent file pin"
        );
    }
    ensure!(
        s.cap.coverage == CapCoverage::CompleteThroughStop
            && s.cap.observed_strang_steps == s.progress.completed_steps
            && s.cap
                .maximum_dimension_seen
                .is_some_and(|v| (1..=256).contains(&v))
            && s.cap.ever_reached == s.cap.maximum_dimension_seen.map(|v| v >= 256),
        "restart cap evidence must end at saved boundary"
    );
    let base = AkltReferenceRequestV3 {
        schema_version: 3,
        contract: v3::CONTRACT_V3.into(),
        model: r.model.clone(),
        variant: r.variant,
        max_bond: r.max_bond,
        max_seconds: r.max_seconds,
        recording_policy: r.recording_policy.clone(),
        heat_capacity_policy: r.heat_capacity_policy.clone(),
        source_revision: r.source_revision.clone(),
        worker_sha256: r.worker_sha256.clone(),
    };
    Ok(resolve_v3_request(&base)?.with_resume(r.clone()))
}
pub(super) fn target_offset(r: &AkltReferenceResumeRequest) -> usize {
    match (r.restart.progress.completed_steps, r.restart.snapshot_index) {
        (154, 13) => 14,
        (346, 3) => 18,
        _ => unreachable!("settings require a validated restart boundary"),
    }
}
pub fn preflight_resume(r: &AkltReferenceResumeRequest) -> Result<Value> {
    let settings = resolve_resume_request(r)?;
    let mut p = preflight_v3(settings.request())?;
    p["schema_version"] = 4.into();
    p["contract"] = CONTRACT_V4.into();
    p["request"] = serde_json::to_value(r)?;
    p["initial_beta"] = r.restart.progress.beta.into();
    p["initial_progress"] = serde_json::to_value(&r.restart.progress)?;
    p["targets"] = serde_json::to_value(settings.targets())?;
    p["remaining_steps"] = (1014 - r.restart.progress.completed_steps).into();
    p["snapshot_policy"]["maximum_count"] = settings.targets().len().into();
    p["failure_recording"] =
        json!({"kind":"first-svd-matrix-v1","file":"../svd-failure.h5","maximum_count":1});
    Ok(p)
}
pub(super) fn load_restart(
    settings: &AkltReferenceSettingsV3,
    directory: &Path,
) -> Result<crate::reference_trajectory::ScheduledStart> {
    let r = &settings
        .resume
        .as_ref()
        .context("missing resume request")?
        .restart;
    // Resolve before reserving child: child/../parent cannot be opened while
    // child is absent. Canonicalize the existing output parent for macOS /var.
    let directory = std::path::absolute(directory)?;
    let base = directory
        .parent()
        .context("missing output parent")?
        .canonicalize()?;
    let joined = base
        .join(directory.file_name().context("missing output name")?)
        .join(&r.directory);
    let mut parent = std::path::PathBuf::new();
    for part in joined.components() {
        match part {
            std::path::Component::ParentDir => {
                ensure!(parent.pop(), "restart path escapes root");
            }
            std::path::Component::CurDir => {}
            other => parent.push(other.as_os_str()),
        }
    }
    for ancestor in parent.ancestors() {
        if ancestor.exists() {
            ensure!(!ancestor.is_symlink(), "symlink restart directory");
        }
    }
    for (name, pin) in &r.files {
        ensure!(
            super::v3_output::fingerprint(&parent.join(name))? == serde_json::to_value(pin)?,
            "restart parent file pin changed: {name}"
        );
    }
    let parent_conditions: Value =
        serde_json::from_slice(&std::fs::read(parent.join("conditions.json"))?)?;
    let cold = r.progress.completed_steps == 346;
    let (parent_schema, parent_contract, count, boundary_index) = if cold {
        (4, CONTRACT_V4, 4, 3)
    } else {
        (3, v3::CONTRACT_V3, 14, 13)
    };
    ensure!(
        parent_conditions["contract"] == parent_contract
            && parent_conditions["schema_version"] == parent_schema,
        "restart parent contract differs from supported boundary"
    );
    if cold {
        // Bound the lineage before recursive verification: T0.3 -> T0.7 -> original v3.
        // Arbitrary/self-referencing resume chains are not supported.
        let parent_request: AkltReferenceResumeRequest =
            serde_json::from_value(parent_conditions["request"].clone())?;
        resolve_resume_request(&parent_request)?;
        ensure!(
            parent_request.restart.progress.completed_steps == 154,
            "T0.3 parent must be the T0.7 continuation"
        );
    }
    let verified =
        super::v3_output::verify_directory(&parent).context("verify pinned parent reference")?;
    ensure!(
        verified["contract"] == parent_contract
            && verified["snapshot_count"] == count
            && verified["available_samples"] == count
            && verified["available_heat_samples"] == count,
        "restart requires the complete retained parent prefix"
    );
    let rows = super::v3_output::read_rows(&parent)?;
    let boundary = &rows[boundary_index];
    ensure!(
        super::v3_output::cap_from(boundary)? == r.cap,
        "restart cap differs from saved boundary"
    );
    let loaded = crate::itebd_checkpoint::load_itebd_checkpoint(
        &parent.join("states.h5"),
        r.snapshot_index,
        &Default::default(),
    )?;
    ensure!(
        loaded.progress.completed_steps == r.progress.completed_steps
            && loaded.progress.beta.to_bits() == r.progress.beta.to_bits()
            && loaded.progress.accumulated_log_norm.to_bits()
                == r.progress.accumulated_log_norm.to_bits(),
        "restart progress differs from saved state"
    );
    ensure!(
        loaded.time_grid
            == crate::itebd_checkpoint::ItebdTimeGrid::Segmented(settings.full_segments().to_vec()),
        "restart time grid changed"
    );
    for (name, pin) in &r.files {
        ensure!(
            super::v3_output::fingerprint(&parent.join(name))? == serde_json::to_value(pin)?,
            "restart parent changed during loading: {name}"
        );
    }
    Ok(crate::reference_trajectory::ScheduledStart {
        state: loaded.state,
        progress: loaded.progress,
        cap: r.cap.clone(),
    })
}
pub fn run_to_directory(r: &AkltReferenceResumeRequest, directory: &Path) -> Result<Value> {
    let start = std::time::Instant::now();
    let settings = resolve_resume_request(r)?;
    super::v3_output::publish_with_engine(
        &settings,
        directory,
        &mut crate::reference_trajectory::NativeEngine(v3::measure_v3_reference),
        &mut || start.elapsed().as_secs_f64(),
    )
}
pub fn verify_failure_file(path: &Path) -> Result<Value> {
    crate::svd_failure::inspect(path)
}
