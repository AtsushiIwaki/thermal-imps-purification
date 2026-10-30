//! Native target snapshots and compact v3 scalar publication.
use super::v3::*;
use super::{AkltSampleValues, CapCoverage, CapEvidence};
use crate::config::{TrotterOrder, TruncationCfg};
use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
use crate::itebd_checkpoint::*;
use crate::reference_output::{atomic_write, exact_state, parse_csv, quote_csv};
use crate::reference_trajectory::{
    run_scheduled_observed_from, Engine, NativeEngine, ReferenceTarget, RunPolicy, ScheduledSample,
    TargetObserver, TrajectorySettings,
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

const EXTRA: &[&str] = &[
    "snapshot_available",
    "snapshot_index",
    "snapshot_reason",
    "heat_available",
    "heat_reason",
    "heat_capacity_variance",
    "heat_diagnostics",
];
const TEXT: &[&str] = &["reason", "cap_coverage", "snapshot_reason", "heat_reason"];
const PHYSICAL: &[&str] = &[
    "energy",
    "mx",
    "mz",
    "free_energy",
    "entropy",
    "parity_delta_energy",
    "parity_delta_mx",
    "parity_delta_mz",
    "direct_energy_difference",
    "maximum_rdm_fixed_point_residual",
    "maximum_rdm_trace_residual",
    "maximum_rdm_hermiticity_residual",
    "maximum_rdm_iterations",
    "minimum_rdm_eigenvalue",
    "bond_dimension_ab",
    "bond_dimension_ba",
    "cap_reached_now",
];
fn columns() -> Vec<&'static str> {
    super::output::COLUMNS
        .split(',')
        .chain(EXTRA.iter().copied())
        .collect()
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.)
}
fn number(row: &Value, key: &str) -> Result<f64> {
    let n = row[key]
        .as_f64()
        .with_context(|| format!("missing numeric {key}"))?;
    ensure!(n.is_finite(), "nonfinite {key}");
    Ok(n)
}
fn integer(row: &Value, key: &str) -> Result<u64> {
    row[key]
        .as_u64()
        .with_context(|| format!("invalid integer {key}"))
}
fn boolean(row: &Value, key: &str) -> Result<bool> {
    row[key]
        .as_bool()
        .with_context(|| format!("invalid boolean {key}"))
}
fn reason(row: &Value, key: &str) -> bool {
    row[key].as_str().is_some_and(|v| !v.trim().is_empty())
}
fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .context("parse reference JSON")
}
pub(super) fn fingerprint(path: &Path) -> Result<Value> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "not a regular reference file: {}",
        path.display()
    );
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok(json!({"sha256":format!("{:x}",hash.finalize()),"bytes":bytes}))
}
fn files(directory: &Path) -> Result<Value> {
    let mut out = json!({});
    for name in ["conditions.json", "samples.csv", "states.h5"] {
        if directory.join(name).exists() {
            out[name] = fingerprint(&directory.join(name))?;
        }
    }
    Ok(out)
}
fn membership(directory: &Path, summary: bool) -> Result<()> {
    ensure!(
        fs::symlink_metadata(directory)?.file_type().is_dir(),
        "reference directory must not be a symlink"
    );
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 member"))?;
        ensure!(
            [
                "conditions.json",
                "samples.csv",
                "summary.json",
                "states.h5"
            ]
            .contains(&name.as_str())
                && entry.file_type()?.is_file(),
            "unexpected/nonregular reference member {name}"
        );
        names.insert(name);
    }
    ensure!(
        names.contains("conditions.json")
            && names.contains("samples.csv")
            && (!summary || names.contains("summary.json")),
        "missing required v3 reference member"
    );
    Ok(())
}
fn empty_rows(settings: &AkltReferenceSettingsV3) -> Vec<Value> {
    settings
        .targets()
        .iter()
        .map(|t| {
            let mut row: Value = columns()
                .iter()
                .map(|key| ((*key).to_owned(), Value::Null))
                .collect();
            for key in ["available", "snapshot_available", "heat_available"] {
                row[key] = false.into();
            }
            for key in ["completed_steps", "cap_observed_strang_steps"] {
                row[key] = 0.into();
            }
            for key in ["beta_actual", "accumulated_log_norm"] {
                row[key] = 0.0.into();
            }
            for key in ["reason", "snapshot_reason", "heat_reason"] {
                row[key] = "target not reached".into();
            }
            row["temperature_tick"] = t.temperature_tick.into();
            row["temperature"] = t.temperature.into();
            row["beta_target"] = t.beta.into();
            row["cap_coverage"] = "unobserved_tail".into();
            if let Some(r) = &settings.resume {
                row["completed_steps"] = r.restart.progress.completed_steps.into();
                row["beta_actual"] = r.restart.progress.beta.into();
                row["accumulated_log_norm"] = r.restart.progress.accumulated_log_norm.into();
                put_cap(&mut row, &r.restart.cap).expect("validated restart cap");
            }
            row
        })
        .collect()
}
fn write_rows(directory: &Path, rows: &[Value]) -> Result<()> {
    let columns = columns();
    let mut text = columns.join(",") + "\n";
    for row in rows {
        let fields = columns
            .iter()
            .map(|key| match &row[*key] {
                Value::Null => String::new(),
                Value::String(s) => quote_csv(s),
                v if *key == "heat_diagnostics" => quote_csv(&v.to_string()),
                v => v.to_string(),
            })
            .collect::<Vec<_>>();
        text.push_str(&fields.join(","));
        text.push('\n');
    }
    atomic_write(&directory.join("samples.csv"), text.as_bytes())
}
pub(super) fn read_rows(directory: &Path) -> Result<Vec<Value>> {
    let parsed = parse_csv(&fs::read_to_string(directory.join("samples.csv"))?)?;
    let expected = columns();
    ensure!(
        parsed
            .first()
            .is_some_and(|h| h.iter().map(String::as_str).eq(expected.iter().copied())),
        "v3 CSV columns mismatch"
    );
    parsed
        .iter()
        .skip(1)
        .map(|fields| {
            ensure!(fields.len() == expected.len(), "malformed v3 CSV row");
            let mut row = json!({});
            for (key, value) in expected.iter().zip(fields) {
                row[*key] = if value.is_empty() {
                    Value::Null
                } else if TEXT.contains(key) {
                    value.clone().into()
                } else {
                    serde_json::from_str(value).with_context(|| format!("invalid CSV {key}"))?
                };
            }
            Ok(row)
        })
        .collect()
}
fn metadata(s: &AkltReferenceSettingsV3) -> ItebdRunMetadata {
    ItebdRunMetadata {
        dtau: 0.005,
        trotter_order: TrotterOrder::Second,
        truncation: TruncationCfg {
            epsilon: 1e-13,
            max_bond: Some(256),
        },
        canonicalize_every: 1,
        record_every_beta: None,
        model_label: Some("aklt_projector".into()),
        git_revision: Some(s.request().source_revision.clone()),
        hermiticity_tolerance: 1e-12,
    }
}
fn read_settings(directory: &Path) -> Result<(AkltReferenceSettingsV3, Value)> {
    let mut conditions = read_json(&directory.join("conditions.json"))?;
    let runtime = conditions
        .as_object_mut()
        .context("conditions must be object")?
        .remove("runtime_at_launch")
        .context("missing launch runtime")?;
    let r: RuntimeEvidence = serde_json::from_value(runtime.clone())?;
    let keys = r
        .environment
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ensure!(
        keys == [
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"
        ]
        .into_iter()
        .collect()
            && r.thread_hint > 0
            && r.visible_cpu_count > 0
            && !r.provider.is_empty(),
        "invalid recorded runtime"
    );
    let settings = if conditions["contract"] == super::v4::CONTRACT_V4 {
        super::v4::resolve_resume_request(&serde_json::from_value(conditions["request"].clone())?)?
    } else {
        resolve_v3_request(&serde_json::from_value::<AkltReferenceRequestV3>(
            conditions["request"].clone(),
        )?)?
    };
    ensure!(
        conditions == settings.preflight()?,
        "v3 conditions/preflight identity mismatch"
    );
    Ok((settings, runtime))
}
pub(super) fn cap_from(row: &Value) -> Result<CapEvidence> {
    Ok(serde_json::from_value(
        json!({"maximum_dimension_seen":row["maximum_dimension_seen"],"ever_reached":row["cap_ever_reached"],
        "observed_strang_steps":row["cap_observed_strang_steps"],"coverage":row["cap_coverage"]}),
    )?)
}
fn put_cap(row: &mut Value, cap: &CapEvidence) -> Result<()> {
    row["maximum_dimension_seen"] = serde_json::to_value(cap.maximum_dimension_seen)?;
    row["cap_ever_reached"] = serde_json::to_value(cap.ever_reached)?;
    row["cap_observed_strang_steps"] = cap.observed_strang_steps.into();
    row["cap_coverage"] = serde_json::to_value(cap.coverage)?;
    Ok(())
}
fn observed_cap(rows: &[Value]) -> Result<CapEvidence> {
    let mut cap = CapEvidence {
        maximum_dimension_seen: None,
        ever_reached: None,
        observed_strang_steps: 0,
        coverage: CapCoverage::UnobservedTail,
    };
    for row in rows {
        let r = cap_from(row)?;
        cap.maximum_dimension_seen = cap.maximum_dimension_seen.max(r.maximum_dimension_seen);
        cap.observed_strang_steps = cap.observed_strang_steps.max(r.observed_strang_steps);
        if r.ever_reached == Some(true) {
            cap.ever_reached = Some(true);
        }
    }
    Ok(cap)
}
struct Publisher<'a> {
    settings: &'a AkltReferenceSettingsV3,
    directory: PathBuf,
    rows: Vec<Value>,
    writer: Option<ItebdTrajectoryWriter>,
    #[cfg(test)]
    fail_after_append: bool,
}
impl<'a> Publisher<'a> {
    fn create(settings: &'a AkltReferenceSettingsV3, directory: &Path) -> Result<Self> {
        fs::create_dir(directory)
            .with_context(|| format!("reserve exclusive v3 directory {}", directory.display()))?;
        let mut conditions = settings.preflight()?;
        conditions["runtime_at_launch"] = serde_json::to_value(capture_runtime()?)?;
        atomic_write(
            &directory.join("conditions.json"),
            &serde_json::to_vec_pretty(&conditions)?,
        )?;
        let rows = empty_rows(settings);
        write_rows(directory, &rows)?;
        Ok(Self {
            settings,
            directory: directory.into(),
            rows,
            writer: None,
            #[cfg(test)]
            fail_after_append: false,
        })
    }
}
impl TargetObserver<AkltValuesV3> for Publisher<'_> {
    fn before_measurement(
        &mut self,
        target: &ReferenceTarget,
        state: &ItebdState,
        progress: &ItebdCheckpointProgress,
    ) -> Result<()> {
        let i = self
            .rows
            .iter()
            .position(|r| r["temperature_tick"] == target.temperature_tick)
            .context("unknown target snapshot")?;
        let index = self
            .rows
            .iter()
            .filter(|r| r["snapshot_available"] == true)
            .count() as u64;
        let dims = crate::itebd_state_view::ItebdStateRef::from(state)
            .validate()?
            .bond_dimensions;
        let mut cap = observed_cap(&self.rows)?;
        cap.maximum_dimension_seen = cap.maximum_dimension_seen.max(dims.into_iter().max());
        if cap.maximum_dimension_seen.is_some_and(|d| d >= 256) {
            cap.ever_reached = Some(true);
        }
        let row = &mut self.rows[i];
        row["beta_actual"] = progress.beta.into();
        row["completed_steps"] = progress.completed_steps.into();
        row["accumulated_log_norm"] = progress.accumulated_log_norm.into();
        row["snapshot_index"] = index.into();
        row["snapshot_reason"] = "snapshot write in progress".into();
        row["reason"] = "measurement not completed".into();
        row["heat_reason"] = "measurement not completed".into();
        put_cap(row, &cap)?;
        write_rows(&self.directory, &self.rows)?;
        let attempt = (|| -> Result<()> {
            if self.writer.is_none() {
                self.writer = Some(ItebdTrajectoryWriter::create_scheduled(
                    self.directory.join("states.h5"),
                    metadata(self.settings),
                    self.settings.hamiltonian(),
                    self.settings.full_segments().to_vec(),
                )?);
            }
            let entry = self
                .writer
                .as_mut()
                .unwrap()
                .append(state.into(), progress)
                .context("append target snapshot")?;
            ensure!(entry.index == index, "snapshot index sequence mismatch");
            #[cfg(test)]
            if self.fail_after_append {
                bail!("injected final flush uncertainty after snapshot append");
            }
            let loaded = load_itebd_checkpoint(
                &self.directory.join("states.h5"),
                entry.index,
                &Default::default(),
            )
            .context("read back target snapshot")?;
            exact_state(state, &loaded.state)?;
            ensure!(
                loaded.progress.beta.to_bits() == progress.beta.to_bits()
                    && loaded.progress.completed_steps == progress.completed_steps
                    && loaded.progress.accumulated_log_norm.to_bits()
                        == progress.accumulated_log_norm.to_bits(),
                "snapshot progress exact round-trip mismatch"
            );
            Ok(())
        })();
        if let Err(error) = attempt {
            self.rows[i]["snapshot_reason"] =
                format!("snapshot persistence failed: {error:#}").into();
            write_rows(&self.directory, &self.rows)
                .context("retain snapshot persistence failure")?;
            return Err(error).context("target snapshot publication failed");
        }
        self.rows[i]["snapshot_available"] = true.into();
        self.rows[i]["snapshot_reason"] = Value::Null;
        write_rows(&self.directory, &self.rows)
    }
    fn sample(&mut self, sample: &ScheduledSample<AkltValuesV3>) -> Result<()> {
        let row = self
            .rows
            .iter_mut()
            .find(|r| r["temperature_tick"] == sample.temperature_tick)
            .context("unknown sample target")?;
        row["beta_actual"] = sample.beta_actual.into();
        row["completed_steps"] = sample.completed_steps.into();
        row["accumulated_log_norm"] = sample.accumulated_log_norm.into();
        row["available"] = sample.available.into();
        row["reason"] = serde_json::to_value(&sample.reason)?;
        if let Some(cap) = &sample.cap {
            put_cap(row, cap)?;
        }
        if let Some(v) = &sample.values {
            let physical = serde_json::to_value(&v.physical)?;
            for key in PHYSICAL {
                row[*key] = physical[*key].clone();
            }
            row["heat_available"] = v.heat.available.into();
            row["heat_reason"] = serde_json::to_value(&v.heat.reason)?;
            row["heat_capacity_variance"] =
                serde_json::to_value(v.heat.report.as_ref().map(|h| h.specific_heat_per_site))?;
            row["heat_diagnostics"] = serde_json::to_value(&v.heat.report)?;
        } else {
            row["heat_reason"] = serde_json::to_value(&sample.reason)?;
        }
        write_rows(&self.directory, &self.rows)
    }
}
pub fn run_to_directory(request: &AkltReferenceRequestV3, directory: &Path) -> Result<Value> {
    let start = std::time::Instant::now();
    let settings = resolve_v3_request(request)?;
    publish_with_engine(
        &settings,
        directory,
        &mut NativeEngine(measure_v3_reference),
        &mut || start.elapsed().as_secs_f64(),
    )
}
pub(super) fn publish_with_engine(
    settings: &AkltReferenceSettingsV3,
    directory: &Path,
    engine: &mut dyn Engine<AkltValuesV3>,
    elapsed: &mut dyn FnMut() -> f64,
) -> Result<Value> {
    // Validate/pin the source before reserving a destination or mutating a state.
    let initial = if settings.resume.is_some() {
        Some(super::v4::load_restart(settings, directory)?)
    } else {
        None
    };
    let _capture = if settings.resume.is_some() {
        Some(crate::svd_failure::activate(
            directory,
            &settings.request().source_revision,
            &settings.request().worker_sha256,
        )?)
    } else {
        None
    };
    let mut publisher = Publisher::create(settings, directory)?;
    let outcome = run_scheduled_observed_from(
        TrajectorySettings {
            hamiltonian: settings.hamiltonian(),
            truncation: settings.truncation(),
            segments: settings.segments(),
            targets: settings.targets(),
            policy: RunPolicy::Aklt {
                max_seconds: None,
                bond_cap: 256,
            },
        },
        &mut publisher,
        engine,
        elapsed,
        initial,
    )?;
    if let Some(writer) = publisher.writer.take() {
        writer.finish().context("finish target snapshot file")?;
    }
    let runtime = read_json(&directory.join("conditions.json"))?["runtime_at_launch"].clone();
    let mut summary = make_summary(
        settings,
        &publisher.rows,
        &outcome.reason,
        &outcome.status,
        &outcome.progress,
        "exact",
        Some(outcome.elapsed_seconds),
        outcome.cap.context("missing cap evidence")?,
        runtime,
    )?;
    summary["files"] = files(directory)?;
    let verified = inspect(directory, summary.clone())?;
    atomic_write(
        &directory.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(verified)
}
fn prefix_beta(settings: &AkltReferenceSettingsV3, steps: u64) -> Result<f64> {
    let mut remaining = steps;
    let mut beta = 0.;
    for segment in settings.full_segments() {
        let n = remaining.min(segment.steps);
        beta += 2. * n as f64 * segment.dtau;
        remaining -= n;
    }
    ensure!(remaining == 0, "progress exceeds reference schedule");
    Ok(beta)
}
fn validate_cap(cap: &CapEvidence, steps: u64) -> Result<()> {
    ensure!(
        cap.maximum_dimension_seen
            .is_none_or(|d| (1..=256).contains(&d))
            && cap.observed_strang_steps <= steps + 1,
        "invalid cap dimensions/observations"
    );
    ensure!(
        (cap.ever_reached == Some(true)) == cap.maximum_dimension_seen.is_some_and(|d| d >= 256),
        "cap hit contradicts observed maximum"
    );
    if cap.maximum_dimension_seen.is_none() {
        ensure!(
            cap.observed_strang_steps == 0,
            "cap steps without dimensions"
        );
    }
    match cap.coverage {
        CapCoverage::CompleteThroughStop => ensure!(
            cap.maximum_dimension_seen.is_some()
                && cap.ever_reached.is_some()
                && cap.observed_strang_steps == steps,
            "incomplete cap coverage"
        ),
        CapCoverage::UnobservedTail => ensure!(
            cap.ever_reached != Some(false),
            "unknown cap tail cannot certify no hit"
        ),
    };
    Ok(())
}
fn validate_heat(row: &Value) -> Result<()> {
    if !boolean(row, "heat_available")? {
        ensure!(
            reason(row, "heat_reason")
                && row["heat_capacity_variance"].is_null()
                && row["heat_diagnostics"].is_null(),
            "missing variance reason or contradictory unavailable heat"
        );
        return Ok(());
    }
    ensure!(
        boolean(row, "available")? && row["heat_reason"].is_null(),
        "variance requires physical values and no failure reason"
    );
    let h: VarianceHeatRecord = serde_json::from_value(row["heat_diagnostics"].clone())
        .context("invalid variance diagnostic schema")?;
    let terms = [
        h.onsite_contribution,
        h.positive_direction_contribution,
        h.negative_direction_contribution,
        h.parity_a_contribution,
        h.parity_b_contribution,
    ];
    ensure!(
        terms.iter().flatten().all(|x| x.is_finite())
            && [
                h.raw_energy_variance_per_site,
                h.energy_variance_per_site,
                h.specific_heat_per_site,
                h.last_positive_shell_magnitude,
                h.last_negative_shell_magnitude,
                h.max_imaginary_residual
            ]
            .iter()
            .all(|x| x.is_finite()),
        "nonfinite variance diagnostics"
    );
    let sum = [
        terms[0][0] + terms[1][0] + terms[2][0],
        terms[0][1] + terms[1][1] + terms[2][1],
    ];
    ensure!(
        close(sum[0], h.raw_energy_variance_per_site)
            && close(terms[3][0] + terms[4][0], sum[0])
            && close(terms[3][1] + terms[4][1], sum[1]),
        "variance decomposition mismatch"
    );
    let tol = 1e-10 * sum[0].hypot(sum[1]).max(1.);
    ensure!(
        sum[1].abs() <= tol
            && h.energy_variance_per_site >= 0.
            && (if h.raw_energy_variance_per_site < 0. {
                h.raw_energy_variance_per_site.abs() <= tol && h.energy_variance_per_site == 0.
            } else {
                close(h.energy_variance_per_site, h.raw_energy_variance_per_site)
            }),
        "variance reality/projection mismatch"
    );
    ensure!(
        (4..=200).contains(&h.max_distance)
            && h.stop_reason == "consecutive_small_shells"
            && h.last_positive_shell_magnitude >= 0.
            && h.last_negative_shell_magnitude >= 0.
            && h.max_imaginary_residual >= 0.,
        "invalid variance stopping diagnostics"
    );
    let beta = number(row, "beta_actual")?;
    ensure!(
        close(
            h.specific_heat_per_site,
            beta * beta * h.energy_variance_per_site
        ) && close(
            h.specific_heat_per_site,
            number(row, "heat_capacity_variance")?
        ),
        "variance beta-squared normalization mismatch"
    );
    Ok(())
}
fn validate_rows(settings: &AkltReferenceSettingsV3, rows: &[Value]) -> Result<()> {
    ensure!(
        rows.len() == settings.targets().len(),
        "reference target row count mismatch"
    );
    let mut previous: Option<CapEvidence> = settings.resume.as_ref().map(|r| r.restart.cap.clone());
    for (row, target) in rows.iter().zip(settings.targets()) {
        (|| -> Result<()> {
            ensure!(
                integer(row, "temperature_tick")? == u64::from(target.temperature_tick)
                    && number(row, "temperature")?.to_bits() == target.temperature.to_bits()
                    && number(row, "beta_target")?.to_bits() == target.beta.to_bits(),
                "target identity mismatch"
            );
            let steps = integer(row, "completed_steps")?;
            ensure!(
                steps
                    >= settings
                        .resume
                        .as_ref()
                        .map_or(0, |r| r.restart.progress.completed_steps)
                    && steps <= target.completed_steps
                    && close(number(row, "beta_actual")?, prefix_beta(settings, steps)?),
                "target progress/grid mismatch"
            );
            let norm = number(row, "accumulated_log_norm")?;
            if let Some(resume) = &settings.resume {
                let saved = &resume.restart.progress;
                if steps == saved.completed_steps {
                    ensure!(
                        number(row, "beta_actual")?.to_bits() == saved.beta.to_bits()
                            && norm.to_bits() == saved.accumulated_log_norm.to_bits(),
                        "restart boundary beta/norm changed"
                    );
                }
            }
            let cap = cap_from(row)?;
            validate_cap(&cap, steps)?;
            if let Some(before) = previous
                .as_ref()
                .filter(|_| cap.maximum_dimension_seen.is_some() || settings.resume.is_some())
            {
                ensure!(
                    cap.maximum_dimension_seen >= before.maximum_dimension_seen
                        && cap.observed_strang_steps >= before.observed_strang_steps
                        && (before.ever_reached != Some(true) || cap.ever_reached == Some(true)),
                    "cap history below restart boundary or earlier observations"
                );
            }
            if cap.maximum_dimension_seen.is_some() {
                previous = Some(cap.clone());
            }
            if boolean(row, "snapshot_available")? {
                ensure!(
                    row["snapshot_index"].as_u64().is_some()
                        && row["snapshot_reason"].is_null()
                        && steps == target.completed_steps,
                    "snapshot target status mismatch"
                );
            } else {
                ensure!(
                    reason(row, "snapshot_reason"),
                    "unavailable snapshot needs reason"
                );
                if !row["snapshot_index"].is_null() {
                    integer(row, "snapshot_index")?;
                }
            }
            if boolean(row, "available")? {
                ensure!(
                    boolean(row, "snapshot_available")?
                        && row["reason"].is_null()
                        && steps == target.completed_steps,
                    "physical availability/snapshot mismatch"
                );
                let v: AkltSampleValues = serde_json::from_value(
                    PHYSICAL
                        .iter()
                        .map(|key| ((*key).to_string(), row[*key].clone()))
                        .collect::<Value>(),
                )?;
                v.validate()?;
                let beta = number(row, "beta_actual")?;
                ensure!(
                    close(
                        v.free_energy,
                        -(3f64.ln() + number(row, "accumulated_log_norm")? / 2.) / beta
                    ) && close(v.entropy, (v.energy - v.free_energy) * beta),
                    "physical thermodynamic identity mismatch"
                );
                ensure!(
                    (-1e-10..=1. + 1e-10).contains(&v.energy)
                        && v.mx.abs() <= 1. + 1e-10
                        && v.mz.abs() <= 1. + 1e-10,
                    "spin-one observable bounds"
                );
                ensure!(
                    (0.0..=1e-12).contains(&v.maximum_rdm_fixed_point_residual)
                        && (0.0..=1e-10).contains(&v.maximum_rdm_trace_residual)
                        && (0.0..=1e-10).contains(&v.maximum_rdm_hermiticity_residual)
                        && v.minimum_rdm_eigenvalue >= -1e-10
                        && v.maximum_rdm_iterations <= 10000
                        && v.bond_dimension_ab <= 256
                        && v.bond_dimension_ba <= 256,
                    "invalid RDM diagnostics"
                );
                let dimension = v.bond_dimension_ab.max(v.bond_dimension_ba);
                ensure!(
                    v.cap_reached_now == (dimension >= 256)
                        && cap.maximum_dimension_seen.is_some_and(|d| d >= dimension)
                        && cap.coverage == CapCoverage::CompleteThroughStop,
                    "physical cap dimensions mismatch"
                );
            } else {
                ensure!(
                    reason(row, "reason") && PHYSICAL.iter().all(|key| row[*key].is_null()),
                    "unavailable physical values/reason mismatch"
                );
            }
            validate_heat(row)
        })()
        .with_context(|| format!("v3 target {}", target.temperature_tick))?;
    }
    Ok(())
}
fn uncertain_index(rows: &[Value]) -> Result<Option<u64>> {
    let mut found = None;
    for row in rows {
        if row["snapshot_reason"]
            .as_str()
            .is_some_and(|r| r.starts_with("snapshot persistence failed"))
        {
            ensure!(
                found.is_none() && !boolean(row, "snapshot_available")?,
                "multiple/contradictory uncertain snapshots"
            );
            found = Some(integer(row, "snapshot_index")?);
        }
    }
    Ok(found)
}
fn snapshot_entries(
    directory: &Path,
    settings: &AkltReferenceSettingsV3,
    rows: &[Value],
    uncertain: Option<u64>,
) -> Result<Vec<Value>> {
    let path = directory.join("states.h5");
    if !path.exists() {
        ensure!(
            !rows.iter().any(|r| r["snapshot_available"] == true),
            "required snapshot checkpoint missing"
        );
        return Ok(Vec::new());
    }
    let names = {
        let file = hdf5_metno::File::open(&path).context("open snapshot checkpoint")?;
        file.group("states")?.member_names()?
    };
    let mut complete = BTreeSet::new();
    let mut all = BTreeSet::new();
    for name in names {
        let index = name.parse::<u64>().context("non-numeric snapshot group")?;
        ensure!(
            index < settings.targets().len() as u64
                && name == format!("{index:06}")
                && all.insert(index),
            "invalid/extra snapshot index"
        );
        let marker = {
            let file = hdf5_metno::File::open(&path)?;
            let group = file.group(&format!("states/{name}"))?;
            if group.attr_names()?.iter().any(|n| n == "complete") {
                Some(group.attr("complete")?.read_scalar::<u8>()?)
            } else {
                None
            }
        };
        ensure!(
            marker.is_none_or(|m| m <= 1),
            "invalid snapshot completion marker"
        );
        if marker == Some(1) {
            complete.insert(index);
        }
    }
    if let Some(i) = uncertain {
        ensure!(
            all.iter().all(|n| *n <= i),
            "snapshots after uncertain write"
        );
    }
    let usable = complete
        .iter()
        .copied()
        .filter(|i| uncertain.is_none_or(|u| *i < u))
        .collect::<Vec<_>>();
    ensure!(
        usable.iter().copied().eq(0..usable.len() as u64),
        "snapshot prefix has a gap"
    );
    let mut entries = Vec::new();
    for index in usable {
        let loaded = load_itebd_checkpoint(&path, index, &Default::default())
            .context("validate native target snapshot")?;
        let target = &settings.targets()[index as usize];
        let p = &loaded.progress;
        let m = &loaded.metadata;
        ensure!(
            p.completed_steps == target.completed_steps
                && close(p.beta, target.beta)
                && loaded.time_grid == ItebdTimeGrid::Segmented(settings.full_segments().to_vec()),
            "snapshot target/grid mismatch"
        );
        ensure!(
            m.dtau == 0.005
                && m.trotter_order == TrotterOrder::Second
                && m.truncation.epsilon == 1e-13
                && m.truncation.max_bond == Some(256)
                && m.canonicalize_every == 1
                && m.record_every_beta.is_none()
                && m.model_label.as_deref() == Some("aklt_projector")
                && m.git_revision.as_deref() == Some(settings.request().source_revision.as_str())
                && m.hermiticity_tolerance == 1e-12,
            "snapshot model/source/settings mismatch"
        );
        match (&loaded.hamiltonian, settings.hamiltonian()) {
            (ItebdHamiltonian::Real(a), ItebdHamiltonian::Real(b)) => ensure!(
                a.two_site_h == b.two_site_h && a.site_energy == b.site_energy,
                "snapshot Hamiltonian mismatch"
            ),
            _ => bail!("snapshot backend mismatch"),
        }
        let row = &rows[index as usize];
        ensure!(
            row["snapshot_index"] == index
                && row["snapshot_available"] == true
                && number(row, "beta_actual")?.to_bits() == p.beta.to_bits()
                && integer(row, "completed_steps")? == p.completed_steps
                && number(row, "accumulated_log_norm")?.to_bits()
                    == p.accumulated_log_norm.to_bits(),
            "snapshot/scalar association mismatch"
        );
        if boolean(row, "available")? {
            let measured = super::measure_reference_with_cap(
                &loaded.state,
                settings.hamiltonian(),
                p.beta,
                p.accumulated_log_norm,
                256,
            )?;
            let values = serde_json::to_value(measured)?;
            for key in ["energy", "mx", "mz", "free_energy", "entropy"] {
                ensure!(
                    close(number(row, key)?, number(&values, key)?),
                    "snapshot physical scalar mismatch: {key}"
                );
            }
            ensure!(
                row["bond_dimension_ab"] == values["bond_dimension_ab"]
                    && row["bond_dimension_ba"] == values["bond_dimension_ba"],
                "snapshot bond dimensions mismatch"
            );
        }
        entries.push(json!({"temperature_tick":target.temperature_tick,"snapshot_index":index,"beta_actual":p.beta,"completed_steps":p.completed_steps,"accumulated_log_norm":p.accumulated_log_norm}));
    }
    ensure!(
        rows.iter()
            .filter(|r| r["snapshot_available"] == true)
            .count()
            == entries.len(),
        "published snapshot missing/incomplete"
    );
    Ok(entries)
}
fn make_summary(
    settings: &AkltReferenceSettingsV3,
    rows: &[Value],
    why: &str,
    status: &str,
    progress: &ItebdCheckpointProgress,
    kind: &str,
    elapsed: Option<f64>,
    cap: CapEvidence,
    runtime: Value,
) -> Result<Value> {
    let contract_name = settings.contract();
    Ok(
        json!({"schema_version":settings.schema_version(),"contract":contract_name,"request":settings.recorded_request()?,"status":status,"reason":why,
        "completed_steps":progress.completed_steps,"beta_actual":progress.beta,"accumulated_log_norm":progress.accumulated_log_norm,
        "completed_steps_kind":kind,"elapsed_seconds":elapsed,"budget_overrun_seconds":null,"cap":cap,
        "snapshot_count":rows.iter().filter(|r|r["snapshot_available"]==true).count(),
        "available_samples":rows.iter().filter(|r|r["available"]==true).count(),"available_heat_samples":rows.iter().filter(|r|r["heat_available"]==true).count(),
        "snapshot_mapping":rows.iter().filter(|r|r["snapshot_available"]==true).map(|r|json!({"temperature_tick":r["temperature_tick"],"snapshot_index":r["snapshot_index"],"beta_actual":r["beta_actual"],"completed_steps":r["completed_steps"],"accumulated_log_norm":r["accumulated_log_norm"]})).collect::<Vec<_>>(),
        "uncertain_snapshot_index":uncertain_index(rows)?,"runtime_at_launch":runtime,"files":{}}),
    )
}
pub fn verify_directory(directory: &Path) -> Result<Value> {
    membership(directory, true)?;
    inspect(directory, read_json(&directory.join("summary.json"))?)
}
fn inspect(directory: &Path, summary: Value) -> Result<Value> {
    membership(directory, false)?;
    let (settings, runtime) = read_settings(directory)?;
    if settings.resume.is_some() {
        super::v4::load_restart(&settings, directory)?;
    }
    let rows = read_rows(directory)?;
    validate_rows(&settings, &rows)?;
    let contract_name = settings.contract();
    ensure!(
        summary["schema_version"] == settings.schema_version()
            && summary["contract"] == contract_name
            && summary["request"] == settings.recorded_request()?
            && summary["runtime_at_launch"] == runtime,
        "v3 summary/request/runtime identity mismatch"
    );
    let fields = [
        "schema_version",
        "contract",
        "request",
        "status",
        "reason",
        "completed_steps",
        "beta_actual",
        "accumulated_log_norm",
        "completed_steps_kind",
        "elapsed_seconds",
        "budget_overrun_seconds",
        "cap",
        "snapshot_count",
        "available_samples",
        "available_heat_samples",
        "snapshot_mapping",
        "uncertain_snapshot_index",
        "runtime_at_launch",
        "files",
    ];
    ensure!(
        summary
            .as_object()
            .context("summary object")?
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == fields.into_iter().collect(),
        "v3 summary schema mismatch"
    );
    ensure!(
        ["complete", "partial"].contains(&summary["status"].as_str().unwrap_or(""))
            && reason(&summary, "reason")
            && ["exact", "retained_lower_bound"]
                .contains(&summary["completed_steps_kind"].as_str().unwrap_or("")),
        "invalid actual stop classification"
    );
    ensure!(
        summary["budget_overrun_seconds"].is_null()
            && (summary["elapsed_seconds"].is_null() || number(&summary, "elapsed_seconds")? >= 0.),
        "invalid unlimited timing"
    );
    let steps = integer(&summary, "completed_steps")?;
    ensure!(
        close(
            number(&summary, "beta_actual")?,
            prefix_beta(&settings, steps)?
        ) && rows
            .iter()
            .all(|r| r["completed_steps"].as_u64().is_some_and(|n| n <= steps)),
        "summary progress mismatch"
    );
    let retained_steps = rows
        .iter()
        .map(|r| integer(r, "completed_steps"))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("missing retained progress")?;
    let norm = number(&summary, "accumulated_log_norm")?;
    if let Some(resume) = &settings.resume {
        let saved = &resume.restart.progress;
        if steps == saved.completed_steps {
            ensure!(
                number(&summary, "beta_actual")?.to_bits() == saved.beta.to_bits()
                    && norm.to_bits() == saved.accumulated_log_norm.to_bits(),
                "summary restart boundary beta/norm changed"
            );
        }
    }
    ensure!(steps == retained_steps, "summary exceeds retained progress");
    ensure!(
        rows.iter()
            .filter(|r| r["completed_steps"].as_u64() == Some(steps))
            .all(|r| {
                close(
                    number(r, "beta_actual").unwrap_or(f64::NAN),
                    number(&summary, "beta_actual").unwrap_or(f64::NAN),
                ) && close(number(r, "accumulated_log_norm").unwrap_or(f64::NAN), norm)
            }),
        "summary beta/norm disagrees with retained progress"
    );
    let cap: CapEvidence = serde_json::from_value(summary["cap"].clone())?;
    validate_cap(&cap, steps)?;
    let observed = observed_cap(&rows)?;
    ensure!(
        cap.maximum_dimension_seen >= observed.maximum_dimension_seen
            && cap.observed_strang_steps >= observed.observed_strang_steps
            && (observed.ever_reached != Some(true) || cap.ever_reached == Some(true)),
        "summary loses cap evidence"
    );
    let pins = files(directory)?;
    ensure!(pins == summary["files"], "v3 file fingerprint mismatch");
    let uncertain = uncertain_index(&rows)?;
    ensure!(
        summary["uncertain_snapshot_index"] == serde_json::to_value(uncertain)?,
        "uncertain snapshot identity mismatch"
    );
    let entries = snapshot_entries(directory, &settings, &rows, uncertain)?;
    ensure!(
        summary["snapshot_mapping"] == serde_json::to_value(&entries)?
            && integer(&summary, "snapshot_count")? == entries.len() as u64
            && integer(&summary, "available_samples")?
                == rows.iter().filter(|r| r["available"] == true).count() as u64
            && integer(&summary, "available_heat_samples")?
                == rows.iter().filter(|r| r["heat_available"] == true).count() as u64,
        "summary acquisition counts/mapping mismatch"
    );
    if summary["status"] == "complete" {
        ensure!(
            entries.len() == settings.targets().len()
                && summary["available_samples"] == settings.targets().len()
                && summary["available_heat_samples"] == settings.targets().len()
                && steps == 1014
                && summary["completed_steps_kind"] == "exact"
                && uncertain.is_none(),
            "complete acquisition missing targets"
        );
    }
    ensure!(
        files(directory)? == pins,
        "v3 files changed during native verification"
    );
    let mut result = summary;
    result["sample_count"] = settings.targets().len().into();
    result["verified_snapshot_count"] = entries.len().into();
    result["checkpoint_verified"] = (!entries.is_empty()).into();
    result["scalar_state_consistency_passed"] = true.into();
    result["variance_remeasured"] = false.into();
    Ok(result)
}
pub fn finalize_interrupted_directory(directory: &Path, why: &str) -> Result<Value> {
    membership(directory, false)?;
    ensure!(!why.trim().is_empty(), "interruption reason required");
    ensure!(
        !directory.join("summary.json").exists(),
        "summary already exists; no interrupted finalization"
    );
    let (settings, runtime) = read_settings(directory)?;
    let mut rows = read_rows(directory)?;
    // A panic leaves later rows at their exact launch template. Recognize only
    // untouched placeholders; do not sanitize arbitrary changed beta/norm/cap.
    let templates = empty_rows(&settings);
    let pending: Vec<bool> = rows
        .iter()
        .zip(&templates)
        .map(|(row, seed)| {
            columns()
                .iter()
                .filter(|key| !["reason", "snapshot_reason", "heat_reason"].contains(key))
                .all(|key| row[*key] == seed[*key])
        })
        .collect();
    let mut recovered_dimensions = vec![None; rows.len()];
    let uncertain = uncertain_index(&rows)?;
    if directory.join("states.h5").exists() {
        let names = {
            let f = hdf5_metno::File::open(directory.join("states.h5"))?;
            f.group("states")?.member_names()?
        };
        for name in names {
            let index = name.parse::<usize>().context("invalid snapshot index")?;
            ensure!(
                index < settings.targets().len(),
                "extra interrupted snapshot"
            );
            if uncertain.is_some_and(|i| index as u64 >= i) {
                continue;
            }
            let marker = {
                let f = hdf5_metno::File::open(directory.join("states.h5"))?;
                let g = f.group(&format!("states/{name}"))?;
                if g.attr_names()?.iter().any(|n| n == "complete") {
                    Some(g.attr("complete")?.read_scalar::<u8>()?)
                } else {
                    None
                }
            };
            if marker != Some(1) {
                continue;
            }
            let loaded = load_itebd_checkpoint(
                &directory.join("states.h5"),
                index as u64,
                &Default::default(),
            )
            .context("recover completed native snapshot")?;
            if pending[index] && settings.resume.is_some() {
                let state = crate::itebd_state_view::ItebdStateRef::from(&loaded.state);
                recovered_dimensions[index] =
                    Some([state.lambda_ab().len(), state.lambda_ba().len()]);
            }
            let row = &mut rows[index];
            ensure!(
                loaded.progress.completed_steps == settings.targets()[index].completed_steps,
                "interrupted snapshot target mismatch"
            );
            row["snapshot_available"] = true.into();
            row["snapshot_index"] = (index as u64).into();
            row["snapshot_reason"] = Value::Null;
            row["beta_actual"] = loaded.progress.beta.into();
            row["completed_steps"] = loaded.progress.completed_steps.into();
            row["accumulated_log_norm"] = loaded.progress.accumulated_log_norm.into();
        }
    }
    if let Some(resume) = &settings.resume {
        let mut cap = resume.restart.cap.clone();
        let mut steps = resume.restart.progress.completed_steps;
        let mut beta = resume.restart.progress.beta;
        let mut norm = resume.restart.progress.accumulated_log_norm;
        for (index, row) in rows.iter_mut().enumerate() {
            if pending[index] {
                // A recovered HDF5 state certifies its dimensions, not the
                // missing inter-target cap history or unrecorded measurements.
                if let Some(dimensions) = recovered_dimensions[index] {
                    for dimension in dimensions {
                        cap.maximum_dimension_seen =
                            cap.maximum_dimension_seen.max(Some(dimension));
                        if dimension >= 256 {
                            cap.ever_reached = Some(true);
                        }
                    }
                } else {
                    row["completed_steps"] = steps.into();
                    row["beta_actual"] = beta.into();
                    row["accumulated_log_norm"] = norm.into();
                }
                cap.mark_unobserved();
                put_cap(row, &cap)?;
            }
            if integer(row, "completed_steps")? >= steps {
                steps = integer(row, "completed_steps")?;
                beta = number(row, "beta_actual")?;
                norm = number(row, "accumulated_log_norm")?;
            }
            cap = cap_from(row)?;
        }
    }
    for row in &mut rows {
        if row["available"] != true {
            row["reason"] = why.into();
            row["heat_reason"] = why.into();
        }
    }
    validate_rows(&settings, &rows)?;
    write_rows(directory, &rows)?;
    let last = rows
        .iter()
        .max_by_key(|r| r["completed_steps"].as_u64().unwrap_or(0))
        .context("missing rows")?;
    let progress = ItebdCheckpointProgress {
        beta: number(last, "beta_actual")?,
        completed_steps: integer(last, "completed_steps")?,
        accumulated_log_norm: number(last, "accumulated_log_norm")?,
        last_step: None,
    };
    let mut summary = make_summary(
        &settings,
        &rows,
        why,
        "partial",
        &progress,
        "retained_lower_bound",
        None,
        observed_cap(&rows)?,
        runtime,
    )?;
    summary["files"] = files(directory)?;
    let verified = inspect(directory, summary.clone())?;
    atomic_write(
        &directory.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itebd::StepInfo;
    use crate::reference_trajectory::{Engine, TargetObserver};

    fn settings() -> AkltReferenceSettingsV3 {
        resolve_v3_request(&serde_json::from_value(serde_json::json!({
            "schema_version":3,"contract":"aklt-projector-reference-v3","model":{"name":"aklt_projector"},
            "variant":"fine","max_bond":256,"max_seconds":null,"recording_policy":"target-snapshots-v1",
            "heat_capacity_policy":"variance-default-v1","source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)
        })).unwrap()).unwrap()
    }
    // Static analytical tensors with labeled progress are integrity fixtures,
    // not a thermal evolution or a finite-temperature physics expectation.
    struct FixtureEngine {
        steps: usize,
        measures: usize,
        fail_heat_at: Option<usize>,
    }
    impl Engine<AkltValuesV3> for FixtureEngine {
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &crate::tensor::Truncation,
        ) -> Result<StepInfo> {
            self.steps += 1;
            Ok(StepInfo {
                max_bond: 1,
                min_singular_value: 1.,
                log_norm: 0.,
            })
        }
        fn canonicalize(&mut self, _: &mut ItebdState, _: &ItebdHamiltonian) -> Result<f64> {
            Ok(0.)
        }
        fn measure(
            &mut self,
            s: &ItebdState,
            h: &ItebdHamiltonian,
            b: f64,
            n: f64,
        ) -> Result<AkltValuesV3> {
            self.measures += 1;
            let mut v = measure_v3_reference(s, h, b, n)?;
            if self.fail_heat_at == Some(self.measures) {
                v.heat = VarianceHeatEvidence {
                    available: false,
                    report: None,
                    reason: Some("variance tail fixture failure".into()),
                };
            }
            Ok(v)
        }
    }
    fn fixture(path: &Path, fail_heat_at: Option<usize>) -> Value {
        publish_with_engine(
            &settings(),
            path,
            &mut FixtureEngine {
                steps: 0,
                measures: 0,
                fail_heat_at,
            },
            &mut || 0.,
        )
        .unwrap()
    }
    struct ContinuationFixture {
        steps: usize,
        setups: usize,
        measures: usize,
        stop_after: Option<usize>,
        panic_after: Option<usize>,
    }
    impl Engine<AkltValuesV3> for ContinuationFixture {
        fn setup(&mut self, h: &ItebdHamiltonian) -> Result<ItebdState> {
            self.setups += 1;
            Ok(ItebdState::infinite_temperature(h)?)
        }
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &crate::tensor::Truncation,
        ) -> Result<StepInfo> {
            assert_ne!(
                self.panic_after,
                Some(self.steps),
                "fixture panic between target snapshots"
            );
            if self.stop_after == Some(self.steps) {
                bail!("fixture stop before next step");
            }
            self.steps += 1;
            Ok(StepInfo {
                max_bond: 1,
                min_singular_value: 1.,
                log_norm: -0.01,
            })
        }
        fn canonicalize(&mut self, _: &mut ItebdState, _: &ItebdHamiltonian) -> Result<f64> {
            Ok(0.)
        }
        fn measure(
            &mut self,
            s: &ItebdState,
            h: &ItebdHamiltonian,
            b: f64,
            n: f64,
        ) -> Result<AkltValuesV3> {
            self.measures += 1;
            measure_v3_reference(s, h, b, n)
        }
    }
    fn resume_fixture(parent: &Path) -> AkltReferenceSettingsV3 {
        let mut old = ContinuationFixture {
            steps: 0,
            setups: 0,
            measures: 0,
            stop_after: Some(154),
            panic_after: None,
        };
        let outcome = publish_with_engine(&settings(), &parent, &mut old, &mut || 0.).unwrap();
        assert_eq!(outcome["snapshot_count"], 14);
        assert_eq!(outcome["available_heat_samples"], 14);
        let rows = read_rows(&parent).unwrap();
        let boundary = &rows[13];
        let mut pins = files(&parent).unwrap();
        pins["summary.json"] = fingerprint(&parent.join("summary.json")).unwrap();
        let mut r = serde_json::to_value(settings().request()).unwrap();
        r["schema_version"] = 4.into();
        r["contract"] = super::super::v4::CONTRACT_V4.into();
        r["source_revision"] = "c".repeat(40).into();
        r["restart"] = json!({"directory":"../parent","snapshot_index":13,"files":pins,
            "progress":{"beta":boundary["beta_actual"],"completed_steps":154,"accumulated_log_norm":boundary["accumulated_log_norm"]},"cap":cap_from(boundary).unwrap()});
        super::super::v4::resolve_resume_request(&serde_json::from_value(r).unwrap()).unwrap()
    }
    // Losing the saved normalization/progress or replaying beta0 must fail this real-I/O check.
    #[test]
    fn resumed_publication_uses_parent_state_and_saves_only_six_new_snapshots() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("parent");
        let child = temp.path().join("child");
        let resumed = resume_fixture(&parent);
        let mut engine = ContinuationFixture {
            steps: 0,
            setups: 0,
            measures: 0,
            stop_after: None,
            panic_after: None,
        };
        let result = publish_with_engine(&resumed, &child, &mut engine, &mut || 0.);
        assert!(result.is_ok(), "continuation failed: {:?}", result.err());
        let result = result.unwrap();
        assert_eq!(engine.setups, 0, "must not initialize beta0");
        assert_eq!(engine.steps, 860);
        assert_eq!(engine.measures, 6);
        assert_eq!(result["snapshot_count"], 6);
        assert_eq!(result["completed_steps"], 1014);
        let first =
            load_itebd_checkpoint(&child.join("states.h5"), 0, &Default::default()).unwrap();
        assert_eq!(first.progress.completed_steps, 178);
        assert!(
            (first.progress.accumulated_log_norm - (-1.78)).abs() < 1e-12,
            "saved normalization must survive restart"
        );
        assert_eq!(
            first.metadata.git_revision.as_deref(),
            Some("cccccccccccccccccccccccccccccccccccccccc")
        );
        assert_eq!(
            first.time_grid,
            ItebdTimeGrid::Segmented(settings().segments().to_vec())
        );
        let bytes = fs::read(parent.join("samples.csv")).unwrap();
        fs::write(
            parent.join("samples.csv"),
            [bytes.as_slice(), b"\n"].concat(),
        )
        .unwrap();
        assert!(
            verify_directory(&child).is_err(),
            "changed parent must invalidate child"
        );
        let unstarted = temp.path().join("unstarted");
        let mut engine = ContinuationFixture {
            steps: 0,
            setups: 0,
            measures: 0,
            stop_after: None,
            panic_after: None,
        };
        assert!(publish_with_engine(&resumed, &unstarted, &mut engine, &mut || 0.).is_err());
        assert!(!unstarted.exists());
        assert_eq!(engine.steps, 0);
        assert_eq!(engine.setups, 0);
    }
    // Reinitializing beta0, losing norm, or accepting a changed ancestor breaks this real HDF5 chain.
    #[test]
    fn t03_publication_preserves_two_level_parent_and_saves_only_cold_states() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("parent");
        let warm = temp.path().join("warm");
        let cold = temp.path().join("cold");
        let resumed = resume_fixture(&parent);
        let mut engine = ContinuationFixture {
            steps: 0,
            setups: 0,
            measures: 0,
            stop_after: Some(192),
            panic_after: None,
        };
        let prefix = publish_with_engine(&resumed, &warm, &mut engine, &mut || 0.).unwrap();
        assert_eq!(prefix["snapshot_count"], 4);
        let rows = read_rows(&warm).unwrap();
        let boundary = &rows[3];
        let mut pins = files(&warm).unwrap();
        pins["summary.json"] = fingerprint(&warm.join("summary.json")).unwrap();
        let mut request = resumed.recorded_request().unwrap();
        request["restart"] = json!({"directory":"../warm","snapshot_index":3,"files":pins,
            "progress":{"completed_steps":346,"beta":boundary["beta_actual"],"accumulated_log_norm":boundary["accumulated_log_norm"]},
            "cap":cap_from(boundary).unwrap()});
        let settings =
            super::super::v4::resolve_resume_request(&serde_json::from_value(request).unwrap());
        assert!(
            settings.is_ok(),
            "T0.3 settings rejected: {:?}",
            settings.err()
        );
        let settings = settings.unwrap();
        let mut engine = ContinuationFixture {
            steps: 0,
            setups: 0,
            measures: 0,
            stop_after: None,
            panic_after: None,
        };
        let result = publish_with_engine(&settings, &cold, &mut engine, &mut || 0.).unwrap();
        assert_eq!((engine.setups, engine.steps, engine.measures), (0, 668, 2));
        assert_eq!(result["snapshot_count"], 2);
        let first = load_itebd_checkpoint(&cold.join("states.h5"), 0, &Default::default()).unwrap();
        assert_eq!(first.progress.completed_steps, 514);
        assert!((first.progress.accumulated_log_norm + 5.14).abs() < 1e-12);
        assert_eq!(
            first.time_grid,
            ItebdTimeGrid::Segmented(settings.full_segments().to_vec())
        );
        assert!(verify_directory(&cold).is_ok());
        let bytes = fs::read(parent.join("samples.csv")).unwrap();
        fs::write(
            parent.join("samples.csv"),
            [bytes.as_slice(), b"\n"].concat(),
        )
        .unwrap();
        let error = verify_directory(&cold).unwrap_err();
        assert!(format!("{error:#}").contains("parent file pin changed"));
    }
    #[test]
    fn interrupted_resume_promotes_only_untouched_pending_tail_from_retained_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let settings = resume_fixture(&temp.path().join("parent"));
        for mode in ["tamper", "csv_lag", "clean"] {
            let child = temp.path().join(mode);
            let mut engine = ContinuationFixture {
                steps: 0,
                setups: 0,
                measures: 0,
                stop_after: None,
                panic_after: Some(192),
            };
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                publish_with_engine(&settings, &child, &mut engine, &mut || 0.)
            }));
            assert!(panic.is_err());
            let before = read_rows(&child).unwrap();
            assert_eq!(before[3]["completed_steps"], 346);
            assert_eq!(before[4]["completed_steps"], 154);
            let state_pin = fingerprint(&child.join("states.h5")).unwrap();
            if mode == "tamper" {
                let mut rows = before.clone();
                rows[4]["accumulated_log_norm"] = 123.0.into();
                write_rows(&child, &rows).unwrap();
                assert!(finalize_interrupted_directory(&child, "SVD failure").is_err());
                assert!(!child.join("summary.json").exists());
                continue;
            }
            if mode == "csv_lag" {
                let mut rows = before.clone();
                rows[3] = empty_rows(&settings)[3].clone();
                write_rows(&child, &rows).unwrap();
            }
            let done = finalize_interrupted_directory(&child, "SVD failure");
            assert!(
                done.is_ok(),
                "recover interrupted target prefix: {:?}",
                done.err()
            );
            let done = done.unwrap();
            assert_eq!(done["snapshot_count"], 4);
            assert_eq!(
                done["available_samples"],
                if mode == "csv_lag" { 3 } else { 4 }
            );
            assert_eq!(
                done["available_heat_samples"],
                if mode == "csv_lag" { 3 } else { 4 }
            );
            assert_eq!(done["completed_steps"], 346);
            assert_eq!(done["completed_steps_kind"], "retained_lower_bound");
            let after = read_rows(&child).unwrap();
            let preserved = if mode == "csv_lag" { 3 } else { 4 };
            assert_eq!(&before[..preserved], &after[..preserved]);
            if mode == "csv_lag" {
                assert_eq!(after[3]["snapshot_available"], true);
                assert_eq!(after[3]["available"], false);
                assert_eq!(after[3]["cap_observed_strang_steps"], 262);
            }
            assert_eq!(after[4]["completed_steps"], 346);
            assert_eq!(
                after[4]["accumulated_log_norm"],
                before[3]["accumulated_log_norm"]
            );
            assert_eq!(after[4]["cap_coverage"], "unobserved_tail");
            assert!(after[4]["cap_ever_reached"].is_null());
            assert_eq!(fingerprint(&child.join("states.h5")).unwrap(), state_pin);
            assert_eq!(
                verify_directory(&child).unwrap()["verified_snapshot_count"],
                4
            );
        }
    }
    struct FailingSvdFixture {
        collision: Option<PathBuf>,
    }
    impl Engine<AkltValuesV3> for FailingSvdFixture {
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &crate::tensor::Truncation,
        ) -> Result<StepInfo> {
            if let Some(path) = &self.collision {
                fs::write(path, b"concurrent file").unwrap();
            }
            let a = crate::tensor::new_index(2);
            let b = crate::tensor::new_index(3);
            let c = crate::tensor::new_index(2);
            let t = crate::tensor::from_fn(&[a.clone(), b, c.clone()], |x| {
                if x == [0, 0, 0] {
                    f64::from_bits(0x7ff8_0000_0000_1234)
                } else {
                    (x[0] + 2 * x[1] + 6 * x[2]) as f64
                }
            });
            crate::tensor::svd_bond(&t, &[c, a], &crate::tensor::Truncation::default());
            bail!("nonfinite SVD unexpectedly accepted")
        }
        fn canonicalize(&mut self, _: &mut ItebdState, _: &ItebdHamiltonian) -> Result<f64> {
            unreachable!()
        }
        fn measure(
            &mut self,
            _: &ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: f64,
        ) -> Result<AkltValuesV3> {
            unreachable!()
        }
    }
    // Capture must preserve matrix layout/bits, and an actual I/O collision must not mask SVD failure.
    #[test]
    fn resume_captures_first_svd_matrix_without_masking_source_error() {
        for collision in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let parent = temp.path().join("parent");
            let child = temp.path().join("child");
            let s = resume_fixture(&parent);
            let path = temp.path().join("svd-failure.h5");
            let mut engine = FailingSvdFixture {
                collision: collision.then(|| path.clone()),
            };
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                publish_with_engine(&s, &child, &mut engine, &mut || 0.)
            }))
            .unwrap_err();
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap();
            assert!(
                message.contains("ComputationError"),
                "original SVD error was replaced: {message}"
            );
            assert!(path.is_file(), "first failed matrix was not captured");
            if collision {
                assert_eq!(fs::read(path).unwrap(), b"concurrent file");
                continue;
            }
            let f = hdf5_metno::File::open(&path).unwrap();
            assert_eq!(
                f.dataset("step").unwrap().read_scalar::<u64>().unwrap(),
                155
            );
            assert_eq!(
                f.dataset("svd_call").unwrap().read_scalar::<u64>().unwrap(),
                1
            );
            assert_eq!(
                f.dataset("shape").unwrap().read_raw::<u64>().unwrap(),
                vec![4, 3]
            );
            let values = f
                .dataset("matrix_col_major")
                .unwrap()
                .read_raw::<f64>()
                .unwrap();
            assert_eq!(values[0].to_bits(), 0x7ff8_0000_0000_1234);
            assert_eq!(
                &values[1..],
                &[6., 1., 7., 2., 8., 3., 9., 4., 10., 5., 11.]
            );
            let verified = super::super::v4::verify_failure_file(&path).unwrap();
            assert_eq!(verified["step"], 155);
            assert_eq!(verified["nonfinite_count"], 1);
            assert_eq!(verified["payload_verified"], true);
            let partial = finalize_interrupted_directory(&child, "SVD fixture failure").unwrap();
            assert_eq!(partial["completed_steps"], 154);
            assert_eq!(partial["snapshot_count"], 0);
            assert_eq!(partial["sample_count"], 6);
            assert_eq!(partial["completed_steps_kind"], "retained_lower_bound");
            drop(f);
            let f = hdf5_metno::File::open_rw(&path).unwrap();
            let mut corrupt = values;
            corrupt[1] += 1.;
            f.dataset("matrix_col_major")
                .unwrap()
                .write_raw(&corrupt)
                .unwrap();
            drop(f);
            assert!(
                super::super::v4::verify_failure_file(&path).is_err(),
                "corrupt matrix payload was accepted"
            );
        }
    }
    fn first_progress(s: &AkltReferenceSettingsV3) -> ItebdCheckpointProgress {
        ItebdCheckpointProgress {
            beta: s.targets()[0].beta,
            completed_steps: s.targets()[0].completed_steps,
            accumulated_log_norm: 0.,
            last_step: None,
        }
    }
    fn repin(path: &Path, name: &str) {
        let mut summary = read_json(&path.join("summary.json")).unwrap();
        summary["files"][name] = fingerprint(&path.join(name)).unwrap();
        atomic_write(
            &path.join("summary.json"),
            &serde_json::to_vec_pretty(&summary).unwrap(),
        )
        .unwrap();
    }
    #[test]
    fn resumed_partial_cannot_reset_saved_norm_or_cap_after_rehash() {
        let temp = tempfile::tempdir().unwrap();
        let settings = resume_fixture(&temp.path().join("parent"));
        for mutation in ["norm", "cap"] {
            let child = temp.path().join(mutation);
            drop(Publisher::create(&settings, &child).unwrap());
            finalize_interrupted_directory(&child, "before first new target").unwrap();
            let mut rows = read_rows(&child).unwrap();
            let mut summary = read_json(&child.join("summary.json")).unwrap();
            if mutation == "norm" {
                for row in &mut rows {
                    row["accumulated_log_norm"] = 0.0.into();
                }
                summary["accumulated_log_norm"] = 0.0.into();
            } else {
                for row in &mut rows {
                    row["maximum_dimension_seen"] = Value::Null;
                    row["cap_ever_reached"] = Value::Null;
                    row["cap_observed_strang_steps"] = 0.into();
                    row["cap_coverage"] = "unobserved_tail".into();
                }
                summary["cap"] = json!({"maximum_dimension_seen":null,"ever_reached":null,"observed_strang_steps":0,"coverage":"unobserved_tail"});
            }
            write_rows(&child, &rows).unwrap();
            summary["files"] = files(&child).unwrap();
            atomic_write(
                &child.join("summary.json"),
                &serde_json::to_vec(&summary).unwrap(),
            )
            .unwrap();
            let error = verify_directory(&child).expect_err("reset restart evidence accepted");
            assert!(
                format!("{error:#}").contains("restart boundary"),
                "{error:#}"
            );
        }
    }
    #[test]
    fn summary_lower_bound_cannot_exceed_retained_progress() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("empty");
        drop(Publisher::create(&settings(), &path).unwrap());
        let mut summary =
            finalize_interrupted_directory(&path, "worker died before evolution").unwrap();
        assert_eq!(summary["completed_steps"], 0);
        summary.as_object_mut().unwrap().retain(|key, _| {
            ![
                "sample_count",
                "verified_snapshot_count",
                "checkpoint_verified",
                "scalar_state_consistency_passed",
                "variance_remeasured",
            ]
            .contains(&key.as_str())
        });
        summary["completed_steps"] = 1014.into();
        summary["beta_actual"] = 10.0.into();
        summary["accumulated_log_norm"] = 123.0.into();
        atomic_write(
            &path.join("summary.json"),
            &serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
        let error = verify_directory(&path).unwrap_err();
        assert!(
            format!("{error:#}").contains("retained progress"),
            "{error:#}"
        );
    }
    #[test]
    fn summary_norm_must_match_latest_rows() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("complete");
        fixture(&path, None);
        let mut summary = read_json(&path.join("summary.json")).unwrap();
        summary["accumulated_log_norm"] = 123.0.into();
        atomic_write(
            &path.join("summary.json"),
            &serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
        let error = verify_directory(&path).unwrap_err();
        assert!(
            format!("{error:#}").contains("retained progress"),
            "{error:#}"
        );
    }
    #[test]
    fn summary_may_record_progress_after_last_saved_target() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("partial");
        fixture(&path, Some(1));
        let mut rows = read_rows(&path).unwrap();
        let steps = settings().targets()[0].completed_steps + 1;
        let beta = prefix_beta(&settings(), steps).unwrap();
        for row in &mut rows[1..] {
            row["completed_steps"] = steps.into();
            row["beta_actual"] = beta.into();
            row["accumulated_log_norm"] = (-0.01).into();
            row["cap_observed_strang_steps"] = steps.into();
        }
        write_rows(&path, &rows).unwrap();
        repin(&path, "samples.csv");
        let mut summary = read_json(&path.join("summary.json")).unwrap();
        summary["reason"] = "interrupted after one additional complete step".into();
        summary["completed_steps"] = steps.into();
        summary["beta_actual"] = beta.into();
        summary["accumulated_log_norm"] = (-0.01).into();
        summary["cap"]["observed_strang_steps"] = steps.into();
        atomic_write(
            &path.join("summary.json"),
            &serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
        let verified = verify_directory(&path).unwrap();
        assert_eq!(verified["snapshot_count"], 1);
        assert_eq!(verified["completed_steps"], steps);
    }
    #[test]
    fn variance_failure_preserves_state_and_physical_values() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("partial");
        let mut engine = FixtureEngine {
            steps: 0,
            measures: 0,
            fail_heat_at: Some(1),
        };
        let summary = publish_with_engine(&settings(), &path, &mut engine, &mut || 0.).unwrap();
        assert_eq!(summary["snapshot_count"], 1);
        assert_eq!(summary["available_samples"], 1);
        assert_eq!(summary["available_heat_samples"], 0);
        assert_eq!(summary["status"], "partial");
        assert_eq!(engine.steps, 50);
        let rows = read_rows(&path).unwrap();
        assert_eq!(rows[0]["snapshot_available"], true);
        assert_eq!(rows[0]["available"], true);
        assert_eq!(rows[0]["heat_available"], false);
        assert_eq!(rows[1]["snapshot_available"], false);
    }
    #[test]
    fn completed_snapshot_without_scalar_survives_finalization() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("interrupted");
        let s = settings();
        let state = ItebdState::infinite_temperature(s.hamiltonian()).unwrap();
        let mut publisher = Publisher::create(&s, &path).unwrap();
        publisher
            .before_measurement(&s.targets()[0], &state, &first_progress(&s))
            .unwrap();
        drop(publisher);
        // Crash after the HDF5 append, before publishing its completed CSV flag.
        let mut lagging = read_rows(&path).unwrap();
        lagging[0]["snapshot_available"] = false.into();
        lagging[0]["snapshot_reason"] = "snapshot write in progress".into();
        write_rows(&path, &lagging).unwrap();
        let reply =
            finalize_interrupted_directory(&path, "worker died before measurement").unwrap();
        assert_eq!(reply["snapshot_count"], 1);
        assert_eq!(reply["available_samples"], 0);
        assert_eq!(reply["status"], "partial");
        assert_eq!(reply["completed_steps_kind"], "retained_lower_bound");
        let row = &read_rows(&path).unwrap()[0];
        assert_eq!(row["snapshot_available"], true);
        assert_eq!(row["available"], false);
    }
    #[test]
    fn twenty_target_states_have_no_duplicate_terminal_copy() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("complete");
        let reply = fixture(&path, None);
        assert_eq!(reply["snapshot_count"], 20);
        assert_eq!(reply["verified_snapshot_count"], 20);
        assert_eq!(reply["available_heat_samples"], 20);
        assert_eq!(reply["status"], "complete");
        assert!(!path.join("state.h5").exists());
        assert_eq!(fs::read_dir(&path).unwrap().count(), 4);
        let dispatched = super::super::output::verify_directory(&path).unwrap();
        assert_eq!(dispatched["verified_snapshot_count"], 20);
    }
    #[test]
    fn snapshot_mapping_rejects_swapped_targets() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("swap");
        fixture(&path, None);
        let mut rows = read_rows(&path).unwrap();
        rows[0]["snapshot_index"] = 1.into();
        rows[1]["snapshot_index"] = 0.into();
        write_rows(&path, &rows).unwrap();
        repin(&path, "samples.csv");
        assert!(format!("{:#}", verify_directory(&path).unwrap_err()).contains("snapshot"));
    }
    #[test]
    fn checkpoint_io_failure_keeps_verified_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("io");
        let s = settings();
        let state = ItebdState::infinite_temperature(s.hamiltonian()).unwrap();
        let mut publisher = Publisher::create(&s, &path).unwrap();
        publisher
            .before_measurement(&s.targets()[0], &state, &first_progress(&s))
            .unwrap();
        let held = temp.path().join("held.h5");
        fs::rename(path.join("states.h5"), &held).unwrap();
        fs::create_dir(path.join("states.h5")).unwrap();
        let p = ItebdCheckpointProgress {
            beta: s.targets()[1].beta,
            completed_steps: s.targets()[1].completed_steps,
            accumulated_log_norm: 0.,
            last_step: None,
        };
        let error = publisher
            .before_measurement(&s.targets()[1], &state, &p)
            .unwrap_err();
        assert!(format!("{error:#}").contains("snapshot"));
        drop(publisher);
        fs::remove_dir(path.join("states.h5")).unwrap();
        fs::rename(&held, path.join("states.h5")).unwrap();
        let reply = finalize_interrupted_directory(&path, &format!("{error:#}")).unwrap();
        assert_eq!(reply["snapshot_count"], 1);
        assert_eq!(reply["uncertain_snapshot_index"], 1);
    }
    #[test]
    fn final_flush_uncertainty_is_not_a_verified_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("flush");
        let s = settings();
        let state = ItebdState::infinite_temperature(s.hamiltonian()).unwrap();
        let mut publisher = Publisher::create(&s, &path).unwrap();
        publisher.fail_after_append = true;
        assert!(publisher
            .before_measurement(&s.targets()[0], &state, &first_progress(&s))
            .is_err());
        drop(publisher);
        let reply = finalize_interrupted_directory(&path, "uncertain final flush").unwrap();
        assert_eq!(reply["snapshot_count"], 0);
        assert_eq!(reply["uncertain_snapshot_index"], 0);
    }
    #[test]
    fn incomplete_marker_cannot_certify_published_state() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("marker");
        fixture(&path, Some(1));
        let file = hdf5_metno::File::open_rw(path.join("states.h5")).unwrap();
        file.group("states/000000")
            .unwrap()
            .attr("complete")
            .unwrap()
            .write_scalar(&0u8)
            .unwrap();
        file.flush().unwrap();
        drop(file);
        repin(&path, "states.h5");
        assert!(verify_directory(&path).is_err());
    }

    #[test]
    fn existing_summary_cannot_be_refinalized() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("existing");
        fixture(&path, Some(1));
        let before = fs::read(path.join("summary.json")).unwrap();
        assert!(finalize_interrupted_directory(&path, "another reason").is_err());
        assert_eq!(fs::read(path.join("summary.json")).unwrap(), before);
    }

    #[test]
    fn snapshot_creation_error_keeps_actual_io_context() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("create");
        let s = settings();
        let state = ItebdState::infinite_temperature(s.hamiltonian()).unwrap();
        let mut publisher = Publisher::create(&s, &path).unwrap();
        fs::create_dir(path.join("states.h5")).unwrap();
        let error = publisher
            .before_measurement(&s.targets()[0], &state, &first_progress(&s))
            .unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains("snapshot"));
        assert!(text.contains("states.h5"));
        assert_eq!(read_rows(&path).unwrap()[0]["snapshot_available"], false);
    }

    #[test]
    fn rehashed_variance_and_native_source_contradictions_are_rejected() {
        use hdf5_metno::types::VarLenUnicode;
        use std::str::FromStr;
        for mutation in ["variance", "source", "extra"] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join(mutation);
            fixture(&path, Some(2));
            if mutation == "variance" {
                let mut rows = read_rows(&path).unwrap();
                rows[0]["heat_capacity_variance"] = 1.0.into();
                rows[0]["heat_diagnostics"]["specific_heat_per_site"] = 1.0.into();
                write_rows(&path, &rows).unwrap();
                repin(&path, "samples.csv");
            } else {
                let f = hdf5_metno::File::open_rw(path.join("states.h5")).unwrap();
                if mutation == "source" {
                    let d = f.dataset("run/metadata_json").unwrap();
                    let mut value: Value =
                        serde_json::from_str(d.read_scalar::<VarLenUnicode>().unwrap().as_str())
                            .unwrap();
                    value["git_revision"] = "c".repeat(40).into();
                    d.write_scalar(&VarLenUnicode::from_str(&value.to_string()).unwrap())
                        .unwrap();
                } else {
                    let g = f.group("states").unwrap().create_group("000020").unwrap();
                    g.new_attr::<u8>()
                        .shape(())
                        .create("complete")
                        .unwrap()
                        .write_scalar(&0)
                        .unwrap();
                }
                f.flush().unwrap();
                drop(f);
                repin(&path, "states.h5");
            }
            assert!(verify_directory(&path).is_err(), "accepted {mutation}");
        }
    }
}
