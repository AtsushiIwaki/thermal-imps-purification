//! Compact AKLT target tables and one coherent native endpoint.

const CAP_COLUMNS: [(&str, &str); 4] = [
    ("maximum_dimension_seen", "maximum_dimension_seen"),
    ("cap_ever_reached", "ever_reached"),
    ("cap_observed_strang_steps", "observed_strang_steps"),
    ("cap_coverage", "coverage"),
];
fn unknown_cap() -> CapEvidence {
    CapEvidence {
        maximum_dimension_seen: None,
        ever_reached: None,
        observed_strang_steps: 0,
        coverage: CapCoverage::UnobservedTail,
    }
}
fn validate_cap(c: &CapEvidence, steps: u64) -> Result<()> {
    ensure!(
        c.maximum_dimension_seen
            .is_none_or(|d| (1..=128).contains(&d)),
        "cap maximum_dimension_seen invalid"
    );
    ensure!(
        c.observed_strang_steps <= steps + 1,
        "cap observed steps exceed coherent progress plus one returned step"
    );
    if c.maximum_dimension_seen.is_none() {
        ensure!(
            c.observed_strang_steps == 0,
            "cap observations require maximum dimension"
        );
    }
    let hit = c.maximum_dimension_seen.is_some_and(|d| d >= 128);
    ensure!(
        (c.ever_reached == Some(true)) == hit,
        "cap ever_reached contradicts observed maximum"
    );
    match c.coverage {
        CapCoverage::CompleteThroughStop => ensure!(
            c.maximum_dimension_seen.is_some()
                && c.ever_reached.is_some()
                && c.observed_strang_steps == steps,
            "complete cap coverage requires all completed Strang observations"
        ),
        CapCoverage::UnobservedTail => ensure!(
            c.ever_reached != Some(false),
            "unobserved cap tail cannot certify no hit"
        ),
    }
    Ok(())
}
fn observed_cap(rows: &[AkltSample]) -> CapEvidence {
    let mut total = unknown_cap();
    for c in rows.iter().filter_map(|r| r.cap.as_ref()) {
        total.maximum_dimension_seen =
            match (total.maximum_dimension_seen, c.maximum_dimension_seen) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
        total.observed_strang_steps = total.observed_strang_steps.max(c.observed_strang_steps);
        if c.ever_reached == Some(true) {
            total.ever_reached = Some(true);
        }
    }
    total
}

use super::*;
use crate::config::{TrotterOrder, TruncationCfg};
use crate::itebd_checkpoint::*;
use crate::reference_output::{atomic_write, exact_state, parse_csv, quote_csv};
use anyhow::bail;
use serde_json::{json, Value};
use std::{fs, path::Path};

pub(super) const COLUMNS: &str = "temperature_tick,temperature,beta_target,beta_actual,completed_steps,available,reason,energy,mx,mz,free_energy,entropy,accumulated_log_norm,parity_delta_energy,parity_delta_mx,parity_delta_mz,direct_energy_difference,maximum_rdm_fixed_point_residual,maximum_rdm_trace_residual,maximum_rdm_hermiticity_residual,maximum_rdm_iterations,minimum_rdm_eigenvalue,bond_dimension_ab,bond_dimension_ba,cap_reached_now,maximum_dimension_seen,cap_ever_reached,cap_observed_strang_steps,cap_coverage";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Summary {
    schema_version: u32,
    contract: String,
    request: AkltReferenceRequest,
    status: String,
    reason: String,
    completed_steps: u64,
    beta_actual: f64,
    accumulated_log_norm: f64,
    cap: CapEvidence,
    completed_steps_kind: String,
    available_samples: usize,
    // Require the field while preserving explicit null for interrupted workers.
    #[serde(deserialize_with = "Option::<f64>::deserialize")]
    elapsed_seconds: Option<f64>,
    #[serde(deserialize_with = "Option::<f64>::deserialize")]
    budget_overrun_seconds: Option<f64>,
    checkpoint_available: bool,
}

pub fn run_to_directory(request: &AkltReferenceRequest, destination: &Path) -> Result<Value> {
    let started = std::time::Instant::now();
    let settings = resolve_request(request)?;
    publish_with(
        &settings,
        destination,
        &mut |s, c| run_with_elapsed(s, c, &mut || started.elapsed().as_secs_f64()),
        &mut || Ok(()),
    )
}
type Runner<'a> = dyn FnMut(&AkltReferenceSettings, &mut dyn FnMut(&AkltSample) -> Result<()>) -> Result<AkltOutcome>
    + 'a;

fn publish_with(
    settings: &AkltReferenceSettings,
    destination: &Path,
    run: &mut Runner<'_>,
    before_summary: &mut dyn FnMut() -> Result<()>,
) -> Result<Value> {
    fs::create_dir(destination).with_context(|| {
        format!(
            "reserve exclusive reference destination {}",
            destination.display()
        )
    })?;
    atomic_write(
        &destination.join("conditions.json"),
        &serde_json::to_vec_pretty(&preflight(settings.request())?)?,
    )?;
    let mut rows = empty_samples(settings);
    write_samples(destination, &rows)?;
    let outcome = run(settings, &mut |sample| {
        let index = rows
            .iter()
            .position(|r| r.temperature_tick == sample.temperature_tick)
            .context("unknown output target tick")?;
        rows[index] = sample.clone();
        write_samples(destination, &rows)
    })?;
    let terminal_started = std::time::Instant::now();
    let checked = std::time::Instant::now();
    validate_samples(settings, &rows)?;
    ensure!(
        serde_json::to_value(&rows)? == serde_json::to_value(&outcome.samples)?,
        "runner samples disagree with published targets"
    );
    let mut native_seconds = checked.elapsed().as_secs_f64();
    if let Some(state) = &outcome.terminal_state {
        let path = destination.join("state.h5");
        let mut writer = ItebdTrajectoryWriter::create_scheduled(
            &path,
            metadata(settings),
            settings.hamiltonian(),
            settings.segments().to_vec(),
        )
        .with_context(|| format!("create terminal checkpoint {}", path.display()))?;
        writer
            .append(state.into(), &outcome.progress)
            .context("append terminal reference checkpoint")?;
        writer
            .finish()
            .context("finish terminal reference checkpoint")?;
        let checked = std::time::Instant::now();
        let loaded = load_itebd_checkpoint(&path, 0, &Default::default())
            .context("round-trip terminal reference checkpoint")?;
        exact_state(state, &loaded.state)?;
        ensure!(
            loaded.progress.beta.to_bits() == outcome.progress.beta.to_bits()
                && loaded.progress.accumulated_log_norm.to_bits()
                    == outcome.progress.accumulated_log_norm.to_bits()
                && loaded.progress.completed_steps == outcome.progress.completed_steps,
            "terminal checkpoint progress round-trip differs"
        );
        native_seconds += checked.elapsed().as_secs_f64();
    }
    let summary = Summary {
        schema_version: settings.request().schema_version,
        contract: settings.request().contract.clone(),
        request: settings.request().clone(),
        status: outcome.status,
        reason: outcome.reason,
        completed_steps: outcome.progress.completed_steps,
        beta_actual: outcome.progress.beta,
        accumulated_log_norm: outcome.progress.accumulated_log_norm,
        cap: outcome.cap.context("AKLT outcome cap evidence missing")?,
        completed_steps_kind: "exact".into(),
        available_samples: rows.iter().filter(|r| r.available).count(),
        elapsed_seconds: Some(outcome.elapsed_seconds),
        budget_overrun_seconds: settings
            .request()
            .max_seconds
            .map(|_| outcome.budget_overrun_seconds),
        checkpoint_available: outcome.terminal_state.is_some(),
    };
    let checked = std::time::Instant::now();
    let mut report = inspect(destination, settings, &rows, &summary)?;
    native_seconds += checked.elapsed().as_secs_f64();
    before_summary().context("finalize reference publication")?;
    atomic_write(
        &destination.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    report["timing"] = json!({"trajectory_seconds":outcome.elapsed_seconds,
        "terminal_publication_seconds":(terminal_started.elapsed().as_secs_f64()-native_seconds).max(0.),"native_verification_seconds":native_seconds});
    Ok(report)
}

fn empty_samples(settings: &AkltReferenceSettings) -> Vec<AkltSample> {
    settings
        .targets()
        .iter()
        .map(|t| AkltSample {
            temperature_tick: t.temperature_tick,
            temperature: t.temperature,
            beta_target: t.beta,
            beta_actual: 0.,
            completed_steps: 0,
            available: false,
            reason: Some("target not completed before publication".into()),
            accumulated_log_norm: 0.,
            values: None,
            cap: Some(unknown_cap()),
        })
        .collect()
}
fn metadata(settings: &AkltReferenceSettings) -> ItebdRunMetadata {
    ItebdRunMetadata {
        dtau: settings.max_dtau(),
        trotter_order: TrotterOrder::Second,
        truncation: TruncationCfg {
            epsilon: settings.truncation().epsilon,
            max_bond: settings.truncation().max_bond,
        },
        canonicalize_every: 1,
        record_every_beta: None,
        model_label: Some("aklt_projector".into()),
        git_revision: Some(settings.request().source_revision.clone()),
        hermiticity_tolerance: 1e-12,
    }
}

fn write_samples(directory: &Path, rows: &[AkltSample]) -> Result<()> {
    let mut csv = format!("{COLUMNS}\n");
    for row in rows {
        let mut value = serde_json::to_value(row)?;
        for (column, key) in CAP_COLUMNS {
            value[column] = value["cap"][key].clone();
        }
        let fields = COLUMNS
            .split(',')
            .map(|key| match &value[key] {
                Value::Null => String::new(),
                Value::String(s) => quote_csv(s),
                v => v.to_string(),
            })
            .collect::<Vec<_>>();
        csv.push_str(&fields.join(","));
        csv.push('\n');
    }
    atomic_write(&directory.join("samples.csv"), csv.as_bytes())
}

fn read_samples(directory: &Path) -> Result<Vec<AkltSample>> {
    let path = directory.join("samples.csv");
    (|| -> Result<_> {
        let csv = parse_csv(&fs::read_to_string(&path)?)?;
        let columns = COLUMNS.split(',').collect::<Vec<_>>();
        ensure!(
            csv.first()
                .is_some_and(|h| h.iter().map(String::as_str).eq(columns.iter().copied())),
            "CSV columns do not match AKLT schema1"
        );
        csv.iter()
            .skip(1)
            .enumerate()
            .map(|(i, row)| {
                ensure!(row.len() == columns.len(), "row {} column count", i + 1);
                let mut value = serde_json::Map::new();
                for (key, field) in columns.iter().zip(row) {
                    let item = if field.is_empty() {
                        Value::Null
                    } else if matches!(*key, "reason" | "cap_coverage") {
                        Value::String(field.clone())
                    } else {
                        serde_json::from_str::<Value>(field)
                            .with_context(|| format!("row {} field {key}", i + 1))?
                    };
                    value.insert((*key).into(), item);
                }
                let available = value["available"]
                    .as_bool()
                    .context("CSV available must be boolean")?;
                for key in &columns[7..25] {
                    if *key != "accumulated_log_norm" {
                        ensure!(
                            available != value[*key].is_null(),
                            "row {} availability disagrees with {key}",
                            i + 1
                        );
                        if !available {
                            value.remove(*key);
                        }
                    }
                }
                let mut cap = serde_json::Map::new();
                for (column, key) in CAP_COLUMNS {
                    cap.insert(
                        key.into(),
                        value.remove(column).context("missing cap column")?,
                    );
                }
                value.insert("cap".into(), Value::Object(cap));
                serde_json::from_value(Value::Object(value))
                    .with_context(|| format!("CSV row {}", i + 1))
            })
            .collect()
    })()
    .with_context(|| format!("read {}", path.display()))
}

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.)
}
fn prefix(settings: &AkltReferenceSettings, steps: u64) -> Result<f64> {
    let mut left = steps;
    let mut beta = 0.;
    for segment in settings.segments() {
        let n = left.min(segment.steps);
        beta += 2. * n as f64 * segment.dtau;
        left -= n;
    }
    ensure!(left == 0, "completed_steps exceeds time grid");
    Ok(beta)
}
fn validate_samples(settings: &AkltReferenceSettings, rows: &[AkltSample]) -> Result<()> {
    ensure!(
        rows.len() == 20,
        "samples.csv requires exactly 20 requested target rows"
    );
    let mut previous: Option<&CapEvidence> = None;
    for (row, target) in rows.iter().zip(settings.targets()) {
        (|| -> Result<()> {
            ensure!(
                row.temperature_tick == target.temperature_tick
                    && row.temperature.to_bits() == target.temperature.to_bits()
                    && row.beta_target.to_bits() == target.beta.to_bits(),
                "target tick/temperature/beta identity mismatch"
            );
            ensure!(
                row.completed_steps <= target.completed_steps
                    && close(row.beta_actual, prefix(settings, row.completed_steps)?),
                "beta_actual/grid-prefix disagreement"
            );
            ensure!(
                row.accumulated_log_norm.is_finite(),
                "nonfinite accumulated_log_norm"
            );
            let cap = row
                .cap
                .as_ref()
                .context("AKLT sample cap evidence missing")?;
            validate_cap(cap, row.completed_steps)?;
            if cap.maximum_dimension_seen.is_some() {
                if let Some(before) = previous {
                    ensure!(
                        cap.maximum_dimension_seen >= before.maximum_dimension_seen
                            && cap.observed_strang_steps >= before.observed_strang_steps
                            && (before.ever_reached != Some(true)
                                || cap.ever_reached == Some(true))
                            && (before.coverage != CapCoverage::UnobservedTail
                                || cap.coverage == CapCoverage::UnobservedTail),
                        "cap evidence is not cumulative"
                    );
                }
                previous = Some(cap);
            }
            if row.available {
                ensure!(
                    row.reason.is_none()
                        && row.completed_steps == target.completed_steps
                        && close(row.beta_actual, target.beta),
                    "available target status/progress mismatch"
                );
                let v = row
                    .values
                    .as_ref()
                    .context("available target missing values")?;
                v.validate()?;
                ensure!(
                    close(
                        v.free_energy,
                        -(3f64.ln() + row.accumulated_log_norm / 2.) / row.beta_actual
                    ) && close(v.entropy, (v.energy - v.free_energy) * row.beta_actual),
                    "free_energy/entropy normalization identity mismatch"
                );
                ensure!(
                    (-1e-10..=1. + 1e-10).contains(&v.energy)
                        && v.mx.abs() <= 1. + 1e-10
                        && v.mz.abs() <= 1. + 1e-10,
                    "spin-one/projector observable bounds"
                );
                ensure!(
                    (0.0..=1e-12).contains(&v.maximum_rdm_fixed_point_residual)
                        && (0.0..=1e-10).contains(&v.maximum_rdm_trace_residual)
                        && (0.0..=1e-10).contains(&v.maximum_rdm_hermiticity_residual)
                        && v.minimum_rdm_eigenvalue >= -1e-10
                        && v.maximum_rdm_iterations <= 10000
                        && v.bond_dimension_ab <= 128
                        && v.bond_dimension_ba <= 128,
                    "invalid RDM diagnostic status"
                );
                let dimension = v.bond_dimension_ab.max(v.bond_dimension_ba);
                ensure!(
                    v.cap_reached_now == (dimension >= 128)
                        && cap.maximum_dimension_seen.is_some_and(|d| d >= dimension)
                        && cap.coverage == CapCoverage::CompleteThroughStop,
                    "available sample cap dimensions/coverage mismatch"
                );
            } else {
                ensure!(
                    row.values.is_none()
                        && row.reason.as_ref().is_some_and(|s| !s.trim().is_empty()),
                    "unavailable target requires reason and empty values"
                );
            }
            Ok(())
        })()
        .with_context(|| format!("samples.csv temperature_tick {}", row.temperature_tick))?;
    }
    Ok(())
}

fn membership(directory: &Path, require_summary: bool) -> Result<bool> {
    ensure!(
        fs::symlink_metadata(directory)?.file_type().is_dir(),
        "bundle directory must be a regular directory, not a symlink"
    );
    let mut names = std::collections::BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 bundle member"))?;
        ensure!(
            ["conditions.json", "summary.json", "samples.csv", "state.h5"].contains(&name.as_str()),
            "unexpected bundle member {name}"
        );
        ensure!(
            entry.file_type()?.is_file(),
            "bundle member {name} must be regular, not a symlink"
        );
        names.insert(name);
    }
    for name in ["conditions.json", "samples.csv"] {
        ensure!(names.contains(name), "missing bundle member {name}");
    }
    ensure!(
        !require_summary || names.contains("summary.json"),
        "incomplete publication: missing summary.json"
    );
    Ok(names.contains("state.h5"))
}
fn read_settings(directory: &Path) -> Result<AkltReferenceSettings> {
    let path = directory.join("conditions.json");
    (|| -> Result<_> {
        let conditions: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let request: AkltReferenceRequest = serde_json::from_value(
            conditions
                .get("request")
                .context("missing request identity")?
                .clone(),
        )?;
        let settings = resolve_request(&request)?;
        ensure!(
            conditions == preflight(&request)?,
            "conditions model/source/settings schema mismatch"
        );
        Ok(settings)
    })()
    .with_context(|| format!("validate {}", path.display()))
}
fn inspect(
    directory: &Path,
    settings: &AkltReferenceSettings,
    rows: &[AkltSample],
    summary: &Summary,
) -> Result<Value> {
    let checkpoint = membership(directory, false)?;
    validate_samples(settings, rows)?;
    let valid_contract = summary.contract == settings.request().contract;
    ensure!(valid_contract && summary.schema_version==settings.request().schema_version && serde_json::to_value(&summary.request)?==serde_json::to_value(settings.request())?,"conditions.json/summary.json request identity mismatch (model/source_revision/worker_sha256)");
    ensure!(
        ["complete", "partial"].contains(&summary.status.as_str())
            && !summary.reason.trim().is_empty(),
        "summary.json invalid status/reason"
    );
    ensure!(
        summary
            .elapsed_seconds
            .is_none_or(|n| n.is_finite() && n >= 0.),
        "summary.json invalid elapsed_seconds"
    );
    ensure!(
        summary.beta_actual.is_finite()
            && close(
                summary.beta_actual,
                prefix(settings, summary.completed_steps)?
            )
            && summary.accumulated_log_norm.is_finite(),
        "summary.json beta/norm/grid mismatch"
    );
    ensure!(
        match (
            settings.request().max_seconds,
            summary.elapsed_seconds,
            summary.budget_overrun_seconds
        ) {
            (None, _, None) => true,
            (Some(limit), Some(elapsed), Some(overrun)) =>
                overrun.is_finite() && overrun >= 0. && close(overrun, (elapsed - limit).max(0.)),
            (Some(_), None, None) => true,
            _ => false,
        },
        "summary.json elapsed/overrun evidence mismatch"
    );
    validate_cap(&summary.cap, summary.completed_steps).context("summary.json cap evidence")?;
    if summary.elapsed_seconds.is_none() {
        ensure!(
            summary.cap.coverage == CapCoverage::UnobservedTail,
            "interrupted cap coverage must remain unknown"
        );
    }
    let observed = observed_cap(rows);
    ensure!(
        summary.cap.maximum_dimension_seen >= observed.maximum_dimension_seen
            && summary.cap.observed_strang_steps == observed.observed_strang_steps
            && (observed.ever_reached != Some(true) || summary.cap.ever_reached == Some(true)),
        "summary.json cap disagrees with retained observations"
    );
    if summary.elapsed_seconds.is_some() {
        ensure!(
            summary.cap.maximum_dimension_seen == observed.maximum_dimension_seen,
            "summary.json cap maximum exceeds observed rows"
        );
    }
    let available = rows.iter().filter(|r| r.available).count();
    ensure!(
        summary.available_samples == available && summary.checkpoint_available == checkpoint,
        "summary.json target/checkpoint availability mismatch"
    );
    ensure!(
        summary.completed_steps_kind
            == if !checkpoint && summary.elapsed_seconds.is_none() {
                "retained_lower_bound"
            } else {
                "exact"
            },
        "summary.json completed_steps_kind inconsistent with endpoint/elapsed evidence"
    );
    let max_steps = rows.iter().map(|r| r.completed_steps).max().unwrap_or(0);
    ensure!(
        summary.completed_steps >= max_steps
            && summary.completed_steps <= settings.targets().last().unwrap().completed_steps,
        "summary.json completed_steps inconsistent"
    );
    ensure!(
        summary.status != "complete"
            || (available == 20
                && checkpoint
                && summary.completed_steps == settings.targets().last().unwrap().completed_steps
                && summary.elapsed_seconds.is_some()
                && summary.cap.coverage == CapCoverage::CompleteThroughStop),
        "summary.json complete status requires all targets, checkpoint and observed cap coverage"
    );
    let mut terminal = None;
    if checkpoint {
        let path = directory.join("state.h5");
        let loaded = (|| -> Result<_> {
            let members = hdf5_metno::File::open(&path)?
                .group("states")?
                .member_names()?;
            ensure!(
                members == ["000000"],
                "requires exactly one terminal snapshot, including pending snapshots"
            );
            let loaded = load_itebd_checkpoint(&path, 0, &Default::default())?;
            ensure!(
                loaded.time_grid == ItebdTimeGrid::Segmented(settings.segments().to_vec()),
                "checkpoint time_grid mismatch"
            );
            ensure!(
                serde_json::to_value(&loaded.metadata)?
                    == serde_json::to_value(metadata(settings))?,
                "checkpoint model/source_revision/settings metadata mismatch"
            );
            match (&loaded.hamiltonian, settings.hamiltonian()) {
                (ItebdHamiltonian::Real(a), ItebdHamiltonian::Real(b)) => ensure!(
                    a.two_site_h == b.two_site_h && a.site_energy == b.site_energy,
                    "checkpoint model matrices mismatch"
                ),
                _ => bail!("checkpoint backend/model mismatch"),
            }
            ensure!(
                matches!(loaded.state, ItebdState::Real(_)),
                "checkpoint state backend mismatch"
            );
            ensure!(
                loaded.progress.completed_steps == summary.completed_steps
                    && loaded.progress.beta.to_bits() == summary.beta_actual.to_bits()
                    && loaded.progress.accumulated_log_norm.to_bits()
                        == summary.accumulated_log_norm.to_bits(),
                "checkpoint/summary completed_steps/beta/norm mismatch"
            );
            ensure!(
                close(
                    loaded.progress.beta,
                    prefix(settings, loaded.progress.completed_steps)?
                ),
                "checkpoint beta/grid prefix mismatch"
            );
            if let Some(row) = rows
                .iter()
                .find(|r| r.completed_steps == loaded.progress.completed_steps && r.available)
            {
                ensure!(
                    close(
                        row.accumulated_log_norm,
                        loaded.progress.accumulated_log_norm
                    ) && close(row.beta_actual, loaded.progress.beta),
                    "checkpoint terminal ledger mismatch"
                );
                let actual = serde_json::to_value(measure_reference(
                    &loaded.state,
                    &loaded.hamiltonian,
                    loaded.progress.beta,
                    loaded.progress.accumulated_log_norm,
                )?)?;
                let expected = serde_json::to_value(row.values.as_ref().unwrap())?;
                for key in [
                    "energy",
                    "mx",
                    "mz",
                    "free_energy",
                    "entropy",
                    "parity_delta_energy",
                    "parity_delta_mx",
                    "parity_delta_mz",
                    "direct_energy_difference",
                    "bond_dimension_ab",
                    "bond_dimension_ba",
                ] {
                    ensure!(
                        close(
                            actual[key].as_f64().unwrap(),
                            expected[key].as_f64().unwrap()
                        ),
                        "terminal sample {key} disagrees with native remeasurement"
                    );
                }
                terminal = Some(true);
            }
            Ok(loaded)
        })()
        .with_context(|| format!("native integrity verification {}", path.display()))?;
        let dimensions = crate::itebd_state_view::ItebdStateRef::from(&loaded.state)
            .validate()?
            .bond_dimensions;
        let dimension = *dimensions.iter().max().unwrap();
        ensure!(
            summary
                .cap
                .maximum_dimension_seen
                .is_some_and(|d| d >= dimension)
                && (dimension < 128 || summary.cap.ever_reached == Some(true)),
            "checkpoint dimensions exceed summary cap evidence"
        );
        if let Some(step) = &loaded.progress.last_step {
            ensure!(
                summary
                    .cap
                    .maximum_dimension_seen
                    .is_some_and(|d| d >= step.max_bond),
                "checkpoint Strang maximum exceeds cap evidence"
            );
        }
        ensure!(
            rows.iter()
                .filter(|r| r.completed_steps == loaded.progress.completed_steps)
                .all(|r| close(r.accumulated_log_norm, loaded.progress.accumulated_log_norm)),
            "state.h5/sample norm ledger mismatch"
        );
    } else {
        ensure!(
            summary.completed_steps == max_steps,
            "summary.json progress without endpoint exceeds retained scalar evidence"
        );
        ensure!(
            rows.iter()
                .filter(|r| r.completed_steps == max_steps)
                .all(|r| close(r.accumulated_log_norm, summary.accumulated_log_norm)),
            "summary.json norm disagrees with retained scalar evidence"
        );
    }
    let contract_name = &summary.contract;
    Ok(
        json!({"schema_version":summary.schema_version,"contract":contract_name,"status":summary.status,"sample_count":rows.len(),"available_samples":available,"checkpoint_available":checkpoint,"checkpoint_verified":checkpoint,"terminal_consistency_passed":terminal,"completed_steps":summary.completed_steps,"completed_steps_kind":summary.completed_steps_kind,"request":summary.request,"beta_actual":summary.beta_actual,"accumulated_log_norm":summary.accumulated_log_norm,"cap":summary.cap,"elapsed_seconds":summary.elapsed_seconds,"budget_overrun_seconds":summary.budget_overrun_seconds}),
    )
}
fn is_v3_directory(directory: &Path) -> Result<bool> {
    ensure!(fs::symlink_metadata(directory)?.file_type().is_dir(), "reference directory must be regular");
    let path=directory.join("conditions.json");
    ensure!(fs::symlink_metadata(&path)?.file_type().is_file(),"conditions.json must be regular");
    let value:Value=serde_json::from_slice(&fs::read(&path)?)?;
    Ok(value["contract"].as_str()==Some(super::v3::CONTRACT_V3))
}
/// Read-only integrity check; does not certify convergence at earlier target betas.
pub fn verify_directory(directory: &Path) -> Result<Value> {
    if is_v3_directory(directory)? {
        return super::v3_output::verify_directory(directory);
    }
    membership(directory, true)
        .with_context(|| format!("verify bundle {}", directory.display()))?;
    let settings = read_settings(directory)?;
    let rows = read_samples(directory)?;
    let summary: Summary = serde_json::from_slice(&fs::read(directory.join("summary.json"))?)
        .context("parse strict summary.json")?;
    inspect(directory, &settings, &rows, &summary)
}
/// Recover only after the supervisor has observed termination. Never evolves or replaces state.
pub fn finalize_interrupted_directory(directory: &Path, reason: &str) -> Result<Value> {
    if is_v3_directory(directory)? {
        return super::v3_output::finalize_interrupted_directory(directory, reason);
    }
    let started = std::time::Instant::now();
    ensure!(
        !reason.trim().is_empty(),
        "interrupted finalization requires an actual reason"
    );
    let checkpoint = membership(directory, false)
        .with_context(|| format!("finalize bundle {}", directory.display()))?;
    let settings = read_settings(directory)?;
    let rows = read_samples(directory)?;
    validate_samples(&settings, &rows)?;
    let mut cap = observed_cap(&rows);
    let (steps, beta, norm) = if checkpoint {
        let loaded = load_itebd_checkpoint(&directory.join("state.h5"), 0, &Default::default())
            .context(
                "state.h5 integrity debt: interrupted checkpoint must be complete and valid",
            )?;
        let dimensions = crate::itebd_state_view::ItebdStateRef::from(&loaded.state)
            .validate()?
            .bond_dimensions;
        let dimension = *dimensions.iter().max().unwrap();
        cap.maximum_dimension_seen = Some(cap.maximum_dimension_seen.unwrap_or(0).max(dimension));
        if dimension >= 128 {
            cap.ever_reached = Some(true);
        }
        (
            loaded.progress.completed_steps,
            loaded.progress.beta,
            loaded.progress.accumulated_log_norm,
        )
    } else {
        let row = rows
            .iter()
            .max_by_key(|r| r.completed_steps)
            .context("missing retained rows")?;
        (
            row.completed_steps,
            row.beta_actual,
            row.accumulated_log_norm,
        )
    };
    cap.mark_unobserved();
    let summary = Summary {
        schema_version: settings.request().schema_version,
        contract: settings.request().contract.clone(),
        request: settings.request().clone(),
        status: "partial".into(),
        reason: reason.into(),
        completed_steps: steps,
        beta_actual: beta,
        accumulated_log_norm: norm,
        cap,
        completed_steps_kind: if checkpoint {
            "exact"
        } else {
            "retained_lower_bound"
        }
        .into(),
        available_samples: rows.iter().filter(|r| r.available).count(),
        elapsed_seconds: None,
        budget_overrun_seconds: None,
        checkpoint_available: checkpoint,
    };
    if directory.join("summary.json").exists() {
        verify_directory(directory)
            .context("existing summary.json integrity debt; retained evidence was not replaced")?;
    }
    let mut report = inspect(directory, &settings, &rows, &summary)
        .context("interrupted finalization integrity debt; retained evidence was not replaced")?;
    let native_seconds = started.elapsed().as_secs_f64();
    let publication = std::time::Instant::now();
    atomic_write(
        &directory.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    report["timing"] = json!({"trajectory_seconds":null,"native_verification_seconds":native_seconds,"terminal_publication_seconds":publication.elapsed().as_secs_f64()});
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_run(
        settings: &AkltReferenceSettings,
        callback: &mut dyn FnMut(&AkltSample) -> Result<()>,
    ) -> Result<AkltOutcome> {
        let state = ItebdState::infinite_temperature(settings.hamiltonian())?;
        let first = &settings.targets()[0];
        let progress = ItebdCheckpointProgress {
            beta: first.beta,
            completed_steps: first.completed_steps,
            accumulated_log_norm: 0.,
            last_step: None,
        };
        let cap = CapEvidence {
            maximum_dimension_seen: Some(1),
            ever_reached: Some(false),
            observed_strang_steps: first.completed_steps,
            coverage: CapCoverage::CompleteThroughStop,
        };
        let mut samples = Vec::new();
        for (i, t) in settings.targets().iter().enumerate() {
            let s = AkltSample {
                temperature_tick: t.temperature_tick,
                temperature: t.temperature,
                beta_target: t.beta,
                beta_actual: first.beta,
                completed_steps: first.completed_steps,
                available: i == 0,
                reason: if i == 0 {
                    None
                } else {
                    Some("time limit".into())
                },
                accumulated_log_norm: 0.,
                values: if i == 0 {
                    Some(measure_reference(
                        &state,
                        settings.hamiltonian(),
                        first.beta,
                        0.,
                    )?)
                } else {
                    None
                },
                cap: Some(cap.clone()),
            };
            callback(&s)?;
            samples.push(s);
        }
        Ok(AkltOutcome {
            samples,
            terminal_state: Some(state),
            progress,
            status: "partial".into(),
            reason: "time limit".into(),
            elapsed_seconds: 3601.,
            budget_overrun_seconds: 1.,
            cap: Some(cap),
        })
    }
    #[test]
    fn aklt_v2_reference_trajectory_publication_keeps_unlimited_timing() -> Result<()> {
        let req: AkltReferenceRequest = serde_json::from_value(
            json!({"schema_version":2,"contract":"aklt-projector-reference-v2","model":{"name":"aklt_projector"},"variant":"fine","max_seconds":null,"source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)}),
        )?;
        let settings = resolve_request(&req)?;
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("v2");
        let report = publish_with(
            &settings,
            &path,
            &mut |settings, callback| {
                let mut result = fixture_run(settings, callback)?;
                result.reason = "fixture numerical stop".into();
                result.elapsed_seconds = 4001.;
                result.budget_overrun_seconds = 0.;
                Ok(result)
            },
            &mut || Ok(()),
        )?;
        assert_eq!(report["schema_version"], 2);
        assert_eq!(report["contract"], "aklt-projector-reference-v2");
        assert_eq!(report["elapsed_seconds"], 4001.);
        assert!(report["budget_overrun_seconds"].is_null());
        assert_eq!(report["checkpoint_verified"], true);
        verify_directory(&path)?;
        Ok(())
    }
    #[test]
    fn aklt_publication_failure_retains_endpoint_and_observed_rows() -> Result<()> {
        let req: AkltReferenceRequest = serde_json::from_value(
            json!({"schema_version":1,"contract":"aklt-projector-reference-v1","model":{"name":"aklt_projector"},"variant":"fine","max_seconds":3600.,"source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)}),
        )?;
        let settings = resolve_request(&req)?;
        let temp = tempfile::tempdir()?;
        let failed = temp.path().join("failed");
        let error = publish_with(&settings, &failed, &mut fixture_run, &mut || {
            anyhow::bail!("injected summary failure")
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("injected summary failure"));
        assert!(!failed.join("summary.json").exists());
        assert!(failed.join("state.h5").is_file());
        let report = finalize_interrupted_directory(&failed, "worker died")?;
        assert_eq!(report["checkpoint_verified"], true);
        assert_eq!(report["available_samples"], 1);
        assert_eq!(report["cap"]["coverage"], "unobserved_tail");
        let good = temp.path().join("good");
        let report = publish_with(&settings, &good, &mut fixture_run, &mut || Ok(()))?;
        assert_eq!(report["timing"]["trajectory_seconds"], 3601.);
        for name in [
            "terminal_publication_seconds",
            "native_verification_seconds",
        ] {
            assert!(report["timing"][name].as_f64().unwrap() >= 0.);
        }
        assert_eq!(report["budget_overrun_seconds"], 1.);
        verify_directory(&good)?;
        Ok(())
    }
}
