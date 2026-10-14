//! Exclusive compact bundles. Finalization is only for a supervisor-observed dead worker.
use super::*;
use crate::config::{TrotterOrder, TruncationCfg};
use crate::itebd_checkpoint::*;
use anyhow::bail;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

const COLUMNS: &str = "temperature_tick,temperature,beta_target,beta_actual,completed_steps,available,reason,energy,mx,mz,zz,free_energy,entropy,accumulated_log_norm,parity_delta_energy,parity_delta_mx,parity_delta_mz,parity_delta_zz,direct_energy_difference,maximum_rdm_fixed_point_residual,maximum_rdm_trace_residual,maximum_rdm_hermiticity_residual,maximum_rdm_iterations,minimum_rdm_eigenvalue,bond_dimension_ab,bond_dimension_ba";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Summary {
    schema_version: u32,
    request: ReferenceRequest,
    status: String,
    reason: String,
    completed_steps: u64,
    completed_steps_kind: String,
    available_samples: usize,
    elapsed_seconds: Option<f64>,
    checkpoint_available: bool,
}

pub fn run_to_directory(request: &ReferenceRequest, destination: &Path) -> Result<Value> {
    let settings = resolve_request(request)?;
    publish_with(
        &settings,
        destination,
        &mut |s, c| run_reference(s, c),
        &mut || Ok(()),
    )
}
type Runner<'a> = dyn FnMut(
        &ReferenceSettings,
        &mut dyn FnMut(&ReferenceSample) -> Result<()>,
    ) -> Result<ReferenceOutcome>
    + 'a;
fn publish_with(
    settings: &ReferenceSettings,
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
    // A panic is deliberately not caught: only the supervisor may finalize a dead process.
    let outcome = run(settings, &mut |sample| {
        let index = rows
            .iter()
            .position(|r| r.temperature_tick == sample.temperature_tick)
            .context("unknown output target tick")?;
        rows[index] = sample.clone();
        write_samples(destination, &rows)
    })?;
    validate_samples(settings, &rows)?;
    ensure!(
        serde_json::to_value(&rows)? == serde_json::to_value(&outcome.samples)?,
        "runner samples disagree with published targets"
    );
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
    }
    let summary = Summary {
        schema_version: 1,
        request: settings.request().clone(),
        status: outcome.status,
        reason: outcome.reason,
        completed_steps: outcome.progress.completed_steps,
        completed_steps_kind: "exact".into(),
        available_samples: rows.iter().filter(|r| r.available).count(),
        elapsed_seconds: Some(outcome.elapsed_seconds),
        checkpoint_available: outcome.terminal_state.is_some(),
    };
    // Verify the native endpoint and all consistency constraints before publishing summary.
    let report = inspect(destination, settings, &rows, &summary)?;
    before_summary().context("finalize reference publication")?;
    atomic_write(
        &destination.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(report)
}
fn empty_samples(settings: &ReferenceSettings) -> Vec<ReferenceSample> {
    settings
        .targets()
        .iter()
        .map(|t| ReferenceSample {
            temperature_tick: t.temperature_tick,
            temperature: t.temperature,
            beta_target: t.beta,
            beta_actual: 0.,
            completed_steps: 0,
            available: false,
            reason: Some("target not completed before publication".into()),
            accumulated_log_norm: 0.,
            values: None,
        })
        .collect()
}
fn metadata(settings: &ReferenceSettings) -> ItebdRunMetadata {
    ItebdRunMetadata {
        dtau: settings.max_dtau(),
        trotter_order: TrotterOrder::Second,
        truncation: TruncationCfg {
            epsilon: settings.truncation().epsilon,
            max_bond: settings.truncation().max_bond,
        },
        canonicalize_every: 1,
        record_every_beta: None,
        model_label: Some("ising_mixed".into()),
        git_revision: Some(settings.request().source_revision.clone()),
        hermiticity_tolerance: 1e-12,
    }
}
fn exact_state(original: &ItebdState, restored: &ItebdState) -> Result<()> {
    use crate::itebd_state_view::ItebdStateRef;
    ensure!(
        matches!(
            (original, restored),
            (ItebdState::Real(_), ItebdState::Real(_))
        ),
        "terminal checkpoint backend differs"
    );
    let a = ItebdStateRef::from(original);
    let b = ItebdStateRef::from(restored);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (x, y) in [(a.a(), b.a()), (a.b(), b.b())] {
        ensure!(
            x.gamma.indices == y.gamma.indices
                && bits(&x.gamma.to_vec::<f64>()?) == bits(&y.gamma.to_vec::<f64>()?),
            "terminal tensor exact-bit round-trip differs"
        );
    }
    ensure!(
        bits(a.lambda_ab()) == bits(b.lambda_ab()) && bits(a.lambda_ba()) == bits(b.lambda_ba()),
        "terminal Schmidt exact-bit round-trip differs"
    );
    Ok(())
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    // Staging is a sibling of the reserved directory, never a completed bundle member.
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().context("publication path has no parent")?;
    let staging_parent = parent
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let staging = staging_parent.join(format!(
        ".ising-reference-{}-{}.pending",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&staging, path)?;
        Ok(())
    })();
    result.with_context(|| {
        format!(
            "publish {} (pending evidence {})",
            path.display(),
            staging.display()
        )
    })
}
fn quote_csv(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}
fn write_samples(directory: &Path, rows: &[ReferenceSample]) -> Result<()> {
    let mut csv = format!("{COLUMNS}\n");
    for row in rows {
        let value = serde_json::to_value(row)?;
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
// Strict RFC-style quoted fields, including embedded newlines; no line-based splitting.
fn parse_csv(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut closed = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' if field.is_empty() && !closed => quoted = true,
                ',' => {
                    row.push(std::mem::take(&mut field));
                    closed = false;
                }
                '\n' | '\r' => {
                    if c == '\r' {
                        ensure!(chars.next() == Some('\n'), "CSV bare carriage return");
                    }
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    closed = false;
                }
                _ => {
                    ensure!(!closed && c != '"', "CSV malformed quotation");
                    field.push(c);
                }
            }
        }
    }
    ensure!(!quoted, "CSV unterminated quotation");
    if !row.is_empty() || !field.is_empty() || closed {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
fn read_samples(directory: &Path) -> Result<Vec<ReferenceSample>> {
    let path = directory.join("samples.csv");
    (|| -> Result<_> {
        let csv = parse_csv(&fs::read_to_string(&path)?)?;
        let columns = COLUMNS.split(',').collect::<Vec<_>>();
        ensure!(
            csv.first()
                .is_some_and(|h| h.iter().map(String::as_str).eq(columns.iter().copied())),
            "CSV columns do not match schema1"
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
                    } else if *key == "reason" {
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
                for key in &columns[7..] {
                    if *key != "accumulated_log_norm" {
                        ensure!(
                            available != value[*key].is_null(),
                            "row {} availability disagrees with {key}",
                            i + 1
                        );
                    }
                }
                if !available {
                    for key in &columns[7..] {
                        if *key != "accumulated_log_norm" {
                            value.remove(*key);
                        }
                    }
                }
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
fn prefix(settings: &ReferenceSettings, steps: u64) -> Result<f64> {
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
fn validate_samples(settings: &ReferenceSettings, rows: &[ReferenceSample]) -> Result<()> {
    ensure!(
        rows.len() == 20,
        "samples.csv requires exactly 20 requested target rows"
    );
    for (row, target) in rows.iter().zip(settings.targets()) {
        let check = (|| -> Result<()> {
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
                    close(v.energy, -v.zz - 1.05 * v.mx - 0.5 * v.mz),
                    "energy/model identity mismatch"
                );
                ensure!(
                    close(
                        v.free_energy,
                        -(std::f64::consts::LN_2 + row.accumulated_log_norm / 2.) / row.beta_actual
                    ) && close(v.entropy, (v.energy - v.free_energy) * row.beta_actual),
                    "free_energy/entropy normalization identity mismatch"
                );
                ensure!(
                    (0.0..=1e-12).contains(&v.maximum_rdm_fixed_point_residual)
                        && (0.0..=1e-10).contains(&v.maximum_rdm_trace_residual)
                        && (0.0..=1e-10).contains(&v.maximum_rdm_hermiticity_residual)
                        && v.minimum_rdm_eigenvalue >= -1e-10
                        && v.maximum_rdm_iterations <= 10000
                        && v.bond_dimension_ab <= settings.truncation().max_bond.unwrap()
                        && v.bond_dimension_ba <= settings.truncation().max_bond.unwrap(),
                    "invalid RDM diagnostic status"
                );
            } else {
                ensure!(
                    row.values.is_none()
                        && row.reason.as_ref().is_some_and(|s| !s.trim().is_empty()),
                    "unavailable target requires reason and empty values"
                );
            }
            Ok(())
        })();
        check.with_context(|| format!("samples.csv temperature_tick {}", row.temperature_tick))?;
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
fn read_settings(directory: &Path) -> Result<ReferenceSettings> {
    let path = directory.join("conditions.json");
    (|| -> Result<_> {
        let conditions: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let request: ReferenceRequest = serde_json::from_value(
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
    settings: &ReferenceSettings,
    rows: &[ReferenceSample],
    summary: &Summary,
) -> Result<Value> {
    let checkpoint = membership(directory, false)?;
    validate_samples(settings, rows)?;
    ensure!(summary.schema_version==1 && serde_json::to_value(&summary.request)?==serde_json::to_value(settings.request())?,"conditions.json/summary.json request identity mismatch (model/source_revision/worker_sha256)");
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
                && summary.elapsed_seconds.is_some()),
        "summary.json complete status requires all targets and checkpoint"
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
                loaded.progress.completed_steps == summary.completed_steps,
                "checkpoint/summary completed_steps mismatch"
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
                    "zz",
                    "free_energy",
                    "entropy",
                    "parity_delta_energy",
                    "parity_delta_mx",
                    "parity_delta_mz",
                    "parity_delta_zz",
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
    }
    Ok(
        json!({"schema_version":1,"status":summary.status,"sample_count":rows.len(),"available_samples":available,"checkpoint_available":checkpoint,"checkpoint_verified":checkpoint,"terminal_consistency_passed":terminal,"completed_steps":summary.completed_steps,"completed_steps_kind":summary.completed_steps_kind,"request":summary.request}),
    )
}
/// Read-only integrity check; does not certify convergence at earlier target betas.
pub fn verify_directory(directory: &Path) -> Result<Value> {
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
    ensure!(
        !reason.trim().is_empty(),
        "interrupted finalization requires an actual reason"
    );
    let checkpoint = membership(directory, false)
        .with_context(|| format!("finalize bundle {}", directory.display()))?;
    let settings = read_settings(directory)?;
    let rows = read_samples(directory)?;
    validate_samples(&settings, &rows)?;
    let completed_steps = if checkpoint {
        load_itebd_checkpoint(&directory.join("state.h5"), 0, &Default::default())
            .context("state.h5 integrity debt: interrupted checkpoint must be complete and valid")?
            .progress
            .completed_steps
    } else {
        rows.iter().map(|r| r.completed_steps).max().unwrap_or(0)
    };
    let summary = Summary {
        schema_version: 1,
        request: settings.request().clone(),
        status: "partial".into(),
        reason: reason.into(),
        completed_steps,
        completed_steps_kind: if checkpoint {
            "exact"
        } else {
            "retained_lower_bound"
        }
        .into(),
        available_samples: rows.iter().filter(|r| r.available).count(),
        elapsed_seconds: None,
        checkpoint_available: checkpoint,
    };
    if directory.join("summary.json").exists() {
        verify_directory(directory)
            .context("existing summary.json integrity debt; retained evidence was not replaced")?;
    }
    let report = inspect(directory, &settings, &rows, &summary)
        .context("interrupted finalization integrity debt; retained evidence was not replaced")?;
    atomic_write(
        &directory.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Identity;
    impl Engine for Identity {
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &Truncation,
        ) -> Result<(crate::itebd::StepInfo, f64)> {
            Ok((
                crate::itebd::StepInfo {
                    max_bond: 1,
                    min_singular_value: 1.,
                    log_norm: 0.,
                },
                0.,
            ))
        }
    }
    #[test]
    fn output_failure_preserves_target_rows() {
        let req=serde_json::from_value::<ReferenceRequest>(serde_json::json!({"schema_version":1,"model":{"name":"ising_mixed","g":1.05,"h":0.5},"variant":"fine","source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})).unwrap();
        let settings = resolve_request(&req).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("bundle");
        let error = publish_with(
            &settings,
            &bundle,
            &mut |s, c| run_engine(s, c, &mut Identity),
            &mut || anyhow::bail!("injected finalization failure"),
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("injected finalization failure"),
            "{error:#}"
        );
        assert!(bundle.join("samples.csv").is_file());
        assert!(!bundle.join("summary.json").exists());
        let report = finalize_interrupted_directory(&bundle, "finalization failed").unwrap();
        assert_eq!(report["available_samples"], 20);
        assert_eq!(report["checkpoint_verified"], true);
    }
    #[test]
    fn evolution_error_retains_quoted_gaps_and_no_endpoint() {
        struct Stops {
            steps: usize,
        }
        impl Engine for Stops {
            fn step(
                &mut self,
                s: &mut ItebdState,
                h: &ItebdHamiltonian,
                dt: f64,
                t: &Truncation,
            ) -> Result<(crate::itebd::StepInfo, f64)> {
                self.steps += 1;
                if self.steps == 51 {
                    bail!("native error: \"quoted\",\nfailed step");
                }
                Identity.step(s, h, dt, t)
            }
        }
        let req=serde_json::from_value::<ReferenceRequest>(json!({"schema_version":1,"model":{"name":"ising_mixed","g":1.05,"h":0.5},"variant":"fine","source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)})).unwrap();
        let settings = resolve_request(&req).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("partial");
        let report = publish_with(
            &settings,
            &bundle,
            &mut |s, c| run_engine(s, c, &mut Stops { steps: 0 }),
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(report["status"], "partial");
        assert_eq!(report["available_samples"], 1);
        assert_eq!(report["checkpoint_available"], false);
        let rows = read_samples(&bundle).unwrap();
        assert_eq!(rows.len(), 20);
        assert_eq!(rows[0].completed_steps, 50);
        for row in &rows[1..] {
            assert!(!row.available);
            assert!(row.values.is_none());
            assert!(row
                .reason
                .as_ref()
                .unwrap()
                .contains("\"quoted\",\nfailed step"));
        }
        assert!(!bundle.join("state.h5").exists());
        verify_directory(&bundle).unwrap();
    }
}
