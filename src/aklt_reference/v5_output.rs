//! Bounded scalar publication; never an output checkpoint or tensor recorder.
use super::{
    v3, v3_output,
    v5::{self, AkltStepSampledRequest, AkltStepSampledSettings},
    v5_input,
};
use super::{AkltSampleValues, CapCoverage, CapEvidence};
use crate::itebd_checkpoint::ItebdCheckpointProgress;
use crate::reference_output::{atomic_write, parse_csv, quote_csv};
use crate::reference_trajectory::step_sampled::{
    run_step_sampled_from, StepObserver, StepSample, StepTrajectorySettings,
};
use crate::reference_trajectory::{Engine, NativeEngine, RunPolicy};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const ID: &[&str] = &[
    "sample_index",
    "completed_steps",
    "beta_actual",
    "temperature",
    "accumulated_log_norm",
    "origin",
    "available",
    "reason",
];
const EXTRA: &[&str] = &[
    "maximum_dimension_seen",
    "cap_ever_reached",
    "cap_observed_strang_steps",
    "cap_coverage",
    "heat_available",
    "heat_reason",
    "heat_capacity_variance",
    "heat_diagnostics",
];
fn columns() -> Vec<&'static str> {
    ID.iter()
        .chain(v3_output::PHYSICAL.iter())
        .chain(EXTRA.iter())
        .copied()
        .collect()
}
fn number(row: &Value, key: &str) -> Result<f64> {
    let x = row[key]
        .as_f64()
        .with_context(|| format!("missing scalar {key}"))?;
    ensure!(x.is_finite(), "nonfinite scalar {key}");
    Ok(x)
}
fn count(row: &Value, key: &str) -> Result<u64> {
    row[key]
        .as_u64()
        .with_context(|| format!("invalid integer {key}"))
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.)
}
fn json_file(path: &Path) -> Result<Value> {
    let bytes =
        fs::read(path).with_context(|| format!("read scalar metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("read scalar metadata {}", path.display()))
}
fn files(directory: &Path) -> Result<Value> {
    Ok(
        json!({"conditions.json":v3_output::fingerprint(&directory.join("conditions.json"))?,
    "samples.csv":v3_output::fingerprint(&directory.join("samples.csv"))?}),
    )
}
fn membership(directory: &Path, summary: bool) -> Result<()> {
    ensure!(
        directory.is_dir() && !directory.is_symlink(),
        "invalid scalar directory"
    );
    let mut names = BTreeSet::new();
    for e in fs::read_dir(directory)? {
        let p = e?.path();
        ensure!(p.is_file() && !p.is_symlink(), "nonregular scalar member");
        names.insert(p.file_name().unwrap().to_string_lossy().into_owned());
    }
    let mut expected = BTreeSet::from(["conditions.json".to_owned(), "samples.csv".to_owned()]);
    if summary {
        expected.insert("summary.json".into());
    }
    ensure!(names == expected, "scalar-only file membership mismatch");
    Ok(())
}
fn blank() -> Value {
    columns()
        .into_iter()
        .map(|k| (k.to_owned(), Value::Null))
        .collect()
}
fn cached_row(s: &AkltStepSampledSettings, parent: &Value) -> Value {
    let mut r = blank();
    for k in v3_output::PHYSICAL.iter().chain(EXTRA.iter()) {
        r[*k] = parent[*k].clone();
    }
    r["sample_index"] = 0.into();
    r["completed_steps"] = 50.into();
    r["beta_actual"] = s.request().restart.progress.beta.into();
    r["temperature"] = (1. / s.request().restart.progress.beta).into();
    r["accumulated_log_norm"] = s.request().restart.progress.accumulated_log_norm.into();
    r["origin"] = "cached_parent".into();
    r["available"] = true.into();
    r
}
fn observation(r: &StepSample<v3::AkltValuesV3>) -> Result<Value> {
    let mut row = blank();
    row["sample_index"] = r.sample_index.into();
    row["completed_steps"] = r.completed_steps.into();
    row["beta_actual"] = r.beta_actual.into();
    row["temperature"] = r.temperature.into();
    row["accumulated_log_norm"] = r.accumulated_log_norm.into();
    row["origin"] = "measured".into();
    row["available"] = r.available.into();
    row["reason"] = serde_json::to_value(&r.reason)?;
    let cap = serde_json::to_value(r.cap.as_ref().context("missing observed cap")?)?;
    for (key, field) in [
        ("maximum_dimension_seen", "maximum_dimension_seen"),
        ("cap_ever_reached", "ever_reached"),
        ("cap_observed_strang_steps", "observed_strang_steps"),
        ("cap_coverage", "coverage"),
    ] {
        row[key] = cap[field].clone();
    }
    if let Some(values) = &r.values {
        let physical = serde_json::to_value(&values.physical)?;
        for key in v3_output::PHYSICAL {
            row[*key] = physical[*key].clone();
        }
        row["heat_available"] = values.heat.available.into();
        row["heat_reason"] = serde_json::to_value(&values.heat.reason)?;
        if let Some(h) = &values.heat.report {
            row["heat_capacity_variance"] = h.specific_heat_per_site.into();
            row["heat_diagnostics"] = serde_json::to_value(h)?;
        }
    } else {
        row["heat_available"] = false.into();
        row["heat_reason"] = serde_json::to_value(&r.reason)?;
    }
    Ok(row)
}
fn write_rows(directory: &Path, rows: &[Value]) -> Result<()> {
    let columns = columns();
    let mut csv = columns.join(",") + "\n";
    for row in rows {
        let fields = columns
            .iter()
            .map(|k| match &row[*k] {
                Value::Null => String::new(),
                Value::String(s) => quote_csv(s),
                v if *k == "heat_diagnostics" => quote_csv(&v.to_string()),
                v => v.to_string(),
            })
            .collect::<Vec<_>>();
        csv.push_str(&fields.join(","));
        csv.push('\n');
    }
    atomic_write(&directory.join("samples.csv"), csv.as_bytes())
}
fn read_rows(directory: &Path) -> Result<Vec<Value>> {
    let csv = parse_csv(&fs::read_to_string(directory.join("samples.csv"))?)?;
    let columns = columns();
    ensure!(
        csv.first()
            .is_some_and(|v| v.iter().map(String::as_str).eq(columns.iter().copied())),
        "step scalar CSV header mismatch"
    );
    ensure!(
        (2..=484).contains(&csv.len()),
        "step scalar row count invalid"
    );
    csv.iter()
        .skip(1)
        .map(|fields| {
            ensure!(fields.len() == columns.len(), "malformed step scalar row");
            columns
                .iter()
                .zip(fields)
                .map(|(k, v)| {
                    let value = if v.is_empty() {
                        Value::Null
                    } else if ["origin", "reason", "cap_coverage", "heat_reason"].contains(k) {
                        Value::String(v.clone())
                    } else {
                        serde_json::from_str(v)
                            .with_context(|| format!("invalid step scalar {k}"))?
                    };
                    Ok(((*k).to_owned(), value))
                })
                .collect::<Result<Value>>()
        })
        .collect()
}
fn validate_rows(s: &AkltStepSampledSettings, rows: &[Value], initial: &Value) -> Result<()> {
    ensure!(
        !rows.is_empty() && rows.len() <= 483,
        "step scalar prefix count invalid"
    );
    ensure!(
        rows[0] == cached_row(s, initial),
        "cached starting scalar row changed"
    );
    let mut previous = s.request().restart.cap.clone();
    for (i, (r, p)) in rows.iter().zip(s.points()).enumerate() {
        ensure!(
            r.as_object().is_some_and(|o| o.len() == columns().len()),
            "step scalar schema mismatch"
        );
        ensure!(
            count(r, "sample_index")? == i as u64
                && count(r, "completed_steps")? == p.completed_steps
                && number(r, "beta_actual")?.to_bits() == p.beta_expected.to_bits()
                && number(r, "temperature")?.to_bits() == (1. / p.beta_expected).to_bits()
                && r["origin"] == if i == 0 { "cached_parent" } else { "measured" },
            "step scalar identity/beta mismatch"
        );
        let beta = number(r, "beta_actual")?;
        let norm = number(r, "accumulated_log_norm")?;
        let available = r["available"]
            .as_bool()
            .context("invalid physical availability")?;
        if available {
            ensure!(
                r["reason"].is_null(),
                "available physical value has failure reason"
            );
            let v: AkltSampleValues = serde_json::from_value(
                v3_output::PHYSICAL
                    .iter()
                    .map(|k| ((*k).to_owned(), r[*k].clone()))
                    .collect(),
            )?;
            v.validate()?;
            ensure!(
                close(v.free_energy, -(3_f64.ln() + norm / 2.) / beta)
                    && close(v.entropy, (v.energy - v.free_energy) * beta),
                "step thermodynamic identity mismatch"
            );
            ensure!(
                v.cap_reached_now == (v.bond_dimension_ab >= 256 || v.bond_dimension_ba >= 256),
                "instantaneous cap mismatch"
            );
        } else {
            ensure!(
                r["reason"].as_str().is_some_and(|x| !x.trim().is_empty())
                    && v3_output::PHYSICAL.iter().all(|k| r[*k].is_null()),
                "unavailable physical evidence contradictory"
            );
        }
        v3_output::validate_heat(r)?;
        ensure!(
            (available && r["heat_available"] == true) || i + 1 == rows.len(),
            "observations after a failed measurement"
        );
        let cap = v3_output::cap_from(r)?;
        v3_output::validate_cap(&cap, p.completed_steps)?;
        ensure!(
            cap.coverage == CapCoverage::CompleteThroughStop
                && cap.observed_strang_steps == p.completed_steps
                && cap.maximum_dimension_seen >= previous.maximum_dimension_seen
                && (previous.ever_reached != Some(true) || cap.ever_reached == Some(true)),
            "observed cap history mismatch"
        );
        if available {
            ensure!(
                cap.maximum_dimension_seen.unwrap_or(0)
                    >= count(r, "bond_dimension_ab")?.max(count(r, "bond_dimension_ba")?) as usize,
                "observed cap below current dimensions"
            );
        }
        previous = cap;
    }
    Ok(())
}
fn read_conditions(directory: &Path) -> Result<(AkltStepSampledSettings, Value)> {
    let c = json_file(&directory.join("conditions.json"))?;
    let request: AkltStepSampledRequest = serde_json::from_value(c["request"].clone())?;
    let s = v5::resolve_request(&request)?;
    let runtime: v3::RuntimeEvidence = serde_json::from_value(c["runtime_at_launch"].clone())?;
    ensure!(
        runtime.thread_hint > 0
            && runtime.visible_cpu_count > 0
            && !runtime.provider.is_empty()
            && runtime
                .environment
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from([
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"
                ]),
        "invalid launch runtime"
    );
    ensure!(
        serde_json::to_value(runtime)? == c["runtime_at_launch"],
        "launch runtime schema mismatch"
    );
    let mut expected = v5::preflight(&request)?;
    for k in ["runtime_at_launch", "cached_start", "input_verification"] {
        expected[k] = c[k].clone();
    }
    ensure!(c == expected, "scalar conditions mismatch");
    v5_input::check_cached_parent(&s, directory, &c["cached_start"], &c["input_verification"])?;
    Ok((s, c))
}
fn beta_at(s: &AkltStepSampledSettings, step: u64) -> Result<f64> {
    ensure!(
        (50..=1014).contains(&step),
        "invalid native completed progress"
    );
    let mut beta = s.request().restart.progress.beta;
    let mut n = 50;
    for seg in s.remaining_segments() {
        for _ in 0..seg.steps {
            if n == step {
                return Ok(beta);
            }
            beta += 2. * seg.dtau;
            n += 1;
        }
    }
    Ok(beta)
}
fn make_summary(
    s: &AkltStepSampledSettings,
    directory: &Path,
    c: &Value,
    rows: &[Value],
    status: &str,
    reason: &str,
    progress: Option<&ItebdCheckpointProgress>,
    native_cap: Option<&CapEvidence>,
    elapsed: Option<f64>,
) -> Result<Value> {
    ensure!(
        matches!(status, "complete" | "partial") && !reason.trim().is_empty(),
        "invalid scalar stop status/reason"
    );
    let last = rows.last().context("missing scalar prefix")?;
    let last_step = count(last, "completed_steps")?;
    let available = rows.iter().filter(|r| r["available"] == true).count();
    let heats = rows.iter().filter(|r| r["heat_available"] == true).count();
    let mut cap = v3_output::cap_from(last)?;
    let lower = if let Some(p) = progress {
        ensure!(
            p.completed_steps >= last_step
                && p.completed_steps <= 1014
                && p.beta.to_bits() == beta_at(s, p.completed_steps)?.to_bits()
                && p.accumulated_log_norm.is_finite(),
            "native progress contradicts scalar prefix"
        );
        if p.completed_steps == last_step {
            ensure!(
                p.accumulated_log_norm.to_bits() == number(last, "accumulated_log_norm")?.to_bits(),
                "native norm differs from scalar boundary"
            );
        }
        cap = native_cap.context("missing native cap")?.clone();
        v3_output::validate_cap(&cap, p.completed_steps)?;
        let before = v3_output::cap_from(last)?;
        ensure!(
            cap.maximum_dimension_seen >= before.maximum_dimension_seen
                && cap.observed_strang_steps >= before.observed_strang_steps
                && (before.ever_reached != Some(true) || cap.ever_reached == Some(true)),
            "native cap history below scalar prefix"
        );
        p.completed_steps
    } else {
        ensure!(
            status == "partial",
            "interrupted scalar result cannot be complete"
        );
        cap.mark_unobserved();
        last_step
    };
    if status == "complete" {
        ensure!(
            rows.len() == 483
                && available == 483
                && heats == 483
                && lower == 1014
                && cap.coverage == CapCoverage::CompleteThroughStop,
            "complete scalar result lacks full acquisition"
        );
    }
    ensure!(
        elapsed.is_none_or(|x| x.is_finite() && x >= 0.),
        "invalid scalar elapsed time"
    );
    Ok(
        json!({"schema_version":5,"contract":v5::CONTRACT_V5,"request":s.request(),"status":status,"reason":reason,
        "planned_samples":483,"cached_samples":1,"new_measurements":rows.len()-1,
        "available_samples":available,"available_heat_samples":heats,"last_observed_step":last_step,
        "completed_steps":progress.map(|p|p.completed_steps),"completed_steps_kind":if progress.is_some(){"exact"}else{"lower_bound"},
        "completed_steps_lower_bound":lower,"beta_actual":progress.map(|p|p.beta),"accumulated_log_norm":progress.map(|p|p.accumulated_log_norm),
        "cap":cap,"elapsed_seconds":elapsed,"runtime_at_launch":c["runtime_at_launch"],
        "output_state_retained":false,"checkpoint_verified":false,"files":files(directory)?}),
    )
}
fn inspect(directory: &Path, summary: &Value, has_summary: bool) -> Result<Value> {
    membership(directory, has_summary)?;
    let (s, c) = read_conditions(directory)?;
    let rows = read_rows(directory)?;
    validate_rows(&s, &rows, &c["cached_start"])?;
    let status = summary["status"]
        .as_str()
        .context("missing scalar status")?;
    let reason = summary["reason"]
        .as_str()
        .context("missing scalar stop reason")?;
    let cap: CapEvidence = serde_json::from_value(summary["cap"].clone())?;
    let progress = match summary["completed_steps_kind"].as_str() {
        Some("exact") => Some(ItebdCheckpointProgress {
            completed_steps: count(summary, "completed_steps")?,
            beta: number(summary, "beta_actual")?,
            accumulated_log_norm: number(summary, "accumulated_log_norm")?,
            last_step: None,
        }),
        Some("lower_bound") => None,
        _ => bail!("invalid scalar progress kind"),
    };
    let elapsed = if summary["elapsed_seconds"].is_null() {
        None
    } else {
        Some(number(summary, "elapsed_seconds")?)
    };
    let expected = make_summary(
        &s,
        directory,
        &c,
        &rows,
        status,
        reason,
        progress.as_ref(),
        Some(&cap),
        elapsed,
    )?;
    ensure!(*summary == expected, "scalar summary/identity mismatch");
    Ok(
        json!({"schema_version":5,"contract":v5::CONTRACT_V5,"status":status,"reason":reason,
        "sample_count":rows.len(),"available_samples":summary["available_samples"],"available_heat_samples":summary["available_heat_samples"],
        "new_measurements":rows.len()-1,"cached_samples":1,"completed_steps":summary["completed_steps"],
        "completed_steps_kind":summary["completed_steps_kind"],"completed_steps_lower_bound":summary["completed_steps_lower_bound"],
        "request":s.request(),"output_state_retained":false,"checkpoint_verified":false,
        "scalar_contract_verified":true,"posthoc_state_scalar_consistency_verified":false,"variance_remeasured":false,
        "input_verification":c["input_verification"]}),
    )
}
struct Publisher<'a> {
    settings: &'a AkltStepSampledSettings,
    directory: PathBuf,
    initial: Value,
    rows: Vec<Value>,
}
impl StepObserver<v3::AkltValuesV3> for Publisher<'_> {
    fn sample(&mut self, sample: &StepSample<v3::AkltValuesV3>) -> Result<()> {
        self.rows.push(observation(sample)?);
        validate_rows(self.settings, &self.rows, &self.initial)?;
        write_rows(&self.directory, &self.rows)
    }
}
pub fn run_to_directory(r: &AkltStepSampledRequest, directory: &Path) -> Result<Value> {
    for k in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        ensure!(
            std::env::var_os(k).is_none(),
            "step-sampled production requires native defaults; unset {k}"
        );
    }
    let start = std::time::Instant::now();
    publish_with_engine(
        r,
        directory,
        &mut NativeEngine(v3::measure_v3_reference),
        &mut || start.elapsed().as_secs_f64(),
    )
}
pub(crate) fn publish_with_engine(
    r: &AkltStepSampledRequest,
    directory: &Path,
    engine: &mut dyn Engine<v3::AkltValuesV3>,
    elapsed: &mut dyn FnMut() -> f64,
) -> Result<Value> {
    let s = v5::resolve_request(r)?;
    let admitted = v5_input::load_start(&s, directory)?;
    if let Some(parent) = directory.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(directory).context("reserve scalar-only directory")?;
    let mut c = v5::preflight(r)?;
    c["runtime_at_launch"] = serde_json::to_value(v3::capture_runtime()?)?;
    c["cached_start"] = admitted.cached_row.clone();
    c["input_verification"] = admitted.verification;
    atomic_write(
        &directory.join("conditions.json"),
        &serde_json::to_vec_pretty(&c)?,
    )?;
    let initial = admitted.cached_row;
    let row = cached_row(&s, &initial);
    let mut publisher = Publisher {
        settings: &s,
        directory: directory.to_path_buf(),
        initial,
        rows: vec![row],
    };
    validate_rows(&s, &publisher.rows, &publisher.initial)?;
    write_rows(directory, &publisher.rows)?;
    let outcome = run_step_sampled_from(
        StepTrajectorySettings {
            hamiltonian: s.hamiltonian(),
            truncation: s.truncation(),
            segments: s.remaining_segments(),
            points: s.new_points(),
            policy: RunPolicy::Aklt {
                max_seconds: None,
                bond_cap: 256,
            },
        },
        &mut publisher,
        engine,
        elapsed,
        admitted.start,
    )?;
    ensure!(
        outcome.samples.len() + 1 == publisher.rows.len(),
        "scalar publication count drift"
    );
    let summary = make_summary(
        &s,
        directory,
        &c,
        &publisher.rows,
        &outcome.status,
        &outcome.reason,
        Some(&outcome.progress),
        outcome.cap.as_ref(),
        Some(outcome.elapsed_seconds),
    )?;
    let verified = inspect(directory, &summary, false)?;
    atomic_write(
        &directory.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(verified)
}
pub fn verify_directory(directory: &Path) -> Result<Value> {
    inspect(
        directory,
        &json_file(&directory.join("summary.json"))?,
        true,
    )
    .with_context(|| format!("verify step-sampled scalars {}", directory.display()))
}
pub fn finalize_interrupted_directory(directory: &Path, why: &str) -> Result<Value> {
    ensure!(
        !why.trim().is_empty(),
        "scalar interruption requires actual reason"
    );
    membership(directory, false)?;
    let (s, c) = read_conditions(directory)?;
    let rows = read_rows(directory)?;
    validate_rows(&s, &rows, &c["cached_start"])?;
    let summary = make_summary(&s, directory, &c, &rows, "partial", why, None, None, None)?;
    let verified = inspect(directory, &summary, false)?;
    atomic_write(
        &directory.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aklt_reference::v3::AkltValuesV3;
    use crate::aklt_reference::{v3, v3_output, v5_input::test_support::parent_fixture};
    use crate::{
        itebd::StepInfo,
        itebd_auto::{ItebdHamiltonian, ItebdState},
        reference_output::parse_csv,
        tensor::Truncation,
    };
    use serde_json::json;
    use std::{collections::BTreeSet, fs, path::PathBuf};

    struct FixtureEngine {
        steps: usize,
        measures: usize,
        stop_at: Option<usize>,
        panic_step: Option<usize>,
        block_csv: Option<PathBuf>,
        template: AkltValuesV3,
    }
    impl FixtureEngine {
        fn new(parent: &Path) -> Self {
            let row = v3_output::read_rows(parent).unwrap().remove(0);
            let physical = serde_json::from_value(
                v3_output::PHYSICAL
                    .iter()
                    .map(|k| ((*k).to_string(), row[*k].clone()))
                    .collect::<Value>(),
            )
            .unwrap();
            let heat = v3::VarianceHeatEvidence {
                available: true,
                reason: None,
                report: Some(serde_json::from_value(row["heat_diagnostics"].clone()).unwrap()),
            };
            Self {
                steps: 0,
                measures: 0,
                stop_at: None,
                panic_step: None,
                block_csv: None,
                template: AkltValuesV3 { physical, heat },
            }
        }
    }
    impl Engine<AkltValuesV3> for FixtureEngine {
        fn setup(&mut self, _: &ItebdHamiltonian) -> Result<ItebdState> {
            bail!("beta-zero setup forbidden")
        }
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &Truncation,
        ) -> Result<StepInfo> {
            self.steps += 1;
            assert_ne!(
                self.panic_step,
                Some(self.steps),
                "controlled worker death after odd step"
            );
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
            _: &ItebdState,
            _: &ItebdHamiltonian,
            beta: f64,
            norm: f64,
        ) -> Result<AkltValuesV3> {
            self.measures += 1;
            if let Some(path) = self.block_csv.take() {
                fs::remove_file(&path)?;
                fs::create_dir(&path)?;
            }
            let mut value = self.template.clone();
            value.physical.free_energy = -(3_f64.ln() + norm / 2.) / beta;
            value.physical.entropy = (value.physical.energy - value.physical.free_energy) * beta;
            let report = value.heat.report.as_mut().unwrap();
            report.specific_heat_per_site = beta * beta * report.energy_variance_per_site;
            if self.stop_at == Some(self.measures) {
                value.heat = v3::VarianceHeatEvidence {
                    available: false,
                    reason: Some("fixture variance tail failure".into()),
                    report: None,
                };
            }
            Ok(value)
        }
    }
    fn summary(path: &Path) -> Value {
        serde_json::from_slice(&fs::read(path.join("summary.json")).unwrap()).unwrap()
    }
    #[test]
    fn scalar_only_complete_has483_rows_and_three_files() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        let mut engine = FixtureEngine::new(&temp.path().join("parent"));
        let result = publish_with_engine(&r, &out, &mut engine, &mut || 0.);
        assert!(
            result.is_ok(),
            "scalar publication failed: {:?}",
            result.err()
        );
        let verified = result.unwrap();
        assert_eq!(engine.steps, 964);
        assert_eq!(engine.measures, 482);
        let rows = parse_csv(&fs::read_to_string(out.join("samples.csv")).unwrap()).unwrap();
        assert_eq!(rows.len(), 484);
        let origin = rows[0].iter().position(|s| s == "origin").unwrap();
        assert_eq!(rows[1][origin], "cached_parent");
        let names = fs::read_dir(&out)
            .unwrap()
            .map(|p| p.unwrap().file_name().into_string().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names,
            BTreeSet::from(["conditions.json", "samples.csv", "summary.json"].map(str::to_owned))
        );
        let s = summary(&out);
        assert_eq!(s["new_measurements"], 482);
        assert_eq!(s["cached_samples"], 1);
        assert_eq!(s["completed_steps"], 1014);
        assert_eq!(s["status"], "complete");
        assert_eq!(verified["checkpoint_verified"], false);
        assert_eq!(verified["output_state_retained"], false);
        assert!(verify_directory(&out).is_ok());
        assert!(!temp.path().join("svd-failure.h5").exists());
    }
    #[test]
    fn scalar_only_heat_failure_retains_physical_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        let mut engine = FixtureEngine::new(&temp.path().join("parent"));
        engine.stop_at = Some(1);
        publish_with_engine(&r, &out, &mut engine, &mut || 0.).unwrap();
        let s = summary(&out);
        assert_eq!(s["status"], "partial");
        assert_eq!(s["completed_steps"], 52);
        assert_eq!(s["available_samples"], 2);
        assert_eq!(s["available_heat_samples"], 1);
        assert_eq!(engine.steps, 2);
        assert!(s["reason"].as_str().unwrap().contains("variance tail"));
        assert!(verify_directory(&out).is_ok());
    }
    #[test]
    fn scalar_only_real_write_failure() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        let mut engine = FixtureEngine::new(&temp.path().join("parent"));
        engine.block_csv = Some(out.join("samples.csv"));
        let error = format!(
            "{:#}",
            publish_with_engine(&r, &out, &mut engine, &mut || 0.).unwrap_err()
        );
        assert!(
            error.contains("publish") && error.contains("samples.csv"),
            "wrong I/O context: {error}"
        );
        assert!(!out.join("summary.json").exists());
        assert!(!out.join("states.h5").exists());
    }
    #[test]
    fn scalar_only_interruption_has_unknown_tail() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        let mut engine = FixtureEngine::new(&temp.path().join("parent"));
        engine.panic_step = Some(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            publish_with_engine(&r, &out, &mut engine, &mut || 0.)
        }));
        assert!(
            result.is_err(),
            "worker death fixture did not reach evolution"
        );
        assert_eq!(engine.steps, 4);
        finalize_interrupted_directory(&out, "confirmed exited worker").unwrap();
        let s = summary(&out);
        assert_eq!(s["completed_steps"], Value::Null);
        assert_eq!(s["completed_steps_lower_bound"], 52);
        assert_eq!(s["last_observed_step"], 52);
        assert_eq!(s["cap"]["coverage"], "unobserved_tail");
        assert_eq!(s["cap"]["ever_reached"], Value::Null);
        assert_eq!(engine.steps, 4);
        assert_eq!(s["output_state_retained"], false);
        assert!(verify_directory(&out).is_ok());
    }
    #[test]
    fn scalar_only_rejects_semantic_drift_after_rehash() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        let mut engine = FixtureEngine::new(&temp.path().join("parent"));
        engine.stop_at = Some(1);
        publish_with_engine(&r, &out, &mut engine, &mut || 0.).unwrap();
        let original = fs::read(out.join("summary.json")).unwrap();
        let mut s: Value = serde_json::from_slice(&original).unwrap();
        s["checkpoint_verified"] = true.into();
        fs::write(out.join("summary.json"), serde_json::to_vec(&s).unwrap()).unwrap();
        assert!(verify_directory(&out).is_err());
        fs::write(out.join("summary.json"), &original).unwrap();
        let mut rows = parse_csv(&fs::read_to_string(out.join("samples.csv")).unwrap()).unwrap();
        let index = rows[0].iter().position(|k| k == "completed_steps").unwrap();
        rows[2][index] = "53".into();
        let csv = rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|x| crate::reference_output::quote_csv(x))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(out.join("samples.csv"), csv).unwrap();
        let mut s: Value = serde_json::from_slice(&original).unwrap();
        s["files"]["samples.csv"] = v3_output::fingerprint(&out.join("samples.csv")).unwrap();
        fs::write(out.join("summary.json"), serde_json::to_vec(&s).unwrap()).unwrap();
        assert!(verify_directory(&out).is_err());
        fs::write(out.join("extra.h5"), b"unrequested").unwrap();
        assert!(verify_directory(&out).is_err());
        assert_eq!(s["planned_samples"], json!(483));
    }
}
