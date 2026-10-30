//! Selected, read-only admission for the step-sampled scalar contract.
use super::{
    v3, v3_output,
    v5::{self, resolve_request, AkltStepSampledRequest, AkltStepSampledSettings},
};
use crate::{
    config::TrotterOrder,
    itebd_auto::ItebdHamiltonian,
    itebd_checkpoint::{load_itebd_checkpoint, ItebdTimeGrid},
    reference_trajectory::ScheduledStart,
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

pub(crate) struct AdmittedStepStart {
    pub start: ScheduledStart,
    pub cached_row: Value,
    pub verification: Value,
}

pub(super) fn parent_directory(destination: &Path, relative: &str) -> Result<PathBuf> {
    let destination = std::path::absolute(destination)?;
    let mut base = destination
        .parent()
        .context("missing destination parent")?
        .to_path_buf();
    let mut missing = Vec::new();
    while !base.exists() {
        ensure!(!base.is_symlink(), "symlink destination ancestor");
        missing.push(
            base.file_name()
                .context("missing destination ancestor")?
                .to_os_string(),
        );
        ensure!(base.pop(), "destination ancestor escape");
    }
    ensure!(!base.is_symlink(), "symlink destination parent");
    // Resolve platform aliases (/var, /tmp) before walking the relative input.
    base = base.canonicalize()?;
    for name in missing.iter().rev() {
        base.push(name);
    }
    base.push(
        destination
            .file_name()
            .context("missing destination name")?,
    );
    for component in Path::new(relative).components() {
        match component {
            Component::ParentDir => ensure!(base.pop(), "step-sampled parent path escape"),
            Component::CurDir => {}
            Component::Normal(name) => base.push(name),
            _ => bail!("step-sampled parent must be relative"),
        }
        ensure!(!base.is_symlink(), "symlink step-sampled parent path");
    }
    Ok(base)
}

fn check_pins(s: &AkltStepSampledSettings, parent: &Path) -> Result<()> {
    for (name, pin) in &s.request().restart.files {
        let actual = v3_output::fingerprint(&parent.join(name))
            .with_context(|| format!("step-sampled parent pin {name}"))?;
        ensure!(
            actual == serde_json::to_value(pin)?,
            "step-sampled parent pin changed: {name}"
        );
    }
    Ok(())
}

pub(crate) fn load_start(
    s: &AkltStepSampledSettings,
    destination: &Path,
) -> Result<AdmittedStepStart> {
    ensure!(
        !destination.exists() && !destination.is_symlink(),
        "step-sampled destination exists"
    );
    let r = &s.request().restart;
    let parent = parent_directory(destination, &r.directory)?;
    check_pins(s, &parent)?;
    let conditions: Value = serde_json::from_slice(&fs::read(parent.join("conditions.json"))?)?;
    ensure!(
        conditions["schema_version"] == 3 && conditions["contract"] == v3::CONTRACT_V3,
        "step-sampled parent must be original v3"
    );
    let original: v3::AkltReferenceRequestV3 =
        serde_json::from_value(conditions["request"].clone())?;
    let mut expected = v3::preflight_v3(&original)?;
    expected["runtime_at_launch"] = conditions["runtime_at_launch"].clone();
    ensure!(
        expected == conditions,
        "step-sampled parent controls/grid changed"
    );
    let rows = v3_output::read_rows(&parent).context("read selected parent scalar table")?;
    ensure!(rows.len() == 20, "step-sampled parent row count mismatch");
    let row = &rows[0];
    ensure!(
        row["temperature_tick"] == 20
            && row["snapshot_available"] == true
            && row["snapshot_index"] == 0
            && row["completed_steps"] == 50
            && row["available"] == true
            && row["heat_available"] == true,
        "step-sampled parent selected row incomplete"
    );
    v3_output::validate_heat(row).context("selected parent variance evidence")?;
    ensure!(
        v3_output::cap_from(row)? == r.cap,
        "selected parent cap differs from boundary row"
    );
    let number = |value: &Value, key: &str| -> Result<f64> {
        let value = value[key]
            .as_f64()
            .with_context(|| format!("missing selected parent {key}"))?;
        ensure!(value.is_finite(), "nonfinite selected parent {key}");
        Ok(value)
    };
    let summary: Value = serde_json::from_slice(&fs::read(parent.join("summary.json"))?)?;
    ensure!(
        summary["contract"] == v3::CONTRACT_V3
            && matches!(summary["status"].as_str(), Some("partial" | "complete")),
        "invalid selected parent summary"
    );
    for name in ["conditions.json", "samples.csv", "states.h5"] {
        ensure!(
            summary["files"][name] == serde_json::to_value(&r.files[name])?,
            "selected parent summary pin mismatch: {name}"
        );
    }
    let mappings = summary["snapshot_mapping"]
        .as_array()
        .context("missing parent snapshot mapping")?;
    let selected = mappings
        .iter()
        .filter(|m| m["snapshot_index"] == 0)
        .collect::<Vec<_>>();
    ensure!(
        selected.len() == 1
            && selected[0]["completed_steps"] == 50
            && selected[0]["temperature_tick"] == 20
            && number(selected[0], "beta_actual")?.to_bits() == r.progress.beta.to_bits()
            && number(selected[0], "accumulated_log_norm")?.to_bits()
                == r.progress.accumulated_log_norm.to_bits(),
        "selected parent progress mapping mismatch"
    );
    let loaded = load_itebd_checkpoint(&parent.join("states.h5"), 0, &Default::default())
        .context("load step-sampled parent snapshot0")?;
    ensure!(
        loaded.progress.completed_steps == 50
            && loaded.progress.beta.to_bits() == r.progress.beta.to_bits()
            && loaded.progress.accumulated_log_norm.to_bits()
                == r.progress.accumulated_log_norm.to_bits()
            && number(row, "beta_actual")?.to_bits() == r.progress.beta.to_bits()
            && number(row, "accumulated_log_norm")?.to_bits()
                == r.progress.accumulated_log_norm.to_bits(),
        "selected parent progress differs from retained state/row"
    );
    ensure!(
        loaded.time_grid == ItebdTimeGrid::Segmented(s.full_segments().to_vec()),
        "selected parent time grid changed"
    );
    let metadata = &loaded.metadata;
    ensure!(
        metadata.git_revision.as_deref() == Some(original.source_revision.as_str()),
        "selected parent HDF5 source differs from historical producer"
    );
    ensure!(
        metadata.dtau == 0.005
            && metadata.trotter_order == TrotterOrder::Second
            && metadata.truncation.epsilon == 1e-13
            && metadata.truncation.max_bond == Some(256)
            && metadata.canonicalize_every == 1
            && metadata.record_every_beta.is_none()
            && metadata.model_label.as_deref() == Some("aklt_projector")
            && metadata.hermiticity_tolerance == 1e-12,
        "selected parent HDF5 settings changed"
    );
    match (&loaded.hamiltonian, s.hamiltonian()) {
        (ItebdHamiltonian::Real(a), ItebdHamiltonian::Real(b)) => ensure!(
            a.two_site_h == b.two_site_h && a.site_energy == b.site_energy,
            "selected parent Hamiltonian changed"
        ),
        _ => bail!("selected parent backend changed"),
    }
    // Physical consistency only. The cached variance is validated, never remeasured.
    let measured = super::measure_reference_with_cap(
        &loaded.state,
        s.hamiltonian(),
        loaded.progress.beta,
        loaded.progress.accumulated_log_norm,
        256,
    )?;
    let values = serde_json::to_value(measured)?;
    for key in ["energy", "mx", "mz", "free_energy", "entropy"] {
        let (a, b) = (number(row, key)?, number(&values, key)?);
        ensure!(
            (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.),
            "selected parent physical scalar mismatch: {key}"
        );
    }
    ensure!(
        row["bond_dimension_ab"] == values["bond_dimension_ab"]
            && row["bond_dimension_ba"] == values["bond_dimension_ba"],
        "selected parent bond dimensions differ"
    );
    check_pins(s, &parent)?;
    Ok(AdmittedStepStart {
        start: ScheduledStart {
            state: loaded.state,
            progress: loaded.progress,
            cap: r.cap.clone(),
        },
        cached_row: row.clone(),
        verification: json!({"kind":"selected-parent-admission-v1","selected_snapshot_verified":true,
            "physical_scalar_consistency_passed":true,"variance_remeasured":false,
            "source_revision":original.source_revision,"snapshot_index":0,"parent_status":summary["status"]}),
    })
}

pub fn preflight_start(r: &AkltStepSampledRequest, destination: &Path) -> Result<Value> {
    let s = resolve_request(r)?;
    let admitted = load_start(&s, destination)?;
    let mut p = v5::preflight(r)?;
    p["input_verification"] = admitted.verification;
    p["cached_start"] = admitted.cached_row;
    Ok(p)
}

// Metadata/pin verification after publication does not load or remeasure a state.
pub(super) fn check_cached_parent(
    s: &AkltStepSampledSettings,
    destination: &Path,
    cached: &Value,
    proof: &Value,
) -> Result<()> {
    let parent = parent_directory(destination, &s.request().restart.directory)?;
    check_pins(s, &parent)?;
    let rows = v3_output::read_rows(&parent)?;
    ensure!(
        rows.len() == 20 && rows.first() == Some(cached),
        "selected parent cached row changed"
    );
    let conditions: Value = serde_json::from_slice(&fs::read(parent.join("conditions.json"))?)?;
    let summary: Value = serde_json::from_slice(&fs::read(parent.join("summary.json"))?)?;
    ensure!(
        proof
            == &json!({"kind":"selected-parent-admission-v1","selected_snapshot_verified":true,
        "physical_scalar_consistency_passed":true,"variance_remeasured":false,
        "source_revision":conditions["request"]["source_revision"],"snapshot_index":0,"parent_status":summary["status"]}),
        "selected parent admission evidence mismatch"
    );
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::aklt_reference::{v3, v3_output};
    use crate::itebd::StepInfo;
    use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
    use crate::reference_trajectory::Engine;
    use crate::tensor::Truncation;
    use serde_json::json;
    use std::{fs, path::PathBuf};

    pub(crate) struct ParentEngine {
        steps: usize,
    }
    impl Engine<v3::AkltValuesV3> for ParentEngine {
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &Truncation,
        ) -> Result<StepInfo> {
            self.steps += 1;
            anyhow::ensure!(self.steps <= 50, "controlled fixture interruption after T2");
            Ok(StepInfo {
                max_bond: 30,
                min_singular_value: 1.,
                log_norm: if self.steps == 1 {
                    -0.4937122532815281
                } else {
                    0.
                },
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
        ) -> Result<v3::AkltValuesV3> {
            v3::measure_v3_reference(s, h, b, n)
        }
    }
    pub(crate) fn refresh(r: &mut AkltStepSampledRequest, parent: &Path) {
        let mut summary: Value =
            serde_json::from_slice(&fs::read(parent.join("summary.json")).unwrap()).unwrap();
        for name in ["conditions.json", "samples.csv", "states.h5"] {
            summary["files"][name] = v3_output::fingerprint(&parent.join(name)).unwrap();
        }
        fs::write(
            parent.join("summary.json"),
            serde_json::to_vec_pretty(&summary).unwrap(),
        )
        .unwrap();
        for (name, pin) in &mut r.restart.files {
            *pin = serde_json::from_value(v3_output::fingerprint(&parent.join(name)).unwrap())
                .unwrap();
        }
    }
    // This is a static analytical state with labeled progress, not thermal evidence.
    pub(crate) fn parent_fixture(root: &Path) -> (AkltStepSampledRequest, PathBuf) {
        let parent = root.join("parent");
        let request = json!({"schema_version":3,"contract":v3::CONTRACT_V3,
            "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,
            "max_seconds":null,"recording_policy":v3::RECORDING_POLICY,"heat_capacity_policy":v3::HEAT_POLICY,
            "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)});
        let settings = v3::resolve_v3_request(&serde_json::from_value(request).unwrap()).unwrap();
        let verified = v3_output::publish_with_engine(
            &settings,
            &parent,
            &mut ParentEngine { steps: 0 },
            &mut || 0.,
        )
        .unwrap();
        assert_eq!(verified["status"], "partial");
        let row = v3_output::read_rows(&parent).unwrap().remove(0);
        // Rank-five projector: <h>=5/9; its partial trace is scalar, so
        // neighboring connected correlations vanish and var(H)/N=20/81.
        assert!((row["energy"].as_f64().unwrap() - 5. / 9.).abs() < 1e-12);
        let raw = row["heat_diagnostics"]["raw_energy_variance_per_site"]
            .as_f64()
            .unwrap();
        assert!((raw - 20. / 81.).abs() < 1e-12);
        let mut r: AkltStepSampledRequest = serde_json::from_value(json!({
            "schema_version":5,"contract":super::super::v5::CONTRACT_V5,
            "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,"max_seconds":null,
            "measurement_policy":super::super::v5::MEASUREMENT_POLICY,
            "recording_policy":super::super::v5::RECORDING_POLICY,"heat_capacity_policy":v3::HEAT_POLICY,
            "source_revision":"c".repeat(40),"worker_sha256":"d".repeat(64),
            "restart":{"directory":"../parent","snapshot_index":0,"files":{},
                "progress":{"completed_steps":50,"beta":row["beta_actual"],"accumulated_log_norm":row["accumulated_log_norm"]},
                "cap":{"maximum_dimension_seen":30,"ever_reached":false,"observed_strang_steps":50,"coverage":"complete_through_stop"}}
        })).unwrap();
        for name in [
            "conditions.json",
            "summary.json",
            "samples.csv",
            "states.h5",
        ] {
            r.restart.files.insert(
                name.into(),
                serde_json::from_value(v3_output::fingerprint(&parent.join(name)).unwrap())
                    .unwrap(),
            );
        }
        (r, root.join("output"))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{parent_fixture, refresh};
    use super::*;
    use std::fs;

    #[test]
    fn selected_parent_admission_uses_completed_row_not_failed_tail() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        let result = preflight_start(&r, &out);
        assert!(
            result.is_ok(),
            "valid completed input rejected: {:#?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(
            result["input_verification"]["selected_snapshot_verified"],
            true
        );
        assert_eq!(result["input_verification"]["variance_remeasured"], false);
        assert_eq!(
            result["cached_start"]["beta_actual"]
                .as_f64()
                .unwrap()
                .to_bits(),
            r.restart.progress.beta.to_bits()
        );
        assert_eq!(
            result["cached_start"]["accumulated_log_norm"]
                .as_f64()
                .unwrap()
                .to_bits(),
            r.restart.progress.accumulated_log_norm.to_bits()
        );
        assert_eq!(result["cached_start"]["maximum_dimension_seen"], 30);
        assert!(!out.exists());
    }

    #[test]
    fn selected_parent_rejects_pin_progress_cap_and_source_drift() {
        let temp = tempfile::tempdir().unwrap();
        let (r, out) = parent_fixture(temp.path());
        for kind in ["progress", "cap", "pin"] {
            let mut bad = r.clone();
            match kind {
                "progress" => bad.restart.progress.accumulated_log_norm += 0.01,
                "cap" => bad.restart.cap.maximum_dimension_seen = Some(31),
                _ => bad.restart.files.get_mut("states.h5").unwrap().sha256 = "0".repeat(64),
            }
            let error = format!("{:#}", preflight_start(&bad, &out).unwrap_err());
            assert!(error.contains(kind), "missing {kind} context: {error}");
            assert!(!out.exists());
        }
        let parent = temp.path().join("parent");
        let mut conditions: Value =
            serde_json::from_slice(&fs::read(parent.join("conditions.json")).unwrap()).unwrap();
        conditions["request"]["source_revision"] = "e".repeat(40).into();
        fs::write(
            parent.join("conditions.json"),
            serde_json::to_vec(&conditions).unwrap(),
        )
        .unwrap();
        let mut bad = r.clone();
        refresh(&mut bad, &parent);
        let error = format!("{:#}", preflight_start(&bad, &out).unwrap_err());
        assert!(error.contains("source"), "wrong source validation: {error}");
    }

    #[test]
    fn selected_parent_rejects_incomplete_snapshot_and_symlink() {
        let temp = tempfile::tempdir().unwrap();
        let (mut r, out) = parent_fixture(temp.path());
        let parent = temp.path().join("parent");
        {
            let file = hdf5_metno::File::open_rw(parent.join("states.h5")).unwrap();
            file.group("states/000000")
                .unwrap()
                .attr("complete")
                .unwrap()
                .write_scalar(&0_u8)
                .unwrap();
        }
        refresh(&mut r, &parent);
        let error = format!("{:#}", preflight_start(&r, &out).unwrap_err());
        assert!(
            error.contains("snapshot"),
            "wrong snapshot context: {error}"
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&parent, temp.path().join("linked")).unwrap();
            r.restart.directory = "../linked".into();
            let error = format!("{:#}", preflight_start(&r, &out).unwrap_err());
            assert!(error.contains("symlink"), "wrong symlink context: {error}");
        }
    }
}
