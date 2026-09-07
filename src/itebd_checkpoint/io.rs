use super::schema::{
    io_error, read_f64_vector, read_root, read_scalar_attribute, read_scalar_dataset,
    read_string_dataset, read_u8_vector, schema_error, validate_create_inputs, write_f64_vector,
    write_root, write_scalar_attribute, write_scalar_dataset, write_string_dataset,
    write_u8_vector, ValidatedRoot,
};
use super::{
    ItebdCheckpointEntry, ItebdCheckpointError, ItebdCheckpointLoadOptions,
    ItebdCheckpointProgress, ItebdRunMetadata, ItebdSnapshotDiagnostics, ItebdStepDiagnostics,
    LoadedItebdCheckpoint,
};
use crate::itebd::StepInfo;
use crate::itebd_auto::{ItebdHamiltonian, ItebdState};
use crate::itebd_complex::{ComplexPurifiedMps, ComplexSite};
use crate::itebd_state_view::{ItebdScalarType, ItebdStateRef, ItebdStateSummary};
use crate::purified_mps::{PurifiedMps, Site};
use crate::tensor::{Idx, Tensor};
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, H5Type};
use num_complex::Complex64;
use std::path::{Path, PathBuf};

pub struct ItebdTrajectoryWriter {
    path: PathBuf,
    tensor_path: String,
    metadata: ItebdRunMetadata,
    scalar_type: ItebdScalarType,
    physical_dim: usize,
    ancilla_dim: usize,
    next_index: u64,
    previous_progress: Option<(u64, f64)>,
    poisoned: bool,
    #[cfg(test)]
    failure_point: Option<WriteFailurePoint>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteFailurePoint {
    MarkerCreation,
    GammaA,
    GammaB,
    FinalMarker,
    FinalFlush,
}

impl ItebdTrajectoryWriter {
    pub fn create(
        path: impl AsRef<Path>,
        metadata: ItebdRunMetadata,
        hamiltonian: &ItebdHamiltonian,
    ) -> Result<Self, ItebdCheckpointError> {
        let path = path.as_ref().to_path_buf();
        let tensor_path = tensor_path(&path)?.to_string();
        let (scalar_type, physical_dim) = validate_create_inputs(&path, &metadata, hamiltonian)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| io_error(&path, error))?;
        }
        let file = File::create_excl(&path).map_err(|error| io_error(&path, error))?;
        write_root(
            &file,
            &path,
            &metadata,
            hamiltonian,
            scalar_type,
            physical_dim,
        )?;
        file.flush().map_err(|error| io_error(&path, error))?;
        Ok(Self {
            path,
            tensor_path,
            metadata,
            scalar_type,
            physical_dim,
            ancilla_dim: physical_dim,
            next_index: 0,
            previous_progress: None,
            poisoned: false,
            #[cfg(test)]
            failure_point: None,
        })
    }

    #[cfg(test)]
    fn inject_failure(&mut self, point: WriteFailurePoint) {
        self.failure_point = Some(point);
    }

    #[cfg(test)]
    fn fail_if_requested(&self, point: WriteFailurePoint) -> Result<(), &'static str> {
        if self.failure_point == Some(point) {
            Err("injected failure")
        } else {
            Ok(())
        }
    }

    pub fn append(
        &mut self,
        state: ItebdStateRef<'_>,
        progress: &ItebdCheckpointProgress,
    ) -> Result<ItebdCheckpointEntry, ItebdCheckpointError> {
        if self.poisoned {
            return Err(ItebdCheckpointError::WriterPoisoned {
                path: self.path.clone(),
            });
        }
        let index = self.next_index;
        let next_index = index.checked_add(1).ok_or_else(|| {
            snapshot_error(
                &self.path,
                index,
                "index",
                "cannot increment snapshot index beyond u64::MAX",
            )
        })?;
        let group_name = encode_snapshot_name(index);
        let summary = state
            .validate()
            .map_err(|source| ItebdCheckpointError::State {
                path: self.path.clone(),
                index,
                source,
            })?;
        validate_snapshot_against_writer(self, index, &summary)?;
        validate_progress(
            &self.path,
            index,
            &self.metadata,
            progress,
            &summary,
            self.previous_progress,
        )?;
        let diagnostics = diagnostics_from(progress, &summary);

        self.write_snapshot(&group_name, state, progress, &summary, &diagnostics)?;

        self.next_index = next_index;
        self.previous_progress = Some((progress.completed_steps, progress.beta));
        Ok(ItebdCheckpointEntry {
            index,
            beta: progress.beta,
            completed_steps: progress.completed_steps,
        })
    }

    pub fn finish(self) -> Result<(), ItebdCheckpointError> {
        if self.poisoned {
            return Err(ItebdCheckpointError::WriterPoisoned { path: self.path });
        }
        let file = File::open_rw(&self.path).map_err(|error| io_error(&self.path, error))?;
        file.flush().map_err(|error| io_error(&self.path, error))
    }

    fn write_snapshot(
        &mut self,
        group_name: &str,
        state: ItebdStateRef<'_>,
        progress: &ItebdCheckpointProgress,
        summary: &ItebdStateSummary,
        diagnostics: &ItebdSnapshotDiagnostics,
    ) -> Result<(), ItebdCheckpointError> {
        let diagnostics_json = serde_json::to_string(diagnostics).map_err(|error| {
            snapshot_error(
                &self.path,
                self.next_index,
                "diagnostics_json",
                format!("cannot serialize diagnostics: {error}"),
            )
        })?;
        let group_path = format!("states/{group_name}");
        let initial_marker_stage =
            format!("create initial completion marker at {group_path}/@complete");
        let final_marker_stage =
            format!("update final completion marker at {group_path}/@complete");
        let result = (|| -> Result<(), ItebdCheckpointError> {
            let file = File::open_rw(&self.path).map_err(|error| {
                snapshot_stage_io_error(
                    &self.path,
                    self.next_index,
                    "open checkpoint for append",
                    error,
                )
            })?;
            let states = file.group("states").map_err(|error| {
                snapshot_field_io_error(&self.path, self.next_index, "states", error)
            })?;
            let group = states.create_group(group_name).map_err(|error| {
                snapshot_field_io_error(&self.path, self.next_index, &group_path, error)
            })?;
            #[cfg(test)]
            self.fail_if_requested(WriteFailurePoint::MarkerCreation)
                .map_err(|error| {
                    snapshot_stage_io_error(
                        &self.path,
                        self.next_index,
                        &initial_marker_stage,
                        error,
                    )
                })?;
            snapshot_stage_io(
                write_scalar_attribute(&group, &self.path, "complete", &0_u8),
                &self.path,
                self.next_index,
                &initial_marker_stage,
            )?;
            snapshot_field_io(
                write_scalar_dataset(&group, &self.path, "beta", &progress.beta),
                &self.path,
                self.next_index,
                &format!("{group_path}/beta"),
            )?;
            snapshot_field_io(
                write_scalar_dataset(
                    &group,
                    &self.path,
                    "completed_steps",
                    &progress.completed_steps,
                ),
                &self.path,
                self.next_index,
                &format!("{group_path}/completed_steps"),
            )?;
            snapshot_field_io(
                write_scalar_dataset(
                    &group,
                    &self.path,
                    "accumulated_log_norm",
                    &progress.accumulated_log_norm,
                ),
                &self.path,
                self.next_index,
                &format!("{group_path}/accumulated_log_norm"),
            )?;
            snapshot_field_io(
                write_string_dataset(&group, &self.path, "diagnostics_json", &diagnostics_json),
                &self.path,
                self.next_index,
                &format!("{group_path}/diagnostics_json"),
            )?;
            snapshot_field_io(
                write_u8_vector(&group, &self.path, "a_role_axes", &summary.a_role_axes),
                &self.path,
                self.next_index,
                &format!("{group_path}/a_role_axes"),
            )?;
            snapshot_field_io(
                write_u8_vector(&group, &self.path, "b_role_axes", &summary.b_role_axes),
                &self.path,
                self.next_index,
                &format!("{group_path}/b_role_axes"),
            )?;
            snapshot_field_io(
                write_f64_vector(&group, &self.path, "lambdaAB", state.lambda_ab()),
                &self.path,
                self.next_index,
                &format!("{group_path}/lambdaAB"),
            )?;
            snapshot_field_io(
                write_f64_vector(&group, &self.path, "lambdaBA", state.lambda_ba()),
                &self.path,
                self.next_index,
                &format!("{group_path}/lambdaBA"),
            )?;
            file.flush().map_err(|error| {
                snapshot_stage_io_error(
                    &self.path,
                    self.next_index,
                    "flush scalar snapshot fields",
                    error,
                )
            })?;
            drop(group);
            drop(states);
            drop(file);

            #[cfg(test)]
            self.fail_if_requested(WriteFailurePoint::GammaA)
                .map_err(|error| {
                    snapshot_field_io_error(
                        &self.path,
                        self.next_index,
                        &format!("{group_path}/GammaA"),
                        error,
                    )
                })?;
            tensor4all_hdf5::append_itensor(
                &self.tensor_path,
                &format!("{group_path}/GammaA"),
                state.a().gamma,
            )
            .map_err(|error| {
                snapshot_field_io_error(
                    &self.path,
                    self.next_index,
                    &format!("{group_path}/GammaA"),
                    error,
                )
            })?;
            #[cfg(test)]
            self.fail_if_requested(WriteFailurePoint::GammaB)
                .map_err(|error| {
                    snapshot_field_io_error(
                        &self.path,
                        self.next_index,
                        &format!("{group_path}/GammaB"),
                        error,
                    )
                })?;
            tensor4all_hdf5::append_itensor(
                &self.tensor_path,
                &format!("{group_path}/GammaB"),
                state.b().gamma,
            )
            .map_err(|error| {
                snapshot_field_io_error(
                    &self.path,
                    self.next_index,
                    &format!("{group_path}/GammaB"),
                    error,
                )
            })?;

            let file = File::open_rw(&self.path).map_err(|error| {
                snapshot_stage_io_error(
                    &self.path,
                    self.next_index,
                    "reopen checkpoint after tensor fields",
                    error,
                )
            })?;
            file.flush().map_err(|error| {
                snapshot_stage_io_error(&self.path, self.next_index, "flush tensor fields", error)
            })?;
            let group = file.group(&group_path).map_err(|error| {
                snapshot_field_io_error(&self.path, self.next_index, &group_path, error)
            })?;
            #[cfg(test)]
            self.fail_if_requested(WriteFailurePoint::FinalMarker)
                .map_err(|error| {
                    snapshot_stage_io_error(&self.path, self.next_index, &final_marker_stage, error)
                })?;
            group
                .attr("complete")
                .and_then(|attribute| attribute.write_scalar(&1_u8))
                .map_err(|error| {
                    snapshot_stage_io_error(&self.path, self.next_index, &final_marker_stage, error)
                })?;
            #[cfg(test)]
            self.fail_if_requested(WriteFailurePoint::FinalFlush)
                .map_err(|error| final_flush_error(&self.path, self.next_index, error))?;
            file.flush()
                .map_err(|error| final_flush_error(&self.path, self.next_index, error))
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

fn snapshot_field_io<T>(
    result: Result<T, ItebdCheckpointError>,
    path: &Path,
    index: u64,
    field: &str,
) -> Result<T, ItebdCheckpointError> {
    result.map_err(|error| add_snapshot_io_context(path, index, "field", field, error))
}

fn snapshot_stage_io<T>(
    result: Result<T, ItebdCheckpointError>,
    path: &Path,
    index: u64,
    stage: &str,
) -> Result<T, ItebdCheckpointError> {
    result.map_err(|error| add_snapshot_io_context(path, index, "stage", stage, error))
}

fn add_snapshot_io_context(
    path: &Path,
    index: u64,
    context_kind: &str,
    context: &str,
    error: ItebdCheckpointError,
) -> ItebdCheckpointError {
    match error {
        ItebdCheckpointError::Io { reason, .. } => io_error(
            path,
            format!("snapshot {index} {context_kind} {context:?}: {reason}"),
        ),
        other => other,
    }
}

fn snapshot_field_io_error(
    path: &Path,
    index: u64,
    field: &str,
    error: impl std::fmt::Display,
) -> ItebdCheckpointError {
    io_error(path, format!("snapshot {index} field {field:?}: {error}"))
}

fn snapshot_stage_io_error(
    path: &Path,
    index: u64,
    stage: &str,
    error: impl std::fmt::Display,
) -> ItebdCheckpointError {
    io_error(path, format!("snapshot {index} stage {stage:?}: {error}"))
}

fn final_flush_error(
    path: &Path,
    index: u64,
    error: impl std::fmt::Display,
) -> ItebdCheckpointError {
    io_error(
        path,
        format!(
            "snapshot {index} stage \"final flush\" failed after the complete marker update; complete marker may already be durable and snapshot persistence is uncertain: {error}"
        ),
    )
}

/// Lists complete, fully loadable checkpoints in numeric index order.
///
/// This validates and loads both tensors in every complete snapshot, so its I/O cost is
/// proportional to the full stored trajectory rather than only its metadata.
pub fn list_itebd_checkpoints(
    path: &Path,
) -> Result<Vec<ItebdCheckpointEntry>, ItebdCheckpointError> {
    let (root, names) = {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        let root = read_root(&file, path, None)?;
        let states = file
            .group("states")
            .map_err(|error| schema_error(path, "states", error.to_string()))?;
        let names = states
            .member_names()
            .map_err(|error| io_error(path, error))?;
        (root, names)
    };

    let mut complete_indices = Vec::new();
    for name in names {
        let marker = read_complete_marker_by_name(path, &name)?;
        match marker {
            None | Some(0) => continue,
            Some(1) => {}
            Some(value) => {
                let index = name.parse::<u64>().unwrap_or(0);
                return Err(snapshot_error(
                    path,
                    index,
                    "complete",
                    format!("must equal 0 or 1, got {value}"),
                ));
            }
        }
        let index = parse_snapshot_name(&name)
            .map_err(|reason| schema_error(path, format!("states/{name}"), reason))?;
        complete_indices.push(index);
    }
    complete_indices.sort_unstable();

    let mut complete = Vec::with_capacity(complete_indices.len());
    let mut previous = None;
    for index in complete_indices {
        let record = read_snapshot(path, index, &root, 1e-12)?;
        if let Some((previous_steps, previous_beta)) = previous {
            if record.progress.completed_steps <= previous_steps {
                return Err(snapshot_error(
                    path,
                    index,
                    "completed_steps",
                    format!("must exceed previous complete value {previous_steps}"),
                ));
            }
            if record.progress.beta <= previous_beta {
                return Err(snapshot_error(
                    path,
                    index,
                    "beta",
                    format!("must exceed previous complete value {previous_beta}"),
                ));
            }
        }
        previous = Some((record.progress.completed_steps, record.progress.beta));
        complete.push(ItebdCheckpointEntry {
            index,
            beta: record.progress.beta,
            completed_steps: record.progress.completed_steps,
        });
    }
    Ok(complete)
}

pub fn load_itebd_checkpoint(
    path: &Path,
    index: u64,
    options: &ItebdCheckpointLoadOptions,
) -> Result<LoadedItebdCheckpoint, ItebdCheckpointError> {
    validate_load_options(path, options)?;
    let root = {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        read_root(&file, path, Some(options.hermiticity_tolerance))?
    };
    let record = read_snapshot(path, index, &root, options.diagnostic_relative_tolerance)?;
    Ok(LoadedItebdCheckpoint {
        state: record.state,
        hamiltonian: root.hamiltonian,
        metadata: root.metadata,
        progress: record.progress,
        diagnostics: record.diagnostics,
    })
}

struct SnapshotRecord {
    state: ItebdState,
    progress: ItebdCheckpointProgress,
    diagnostics: ItebdSnapshotDiagnostics,
}

fn read_snapshot(
    path: &Path,
    index: u64,
    root: &ValidatedRoot,
    diagnostic_tolerance: f64,
) -> Result<SnapshotRecord, ItebdCheckpointError> {
    let group_name = encode_snapshot_name(index);
    let group_path = format!("states/{group_name}");
    let (
        beta,
        completed_steps,
        accumulated_log_norm,
        diagnostics,
        a_axes,
        b_axes,
        lambda_ab,
        lambda_ba,
    ) = {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        let group = file.group(&group_path).map_err(|error| {
            snapshot_error(
                path,
                index,
                "group",
                format!("required group is missing: {error}"),
            )
        })?;
        match read_complete_marker(&group, path, index)? {
            Some(1) => {}
            None | Some(0) => {
                return Err(ItebdCheckpointError::Incomplete {
                    path: path.to_path_buf(),
                    index,
                })
            }
            Some(value) => {
                return Err(snapshot_error(
                    path,
                    index,
                    "complete",
                    format!("must equal 0 or 1, got {value}"),
                ))
            }
        }
        let beta = snapshotize(
            read_scalar_dataset::<f64>(&group, path, "beta", "f64"),
            path,
            index,
        )?;
        let completed_steps = snapshotize(
            read_scalar_dataset::<u64>(&group, path, "completed_steps", "u64"),
            path,
            index,
        )?;
        let accumulated_log_norm = snapshotize(
            read_scalar_dataset::<f64>(&group, path, "accumulated_log_norm", "f64"),
            path,
            index,
        )?;
        let diagnostics_json = snapshotize(
            read_string_dataset(&group, path, "diagnostics_json"),
            path,
            index,
        )?;
        let diagnostics = serde_json::from_str::<ItebdSnapshotDiagnostics>(&diagnostics_json)
            .map_err(|error| {
                snapshot_error(
                    path,
                    index,
                    "diagnostics_json",
                    format!("invalid diagnostics record: {error}"),
                )
            })?;
        let a_axes = snapshotize(read_u8_vector(&group, path, "a_role_axes", 4), path, index)?;
        let b_axes = snapshotize(read_u8_vector(&group, path, "b_role_axes", 4), path, index)?;
        let lambda_ab = snapshotize(
            read_f64_vector(&group, path, "lambdaAB", diagnostics.bond_dimensions[0]),
            path,
            index,
        )?;
        let lambda_ba = snapshotize(
            read_f64_vector(&group, path, "lambdaBA", diagnostics.bond_dimensions[1]),
            path,
            index,
        )?;
        (
            beta,
            completed_steps,
            accumulated_log_norm,
            diagnostics,
            axes_array(path, index, "a_role_axes", &a_axes)?,
            axes_array(path, index, "b_role_axes", &b_axes)?,
            lambda_ab,
            lambda_ba,
        )
    };

    preflight_gamma_schemas(path, index, root.scalar_type, &group_path)?;
    let tensor_path_value = tensor_path(path)?;
    let gamma_a = tensor4all_hdf5::load_itensor(tensor_path_value, &format!("{group_path}/GammaA"))
        .map_err(|error| snapshot_error(path, index, "GammaA", error.to_string()))?;
    let gamma_b = tensor4all_hdf5::load_itensor(tensor_path_value, &format!("{group_path}/GammaB"))
        .map_err(|error| snapshot_error(path, index, "GammaB", error.to_string()))?;
    let state = build_state(
        path,
        index,
        root.scalar_type,
        gamma_a,
        gamma_b,
        a_axes,
        b_axes,
        lambda_ab,
        lambda_ba,
    )?;
    let summary =
        ItebdStateRef::from(&state)
            .validate()
            .map_err(|source| ItebdCheckpointError::State {
                path: path.to_path_buf(),
                index,
                source,
            })?;
    validate_loaded_summary(
        path,
        index,
        root,
        &summary,
        &diagnostics,
        diagnostic_tolerance,
    )?;
    let progress = ItebdCheckpointProgress {
        beta,
        completed_steps,
        accumulated_log_norm,
        last_step: diagnostics.last_step.as_ref().map(step_from_diagnostics),
    };
    validate_progress(path, index, &root.metadata, &progress, &summary, None)?;

    Ok(SnapshotRecord {
        state,
        progress,
        diagnostics,
    })
}

fn preflight_gamma_schemas(
    path: &Path,
    index: u64,
    scalar_type: ItebdScalarType,
    snapshot_path: &str,
) -> Result<(), ItebdCheckpointError> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    for gamma_name in ["GammaA", "GammaB"] {
        let gamma_path = format!("{snapshot_path}/{gamma_name}");
        let gamma = file.group(&gamma_path).map_err(|error| {
            snapshot_error(
                path,
                index,
                gamma_name,
                format!("required tensor group is missing: {error}"),
            )
        })?;
        preflight_gamma_group(path, index, gamma_name, scalar_type, &gamma)?;
    }
    Ok(())
}

fn preflight_gamma_group(
    path: &Path,
    index: u64,
    field: &str,
    scalar_type: ItebdScalarType,
    gamma: &Group,
) -> Result<(), ItebdCheckpointError> {
    require_object_identity(path, index, field, gamma, "ITensor")?;
    let inds_field = format!("{field}/inds");
    let inds = gamma.group("inds").map_err(|error| {
        snapshot_error(
            path,
            index,
            &inds_field,
            format!("required group is missing: {error}"),
        )
    })?;
    require_object_identity(path, index, &inds_field, &inds, "IndexSet")?;
    let length = read_exact_scalar_dataset_at::<i64>(
        &inds,
        path,
        index,
        &format!("{inds_field}/length"),
        "length",
        "i64",
    )?;
    if length != 4 {
        return Err(snapshot_error(
            path,
            index,
            format!("{inds_field}/length"),
            format!("rank-4 Gamma requires length 4, got {length}"),
        ));
    }

    let mut element_count = 1_usize;
    for ordinal in 1..=4 {
        let child_name = format!("index_{ordinal}");
        let index_field = format!("{inds_field}/{child_name}");
        let index_group = inds.group(&child_name).map_err(|error| {
            snapshot_error(
                path,
                index,
                &index_field,
                format!("required group is missing: {error}"),
            )
        })?;
        require_object_identity(path, index, &index_field, &index_group, "Index")?;
        let space_type = read_exact_string_attribute_at(
            &index_group,
            path,
            index,
            &format!("{index_field}/space_type"),
            "space_type",
        )?;
        if space_type != "Int" {
            return Err(snapshot_error(
                path,
                index,
                format!("{index_field}/space_type"),
                format!("expected \"Int\", got {space_type:?}"),
            ));
        }
        read_exact_scalar_dataset_at::<u64>(
            &index_group,
            path,
            index,
            &format!("{index_field}/id"),
            "id",
            "u64",
        )?;
        let dimension = read_exact_scalar_dataset_at::<i64>(
            &index_group,
            path,
            index,
            &format!("{index_field}/dim"),
            "dim",
            "i64",
        )?;
        let dimension = usize::try_from(dimension).map_err(|_| {
            snapshot_error(
                path,
                index,
                format!("{index_field}/dim"),
                format!("must be positive and fit in usize, got {dimension}"),
            )
        })?;
        if dimension == 0 {
            return Err(snapshot_error(
                path,
                index,
                format!("{index_field}/dim"),
                "must be positive",
            ));
        }
        element_count = element_count.checked_mul(dimension).ok_or_else(|| {
            snapshot_error(
                path,
                index,
                format!("{field}/storage/data"),
                "element count overflows usize",
            )
        })?;
        let direction = read_exact_scalar_dataset_at::<i64>(
            &index_group,
            path,
            index,
            &format!("{index_field}/dir"),
            "dir",
            "i64",
        )?;
        if direction != 0 {
            return Err(snapshot_error(
                path,
                index,
                format!("{index_field}/dir"),
                format!("must equal 0, got {direction}"),
            ));
        }
        read_exact_scalar_dataset_at::<i64>(
            &index_group,
            path,
            index,
            &format!("{index_field}/plev"),
            "plev",
            "i64",
        )?;
        let tags_field = format!("{index_field}/tags");
        let tags = index_group.group("tags").map_err(|error| {
            snapshot_error(
                path,
                index,
                &tags_field,
                format!("required group is missing: {error}"),
            )
        })?;
        require_object_identity(path, index, &tags_field, &tags, "TagSet")?;
        read_exact_string_dataset_at(&tags, path, index, &format!("{tags_field}/tags"), "tags")?;
    }

    let storage_field = format!("{field}/storage");
    let storage = gamma.group("storage").map_err(|error| {
        snapshot_error(
            path,
            index,
            &storage_field,
            format!("required group is missing: {error}"),
        )
    })?;
    let expected_storage = match scalar_type {
        ItebdScalarType::Real => "Dense{Float64}",
        ItebdScalarType::Complex => "Dense{ComplexF64}",
    };
    require_object_identity(path, index, &storage_field, &storage, expected_storage)?;
    let data_field = format!("{storage_field}/data");
    match scalar_type {
        ItebdScalarType::Real => require_exact_vector_dataset::<f64>(
            &storage,
            path,
            index,
            &data_field,
            "data",
            element_count,
            "f64",
        ),
        ItebdScalarType::Complex => require_exact_vector_dataset::<Complex64>(
            &storage,
            path,
            index,
            &data_field,
            "data",
            element_count,
            "Complex64",
        ),
    }
}

fn require_object_identity(
    path: &Path,
    index: u64,
    field: &str,
    group: &Group,
    expected_type: &str,
) -> Result<(), ItebdCheckpointError> {
    let type_field = format!("{field}/type");
    let type_name = read_exact_string_attribute_at(group, path, index, &type_field, "type")?;
    if type_name != expected_type {
        return Err(snapshot_error(
            path,
            index,
            type_field,
            format!("expected {expected_type:?}, got {type_name:?}"),
        ));
    }
    let version_field = format!("{field}/version");
    let version = read_exact_scalar_attribute_at::<i64>(
        group,
        path,
        index,
        &version_field,
        "version",
        "i64",
    )?;
    if version != 1 {
        return Err(snapshot_error(
            path,
            index,
            version_field,
            format!("expected version 1, got {version}"),
        ));
    }
    Ok(())
}

fn read_exact_string_attribute_at(
    group: &Group,
    path: &Path,
    index: u64,
    field: &str,
    name: &str,
) -> Result<String, ItebdCheckpointError> {
    let attribute = group.attr(name).map_err(|error| {
        snapshot_error(
            path,
            index,
            field,
            format!("required attribute is missing: {error}"),
        )
    })?;
    if !attribute.shape().is_empty() {
        return Err(snapshot_error(path, index, field, "must be scalar"));
    }
    let dtype = attribute.dtype().map_err(|error| {
        snapshot_error(path, index, field, format!("cannot inspect dtype: {error}"))
    })?;
    if !dtype.is::<VarLenUnicode>() {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!("must have VarLenUnicode dtype, got {dtype}"),
        ));
    }
    attribute
        .read_scalar::<VarLenUnicode>()
        .map(|value| value.as_str().to_string())
        .map_err(|error| snapshot_error(path, index, field, error.to_string()))
}

fn read_exact_scalar_attribute_at<T: H5Type>(
    group: &Group,
    path: &Path,
    index: u64,
    field: &str,
    name: &str,
    expected_type: &str,
) -> Result<T, ItebdCheckpointError> {
    let attribute = group.attr(name).map_err(|error| {
        snapshot_error(
            path,
            index,
            field,
            format!("required attribute is missing: {error}"),
        )
    })?;
    if !attribute.shape().is_empty() {
        return Err(snapshot_error(path, index, field, "must be scalar"));
    }
    let dtype = attribute.dtype().map_err(|error| {
        snapshot_error(path, index, field, format!("cannot inspect dtype: {error}"))
    })?;
    if !dtype.is::<T>() {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!("must have {expected_type} dtype, got {dtype}"),
        ));
    }
    attribute
        .read_scalar::<T>()
        .map_err(|error| snapshot_error(path, index, field, error.to_string()))
}

fn read_exact_scalar_dataset_at<T: H5Type>(
    group: &Group,
    path: &Path,
    index: u64,
    field: &str,
    name: &str,
    expected_type: &str,
) -> Result<T, ItebdCheckpointError> {
    let dataset = group.dataset(name).map_err(|error| {
        snapshot_error(
            path,
            index,
            field,
            format!("required dataset is missing: {error}"),
        )
    })?;
    if !dataset.shape().is_empty() {
        return Err(snapshot_error(path, index, field, "must be scalar"));
    }
    let dtype = dataset.dtype().map_err(|error| {
        snapshot_error(path, index, field, format!("cannot inspect dtype: {error}"))
    })?;
    if !dtype.is::<T>() {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!("must have {expected_type} dtype, got {dtype}"),
        ));
    }
    dataset
        .read_scalar::<T>()
        .map_err(|error| snapshot_error(path, index, field, error.to_string()))
}

fn read_exact_string_dataset_at(
    group: &Group,
    path: &Path,
    index: u64,
    field: &str,
    name: &str,
) -> Result<String, ItebdCheckpointError> {
    let dataset = group.dataset(name).map_err(|error| {
        snapshot_error(
            path,
            index,
            field,
            format!("required dataset is missing: {error}"),
        )
    })?;
    if !dataset.shape().is_empty() {
        return Err(snapshot_error(path, index, field, "must be scalar"));
    }
    let dtype = dataset.dtype().map_err(|error| {
        snapshot_error(path, index, field, format!("cannot inspect dtype: {error}"))
    })?;
    if !dtype.is::<VarLenUnicode>() {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!("must have VarLenUnicode dtype, got {dtype}"),
        ));
    }
    dataset
        .read_scalar::<VarLenUnicode>()
        .map(|value| value.as_str().to_string())
        .map_err(|error| snapshot_error(path, index, field, error.to_string()))
}

fn require_exact_vector_dataset<T: H5Type>(
    group: &Group,
    path: &Path,
    index: u64,
    field: &str,
    name: &str,
    expected_length: usize,
    expected_type: &str,
) -> Result<(), ItebdCheckpointError> {
    let dataset = group.dataset(name).map_err(|error| {
        snapshot_error(
            path,
            index,
            field,
            format!("required dataset is missing: {error}"),
        )
    })?;
    if dataset.shape() != [expected_length] {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!(
                "must have shape [{expected_length}], got {:?}",
                dataset.shape()
            ),
        ));
    }
    let dtype = dataset.dtype().map_err(|error| {
        snapshot_error(path, index, field, format!("cannot inspect dtype: {error}"))
    })?;
    if !dtype.is::<T>() {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!("must have {expected_type} dtype, got {dtype}"),
        ));
    }
    Ok(())
}

fn build_state(
    path: &Path,
    index: u64,
    scalar_type: ItebdScalarType,
    gamma_a: Tensor,
    gamma_b: Tensor,
    a_axes: [usize; 4],
    b_axes: [usize; 4],
    lambda_ab: Vec<f64>,
    lambda_ba: Vec<f64>,
) -> Result<ItebdState, ItebdCheckpointError> {
    let a_roles = roles_from_axes(path, index, "a_role_axes", &gamma_a, a_axes)?;
    let b_roles = roles_from_axes(path, index, "b_role_axes", &gamma_b, b_axes)?;
    let lambda_bond_ab = a_roles[3].clone();
    let lambda_bond_ba = a_roles[0].clone();
    Ok(match scalar_type {
        ItebdScalarType::Real => ItebdState::Real(PurifiedMps {
            a: Site {
                gamma: gamma_a,
                left: a_roles[0].clone(),
                phys: a_roles[1].clone(),
                anc: a_roles[2].clone(),
                right: a_roles[3].clone(),
            },
            b: Site {
                gamma: gamma_b,
                left: b_roles[0].clone(),
                phys: b_roles[1].clone(),
                anc: b_roles[2].clone(),
                right: b_roles[3].clone(),
            },
            lambda_ab,
            lambda_bond_ab,
            lambda_ba,
            lambda_bond_ba,
        }),
        ItebdScalarType::Complex => ItebdState::Complex(ComplexPurifiedMps {
            a: ComplexSite {
                gamma: gamma_a,
                left: a_roles[0].clone(),
                phys: a_roles[1].clone(),
                anc: a_roles[2].clone(),
                right: a_roles[3].clone(),
            },
            b: ComplexSite {
                gamma: gamma_b,
                left: b_roles[0].clone(),
                phys: b_roles[1].clone(),
                anc: b_roles[2].clone(),
                right: b_roles[3].clone(),
            },
            lambda_ab,
            lambda_bond_ab,
            lambda_ba,
            lambda_bond_ba,
        }),
    })
}

fn roles_from_axes(
    path: &Path,
    index: u64,
    field: &str,
    tensor: &Tensor,
    axes: [usize; 4],
) -> Result<[Idx; 4], ItebdCheckpointError> {
    if tensor.indices.len() != 4 {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!(
                "cannot resolve four roles from rank-{} tensor",
                tensor.indices.len()
            ),
        ));
    }
    Ok(axes.map(|axis| tensor.indices[axis].clone()))
}

fn axes_array(
    path: &Path,
    index: u64,
    field: &str,
    values: &[u8],
) -> Result<[usize; 4], ItebdCheckpointError> {
    let axes = [
        values[0] as usize,
        values[1] as usize,
        values[2] as usize,
        values[3] as usize,
    ];
    let mut sorted = axes;
    sorted.sort_unstable();
    if sorted != [0, 1, 2, 3] {
        return Err(snapshot_error(
            path,
            index,
            field,
            format!("must be a permutation of [0, 1, 2, 3], got {axes:?}"),
        ));
    }
    Ok(axes)
}

fn validate_snapshot_against_writer(
    writer: &ItebdTrajectoryWriter,
    index: u64,
    summary: &ItebdStateSummary,
) -> Result<(), ItebdCheckpointError> {
    if summary.scalar_type != writer.scalar_type {
        return Err(snapshot_error(
            &writer.path,
            index,
            "scalar_type",
            format!(
                "state backend {:?} does not match file backend {:?}",
                summary.scalar_type, writer.scalar_type
            ),
        ));
    }
    if summary.physical_dim != writer.physical_dim {
        return Err(snapshot_error(
            &writer.path,
            index,
            "physical_dim",
            format!(
                "state dimension {} does not match file dimension {}",
                summary.physical_dim, writer.physical_dim
            ),
        ));
    }
    if summary.ancilla_dim != writer.ancilla_dim {
        return Err(snapshot_error(
            &writer.path,
            index,
            "ancilla_dim",
            format!(
                "state dimension {} does not match file dimension {}",
                summary.ancilla_dim, writer.ancilla_dim
            ),
        ));
    }
    if let Some(cap) = writer.metadata.truncation.max_bond {
        if summary
            .bond_dimensions
            .into_iter()
            .any(|dimension| dimension > cap)
        {
            return Err(snapshot_error(
                &writer.path,
                index,
                "bond_dimensions",
                format!(
                    "state bond dimensions {:?} exceed configured cap {cap}",
                    summary.bond_dimensions
                ),
            ));
        }
    }
    Ok(())
}

fn validate_loaded_summary(
    path: &Path,
    index: u64,
    root: &ValidatedRoot,
    summary: &ItebdStateSummary,
    diagnostics: &ItebdSnapshotDiagnostics,
    tolerance: f64,
) -> Result<(), ItebdCheckpointError> {
    if summary.scalar_type != root.scalar_type {
        return Err(snapshot_error(
            path,
            index,
            "scalar_type",
            "loaded tensor storage does not match root backend",
        ));
    }
    if summary.physical_dim != root.physical_dim {
        return Err(snapshot_error(
            path,
            index,
            "physical_dim",
            format!(
                "loaded state dimension {} does not match root {}",
                summary.physical_dim, root.physical_dim
            ),
        ));
    }
    if summary.ancilla_dim != root.ancilla_dim {
        return Err(snapshot_error(
            path,
            index,
            "ancilla_dim",
            format!(
                "loaded state dimension {} does not match root {}",
                summary.ancilla_dim, root.ancilla_dim
            ),
        ));
    }
    if diagnostics.bond_dimensions != summary.bond_dimensions {
        return Err(snapshot_error(
            path,
            index,
            "diagnostics_json.bond_dimensions",
            format!(
                "stored {:?} does not match actual {:?}",
                diagnostics.bond_dimensions, summary.bond_dimensions
            ),
        ));
    }
    for axis in 0..2 {
        let stored = diagnostics.schmidt_norms[axis];
        let actual = summary.schmidt_norms[axis];
        let scale = 1.0_f64.max(actual.abs()).max(stored.abs());
        if !stored.is_finite() || stored <= 0.0 || (stored - actual).abs() > tolerance * scale {
            return Err(snapshot_error(
                path,
                index,
                format!("diagnostics_json.schmidt_norms[{axis}]"),
                format!("stored {stored} does not match actual {actual} within {tolerance}"),
            ));
        }
    }
    if let Some(cap) = root.metadata.truncation.max_bond {
        if summary
            .bond_dimensions
            .into_iter()
            .any(|dimension| dimension > cap)
        {
            return Err(snapshot_error(
                path,
                index,
                "bond_dimensions",
                format!(
                    "actual dimensions {:?} exceed cap {cap}",
                    summary.bond_dimensions
                ),
            ));
        }
    }
    Ok(())
}

fn validate_progress(
    path: &Path,
    index: u64,
    metadata: &ItebdRunMetadata,
    progress: &ItebdCheckpointProgress,
    summary: &ItebdStateSummary,
    previous: Option<(u64, f64)>,
) -> Result<(), ItebdCheckpointError> {
    let expected_beta = 2.0 * progress.completed_steps as f64 * metadata.dtau;
    let scale = 1.0_f64.max(progress.beta.abs()).max(expected_beta.abs());
    let valid_beta = progress.beta.is_finite()
        && progress.beta >= 0.0
        && expected_beta.is_finite()
        && (progress.beta - expected_beta).abs() <= 1e-12 * scale;
    if !valid_beta {
        return Err(snapshot_error(
            path,
            index,
            "beta",
            format!(
                "must match 2 * completed_steps * dtau = {expected_beta} within relative tolerance 1e-12"
            ),
        ));
    }
    if !progress.accumulated_log_norm.is_finite() {
        return Err(snapshot_error(
            path,
            index,
            "accumulated_log_norm",
            "must be finite",
        ));
    }
    if progress.completed_steps == 0
        && (progress.beta != 0.0
            || progress.accumulated_log_norm != 0.0
            || progress.last_step.is_some())
    {
        return Err(snapshot_error(
            path,
            index,
            "progress",
            "step zero requires beta=0, accumulated_log_norm=0, and no last_step",
        ));
    }
    if let Some(step) = progress.last_step {
        let current_max = summary.bond_dimensions[0].max(summary.bond_dimensions[1]);
        if step.max_bond == 0 || step.max_bond < current_max {
            return Err(snapshot_error(
                path,
                index,
                "diagnostics_json.last_step.max_bond",
                format!("must be at least current maximum bond dimension {current_max}"),
            ));
        }
        if !step.min_singular_value.is_finite() || step.min_singular_value < 0.0 {
            return Err(snapshot_error(
                path,
                index,
                "diagnostics_json.last_step.min_singular_value",
                "must be finite and non-negative",
            ));
        }
        if !step.log_norm.is_finite() {
            return Err(snapshot_error(
                path,
                index,
                "diagnostics_json.last_step.log_norm",
                "must be finite",
            ));
        }
    }
    if let Some((previous_steps, previous_beta)) = previous {
        if progress.completed_steps <= previous_steps {
            return Err(snapshot_error(
                path,
                index,
                "completed_steps",
                format!("must exceed previous value {previous_steps}"),
            ));
        }
        if progress.beta <= previous_beta {
            return Err(snapshot_error(
                path,
                index,
                "beta",
                format!("must exceed previous value {previous_beta}"),
            ));
        }
    }
    Ok(())
}

fn diagnostics_from(
    progress: &ItebdCheckpointProgress,
    summary: &ItebdStateSummary,
) -> ItebdSnapshotDiagnostics {
    ItebdSnapshotDiagnostics {
        bond_dimensions: summary.bond_dimensions,
        schmidt_norms: summary.schmidt_norms,
        last_step: progress.last_step.map(|step| ItebdStepDiagnostics {
            max_bond: step.max_bond,
            min_singular_value: step.min_singular_value,
            log_norm: step.log_norm,
        }),
    }
}

fn step_from_diagnostics(value: &ItebdStepDiagnostics) -> StepInfo {
    StepInfo {
        max_bond: value.max_bond,
        min_singular_value: value.min_singular_value,
        log_norm: value.log_norm,
    }
}

fn validate_load_options(
    path: &Path,
    options: &ItebdCheckpointLoadOptions,
) -> Result<(), ItebdCheckpointError> {
    for (field, value) in [
        ("hermiticity_tolerance", options.hermiticity_tolerance),
        (
            "diagnostic_relative_tolerance",
            options.diagnostic_relative_tolerance,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(schema_error(
                path,
                format!("load_options.{field}"),
                "must be finite and non-negative",
            ));
        }
    }
    Ok(())
}

fn read_complete_marker_by_name(
    path: &Path,
    name: &str,
) -> Result<Option<u8>, ItebdCheckpointError> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let group = file
        .group(&format!("states/{name}"))
        .map_err(|error| schema_error(path, format!("states/{name}"), error.to_string()))?;
    read_complete_marker(&group, path, name.parse::<u64>().unwrap_or(0))
}

fn read_complete_marker(
    group: &Group,
    path: &Path,
    index: u64,
) -> Result<Option<u8>, ItebdCheckpointError> {
    let names = group.attr_names().map_err(|error| io_error(path, error))?;
    if !names.iter().any(|name| name == "complete") {
        return Ok(None);
    }
    snapshotize(
        read_scalar_attribute::<u8>(group, path, "complete", "u8"),
        path,
        index,
    )
    .map(Some)
}

fn encode_snapshot_name(index: u64) -> String {
    format!("{index:06}")
}

fn parse_snapshot_name(name: &str) -> Result<u64, String> {
    if name.is_empty() || !name.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "complete snapshot group name {name:?} must contain only ASCII digits"
        ));
    }
    let index = name
        .parse::<u64>()
        .map_err(|error| format!("snapshot index is not representable as u64: {error}"))?;
    if encode_snapshot_name(index) != name {
        return Err(format!(
            "complete snapshot group name {name:?} is not the canonical encoding of index {index}"
        ));
    }
    Ok(index)
}

fn tensor_path(path: &Path) -> Result<&str, ItebdCheckpointError> {
    path.to_str()
        .ok_or_else(|| schema_error(path, "path", "must be valid UTF-8 for tensor4all-hdf5"))
}

fn snapshotize<T>(
    result: Result<T, ItebdCheckpointError>,
    path: &Path,
    index: u64,
) -> Result<T, ItebdCheckpointError> {
    result.map_err(|error| match error {
        ItebdCheckpointError::Schema { field, reason, .. } => {
            snapshot_error(path, index, field, reason)
        }
        other => other,
    })
}

fn snapshot_error(
    path: &Path,
    index: u64,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> ItebdCheckpointError {
    ItebdCheckpointError::Snapshot {
        path: path.to_path_buf(),
        index,
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        encode_snapshot_name, parse_snapshot_name, ItebdTrajectoryWriter, WriteFailurePoint,
    };
    use crate::config::{TrotterOrder, TruncationCfg};
    use crate::itebd_auto::ItebdHamiltonian;
    use crate::itebd_checkpoint::{
        load_itebd_checkpoint, ItebdCheckpointError, ItebdCheckpointLoadOptions,
        ItebdCheckpointProgress, ItebdRunMetadata,
    };
    use crate::itebd_state_view::ItebdStateRef;
    use crate::model::Tfim;
    use crate::purified_mps::infinite_temperature;

    #[test]
    fn snapshot_names_are_canonical_and_extend_beyond_six_digits() {
        for (index, encoded) in [
            (0, "000000"),
            (999_999, "999999"),
            (1_000_000, "1000000"),
            (u64::MAX, "18446744073709551615"),
        ] {
            assert_eq!(encode_snapshot_name(index), encoded);
            assert_eq!(parse_snapshot_name(encoded).unwrap(), index);
        }
        for malformed in ["", "0", "0000000", "00000x", "１２３４５６"] {
            assert!(parse_snapshot_name(malformed).is_err(), "{malformed:?}");
        }
    }

    #[test]
    fn failures_after_snapshot_group_creation_poison_append_and_finish_but_preserve_prior_snapshot()
    {
        for (point, expected_context) in [
            (
                WriteFailurePoint::MarkerCreation,
                "stage \"create initial completion marker at states/000001/@complete\"",
            ),
            (WriteFailurePoint::GammaA, "field \"states/000001/GammaA\""),
            (WriteFailurePoint::GammaB, "field \"states/000001/GammaB\""),
            (
                WriteFailurePoint::FinalMarker,
                "stage \"update final completion marker at states/000001/@complete\"",
            ),
            (WriteFailurePoint::FinalFlush, "stage \"final flush\""),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(format!("fault-{point:?}.h5"));
            let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
            let state = infinite_temperature(2);
            let mut writer = ItebdTrajectoryWriter::create(&path, metadata(), &h).unwrap();
            writer
                .append(ItebdStateRef::from(&state), &progress(0))
                .unwrap();
            writer.inject_failure(point);

            let error = writer
                .append(ItebdStateRef::from(&state), &progress(1))
                .unwrap_err();
            assert!(matches!(
                error,
                ItebdCheckpointError::Io { path: error_path, reason }
                    if error_path == path
                        && reason.contains("snapshot 1")
                        && reason.contains(expected_context)
                        && reason.contains("injected failure")
                        && (point != WriteFailurePoint::FinalFlush
                            || (reason.contains("complete marker may already be durable")
                                && reason.contains("snapshot persistence is uncertain")))
            ));
            assert!(matches!(
                writer.append(ItebdStateRef::from(&state), &progress(1)),
                Err(ItebdCheckpointError::WriterPoisoned { path: error_path })
                    if error_path == path
            ));
            assert!(matches!(
                writer.finish(),
                Err(ItebdCheckpointError::WriterPoisoned { path: error_path })
                    if error_path == path
            ));

            let loaded =
                load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
            assert_eq!(loaded.progress.completed_steps, 0, "{point:?}");
            if point == WriteFailurePoint::FinalFlush {
                let file = hdf5_metno::File::open(&path).unwrap();
                assert_eq!(
                    file.group("states/000001")
                        .unwrap()
                        .attr("complete")
                        .unwrap()
                        .read_scalar::<u8>()
                        .unwrap(),
                    1,
                    "final-flush uncertainty must retain the already-updated marker"
                );
            }
        }
    }

    #[test]
    fn failed_snapshot_group_creation_attempt_poisons_writer_and_preserves_prior_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("group-creation-failure.h5");
        let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
        let state = infinite_temperature(2);
        let mut writer = ItebdTrajectoryWriter::create(&path, metadata(), &h).unwrap();
        writer
            .append(ItebdStateRef::from(&state), &progress(0))
            .unwrap();
        let file = hdf5_metno::File::open_rw(&path).unwrap();
        file.group("states")
            .unwrap()
            .create_group("000001")
            .unwrap();
        file.flush().unwrap();
        drop(file);

        let error = writer
            .append(ItebdStateRef::from(&state), &progress(1))
            .unwrap_err();
        assert!(matches!(
            error,
            ItebdCheckpointError::Io { path: error_path, reason }
                if error_path == path
                    && reason.contains("snapshot 1")
                    && reason.contains("field \"states/000001\"")
                    && reason.contains("already exists")
        ));
        assert!(matches!(
            writer.append(ItebdStateRef::from(&state), &progress(1)),
            Err(ItebdCheckpointError::WriterPoisoned { path: error_path })
                if error_path == path
        ));
        assert!(matches!(
            writer.finish(),
            Err(ItebdCheckpointError::WriterPoisoned { path: error_path })
                if error_path == path
        ));
        let loaded =
            load_itebd_checkpoint(&path, 0, &ItebdCheckpointLoadOptions::default()).unwrap();
        assert_eq!(loaded.progress.completed_steps, 0);
    }

    #[test]
    fn rejected_prevalidation_does_not_create_a_group_or_poison_the_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prevalidation.h5");
        let h = ItebdHamiltonian::Real(Tfim { j: 1.0, g: 0.7 }.local());
        let state = infinite_temperature(2);
        let mut writer = ItebdTrajectoryWriter::create(&path, metadata(), &h).unwrap();
        let mut invalid = progress(0);
        invalid.beta = 0.25;
        assert!(matches!(
            writer.append(ItebdStateRef::from(&state), &invalid),
            Err(ItebdCheckpointError::Snapshot { path: error_path, index: 0, field, .. })
                if error_path == path && field == "beta"
        ));
        let file = hdf5_metno::File::open(&path).unwrap();
        assert!(file
            .group("states")
            .unwrap()
            .member_names()
            .unwrap()
            .is_empty());
        drop(file);
        assert_eq!(
            writer
                .append(ItebdStateRef::from(&state), &progress(0))
                .unwrap()
                .index,
            0
        );
        writer.finish().unwrap();
    }

    fn metadata() -> ItebdRunMetadata {
        ItebdRunMetadata {
            dtau: 0.05,
            trotter_order: TrotterOrder::Second,
            truncation: TruncationCfg {
                epsilon: 1e-13,
                max_bond: Some(64),
            },
            canonicalize_every: 3,
            record_every_beta: None,
            model_label: None,
            git_revision: None,
            hermiticity_tolerance: 1e-12,
        }
    }

    fn progress(completed_steps: u64) -> ItebdCheckpointProgress {
        ItebdCheckpointProgress {
            beta: 2.0 * completed_steps as f64 * 0.05,
            completed_steps,
            accumulated_log_norm: if completed_steps == 0 { 0.0 } else { -0.1 },
            last_step: None,
        }
    }
}
