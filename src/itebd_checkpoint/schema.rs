use super::{ItebdCheckpointError, ItebdRunMetadata};
use crate::config::{TrotterOrder, TruncationCfg};
use crate::itebd_auto::ItebdHamiltonian;
use crate::itebd_complex::ComplexLocalHamiltonian;
use crate::itebd_state_view::ItebdScalarType;
use crate::model::LocalHamiltonian;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, H5Type, Location};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::str::FromStr;

pub(super) const FILE_TYPE: &str = "TwoSiteItebdPurificationRun";
pub(super) const SCHEMA_VERSION: u32 = 1;
pub(super) const UNIT_CELL_SIZE: u32 = 2;
pub(super) const BASIS_ORDER: &str = "first-site-fastest";

pub(super) struct ValidatedRoot {
    pub metadata: ItebdRunMetadata,
    pub scalar_type: ItebdScalarType,
    pub physical_dim: usize,
    pub ancilla_dim: usize,
    pub hamiltonian: ItebdHamiltonian,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataRecord {
    dtau: f64,
    trotter_order: TrotterOrder,
    truncation: TruncationRecord,
    canonicalize_every: usize,
    record_every_beta: Option<f64>,
    model_label: Option<String>,
    git_revision: Option<String>,
    hermiticity_tolerance: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TruncationRecord {
    epsilon: f64,
    max_bond: Option<usize>,
}

impl From<&ItebdRunMetadata> for MetadataRecord {
    fn from(value: &ItebdRunMetadata) -> Self {
        Self {
            dtau: value.dtau,
            trotter_order: value.trotter_order,
            truncation: TruncationRecord {
                epsilon: value.truncation.epsilon,
                max_bond: value.truncation.max_bond,
            },
            canonicalize_every: value.canonicalize_every,
            record_every_beta: value.record_every_beta,
            model_label: value.model_label.clone(),
            git_revision: value.git_revision.clone(),
            hermiticity_tolerance: value.hermiticity_tolerance,
        }
    }
}

impl From<MetadataRecord> for ItebdRunMetadata {
    fn from(value: MetadataRecord) -> Self {
        Self {
            dtau: value.dtau,
            trotter_order: value.trotter_order,
            truncation: TruncationCfg {
                epsilon: value.truncation.epsilon,
                max_bond: value.truncation.max_bond,
            },
            canonicalize_every: value.canonicalize_every,
            record_every_beta: value.record_every_beta,
            model_label: value.model_label,
            git_revision: value.git_revision,
            hermiticity_tolerance: value.hermiticity_tolerance,
        }
    }
}

pub(super) fn validate_create_inputs(
    path: &Path,
    metadata: &ItebdRunMetadata,
    hamiltonian: &ItebdHamiltonian,
) -> Result<(ItebdScalarType, usize), ItebdCheckpointError> {
    validate_metadata(path, metadata)?;
    let (scalar_type, physical_dim) = match hamiltonian {
        ItebdHamiltonian::Real(hamiltonian) => {
            let dimension =
                validate_real_hamiltonian(path, hamiltonian, metadata.hermiticity_tolerance)?;
            (ItebdScalarType::Real, dimension)
        }
        ItebdHamiltonian::Complex(hamiltonian) => {
            if hamiltonian.hermiticity_tolerance().to_bits()
                != metadata.hermiticity_tolerance.to_bits()
            {
                return Err(schema_error(
                    path,
                    "run/metadata_json.hermiticity_tolerance",
                    format!(
                        "metadata tolerance {} must exactly match complex Hamiltonian tolerance {}",
                        metadata.hermiticity_tolerance,
                        hamiltonian.hermiticity_tolerance()
                    ),
                ));
            }
            validate_complex_hamiltonian_parts(
                path,
                hamiltonian.two_site_h(),
                hamiltonian.site_energy(),
                metadata.hermiticity_tolerance,
            )?;
            (ItebdScalarType::Complex, hamiltonian.dim())
        }
    };
    Ok((scalar_type, physical_dim))
}

pub(super) fn write_root(
    file: &File,
    path: &Path,
    metadata: &ItebdRunMetadata,
    hamiltonian: &ItebdHamiltonian,
    scalar_type: ItebdScalarType,
    physical_dim: usize,
) -> Result<(), ItebdCheckpointError> {
    write_string_attribute(file, path, "type", FILE_TYPE)?;
    write_scalar_attribute(file, path, "schema_version", &SCHEMA_VERSION)?;
    write_scalar_attribute(file, path, "unit_cell_size", &UNIT_CELL_SIZE)?;
    write_string_attribute(
        file,
        path,
        "scalar_type",
        match scalar_type {
            ItebdScalarType::Real => "real",
            ItebdScalarType::Complex => "complex",
        },
    )?;
    write_scalar_attribute(file, path, "physical_dim", &(physical_dim as u64))?;
    write_scalar_attribute(file, path, "ancilla_dim", &(physical_dim as u64))?;
    write_string_attribute(file, path, "basis_order", BASIS_ORDER)?;

    let run = file
        .create_group("run")
        .map_err(|error| io_error(path, error))?;
    let metadata_json =
        serde_json::to_string(&MetadataRecord::from(metadata)).map_err(|error| {
            schema_error(
                path,
                "run/metadata_json",
                format!("cannot serialize metadata: {error}"),
            )
        })?;
    write_string_dataset(&run, path, "metadata_json", &metadata_json)?;
    let hamiltonian_group = run
        .create_group("hamiltonian")
        .map_err(|error| io_error(path, error))?;
    match hamiltonian {
        ItebdHamiltonian::Real(hamiltonian) => {
            write_f64_vector(
                &hamiltonian_group,
                path,
                "two_site_h_real",
                hamiltonian.two_site_h.as_slice(),
            )?;
            write_f64_vector(
                &hamiltonian_group,
                path,
                "site_energy_real",
                hamiltonian.site_energy.as_slice(),
            )?;
        }
        ItebdHamiltonian::Complex(hamiltonian) => {
            write_complex_matrix(
                &hamiltonian_group,
                path,
                "two_site_h",
                hamiltonian.two_site_h(),
            )?;
            write_complex_matrix(
                &hamiltonian_group,
                path,
                "site_energy",
                hamiltonian.site_energy(),
            )?;
        }
    }
    file.create_group("states")
        .map_err(|error| io_error(path, error))?;
    Ok(())
}

pub(super) fn read_root(
    file: &File,
    path: &Path,
    acceptance_tolerance: Option<f64>,
) -> Result<ValidatedRoot, ItebdCheckpointError> {
    let file_type = read_string_attribute(file, path, "type")?;
    if file_type != FILE_TYPE {
        return Err(schema_error(
            path,
            "type",
            format!("expected {FILE_TYPE:?}, got {file_type:?}"),
        ));
    }
    let version = read_scalar_attribute::<u32>(file, path, "schema_version", "u32")?;
    if version != SCHEMA_VERSION {
        return Err(schema_error(
            path,
            "schema_version",
            format!("unsupported schema version {version}; expected {SCHEMA_VERSION}"),
        ));
    }
    let unit_cell_size = read_scalar_attribute::<u32>(file, path, "unit_cell_size", "u32")?;
    if unit_cell_size != UNIT_CELL_SIZE {
        return Err(schema_error(
            path,
            "unit_cell_size",
            format!("expected {UNIT_CELL_SIZE}, got {unit_cell_size}"),
        ));
    }
    let scalar_type = match read_string_attribute(file, path, "scalar_type")?.as_str() {
        "real" => ItebdScalarType::Real,
        "complex" => ItebdScalarType::Complex,
        value => {
            return Err(schema_error(
                path,
                "scalar_type",
                format!("expected \"real\" or \"complex\", got {value:?}"),
            ))
        }
    };
    let physical_dim = read_dimension_attribute(file, path, "physical_dim")?;
    let ancilla_dim = read_dimension_attribute(file, path, "ancilla_dim")?;
    if ancilla_dim != physical_dim {
        return Err(schema_error(
            path,
            "ancilla_dim",
            format!(
                "must equal physical_dim {physical_dim} for a purified iTEBD state, got {ancilla_dim}"
            ),
        ));
    }
    let basis_order = read_string_attribute(file, path, "basis_order")?;
    if basis_order != BASIS_ORDER {
        return Err(schema_error(
            path,
            "basis_order",
            format!("expected {BASIS_ORDER:?}, got {basis_order:?}"),
        ));
    }

    let run = file
        .group("run")
        .map_err(|error| schema_error(path, "run", error.to_string()))?;
    let metadata_json = read_string_dataset(&run, path, "metadata_json")?;
    let metadata: ItebdRunMetadata = serde_json::from_str::<MetadataRecord>(&metadata_json)
        .map(Into::into)
        .map_err(|error| {
            schema_error(
                path,
                "run/metadata_json",
                format!("invalid schema-version-{version} metadata: {error}"),
            )
        })?;
    validate_metadata(path, &metadata)?;
    let tolerance = acceptance_tolerance.unwrap_or(metadata.hermiticity_tolerance);
    validate_tolerance(path, "hermiticity_tolerance", tolerance)?;

    let group = run
        .group("hamiltonian")
        .map_err(|error| schema_error(path, "run/hamiltonian", error.to_string()))?;
    let matrix_dimension = physical_dim
        .checked_mul(physical_dim)
        .ok_or_else(|| schema_error(path, "physical_dim", "square overflows usize"))?;
    let matrix_length = matrix_dimension
        .checked_mul(matrix_dimension)
        .ok_or_else(|| schema_error(path, "physical_dim", "fourth power overflows usize"))?;
    let hamiltonian = match scalar_type {
        ItebdScalarType::Real => {
            let two_site_h = schema_field(
                read_f64_vector(&group, path, "two_site_h_real", matrix_length),
                path,
                "run/hamiltonian/two_site_h_real",
            )?;
            let site_energy = schema_field(
                read_f64_vector(&group, path, "site_energy_real", matrix_length),
                path,
                "run/hamiltonian/site_energy_real",
            )?;
            let local = LocalHamiltonian {
                two_site_h: DMatrix::from_column_slice(
                    matrix_dimension,
                    matrix_dimension,
                    &two_site_h,
                ),
                site_energy: DMatrix::from_column_slice(
                    matrix_dimension,
                    matrix_dimension,
                    &site_energy,
                ),
            };
            let actual_dim = validate_real_hamiltonian(path, &local, tolerance)?;
            if actual_dim != physical_dim {
                return Err(schema_error(
                    path,
                    "physical_dim",
                    format!(
                        "Hamiltonian physical dimension {actual_dim} does not match root {physical_dim}"
                    ),
                ));
            }
            ItebdHamiltonian::Real(local)
        }
        ItebdScalarType::Complex => {
            let two_site_h =
                read_complex_matrix(&group, path, "two_site_h", matrix_dimension, matrix_length)?;
            let site_energy =
                read_complex_matrix(&group, path, "site_energy", matrix_dimension, matrix_length)?;
            let local =
                ComplexLocalHamiltonian::try_new_with_tolerance(two_site_h, site_energy, tolerance)
                    .map_err(|error| {
                        schema_error(
                            path,
                            "run/hamiltonian",
                            format!("Hamiltonian rejected at load tolerance {tolerance}: {error}"),
                        )
                    })?;
            if local.dim() != physical_dim {
                return Err(schema_error(
                    path,
                    "physical_dim",
                    format!(
                        "Hamiltonian physical dimension {} does not match root {physical_dim}",
                        local.dim()
                    ),
                ));
            }
            ItebdHamiltonian::Complex(local)
        }
    };
    file.group("states")
        .map_err(|error| schema_error(path, "states", error.to_string()))?;
    Ok(ValidatedRoot {
        metadata,
        scalar_type,
        physical_dim,
        ancilla_dim,
        hamiltonian,
    })
}

fn validate_metadata(path: &Path, metadata: &ItebdRunMetadata) -> Result<(), ItebdCheckpointError> {
    if !metadata.dtau.is_finite() || metadata.dtau <= 0.0 {
        return Err(schema_error(
            path,
            "run/metadata_json.dtau",
            "must be finite and positive",
        ));
    }
    if !metadata.truncation.epsilon.is_finite() || metadata.truncation.epsilon <= 0.0 {
        return Err(schema_error(
            path,
            "run/metadata_json.truncation.epsilon",
            "must be finite and positive",
        ));
    }
    if metadata.truncation.max_bond == Some(0) {
        return Err(schema_error(
            path,
            "run/metadata_json.truncation.max_bond",
            "must be positive when present",
        ));
    }
    if metadata.canonicalize_every == 0 {
        return Err(schema_error(
            path,
            "run/metadata_json.canonicalize_every",
            "must be at least 1",
        ));
    }
    if let Some(interval) = metadata.record_every_beta {
        if !interval.is_finite() || interval <= 0.0 {
            return Err(schema_error(
                path,
                "run/metadata_json.record_every_beta",
                "must be finite and positive when present",
            ));
        }
    }
    validate_tolerance(
        path,
        "run/metadata_json.hermiticity_tolerance",
        metadata.hermiticity_tolerance,
    )
}

fn validate_real_hamiltonian(
    path: &Path,
    hamiltonian: &LocalHamiltonian,
    tolerance: f64,
) -> Result<usize, ItebdCheckpointError> {
    validate_real_matrix_shape(
        path,
        "run/hamiltonian/two_site_h_real",
        &hamiltonian.two_site_h,
    )?;
    validate_real_matrix_shape(
        path,
        "run/hamiltonian/site_energy_real",
        &hamiltonian.site_energy,
    )?;
    if hamiltonian.two_site_h.shape() != hamiltonian.site_energy.shape() {
        return Err(schema_error(
            path,
            "run/hamiltonian",
            format!(
                "matrix shapes differ: {:?} and {:?}",
                hamiltonian.two_site_h.shape(),
                hamiltonian.site_energy.shape()
            ),
        ));
    }
    let matrix_dimension = hamiltonian.two_site_h.nrows();
    let physical_dim = exact_square_root(matrix_dimension).ok_or_else(|| {
        schema_error(
            path,
            "run/hamiltonian/two_site_h_real",
            format!("matrix dimension {matrix_dimension} is not a positive perfect square"),
        )
    })?;
    validate_real_hermitian(
        path,
        "run/hamiltonian/two_site_h_real",
        &hamiltonian.two_site_h,
        tolerance,
    )?;
    validate_real_hermitian(
        path,
        "run/hamiltonian/site_energy_real",
        &hamiltonian.site_energy,
        tolerance,
    )?;
    Ok(physical_dim)
}

fn validate_complex_hamiltonian_parts(
    path: &Path,
    two_site_h: &DMatrix<Complex64>,
    site_energy: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<(), ItebdCheckpointError> {
    if two_site_h.shape() != site_energy.shape() {
        return Err(schema_error(
            path,
            "run/hamiltonian",
            format!(
                "matrix shapes differ: {:?} and {:?}",
                two_site_h.shape(),
                site_energy.shape()
            ),
        ));
    }
    ComplexLocalHamiltonian::try_new_with_tolerance(
        two_site_h.clone(),
        site_energy.clone(),
        tolerance,
    )
    .map(|_| ())
    .map_err(|error| schema_error(path, "run/hamiltonian", error.to_string()))
}

fn validate_real_matrix_shape(
    path: &Path,
    field: &str,
    matrix: &DMatrix<f64>,
) -> Result<(), ItebdCheckpointError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(schema_error(
            path,
            field,
            format!("must be nonempty and square, got {:?}", matrix.shape()),
        ));
    }
    if let Some((offset, value)) = matrix
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(schema_error(
            path,
            field,
            format!("entry at column-major offset {offset} is non-finite: {value}"),
        ));
    }
    Ok(())
}

fn validate_real_hermitian(
    path: &Path,
    field: &str,
    matrix: &DMatrix<f64>,
    tolerance: f64,
) -> Result<(), ItebdCheckpointError> {
    let scale = matrix.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(());
    }
    let matrix_norm = matrix
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value / scale));
    let mut residual_norm = 0.0_f64;
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let residual = matrix[(row, column)] / scale - matrix[(column, row)] / scale;
            residual_norm = residual_norm.hypot(residual);
        }
    }
    let residual = if scale >= matrix_norm.recip() {
        residual_norm / matrix_norm
    } else {
        scale * residual_norm
    };
    if !residual.is_finite() || residual > tolerance {
        return Err(schema_error(
            path,
            field,
            format!("Hermiticity residual {residual} exceeds tolerance {tolerance}"),
        ));
    }
    Ok(())
}

fn exact_square_root(value: usize) -> Option<usize> {
    let root = (value as f64).sqrt() as usize;
    [root, root.saturating_add(1)]
        .into_iter()
        .find(|candidate| candidate.checked_mul(*candidate) == Some(value) && *candidate > 0)
}

fn validate_tolerance(
    path: &Path,
    field: &str,
    tolerance: f64,
) -> Result<(), ItebdCheckpointError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        Err(schema_error(path, field, "must be finite and non-negative"))
    } else {
        Ok(())
    }
}

fn write_complex_matrix(
    group: &Group,
    path: &Path,
    stem: &str,
    matrix: &DMatrix<Complex64>,
) -> Result<(), ItebdCheckpointError> {
    let real: Vec<_> = matrix.iter().map(|value| value.re).collect();
    let imag: Vec<_> = matrix.iter().map(|value| value.im).collect();
    write_f64_vector(group, path, &format!("{stem}_real"), &real)?;
    write_f64_vector(group, path, &format!("{stem}_imag"), &imag)
}

fn read_complex_matrix(
    group: &Group,
    path: &Path,
    stem: &str,
    matrix_dimension: usize,
    matrix_length: usize,
) -> Result<DMatrix<Complex64>, ItebdCheckpointError> {
    let real_name = format!("{stem}_real");
    let imag_name = format!("{stem}_imag");
    let real = schema_field(
        read_f64_vector(group, path, &real_name, matrix_length),
        path,
        format!("run/hamiltonian/{real_name}"),
    )?;
    let imag = schema_field(
        read_f64_vector(group, path, &imag_name, matrix_length),
        path,
        format!("run/hamiltonian/{imag_name}"),
    )?;
    let values: Vec<_> = real
        .into_iter()
        .zip(imag)
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    Ok(DMatrix::from_column_slice(
        matrix_dimension,
        matrix_dimension,
        &values,
    ))
}

fn schema_field<T>(
    result: Result<T, ItebdCheckpointError>,
    path: &Path,
    field: impl Into<String>,
) -> Result<T, ItebdCheckpointError> {
    let field = field.into();
    result.map_err(|error| match error {
        ItebdCheckpointError::Schema { reason, .. } => schema_error(path, field, reason),
        other => other,
    })
}

pub(super) fn write_string_dataset(
    group: &Group,
    path: &Path,
    name: &str,
    value: &str,
) -> Result<(), ItebdCheckpointError> {
    let value = VarLenUnicode::from_str(value)
        .map_err(|error| schema_error(path, name, error.to_string()))?;
    group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create(name)
        .and_then(|dataset| dataset.write_scalar(&value))
        .map_err(|error| dataset_write_error(group, path, name, error))
}

pub(super) fn read_string_dataset(
    group: &Group,
    path: &Path,
    name: &str,
) -> Result<String, ItebdCheckpointError> {
    let dataset = group.dataset(name).map_err(|error| {
        schema_error(path, name, format!("required dataset is missing: {error}"))
    })?;
    if !dataset.shape().is_empty() {
        return Err(schema_error(path, name, "must be scalar"));
    }
    let dtype = dataset
        .dtype()
        .map_err(|error| schema_error(path, name, format!("cannot inspect dtype: {error}")))?;
    if !dtype.is::<VarLenUnicode>() {
        return Err(schema_error(
            path,
            name,
            format!("must have VarLenUnicode dtype, got {dtype}"),
        ));
    }
    dataset
        .read_scalar::<VarLenUnicode>()
        .map(|value| value.as_str().to_string())
        .map_err(|error| schema_error(path, name, error.to_string()))
}

pub(super) fn write_scalar_dataset<T: H5Type>(
    group: &Group,
    path: &Path,
    name: &str,
    value: &T,
) -> Result<(), ItebdCheckpointError> {
    group
        .new_dataset::<T>()
        .shape(())
        .create(name)
        .and_then(|dataset| dataset.write_scalar(value))
        .map_err(|error| dataset_write_error(group, path, name, error))
}

pub(super) fn read_scalar_dataset<T: H5Type>(
    group: &Group,
    path: &Path,
    name: &str,
    expected_type: &str,
) -> Result<T, ItebdCheckpointError> {
    let dataset = group.dataset(name).map_err(|error| {
        schema_error(path, name, format!("required dataset is missing: {error}"))
    })?;
    if !dataset.shape().is_empty() {
        return Err(schema_error(path, name, "must be scalar"));
    }
    let dtype = dataset
        .dtype()
        .map_err(|error| schema_error(path, name, format!("cannot inspect dtype: {error}")))?;
    if !dtype.is::<T>() {
        return Err(schema_error(
            path,
            name,
            format!("must have {expected_type} dtype, got {dtype}"),
        ));
    }
    dataset
        .read_scalar::<T>()
        .map_err(|error| schema_error(path, name, error.to_string()))
}

pub(super) fn write_f64_vector(
    group: &Group,
    path: &Path,
    name: &str,
    values: &[f64],
) -> Result<(), ItebdCheckpointError> {
    group
        .new_dataset::<f64>()
        .shape(values.len())
        .create(name)
        .and_then(|dataset| dataset.write_raw(values))
        .map_err(|error| dataset_write_error(group, path, name, error))
}

pub(super) fn write_u8_vector(
    group: &Group,
    path: &Path,
    name: &str,
    values: &[u8],
) -> Result<(), ItebdCheckpointError> {
    group
        .new_dataset::<u8>()
        .shape(values.len())
        .create(name)
        .and_then(|dataset| dataset.write_raw(values))
        .map_err(|error| dataset_write_error(group, path, name, error))
}

pub(super) fn read_f64_vector(
    group: &Group,
    path: &Path,
    name: &str,
    expected_length: usize,
) -> Result<Vec<f64>, ItebdCheckpointError> {
    read_exact_vector(group, path, name, expected_length, "f64")
}

pub(super) fn read_u8_vector(
    group: &Group,
    path: &Path,
    name: &str,
    expected_length: usize,
) -> Result<Vec<u8>, ItebdCheckpointError> {
    read_exact_vector(group, path, name, expected_length, "u8")
}

fn read_exact_vector<T: H5Type>(
    group: &Group,
    path: &Path,
    name: &str,
    expected_length: usize,
    expected_type: &str,
) -> Result<Vec<T>, ItebdCheckpointError> {
    let dataset = group.dataset(name).map_err(|error| {
        schema_error(path, name, format!("required dataset is missing: {error}"))
    })?;
    if dataset.shape() != [expected_length] {
        return Err(schema_error(
            path,
            name,
            format!(
                "must have shape [{expected_length}], got {:?}",
                dataset.shape()
            ),
        ));
    }
    let dtype = dataset
        .dtype()
        .map_err(|error| schema_error(path, name, format!("cannot inspect dtype: {error}")))?;
    if !dtype.is::<T>() {
        return Err(schema_error(
            path,
            name,
            format!("must have {expected_type} dtype, got {dtype}"),
        ));
    }
    dataset
        .read_raw::<T>()
        .map_err(|error| schema_error(path, name, error.to_string()))
}

fn write_string_attribute(
    location: &Location,
    path: &Path,
    name: &str,
    value: &str,
) -> Result<(), ItebdCheckpointError> {
    let value = VarLenUnicode::from_str(value)
        .map_err(|error| schema_error(path, name, error.to_string()))?;
    location
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .and_then(|attribute| attribute.write_scalar(&value))
        .map_err(|error| attribute_write_error(location, path, name, error))
}

pub(super) fn write_scalar_attribute<T: H5Type>(
    location: &Location,
    path: &Path,
    name: &str,
    value: &T,
) -> Result<(), ItebdCheckpointError> {
    location
        .new_attr::<T>()
        .shape(())
        .create(name)
        .and_then(|attribute| attribute.write_scalar(value))
        .map_err(|error| attribute_write_error(location, path, name, error))
}

fn dataset_write_error(
    group: &Group,
    path: &Path,
    name: &str,
    error: impl std::fmt::Display,
) -> ItebdCheckpointError {
    io_error(
        path,
        format!(
            "HDF5 object {:?} dataset {name:?} write failed: {error}",
            group.name()
        ),
    )
}

fn attribute_write_error(
    location: &Location,
    path: &Path,
    name: &str,
    error: impl std::fmt::Display,
) -> ItebdCheckpointError {
    io_error(
        path,
        format!(
            "HDF5 object {:?} attribute {name:?} write failed: {error}",
            location.name()
        ),
    )
}

fn read_string_attribute(
    location: &Location,
    path: &Path,
    name: &str,
) -> Result<String, ItebdCheckpointError> {
    read_scalar_attribute::<VarLenUnicode>(location, path, name, "VarLenUnicode")
        .map(|value| value.as_str().to_string())
}

pub(super) fn read_scalar_attribute<T: H5Type>(
    location: &Location,
    path: &Path,
    name: &str,
    expected_type: &str,
) -> Result<T, ItebdCheckpointError> {
    let attribute = location.attr(name).map_err(|error| {
        schema_error(
            path,
            name,
            format!("required attribute is missing: {error}"),
        )
    })?;
    if !attribute.shape().is_empty() {
        return Err(schema_error(path, name, "must be scalar"));
    }
    let dtype = attribute
        .dtype()
        .map_err(|error| schema_error(path, name, format!("cannot inspect dtype: {error}")))?;
    if !dtype.is::<T>() {
        return Err(schema_error(
            path,
            name,
            format!("must have {expected_type} dtype, got {dtype}"),
        ));
    }
    attribute
        .read_scalar::<T>()
        .map_err(|error| schema_error(path, name, error.to_string()))
}

fn read_dimension_attribute(
    file: &File,
    path: &Path,
    name: &str,
) -> Result<usize, ItebdCheckpointError> {
    let value = read_scalar_attribute::<u64>(file, path, name, "u64")?;
    let value =
        usize::try_from(value).map_err(|_| schema_error(path, name, "does not fit in usize"))?;
    if value == 0 {
        return Err(schema_error(path, name, "must be positive"));
    }
    Ok(value)
}

pub(super) fn schema_error(
    path: &Path,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> ItebdCheckpointError {
    ItebdCheckpointError::Schema {
        path: path.to_path_buf(),
        field: field.into(),
        reason: reason.into(),
    }
}

pub(super) fn io_error(path: &Path, error: impl std::fmt::Display) -> ItebdCheckpointError {
    ItebdCheckpointError::Io {
        path: path.to_path_buf(),
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        write_f64_vector, write_scalar_attribute, write_scalar_dataset, write_string_attribute,
        write_string_dataset, write_u8_vector,
    };
    use crate::itebd_checkpoint::ItebdCheckpointError;
    use hdf5_metno::types::VarLenUnicode;
    use std::str::FromStr;

    fn io_reason(error: ItebdCheckpointError, path: &std::path::Path) -> String {
        match error {
            ItebdCheckpointError::Io {
                path: error_path,
                reason,
            } if error_path == path => reason,
            other => panic!("expected I/O error for {path:?}, got {other:?}"),
        }
    }

    fn assert_collision_context(reason: &str, object: &str, field: &str, failure: &str) {
        assert!(
            reason.contains(&format!("object {object:?}")),
            "missing object context in {reason:?}"
        );
        assert!(
            reason.contains(field),
            "missing field context in {reason:?}"
        );
        assert!(
            reason.contains(failure),
            "missing underlying HDF5 failure in {reason:?}"
        );
    }

    #[test]
    fn shared_writers_report_hdf5_object_field_and_underlying_collision() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("writer-context.h5");
        let file = hdf5_metno::File::create(&path).unwrap();

        let string_dataset = file.create_group("string_dataset").unwrap();
        string_dataset
            .new_dataset::<VarLenUnicode>()
            .shape(())
            .create("value")
            .unwrap();
        let reason = io_reason(
            write_string_dataset(&string_dataset, &path, "value", "x").unwrap_err(),
            &path,
        );
        assert_collision_context(
            &reason,
            "/string_dataset",
            "dataset \"value\"",
            "unknown library error",
        );

        let scalar_dataset = file.create_group("scalar_dataset").unwrap();
        scalar_dataset
            .new_dataset::<u64>()
            .shape(())
            .create("value")
            .unwrap();
        let reason = io_reason(
            write_scalar_dataset(&scalar_dataset, &path, "value", &1_u64).unwrap_err(),
            &path,
        );
        assert_collision_context(
            &reason,
            "/scalar_dataset",
            "dataset \"value\"",
            "unknown library error",
        );

        let f64_vector = file.create_group("f64_vector").unwrap();
        f64_vector
            .new_dataset::<f64>()
            .shape(1)
            .create("value")
            .unwrap();
        let reason = io_reason(
            write_f64_vector(&f64_vector, &path, "value", &[1.0]).unwrap_err(),
            &path,
        );
        assert_collision_context(
            &reason,
            "/f64_vector",
            "dataset \"value\"",
            "unknown library error",
        );

        let u8_vector = file.create_group("u8_vector").unwrap();
        u8_vector
            .new_dataset::<u8>()
            .shape(1)
            .create("value")
            .unwrap();
        let reason = io_reason(
            write_u8_vector(&u8_vector, &path, "value", &[1]).unwrap_err(),
            &path,
        );
        assert_collision_context(
            &reason,
            "/u8_vector",
            "dataset \"value\"",
            "unknown library error",
        );

        let string_attribute = file.create_group("string_attribute").unwrap();
        string_attribute
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("value")
            .unwrap()
            .write_scalar(&VarLenUnicode::from_str("old").unwrap())
            .unwrap();
        let reason = io_reason(
            write_string_attribute(&string_attribute, &path, "value", "new").unwrap_err(),
            &path,
        );
        assert_collision_context(
            &reason,
            "/string_attribute",
            "attribute \"value\"",
            "attribute already exists",
        );

        let scalar_attribute = file.create_group("scalar_attribute").unwrap();
        scalar_attribute
            .new_attr::<u8>()
            .shape(())
            .create("value")
            .unwrap()
            .write_scalar(&0_u8)
            .unwrap();
        let reason = io_reason(
            write_scalar_attribute(&scalar_attribute, &path, "value", &1_u8).unwrap_err(),
            &path,
        );
        assert_collision_context(
            &reason,
            "/scalar_attribute",
            "attribute \"value\"",
            "attribute already exists",
        );
    }
}
