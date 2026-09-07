use crate::itebd_auto::ItebdState;
use crate::itebd_complex::{ComplexPurifiedMps, ComplexSite};
use crate::purified_mps::{PurifiedMps, Site};
use crate::tensor::{Idx, Tensor};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItebdScalarType {
    Real,
    Complex,
}

#[derive(Clone, Copy)]
pub enum ItebdStateRef<'a> {
    Real(&'a PurifiedMps),
    Complex(&'a ComplexPurifiedMps),
}

#[derive(Clone, Copy)]
pub struct ItebdSiteRef<'a> {
    pub gamma: &'a Tensor,
    pub left: &'a Idx,
    pub physical: &'a Idx,
    pub ancilla: &'a Idx,
    pub right: &'a Idx,
}

#[derive(Debug)]
pub struct ItebdStateSummary {
    pub scalar_type: ItebdScalarType,
    pub physical_dim: usize,
    pub ancilla_dim: usize,
    pub bond_dimensions: [usize; 2],
    pub schmidt_norms: [f64; 2],
    pub a_role_axes: [u8; 4],
    pub b_role_axes: [u8; 4],
}

#[derive(Debug, thiserror::Error)]
pub enum ItebdStateValidationError {
    #[error("invalid iTEBD state field `{field}`: {reason}")]
    InvalidField { field: String, reason: String },
    #[error("failed to access iTEBD state tensor `{field}`: {reason}")]
    Tensor { field: String, reason: String },
}

impl<'a> From<&'a PurifiedMps> for ItebdStateRef<'a> {
    fn from(state: &'a PurifiedMps) -> Self {
        Self::Real(state)
    }
}

impl<'a> From<&'a ComplexPurifiedMps> for ItebdStateRef<'a> {
    fn from(state: &'a ComplexPurifiedMps) -> Self {
        Self::Complex(state)
    }
}

impl<'a> From<&'a ItebdState> for ItebdStateRef<'a> {
    fn from(state: &'a ItebdState) -> Self {
        match state {
            ItebdState::Real(state) => Self::Real(state),
            ItebdState::Complex(state) => Self::Complex(state),
        }
    }
}

impl<'a> ItebdStateRef<'a> {
    pub fn a(self) -> ItebdSiteRef<'a> {
        match self {
            Self::Real(state) => ItebdSiteRef::from(&state.a),
            Self::Complex(state) => ItebdSiteRef::from(&state.a),
        }
    }

    pub fn b(self) -> ItebdSiteRef<'a> {
        match self {
            Self::Real(state) => ItebdSiteRef::from(&state.b),
            Self::Complex(state) => ItebdSiteRef::from(&state.b),
        }
    }

    pub fn lambda_ab(self) -> &'a [f64] {
        match self {
            Self::Real(state) => &state.lambda_ab,
            Self::Complex(state) => &state.lambda_ab,
        }
    }

    pub fn lambda_ba(self) -> &'a [f64] {
        match self {
            Self::Real(state) => &state.lambda_ba,
            Self::Complex(state) => &state.lambda_ba,
        }
    }

    pub fn scalar_type(self) -> ItebdScalarType {
        match self {
            Self::Real(_) => ItebdScalarType::Real,
            Self::Complex(_) => ItebdScalarType::Complex,
        }
    }

    pub fn validate(self) -> Result<ItebdStateSummary, ItebdStateValidationError> {
        let (bond_ab, bond_ba) = self.bonds();
        validate_distinct_roles(self.a(), self.b(), bond_ab, bond_ba)?;
        validate_shared_bonds(self.a(), self.b(), bond_ab, bond_ba)?;

        let a_role_axes = validate_site("a", self.a(), self.scalar_type())?;
        let b_role_axes = validate_site("b", self.b(), self.scalar_type())?;

        require_same_dimension(
            "a.physical",
            self.a().physical,
            "b.physical",
            self.b().physical,
        )?;
        require_same_dimension("a.ancilla", self.a().ancilla, "b.ancilla", self.b().ancilla)?;
        require_same_dimension("physical", self.a().physical, "ancilla", self.a().ancilla)?;

        let norm_ab = validate_schmidt("lambda_ab", self.lambda_ab(), bond_ab)?;
        let norm_ba = validate_schmidt("lambda_ba", self.lambda_ba(), bond_ba)?;

        Ok(ItebdStateSummary {
            scalar_type: self.scalar_type(),
            physical_dim: self.a().physical.dim,
            ancilla_dim: self.a().ancilla.dim,
            bond_dimensions: [bond_ab.dim, bond_ba.dim],
            schmidt_norms: [norm_ab, norm_ba],
            a_role_axes,
            b_role_axes,
        })
    }

    fn bonds(self) -> (&'a Idx, &'a Idx) {
        match self {
            Self::Real(state) => (&state.lambda_bond_ab, &state.lambda_bond_ba),
            Self::Complex(state) => (&state.lambda_bond_ab, &state.lambda_bond_ba),
        }
    }
}

impl<'a> From<&'a Site> for ItebdSiteRef<'a> {
    fn from(site: &'a Site) -> Self {
        Self {
            gamma: &site.gamma,
            left: &site.left,
            physical: &site.phys,
            ancilla: &site.anc,
            right: &site.right,
        }
    }
}

impl<'a> From<&'a ComplexSite> for ItebdSiteRef<'a> {
    fn from(site: &'a ComplexSite) -> Self {
        Self {
            gamma: &site.gamma,
            left: &site.left,
            physical: &site.phys,
            ancilla: &site.anc,
            right: &site.right,
        }
    }
}

fn validate_distinct_roles(
    a: ItebdSiteRef<'_>,
    b: ItebdSiteRef<'_>,
    bond_ab: &Idx,
    bond_ba: &Idx,
) -> Result<(), ItebdStateValidationError> {
    let roles = [
        ("bond_ab", bond_ab),
        ("bond_ba", bond_ba),
        ("a.physical", a.physical),
        ("a.ancilla", a.ancilla),
        ("b.physical", b.physical),
        ("b.ancilla", b.ancilla),
    ];
    for (position, (left_name, left)) in roles.iter().enumerate() {
        require_positive_dimension(left_name, left)?;
        for (right_name, right) in &roles[position + 1..] {
            if left == right {
                return Err(invalid(
                    format!("{left_name}/{right_name}"),
                    "role identities must be distinct",
                ));
            }
        }
    }
    Ok(())
}

fn validate_shared_bonds(
    a: ItebdSiteRef<'_>,
    b: ItebdSiteRef<'_>,
    bond_ab: &Idx,
    bond_ba: &Idx,
) -> Result<(), ItebdStateValidationError> {
    require_same_index("a.right", a.right, "b.left", b.left)?;
    require_same_index("a.right", a.right, "lambda_bond_ab", bond_ab)?;
    require_same_index("b.right", b.right, "a.left", a.left)?;
    require_same_index("b.right", b.right, "lambda_bond_ba", bond_ba)?;
    Ok(())
}

fn validate_site(
    label: &str,
    site: ItebdSiteRef<'_>,
    scalar_type: ItebdScalarType,
) -> Result<[u8; 4], ItebdStateValidationError> {
    let gamma_field = format!("{label}.gamma");
    if site.gamma.indices.len() != 4 {
        return Err(invalid(
            gamma_field,
            format!("expected rank 4, got {}", site.gamma.indices.len()),
        ));
    }
    match scalar_type {
        ItebdScalarType::Real if site.gamma.is_complex() => {
            return Err(invalid(gamma_field, "expected f64 storage"));
        }
        ItebdScalarType::Complex if !site.gamma.is_complex() => {
            return Err(invalid(gamma_field, "expected Complex64 storage"));
        }
        _ => {}
    }
    let storage = site.gamma.storage().map_err(|error| {
        tensor_error(
            gamma_field.clone(),
            format!("cannot inspect tensor storage: {error}"),
        )
    })?;
    if !storage.is_dense() {
        return Err(invalid(gamma_field, "expected dense storage"));
    }
    let gamma_dimensions = site.gamma.dims();
    if storage.logical_dims() != gamma_dimensions {
        return Err(invalid(
            gamma_field,
            format!(
                "storage shape {:?} does not match gamma index dimensions {gamma_dimensions:?}",
                storage.logical_dims()
            ),
        ));
    }

    let roles = [
        ("left", site.left),
        ("physical", site.physical),
        ("ancilla", site.ancilla),
        ("right", site.right),
    ];
    let mut positions = [0_u8; 4];
    for (role_number, (role_name, declared)) in roles.iter().enumerate() {
        require_positive_dimension(&format!("{label}.{role_name}"), declared)?;
        let matching_positions: Vec<_> = site
            .gamma
            .indices
            .iter()
            .enumerate()
            .filter_map(|(position, actual)| (actual == *declared).then_some(position))
            .collect();
        if matching_positions.len() != 1 {
            let same_id = site
                .gamma
                .indices
                .iter()
                .filter(|actual| actual.id == declared.id)
                .count();
            return Err(invalid(
                format!("{label}.gamma.{role_name}"),
                format!(
                    "expected exactly one axis with the declared identity, found {}; axes sharing only its raw id: {same_id}",
                    matching_positions.len()
                ),
            ));
        }
        let position = matching_positions[0];
        let actual = &site.gamma.indices[position];
        require_positive_dimension(&format!("{label}.gamma[{position}]"), actual)?;
        if actual.dim != declared.dim {
            return Err(invalid(
                format!("{label}.{role_name}"),
                format!(
                    "declared dimension {} does not match gamma axis {position} dimension {}",
                    declared.dim, actual.dim
                ),
            ));
        }
        positions[role_number] = position as u8;
    }

    let mut sorted_positions = positions;
    sorted_positions.sort_unstable();
    if sorted_positions != [0, 1, 2, 3] {
        return Err(invalid(
            gamma_field.clone(),
            "role axes do not form a permutation of the tensor axes",
        ));
    }

    let element_count = site
        .gamma
        .indices
        .iter()
        .try_fold(1_usize, |count, index| count.checked_mul(index.dim))
        .ok_or_else(|| invalid(gamma_field.clone(), "element count overflows usize"))?;
    match scalar_type {
        ItebdScalarType::Real => {
            let values = site.gamma.to_vec::<f64>().map_err(|error| {
                tensor_error(
                    gamma_field.clone(),
                    format!("cannot read f64 values: {error}"),
                )
            })?;
            validate_gamma_values(&gamma_field, element_count, values.len(), || {
                values.iter().all(|value| value.is_finite())
            })?;
        }
        ItebdScalarType::Complex => {
            let values = site.gamma.to_vec::<Complex64>().map_err(|error| {
                tensor_error(
                    gamma_field.clone(),
                    format!("cannot read Complex64 values: {error}"),
                )
            })?;
            validate_gamma_values(&gamma_field, element_count, values.len(), || {
                values
                    .iter()
                    .all(|value| value.re.is_finite() && value.im.is_finite())
            })?;
        }
    }
    Ok(positions)
}

fn validate_gamma_values(
    field: &str,
    expected_count: usize,
    actual_count: usize,
    all_finite: impl FnOnce() -> bool,
) -> Result<(), ItebdStateValidationError> {
    if actual_count != expected_count {
        return Err(invalid(
            field,
            format!("expected {expected_count} values, got {actual_count}"),
        ));
    }
    if !all_finite() {
        return Err(invalid(field, "contains a non-finite value"));
    }
    Ok(())
}

fn validate_schmidt(
    field: &str,
    values: &[f64],
    bond: &Idx,
) -> Result<f64, ItebdStateValidationError> {
    if values.len() != bond.dim {
        return Err(invalid(
            field,
            format!(
                "length {} does not match bond dimension {}",
                values.len(),
                bond.dim
            ),
        ));
    }
    if let Some((position, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            format!("{field}[{position}]"),
            format!("must be finite and non-negative, got {value}"),
        ));
    }
    let scale = values.iter().copied().fold(0.0_f64, f64::max);
    let norm = if scale == 0.0 {
        0.0
    } else {
        scale
            * values
                .iter()
                .map(|value| (value / scale).powi(2))
                .sum::<f64>()
                .sqrt()
    };
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid(
            field,
            format!("norm must be finite and positive, got {norm}"),
        ));
    }
    Ok(norm)
}

fn require_same_index(
    left_field: &str,
    left: &Idx,
    right_field: &str,
    right: &Idx,
) -> Result<(), ItebdStateValidationError> {
    if left.dim != right.dim {
        return Err(invalid(
            format!("{left_field}/{right_field}"),
            format!("dimensions differ: {} != {}", left.dim, right.dim),
        ));
    }
    if left != right {
        return Err(invalid(
            format!("{left_field}/{right_field}"),
            "identities differ",
        ));
    }
    Ok(())
}

fn require_same_dimension(
    left_field: &str,
    left: &Idx,
    right_field: &str,
    right: &Idx,
) -> Result<(), ItebdStateValidationError> {
    if left.dim != right.dim {
        return Err(invalid(
            format!("{left_field}/{right_field}"),
            format!("dimensions differ: {} != {}", left.dim, right.dim),
        ));
    }
    Ok(())
}

fn require_positive_dimension(field: &str, index: &Idx) -> Result<(), ItebdStateValidationError> {
    if index.dim == 0 {
        return Err(invalid(field, "dimension must be positive"));
    }
    Ok(())
}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ItebdStateValidationError {
    ItebdStateValidationError::InvalidField {
        field: field.into(),
        reason: reason.into(),
    }
}

fn tensor_error(field: impl Into<String>, reason: impl Into<String>) -> ItebdStateValidationError {
    ItebdStateValidationError::Tensor {
        field: field.into(),
        reason: reason.into(),
    }
}
