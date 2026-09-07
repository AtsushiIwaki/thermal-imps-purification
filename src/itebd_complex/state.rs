use super::tensor::complex_from_fn;
use crate::itebd_error::ItebdError;
use crate::tensor::{new_index, Idx, Tensor};
use num_complex::Complex64;

const TOPOLOGY_STAGE: &str = "complex_state_topology";

#[derive(Clone)]
pub struct ComplexSite {
    pub gamma: Tensor,
    pub left: Idx,
    pub phys: Idx,
    pub anc: Idx,
    pub right: Idx,
}

#[derive(Clone)]
pub struct ComplexPurifiedMps {
    pub a: ComplexSite,
    pub b: ComplexSite,
    pub lambda_ab: Vec<f64>,
    pub lambda_bond_ab: Idx,
    pub lambda_ba: Vec<f64>,
    pub lambda_bond_ba: Idx,
}

impl ComplexPurifiedMps {
    pub fn infinite_temperature(dimension: usize) -> Result<Self, ItebdError> {
        if dimension == 0 {
            return Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 0 });
        }
        let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
        let bond_ab = new_index(1);
        let bond_ba = new_index(1);
        let a = infinite_temperature_site(&bond_ba, &bond_ab, dimension, amplitude)?;
        let b = infinite_temperature_site(&bond_ab, &bond_ba, dimension, amplitude)?;
        let state = Self {
            a,
            b,
            lambda_ab: vec![1.0],
            lambda_bond_ab: bond_ab,
            lambda_ba: vec![1.0],
            lambda_bond_ba: bond_ba,
        };
        state.validate_topology()?;
        Ok(state)
    }

    pub fn validate_topology(&self) -> Result<(), ItebdError> {
        validate_site("A", &self.a)?;
        validate_site("B", &self.b)?;

        require_index_match("A.right", &self.a.right, "B.left", &self.b.left)?;
        require_index_match(
            "A.right/B.left",
            &self.a.right,
            "lambda_bond_ab",
            &self.lambda_bond_ab,
        )?;
        require_index_match("B.right", &self.b.right, "A.left", &self.a.left)?;
        require_index_match(
            "B.right/A.left",
            &self.b.right,
            "lambda_bond_ba",
            &self.lambda_bond_ba,
        )?;
        if self.lambda_bond_ab == self.lambda_bond_ba {
            return Err(topology_error(
                "lambda_bond_ab and lambda_bond_ba must be distinct",
            ));
        }
        validate_schmidt_values("lambda_ab", &self.lambda_ab, &self.lambda_bond_ab)?;
        validate_schmidt_values("lambda_ba", &self.lambda_ba, &self.lambda_bond_ba)?;
        if self.a.phys.dim != self.b.phys.dim {
            return Err(topology_error(format!(
                "A.phys dimension {} does not match B.phys dimension {}",
                self.a.phys.dim, self.b.phys.dim
            )));
        }
        Ok(())
    }
}

fn infinite_temperature_site(
    left: &Idx,
    right: &Idx,
    dimension: usize,
    amplitude: Complex64,
) -> Result<ComplexSite, ItebdError> {
    let phys = new_index(dimension);
    let anc = new_index(dimension);
    let gamma = complex_from_fn(
        &[left.clone(), phys.clone(), anc.clone(), right.clone()],
        |index| {
            if index[1] == index[2] {
                amplitude
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )?;
    Ok(ComplexSite {
        gamma,
        left: left.clone(),
        phys,
        anc,
        right: right.clone(),
    })
}

fn validate_site(label: &'static str, site: &ComplexSite) -> Result<(), ItebdError> {
    if !site.gamma.is_complex() {
        return Err(topology_error(format!(
            "{label}.gamma must use Complex64 storage"
        )));
    }
    let expected = [&site.left, &site.phys, &site.anc, &site.right];
    if site.gamma.indices.len() != expected.len() {
        return Err(topology_error(format!(
            "{label}.gamma must have four indices, got {}",
            site.gamma.indices.len()
        )));
    }
    for (position, ((actual, expected), field)) in site
        .gamma
        .indices
        .iter()
        .zip(expected)
        .zip(["left", "phys", "anc", "right"])
        .enumerate()
    {
        if actual.dim != expected.dim {
            return Err(topology_error(format!(
                "{label}.gamma index {position} dimension {} does not match declared {label}.{field} dimension {}",
                actual.dim, expected.dim
            )));
        }
        if actual != expected {
            return Err(topology_error(format!(
                "{label}.gamma index {position} does not match declared {label}.{field}"
            )));
        }
    }
    if site.phys.dim == 0 || site.anc.dim == 0 || site.phys.dim != site.anc.dim {
        return Err(topology_error(format!(
            "{label} physical/ancilla dimensions must be equal and positive, got {}/{}",
            site.phys.dim, site.anc.dim
        )));
    }
    let values = site
        .gamma
        .to_vec::<Complex64>()
        .map_err(|error| topology_error(format!("cannot materialize {label}.gamma: {error}")))?;
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(topology_error(format!(
            "{label}.gamma contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_schmidt_values(
    label: &'static str,
    values: &[f64],
    bond: &Idx,
) -> Result<(), ItebdError> {
    if values.is_empty() || values.len() != bond.dim {
        return Err(topology_error(format!(
            "{label} length {} does not match positive bond dimension {}",
            values.len(),
            bond.dim
        )));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        return Err(topology_error(format!(
            "{label}[{index}] must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

fn require_index_match(
    left_label: &'static str,
    left: &Idx,
    right_label: &'static str,
    right: &Idx,
) -> Result<(), ItebdError> {
    if left.dim != right.dim {
        return Err(topology_error(format!(
            "{left_label} dimension {} must match {right_label} dimension {}",
            left.dim, right.dim
        )));
    }
    if left != right {
        return Err(topology_error(format!(
            "{left_label} must fully match {right_label}"
        )));
    }
    Ok(())
}

fn topology_error(message: impl ToString) -> ItebdError {
    ItebdError::TensorOperation {
        stage: TOPOLOGY_STAGE,
        message: message.to_string(),
    }
}
