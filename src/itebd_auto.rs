use crate::itebd::StepInfo;
use crate::itebd_complex::{
    canonicalize_complex, energy_density_complex, imaginary_time_step_complex,
    imaginary_time_step_second_order_complex, local_expectation_complex, real_observable,
    specific_heat_complex, specific_heat_complex_with_options, validate_local_operator,
    ComplexLocalHamiltonian, ComplexPurifiedMps,
};
use crate::itebd_error::ItebdError;
use crate::model::LocalHamiltonian;
use crate::purified_mps::PurifiedMps;
use crate::specific_heat::{SpecificHeatOptions, SpecificHeatReport};
use crate::tensor::Truncation;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub enum ItebdHamiltonian {
    Real(LocalHamiltonian),
    Complex(ComplexLocalHamiltonian),
}

pub enum ItebdState {
    Real(PurifiedMps),
    Complex(ComplexPurifiedMps),
}

impl ItebdHamiltonian {
    pub fn try_from_complex(
        two_site_h: DMatrix<Complex64>,
        site_energy: DMatrix<Complex64>,
    ) -> Result<Self, ItebdError> {
        let validated = ComplexLocalHamiltonian::try_new(two_site_h, site_energy)?;
        let is_exactly_real = validated.two_site_h().iter().all(|value| value.im == 0.0)
            && validated.site_energy().iter().all(|value| value.im == 0.0);
        if is_exactly_real {
            let (two_site_h, site_energy) = validated.into_parts();
            Ok(Self::Real(LocalHamiltonian {
                two_site_h: two_site_h.map(|value| value.re),
                site_energy: site_energy.map(|value| value.re),
            }))
        } else {
            Ok(Self::Complex(validated))
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Real(hamiltonian) => hamiltonian.dim(),
            Self::Complex(hamiltonian) => hamiltonian.dim(),
        }
    }
}

impl ItebdState {
    pub fn infinite_temperature(hamiltonian: &ItebdHamiltonian) -> Result<Self, ItebdError> {
        match hamiltonian {
            ItebdHamiltonian::Real(hamiltonian) => {
                let dimension = hamiltonian.dim();
                if dimension == 0 {
                    return Err(ItebdError::InvalidPhysicalDimension { matrix_dim: 0 });
                }
                Ok(Self::Real(crate::purified_mps::infinite_temperature(
                    dimension,
                )))
            }
            ItebdHamiltonian::Complex(hamiltonian) => Ok(Self::Complex(
                ComplexPurifiedMps::infinite_temperature(hamiltonian.dim())?,
            )),
        }
    }
}

fn backend_mismatch(state: &'static str, hamiltonian: &'static str) -> ItebdError {
    ItebdError::BackendMismatch { state, hamiltonian }
}

pub fn imaginary_time_step_auto(
    state: &mut ItebdState,
    hamiltonian: &ItebdHamiltonian,
    dtau: f64,
    truncation: &Truncation,
) -> Result<StepInfo, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(hamiltonian)) => Ok(
            crate::itebd::imaginary_time_step(state, hamiltonian, dtau, truncation),
        ),
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(hamiltonian)) => {
            imaginary_time_step_complex(state, hamiltonian, dtau, truncation)
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}

pub fn imaginary_time_step_second_order_auto(
    state: &mut ItebdState,
    hamiltonian: &ItebdHamiltonian,
    dtau: f64,
    truncation: &Truncation,
) -> Result<StepInfo, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(hamiltonian)) => Ok(
            crate::itebd::imaginary_time_step_second_order(state, hamiltonian, dtau, truncation),
        ),
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(hamiltonian)) => {
            imaginary_time_step_second_order_complex(state, hamiltonian, dtau, truncation)
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}

pub fn canonicalize_auto(
    state: &mut ItebdState,
    hamiltonian: &ItebdHamiltonian,
) -> Result<f64, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(_)) => {
            Ok(crate::canonicalize::canonicalize(state))
        }
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(hamiltonian)) => {
            canonicalize_complex(state, hamiltonian.hermiticity_tolerance())
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}

pub fn energy_density_auto(
    state: &ItebdState,
    hamiltonian: &ItebdHamiltonian,
) -> Result<f64, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(hamiltonian)) => {
            Ok(crate::observable::energy_density(state, hamiltonian))
        }
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(hamiltonian)) => {
            energy_density_complex(state, hamiltonian)
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}

pub fn local_expectation_auto(
    state: &ItebdState,
    hamiltonian: &ItebdHamiltonian,
    operator: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<f64, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(hamiltonian)) => {
            validate_local_operator(operator, hamiltonian.dim(), tolerance)?;
            let real_operator = operator.map(|value| value.re);
            let real = crate::observable::magnetization(state, &real_operator);
            if operator.iter().all(|value| value.im == 0.0) {
                return Ok(real);
            }
            let imaginary_operator = operator.map(|value| value.im);
            let imaginary = crate::observable::magnetization(state, &imaginary_operator);
            real_observable(
                "automatic local observable",
                Complex64::new(real, imaginary),
                tolerance,
            )
        }
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(_)) => {
            local_expectation_complex(state, operator, tolerance)
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}

pub fn specific_heat_auto(
    state: &ItebdState,
    hamiltonian: &ItebdHamiltonian,
    beta: f64,
) -> Result<f64, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(hamiltonian)) => {
            crate::variance::specific_heat(state, hamiltonian, beta)
        }
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(hamiltonian)) => {
            specific_heat_complex(state, hamiltonian, beta)
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}

pub fn specific_heat_with_options_auto(
    state: &ItebdState,
    hamiltonian: &ItebdHamiltonian,
    beta: f64,
    options: &SpecificHeatOptions,
) -> Result<SpecificHeatReport, ItebdError> {
    match (state, hamiltonian) {
        (ItebdState::Real(state), ItebdHamiltonian::Real(hamiltonian)) => {
            crate::variance::specific_heat_with_options(state, hamiltonian, beta, options)
        }
        (ItebdState::Complex(state), ItebdHamiltonian::Complex(hamiltonian)) => {
            specific_heat_complex_with_options(state, hamiltonian, beta, options)
        }
        (ItebdState::Real(_), ItebdHamiltonian::Complex(_)) => {
            Err(backend_mismatch("real", "complex"))
        }
        (ItebdState::Complex(_), ItebdHamiltonian::Real(_)) => {
            Err(backend_mismatch("complex", "real"))
        }
    }
}
