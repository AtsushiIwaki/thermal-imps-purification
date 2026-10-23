//! Finite-temperature iMPS purification with iTEBD.

pub mod canonicalize;
pub mod config;
#[cfg(test)]
pub(crate) mod contraction_benchmark;
#[cfg(test)]
mod contraction_boundary_audit;
#[cfg(test)]
pub(crate) mod contraction_contract_candidate;
pub(crate) mod contraction_pairwise;
pub mod exact;
mod git_process;
pub mod ising_reference;
pub mod itebd;
pub mod itebd_auto;
pub mod itebd_checkpoint;
pub mod itebd_complex;
pub mod itebd_error;
pub mod itebd_rdm;
pub mod itebd_state_view;
pub mod model;
pub mod observable;
pub mod purified_mps;
pub mod runner;
pub mod solve_run;
pub mod specific_heat;
pub mod tensor;
#[cfg(test)]
pub(crate) mod test_tensor_support;
pub mod transfer;
pub mod variance;

mod reference_trajectory;

pub mod aklt_reference;
