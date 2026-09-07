use thermal_imps_purification::config::{TrotterOrder, TruncationCfg};
use thermal_imps_purification::itebd_auto::{
    canonicalize_auto, imaginary_time_step_auto, imaginary_time_step_second_order_auto,
    ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::itebd_checkpoint::{ItebdCheckpointProgress, ItebdRunMetadata};
use thermal_imps_purification::itebd_complex::{ComplexLocalHamiltonian, ComplexPurifiedMps, ComplexSite};
use thermal_imps_purification::itebd_error::ItebdError;
use thermal_imps_purification::purified_mps::PurifiedMps;
use thermal_imps_purification::tensor::{Tensor, Truncation};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

pub fn real_tfim_metadata() -> ItebdRunMetadata {
    ItebdRunMetadata {
        dtau: 0.05,
        trotter_order: TrotterOrder::Second,
        truncation: TruncationCfg {
            epsilon: 1e-13,
            max_bond: Some(64),
        },
        canonicalize_every: 3,
        record_every_beta: None,
        model_label: Some("TFIM J=1 g=0.7".to_string()),
        git_revision: None,
        hermiticity_tolerance: 1e-12,
    }
}

pub fn phase_tfim() -> ItebdHamiltonian {
    let real = thermal_imps_purification::model::Tfim { j: 1.0, g: 0.7 }.local();
    let u = DMatrix::from_diagonal(&DVector::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ]));
    let u2 = u.kronecker(&u);
    let rotate =
        |matrix: &DMatrix<f64>| &u2 * matrix.map(|value| Complex64::new(value, 0.0)) * u2.adjoint();
    ItebdHamiltonian::Complex(
        ComplexLocalHamiltonian::try_new_with_tolerance(
            rotate(&real.two_site_h),
            rotate(&real.site_energy),
            1e-12,
        )
        .unwrap(),
    )
}

#[allow(dead_code)]
pub fn promote_real_state(state: &PurifiedMps) -> ComplexPurifiedMps {
    let promote_site = |site: &thermal_imps_purification::purified_mps::Site| ComplexSite {
        gamma: Tensor::from_dense(
            site.gamma.indices.clone(),
            site.gamma
                .to_vec::<f64>()
                .unwrap()
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect(),
        )
        .unwrap(),
        left: site.left.clone(),
        phys: site.phys.clone(),
        anc: site.anc.clone(),
        right: site.right.clone(),
    };
    ComplexPurifiedMps {
        a: promote_site(&state.a),
        b: promote_site(&state.b),
        lambda_ab: state.lambda_ab.clone(),
        lambda_bond_ab: state.lambda_bond_ab.clone(),
        lambda_ba: state.lambda_ba.clone(),
        lambda_bond_ba: state.lambda_bond_ba.clone(),
    }
}

#[allow(dead_code)]
pub fn advance(
    state: &mut ItebdState,
    h: &ItebdHamiltonian,
    metadata: &ItebdRunMetadata,
    progress: &mut ItebdCheckpointProgress,
    count: u64,
) -> Result<(), ItebdError> {
    let trunc = Truncation {
        epsilon: metadata.truncation.epsilon,
        max_bond: metadata.truncation.max_bond,
    };
    for _ in 0..count {
        let info = match metadata.trotter_order {
            TrotterOrder::First => imaginary_time_step_auto(state, h, metadata.dtau, &trunc)?,
            TrotterOrder::Second => {
                imaginary_time_step_second_order_auto(state, h, metadata.dtau, &trunc)?
            }
        };
        progress.accumulated_log_norm += info.log_norm;
        progress.completed_steps += 1;
        if progress.completed_steps % metadata.canonicalize_every as u64 == 0 {
            progress.accumulated_log_norm += canonicalize_auto(state, h)?;
        }
        progress.beta = 2.0 * progress.completed_steps as f64 * metadata.dtau;
        progress.last_step = Some(info);
    }
    Ok(())
}

#[allow(dead_code)]
pub fn rdm(
    state: &ItebdState,
    start: thermal_imps_purification::itebd_rdm::RdmParity,
    length: usize,
) -> DMatrix<Complex64> {
    use thermal_imps_purification::itebd_rdm::{reduced_density_matrix_auto, AutoRdmResult, RdmOptions};
    match reduced_density_matrix_auto(state, start, length, &RdmOptions::default()).unwrap() {
        AutoRdmResult::Real(r) => r.density_matrix.map(Complex64::from),
        AutoRdmResult::Complex(r) => r.density_matrix,
    }
}

#[allow(dead_code)]
pub fn assert_report_bits(
    a: &thermal_imps_purification::itebd_rdm::RdmReport,
    b: &thermal_imps_purification::itebd_rdm::RdmReport,
) {
    // Exhaustive destructuring makes newly added report fields require a comparison.
    use thermal_imps_purification::itebd_rdm::{RdmEnvironmentDiagnostics, RdmReport};
    let bits = |r: &RdmReport| {
        let RdmReport {
            start,
            length,
            left,
            right,
            raw_trace,
            trace_residual,
            hermiticity_residual,
            minimum_eigenvalue,
            largest_intermediate_elements,
        } = r;
        let mut bits = vec![
            u64::from(*start == thermal_imps_purification::itebd_rdm::RdmParity::B),
            *length as u64,
            raw_trace.re.to_bits(),
            raw_trace.im.to_bits(),
            trace_residual.to_bits(),
            hermiticity_residual.to_bits(),
            minimum_eigenvalue.to_bits(),
            *largest_intermediate_elements as u64,
        ];
        for d in [left, right] {
            let RdmEnvironmentDiagnostics {
                eigenvalue,
                iterations,
                relative_residual,
            } = d;
            bits.extend([
                eigenvalue.to_bits(),
                *iterations as u64,
                relative_residual.to_bits(),
            ]);
        }
        bits
    };
    assert_eq!(bits(a), bits(b));
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub struct StateSnapshot {
    complex: bool,
    gamma_bits: [Vec<u64>; 2],
    lambda_bits: [Vec<u64>; 2],
    roles_and_axes: Vec<(thermal_imps_purification::tensor::Idx, usize)>,
    progress_bits: Vec<u64>,
}

#[allow(dead_code)]
pub fn snapshot(state: &ItebdState, progress: &ItebdCheckpointProgress) -> StateSnapshot {
    use thermal_imps_purification::itebd_state_view::ItebdStateRef;
    let view = ItebdStateRef::from(state);
    let bits = |site: thermal_imps_purification::itebd_state_view::ItebdSiteRef<'_>| {
        if site.gamma.is_complex() {
            site.gamma
                .to_vec::<Complex64>()
                .unwrap()
                .iter()
                .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                .collect()
        } else {
            site.gamma
                .to_vec::<f64>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        }
    };
    let mut roles_and_axes = Vec::new();
    for site in [view.a(), view.b()] {
        roles_and_axes.extend(
            [site.left, site.physical, site.ancilla, site.right]
                .into_iter()
                .chain(site.gamma.indices.iter())
                .map(|i| (i.clone(), i.dim)),
        );
    }
    let bonds = match state {
        ItebdState::Real(s) => [&s.lambda_bond_ab, &s.lambda_bond_ba],
        ItebdState::Complex(s) => [&s.lambda_bond_ab, &s.lambda_bond_ba],
    };
    roles_and_axes.extend(bonds.into_iter().map(|i| (i.clone(), i.dim)));
    let ItebdCheckpointProgress {
        beta,
        completed_steps,
        accumulated_log_norm,
        last_step,
    } = progress;
    let mut progress_bits = vec![
        beta.to_bits(),
        *completed_steps,
        accumulated_log_norm.to_bits(),
        u64::from(last_step.is_some()),
    ];
    if let Some(thermal_imps_purification::itebd::StepInfo {
        max_bond,
        min_singular_value,
        log_norm,
    }) = last_step
    {
        progress_bits.extend([
            *max_bond as u64,
            min_singular_value.to_bits(),
            log_norm.to_bits(),
        ]);
    }
    StateSnapshot {
        complex: view.a().gamma.is_complex(),
        gamma_bits: [bits(view.a()), bits(view.b())],
        lambda_bits: [
            view.lambda_ab().iter().map(|v| v.to_bits()).collect(),
            view.lambda_ba().iter().map(|v| v.to_bits()).collect(),
        ],
        roles_and_axes,
        progress_bits,
    }
}
