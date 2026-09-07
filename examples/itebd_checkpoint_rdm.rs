//! Run with a new destination path; existing files are never replaced.
use thermal_imps_purification::config::{TrotterOrder, TruncationCfg};
use thermal_imps_purification::itebd::free_energy_from_log_norm;
use thermal_imps_purification::itebd_auto::{
    canonicalize_auto, energy_density_auto, imaginary_time_step_auto,
    imaginary_time_step_second_order_auto, ItebdHamiltonian, ItebdState,
};
use thermal_imps_purification::itebd_checkpoint::{
    load_itebd_checkpoint, ItebdCheckpointLoadOptions, ItebdCheckpointProgress, ItebdRunMetadata,
    ItebdTrajectoryWriter,
};
use thermal_imps_purification::itebd_error::ItebdError;
use thermal_imps_purification::itebd_rdm::{
    reduced_density_matrix_auto, AutoRdmResult, RdmOptions, RdmParity,
};
use thermal_imps_purification::tensor::Truncation;
use std::path::PathBuf;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: itebd_checkpoint_rdm NEW_OUTPUT.h5")?,
    );
    let h = ItebdHamiltonian::Real(thermal_imps_purification::model::Tfim { j: 1.0, g: 0.7 }.local());
    let metadata = ItebdRunMetadata {
        dtau: 0.05,
        trotter_order: TrotterOrder::Second,
        truncation: TruncationCfg {
            epsilon: 1e-13,
            max_bond: Some(64),
        },
        canonicalize_every: 3,
        record_every_beta: None,
        model_label: Some("TFIM J=1 g=0.7".into()),
        git_revision: None,
        hermiticity_tolerance: 1e-12,
    };
    let mut state = ItebdState::infinite_temperature(&h)?;
    let mut progress = ItebdCheckpointProgress {
        beta: 0.0,
        completed_steps: 0,
        accumulated_log_norm: 0.0,
        last_step: None,
    };
    let mut writer = ItebdTrajectoryWriter::create(&path, metadata.clone(), &h)?;
    writer.append((&state).into(), &progress)?;
    advance(&mut state, &h, &metadata, &mut progress, 5)?;
    writer.append((&state).into(), &progress)?;
    writer.finish()?;
    let mut loaded = load_itebd_checkpoint(&path, 1, &ItebdCheckpointLoadOptions::default())?;
    advance(
        &mut loaded.state,
        &loaded.hamiltonian,
        &loaded.metadata,
        &mut loaded.progress,
        5,
    )?;
    println!(
        "beta={} energy={:.15} free_energy={:.15}",
        loaded.progress.beta,
        energy_density_auto(&loaded.state, &loaded.hamiltonian)?,
        free_energy_from_log_norm(
            loaded.progress.accumulated_log_norm / 2.0,
            loaded.progress.beta,
            2
        )
    );
    for start in [RdmParity::A, RdmParity::B] {
        let report =
            match reduced_density_matrix_auto(&loaded.state, start, 2, &RdmOptions::default())? {
                AutoRdmResult::Real(r) => r.report,
                AutoRdmResult::Complex(r) => r.report,
            };
        println!(
            "{start:?} length=2 trace_residual={:e} minimum_eigenvalue={:e} report={report:?}",
            report.trace_residual, report.minimum_eigenvalue
        );
    }
    Ok(())
}
