use super::SolveRunError;
use crate::config::RunConfig;
use crate::itebd_auto::ItebdHamiltonian;
use crate::itebd_checkpoint::ItebdRunMetadata;
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::path::{Component, Path, PathBuf};

pub(super) fn paths(cfg: &RunConfig, config_path: Option<&Path>) -> Result<(), SolveRunError> {
    let checkpoint = cfg.checkpoint.as_ref().unwrap();
    let destinations = [Path::new(&cfg.output.path), Path::new(&checkpoint.path)];
    let mut resolved = Vec::new();
    for path in destinations {
        if path.as_os_str().is_empty() {
            return Err(SolveRunError::Configuration(
                "output paths must be nonempty".into(),
            ));
        }
        let full = resolve(path)?;
        match std::fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(SolveRunError::Configuration(format!(
                    "destination already exists: {}",
                    path.display()
                )))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(source) => {
                return Err(SolveRunError::Path {
                    path: path.into(),
                    source,
                })
            }
        }
        resolved.push(full);
    }
    let conflicts = |a: &Path, b: &Path| a.starts_with(b) || b.starts_with(a);
    if conflicts(&resolved[0], &resolved[1]) {
        return Err(SolveRunError::Configuration(
            "JSON and checkpoint destinations conflict".into(),
        ));
    }
    for path in config_path
        .into_iter()
        .chain(cfg.restart.as_ref().map(|r| Path::new(&r.path)))
    {
        if path.as_os_str().is_empty() {
            return Err(SolveRunError::Configuration(
                "input paths must be nonempty".into(),
            ));
        }
        let full = resolve(path)?;
        if resolved.iter().any(|dest| conflicts(dest, &full)) {
            return Err(SolveRunError::Configuration(format!(
                "output conflicts with input {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn resolve(path: &Path) -> Result<PathBuf, SolveRunError> {
    let mut result = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().map_err(|source| SolveRunError::Path {
            path: path.into(),
            source,
        })?
    };
    for component in path.components() {
        match component {
            Component::CurDir => (),
            Component::ParentDir => {
                result.pop();
            }
            other => {
                result.push(other.as_os_str());
                match std::fs::symlink_metadata(&result) {
                    Ok(_) => {
                        result = std::fs::canonicalize(&result).map_err(|source| {
                            SolveRunError::Path {
                                path: path.into(),
                                source,
                            }
                        })?
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                    Err(source) => {
                        return Err(SolveRunError::Path {
                            path: path.into(),
                            source,
                        })
                    }
                }
            }
        }
    }
    Ok(result)
}

pub(super) fn compatible(
    cfg: &RunConfig,
    expected: &ItebdHamiltonian,
    loaded: &ItebdHamiltonian,
    metadata: &ItebdRunMetadata,
) -> Result<(), SolveRunError> {
    for (field, matches) in [
        (
            "Hamiltonian.two_site_h",
            same_matrix_values(two_site_h(expected), two_site_h(loaded)),
        ),
        (
            "Hamiltonian.site_energy",
            same_matrix_values(site_energy(expected), site_energy(loaded)),
        ),
        ("dtau", metadata.dtau == cfg.evolution.dtau),
        (
            "trotter_order",
            metadata.trotter_order == cfg.evolution.trotter_order,
        ),
        (
            "truncation.epsilon",
            metadata.truncation.epsilon == cfg.truncation.epsilon,
        ),
        (
            "truncation.max_bond",
            metadata.truncation.max_bond == cfg.truncation.max_bond,
        ),
        (
            "canonicalize_every",
            metadata.canonicalize_every == cfg.run.canonicalize_every,
        ),
        (
            "record_every_beta",
            metadata
                .record_every_beta
                .is_none_or(|v| v == cfg.evolution.record_every_beta),
        ),
    ] {
        if !matches {
            return Err(SolveRunError::RestartMismatch { field });
        }
    }
    Ok(())
}

enum MatrixRef<'a> {
    Real(&'a DMatrix<f64>),
    Complex(&'a DMatrix<Complex64>),
}

fn two_site_h(hamiltonian: &ItebdHamiltonian) -> MatrixRef<'_> {
    match hamiltonian {
        ItebdHamiltonian::Real(hamiltonian) => MatrixRef::Real(&hamiltonian.two_site_h),
        ItebdHamiltonian::Complex(hamiltonian) => MatrixRef::Complex(hamiltonian.two_site_h()),
    }
}

fn site_energy(hamiltonian: &ItebdHamiltonian) -> MatrixRef<'_> {
    match hamiltonian {
        ItebdHamiltonian::Real(hamiltonian) => MatrixRef::Real(&hamiltonian.site_energy),
        ItebdHamiltonian::Complex(hamiltonian) => MatrixRef::Complex(hamiltonian.site_energy()),
    }
}

fn same_matrix_values(a: MatrixRef<'_>, b: MatrixRef<'_>) -> bool {
    match (a, b) {
        (MatrixRef::Real(a), MatrixRef::Real(b)) => {
            a.shape() == b.shape()
                && a.iter()
                    .zip(b.iter())
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a == b)
        }
        (MatrixRef::Real(real), MatrixRef::Complex(complex))
        | (MatrixRef::Complex(complex), MatrixRef::Real(real)) => {
            real.shape() == complex.shape()
                && real.iter().zip(complex.iter()).all(|(real, complex)| {
                    real.is_finite()
                        && complex.re.is_finite()
                        && complex.im.is_finite()
                        && *real == complex.re
                        && complex.im == 0.0
                })
        }
        (MatrixRef::Complex(a), MatrixRef::Complex(b)) => {
            a.shape() == b.shape()
                && a.iter().zip(b.iter()).all(|(a, b)| {
                    a.re.is_finite()
                        && a.im.is_finite()
                        && b.re.is_finite()
                        && b.im.is_finite()
                        && a == b
                })
        }
    }
}
