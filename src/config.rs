//! Run configuration: TOML in, validated and resolved run parameters out.
//!
//! Input parsing goes through a flat `RawModel` to avoid `toml`'s internally-tagged-enum
//! quirks; `ModelSpec` itself is (de)serialized only over JSON (self-describing) in `runner`.

mod matrix;

pub use matrix::{MatrixInput, MatrixModelInput, ObservableInput, ResolvedModel};

use std::collections::BTreeMap;

use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::exact;
use crate::itebd_auto::ItebdHamiltonian;
use crate::itebd_complex::validate_local_operator;
use crate::itebd_error::ItebdError;
use crate::model::{
    pauli_x, pauli_z, sz1, AkltProjector, BilinearBiquadratic, LocalHamiltonian, Tfim, Xy,
};

/// Quadrature points for exact free-fermion references (matches the prior demo binaries).
const EXACT_NK: usize = 4000;

#[derive(Debug, Clone, Serialize)]
pub struct RunConfig {
    pub model: ModelSpec,
    pub evolution: Evolution,
    pub truncation: TruncationCfg,
    pub run: RunOpts,
    pub output: OutputCfg,
    pub checkpoint: Option<CheckpointCfg>,
    pub restart: Option<RestartCfg>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCfg {
    pub path: String,
    pub every_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestartCfg {
    pub path: String,
    #[serde(default)]
    pub snapshot: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSpec {
    Tfim { j: f64, g: f64 },
    Xy { gamma: f64, h: f64 },
    BilinearBiquadratic { j1: f64, j2: f64 },
    Aklt,
    AkltProjector,
    Heisenberg,
    Matrix(MatrixModelInput),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrotterOrder {
    First,
    Second,
}

impl Default for TrotterOrder {
    fn default() -> Self {
        Self::Second
    }
}

impl Serialize for TrotterOrder {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(match self {
            Self::First => 1,
            Self::Second => 2,
        })
    }
}

impl<'de> Deserialize<'de> for TrotterOrder {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match u8::deserialize(deserializer)? {
            1 => Ok(Self::First),
            2 => Ok(Self::Second),
            _ => Err(serde::de::Error::custom(
                "evolution.trotter_order must be 1 or 2",
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evolution {
    pub dtau: f64,
    pub beta_max: f64,
    pub record_every_beta: f64,
    #[serde(default)]
    pub trotter_order: TrotterOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationCfg {
    pub epsilon: f64,
    #[serde(default)]
    pub max_bond: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOpts {
    #[serde(default = "default_canonicalize_every")]
    pub canonicalize_every: usize,
}
impl Default for RunOpts {
    fn default() -> Self {
        Self {
            canonicalize_every: default_canonicalize_every(),
        }
    }
}
fn default_canonicalize_every() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputCfg {
    pub path: String,
    #[serde(default)]
    pub include_exact: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactRefs {
    pub u: f64,
    pub c: f64,
    pub f: f64,
    pub magnetization: f64,
}

// --- Input layer: flat TOML shapes converted to the resolved config above. ---

#[derive(Debug, Deserialize)]
struct RawConfig {
    model: RawModel,
    evolution: Evolution,
    truncation: TruncationCfg,
    #[serde(default)]
    run: RunOpts,
    output: OutputCfg,
    checkpoint: Option<CheckpointCfg>,
    restart: Option<RestartCfg>,
}

#[derive(Debug, Deserialize)]
struct RawModel {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    j: RawPresetParameter,
    #[serde(default)]
    g: RawPresetParameter,
    #[serde(default)]
    gamma: RawPresetParameter,
    #[serde(default)]
    h: RawPresetParameter,
    #[serde(default)]
    j1: RawPresetParameter,
    #[serde(default)]
    j2: RawPresetParameter,
    version: Option<u32>,
    local_dim: Option<usize>,
    basis_order: Option<String>,
    label: Option<String>,
    two_site_h: Option<MatrixInput>,
    site_energy: Option<MatrixInput>,
    observable: Option<ObservableInput>,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde::de::IgnoredAny>,
}

#[derive(Debug)]
struct RawPresetParameter {
    value: Option<f64>,
    supplied: bool,
}

impl Default for RawPresetParameter {
    fn default() -> Self {
        Self {
            value: None,
            supplied: false,
        }
    }
}

impl<'de> Deserialize<'de> for RawPresetParameter {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            value: Option::<f64>::deserialize(deserializer)?,
            supplied: true,
        })
    }
}

impl RawModel {
    fn resolve(&self) -> Result<ModelSpec, String> {
        let need = |o: Option<f64>, name: &str| {
            o.ok_or_else(|| format!("model.{name} is required for type \"{}\"", self.kind))
        };
        Ok(match self.kind.as_str() {
            "tfim" => ModelSpec::Tfim {
                j: need(self.j.value, "j")?,
                g: need(self.g.value, "g")?,
            },
            "xy" => ModelSpec::Xy {
                gamma: need(self.gamma.value, "gamma")?,
                h: need(self.h.value, "h")?,
            },
            "bilinear_biquadratic" => {
                ModelSpec::BilinearBiquadratic {
                    j1: need(self.j1.value, "j1")?,
                    j2: need(self.j2.value, "j2")?,
                }
            }
            "aklt" => ModelSpec::Aklt,
            "aklt_projector" => ModelSpec::AkltProjector,
            "heisenberg" => ModelSpec::Heisenberg,
            "matrix" => {
                for (name, present) in [
                    ("j", self.j.supplied),
                    ("g", self.g.supplied),
                    ("gamma", self.gamma.supplied),
                    ("h", self.h.supplied),
                    ("j1", self.j1.supplied),
                    ("j2", self.j2.supplied),
                ] {
                    if present {
                        return Err(format!(
                            "model.{name} is a preset-only key and is invalid for type \"matrix\""
                        ));
                    }
                }
                if let Some(name) = self.unknown.keys().next() {
                    return Err(format!("unknown matrix-model key model.{name}"));
                }
                let required = |present: bool, name: &str| {
                    present
                        .then_some(())
                        .ok_or_else(|| format!("model.{name} is required for type \"matrix\""))
                };
                required(self.version.is_some(), "version")?;
                required(self.local_dim.is_some(), "local_dim")?;
                required(self.basis_order.is_some(), "basis_order")?;
                required(self.two_site_h.is_some(), "two_site_h")?;
                required(self.site_energy.is_some(), "site_energy")?;
                required(self.observable.is_some(), "observable")?;
                ModelSpec::Matrix(MatrixModelInput {
                    version: self.version.unwrap(),
                    local_dim: self.local_dim.unwrap(),
                    basis_order: self.basis_order.clone().unwrap(),
                    label: self.label.clone(),
                    two_site_h: self.two_site_h.clone().unwrap(),
                    site_energy: self.site_energy.clone().unwrap(),
                    observable: self.observable.clone().unwrap(),
                })
            }
            other => {
                return Err(format!(
                    "unknown model type \"{other}\" (expected tfim, xy, bilinear_biquadratic, aklt, aklt_projector, heisenberg, matrix)"
                ))
            }
        })
    }
}

impl RunConfig {
    /// Parse TOML text into a validated, resolved config.
    pub fn from_toml_str(text: &str) -> Result<RunConfig, String> {
        let raw: RawConfig = toml::from_str(text).map_err(|e| format!("TOML parse error: {e}"))?;
        Self::from_raw(raw)
    }

    /// Parse JSON text into a validated, resolved config.
    pub fn from_json_str(text: &str) -> Result<RunConfig, String> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|e| format!("JSON parse error: {e}"))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawConfig) -> Result<RunConfig, String> {
        let cfg = RunConfig {
            model: raw.model.resolve()?,
            evolution: raw.evolution,
            truncation: raw.truncation,
            run: raw.run,
            output: raw.output,
            checkpoint: raw.checkpoint,
            restart: raw.restart,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let e = &self.evolution;
        if !e.dtau.is_finite() || !(e.dtau > 0.0) {
            return Err("evolution.dtau must be finite and > 0".into());
        }
        if !e.beta_max.is_finite() || !(e.beta_max > 0.0) {
            return Err("evolution.beta_max must be finite and > 0".into());
        }
        if !e.record_every_beta.is_finite() || !(e.record_every_beta > 0.0) {
            return Err("evolution.record_every_beta must be finite and > 0".into());
        }
        if !self.truncation.epsilon.is_finite() || !(self.truncation.epsilon > 0.0) {
            return Err("truncation.epsilon must be finite and > 0".into());
        }
        if matches!(self.truncation.max_bond, Some(0)) {
            return Err("truncation.max_bond must be >= 1".into());
        }
        if self.run.canonicalize_every < 1 {
            return Err("run.canonicalize_every must be >= 1".into());
        }
        if matches!(self.checkpoint, Some(CheckpointCfg { every_steps: 0, .. })) {
            return Err("checkpoint.every_steps must be >= 1".into());
        }
        if self.restart.is_some() && self.checkpoint.is_none() {
            return Err("restart requires a checkpoint destination".into());
        }
        if let Some(checkpoint) = &self.checkpoint {
            if self.output.path.trim().is_empty() {
                return Err("output.path must not be blank in checkpoint mode".into());
            }
            if checkpoint.path.trim().is_empty() {
                return Err("checkpoint.path must not be blank".into());
            }
            if self
                .restart
                .as_ref()
                .is_some_and(|restart| restart.path.trim().is_empty())
            {
                return Err("restart.path must not be blank".into());
            }
        }
        let finite = match &self.model {
            ModelSpec::Tfim { j, g } => j.is_finite() && g.is_finite(),
            ModelSpec::Xy { gamma, h } => gamma.is_finite() && h.is_finite(),
            ModelSpec::BilinearBiquadratic { j1, j2 } => j1.is_finite() && j2.is_finite(),
            ModelSpec::Aklt
            | ModelSpec::AkltProjector
            | ModelSpec::Heisenberg
            | ModelSpec::Matrix(_) => true,
        };
        if !finite {
            return Err("model parameters must be finite".into());
        }
        if matches!(&self.model, ModelSpec::Matrix(_)) {
            self.model.resolve().map_err(|error| error.to_string())?;
        }
        self.schedule()?;
        if self.output.include_exact && !self.model.has_exact() {
            return Err(
                "output.include_exact = true is only valid for models with a closed-form reference (tfim, xy)"
                    .into(),
            );
        }
        Ok(())
    }

    pub(crate) fn schedule(&self) -> Result<(u64, u64), String> {
        const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;
        let denominator = 2.0 * self.evolution.dtau;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("step beta denominator must be finite and > 0".into());
        }
        let target_ratio = self.evolution.beta_max / denominator;
        let stride_ratio = self.evolution.record_every_beta / denominator;
        let target = target_ratio.round();
        let stride = stride_ratio.round().max(1.0);
        for (name, ratio, rounded) in [
            ("target steps", target_ratio, target),
            ("observation stride", stride_ratio, stride),
        ] {
            if !ratio.is_finite()
                || !rounded.is_finite()
                || ratio >= MAX_EXACT_INTEGER
                || rounded >= MAX_EXACT_INTEGER
            {
                return Err(format!("{name} must be finite and below 2^53"));
            }
            if rounded > usize::MAX as f64 || rounded > u64::MAX as f64 {
                return Err(format!("{name} exceeds the platform step bound"));
            }
        }
        let final_beta = 2.0 * target * self.evolution.dtau;
        if !final_beta.is_finite() {
            return Err("scheduled final beta must be finite".into());
        }
        Ok((target as u64, stride as u64))
    }
}

impl ModelSpec {
    fn builtin_hamiltonian(&self) -> Option<LocalHamiltonian> {
        match self {
            ModelSpec::Tfim { j, g } => Some(Tfim { j: *j, g: *g }.local()),
            ModelSpec::Xy { gamma, h } => Some(
                Xy {
                    gamma: *gamma,
                    h: *h,
                }
                .local(),
            ),
            ModelSpec::BilinearBiquadratic { j1, j2 } => {
                Some(BilinearBiquadratic { j1: *j1, j2: *j2 }.local())
            }
            ModelSpec::Aklt => Some(BilinearBiquadratic::aklt().local()),
            ModelSpec::AkltProjector => Some(AkltProjector.local()),
            ModelSpec::Heisenberg => Some(BilinearBiquadratic::heisenberg().local()),
            ModelSpec::Matrix(_) => None,
        }
    }

    fn builtin_magnetization_op(&self) -> Option<DMatrix<f64>> {
        match self {
            ModelSpec::Tfim { .. } => Some(pauli_x()),
            ModelSpec::Xy { .. } => Some(pauli_z()),
            ModelSpec::BilinearBiquadratic { .. }
            | ModelSpec::Aklt
            | ModelSpec::AkltProjector
            | ModelSpec::Heisenberg => Some(sz1()),
            ModelSpec::Matrix(_) => None,
        }
    }

    pub fn resolve(&self) -> Result<ResolvedModel, ItebdError> {
        match self {
            ModelSpec::Matrix(matrix) => matrix.resolve(),
            _ => {
                let local = self.builtin_hamiltonian().unwrap();
                let hamiltonian = ItebdHamiltonian::try_from_complex(
                    local.two_site_h.map(|value| Complex64::new(value, 0.0)),
                    local.site_energy.map(|value| Complex64::new(value, 0.0)),
                )?;
                let observable = self
                    .builtin_magnetization_op()
                    .unwrap()
                    .map(|value| Complex64::new(value, 0.0));
                validate_local_operator(&observable, hamiltonian.dim(), 1e-12)?;
                Ok(ResolvedModel {
                    hamiltonian,
                    observable,
                    observable_name: "magnetization".into(),
                })
            }
        }
    }

    pub fn hamiltonian(&self) -> Result<LocalHamiltonian, ItebdError> {
        if let ModelSpec::Matrix(matrix) = self {
            return matrix.real_hamiltonian();
        }
        let local = self.builtin_hamiltonian().unwrap();
        match ItebdHamiltonian::try_from_complex(
            local.two_site_h.map(|value| Complex64::new(value, 0.0)),
            local.site_energy.map(|value| Complex64::new(value, 0.0)),
        )? {
            ItebdHamiltonian::Real(hamiltonian) => Ok(hamiltonian),
            ItebdHamiltonian::Complex(_) => unreachable!("built-in Hamiltonians are real"),
        }
    }

    /// Single-site operator whose expectation is reported as `magnetization`:
    /// TFIM → σx (transverse field), XY → σz, spin-1 → Sz = diag(1,0,-1).
    pub fn magnetization_op(&self) -> Result<DMatrix<f64>, ItebdError> {
        if let Some(operator) = self.builtin_magnetization_op() {
            return Ok(operator);
        }
        match self {
            ModelSpec::Matrix(matrix) => matrix.real_observable(),
            _ => unreachable!(),
        }
    }

    pub fn has_exact(&self) -> bool {
        matches!(self, ModelSpec::Tfim { .. } | ModelSpec::Xy { .. })
    }

    /// Closed-form references at inverse temperature `beta`, when available.
    pub fn exact(&self, beta: f64) -> Option<ExactRefs> {
        match self {
            ModelSpec::Tfim { j, g } => Some(ExactRefs {
                u: exact::exact_energy_density(*j, *g, beta, EXACT_NK),
                c: exact::exact_specific_heat(*j, *g, beta, EXACT_NK),
                f: exact::free_energy_density(*j, *g, beta, EXACT_NK),
                magnetization: exact::exact_magnetization_x(*j, *g, beta, EXACT_NK),
            }),
            ModelSpec::Xy { gamma, h } => Some(ExactRefs {
                u: exact::xy_energy_density(*gamma, *h, beta, EXACT_NK),
                c: exact::xy_specific_heat(*gamma, *h, beta, EXACT_NK),
                f: exact::xy_free_energy_density(*gamma, *h, beta, EXACT_NK),
                magnetization: exact::xy_magnetization_z(*gamma, *h, beta, EXACT_NK),
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{pauli_x, pauli_z, sz1};

    fn cfg_toml(model_block: &str, include_exact: bool) -> String {
        format!(
            "{model_block}\n\
             [evolution]\n\
             dtau = 0.01\n\
             beta_max = 1.0\n\
             record_every_beta = 0.5\n\
             [truncation]\n\
             epsilon = 1e-12\n\
             max_bond = 16\n\
             [output]\n\
             path = \"out.json\"\n\
             include_exact = {include_exact}\n"
        )
    }

    #[test]
    fn input_trotter_order_defaults_to_second() {
        let cfg = RunConfig::from_toml_str(&cfg_toml(
            "[model]\ntype = \"tfim\"\nj = 1.0\ng = 0.7",
            false,
        ))
        .unwrap();
        assert_eq!(cfg.evolution.trotter_order, TrotterOrder::Second);
    }

    #[test]
    fn parses_explicit_first_order() {
        let text = cfg_toml("[model]\ntype = \"aklt\"", false)
            .replace("dtau = 0.01", "dtau = 0.01\ntrotter_order = 1");
        let evolution = RunConfig::from_toml_str(&text).unwrap().evolution;
        assert_eq!(evolution.trotter_order, TrotterOrder::First);
    }

    #[test]
    fn parses_and_serializes_second_order_as_integer() {
        let text = cfg_toml("[model]\ntype = \"aklt\"", false)
            .replace("dtau = 0.01", "dtau = 0.01\ntrotter_order = 2");
        let evolution = RunConfig::from_toml_str(&text).unwrap().evolution;
        assert_eq!(evolution.trotter_order, TrotterOrder::Second);
        assert_eq!(serde_json::to_value(evolution).unwrap()["trotter_order"], 2);
    }

    #[test]
    fn rejects_unsupported_trotter_order() {
        let text = cfg_toml("[model]\ntype = \"aklt\"", false)
            .replace("dtau = 0.01", "dtau = 0.01\ntrotter_order = 3");
        let error = RunConfig::from_toml_str(&text).unwrap_err();
        assert!(
            error.contains("evolution.trotter_order must be 1 or 2"),
            "{error}"
        );
    }

    #[test]
    fn parses_tfim_with_params() {
        let cfg = RunConfig::from_toml_str(&cfg_toml(
            "[model]\ntype = \"tfim\"\nj = 1.0\ng = 0.7",
            false,
        ))
        .unwrap();
        match cfg.model {
            ModelSpec::Tfim { j, g } => {
                assert_eq!(j, 1.0);
                assert_eq!(g, 0.7);
            }
            _ => panic!("wrong variant"),
        }
        assert_eq!(cfg.run.canonicalize_every, 1, "default canonicalize_every");
        assert_eq!(cfg.model.hamiltonian().unwrap().dim(), 2);
        assert!(cfg.model.has_exact());
    }

    #[test]
    fn parses_aklt_preset() {
        let cfg = RunConfig::from_toml_str(&cfg_toml("[model]\ntype = \"aklt\"", false)).unwrap();
        assert!(matches!(cfg.model, ModelSpec::Aklt));
        assert_eq!(cfg.model.hamiltonian().unwrap().dim(), 3);
        assert!(!cfg.model.has_exact());
        assert!(cfg.model.exact(1.0).is_none());
    }

    #[test]
    fn magnetization_op_is_model_appropriate() {
        assert_eq!(
            ModelSpec::Tfim { j: 1.0, g: 0.5 }
                .magnetization_op()
                .unwrap(),
            pauli_x()
        );
        assert_eq!(
            ModelSpec::Xy { gamma: 0.5, h: 0.5 }
                .magnetization_op()
                .unwrap(),
            pauli_z()
        );
        assert_eq!(ModelSpec::Aklt.magnetization_op().unwrap(), sz1());
    }

    #[test]
    fn xy_exact_matches_exact_module() {
        let m = ModelSpec::Xy { gamma: 0.5, h: 0.5 };
        let r = m.exact(0.8).unwrap();
        assert_eq!(r.u, crate::exact::xy_energy_density(0.5, 0.5, 0.8, 4000));
        assert_eq!(
            r.magnetization,
            crate::exact::xy_magnetization_z(0.5, 0.5, 0.8, 4000)
        );
    }

    #[test]
    fn missing_required_param_is_error() {
        // tfim without g
        assert!(
            RunConfig::from_toml_str(&cfg_toml("[model]\ntype = \"tfim\"\nj = 1.0", false))
                .is_err()
        );
    }

    #[test]
    fn unknown_model_is_error() {
        assert!(RunConfig::from_toml_str(&cfg_toml("[model]\ntype = \"ising3d\"", false)).is_err());
    }

    #[test]
    fn rejects_include_exact_without_reference() {
        assert!(RunConfig::from_toml_str(&cfg_toml("[model]\ntype = \"aklt\"", true)).is_err());
    }

    #[test]
    fn rejects_nonpositive_dtau() {
        let toml = "[model]\ntype = \"aklt\"\n[evolution]\ndtau = 0.0\nbeta_max = 1.0\nrecord_every_beta = 0.5\n[truncation]\nepsilon = 1e-12\n[output]\npath = \"out.json\"\n";
        assert!(RunConfig::from_toml_str(toml).is_err());
    }

    #[test]
    fn rejects_zero_max_bond() {
        let toml = "[model]\ntype = \"aklt\"\n[evolution]\ndtau = 0.01\nbeta_max = 1.0\nrecord_every_beta = 0.5\n[truncation]\nepsilon = 1e-12\nmax_bond = 0\n[output]\npath = \"out.json\"\n";
        assert!(RunConfig::from_toml_str(toml).is_err());
    }

    #[test]
    fn schedule_preserves_legacy_rounding_and_stride_floor() {
        let text = cfg_toml("[model]\ntype = \"aklt\"", false)
            .replace("beta_max = 1.0", "beta_max = 0.35")
            .replace("record_every_beta = 0.5", "record_every_beta = 0.001")
            .replace("dtau = 0.01", "dtau = 0.05");
        assert_eq!(
            RunConfig::from_toml_str(&text).unwrap().schedule().unwrap(),
            (3, 1)
        );
    }

    #[test]
    fn tfim_exact_matches_exact_module() {
        let m = ModelSpec::Tfim { j: 1.0, g: 0.7 };
        let r = m.exact(0.8).unwrap();
        assert_eq!(r.u, crate::exact::exact_energy_density(1.0, 0.7, 0.8, 4000));
        assert_eq!(
            r.magnetization,
            crate::exact::exact_magnetization_x(1.0, 0.7, 0.8, 4000)
        );
    }

    #[test]
    fn magnetization_op_spin1_variants() {
        assert_eq!(ModelSpec::Heisenberg.magnetization_op().unwrap(), sz1());
        assert_eq!(
            ModelSpec::BilinearBiquadratic { j1: 1.0, j2: 0.5 }
                .magnetization_op()
                .unwrap(),
            sz1()
        );
    }
}
