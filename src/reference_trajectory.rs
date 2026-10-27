//! Shared fixed-target reference evolution and cooperative stop evidence.
use crate::{
    itebd::StepInfo,
    itebd_auto::{ItebdHamiltonian, ItebdState},
    itebd_checkpoint::{ItebdCheckpointProgress, ItebdStepSegment},
    tensor::Truncation,
};
use anyhow::{ensure, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceTarget {
    pub temperature_tick: u32,
    pub temperature: f64,
    pub beta: f64,
    pub completed_steps: u64,
}

pub(crate) fn nested_schedule(split: u64) -> Result<(Vec<ItebdStepSegment>, Vec<ReferenceTarget>)> {
    ensure!(
        matches!(split, 1 | 2),
        "nested schedule split must be 1 or 2"
    );
    let mut segments = Vec::with_capacity(20);
    let mut targets = Vec::with_capacity(20);
    let mut completed_steps = 0;
    for j in (1_u32..=20).rev() {
        let denominator = u64::from(j) * u64::from(j + 1);
        let coarse_steps = if j == 20 {
            25
        } else {
            500_u64.div_ceil(denominator)
        };
        let delta_beta = if j == 20 {
            0.5
        } else {
            10.0 / denominator as f64
        };
        let steps = coarse_steps * split;
        segments.push(ItebdStepSegment {
            steps,
            dtau: delta_beta / (2.0 * steps as f64),
        });
        completed_steps += steps;
        targets.push(ReferenceTarget {
            temperature_tick: j,
            temperature: j as f64 / 10.0,
            beta: 10.0 / j as f64,
            completed_steps,
        });
    }
    Ok((segments, targets))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapCoverage {
    CompleteThroughStop,
    UnobservedTail,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapEvidence {
    pub maximum_dimension_seen: Option<usize>,
    pub ever_reached: Option<bool>,
    pub observed_strang_steps: u64,
    pub coverage: CapCoverage,
}
impl CapEvidence {
    fn empty() -> Self {
        Self {
            maximum_dimension_seen: None,
            ever_reached: None,
            observed_strang_steps: 0,
            coverage: CapCoverage::UnobservedTail,
        }
    }
    fn observe(&mut self, dimension: usize, limit: usize) {
        self.maximum_dimension_seen = Some(self.maximum_dimension_seen.unwrap_or(0).max(dimension));
        self.ever_reached = Some(self.ever_reached == Some(true) || dimension >= limit);
    }
    pub(crate) fn mark_unobserved(&mut self) {
        self.coverage = CapCoverage::UnobservedTail;
        if self.ever_reached != Some(true) {
            self.ever_reached = None;
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) enum RunPolicy {
    IsingV1,
    Aklt {
        max_seconds: Option<f64>,
        bond_cap: usize,
    },
}
pub(crate) struct TrajectorySettings<'a> {
    pub hamiltonian: &'a ItebdHamiltonian,
    pub truncation: &'a Truncation,
    pub segments: &'a [ItebdStepSegment],
    pub targets: &'a [ReferenceTarget],
    pub policy: RunPolicy,
}
pub(crate) trait Measurement: Clone {
    fn validate(&self) -> Result<()>;
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledSample<V> {
    pub temperature_tick: u32,
    pub temperature: f64,
    pub beta_target: f64,
    pub beta_actual: f64,
    pub completed_steps: u64,
    pub available: bool,
    pub reason: Option<String>,
    pub accumulated_log_norm: f64,
    #[serde(flatten)]
    pub values: Option<V>,
    pub cap: Option<CapEvidence>,
}
pub struct ScheduledOutcome<V> {
    pub samples: Vec<ScheduledSample<V>>,
    pub terminal_state: Option<ItebdState>,
    pub progress: ItebdCheckpointProgress,
    pub status: String,
    pub reason: String,
    pub elapsed_seconds: f64,
    pub budget_overrun_seconds: f64,
    pub cap: Option<CapEvidence>,
}
pub(crate) trait Engine<V: Measurement> {
    fn setup(&mut self, h: &ItebdHamiltonian) -> Result<ItebdState> {
        Ok(ItebdState::infinite_temperature(h)?)
    }
    fn step(
        &mut self,
        s: &mut ItebdState,
        h: &ItebdHamiltonian,
        dt: f64,
        t: &Truncation,
    ) -> Result<StepInfo>;
    fn canonicalize(&mut self, s: &mut ItebdState, h: &ItebdHamiltonian) -> Result<f64>;
    fn measure(&mut self, s: &ItebdState, h: &ItebdHamiltonian, b: f64, n: f64) -> Result<V>;
}
pub(crate) struct NativeEngine<V>(pub fn(&ItebdState, &ItebdHamiltonian, f64, f64) -> Result<V>);
impl<V: Measurement> Engine<V> for NativeEngine<V> {
    fn step(
        &mut self,
        s: &mut ItebdState,
        h: &ItebdHamiltonian,
        dt: f64,
        t: &Truncation,
    ) -> Result<StepInfo> {
        crate::itebd_auto::imaginary_time_step_second_order_auto(s, h, dt, t)
            .context("Strang evolution step")
    }
    fn canonicalize(&mut self, s: &mut ItebdState, h: &ItebdHamiltonian) -> Result<f64> {
        crate::itebd_auto::canonicalize_auto(s, h)
            .context("canonicalization after complete Strang step")
    }
    fn measure(&mut self, s: &ItebdState, h: &ItebdHamiltonian, b: f64, n: f64) -> Result<V> {
        (self.0)(s, h, b, n)
    }
}
fn stop(reason: String, stopped: &mut Option<String>, reasons: &mut Vec<String>) {
    reasons.push(reason.clone());
    *stopped = Some(reason);
}
pub(crate) fn run_scheduled<V: Measurement>(
    settings: TrajectorySettings<'_>,
    callback: &mut dyn FnMut(&ScheduledSample<V>) -> Result<()>,
    engine: &mut dyn Engine<V>,
    elapsed: &mut dyn FnMut() -> f64,
) -> Result<ScheduledOutcome<V>> {
    ensure!(
        settings.segments.len() == settings.targets.len() && !settings.targets.is_empty(),
        "reference schedule length mismatch"
    );
    let (limit, bond_cap) = match settings.policy {
        RunPolicy::IsingV1 => (None, None),
        RunPolicy::Aklt {
            max_seconds,
            bond_cap,
        } => {
            ensure!(
                max_seconds.is_none_or(|v| v.is_finite() && v > 0.) && bond_cap > 0,
                "invalid reference run policy"
            );
            (max_seconds, Some(bond_cap))
        }
    };
    let expired = |clock: &mut dyn FnMut() -> f64| limit.is_some_and(|v| clock() >= v);
    let mut cap = bond_cap.map(|_| CapEvidence::empty());
    let mut progress = ItebdCheckpointProgress {
        beta: 0.,
        completed_steps: 0,
        accumulated_log_norm: 0.,
        last_step: None,
    };
    let mut stopped = None;
    let mut reasons = Vec::new();
    let mut state = if expired(elapsed) {
        stop(
            "time limit before initial setup".into(),
            &mut stopped,
            &mut reasons,
        );
        None
    } else {
        let state = engine
            .setup(settings.hamiltonian)
            .context("reference initial state setup")?;
        let summary = crate::itebd_state_view::ItebdStateRef::from(&state)
            .validate()
            .context("reference initial state validation")?;
        if let (Some(c), Some(b)) = (&mut cap, bond_cap) {
            for d in summary.bond_dimensions {
                c.observe(d, b);
            }
            c.coverage = CapCoverage::CompleteThroughStop;
        }
        if expired(elapsed) {
            stop(
                "time limit after initial setup".into(),
                &mut stopped,
                &mut reasons,
            );
        }
        Some(state)
    };
    let mut endpoint_valid = state.is_some();
    let mut samples = Vec::with_capacity(settings.targets.len());
    for (segment, target) in settings.segments.iter().zip(settings.targets) {
        if stopped.is_none() {
            for _ in 0..segment.steps {
                if expired(elapsed) {
                    stop(
                        "time limit before evolution step".into(),
                        &mut stopped,
                        &mut reasons,
                    );
                    break;
                }
                let s = state
                    .as_mut()
                    .context("missing initialized reference state")?;
                let attempt = (|| -> Result<_> {
                    let step =
                        engine.step(s, settings.hamiltonian, segment.dtau, settings.truncation)?;
                    if let (Some(c), Some(b)) = (&mut cap, bond_cap) {
                        c.observed_strang_steps += 1;
                        if step.max_bond > 0 {
                            c.observe(step.max_bond, b);
                        }
                    }
                    ensure!(
                        step.min_singular_value.is_finite() && step.max_bond > 0,
                        "invalid evolution step diagnostics"
                    );
                    let canonical = engine.canonicalize(s, settings.hamiltonian)?;
                    let norm = progress.accumulated_log_norm + step.log_norm + canonical;
                    ensure!(
                        step.log_norm.is_finite() && canonical.is_finite() && norm.is_finite(),
                        "nonfinite evolution normalization"
                    );
                    if let (Some(c), Some(b)) = (&mut cap, bond_cap) {
                        let summary = crate::itebd_state_view::ItebdStateRef::from(&*s)
                            .validate()
                            .context("reference evolved state validation")?;
                        for d in summary.bond_dimensions {
                            c.observe(d, b);
                        }
                    }
                    Ok((step, norm))
                })();
                match attempt {
                    Ok((step, norm)) => {
                        progress.beta += 2. * segment.dtau;
                        progress.completed_steps += 1;
                        progress.accumulated_log_norm = norm;
                        progress.last_step = Some(step);
                        if expired(elapsed) {
                            stop(
                                "time limit after complete evolution step".into(),
                                &mut stopped,
                                &mut reasons,
                            );
                            break;
                        }
                    }
                    Err(error) => {
                        endpoint_valid = false;
                        if let Some(c) = &mut cap {
                            c.mark_unobserved();
                        }
                        stop(
                            format!(
                                "evolution failed at step {}: {error:#}",
                                progress.completed_steps + 1
                            ),
                            &mut stopped,
                            &mut reasons,
                        );
                        break;
                    }
                }
            }
        }
        if stopped.is_none() && expired(elapsed) {
            stop(
                "time limit before target measurement".into(),
                &mut stopped,
                &mut reasons,
            );
        }
        let measured = if let Some(reason) = &stopped {
            Err(anyhow::anyhow!(reason.clone()))
        } else {
            (|| -> Result<V> {
                ensure!(
                    progress.completed_steps == target.completed_steps
                        && (progress.beta - target.beta).abs() <= 1e-12 * target.beta.max(1.),
                    "target beta/step roundoff agreement failed"
                );
                let v = engine.measure(
                    state.as_ref().context("missing measurement state")?,
                    settings.hamiltonian,
                    progress.beta,
                    progress.accumulated_log_norm,
                )?;
                v.validate()?;
                Ok(v)
            })()
        };
        let (values, reason) = match measured {
            Ok(v) => (Some(v), None),
            Err(error) => {
                let reason = format!("target {} unavailable: {error:#}", target.temperature_tick);
                if stopped.is_none() {
                    if bond_cap.is_some() {
                        stop(reason.clone(), &mut stopped, &mut reasons);
                    } else {
                        reasons.push(reason.clone());
                    }
                }
                (None, Some(reason))
            }
        };
        let sample = ScheduledSample {
            temperature_tick: target.temperature_tick,
            temperature: target.temperature,
            beta_target: target.beta,
            beta_actual: progress.beta,
            completed_steps: progress.completed_steps,
            available: values.is_some(),
            reason,
            accumulated_log_norm: progress.accumulated_log_norm,
            values,
            cap: cap.clone(),
        };
        callback(&sample).with_context(|| {
            format!(
                "reference output callback at target {}",
                target.temperature_tick
            )
        })?;
        samples.push(sample);
        // A completed measurement and publication are retained even if they cross the deadline.
        if stopped.is_none() && expired(elapsed) {
            stop(
                "time limit after target measurement/publication".into(),
                &mut stopped,
                &mut reasons,
            );
        }
    }
    let terminal_state = if endpoint_valid {
        let s = state.take().context("missing terminal state")?;
        match crate::itebd_state_view::ItebdStateRef::from(&s).validate() {
            Ok(_) => Some(s),
            Err(error) => {
                reasons.push(format!("terminal state validation: {error}"));
                if let Some(c) = &mut cap {
                    c.mark_unobserved();
                }
                None
            }
        }
    } else {
        None
    };
    let elapsed_seconds = elapsed();
    Ok(ScheduledOutcome {
        samples,
        terminal_state,
        progress,
        status: if reasons.is_empty() {
            "complete"
        } else {
            "partial"
        }
        .into(),
        reason: if reasons.is_empty() {
            "completed all requested targets".into()
        } else {
            reasons.join("; ")
        },
        elapsed_seconds,
        budget_overrun_seconds: limit.map_or(0., |v| (elapsed_seconds - v).max(0.)),
        cap,
    })
}
pub(crate) fn expectation(rho: &DMatrix<Complex64>, operator: &DMatrix<Complex64>) -> Result<f64> {
    ensure!(
        rho.is_square() && rho.shape() == operator.shape(),
        "RDM expectation shape mismatch"
    );
    // Tr(rho O), with bra/ket orientation supplied by the matrix product.
    let value = (rho * operator).trace();
    ensure!(
        value.re.is_finite()
            && value.im.is_finite()
            && value.im.abs() <= 1e-10 * value.re.abs().max(1.),
        "RDM expectation is not finite and real: {value}"
    );
    Ok(value.re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        itebd::StepInfo,
        itebd_auto::{ItebdHamiltonian, ItebdState},
        itebd_checkpoint::ItebdStepSegment,
        tensor::Truncation,
    };
    use anyhow::{bail, ensure, Result};
    use std::{cell::Cell, rc::Rc};
    #[derive(Clone, Debug)]
    struct Value(f64);
    impl Measurement for Value {
        fn validate(&self) -> Result<()> {
            ensure!(self.0.is_finite(), "nonfinite measurement");
            Ok(())
        }
    }
    struct Fixture {
        ham: ItebdHamiltonian,
        trunc: Truncation,
        segments: Vec<ItebdStepSegment>,
        targets: Vec<ReferenceTarget>,
    }
    impl Fixture {
        fn new() -> Self {
            Self {
                ham: ItebdHamiltonian::Real(crate::model::AkltProjector.local()),
                trunc: Truncation {
                    epsilon: 1e-13,
                    max_bond: Some(128),
                },
                segments: vec![
                    ItebdStepSegment {
                        steps: 2,
                        dtau: 0.05
                    };
                    2
                ],
                targets: vec![
                    ReferenceTarget {
                        temperature_tick: 50,
                        temperature: 5.,
                        beta: 0.2,
                        completed_steps: 2,
                    },
                    ReferenceTarget {
                        temperature_tick: 25,
                        temperature: 2.5,
                        beta: 0.4,
                        completed_steps: 4,
                    },
                ],
            }
        }
        fn settings(&self) -> TrajectorySettings<'_> {
            TrajectorySettings {
                hamiltonian: &self.ham,
                truncation: &self.trunc,
                segments: &self.segments,
                targets: &self.targets,
                policy: RunPolicy::Aklt {
                    max_seconds: Some(3600.),
                    bond_cap: 128,
                },
            }
        }
    }
    #[derive(Clone, Copy, PartialEq)]
    enum Phase {
        Setup,
        Step,
        Canonical,
        Validation,
        Measurement,
        NonfiniteMeasurement,
        Callback,
    }
    struct Seam {
        clock: Rc<Cell<f64>>,
        steps: usize,
        measures: usize,
        setups: usize,
        deadline: Option<(Phase, usize)>,
        failure: Option<(Phase, usize)>,
        hit: bool,
    }
    impl Seam {
        fn new() -> Self {
            Self {
                clock: Rc::new(Cell::new(0.)),
                steps: 0,
                measures: 0,
                setups: 0,
                deadline: None,
                failure: None,
                hit: false,
            }
        }
        fn visit(&self, p: Phase, n: usize) {
            if self.deadline == Some((p, n)) {
                self.clock.set(3601.);
            }
        }
    }
    impl Engine<Value> for Seam {
        fn setup(&mut self, h: &ItebdHamiltonian) -> Result<ItebdState> {
            self.setups += 1;
            self.visit(Phase::Setup, self.setups);
            Ok(ItebdState::infinite_temperature(h)?)
        }
        fn step(
            &mut self,
            s: &mut ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            _: &Truncation,
        ) -> Result<StepInfo> {
            self.steps += 1;
            self.visit(Phase::Step, self.steps);
            if self.failure == Some((Phase::Step, self.steps)) {
                if let ItebdState::Real(real) = s {
                    real.lambda_ab[0] *= 2.;
                }
                bail!("originating step failure after mutation");
            }
            Ok(StepInfo {
                max_bond: if self.hit && self.steps == 1 { 128 } else { 1 },
                min_singular_value: 1.,
                log_norm: 0.2,
            })
        }
        fn canonicalize(&mut self, s: &mut ItebdState, _: &ItebdHamiltonian) -> Result<f64> {
            self.visit(Phase::Canonical, self.steps);
            if self.failure == Some((Phase::Canonical, self.steps)) {
                bail!("originating canonicalization failure");
            }
            if self.failure == Some((Phase::Validation, self.steps)) {
                if let ItebdState::Real(real) = s {
                    real.lambda_ab[0] = f64::NAN;
                }
                self.visit(Phase::Validation, self.steps);
            }
            Ok(0.3)
        }
        fn measure(
            &mut self,
            _: &ItebdState,
            _: &ItebdHamiltonian,
            _: f64,
            norm: f64,
        ) -> Result<Value> {
            self.measures += 1;
            self.visit(Phase::Measurement, self.measures);
            if self.failure == Some((Phase::Measurement, self.measures)) {
                bail!("originating measurement failure");
            }
            Ok(Value(
                if self.failure == Some((Phase::NonfiniteMeasurement, self.measures)) {
                    f64::NAN
                } else {
                    norm
                },
            ))
        }
    }
    fn run(f: &Fixture, engine: &mut Seam) -> Result<ScheduledOutcome<Value>> {
        let clock = engine.clock.clone();
        let callback_clock = engine.clock.clone();
        let deadline = engine.deadline;
        let mut callbacks = 0;
        run_scheduled(
            f.settings(),
            &mut |_| {
                callbacks += 1;
                if deadline == Some((Phase::Callback, callbacks)) {
                    callback_clock.set(3601.);
                }
                Ok(())
            },
            engine,
            &mut || clock.get(),
        )
    }
    #[test]
    fn aklt_unlimited_clock_keeps_cap_evidence() {
        let f = Fixture::new();
        let mut settings = f.settings();
        settings.policy = RunPolicy::Aklt {
            max_seconds: None,
            bond_cap: 128,
        };
        let mut e = Seam::new();
        e.clock.set(4000.);
        e.hit = true;
        let clock = e.clock.clone();
        let out = run_scheduled(settings, &mut |_| Ok(()), &mut e, &mut || clock.get()).unwrap();
        assert_eq!(e.steps, 4);
        assert!(out.samples.iter().all(|s| s.available));
        assert_eq!(out.elapsed_seconds, 4000.);
        let cap = out.cap.unwrap();
        assert_eq!(cap.ever_reached, Some(true));
        assert_eq!(cap.maximum_dimension_seen, Some(128));
        assert_eq!(cap.observed_strang_steps, 4);
        assert_eq!(cap.coverage, CapCoverage::CompleteThroughStop);
        let mut old = Seam::new();
        old.clock.set(4000.);
        let bounded = run(&f, &mut old).unwrap();
        assert_eq!(old.steps, 0);
        assert!(bounded.reason.contains("time limit"));
        for (phase, n, cause) in [
            (Phase::Step, 3, "originating step"),
            (Phase::Canonical, 3, "originating canonicalization"),
            (Phase::Validation, 3, "lambda_ab"),
            (Phase::NonfiniteMeasurement, 2, "nonfinite measurement"),
        ] {
            let mut settings = f.settings();
            settings.policy = RunPolicy::Aklt {
                max_seconds: None,
                bond_cap: 128,
            };
            let mut e = Seam::new();
            e.clock.set(4000.);
            e.failure = Some((phase, n));
            let clock = e.clock.clone();
            let out =
                run_scheduled(settings, &mut |_| Ok(()), &mut e, &mut || clock.get()).unwrap();
            assert!(out.samples[0].available);
            assert!(!out.samples[1].available);
            assert!(out.reason.contains(cause), "{}", out.reason);
            assert!(!out.reason.contains("time limit"));
            assert_eq!(
                out.terminal_state.is_some(),
                phase == Phase::NonfiniteMeasurement
            );
            if phase != Phase::NonfiniteMeasurement {
                assert_eq!(out.cap.unwrap().coverage, CapCoverage::UnobservedTail);
            }
        }
    }
    #[test]
    fn aklt_deadline_retains_completed_measurement() {
        for phase in [Phase::Measurement, Phase::Callback] {
            let mut e = Seam::new();
            e.deadline = Some((phase, 1));
            let out = run(&Fixture::new(), &mut e).unwrap();
            assert!(out.samples[0].available);
            assert!(!out.samples[1].available);
            assert_eq!(e.steps, 2);
            assert_eq!(e.measures, 1);
            assert_eq!(out.progress.beta, 0.2);
            assert_eq!(out.progress.accumulated_log_norm, 1.);
            assert_eq!(out.samples[0].values.as_ref().unwrap().0, 1.);
            assert!(out.terminal_state.is_some());
            assert_eq!(out.budget_overrun_seconds, 1.);
            assert!(out.reason.contains("time limit"));
        }
    }
    #[test]
    fn aklt_setup_and_step_deadlines_keep_actual_progress() {
        for (phase, n, steps, beta) in [
            (Phase::Setup, 1, 0, 0.),
            (Phase::Canonical, 1, 1, 0.1),
            (Phase::Canonical, 2, 2, 0.2),
        ] {
            let mut e = Seam::new();
            e.deadline = Some((phase, n));
            let out = run(&Fixture::new(), &mut e).unwrap();
            assert_eq!(out.progress.completed_steps, steps);
            assert_eq!(out.progress.beta, beta);
            assert!(out.terminal_state.is_some());
            assert_eq!(e.measures, 0);
            assert!(out
                .samples
                .iter()
                .all(|s| !s.available && s.beta_actual == beta));
        }
        let mut e = Seam::new();
        e.clock.set(3601.);
        let out = run(&Fixture::new(), &mut e).unwrap();
        assert_eq!(e.setups, 0);
        assert_eq!(e.steps, 0);
        assert!(out.terminal_state.is_none());
        assert_eq!(out.cap.as_ref().unwrap().ever_reached, None);
    }
    #[test]
    fn aklt_failure_precedes_deadline() {
        for (phase, n, cause) in [
            (Phase::Step, 3, "originating step"),
            (Phase::Canonical, 3, "originating canonicalization"),
            (Phase::Validation, 3, "lambda_ab"),
            (Phase::Measurement, 2, "originating measurement"),
        ] {
            let mut e = Seam::new();
            e.deadline = Some((phase, n));
            e.failure = Some((phase, n));
            let out = run(&Fixture::new(), &mut e).unwrap();
            assert!(out.samples[0].available);
            assert!(!out.samples[1].available);
            assert!(out.reason.contains(cause), "{}", out.reason);
            assert!(!out.reason.contains("time limit"));
            assert_eq!(out.terminal_state.is_some(), phase == Phase::Measurement);
            assert_eq!(
                out.progress.completed_steps,
                if phase == Phase::Measurement { 4 } else { 2 }
            );
            assert_eq!(out.budget_overrun_seconds, 1.);
        }
    }
    #[test]
    fn aklt_cap_hit_survives_shrink_and_unknown_tail() {
        let mut e = Seam::new();
        e.hit = true;
        let out = run(&Fixture::new(), &mut e).unwrap();
        let cap = out.cap.unwrap();
        assert_eq!(cap.maximum_dimension_seen, Some(128));
        assert_eq!(cap.ever_reached, Some(true));
        assert_eq!(cap.observed_strang_steps, 4);
        assert_eq!(cap.coverage, CapCoverage::CompleteThroughStop);
        for hit in [false, true] {
            let mut e = Seam::new();
            e.hit = hit;
            e.failure = Some((Phase::Step, 3));
            let out = run(&Fixture::new(), &mut e).unwrap();
            let cap = out.cap.unwrap();
            assert_eq!(cap.ever_reached, if hit { Some(true) } else { None });
            assert_eq!(cap.observed_strang_steps, 2);
            assert_eq!(cap.coverage, CapCoverage::UnobservedTail);
            assert!(out.terminal_state.is_none());
        }
        let mut e = Seam::new();
        e.hit = true;
        e.failure = Some((Phase::Canonical, 1));
        let out = run(&Fixture::new(), &mut e).unwrap();
        let cap = out.cap.unwrap();
        assert_eq!(cap.ever_reached, Some(true));
        assert_eq!(cap.maximum_dimension_seen, Some(128));
        assert_eq!(cap.observed_strang_steps, 1);
        assert_eq!(out.progress.completed_steps, 0);
        assert_eq!(cap.coverage, CapCoverage::UnobservedTail);
    }
    #[test]
    fn aklt_reference_complex_trace_orientation() {
        use crate::model::{id2, pauli_z};
        use nalgebra::DMatrix;
        use num_complex::Complex64;
        // Independent component sum uses rho_ij O_ji, including complex conjugate off-diagonals.
        for phase in [0., 0.2] {
            let rho = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(0.7, 0.),
                    Complex64::new(0.1, phase),
                    Complex64::new(0.1, -phase),
                    Complex64::new(0.3, 0.),
                ],
            );
            let op = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(2., 0.),
                    Complex64::new(0.4, if phase == 0. { 0. } else { -0.3 }),
                    Complex64::new(0.4, if phase == 0. { 0. } else { 0.3 }),
                    Complex64::new(-1., 0.),
                ],
            );
            let mut oracle = Complex64::new(0., 0.);
            for a in 0..2 {
                for b in 0..2 {
                    oracle += rho[(a, b)] * op[(b, a)];
                }
            }
            assert!((expectation(&rho, &op).unwrap() - oracle.re).abs() < 1e-14);
            let joint = rho.kronecker(&DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
            ])));
            let first = id2()
                .map(|x| Complex64::new(x, 0.))
                .kronecker(&pauli_z().map(|x| Complex64::new(x, 0.)));
            assert!((expectation(&joint, &first).unwrap() - 1.).abs() < 1e-14);
        }
    }
}
