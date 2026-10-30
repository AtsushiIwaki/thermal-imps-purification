//! Step identity is independent of historical temperature targets.
use super::*;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct StepMeasurementPoint {
    pub sample_index: u32,
    pub completed_steps: u64,
    pub beta_expected: f64,
}

pub(crate) struct StepTrajectorySettings<'a> {
    pub hamiltonian: &'a ItebdHamiltonian,
    pub truncation: &'a Truncation,
    pub segments: &'a [ItebdStepSegment],
    pub points: &'a [StepMeasurementPoint],
    pub policy: RunPolicy,
}
#[derive(Clone)]
pub(crate) struct StepSample<V> {
    pub sample_index: u32,
    pub completed_steps: u64,
    pub beta_actual: f64,
    pub temperature: f64,
    pub accumulated_log_norm: f64,
    pub available: bool,
    pub reason: Option<String>,
    pub values: Option<V>,
    pub cap: Option<CapEvidence>,
}
pub(crate) trait StepObserver<V> {
    fn sample(&mut self, row: &StepSample<V>) -> Result<()>;
}
pub(crate) struct StepSampledOutcome<V> {
    pub samples: Vec<StepSample<V>>,
    pub progress: ItebdCheckpointProgress,
    pub status: String,
    pub reason: String,
    pub elapsed_seconds: f64,
    pub cap: Option<CapEvidence>,
}
pub(crate) fn run_step_sampled_from<V: Measurement>(
    settings: StepTrajectorySettings<'_>,
    observer: &mut dyn StepObserver<V>,
    engine: &mut dyn Engine<V>,
    elapsed: &mut dyn FnMut() -> f64,
    start: ScheduledStart,
) -> Result<StepSampledOutcome<V>> {
    let (limit, bound) = match settings.policy {
        RunPolicy::Aklt {
            max_seconds,
            bond_cap,
        } => (max_seconds, bond_cap),
        _ => anyhow::bail!("step observation requires an AKLT restart policy"),
    };
    ensure!(
        bound > 0 && limit.is_none_or(|v| v.is_finite() && v > 0.),
        "invalid step observation policy"
    );
    ensure!(
        !settings.segments.is_empty() && !settings.points.is_empty(),
        "empty step observation schedule"
    );
    let total = settings
        .segments
        .iter()
        .try_fold(start.progress.completed_steps, |n, s| {
            ensure!(
                s.steps > 0 && s.dtau.is_finite() && s.dtau > 0.,
                "invalid step segment"
            );
            n.checked_add(s.steps).context("step schedule overflow")
        })?;
    let mut previous = start.progress.completed_steps;
    for (i, p) in settings.points.iter().enumerate() {
        ensure!(
            p.sample_index == i as u32 + 1
                && p.completed_steps > previous
                && p.completed_steps <= total
                && p.beta_expected.is_finite()
                && p.beta_expected > 0.,
            "invalid step observation identity"
        );
        previous = p.completed_steps;
    }
    ensure!(
        previous == total,
        "step observation must include the endpoint"
    );
    let mut state = start.state;
    let mut progress = start.progress;
    ensure!(
        progress.beta.is_finite()
            && progress.beta > 0.
            && progress.accumulated_log_norm.is_finite(),
        "invalid step restart progress"
    );
    let mut cap = Some(start.cap);
    let initial = crate::itebd_state_view::ItebdStateRef::from(&state).validate()?;
    if let Some(c) = &mut cap {
        for dimension in initial.bond_dimensions {
            c.observe(dimension, bound);
        }
    }
    let mut samples = Vec::with_capacity(settings.points.len());
    let mut cursor = 0;
    let mut reason = None;
    let expired = |clock: &mut dyn FnMut() -> f64| limit.is_some_and(|v| clock() >= v);
    'evolution: for segment in settings.segments {
        for _ in 0..segment.steps {
            if expired(elapsed) {
                reason = Some("time limit before evolution step".into());
                break 'evolution;
            }
            if let Err(error) = advance_complete_step(
                &mut state,
                settings.hamiltonian,
                settings.truncation,
                segment.dtau,
                engine,
                &mut progress,
                &mut cap,
                Some(bound),
            ) {
                if let Some(c) = &mut cap {
                    c.mark_unobserved();
                }
                reason = Some(format!(
                    "evolution failed at step {}: {error:#}",
                    progress.completed_steps + 1
                ));
                break 'evolution;
            }
            if expired(elapsed) {
                reason = Some("time limit after complete evolution step".into());
                break 'evolution;
            }
            let point = &settings.points[cursor];
            if progress.completed_steps != point.completed_steps {
                continue;
            }
            let measured = (|| -> Result<V> {
                ensure!(
                    progress.beta.to_bits() == point.beta_expected.to_bits(),
                    "step beta/progress mismatch"
                );
                let value = engine.measure(
                    &state,
                    settings.hamiltonian,
                    progress.beta,
                    progress.accumulated_log_norm,
                )?;
                value.validate()?;
                Ok(value)
            })();
            let (values, failure) = match measured {
                Ok(value) => (Some(value), None),
                Err(error) => (
                    None,
                    Some(format!(
                        "measurement at step {} unavailable: {error:#}",
                        progress.completed_steps
                    )),
                ),
            };
            let sample = StepSample {
                sample_index: point.sample_index,
                completed_steps: progress.completed_steps,
                beta_actual: progress.beta,
                temperature: 1. / progress.beta,
                accumulated_log_norm: progress.accumulated_log_norm,
                available: values.is_some(),
                reason: failure.clone(),
                values,
                cap: cap.clone(),
            };
            observer.sample(&sample).with_context(|| {
                format!("scalar publication at step {}", progress.completed_steps)
            })?;
            let measurement_stop = sample
                .values
                .as_ref()
                .and_then(Measurement::stop_reason)
                .map(str::to_owned);
            samples.push(sample);
            cursor += 1;
            if let Some(failure) = failure.or(measurement_stop) {
                reason = Some(failure);
                break 'evolution;
            }
        }
    }
    let complete =
        reason.is_none() && cursor == settings.points.len() && progress.completed_steps == total;
    Ok(StepSampledOutcome {
        samples,
        progress,
        status: if complete { "complete" } else { "partial" }.into(),
        reason: reason.unwrap_or_else(|| {
            if complete {
                "completed all step observations"
            } else {
                "incomplete step observations"
            }
            .into()
        }),
        elapsed_seconds: elapsed(),
        cap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aklt_reference::v3;
    use crate::reference_output::exact_state;
    use serde_json::json;

    #[derive(Clone)]
    struct Scalar {
        stop: bool,
    }
    impl Measurement for Scalar {
        fn validate(&self) -> Result<()> {
            Ok(())
        }
        fn stop_reason(&self) -> Option<&str> {
            self.stop.then_some("fixture variance tail failed")
        }
    }
    #[derive(Default)]
    struct Probe {
        setup_calls: usize,
        dtau_bits: Vec<u64>,
        canonical_calls: usize,
        betas: Vec<f64>,
        stop_at: Option<usize>,
        last: Option<ItebdState>,
    }
    fn clone_state(s: &ItebdState) -> ItebdState {
        match s {
            ItebdState::Real(s) => {
                use crate::purified_mps::{PurifiedMps, Site};
                let site = |a: &Site| Site {
                    gamma: a.gamma.clone(),
                    left: a.left.clone(),
                    phys: a.phys.clone(),
                    anc: a.anc.clone(),
                    right: a.right.clone(),
                };
                ItebdState::Real(PurifiedMps {
                    a: site(&s.a),
                    b: site(&s.b),
                    lambda_ab: s.lambda_ab.clone(),
                    lambda_bond_ab: s.lambda_bond_ab.clone(),
                    lambda_ba: s.lambda_ba.clone(),
                    lambda_bond_ba: s.lambda_bond_ba.clone(),
                })
            }
            ItebdState::Complex(s) => ItebdState::Complex(s.clone()),
        }
    }
    impl Engine<Scalar> for Probe {
        fn setup(&mut self, h: &ItebdHamiltonian) -> Result<ItebdState> {
            self.setup_calls += 1;
            Ok(ItebdState::infinite_temperature(h)?)
        }
        fn step(
            &mut self,
            _: &mut ItebdState,
            _: &ItebdHamiltonian,
            dt: f64,
            _: &Truncation,
        ) -> Result<StepInfo> {
            self.dtau_bits.push(dt.to_bits());
            Ok(StepInfo {
                max_bond: 1,
                min_singular_value: 1.,
                log_norm: 0.,
            })
        }
        fn canonicalize(&mut self, s: &mut ItebdState, _: &ItebdHamiltonian) -> Result<f64> {
            self.canonical_calls += 1;
            self.last = Some(clone_state(s));
            Ok(0.)
        }
        fn measure(
            &mut self,
            _: &ItebdState,
            _: &ItebdHamiltonian,
            b: f64,
            _: f64,
        ) -> Result<Scalar> {
            self.betas.push(b);
            Ok(Scalar {
                stop: self.stop_at == Some(self.betas.len()),
            })
        }
    }
    #[derive(Default)]
    struct Observer {
        steps: Vec<u64>,
    }
    impl StepObserver<Scalar> for Observer {
        fn sample(&mut self, r: &StepSample<Scalar>) -> Result<()> {
            self.steps.push(r.completed_steps);
            Ok(())
        }
    }
    fn base() -> v3::AkltReferenceSettingsV3 {
        v3::resolve_v3_request(
            &serde_json::from_value(json!({"schema_version":3,"contract":v3::CONTRACT_V3,
            "model":{"name":"aklt_projector"},"variant":"fine","max_bond":256,"max_seconds":null,
            "recording_policy":v3::RECORDING_POLICY,"heat_capacity_policy":v3::HEAT_POLICY,
            "source_revision":"a".repeat(40),"worker_sha256":"b".repeat(64)}))
            .unwrap(),
        )
        .unwrap()
    }
    fn start(h: &ItebdHamiltonian) -> ScheduledStart {
        ScheduledStart {
            state: ItebdState::infinite_temperature(h).unwrap(),
            progress: ItebdCheckpointProgress {
                beta: 0.5000000000000002,
                completed_steps: 50,
                accumulated_log_norm: -0.4937122532815281,
                last_step: None,
            },
            cap: CapEvidence {
                maximum_dimension_seen: Some(30),
                ever_reached: Some(false),
                observed_strang_steps: 50,
                coverage: CapCoverage::CompleteThroughStop,
            },
        }
    }
    fn points(segments: &[ItebdStepSegment], initial: f64) -> Vec<StepMeasurementPoint> {
        let mut beta = initial;
        let mut step = 50;
        let mut result = Vec::new();
        for segment in segments {
            for _ in 0..segment.steps {
                beta += 2. * segment.dtau;
                step += 1;
                if (step - 50) % 2 == 0 {
                    result.push(StepMeasurementPoint {
                        sample_index: result.len() as u32 + 1,
                        completed_steps: step,
                        beta_expected: beta,
                    });
                }
            }
        }
        result
    }
    #[test]
    fn step_sampled_preserves_sequence_and_cadence() {
        let b = base();
        let s = start(b.hamiltonian());
        let segments = &b.segments()[1..];
        let points = points(segments, s.progress.beta);
        let mut engine = Probe::default();
        let mut observer = Observer::default();
        let mut reference_state = clone_state(&s.state);
        let mut reference_progress = s.progress.clone();
        let mut reference_cap = Some(s.cap.clone());
        let mut unobserved = Probe::default();
        let result = run_step_sampled_from(
            StepTrajectorySettings {
                hamiltonian: b.hamiltonian(),
                truncation: b.truncation(),
                segments,
                points: &points,
                policy: RunPolicy::Aklt {
                    max_seconds: None,
                    bond_cap: 256,
                },
            },
            &mut observer,
            &mut engine,
            &mut || 0.,
            s,
        );
        assert!(
            result.is_ok(),
            "step runner rejected valid sequence: {:?}",
            result.err()
        );
        let outcome = result.unwrap();
        assert_eq!(engine.setup_calls, 0);
        assert_eq!(engine.dtau_bits.len(), 964);
        assert_eq!(engine.canonical_calls, 964);
        assert_eq!(outcome.samples.len(), 482);
        assert_eq!(observer.steps, (52..=1014).step_by(2).collect::<Vec<_>>());
        assert_eq!(outcome.progress.completed_steps, 1014);
        assert_eq!(
            outcome.progress.accumulated_log_norm.to_bits(),
            (-0.4937122532815281_f64).to_bits()
        );
        assert_eq!(
            outcome.progress.beta.to_bits(),
            9.999999999999915_f64.to_bits()
        );
        for segment in segments {
            for _ in 0..segment.steps {
                advance_complete_step(
                    &mut reference_state,
                    b.hamiltonian(),
                    b.truncation(),
                    segment.dtau,
                    &mut unobserved,
                    &mut reference_progress,
                    &mut reference_cap,
                    Some(256),
                )
                .unwrap();
            }
        }
        assert_eq!(engine.dtau_bits, unobserved.dtau_bits);
        exact_state(engine.last.as_ref().unwrap(), &reference_state).unwrap();
        for sample in &outcome.samples {
            assert_eq!(sample.temperature, 1. / sample.beta_actual);
        }
        assert_eq!(outcome.status, "complete");
    }
    #[test]
    fn step_sampled_crosses_odd_segment_boundary() {
        let b = base();
        let s = start(b.hamiltonian());
        let segments = [
            ItebdStepSegment {
                steps: 1,
                dtau: 0.005,
            },
            ItebdStepSegment {
                steps: 3,
                dtau: 0.004,
            },
        ];
        let points = points(&segments, s.progress.beta);
        let mut engine = Probe::default();
        let mut observer = Observer::default();
        let result = run_step_sampled_from(
            StepTrajectorySettings {
                hamiltonian: b.hamiltonian(),
                truncation: b.truncation(),
                segments: &segments,
                points: &points,
                policy: RunPolicy::Aklt {
                    max_seconds: None,
                    bond_cap: 256,
                },
            },
            &mut observer,
            &mut engine,
            &mut || 0.,
            s,
        )
        .unwrap();
        assert_eq!(observer.steps, [52, 54]);
        assert_eq!(engine.dtau_bits.len(), 4);
        assert_eq!(result.samples.len(), 2);
    }
    #[test]
    fn step_sampled_stops_at_failed_measurement() {
        let b = base();
        let s = start(b.hamiltonian());
        let segments = &b.segments()[1..];
        let points = points(segments, s.progress.beta);
        let mut engine = Probe {
            stop_at: Some(2),
            ..Default::default()
        };
        let mut observer = Observer::default();
        let result = run_step_sampled_from(
            StepTrajectorySettings {
                hamiltonian: b.hamiltonian(),
                truncation: b.truncation(),
                segments,
                points: &points,
                policy: RunPolicy::Aklt {
                    max_seconds: None,
                    bond_cap: 256,
                },
            },
            &mut observer,
            &mut engine,
            &mut || 0.,
            s,
        )
        .unwrap();
        assert_eq!(observer.steps, [52, 54]);
        assert_eq!(engine.dtau_bits.len(), 4);
        assert_eq!(result.status, "partial");
        assert!(result.reason.contains("variance tail"));
        assert!(result.samples[1].values.is_some());
    }
}
