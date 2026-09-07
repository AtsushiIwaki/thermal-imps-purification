#![allow(dead_code)] // Shared by the real and complex kernels added in later tasks.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itebd_error::ItebdError;
    use num_complex::Complex64;

    fn shell(distance: usize, value: f64) -> ParityShell<f64> {
        ParityShell::new(distance, [value, value], [value, value])
    }

    fn shell_with_canceled_real_parts(
        distance: usize,
        paired_imaginary: f64,
    ) -> ParityShell<Complex64> {
        let positive = [Complex64::new(100.0, 0.0), Complex64::new(-100.0, 0.0)];
        let expected_negative = if distance & 1 == 0 {
            [positive[0], positive[1]]
        } else {
            [positive[1], positive[0]]
        };
        ParityShell::new(
            distance,
            positive,
            [
                expected_negative[0] + Complex64::new(0.0, paired_imaginary),
                expected_negative[1] + Complex64::new(0.0, paired_imaginary),
            ],
        )
    }

    fn zero_tail(_distance: usize) -> Result<ParityShell<f64>, ItebdError> {
        Ok(ParityShell::zero())
    }

    #[test]
    fn defaults_match_the_approved_contract() {
        assert_eq!(
            SpecificHeatOptions::default(),
            SpecificHeatOptions {
                max_distance: 200,
                relative_tolerance: 1e-6,
                absolute_tolerance: 1e-12,
                consecutive_small_shells: 3,
                reality_tolerance: 1e-10,
            }
        );
    }

    #[test]
    fn beta_and_tail_count_errors_are_typed() {
        assert!(matches!(validate_beta(-0.1),
            Err(ItebdError::InvalidSpecificHeatBeta { value }) if value == -0.1));
        assert!(matches!(validate_beta(f64::NAN),
            Err(ItebdError::InvalidSpecificHeatBeta { value }) if value.is_nan()));
        assert!(matches!(validate_beta(f64::INFINITY),
            Err(ItebdError::InvalidSpecificHeatBeta { value }) if value.is_infinite()));
        assert!(validate_beta(0.0).is_ok());

        let zero_count = SpecificHeatOptions {
            consecutive_small_shells: 0,
            ..SpecificHeatOptions::default()
        };
        assert!(matches!(
            zero_count.validate(),
            Err(ItebdError::InvalidSpecificHeatCount {
                name: "consecutive_small_shells",
                value: 0,
                minimum: 1
            })
        ));

        let insufficient_samples = SpecificHeatOptions {
            max_distance: 3,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        assert!(matches!(
            insufficient_samples.validate(),
            Err(ItebdError::InvalidSpecificHeatCount {
                name: "max_distance",
                value: 3,
                minimum: 4
            })
        ));
    }

    #[test]
    fn options_reject_zero_max_distance_and_unrepresentable_tail_count() {
        let zero_distance = SpecificHeatOptions {
            max_distance: 0,
            ..SpecificHeatOptions::default()
        };
        assert!(matches!(
            zero_distance.validate(),
            Err(ItebdError::InvalidSpecificHeatCount {
                name: "max_distance",
                value: 0,
                minimum: 4,
            })
        ));

        let unrepresentable_tail = SpecificHeatOptions {
            max_distance: usize::MAX,
            consecutive_small_shells: usize::MAX,
            ..SpecificHeatOptions::default()
        };
        assert!(matches!(
            unrepresentable_tail.validate(),
            Err(ItebdError::InvalidSpecificHeatCount {
                name: "max_distance",
                value: usize::MAX,
                minimum: usize::MAX,
            })
        ));
    }

    #[test]
    fn options_reject_nonfinite_or_negative_tolerances() {
        for (name, options) in [
            (
                "relative_tolerance",
                SpecificHeatOptions {
                    relative_tolerance: -1e-6,
                    ..SpecificHeatOptions::default()
                },
            ),
            (
                "absolute_tolerance",
                SpecificHeatOptions {
                    absolute_tolerance: f64::NAN,
                    ..SpecificHeatOptions::default()
                },
            ),
            (
                "reality_tolerance",
                SpecificHeatOptions {
                    reality_tolerance: f64::INFINITY,
                    ..SpecificHeatOptions::default()
                },
            ),
        ] {
            assert!(matches!(options.validate(),
                Err(ItebdError::InvalidTolerance { name: actual, .. }) if actual == name));
        }
    }

    #[test]
    fn report_reconstructs_both_ways() {
        let report = accumulate_specific_heat(
            2.0,
            &SpecificHeatOptions {
                relative_tolerance: 0.0,
                absolute_tolerance: 1e-8,
                ..SpecificHeatOptions::default()
            },
            [2.0_f64, 4.0],
            ParityShell::new(1, [0.5, 0.25], [0.25, 0.5]),
            zero_tail,
        )
        .unwrap();
        assert_eq!(
            Complex64::new(report.raw_energy_variance_per_site, 0.0),
            report.onsite_contribution
                + report.positive_direction_contribution
                + report.negative_direction_contribution
        );
        assert_eq!(
            Complex64::new(report.raw_energy_variance_per_site, 0.0),
            report.parity_a_contribution + report.parity_b_contribution
        );
        assert_eq!(
            report.specific_heat_per_site,
            4.0 * report.energy_variance_per_site
        );
        assert_eq!(report.stop_reason, TailStopReason::ConsecutiveSmallShells);
        assert_eq!(report.max_distance, 4);
    }

    #[test]
    fn an_increasing_oscillatory_tail_does_not_stop_early() {
        let values = [1e-2, -2e-2, 5e-3, 0.0, 0.0, 0.0];
        let report = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions {
                max_distance: 7,
                relative_tolerance: 0.0,
                absolute_tolerance: 1e-3,
                consecutive_small_shells: 3,
                ..SpecificHeatOptions::default()
            },
            [1.0_f64, 1.0],
            shell(1, 0.0),
            |distance| Ok(shell(distance, values[distance - 2])),
        )
        .unwrap();

        assert_eq!(report.max_distance, 7);
        assert_eq!(report.last_positive_shell_magnitude, 0.0);
        assert_eq!(report.last_negative_shell_magnitude, 0.0);
        assert_eq!(report.raw_energy_variance_per_site, 0.99);
    }

    #[test]
    fn accumulated_local_contribution_sets_the_relative_tail_scale() {
        let report = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions {
                max_distance: 4,
                relative_tolerance: 1e-6,
                absolute_tolerance: 0.0,
                consecutive_small_shells: 3,
                ..SpecificHeatOptions::default()
            },
            [1e6_f64, 1e6],
            shell(1, 0.0),
            |distance| Ok(shell(distance, 0.5)),
        )
        .unwrap();

        assert_eq!(report.max_distance, 4);
    }

    #[test]
    fn canceled_transient_shell_does_not_permanently_loosen_the_tail() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions {
                max_distance: 5,
                relative_tolerance: 1e-6,
                absolute_tolerance: 0.0,
                consecutive_small_shells: 3,
                ..SpecificHeatOptions::default()
            },
            [0.0_f64, 0.0],
            shell(1, 1e6),
            |distance| Ok(shell(distance, if distance == 2 { -1e6 } else { 0.5 })),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::SpecificHeatTailNonConvergence {
                max_distance: 5,
                ..
            }
        ));
    }

    #[test]
    fn complex_shell_cancellation_preserves_complex_diagnostics() {
        let report = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions::default(),
            [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            ParityShell::new(
                1,
                [Complex64::new(0.0, -0.5), Complex64::new(0.0, -0.5)],
                [Complex64::new(0.0, 0.5), Complex64::new(0.0, 0.5)],
            ),
            |_distance| Ok(ParityShell::zero()),
        )
        .unwrap();

        assert_eq!(report.raw_energy_variance_per_site, 1.0);
        assert_ne!(report.positive_direction_contribution.im, 0.0);
        assert_eq!(report.max_imaginary_residual, 0.0);
    }

    #[test]
    fn paired_shell_reality_is_checked_before_later_shells_cancel_it() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions {
                max_distance: 5,
                relative_tolerance: 0.0,
                absolute_tolerance: 1e-12,
                consecutive_small_shells: 3,
                ..SpecificHeatOptions::default()
            },
            [Complex64::new(1.0, 0.0); 2],
            shell_with_canceled_real_parts(1, 2e-10),
            |distance| {
                if distance == 2 {
                    Ok(shell_with_canceled_real_parts(distance, -2e-10))
                } else {
                    Ok(ParityShell::zero())
                }
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::NonRealSpecificHeat {
                stage: "paired_shell",
                imaginary,
                tolerance,
            } if imaginary == 2e-10 && tolerance == 1e-10
        ));
    }

    #[test]
    fn report_retains_the_largest_paired_shell_imaginary_residual() {
        let paired_imaginary = 5e-11;
        let report = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions {
                max_distance: 5,
                relative_tolerance: 0.0,
                absolute_tolerance: 1e-12,
                consecutive_small_shells: 3,
                ..SpecificHeatOptions::default()
            },
            [Complex64::new(1.0, 0.0); 2],
            shell_with_canceled_real_parts(1, paired_imaginary),
            |distance| {
                if distance == 2 {
                    Ok(shell_with_canceled_real_parts(distance, -paired_imaginary))
                } else {
                    Ok(ParityShell::zero())
                }
            },
        )
        .unwrap();

        assert_eq!(report.max_imaginary_residual, paired_imaginary);
    }

    #[test]
    fn excessive_imaginary_parity_total_is_rejected() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions::default(),
            [Complex64::new(1.0, 1e-4), Complex64::new(1.0, 0.0)],
            ParityShell::new(
                1,
                [Complex64::new(0.0, 0.0); 2],
                [Complex64::new(0.0, 0.0); 2],
            ),
            |_distance| Ok(ParityShell::zero()),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::NonRealSpecificHeat {
                stage: "parity_a",
                ..
            }
        ));
    }

    #[test]
    fn direction_mismatch_is_rejected() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions::default(),
            [0.0_f64, 0.0],
            ParityShell::new(1, [1.0, 0.0], [0.0, 0.0]),
            zero_tail,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::SpecificHeatDirectionMismatch {
                parity: 1,
                distance: 1,
                ..
            }
        ));
    }

    #[test]
    fn cap_without_a_converged_tail_is_rejected() {
        let options = SpecificHeatOptions {
            max_distance: 4,
            consecutive_small_shells: 3,
            relative_tolerance: 0.0,
            absolute_tolerance: 1e-6,
            ..SpecificHeatOptions::default()
        };
        let error =
            accumulate_specific_heat(1.0, &options, [0.0_f64, 0.0], shell(1, 1.0), |distance| {
                Ok(shell(distance, 1.0))
            })
            .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::SpecificHeatTailNonConvergence {
                max_distance: 4,
                consecutive_small_shells: 3,
                ..
            }
        ));
    }

    #[test]
    fn nonfinite_correlation_value_is_rejected_with_its_location() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions::default(),
            [f64::NAN, 0.0],
            shell(1, 0.0),
            zero_tail,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::NonFiniteSpecificHeatValue {
                stage: "onsite",
                parity: 0,
                distance: 0,
                ..
            }
        ));
    }

    #[test]
    fn small_negative_variance_is_projected_to_zero() {
        let report = accumulate_specific_heat(
            2.0,
            &SpecificHeatOptions::default(),
            [-1e-11_f64, -1e-11],
            shell(1, 0.0),
            zero_tail,
        )
        .unwrap();

        assert_eq!(report.raw_energy_variance_per_site, -1e-11);
        assert_eq!(report.energy_variance_per_site, 0.0);
        assert_eq!(report.specific_heat_per_site, 0.0);
    }

    #[test]
    fn materially_negative_variance_is_rejected() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions::default(),
            [-1e-6_f64, -1e-6],
            shell(1, 0.0),
            zero_tail,
        )
        .unwrap_err();

        assert!(matches!(error,
            ItebdError::NegativeEnergyVariance { value, .. } if value == -1e-6));
    }

    #[test]
    fn overflowed_onsite_accumulation_is_rejected() {
        let error = accumulate_specific_heat(
            1.0,
            &SpecificHeatOptions::default(),
            [f64::MAX, f64::MAX],
            shell(1, 0.0),
            zero_tail,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::NonFiniteSpecificHeatValue {
                stage: "onsite_contribution",
                distance: 0,
                ..
            }
        ));
    }

    #[test]
    fn overflowed_tail_threshold_is_rejected() {
        let options = SpecificHeatOptions {
            max_distance: 4,
            relative_tolerance: f64::MAX,
            absolute_tolerance: 0.0,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        let error =
            accumulate_specific_heat(1.0, &options, [1.0_f64, 1.0], shell(1, 0.0), |distance| {
                Ok(shell(distance, if distance == 2 { 2.0 } else { 0.0 }))
            })
            .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::NonFiniteSpecificHeatValue {
                stage: "positive_tail_threshold",
                distance: 2,
                ..
            }
        ));
    }

    #[test]
    fn overflowed_specific_heat_is_rejected() {
        let error = accumulate_specific_heat(
            f64::MAX,
            &SpecificHeatOptions::default(),
            [1.0_f64, 1.0],
            shell(1, 0.0),
            zero_tail,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ItebdError::NonFiniteSpecificHeatValue {
                stage: "specific_heat_per_site",
                ..
            }
        ));
    }

    #[test]
    fn tail_shell_uses_the_requested_distance_for_pairing() {
        let options = SpecificHeatOptions {
            max_distance: 4,
            relative_tolerance: 1.0,
            absolute_tolerance: 0.0,
            consecutive_small_shells: 3,
            ..SpecificHeatOptions::default()
        };
        let report =
            accumulate_specific_heat(1.0, &options, [1.0_f64, 1.0], shell(1, 0.0), |distance| {
                if distance == 2 {
                    // These values are paired at even distance, but deliberately labelled odd.
                    Ok(ParityShell::new(1, [1.0, 2.0], [1.0, 2.0]))
                } else {
                    Ok(shell(distance, 0.0))
                }
            })
            .unwrap();

        assert_eq!(report.max_distance, 4);
    }
}
use crate::itebd_error::ItebdError;
use num_complex::Complex64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecificHeatOptions {
    pub max_distance: usize,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub consecutive_small_shells: usize,
    pub reality_tolerance: f64,
}

impl Default for SpecificHeatOptions {
    fn default() -> Self {
        Self {
            max_distance: 200,
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-12,
            consecutive_small_shells: 3,
            reality_tolerance: 1e-10,
        }
    }
}

impl SpecificHeatOptions {
    pub fn validate(&self) -> Result<(), ItebdError> {
        for (name, value) in [
            ("relative_tolerance", self.relative_tolerance),
            ("absolute_tolerance", self.absolute_tolerance),
            ("reality_tolerance", self.reality_tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ItebdError::InvalidTolerance { name, value });
            }
        }
        if self.consecutive_small_shells == 0 {
            return Err(ItebdError::InvalidSpecificHeatCount {
                name: "consecutive_small_shells",
                value: 0,
                minimum: 1,
            });
        }
        if self.max_distance <= self.consecutive_small_shells {
            return Err(ItebdError::InvalidSpecificHeatCount {
                name: "max_distance",
                value: self.max_distance,
                minimum: self.consecutive_small_shells.saturating_add(1),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailStopReason {
    ConsecutiveSmallShells,
}

/// Diagnostics and decompositions for a bidirectional specific-heat estimate.
///
/// The direction and parity decompositions reconstruct [`Self::raw_energy_variance_per_site`].
/// [`Self::energy_variance_per_site`] may differ only when a small negative raw variance is
/// projected to zero within the configured reality tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecificHeatReport {
    /// Unprojected variance reconstructed by the direction and parity decompositions.
    pub raw_energy_variance_per_site: f64,
    /// Validated variance after the documented small-negative projection.
    pub energy_variance_per_site: f64,
    pub specific_heat_per_site: f64,
    pub onsite_contribution: Complex64,
    pub positive_direction_contribution: Complex64,
    pub negative_direction_contribution: Complex64,
    pub parity_a_contribution: Complex64,
    pub parity_b_contribution: Complex64,
    pub max_distance: usize,
    pub stop_reason: TailStopReason,
    pub last_positive_shell_magnitude: f64,
    pub last_negative_shell_magnitude: f64,
    pub max_imaginary_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DirectionPair<T> {
    pub(crate) positive: T,
    pub(crate) negative: T,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ParityShell<T> {
    pub(crate) distance: usize,
    pub(crate) directions: DirectionPair<[T; 2]>,
}

impl<T> ParityShell<T> {
    pub(crate) fn new(distance: usize, positive: [T; 2], negative: [T; 2]) -> Self {
        Self {
            distance,
            directions: DirectionPair { positive, negative },
        }
    }
}

impl<T: CorrelationValue> ParityShell<T> {
    pub(crate) fn zero() -> Self {
        Self::new(0, [T::zero(), T::zero()], [T::zero(), T::zero()])
    }
}

pub(crate) trait CorrelationValue: Copy {
    fn as_complex(self) -> Complex64;
    fn zero() -> Self;
}

impl CorrelationValue for f64 {
    fn as_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }

    fn zero() -> Self {
        0.0
    }
}

impl CorrelationValue for Complex64 {
    fn as_complex(self) -> Complex64 {
        self
    }

    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
}

pub(crate) fn validate_beta(beta: f64) -> Result<(), ItebdError> {
    if !beta.is_finite() || beta < 0.0 {
        return Err(ItebdError::InvalidSpecificHeatBeta { value: beta });
    }
    Ok(())
}

pub(crate) fn accumulate_specific_heat<T, F>(
    beta: f64,
    options: &SpecificHeatOptions,
    onsite: [T; 2],
    first_shell: ParityShell<T>,
    mut shell_at_distance: F,
) -> Result<SpecificHeatReport, ItebdError>
where
    T: CorrelationValue,
    F: FnMut(usize) -> Result<ParityShell<T>, ItebdError>,
{
    validate_beta(beta)?;
    options.validate()?;

    let onsite = [
        checked_value(onsite[0], "onsite", 0, 0)?,
        checked_value(onsite[1], "onsite", 1, 0)?,
    ];
    let onsite_contribution =
        checked_complex((onsite[0] + onsite[1]) * 0.5, "onsite_contribution", 2, 0)?;
    let mut positive_direction_contribution = Complex64::new(0.0, 0.0);
    let mut negative_direction_contribution = Complex64::new(0.0, 0.0);
    let mut parity_contribution = [onsite[0] * 0.5, onsite[1] * 0.5];

    let (mut last_positive, mut last_negative, mut max_paired_shell_imaginary_residual) =
        add_shell(
            first_shell,
            &mut positive_direction_contribution,
            &mut negative_direction_contribution,
            &mut parity_contribution,
            options,
        )?;

    let mut consecutive_small = 0;
    for distance in 2..=options.max_distance {
        let mut shell = shell_at_distance(distance)?;
        shell.distance = distance;
        let (positive_magnitude, negative_magnitude, paired_shell_imaginary_residual) = add_shell(
            shell,
            &mut positive_direction_contribution,
            &mut negative_direction_contribution,
            &mut parity_contribution,
            options,
        )?;
        last_positive = positive_magnitude;
        last_negative = negative_magnitude;
        max_paired_shell_imaginary_residual =
            max_paired_shell_imaginary_residual.max(paired_shell_imaginary_residual);
        let reference_scale = accumulated_reference_scale(
            onsite_contribution,
            positive_direction_contribution,
            negative_direction_contribution,
            distance,
        )?;

        let positive_small =
            last_positive <= tail_threshold(options, reference_scale, "positive", distance)?;
        let negative_small =
            last_negative <= tail_threshold(options, reference_scale, "negative", distance)?;
        if positive_small && negative_small {
            consecutive_small += 1;
            if consecutive_small == options.consecutive_small_shells {
                return finish_report(
                    beta,
                    options,
                    onsite_contribution,
                    positive_direction_contribution,
                    negative_direction_contribution,
                    parity_contribution,
                    distance,
                    last_positive,
                    last_negative,
                    max_paired_shell_imaginary_residual,
                );
            }
        } else {
            consecutive_small = 0;
        }
    }

    Err(ItebdError::SpecificHeatTailNonConvergence {
        max_distance: options.max_distance,
        last_positive,
        last_negative,
        consecutive_small_shells: options.consecutive_small_shells,
    })
}

fn add_shell<T: CorrelationValue>(
    shell: ParityShell<T>,
    positive_total: &mut Complex64,
    negative_total: &mut Complex64,
    parity_total: &mut [Complex64; 2],
    options: &SpecificHeatOptions,
) -> Result<(f64, f64, f64), ItebdError> {
    let positive = [
        checked_value(shell.directions.positive[0], "positive", 0, shell.distance)?,
        checked_value(shell.directions.positive[1], "positive", 1, shell.distance)?,
    ];
    let negative = [
        checked_value(shell.directions.negative[0], "negative", 0, shell.distance)?,
        checked_value(shell.directions.negative[1], "negative", 1, shell.distance)?,
    ];

    for parity in 0..2 {
        let expected = positive[parity ^ (shell.distance & 1)].conj();
        let residual = checked_scalar(
            (negative[parity] - expected).norm(),
            "direction_pair_residual",
            parity,
            shell.distance,
        )?;
        let reference_scale = checked_scalar(
            negative[parity].norm().max(expected.norm()).max(1.0),
            "direction_pair_reference_scale",
            parity,
            shell.distance,
        )?;
        let tolerance = checked_scalar(
            options.reality_tolerance * reference_scale,
            "direction_pair_tolerance",
            parity,
            shell.distance,
        )?;
        if residual > tolerance {
            return Err(ItebdError::SpecificHeatDirectionMismatch {
                parity,
                distance: shell.distance,
                residual,
                tolerance,
            });
        }
    }

    let paired_shell = checked_complex(
        (positive[0] + positive[1] + negative[0] + negative[1]) * 0.5,
        "paired_shell",
        2,
        shell.distance,
    )?;
    let paired_shell_imaginary_residual = paired_shell.im.abs();
    let paired_shell_tolerance = reality_tolerance(
        paired_shell,
        options.reality_tolerance,
        "paired_shell_reality_tolerance",
        shell.distance,
    )?;
    if paired_shell_imaginary_residual > paired_shell_tolerance {
        return Err(ItebdError::NonRealSpecificHeat {
            stage: "paired_shell",
            imaginary: paired_shell.im,
            tolerance: paired_shell_tolerance,
        });
    }

    *positive_total = checked_complex(
        *positive_total + (positive[0] + positive[1]) * 0.5,
        "positive_direction_contribution",
        2,
        shell.distance,
    )?;
    *negative_total = checked_complex(
        *negative_total + (negative[0] + negative[1]) * 0.5,
        "negative_direction_contribution",
        2,
        shell.distance,
    )?;
    for parity in 0..2 {
        parity_total[parity] = checked_complex(
            parity_total[parity] + (positive[parity] + negative[parity]) * 0.5,
            "parity_contribution",
            parity,
            shell.distance,
        )?;
    }
    let positive_magnitude = checked_scalar(
        positive[0].norm().max(positive[1].norm()),
        "positive_shell_magnitude",
        2,
        shell.distance,
    )?;
    let negative_magnitude = checked_scalar(
        negative[0].norm().max(negative[1].norm()),
        "negative_shell_magnitude",
        2,
        shell.distance,
    )?;
    Ok((
        positive_magnitude,
        negative_magnitude,
        paired_shell_imaginary_residual,
    ))
}

fn accumulated_reference_scale(
    onsite: Complex64,
    positive_total: Complex64,
    negative_total: Complex64,
    distance: usize,
) -> Result<f64, ItebdError> {
    checked_scalar(
        onsite
            .norm()
            .max(positive_total.norm())
            .max(negative_total.norm())
            .max((onsite + positive_total + negative_total).norm())
            .max(1.0),
        "tail_reference_scale",
        2,
        distance,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_report(
    beta: f64,
    options: &SpecificHeatOptions,
    onsite_contribution: Complex64,
    positive_direction_contribution: Complex64,
    negative_direction_contribution: Complex64,
    parity_contribution: [Complex64; 2],
    max_distance: usize,
    last_positive_shell_magnitude: f64,
    last_negative_shell_magnitude: f64,
    max_paired_shell_imaginary_residual: f64,
) -> Result<SpecificHeatReport, ItebdError> {
    let total_from_directions = checked_complex(
        onsite_contribution + positive_direction_contribution + negative_direction_contribution,
        "total_direction",
        2,
        max_distance,
    )?;
    let total_from_parity = checked_complex(
        parity_contribution[0] + parity_contribution[1],
        "total_parity",
        2,
        max_distance,
    )?;
    let mut max_imaginary_residual = max_paired_shell_imaginary_residual;
    for (stage, value) in [
        ("parity_a", parity_contribution[0]),
        ("parity_b", parity_contribution[1]),
        ("total_direction", total_from_directions),
        ("total_parity", total_from_parity),
    ] {
        let imaginary = value.im.abs();
        max_imaginary_residual = max_imaginary_residual.max(imaginary);
        let tolerance = reality_tolerance(
            value,
            options.reality_tolerance,
            "reality_tolerance",
            max_distance,
        )?;
        if imaginary > tolerance {
            return Err(ItebdError::NonRealSpecificHeat {
                stage,
                imaginary: value.im,
                tolerance,
            });
        }
    }

    let reconstruction_residual = checked_scalar(
        (total_from_directions - total_from_parity).norm(),
        "reconstruction_residual",
        2,
        max_distance,
    )?;
    let reconstruction_scale = checked_scalar(
        total_from_directions
            .norm()
            .max(total_from_parity.norm())
            .max(1.0),
        "reconstruction_reference_scale",
        2,
        max_distance,
    )?;
    let reconstruction_tolerance = checked_scalar(
        options.reality_tolerance * reconstruction_scale,
        "reconstruction_tolerance",
        2,
        max_distance,
    )?;
    if reconstruction_residual > reconstruction_tolerance {
        return Err(ItebdError::SpecificHeatDirectionMismatch {
            parity: 2,
            distance: max_distance,
            residual: reconstruction_residual,
            tolerance: reconstruction_tolerance,
        });
    }

    let raw_energy_variance_per_site = total_from_directions.re;
    let negative_tolerance = reality_tolerance(
        total_from_directions,
        options.reality_tolerance,
        "negative_variance_tolerance",
        max_distance,
    )?;
    let energy_variance_per_site = if raw_energy_variance_per_site < 0.0 {
        if raw_energy_variance_per_site.abs() <= negative_tolerance {
            0.0
        } else {
            return Err(ItebdError::NegativeEnergyVariance {
                value: raw_energy_variance_per_site,
                tolerance: negative_tolerance,
            });
        }
    } else {
        raw_energy_variance_per_site
    };
    Ok(SpecificHeatReport {
        raw_energy_variance_per_site,
        energy_variance_per_site,
        specific_heat_per_site: checked_scalar(
            beta * beta * energy_variance_per_site,
            "specific_heat_per_site",
            2,
            max_distance,
        )?,
        onsite_contribution,
        positive_direction_contribution,
        negative_direction_contribution,
        parity_a_contribution: parity_contribution[0],
        parity_b_contribution: parity_contribution[1],
        max_distance,
        stop_reason: TailStopReason::ConsecutiveSmallShells,
        last_positive_shell_magnitude,
        last_negative_shell_magnitude,
        max_imaginary_residual,
    })
}

fn checked_value<T: CorrelationValue>(
    value: T,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<Complex64, ItebdError> {
    checked_complex(value.as_complex(), stage, parity, distance)
}

fn checked_complex(
    value: Complex64,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<Complex64, ItebdError> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(ItebdError::NonFiniteSpecificHeatValue {
            stage,
            parity,
            distance,
            real: value.re,
            imaginary: value.im,
        });
    }
    Ok(value)
}

fn checked_scalar(
    value: f64,
    stage: &'static str,
    parity: usize,
    distance: usize,
) -> Result<f64, ItebdError> {
    if !value.is_finite() {
        return Err(ItebdError::NonFiniteSpecificHeatValue {
            stage,
            parity,
            distance,
            real: value,
            imaginary: 0.0,
        });
    }
    Ok(value)
}

fn tail_threshold(
    options: &SpecificHeatOptions,
    running_scale: f64,
    direction: &'static str,
    distance: usize,
) -> Result<f64, ItebdError> {
    let reference_scale = checked_scalar(
        running_scale.max(1.0),
        match direction {
            "positive" => "positive_tail_reference_scale",
            "negative" => "negative_tail_reference_scale",
            _ => unreachable!("tail direction is fixed by the accumulator"),
        },
        2,
        distance,
    )?;
    let relative_component = checked_scalar(
        options.relative_tolerance * reference_scale,
        match direction {
            "positive" => "positive_tail_threshold",
            "negative" => "negative_tail_threshold",
            _ => unreachable!("tail direction is fixed by the accumulator"),
        },
        2,
        distance,
    )?;
    checked_scalar(
        options.absolute_tolerance + relative_component,
        match direction {
            "positive" => "positive_tail_threshold",
            "negative" => "negative_tail_threshold",
            _ => unreachable!("tail direction is fixed by the accumulator"),
        },
        2,
        distance,
    )
}

fn reality_tolerance(
    value: Complex64,
    relative_tolerance: f64,
    stage: &'static str,
    distance: usize,
) -> Result<f64, ItebdError> {
    let reference_scale = checked_scalar(value.norm().max(1.0), stage, 2, distance)?;
    checked_scalar(relative_tolerance * reference_scale, stage, 2, distance)
}
