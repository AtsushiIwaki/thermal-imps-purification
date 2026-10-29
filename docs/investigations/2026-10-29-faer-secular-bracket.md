# Faer secular-root bracket repair

2026-10-29: The bounded repair preserves the valid pole interval when secant
extrapolation fails. A small independent polynomial-root regression reproduces
the previous below-pole return, and the captured finite963x963 matrix now
factorizes with finite, accurate singular vectors under the selected24-way
parallelism setting. Implementation is on the local repair branch, pending
integration. No thermal trajectory or reference publication is part of this task.

## Cause and change

The earlier diagnostic isolated a secant estimate outside the active pole
interval. Its candidates were still cached and then merged independently into
the fallback bracket. This reversed the bracket, skipped bisection through its
negative width, and returned an invalid midpoint. The resulting below-pole root
made `perturb_col0` take a square root of a negative product. Successful SVD then
returned NaN vectors. That diagnosis applies to the captured finite matrix;
identity with the original production step402 failure remains unproved.

`vendor/faer/src/linalg/svd/bidiag_svd.rs` now:

- Rejects nonfinite/out-of-domain secant proposals before evaluation or caching.
- Checks opposite-side proposals, the current seed domain and seed-function finiteness.
- Intersects candidate bounds atomically with the existing valid bracket.
- Uses the existing bisection for an invalid final secant estimate.

The repair preserves dependencies, tolerances, iteration controls, backend
selection and the retained terminal2x2 kernel. It does not clamp a negative
reconstruction weight or substitute a decomposition/backend. It does not claim
to qualify arbitrary nonfinite callback behavior in other root-finder stages.

## Independent regression

The function `1 - 2/sigma^2 - 0.01/(sigma^2-1)` is increasing on each relevant
pole interval. Setting z=sigma² gives `z² - 3.01z + 2 = 0`. Independently derived
positive roots are1.4211987153889648 and0.995085027209613. The wide outer bracket
[1,1e6] previously returned0.9979070647721313. The regression asserts the
correct bracket and analytical root, exercises both shift directions, and adds
an explicit last-root branch case. A preliminary bounded64-case search was
reduced to this one small retained regression. RED and GREEN output is retained
by the controlling Infinite_MST task; old matrix outputs are not test oracles.

## Fixed-matrix speed and correctness

`tests/support/faer_svd_speed.rs` is a direct-rustc control linked to either
release Faer build with the same source/compiler/features. It compares the
captured finite963x963 input and a deterministic, dense, diagonally dominated
963x963 input. Their checksums match across versions. Input loading/generation,
process startup, finiteness checks and reconstruction/orthogonality are excluded
from the measured `thin_svd` interval; normal SVD allocations are included.

Each condition uses ABBA blocks (before/after/after/before), one warmup and five
measured calls per block, giving ten timed calls per version/condition and96
factorizations total. RAYON_NUM_THREADS is pinned to the selected1 or24 in fresh
processes; other four thread settings are1. Faer uses Seq or rayon(selected).
These are controlled parallelism/pool settings, not a measurement of active
worker utilization or a change to purification production defaults. No compiler
or other task numerical work ran concurrently. Total benchmark wall36.168s.

| Input | Parallelism setting | Before median ms (IQR) | After median ms (IQR) | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Captured | 1 | 304.901 (302.670–307.979) | 302.906 (301.891–304.188) | Both valid; elapsed−0.65% |
| Captured | 24 | 369.147 (364.378–376.193) | 376.131 (370.717–390.176) | Before invalid; after valid; no valid-result speed ratio |
| Healthy | 1 | 328.128 (327.340–329.777) | 327.361 (326.653–328.533) | Both valid; elapsed−0.23% |
| Healthy | 24 | 401.068 (389.758–409.027) | 396.635 (389.994–413.009) | Both valid; elapsed−1.11% |

Every repaired measured factorization was finite. Captured24 maximum relative
reconstruction error across the checked block endpoints is3.92964e-15, with
U/V orthogonality Frobenius residuals at most1.253e-13/1.443e-13. Captured1
already worked before the repair; both versions reconstruct to4.45967e-15.
Healthy before/after identities match within the reported checks. The healthy
medians and overlapping IQRs show no apparent slowdown in this limited sample;
they do not establish a statistically significant speedup, other-size behavior,
optimal thread count or end-to-end purification time. The captured24 elapsed
increase of1.89% is a comparison against invalid baseline work.

## Verification ownership

Focused RED/GREEN, final Native/vendor/HDF5 gates, source identities, peer review,
protected output hashes and integration disposition are recorded under the
controlling Infinite_MST
`codes/imst_rs/results/records/aklt-purification-dmax256-resume-svd-repair/bracket-repair/`.
Daily Native checks include actual checkpoint save/read; the controlling main
HDF5 target additionally reads Git-managed fixed fixtures. Full unchanged main
and Python suites, parameter sweeps and thermal retries are unselected. The
captured matrix is a bounded reproduction input; correctness is assessed through
factorization/orthogonality identities rather than historical reference values.

Final software gates passed495 Native tests/42 binaries (16 optional ignored,
one known ARM64 warning, no unknown warning), six vendor regressions including
the explicit last-root case, and one each real/complex/special SVD test. Doc tests
discovered0 examples. The controlling main native HDF5 target passed14 tests
with4 optional exporters ignored. Native routine wall285.710s includes build and
execution; its existing wrapper does not expose their split. Separate Faer
release library builds took25.87s before and26.195s after; the final vendor-test
process took74.602s, predominantly74s compilation. Main HDF5 wall81.660s includes
77s build and2.41s execution. Full unchanged Python/main suites were unselected.
