# Native SVD repairs

Faer0.24.4 is retained as required source for this standalone purification crate.
Cargo.toml selects vendor/faer with [patch.crates-io]. No external project path
or shared Cargo cache modification is used. Tensor4all and tenferro Git pins
remain unchanged.

Upstream package checksum: 5ab6df3dd147fe8d702a288b95bcd8fcc499ab572fc80da6828f60cd4d524d67.
Upstream VCS:0539947ffb757a739d7e703a7d2fa0c792a909c1. Retain LICENSE and all
COPYING notices, including COPYING.LAPACK.BSD for the DLASV2 adaptation.

The sole library change from the published0.24.4 sources is
src/linalg/svd/bidiag_svd.rs: the reviewed Infinite_MST terminal2x2 SVD repair
and its five small regressions, plus the secular-root bracket repair below.
The terminal2x2 repair handles the remaining2x2 bidiagonal block
analytically before general shifted QR, with scaled ratios and native rotations.
Sign correction, sorting, tolerances, iteration budgets, backends and public APIs
are preserved. This common path also serves divide-and-conquer leaves and the
real bidiagonal stage of complex SVD. No fallback or normal matrix is added.
The unrelated Infinite_MST Schur patch is not included.

Source was copied from Infinite_MST at21dc3e69f85908a926b4ea76f88037009f099268,
then its Schur file restored from the pristine registry source. All library files
were compared against that pristine source; only bidiag_svd.rs differs.
Packaged examples, optional data, cache markers and broad upstream development
lockfile are omitted. The normalized manifest retains only the test dependencies
needed by selected library tests; production dependencies/features are unchanged.

Verification uses the application public-API clustered3x3 regression, the six
imst_svd_regressions (including the bracket regression), and selected
real/complex/special SVD tests. Follow with
the standalone routine/doc gates and native checkpoint I/O. A passing control
does not prove the earlier thermal failure had the same root cause.
Keep the source, license/provenance and regressions until a reviewed upstream
replacement passes equivalent checks. Vendor source is not build cleanup debt.

The standalone Cargo.lock retains production locked versions and the three
small test-only dependencies needed by selected library tests. On ARM64 the
unmodified upstream matmul module emits one dead_code warning for its x86-only
PAR_THRESHOLD_MNK constant. The existing verification wrapper records this exact
identity in scripts/verify_known_warnings.json; no unrelated library edit or
blanket warning suppression is used.

## Secular-root bracket repair (2026-10-29)

A captured finite963x963 AKLT gate matrix made unmodified and terminal2x2-
patched Faer return successful SVD with NaN vectors. Divide-and-conquer's
secular root finder accepted out-of-domain secant candidates into the fallback
bracket, reversed its bounds, skipped bisection and returned an invalid midpoint.
The resulting below-pole root produced a negative squared reconstruction weight
and sqrt NaN. This is distinct from the terminal2x2 convergence defect.

The root finder now rejects nonfinite/out-of-domain proposals before evaluating
or caching them, checks opposite-side proposals, and intersects candidate bounds
atomically with the valid fallback bracket. An invalid/outside final secant
estimate uses the existing bisection. Tolerances, iteration limits, dependencies,
backend choices and the earlier repair are unchanged; no absolute-value/clamp
repair masks a negative reconstruction product. A small two-pole secular
regression has independently derived polynomial roots and protects both inner
and outer intervals. The captured matrix is only a scoped extended reproduction
input, with reconstruction/orthogonality identities as independent oracles.

Before/after timing uses the same release-built
`tests/support/faer_svd_speed.rs`, one captured and one deterministic healthy
963x963 matrix, and selected1/24-thread controls. Invalid baseline timings are
labelled explicitly; they are not valid-result speedups. The detailed evidence
and limitations are in the
[repair investigation](../../docs/investigations/2026-10-29-faer-secular-bracket.md).
