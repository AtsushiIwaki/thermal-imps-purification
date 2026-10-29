# Native terminal 2x2 SVD repair

Faer0.24.4 is retained as required source for this standalone purification crate.
Cargo.toml selects vendor/faer with [patch.crates-io]. No external project path
or shared Cargo cache modification is used. Tensor4all and tenferro Git pins
remain unchanged.

Upstream package checksum: 5ab6df3dd147fe8d702a288b95bcd8fcc499ab572fc80da6828f60cd4d524d67.
Upstream VCS:0539947ffb757a739d7e703a7d2fa0c792a909c1. Retain LICENSE and all
COPYING notices, including COPYING.LAPACK.BSD for the DLASV2 adaptation.

The sole library change from the published0.24.4 sources is
src/linalg/svd/bidiag_svd.rs: the reviewed Infinite_MST terminal2x2 SVD repair
and its five small regressions. It handles the remaining2x2 bidiagonal block
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

Verification uses the application public-API clustered3x3 regression, the five
imst_svd_regressions, and selected real/complex/special SVD tests. Follow with
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
