# Investigation: complex heat and canonical matrix decompositions

## Latest conclusion — 2026-09-08

- Status: the PSD-factor and complex gauge-SVD reconstruction defects are independently reproduced and repaired; the full Linux routine suite and fixed scientific heat driver pass. The latter returns 28 valid rows, with both models selecting the .1 to .05 refinement window.
- Established: the original XX test fails at `45007b4c26bc7e13525088fa0d1a359324b8611a` with parity-A imaginary residual `4.809681007551957e-9` against `1e-10`. A weakly coupled 3x3 positive matrix reproduces PSD-factor reconstruction error `1.412799348810722e-3`; a captured 10x10 complex gauge matrix independently reproduces SVD reconstruction error `6.27143088881496e-3`.
- Repair: use the already pinned tensor4all Hermitian eigensolver in both canonicalizers, and its full-rank SVD factorization in the complex canonicalizer. Keep floors, dependency pins, Schmidt-slot counts, physical/ancilla conventions, and all pre-existing heat assertions unchanged.
- Not established: universal conditioning, off-canonical heat accuracy, or arbitrary-model/low-temperature convergence. The real canonicalizer's SVD and other uses of nalgebra are outside this demonstrated complex-SVD repair.
- Superseded: an eigen-only repair passed the original test but failed the smaller-step control. A first SVD adapter zeroed U columns for null singular slots; review and a minimal state regression exposed a normalization change, corrected by retaining full U.
- Remaining platform verification: hosted CI and macOS execution of this repair; current project priority remains in [STATE.md](../../STATE.md).

## Reproduction and causal separation

The starting source is `45007b4c26bc7e13525088fa0d1a359324b8611a`. The user requested repair of
the numerical failure independently of the preceding HDF5 process-launch repair. The only
pre-existing working change was the mode of `scripts/smoke.py`, which was preserved.
The [task plan](../superpowers/plans/2026-09-08-complex-heat-reality.md) records path ownership,
caller contracts and scope expansions.

The Linux command is:

```sh
CARGO_INCREMENTAL=0 cargo test --locked --offline --test specific_heat twisted_xx_matches_untwisted_report_and_exact_heat -- --exact --nocapture
```

It selects one test and fails with the recorded imaginary residual. The
[hosted run at the same commit](https://github.com/AtsushiIwaki/thermal-imps-purification/actions/runs/34230916826)
also passes the repaired beta-zero checkpoint regression and fails this heat test with the same
number; macOS completes successfully. These are observations of the pre-numerical-repair commit.

Local evidence is retained under ignored `results/complex-heat-reality-20260908/`, including the
`run.sh` runner, separate exit files, raw output and diagnostic source. The primary agent owns
verification. Commands use the existing WSL Ubuntu GNU/Linux toolchain/cache, Rust 1.96.1,
`CARGO_INCREMENTAL=0` and `CARGO_BUILD_JOBS=4`. Dependencies are resolved from the pinned lock,
without local source patches. Formatting uses installed Windows `rustfmt +stable` 1.9.0;
compilation uses the repository's Rust 1.96.1.

### PSD factors

For phase-twisted XX at beta .6, dtau .05, cutoff 1e-12, cap32, diagnostic output found
eigenpair residual up to `3.4731551151288608e-3` despite Hermiticity residual near `4e-18`.
The input's Hermiticity was not the source of that discrepancy.

The analytic regression uses diagonal `[1, 1e-6, 1e-3]`, with entries (1,2) and (2,1)
coupled by `1e-18` (and a conjugated imaginary component in the complex variant). All eigenvalues
are above the existing 1e-12 floor. The former nalgebra factor fails `F F^H = rho` by
`1.412799348810722e-3`; the replacement passes reconstruction at `1e-14` and inverse whitening
at `1e-12`. A separate Fourier-basis PSD matrix exercises materially nonreal conjugation.
Both f64 and Complex64 production paths have direct reconstruction tests.

Changing only the complex eigensolver removed the original imaginary failure but exposed
real/complex variance disagreement near `5.6e-7`: the real canonicalizer used the same flawed
PSD-factor routine. Updating both removed that mismatch without changing its `1e-8` assertion.

### Gauge SVD

An eigen-only repair is insufficient. At dtau .025 with the same beta/cutoff/cap, the complex
gauge SVD reconstructs its own input with error `6.27143088881496e-3`; the final parity imaginary
residual is `-3.220737922687396e-6`. Recanonicalizing the final state does not undo this physical
error. The new smaller-step integration test rejects it with the original default reality bound.

[canonical-svd-xx.json](../../tests/fixtures/canonical-svd-xx.json) stores the captured 10x10
`Y*X` matrix as 100 column-major `[real, imaginary]` pairs. It was captured during the dtau .025
diagnostic trajectory after the eigensolver repair and before replacing the gauge SVD. It is
project-generated numerical data, not an upstream source extract. Its SHA-256 is
`06e6f7f1a2fe81ee5ec4c9a98421231574292aa34a01a5db6325586d059f847f`.
The regression directly checks reconstruction below `1e-14` and U/V orthogonality below `1e-12`.

The full-rank tensor4all factorization returns U and S*V^H (`Canonical::Left`). Dividing
nonzero rows of the second factor by the corresponding singular value recovers V^H. Exactly
zero rows remain zero and become incoming Gamma rows weighted by zero Schmidt values. Full U
must remain intact: an outgoing zero-Schmidt column still contributes to `site_gram_scalar`.
No new truncation rule is applied, including to a singular value of 1e-16 or an exact zero slot.

The first adapter instead used `Canonical::Right` and zeroed U columns. Full verification caught
a TFIM free-energy error of about .256. Independent review identified the unweighted Gram
normalization interaction; a normalized product state with BA weights `[1,0]`, AB weight `[1]`,
and an orthogonal inactive outgoing column reproduced spurious log normalization
`-0.3465735902799727`. The final adapter passes its zero-log, unit-cell norm, direct periodic
norm, product-observable and retained-dimension checks. The TFIM thermodynamics regression then
passes with free-energy error `4.048e-5`. This intermediate failure was fixed, not waived.

Unweighted full-identity Gram diagnostics on a zero-Schmidt slot can legitimately report one;
they must not be interpreted as an active-subspace physical error. Existing nonreal-gauge tests
also check direct periodic norm/energy preservation, idempotence and canonical Gram residuals
on their full-support fixtures.

## Verification evidence

| Check | Recorded result | Evidence basename |
| --- | --- | --- |
| Original exact XX test | Failed with parity imaginary 4.809681e-9 | `baseline.log` |
| Analytic PSD regression, old solver | Failed with reconstruction 1.412799e-3 | `red-factors.log` |
| Analytic PSD regression, tensor backend | Passed | `green-factors.log` |
| Smaller-step XX after eigen-only repair | Failed with parity imaginary -3.220738e-6 | `red-fine-heat.log` |
| Captured SVD matrix, old solver | Failed with reconstruction 6.271431e-3 | `red-svd.log` |
| Captured SVD matrix, tensor backend | Passed | `green-svd.log` |
| Initial adapter full routine | Failed TFIM free-energy assertion | `zero-u-routine.log` |
| Zero-support product state, initial adapter | Failed with spurious log -0.346574 | `red-zero-support.log` |
| Zero-support product state, full-U adapter | Passed | `green-zero-support.log` |
| TFIM free-energy regression, full-U adapter | Passed | `green-thermodynamics.log` |
| Final Linux routine (`cargo test --all-targets`) | 32 binaries; 422 passed, 0 failed, 16 ignored; zero warnings | `final-routine.log` |
| Linux documentation tests | Passed; zero doctests selected | `final-doc.log` |
| Linux Python harness | 54 passed | `final-python.log` |
| Fixed release complex-heat driver | 1 selected test passed; 28 valid rows, 0 blocked; 2 selected windows | `final-heat-evidence.log` |

The full routine includes the contraction AST boundary audit, independent real/complex
correlation oracles, nonreal virtual-gauge and canonical norm/energy/idempotence checks,
thermodynamics, exact checkpoint round trips and restart/CLI tests. No existing assertion was
removed or weakened. The separate scientific driver is recorded below.

The final read-only code review found no remaining substantive issues after the zero-support
correction. It inspected actual source, the fixture and investigation, and did not run builds.
The Linux Python harness passed all 54 tests. An additional Windows Python attempt failed two
tests and errored in one: the unchanged harness assumes executable POSIX shell fixtures,
POSIX-valid unusual filenames, and POSIX path resolution. No harness source changed in this
repair; this is not a claim of Windows full-suite coverage.

The final source/fixture/manifest/lock fingerprints are retained in
`tested-source-sha256.json` alongside these logs. The tested working change is based on the
captured commit, not a claim that its unmodified contents passed. `Cargo.toml` and `Cargo.lock`
remain unchanged. The scientific command is the existing documented driver, with no modified
thresholds or row selection:

```sh
CARGO_INCREMENTAL=0 cargo test --locked --offline --release --test specific_heat_benchmark benchmark_complex_specific_heat_validation -- --ignored --exact --nocapture
```

### Fixed scientific heat grid

The command selected one test, passed it and exited 0. The unchanged driver evolved
infinite-temperature purifications with second-order steps and canonicalization every step:
phase-rotated TFIM (`J=1,g=.7,beta=1`) and phase-twisted XX (`gamma=0,h=.4,beta=.6`, phase
`pi/5`). Each used `dtau={.1,.05,.025,.0125}` for primary cutoff `1e-14`, cap128;
cutoff sensitivity `1e-13`, cap128; and cap sensitivity `1e-14`, cap64. Those 24 rows plus
strict/relaxed tail checks at the two selected endpoints gave 28 valid reports, zero blocked
rows, maximum imaginary residual `2.288893903873734e-16`, maximum direction-total difference
`6.6058269965196814e-15` and maximum observed bond21, below both caps.

The fixed rule selects the first qualifying adjacent window, `.1 -> .05` for both models:

| Model | Coarse heat error | Fine heat error | Order | Cutoff change / fine error | Cap change / fine error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase-rotated TFIM | 1.7328346811212869e-4 | 4.4788329350453626e-5 | 1.9519392642464291 | 8.9411439235852804e-4 | 0 |
| Phase-twisted XX | 3.0509143623236312e-5 | 7.7780217409006980e-6 | 1.9717665122698715 | 9.5824818183244469e-4 | 0 |

The gates require decreasing error, order `>=1.6` and cutoff/cap fractions `<=.05`. Independent
symmetric energy derivatives use one and two time steps on each side. At the selected endpoints,
their mutual difference is `7.6717879015855694e-3` (TFIM) and `2.8128509568930982e-4` (XX),
below the driver's limits `1.0183017979405184e-2` and `3.9291806062974333e-4`. These limits
are the larger derivative error against the exact heat; passing is not a claim that the
derivative and variance estimators agree to machine precision. Exact references use nk32000;
this run did not perform a new reference-resolution sensitivity study.

Strict tail settings (relative `3e-7`, absolute `3e-13`, max distance300) shift the selected
heat by zero. Relaxed settings (`3e-6`, `3e-12`, distance120) shift TFIM by
`2.6503738093097695e-10` and XX by `3.0645322390832064e-10`, below their selected exact-reference
errors. The default is `1e-6`, `1e-12`, distance200; all use three consecutive small shells
and the unchanged reality tolerance `1e-10`.

The `.05 -> .025` windows are also candidates (orders `1.798512860993063` and
`1.7873730553002734`). The finest `.025 -> .0125` windows are rejected with orders
`1.5114892569028171` and `.7557487761065906`, despite finite reality-valid reports. This is
limited refinement evidence, not arbitrary precision, low-temperature convergence, variation
of initial conditions or general Hamiltonian qualification. The infinite translation-invariant
estimator has no finite-system-size extrapolation in this driver. Earlier matrix CLI, RDM and
AKLT scientific grids were not rerun; their recorded results remain historical evidence.

Raw scientific log SHA-256:
`f2376b5d75affd1bf79e817b5496cbbf7e91491229bfc1174e3dd93cb4454ba7`.

The final documentation review checked these claims against the unchanged driver and observed
log, including selection/rejection, counts and hash, and found no substantive issues. Final
source fingerprints still match the tested files, changed-document local links resolve,
owned complex-source/test formatting and `git diff --check` pass, and the staged view is empty.
The only unrelated difference remains the pre-existing `scripts/smoke.py` mode. Temporary
diagnostics were removed from production/tests and retained only as ignored evidence. Changes
were uncommitted and unpublished at this verification checkpoint.
