# Numerical qualification

On 2026-09-07, all four fixed release qualification tests passed on committed code
`e9e732a99f6e2937dcd4d112469bc93e451c0b37`: 16 matrix CLI rows, 32 valid checkpoint/RDM rows,
eight second-order AKLT normalization trajectory pairs, and one first-order AKLT control.
There were no failed or blocked required rows and no relaxed thresholds. This establishes the
specific refinement, persistence and normalization checks below, within their measured windows.
The broader [known limitations](limitations.md) still apply.

## Extraction compatibility and local build coverage

The source baseline `55ed6ee4c1940dbfd2be13374ce3caf45169f515` and extracted code
`e9e732a99f6e2937dcd4d112469bc93e451c0b37` passed eight fixed compatibility cases and
536 comparison groups. These covered first/second-order real TFIM, phase-rotated complex TFIM,
AKLT projector, off-cadence observations/checkpoints, and source-file restart in the extracted
build. Maximum absolute and scaled differences were 0 for all compared physical and CLI fields,
with fixed tolerance `abs(a-b)<=1e-10*max(1,abs(a),abs(b))`. Loading the same checkpoint preserved
stored components bit-exactly; fresh trajectory Gamma gauges/UUIDs were not compared. Explicitly
complex storage with an all-real Hamiltonian retained its backend, and incompatible historical
AKLT/projector restart was rejected. Detailed compatibility harnesses and original command
records remain in the private source extraction evidence; they are not public dependencies.

On 2026-09-07, commit `a2714c2c64e36477979d55e8f656885688876888` passed an independent
local macOS arm64 build from a `git clone --no-hardlinks` checkout, an observed empty dedicated
Cargo home, and an empty target directory. The recorded locked fetch preceded a successful
release all-target build, including vendored static HDF5. Metadata found 233 packages (one local
root, 209 registry and 23 Git packages), all under that clone or its dedicated cache, with no
source-repository or `refs/` dependency. Lockfile bytes remained unchanged. Complete original
stdout/stderr, argv, statuses and initial empty-cache/clone proof are retained privately.

That cold-environment run passed routine verification (31 binaries; 409 passed, zero failed,
16 ignored; zero warning categories), documentation tests (zero selected, zero failed), 49 Python
tests, and real/complex smoke cases with two finite records each through beta .2. A separate local
execution of the workflow's verification block also passed. These are macOS 26.6.2 arm64 local
results: **Linux, GitHub-hosted Linux and GitHub-hosted macOS are unverified**. The CI matrix is
prepared, but no remote or GitHub Actions run exists.

Finalization changed only smoke-path serialization, its regression test, and public documentation.
The final Python suite passed 50 tests; both focused README Rust tests passed; ordinary and
quote/backslash/Unicode/tab/DEL output-directory smoke runs each passed real and complex solver
cases through beta .2. The smoke regression checks config output paths and argument-list process
invocation; real solver runs check the production TOML/JSON parsers. No numerical kernels,
measurement drivers, library/config contracts, or dependency resolutions changed, so the accepted
routine, compatibility and scientific results above were retained without repeating their grids.

## Reproducibility

The tested code was extracted from `55ed6ee4c1940dbfd2be13374ce3caf45169f515`. All three
measurement drivers and their four support files were compared byte-for-byte with that source
after the sole import substitution `imps_purification` → `thermal_imps_purification`.
No production kernel or measurement-driver change was made for this qualification.
[summary.json](validation/summary.json) records driver hashes, manifest/lock hashes, actual argv,
statuses, raw-output hashes, and full parsed CLI/refinement/sensitivity rows.
[checkpoint-rdm.json](validation/checkpoint-rdm.json) retains all 32 rows and their diagnostics.

Environment: Rust 1.96.1 (`31fca3adb283cc9dfd56b49cdee9a96eb9c96ffd`), LLVM 22.1.2,
`aarch64-apple-darwin`, macOS 26.6.2 build 25G83. Every Cargo command used
`CARGO_INCREMENTAL=0`. Compilation reused the extraction baseline's existing target cache and
cached dependencies. This is warm-cache numerical evidence, not a cold-cache build or CI result.
The lock retains tensor4all `880032daea2a84d7ddfe84a2594a6d160631afc2` and tenferro
`f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266`.

Run from the repository root, sequentially. Use a new RDM output filename for each run; its driver
opens `ITEBD_EVIDENCE_PATH` exclusively before measuring. Capture stdout/stderr independently.

```sh
CARGO_INCREMENTAL=0 cargo test --release --test solve_matrix_evidence --test itebd_checkpoint_rdm_evidence --test aklt_projector_evidence -- --list
CARGO_INCREMENTAL=0 cargo test --release --test solve_matrix_evidence matrix_cli_refinement -- --ignored --exact --nocapture
mkdir -p results
ITEBD_EVIDENCE_PATH="$PWD/results/qualification-rdm-001.json" CARGO_INCREMENTAL=0 cargo test --release --test itebd_checkpoint_rdm_evidence checkpoint_rdm_refinement_matrix -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test aklt_projector_evidence projector_old_normalization_grid -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test aklt_projector_evidence projector_old_first_order_control -- --ignored --exact --nocapture
```

The initial [list output](validation/list.log) contains seven tests across the three targets.
Each of the four exact measurement commands selected **one test**, passed it, and exited 0.
The RDM output filename was never used before the run. All measurement stdout/stderr and command
records remain in ignored local evidence. The public RDM derivative removes the host hostname and
process ID and substitutes OS version; every numerical row, setting, refinement and sensitivity
is unchanged. Public command metadata omits private working/cache/output absolute paths.
Published matrix/AKLT stdout removes only trailing blank lines; list stdout is byte-identical to
the raw stdout. Output hashes
identify both the public derivatives and private originals without revealing private paths.
Elapsed times in the command metadata are execution provenance only; no performance claim or
wall-clock threshold is made.

## Phase-TFIM matrix CLI refinement

Parameters: `J=1`, `g=0.7`, `beta=1`, phase rotation `U=diag(1,i)` applied to Hamiltonian and
local Pauli-X observable; second-order Strang evolution; canonicalization every step;
`dtau={0.1,0.05}`, cutoff `{1e-13,1e-14}`, cap `{64,128}`, TOML and JSON. This gives 16 CLI
rows plus independent direct evolution for each numerical setting. Primary selection was
cutoff `1e-14`, cap128, fixed before execution.

All 32 observable refinement comparisons had positive decreasing errors. Primary results were
identical between TOML and JSON:

| Observable | Error at dtau=.1 | Error at dtau=.05 | Primary order |
| --- | ---: | ---: | ---: |
| u | 2.7112819632e-5 | 6.8586732400e-6 | 1.9829737261 |
| c | 1.7326794164e-4 | 4.4772802885e-5 | 1.9523102067 |
| f | 1.6702096382e-4 | 3.9718788683e-5 | 2.0721356652 |
| local | 2.8600180091e-4 | 7.1797570034e-5 | 1.9940173090 |

The required primary order is `>=1.6`. Maximum cutoff/cap sensitivity relative to primary
exact-reference error was **0.019197167510241763**, below the fixed `<=0.05` gate. Maximum
direct/CLI absolute and scaled differences and TOML/JSON absolute difference were all **0**;
the driver gate is `abs(a-b) <= 1e-10 * max(1,abs(a),abs(b))` for u/c/f/local. CLI and direct
reported bonds also matched. Maximum observed bond was 10, below both caps.

The exact-reference 16,384→32,768-point diagnostics changed u by `1.8873791418627661e-11`,
c by `3.501038348119323e-8`, f by `9.992007221626409e-15`, and local by
`1.4876988529977098e-10`. These are resolution diagnostics, not rigorous error bounds.
All measured rows and gates are in [matrix.log](validation/matrix.log).

## Checkpoint and RDM refinement

The 32 rows use real and phase-rotated TFIM with the same `J=1,g=.7,beta=1`, second order and
canonicalization every step; `dtau={.1,.05}`, cutoff `{1e-13,1e-14}`, cap `{64,128}`, and
fixed-point tolerance `{1e-10,1e-12}`. The primary selection is cutoff `1e-13`, cap128,
tolerance `1e-12`. RDM options retain at most 10,000 iterations, trace/Hermiticity/positivity
thresholds `1e-10`, 1,048,576 output elements and 16,777,216 intermediate elements.
Each row checks A/B parities at lengths 1 and 2, with checkpointing at an intermediate step.

All 32 rows were valid. Stored state/progress component comparisons were bit-exact, round-trip
RDM Frobenius discrepancy was **0**, and every recorded continuation discrepancy (energy, free
energy, log norm, Schmidt values, RDM) was **0**, against the fixed continuation `<=1e-10` gate.
All 32 RDM observable refinement comparisons decreased. Primary orders were:

| Backend | Energy | Local observable |
| --- | ---: | ---: |
| real | 1.9808223270800112 | 1.9933153783484847 |
| complex | 1.98082232694074 | 1.9933153783192084 |

These exceed `>=1.6`; maximum sensitivity fraction was **0.003822464152019367**, below the
strict `<.05` gate. Maximum direct/RDM disagreement was `8.41826608422025e-11` for energy and
`3.5181768609504616e-10` for local. The gate is
`<=max(1e-10,.1*direct_error)`, not an absolute `1e-10` bound; maximum disagreement/gate ratio
was `3.313264706817879e-5`. Maximum RDM errors over the full grid were `2.7176380604787553e-5`
and `2.8600147649221475e-4` respectively. Maximum bond was 10.

Maximum observable imaginary part was 0, normalized trace residual `2.220446049250313e-16`,
Hermiticity residual `1.7772239894833365e-16`, and fixed-point residual
`5.054746793120586e-11` across the mixed-tolerance grid. Minimum RDM eigenvalue was
`0.031208006521612423`. The central-difference step diagnostic (`1e-4` versus `5e-5`, nk16384)
shifted energy by `8.504308368628699e-10` and local by `2.353672812205332e-10`; primary errors
were above those diagnostic floors. This is not a rigorous reference error bound or proof of
unique transfer fixed points.

## AKLT normalization controls

The measured identity is `h_old=2*P2-2*I/3`, with old beta and dtau half the projector values:
`u_projector=u_old/2+1/3`, `f_projector=f_old/2+1/3`; heat, Sz and normalized RDMs agree.
Second order uses projector `dtau={.05,.025}`, cutoff `{1e-10,1e-12}`, cap `{16,32}` and
beta endpoints `{.2,.4}`: eight trajectory pairs, 16 trajectories, 32 endpoint samples.
First order uses projector dtau `.05`, cutoff `1e-12`, cap32, beta `.2`: one trajectory pair,
two trajectories, two endpoint samples. Canonicalization is performed every step.

| Mapping field | Maximum second-order difference | First-order difference | Fixed tolerance |
| --- | ---: | ---: | ---: |
| matched beta | 0 | 0 | 1e-14 scaled |
| u | 4.2914860554e-10 | 1.9984014443e-15 | 1e-8 scaled |
| f | 4.6144421617e-11 | 7.1054273576e-15 | 1e-8 scaled |
| c | 2.6831162292e-10 | 3.1225022568e-17 | 1e-7 scaled |
| Sz | 2.3035064636e-10 | 1.1469266380e-16 | 1e-8 scaled |
| A RDM | 6.3833442658e-9 | 7.4570489953e-16 | 1e-8 Frobenius |
| B RDM | 6.3832367628e-9 | 8.8332133267e-16 | 1e-8 Frobenius |

The table reports absolute differences; scalar gates scale by `max(1,abs(actual),abs(expected))`.
The largest second-order scaled f discrepancy was `2.0594993245499716e-11`.
Projector sensitivity maxima, with reference cutoff `1e-12`, cap32 and fine dtau `.025`, are:

| Varied setting | u shift | c shift | f shift |
| --- | ---: | ---: | ---: |
| cutoff | 2.6784807339e-7 | 4.9585142687e-7 | 4.4639122621e-8 |
| cap | 1.3420855405e-7 | 4.1432886908e-7 | 1.1775556086e-8 |
| dtau | 2.3859369228e-6 | 1.4571336339e-8 | 2.3604708681e-6 |

Cutoff comparisons hold cap/dtau/beta fixed; cap comparisons hold cutoff/dtau/beta fixed;
step comparisons use cutoff `1e-12`, cap32. Fourteen of 32 second-order endpoint samples reached
their configured cap; maximum endpoint and trajectory bond was 32. The first-order endpoints
had bond17, below cap32. These results establish normalization mapping over this grid, **not
bond convergence, low-temperature convergence, or a ground-state limit**. Full samples,
mappings and sensitivities are in [aklt-grid.log](validation/aklt-grid.log) and
[aklt-first.log](validation/aklt-first.log).

## Retained coverage and additional commands

The routine wrapper runs `cargo test --all-targets`, including the non-ignored
`itebd_second_order` TFIM (`J=1,g=.7`) and XY (`gamma=.5,h=.7`) exact-observable checks;
complex-heat independent negative overlap/tail streams, directional mismatch, paired-shell
reality and tail nonconvergence checks; and `projector_high_temperature_control` at beta `.01`.
The TFIM/XY checks use beta1, dtau `.05`, cutoff `1e-13`, cap64, requiring second-order u/f
improvement, u/f/local error `<5e-3`, and heat error `<3e-2`. The AKLT high-temperature control
requires `abs(beta*f+ln(3))<=.01` and `abs(Sz)<=1e-8`. Those are routine controls, not new
low-temperature evidence. Their inclusion was inspected in source and confirmed by a fresh
`CARGO_INCREMENTAL=0 cargo test --release --all-targets -- --list` ([list](validation/coverage-list.log));
this list compiled all targets but did not execute those tests. The previously accepted extraction
routine run passed 409 tests with 16 ignored and zero warnings; it was not redundantly rerun here.

Other retained ignored scientific/benchmark drivers remain available below. They were **not
rerun** here because their numerical kernels were unchanged. These commands carry their existing
assertions and known limitations; listing them does not claim every broader numerical window
passes. Timing outputs remain informational.

```sh
CARGO_INCREMENTAL=0 cargo test --release --test itebd_second_order benchmark_first_vs_second_order_accuracy -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test itebd_second_order benchmark_first_vs_second_order_timing -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test itebd_complex_benchmark benchmark_complex_accuracy -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test itebd_complex_benchmark benchmark_complex_timing -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test specific_heat_benchmark benchmark_complex_specific_heat_validation -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test aklt_finite_t low_temperature_energy_approaches_ground_state -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --test aklt_finite_t specific_heat_matches_numerical_beta_derivative -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --lib runner::tests::aklt_sweep_matches_recorded_demo -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --lib variance::tests::specific_heat_tfim_low_temperature -- --ignored --exact --nocapture
CARGO_INCREMENTAL=0 cargo test --release --lib contraction_boundary_audit::benchmark_task9_remaining_nary_topologies -- --ignored --exact --nocapture
ITEBD_EVIDENCE_PATH="$PWD/results/rdm-timing-001.json" CARGO_INCREMENTAL=0 cargo test --release --lib itebd_rdm::benchmarks::checkpoint_rdm_timing_matrix -- --ignored --exact --nocapture
```

Use a fresh evidence output for the last command too. The remaining ignored
`solve_run::tests::checkpoint_child` is an internal subprocess helper, executed by its ordinary
parent interruption test with a temporary directory and environment; it is not a standalone
scientific measurement. Avoid invoking all ignored tests indiscriminately.

## Artifact fingerprints

SHA-256 for the compact public evidence:

| Artifact | SHA-256 |
| --- | --- |
| `aklt-first.log` | `ae4fd8fc4f5d03383c558f335d1fdcf4cbcc9de47eac80a2ed2d40f8c70306d8` |
| `aklt-grid.log` | `0c8e104c28e9f8c66b70e3afb9a80879ba2615add39aa36f973b2ac9867a6bdb` |
| `checkpoint-rdm.json` | `56660c6063d6fe8d27e6c44073d3104e8582b759f42a04cdfb3bff76ea81be9f` |
| `coverage-list.log` | `842d5c8e02e6a2c07b95d345c2d85b9a80c3ece8d3d09b49702eabcb50159fe8` |
| `list.log` | `ef63851dfb72a6bee66d562addcbef9b9c7ee219d4ec34511f73da85ee000862` |
| `matrix.log` | `6a526bff07ec20c32e73a0532b7624850512616fb4bd50e669c8191d27b48e47` |
| `summary.json` | `c9cc2bca901cdf598f1f5e05572bc0416eb90bb8076844ae6aef9e650f62cbbd` |

`Cargo.toml`: `fe610a7c025890af5a5c126b45e6b5d8bd4f1f76d7da17b005adf604e5b95802`.

`Cargo.lock`: `02b4a6ea7d36c9c05b7b028b17cdc558aae4e2ee6da63b5daff46d2ff0bd27db`.
