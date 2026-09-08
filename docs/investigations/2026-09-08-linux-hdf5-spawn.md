# Investigation: GNU/Linux HDF5 and Git process launch

## Latest conclusion — 2026-09-08

- Status: the GNU/Linux descriptor and spawn-window defects are reproduced and repaired; focused checks pass. The full local routine run is blocked by a separate numerical failure that reproduces identically on the original commit.
- Established: the [first hosted CI run](https://github.com/AtsushiIwaki/thermal-imps-purification/actions/runs/34218807282) built successfully on both platforms. macOS completed verification; Ubuntu failed `checkpoint_restart_from_zero_does_not_repeat_the_initial_observation` at `tests/solve_beta_zero.rs:116` while writing snapshot 0 GammaB to a complex checkpoint.
- Observed error: `H5Fcreate(): ... errno = 17 ... File exists`. The pinned `tensor4all_hdf5::append_itensor` calls `hdf5_metno::File::append`, which tries read/write open first and falls back to exclusive creation on any open failure. The final message therefore does not identify why the original open failed.
- Established locally: the standard GNU/Linux launcher passes an open HDF5 descriptor to a held child, and the parent's reopen fails with errno 11. Closing child descriptors alone is insufficient: removing only the spawn synchronization from the repaired launcher also fails the FIFO-gated close/reopen regression with errno 11.
- Repair: use GNU close-from file actions and the existing shared HDF5 mutex across native spawn. Preserve macOS behavior, release synchronization before pipe draining/waiting, and keep other targets on the existing standard launcher.
- Not established: a repaired hosted CI run or a native macOS run of the changed source. The original CI error masks the underlying open error; the new regressions establish a concrete matching failure mechanism, not a trace of the original runner's file descriptors.
- Scope decision: the user selected completion of the HDF5 repair and a separate task for the numerical failure. Full Linux CI success remains outstanding; no numerical code, threshold, test selection, dependency pin or HDF5 locking setting is changed by this repair.
- Next action: publish the reviewed HDF5 repair and observe hosted CI; investigate the independent numerical failure separately.

## Historical observations

### CI log and dependency inspection

The failing target selected six tests; five passed and one failed. Earlier checkpoint,
RDM, real/complex evolution and second-order targets passed in that run. No numerical
tolerance or schema assertion failed in the reported test: failure occurred while
creating the first checkpoint for the complex branch.

The pinned HDF5 sec2 driver opens native descriptors without O_CLOEXEC. Its nonblocking
flock and descriptor lifetime must be considered alongside process creation. The
existing macOS launcher closes unmentioned descriptors at exec and takes the shared
HDF5 mutex across `posix_spawnp`, releasing it before draining stdout or waiting.

GNU libc supplies `posix_spawn_file_actions_addclosefrom_np` starting with glibc 2.34;
the [upstream introduction](https://sourceware.org/pipermail/glibc-cvs/2021q3/073654.html)
documents closing the child's descriptors at and above a bound without changing the
parent descriptor table. This is the GNU/Linux counterpart to macOS's
`POSIX_SPAWN_CLOEXEC_DEFAULT`.

The [task plan](../superpowers/plans/2026-09-08-linux-hdf5-spawn.md) owns acceptance and
scope. Detailed local commands and raw output are retained under ignored
`results/linux-hdf5-spawn-20260908/`.

## Local verification

The primary agent owns this verification. The base is
`641bd45cc4b7bc1b954de940b5b3553020696f66`; the repair is on `codex/linux-hdf5-spawn`.
GNU/Linux checks use an independent checkout in WSL2 Ubuntu 22.04.3, glibc 2.35,
Rust 1.96.1 and CMake 4.4.3. Commands set `CARGO_INCREMENTAL=0` and compile with
`CARGO_BUILD_JOBS=4`; test threads were not serialized. This is local WSL evidence,
not a rerun on GitHub's `ubuntu-latest` image.

| Check | Result | Local evidence |
| --- | --- | --- |
| Held-child HDF5 descriptor/reopen regression, unchanged production launcher | Failed as required: child reported `inherited`, parent `H5Fopen` failed with errno 11 | `red-descriptor.log` |
| Native spawn synchronization removed only in the disposable repaired checkout | Failed as required: FIFO-gated real close/reopen failed with errno 11; production source restored afterward | `red-spawn-window.log` |
| Repaired `git_process::tests::` | 10 passed | `final-launcher.log` |
| `runner::tests::git_revision_contract` | 1 passed | `green-callers.log` |
| `solve_beta_zero` target, including original failing test | 6 passed | `green-beta-zero.log`, `final-routine.log` |
| Python harness suite | 54 passed | `final-python.log` |
| Full `scripts/verify.py routine` | Failed at the specific-heat test described below | `final-routine.log` |
| Clean original commit, exact same specific-heat test | Failed with identical imaginary residual | `baseline-heat.log`, `heat-baseline-head.txt`, `heat-baseline-status.txt` |
| Linux documentation tests | Passed; zero doctests selected | `final-doc.log` |
| Linux locked debug build and real/complex smoke | Passed; three records each at beta 0, .1, .2 | `final-build.log`, `final-smoke.log` |
| Dependency lock and tested source identity | Lock unchanged; tested manifest, lock and launcher sources byte-identical to the working tree | `final-lock.log`, `tested-source-sha256.txt` |
| Windows locked release build and real/complex smoke | Passed; three records each at beta 0, .1, .2 | `windows-build.log`, `windows-smoke.log` |

Exact focused and final commands are recorded in the
[plan](../superpowers/plans/2026-09-08-linux-hdf5-spawn.md); execution scripts, separate
exit statuses and complete stdout/stderr remain in the ignored evidence directory.
Windows uses MSVC, Rust 1.96.1 and CMake 4.4.3. Its smoke check exercises the existing
standard-launcher branch; it does not establish full Windows test-suite coverage.
An independent code review found no actionable launcher findings. Its documentation findings
were resolved by correcting the sole production caller and distinguishing historical extraction
coverage from current CI evidence. Owned Rust files pass `rustfmt --check`; the working diff
passes `git diff --check`. The pre-existing `scripts/smoke.py` mode change is outside this repair.

## Separate numerical failure

The full local routine run passed the earlier HDF5/checkpoint targets, then failed
`tests/specific_heat.rs:386`, `twisted_xx_matches_untwisted_report_and_exact_heat`:

```text
NonRealSpecificHeat { stage: "parity_a", imaginary: 4.809681007551957e-9, tolerance: 1e-10 }
```

An independent clean checkout of the original commit, with the same toolchain and dependencies,
fails the exact test with the identical value:

```sh
CARGO_INCREMENTAL=0 cargo test --locked --test specific_heat twisted_xx_matches_untwisted_report_and_exact_heat -- --exact --nocapture
```

This test evolves tensors directly without invoking Git or HDF5. Its phase-twisted XX case
uses beta .6, dtau .05, phase pi/5, cutoff 1e-12 and bond cap 32. The baseline comparison
establishes that this failure predates the launcher repair; it does not explain its numerical
cause. Investigation must preserve the reality assertion and inspect convergence and canonical
residuals before considering a numerical change. It remains a separate task by user choice.
