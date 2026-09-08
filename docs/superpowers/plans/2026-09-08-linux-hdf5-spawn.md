# Linux HDF5 process-launch repair plan

**Goal:** Prevent runtime Git provenance capture from retaining HDF5 descriptors or racing HDF5 close/reopen on GNU/Linux.

**Architecture:** First extend the existing public descriptor regression to GNU/Linux and observe its result on the unchanged launcher. If descriptor inheritance is reproduced, reuse the native macOS spawn implementation with GNU close-from file actions, sharing the pinned HDF5 synchronization lock only across spawn. Keep pipe draining and child waiting outside that lock.

**Tech stack:** Rust 1.96.1, pinned hdf5-metno/libc, GNU/Linux posix_spawn, existing macOS native launcher.

## Scope and contracts

- Base: `641bd45cc4b7bc1b954de940b5b3553020696f66`; working branch `codex/linux-hdf5-spawn`.
- Pre-existing change: `scripts/smoke.py` mode 100755 -> 100644; preserve and exclude.
- Editable paths: `Cargo.toml`, `src/git_process.rs`, `src/git_process/macos.rs` (move to `src/git_process/posix.rs`), `src/git_process/tests.rs`, `README.md`, `STATE.md`, `knowledge/index.md`, `knowledge/topics/hdf5-process-launch.md`, `docs/validation.md`, `docs/provenance.md`, this plan, and `docs/investigations/2026-09-08-linux-hdf5-spawn.md`.
- Generated evidence only: `results/linux-hdf5-spawn-20260908/`; Linux tools/cache/independent checkout at `/home/iwaki/.cache/thermal-ci-20260908-641bd45` inside WSL Ubuntu.
- Caller contract: `git_stdout(&[&str]) -> Option<Vec<u8>>`; success captures stdout, nonzero exit returns None, spawn/input errors remain errors in the private helper; PATH/cwd/environment and the runner's short-revision formatting remain unchanged. Checkpoints retain that metadata; they do not independently launch Git.
- No numerical behavior, checkpoint schema, exclusive-creation rule, failure-context mapping, dependency pin, or lockfile change. tensor4all remains `880032daea2a84d7ddfe84a2594a6d160631afc2`; tenferro remains `f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266`.
- GNU/Linux native close-from requires glibc >=2.34; document that floor. macOS keeps CLOEXEC_DEFAULT. Other targets keep the existing standard launcher; no new coverage claim for them.
- Verification/review/cleanup owner: primary agent. Retain generated evidence and caches. Do not delete existing artifacts or commit/push without a concrete final review.

## Steps

- [x] Add GNU/Linux `holds_path` using `/proc/self/fd` read-link inspection. Enable the existing held-child descriptor/reopen regression and child entry on GNU/Linux, without changing production code.
- [x] Run `CARGO_INCREMENTAL=0 cargo test --locked --lib git_process::tests::open_hdf5_file_is_not_inherited_and_reopens_before_child_exit -- --exact --nocapture`. Require a real descriptor-inheritance or reopen failure; if it passes, investigate further before changing the launcher.
- [x] Move the native launcher to `posix.rs`. Select it with `cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))` and expand the existing libc target dependency to that same condition.
- [x] In GNU/Linux file actions, after duplicating stdout, call `libc::posix_spawn_file_actions_addclosefrom_np(&mut actions.0, 3)`. Retain the explicit pipe close for macOS. Set SIGPIPE defaults on both; add CLOEXEC_DEFAULT only on macOS. Continue calling `posix_spawnp` inside `hdf5_metno::sync::sync` and release the lock before reading/waiting.
- [x] Enable existing native-launch regressions, including the FIFO-gated transient close/reopen test and revision descriptor scans, on GNU/Linux. Verify regression strength by temporarily omitting only the synchronization for a negative-control run in the independent Linux checkout, then restore the fixed source.
- [x] Run the whole `git_process::tests::` group and the original `solve_beta_zero` target. Also run `runner::tests::git_revision_contract`; the launcher group separately checks changes to disposable repository HEADs.
- [x] Run final GNU/Linux `CARGO_INCREMENTAL=0 python3 scripts/verify.py routine`, `CARGO_INCREMENTAL=0 cargo test --locked --doc`, Python harness tests, and the existing real/complex smoke driver with a fresh output directory. These include persistence exact round trips, malformed schemas, real I/O failures, and failure injection already in the suite.
- [x] Run a Windows build/smoke check using the existing release cache to verify the unchanged standard-launcher branch. Do not claim macOS execution of the modified launcher without hosted or native macOS evidence.
- [x] Review actual changed code and the runner caller plus metadata retention; check staged/unstaged/untracked/committed views, explicit file formatting, lock preservation, and `git diff --check`. Update STATE and the dated investigation with exact evidence and the hosted CI limitation.

## Acceptance

The intended gate required the held-child regression to fail before the repair and pass after;
the transient-window negative control to fail while the coordinated version passes; the original
beta-zero regression and full GNU/Linux routine suite to pass without relaxed assertions, skipped
tests, disabled HDF5 locking or test-thread serialization; and Windows smoke to continue passing.

Final outcome: all listed execution/review steps were performed. Both negative controls and all
focused repair checks satisfy their gates. Python (54), beta-zero (6), launcher (10), caller (1),
Linux/Windows build and real/complex smoke checks pass. Linux documentation tests succeed with
zero selected. The full routine **does not pass**: a separate phase-twisted XX specific-heat test
fails identically on a clean original commit. By explicit user choice, numerical investigation is
a separate task. Thus the bounded HDF5 repair is reviewed and locally verified, while the overall
full-suite acceptance gate remains outstanding. See the
[investigation](../../investigations/2026-09-08-linux-hdf5-spawn.md) for the exact residual and logs.

At the pre-commit review, the repair had not been committed or pushed; the user subsequently
authorized committing it. Its hosted CI result is not established. The
pre-existing smoke script mode change remains untouched. Evidence and independent Linux
checkouts/caches are retained for the follow-up; no cleanup of unrelated artifacts is needed.
