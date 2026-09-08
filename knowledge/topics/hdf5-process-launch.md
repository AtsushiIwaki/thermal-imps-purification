# HDF5 and process launch

The [runtime Git helper](../../src/git_process.rs) supplies provenance through
[runner metadata](../../src/runner.rs), which the [solve driver](../../src/solve_run.rs) retains.
On macOS, its [native launcher](../../src/git_process/macos.rs) addresses two distinct descriptor
lifetime concerns: retention after exec and the transient interval during spawn. Closing
unmentioned descriptors at exec alone does not coordinate concurrent HDF5 close/open operations
during that earlier interval.

The macOS helper sets `POSIX_SPAWN_CLOEXEC_DEFAULT` and wraps `posix_spawnp` in
`hdf5_metno::sync::sync`, using the shared HDF5 mutex. It releases that mutex before draining
stdout or waiting for the child. The [lockfile](../../Cargo.lock) resolves the HDF5 dependency;
this coordination relies on operations sharing that synchronization lock. It is not a separate
application mutex around only selected writers.

Two macOS [public regressions](../../src/git_process/tests.rs) cover the boundaries:

- `open_hdf5_file_is_not_inherited_and_reopens_before_child_exit` observes the child's descriptors
  and checks that the parent can reopen HDF5 while the child remains alive.
- `hdf5_close_reopen_is_safe_during_native_spawn` gates native spawn file actions and exercises
  concurrent HDF5 close/reopen during that interval.

These links describe implementation and regression coverage. This page adds no execution result
or independent reproduction of an external diagnostic trace. Non-macOS builds retain standard
process launching. Independently launched subprocesses and native operations bypassing the
shared mutex are outside this coordination. A separate dependency copy or backend with its own
synchronization lock requires rechecking the contract. The helper does not strengthen the
[checkpoint publication or power-loss guarantees](../../docs/limitations.md).
