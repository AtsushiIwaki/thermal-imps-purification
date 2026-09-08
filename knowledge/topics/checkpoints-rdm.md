# Checkpoints and interval RDMs

[Checkpoint storage](../../src/itebd_checkpoint/mod.rs) retains tensor components, Schmidt values,
index roles, backend, authoritative Hamiltonian, progress and metadata. Save/load does not
canonicalize or reclassify the state. [Round-trip tests](../../tests/itebd_checkpoints.rs) compare
stored component bits, including signed zeros and complex storage with all-real payloads.
Independent trajectories can choose different Gamma gauges and index identities: compare their
physical quantities rather than requiring fresh tensor components to match.

The [solve driver](../../src/solve_run.rs) checks continuation compatibility and adopts the loaded
Hamiltonian, backend and progress. Step counts, cadence and accumulated normalization remain
part of the state of the run. [Off-cadence continuation tests](../../tests/itebd_checkpoint_continuation.rs)
cover both backends and Trotter orders, including a split between scheduled canonicalizations.
The [numerical conventions](../../docs/numerical-conventions.md) define per-site free energy and
beta advancement; restart must not insert an extra normalization or reset global schedules.

[Interval RDMs](../../src/itebd_rdm/interval.rs) are approximate fixed-point reconstructions,
even when checkpoint storage is exact. Independent identity-seeded
[boundary solves](../../src/itebd_rdm/environment.rs) must pass residual checks within their
iteration budgets. Rows are ket tuples and columns are bra tuples in first-site-fastest order:
`(s1,s2)` maps to `s1+d*s2`. Ancillas are traced out. A/B parity selects the first retained site's
physical parity; quantities average those parities where specified by the
[conventions](../../docs/numerical-conventions.md).

Raw normalized matrices are checked for trace, Hermiticity and positivity; failing matrices are
not repaired by clipping or Hermitian projection. A degenerate dominant transfer sector can make
identity-seeded boundaries sector-dependent despite small residuals. The
[RDM options and reports](../../src/itebd_rdm/mod.rs) bound planned output/intermediate elements
and occurrence-index metadata, not aggregate allocation, metadata storage or hidden backend
workspace. Dependency allocation during rejection of underreported corrupt dimensions also lies
outside that bound. See the complete [resource and sector limits](../../docs/limitations.md).

The [checkpoint/RDM measurement driver](../../tests/itebd_checkpoint_rdm_evidence.rs) and
[published refinement report](../../docs/validation.md#checkpoint-and-rdm-refinement) separately
assess component equality, continuation, direct/RDM agreement and exact-reference refinement.
A small storage or fixed-point residual alone does not prove thermodynamic convergence.

JSON observations and HDF5 trajectories are separate publications. Use fresh output destinations
for each new or resumed run, keeping the restart source read-only. After interruption, inspect
complete HDF5 snapshots and trim any preceding JSON segment to the selected restart step before
joining its suffix. Completion markers and flushes do not provide a joint transaction or a
power-loss guarantee; same-destination concurrent writers are unsupported. These
[publication limits](../../docs/limitations.md) remain part of the persistence contract.
