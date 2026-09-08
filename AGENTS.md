# Repository guidance

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and use the
[development workflow](docs/development-workflow.md) for task scope, verification, review, and
cleanup. These repository documents are self-contained; the `docs/superpowers/` directory is an
organizational convention and does not require an external skill installation.

## Numerical and tensor contracts

This crate implements finite-temperature purification for an infinite, translation-invariant
two-site unit cell evolved with iTEBD. Preserve the physical/ancilla split, alternating bond
orientation, canonical form, and the documented index order. Read
[numerical conventions](docs/numerical-conventions.md), [validation](docs/validation.md), and
[limitations](docs/limitations.md) before changing numerical behavior.

Express production tensor networks with `tensor4all` through explicit leg identities, retained
indices, and conjugation. Factor operators and cache intermediates that remain fixed across
iterative matrix-vector products; keep those products matrix-free instead of materializing a full
dense effective operator. Prefer the repository's pairwise contraction adapter and staged
contractions that bound intermediate size or enable reuse. Production N-ary contractions are not
generally permitted: the live AST boundary audit defines the narrow allowed ownership and must
remain green. Use an explicit outer product only for intentionally disconnected operands.

For any contraction change, compare against an independent oracle for both real and complex
inputs, including conjugation, index order, and relevant physical or canonical invariants.
Software tests alone do not establish a scientific result. Claims also need applicable evidence
for convergence and sensitivity to bond dimension, cutoff or tolerance, system size, time step,
fitting window, and initial conditions, plus limiting cases or an independent estimator.

Keep `tensor4all` pinned to `880032daea2a84d7ddfe84a2594a6d160631afc2` and the resolved
`tenferro` packages pinned to `f68b33d9e06cc0af8d2b9c40f7c52d8972cd1266`. Do not replace
them with local paths, patches, symlinks, or build-time dependencies on an external source tree.

## Change discipline

Define exact editable paths, the captured base, pre-existing changes, affected caller contracts,
early checks, completion evidence, and cleanup ownership before substantial work. Preserve
unrelated changes and inspect committed, unstaged, staged, and untracked views before staging.
Format only owned files; never recursively format `src/lib.rs`.

Persistence changes require exact round trips, deterministic real I/O failures together with
failure injection for otherwise hard-to-reach stages, and malformed schema or dependency inputs.
Assert the production error context as well as its underlying cause. Run focused checks early,
then the task-appropriate final verification with one named owner and fresh output. Review the
actual diff and affected callers through ordinary repository review; resolve substantive findings
before completion.

When `STATE.md` is present, it owns the current position, constraints, next action, and links.
Detailed evidence belongs in dated reports or investigation notes rather than a duplicate live
status section.
