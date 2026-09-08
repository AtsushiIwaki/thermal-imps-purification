# Development workflow

Use this guide from the repository root. It supports ordinary Git review and local commands; it
does not depend on a particular model, agent framework, hosted service, or global skill.
Before changing numerical behavior, use the authoritative
[numerical conventions](numerical-conventions.md), [validation record](validation.md), and
[limitations](limitations.md). The [knowledge index](../knowledge/index.md) routes to focused
explanations without replacing those evidence owners.

## Prepare a bounded task

Capture the base before edits and record the exact editable paths, pre-existing changes, affected
caller signatures and types, relevant numerical or persistence contracts, early checks, final
verification owner, evidence location, and cleanup owner. Use the
[task handoff template](superpowers/templates/task-handoff.md) when useful.

```sh
TASK_BASE=$(git rev-parse HEAD)
git status --short
```

Inspect four distinct views before handoff and before commit:

```sh
git diff --name-only --no-renames "$TASK_BASE" HEAD
git diff --name-only --no-renames
git diff --cached --name-only --no-renames
git ls-files --others --exclude-standard
```

They show committed changes since the captured base, unstaged tracked changes, staged changes,
and untracked files. Compare their union with the recorded scope and entry status. Inspect new
files directly because ordinary `git diff` omits untracked content. Preserve unrelated changes,
format only owned files, and stage explicit paths.

## Check affected interfaces early

Choose focused checks from the changed caller contract before building dependent behavior.
Confirm that a test filter executes a nonzero number of tests: listing a test proves discovery,
while executing it proves its assertions passed. A misspelled filter may exit successfully after
running no tests.

For checkpoint implementation and serialization changes, run:

```sh
CARGO_INCREMENTAL=0 cargo test --lib itebd_checkpoint::
CARGO_INCREMENTAL=0 cargo test --test itebd_state_view --test itebd_checkpoints --test itebd_checkpoint_continuation
```

These targets cover exact state/checkpoint round trips, malformed stored schemas, authoritative
dimensions, continuation behavior, real HDF5 failures, and injected failures through production
error mapping. Keep real failure coverage alongside injection.

For tensor call sites or fixtures, run the live policy check:

```sh
CARGO_INCREMENTAL=0 cargo test --lib contraction_boundary_audit::repository_contraction_boundary_is_ast_audited -- --exact
```

This checks the repository contraction boundary, not numerical correctness. Contraction changes
also need independent real/complex numerical oracles, conjugation and index-order checks, and the
relevant invariants.

For Python verification tooling, parsers, or smoke drivers, run:

```sh
python3 -m unittest discover -s scripts/tests
```

## Complete verification

Assign one owner to run final verification and retain fresh command, exit, and output evidence.
Use focused commands during edits. Repeat a successful command when relevant code later changed,
a failure intervened, or an unresolved risk justifies a rerun.

```sh
CARGO_INCREMENTAL=0 python3 scripts/verify.py routine
CARGO_INCREMENTAL=0 cargo test --doc
```

The routine wrapper still executes `cargo test --all-targets`; its compact successful output does
not reduce target scope. Documentation tests are separate. A failing wrapper should be diagnosed
with the smallest direct Cargo target that exposes the failure, without replacing the final full
scope. The exact ignored release benchmarks and scientific evidence commands are listed in
[validation](validation.md); run the relevant named command rather than all ignored tests.
Software checks do not by themselves establish physical convergence or a scientific claim.

Review the full scoped diff and affected callers, including untracked additions before staging.
Resolve substantive findings, then check whitespace and all four Git views again. Record what
changed, command results, evidence links, limitations, and any out-of-scope files.

## State, investigations, and cleanup

When `STATE.md` is present, it is the live owner of current position, unresolved constraints, next
action, and relevant links. Acceptance belongs in the active plan; dated reports own historical
evidence. Investigation notes begin with the dated current conclusion using the
[investigation template](superpowers/templates/investigation-note.md), while preserving labeled
earlier observations.

Cleanup has one owner and an exact authorized checkout path and Git ref scope. Before removal,
inventory and preserve ignored or untracked artifacts that must survive. Verify filesystem
removal, Git worktree registration, and branch/ref postconditions separately. If a command stalls,
an unexpected path or ref appears, or filesystem and Git state disagree, stop destructive retries,
preserve recoverable residue without overwriting it, and report the unresolved disposition. Never
broaden cleanup or prune unrelated registrations to make a check pass.
