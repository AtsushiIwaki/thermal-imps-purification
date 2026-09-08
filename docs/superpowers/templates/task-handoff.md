# Task handoff template

This is a template. Replace every angle-bracket prompt before publishing an instantiated handoff.
Use repository-relative paths and public commit IDs in the shared report. Keep an absolute local
checkout path only in local execution context when it is needed to select the right checkout.

## Objective and scope

- Task: <task identifier and requirements source>
- Outcome: <observable result and acceptance conditions>
- Base: <captured public commit ID>
- Editable paths: <exact repository-relative paths>
- Pre-existing changes: <staged, unstaged, and untracked paths with ownership, or checked none>
- State ownership: <when STATE.md exists, link it as live-state owner; name the plan/report owners>
- Cleanup ownership: <one owner and exact path/ref scope, or explicitly outside this task>

## Contracts and inputs

- Affected callers: <signatures, shared types, error conversions, or explicitly none>
- Numerical/persistence contracts: <exact index order, tolerance, schema, and error behavior>
- Dependency boundary: <authoritative pinned revision/API and prohibited local substitutes>
- Scope changes: <how an additional path is recorded before editing>

## Verification order

| Stage | Exact command or check | What it establishes | Passing evidence |
| --- | --- | --- | --- |
| Early | <small affected-interface check> | <specific assumption or defect> | <nonzero selection and invariant> |
| During | <focused regression and caller check> | <changed behavior> | <command, exit, and fresh output> |
| Completion | <full, doc, release, scientific, or content checks and their owner> | <scope and limits> | <evidence link> |

For persistence work, include exact round trips, malformed schemas, and a real I/O failure beside
injection. For tensor work, include the live AST audit and independent real/complex numerical
checks. For scientific claims, name convergence and sensitivity evidence separately from software
tests. Use the four Git views in [the workflow](../../development-workflow.md) before submission.

## Report and coordination

- Report: <repository-relative or public report location>
- Evidence: <commands, exits, fresh outputs, and long-log links>
- Handoff reply: <status, commit, checks, concerns, and report location>
- Investigation: <latest-conclusion link if findings changed, or not applicable>
- Remaining concerns: <limitations or none>

Do not leave prompts in an instantiated handoff. Report scope changes and blockers when discovered;
otherwise communicate material findings and completion. Do not add approval gates or require a
specific model, vendor API, global skill, or dispatched agent.
