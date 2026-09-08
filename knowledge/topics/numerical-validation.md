# Numerical validation

A passing software test checks its assertions at its fixture settings. A scientific conclusion
also needs a reference or limiting case and evidence that the relevant numerical errors are
controlled. Use the [validation report](../../docs/validation.md) for measured settings, tables,
commands and provenance, and retain the [known limitations](../../docs/limitations.md).

Under the [conventions](../../docs/numerical-conventions.md), a complete step advances beta by
`2*dtau`. New input defaults to second-order Strang evolution; explicit first order remains
available, while legacy result metadata without an order means first order. The selected default
does not guarantee a quadratic observed slope for every model, observable or grid.

Use a bounded validation sequence:

1. Check an independent reference and its resolution. Refine quadrature and finite-difference
   steps where used; their changes are diagnostics, not rigorous error bounds. Check limiting
   cases and physical invariants as appropriate to the model.
2. Select and report a step-size window with positive decreasing errors. Nominal Trotter order
   can be obscured by truncation, active bond caps, canonicalization and reference floors. Report
   failed or excluded windows instead of extending a coarse-window conclusion to finer grids.
3. Vary cutoff and bond cap at fixed comparison settings, and fixed-point/tail tolerances for
   estimators that use them. Compare their sensitivity with the claimed discretization error.
4. Compare independent estimators: direct versus RDM observables, or variance heat versus an
   energy derivative. Agreement between two paths alone does not establish reference accuracy.

The [second-order controls and accuracy/timing drivers](../../tests/itebd_second_order.rs)
implement exact-observable checks and explicit window selection. The
[matrix CLI qualification](../../tests/solve_matrix_evidence.rs) records reference-resolution
changes and a fixed cutoff/cap grid. The report's
[checkpoint/RDM qualification](../../docs/validation.md#checkpoint-and-rdm-refinement) separates
component round trips, continuation and physical-observable refinement.

The [complex fine-grid and off-canonical limits](../../docs/limitations.md) still apply.
Canonicalization floors can dominate a refinement sequence; a successful selected window does
not resolve that floor. In particular, the documented beta=.2 real/phase free-energy discrepancy
has an unresolved normalization cause. Do not infer a cause from agreement of other observables.

Timing comparisons require the workload, dimensions, iterations, build mode, hardware and sample
method. The [retained benchmark commands](../../docs/validation.md#retained-coverage-and-additional-commands)
produce informational timings; execution-provenance elapsed times are not performance claims.
No routine wall-clock threshold or universal first/second-order speed ranking follows from them.
