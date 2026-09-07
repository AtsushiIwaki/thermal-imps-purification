# Numerical conventions

Each complete imaginary-time step advances inverse temperature by `2 * dtau`, because the
purification evolves by `exp(-beta H / 2)` on the ket. A missing `trotter_order` in a new
configuration selects the second-order Strang sequence `AB/2 -> BA -> AB/2`. Historical result
JSON without this metadata is interpreted as first order.

The two-site unit cell averages its A-start and B-start bond energies and reports energy, free
energy, and specific heat per physical site. Free energy is reconstructed from the accumulated
log normalization divided by two sites. Canonicalization returns contribute to this accumulated
normalization; a restart preserves the completed step count, cadence, and stored normalization.

Matrix configurations use `basis_order = "first_site_fastest"`. For local dimension `d`, the
two-site tuple `(s1, s2)` maps to `s1 + d*s2`. RDM row indices are ket tuples and columns are bra
tuples in the same convention. Ancilla indices are traced out. `RdmParity::A` and `RdmParity::B`
select the physical parity of the first retained site; reported two-site quantities average the
two parities where specified.

Real/complex state storage is classified from both Hamiltonian matrices before state creation.
Only exactly zero imaginary entries, including signed zero, select the real backend. Any nonzero
Hamiltonian imaginary component selects complex storage; a complex observable alone does not.
The choice persists across checkpoints. Complex contractions use conjugated bra legs, and input
matrices are checked for finiteness, shape, and Hermiticity rather than silently repaired.

Public fallible APIs return `ItebdError` variants for invalid configuration, topology, backend
mismatch, Hermiticity/reality failures, tensor operations, fixed-point failure, specific-heat
tail nonconvergence, resource limits, and persistence errors. The scalar specific-heat API does
not return a partial value when its bidirectional tail criterion fails.

JSON observations and HDF5 trajectories are published independently. JSON replacement is atomic
within its directory, while HDF5 completion markers distinguish complete snapshots. The pair is
not a transaction: after interruption, inspect complete HDF5 snapshots and join JSON segments
only through the selected restart step. Every new or resumed run must use fresh output paths.
