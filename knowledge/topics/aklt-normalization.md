# AKLT normalization

The [model implementation](../../src/model.rs) keeps the historical `aklt` convention and adds
the coefficient-one `aklt_projector` preset. For `x = S_i · S_(i+1)`,
`P2 = (x*x + 3*x + 2*I)/6` and `h_old = 2*P2 - 2*I/3`.
The [independent projector tests](../../tests/aklt_projector.rs) check the spin-two subspace,
spectrum, algebraic relation, exponential and limiting controls.

Matched Gibbs states use:

| Quantity | Mapping |
| --- | --- |
| Inverse temperature | `beta_old = beta_projector/2` |
| Evolution step | `dtau_old = dtau_projector/2` |
| Energy per site | `u_projector = u_old/2 + 1/3` |
| Free energy per site | `f_projector = f_old/2 + 1/3` |
| Specific heat, Sz, normalized RDMs | Equal at matched Gibbs states |

The [normalization driver](../../tests/aklt_projector_evidence.rs) compares these quantities for
both orders, including both RDM parities. The [public normalization tables](../../docs/validation.md#aklt-normalization-controls)
own the finite grid, discrepancies, tolerances and sensitivity results. Active bond caps in that
grid prevent a bond-convergence conclusion. Its finite-temperature mapping and high-temperature
control do not establish low-temperature convergence or approach to the ground state; retain the
[AKLT limits](../../docs/limitations.md).

Historical and projector checkpoint files are not interchangeable restarts. Their Hamiltonian
values differ, and the [restart driver](../../src/solve_run.rs) enforces compatibility of the
actual Hamiltonian. Changing a label does not perform the temperature/energy conversion. Use
fresh trajectories when changing normalization and record the selected preset explicitly.
