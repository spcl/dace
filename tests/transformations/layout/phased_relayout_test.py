# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The layout-transformation PHASE: 4 row-major nests then 4 column-major nests over one array.

This is the didactic case for a MID-FLIGHT layout change. Eight nests form a chain over ``A`` and a
running array; each nest reads the PREVIOUS output reversed (``O_{k-1}[i, N-1-j]``) so the nests
survive maximal fusion (a reversal is non-pointwise -- the multinest_programs lesson -- and stays
row-major friendly), plus a straight ``P[i,j]``, plus ``A``:

    nests 0-3 (row-major happy):     O_k[i,j] = O_{k-1}[i,N-1-j] + A[i,j] + P[i,j]
    nests 4-7 (column-major happy):  O_k[i,j] = O_{k-1}[i,N-1-j] + A[j,i] + P[i,j]

(nest 0 seeds with ``O_0 = A + P``.) In the transposed nests the reversed ``O_{k-1}`` and the straight
``P`` are a MAJORITY of row-major accesses that pins the canonical schedule to ``(i, j)`` (so schedule
permutation cannot dissolve the transpose; the remedy for ``A`` is a LAYOUT, not a schedule). Under
that pinned schedule ``A[i,j]`` is contiguous only row-major and ``A[j,i]`` only column-major, so
``A``'s two halves genuinely disagree; ``P`` and the ``O`` chain stay row-major throughout.

No single layout of ``A`` is good for all eight nests: row-major loses the last four, column-major
loses the first four. The global assignment's per-array Viterbi DP (``per_array_dp``) prices the one
relayout against the four nests of benefit it buys and inserts a SINGLE layout-transformation phase
at the 3->4 boundary -- the trajectory ``[identity]*4 + [perm10]*4``. ``brute_force_trajectories``
confirms the DP is optimal; ``break_even_uses`` shows why the transpose pays (four consuming nests >
the break-even count). ``A`` is a read-only input, so the relayout is a pure transpose copy of ``A``
at the boundary; no nest both reads and writes ``A`` (a parallel map may not do that).

This is the local, single-process shape of the OMEN transpose (mpi_omen_transpose_test.py): the same
"two phases want opposite layouts, pay one transpose between them" decision, here priced by the cost
model instead of hand-placed.
"""
import numpy
import pytest

import dace
from dace.transformation.layout.apply_assignment import apply_assignment
from dace.transformation.layout.assignment_costs import assignment_arrays, model_costs
from dace.transformation.layout.cost_model.relayout import break_even_uses
from dace.transformation.layout.global_assign import (brute_force_trajectories, conflict_report, format_conflict_report,
                                                      per_array_dp, to_assignment)
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N")


@dace.program
def phased(A: dace.float64[N, N], P: dace.float64[N, N], O0: dace.float64[N, N], O1: dace.float64[N, N],
           O2: dace.float64[N, N], O3: dace.float64[N, N], O4: dace.float64[N, N], O5: dace.float64[N, N],
           O6: dace.float64[N, N], O7: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        O0[i, j] = A[i, j] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O1[i, j] = O0[i, N - 1 - j] + A[i, j] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O2[i, j] = O1[i, N - 1 - j] + A[i, j] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O3[i, j] = O2[i, N - 1 - j] + A[i, j] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O4[i, j] = O3[i, N - 1 - j] + A[j, i] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O5[i, j] = O4[i, N - 1 - j] + A[j, i] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O6[i, j] = O5[i, N - 1 - j] + A[j, i] + P[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O7[i, j] = O6[i, N - 1 - j] + A[j, i] + P[i, j]


def oracle(A, P):
    out = {}
    prev = None
    for k in range(8):
        a = A if k < 4 else A.T
        cur = (a + P) if k == 0 else (prev[:, ::-1] + a + P)
        out[f"O{k}"] = cur
        prev = cur
    return out


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((n, n)), "P": rng.random((n, n))}


def build():
    sdfg = phased.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def run_and_check(sdfg, n, seed=0):
    inp = make_inputs(n, seed)
    outs = {f"O{k}": numpy.zeros((n, n)) for k in range(8)}
    sdfg(A=inp["A"].copy(), P=inp["P"].copy(), **outs, N=n)
    ref = oracle(inp["A"], inp["P"])
    for name, r in ref.items():
        assert numpy.allclose(outs[name], r), f"{name} diverges from the oracle"


def test_phased_relayout_inserts_one_layout_phase(n=256):
    """The DP finds A's single mid-flight transpose: identity for the 4 row-major nests, perm10 for
    the 4 column-major nests, one boundary change -- and it is provably optimal + bit-exact applied."""
    sdfg, kernels = build()
    assert len(kernels) == 8, [k.state.label for k in kernels]  # 4 row + 4 transposed nests survive fusion
    assert "A" in assignment_arrays(sdfg, kernels)

    costs = model_costs(sdfg, kernels, symbols={"N": n})
    rows = {r.array: r for r in conflict_report(costs, len(kernels))}
    print(format_conflict_report(list(rows.values())))
    assert rows["A"].conflicting
    assert rows["A"].per_kernel_preference == ["identity"] * 4 + ["perm10"] * 4  # the 4/4 disagreement
    assert not rows["P"].conflicting  # the schedule-pinning array never conflicts

    dp = per_array_dp(costs, len(kernels))
    oracle_traj = brute_force_trajectories(costs, len(kernels))
    for array in dp:
        assert dp[array].cost == pytest.approx(oracle_traj[array].cost), array  # DP == brute-force oracle
    assert dp["A"].tags == ["identity"] * 4 + ["perm10"] * 4  # ONE layout-transformation phase
    assert dp["A"].changes() == 1

    single = per_array_dp(costs, len(kernels), allow_changes=False)
    assert dp["A"].cost < single["A"].cost  # the mid-flight transpose beats carrying either layout

    # Why it pays: the transpose amortizes over the four column-major nests (break-even < 4).
    row_cost = costs.node_cost[("A", 4, "identity")]  # nest 4 wants perm10; identity here is the strided read
    col_cost = costs.node_cost[("A", 4, "perm10")]
    relayout = costs.relayout_cost[("A", "identity", "perm10")]
    breakeven = break_even_uses(row_cost, col_cost, relayout)
    assert breakeven is not None and breakeven <= 4

    assignment = to_assignment(dp, costs.layouts)
    assert "A" in assignment
    applied = apply_assignment(sdfg, kernels, assignment)
    assert len(applied.boundary_states) >= 1  # the materialized layout-transformation phase
    sdfg.validate()
    run_and_check(sdfg, 64, seed=7)


if __name__ == "__main__":
    test_phased_relayout_inserts_one_layout_phase()
    print("phased relayout test PASS")
