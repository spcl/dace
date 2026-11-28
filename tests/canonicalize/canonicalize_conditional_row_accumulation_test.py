# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC's ``IF (ZQXN(JN) < ZEPSEC)`` update of the one row ``ZQXN(NCLDQV)``, summed over the species ``JN``.

Every species iteration adds its own row to row ``4`` and zeroes its own row, under a data-dependent guard. Rows ``0..3``
never meet row ``4``, so the species loop is a ``+`` reduction into row ``4`` with the identity ``0`` where the guard is
false, and canonicalization should lift it.

Two things stopped it. The frontend orders the snapshot of ``ZQXN(4, JL)`` before the zeroing write of
``ZQXN(JN-1, JL)`` with an ordering edge, and ``JN - 1 != 4`` needs ``JN < 5`` from the loop header: with the edge,
``FoldScalarReadCopies`` kept the snapshot, and ``AugAssignToWCR`` never saw ``zq[4] = zq[4] + x``. The fold now asks
the facts at the state (the enclosing loops' ranges) whether the written row lies apart from the read one.

Once the accumulation is a WCR on row ``4``, the species loop's two writes (row ``JN-1`` zeroed, row ``4`` summed) leave
the nested SDFG through one connector as the box ``JN-1:5``. ``LoopToMap`` now reads the accesses of the nested SDFG one
by one (``exact_accesses``), and so do the pass that decides whether an update accumulates (``accumulates``) and the
revert of a WCR that conflicts with nothing: a box that varies with ``JN`` must not vouch for the write to row ``4``,
which every iteration makes.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N")


@dace.program
def conditional_row_accumulation(zq: dace.float64[5, N], kidia: dace.int32, kfdia: dace.int32, zepsec: dace.float64):
    for jn in range(1, 5):
        for jl in range(kidia, kfdia + 1):
            if zq[jn - 1, jl - 1] < zepsec:
                zq[4, jl - 1] = zq[4, jl - 1] + zq[jn - 1, jl - 1]
                zq[jn - 1, jl - 1] = 0.0


def canonical_sdfg() -> dace.SDFG:
    sdfg = conditional_row_accumulation.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    return sdfg


def test_the_sum_into_the_one_row_is_a_reduction_beside_the_plain_zeroing_of_another_row():
    sdfg = canonical_sdfg()

    stores = [
        (str(edge.data.subset), edge.data.wcr)
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet)
        for edge in state.out_edges(node)
        if edge.data.data == "zq"
    ]

    reductions = [subset for subset, wcr in stores if wcr is not None]
    assert len(reductions) == 1 and reductions[0].startswith("4,"), stores
    assert len(stores) == 2, stores


def test_a_conditional_accumulation_into_one_row_lifts_as_a_reduction_over_the_species():
    sdfg = canonical_sdfg()

    assert not [
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)
    ]


def test_the_lifted_species_loop_computes_the_same_sums_on_many_threads():
    sdfg = canonical_sdfg()
    # Pinned on the maps themselves: the environment's thread count must not decide whether a race can show
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.omp_num_threads = 8
    n = 20000
    zq = np.random.default_rng(0).uniform(-1.0, 1.0, (5, n))
    expected = zq.copy()
    for jn in range(1, 5):
        taken = expected[jn - 1] < 0.9
        expected[4, taken] += expected[jn - 1, taken]
        expected[jn - 1, taken] = 0.0
    sdfg(zq=zq, kidia=1, kfdia=n, zepsec=0.9, N=n)
    assert np.allclose(zq, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    test_the_sum_into_the_one_row_is_a_reduction_beside_the_plain_zeroing_of_another_row()
    test_a_conditional_accumulation_into_one_row_lifts_as_a_reduction_over_the_species()
    test_the_lifted_species_loop_computes_the_same_sums_on_many_threads()
