# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC's ``IF (ZQXN(JN) < ZEPSEC)`` update of the one row ``ZQXN(NCLDQV)``, summed over the species ``JN``.

Every species iteration adds its own row to row ``4`` and zeroes its own row, under a data-dependent guard. Rows ``0..3``
never meet row ``4``, so the species loop is a ``+`` reduction into row ``4`` with the identity ``0`` where the guard is
false, and canonicalization should lift it.

Two things stopped it. The frontend orders the snapshot of ``ZQXN(4, JL)`` before the zeroing write of
``ZQXN(JN-1, JL)`` with an ordering edge, and ``JN - 1 != 4`` needs ``JN < 5`` from the loop header: with the edge,
``FoldScalarReadCopies`` kept the snapshot, and ``AugAssignToWCR`` never saw ``zq[4] = zq[4] + x``. The fold now asks
the facts at the state (the enclosing loops' ranges) whether the written row lies apart from the read one.

Still open: once the accumulation is a WCR, ``LoopToMap`` unions the loop's two writes (row ``JN-1`` and the WCR row 4)
into the box ``JN-1:5`` and finds the read of row ``JN-1`` in conflict with it.
"""

import pytest

import dace
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


def test_the_update_of_the_one_row_reads_it_in_place_beside_a_write_of_another_row():
    sdfg = canonical_sdfg()

    update = next(
        (state, node)
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.Tasklet) and node.label.startswith("_Add_")
    )
    state, tasklet = update
    reads = [state.memlet_path(e)[0].src for e in state.in_edges(tasklet)]

    assert any(isinstance(src, dace.nodes.AccessNode) and src.data == "zq" for src in reads), reads


@pytest.mark.xfail(strict=True, reason="LoopToMap unions the WCR row and the zeroed row into one write box")
def test_a_conditional_accumulation_into_one_row_lifts_as_a_reduction_over_the_species():
    sdfg = canonical_sdfg()

    assert not [
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)
    ]


if __name__ == "__main__":
    test_the_update_of_the_one_row_reads_it_in_place_beside_a_write_of_another_row()
    # test_a_conditional_accumulation_into_one_row_lifts_as_a_reduction_over_the_species() -- xfail until lifted
