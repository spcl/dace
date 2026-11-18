# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A chain of copies within one array becomes one map when no copy writes what another reads or writes."""

import numpy as np

import dace
from dace.transformation.passes.offloading.fuse_copy_chains import FuseCopyChains

N = dace.symbol("N")


@dace.program
def corner_copies(q: dace.float64[6, 6, N]):
    q[0, 0] = q[0, 5]
    q[1, 0] = q[1, 4]
    q[0, 1] = q[0, 4]


@dace.program
def dependent_copies(q: dace.float64[6, 6, N]):
    q[0, 0] = q[0, 5]
    q[0, 5] = q[1, 5]


def copy_edges(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for state in sdfg.all_states()
        for e in state.edges()
        if isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
    )


def test_independent_copies_within_an_array_are_one_map():
    sdfg = corner_copies.to_sdfg(simplify=True)
    assert copy_edges(sdfg) == 3, "the frontend chains one copy per statement"
    assert FuseCopyChains().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert copy_edges(sdfg) == 0
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert len(maps) == 1 and [str(s) for s in maps[0].map.range.size()] == ["N"]

    q = np.random.default_rng(20261118).random((6, 6, 7))
    ref = q.copy()
    ref[0, 0], ref[1, 0], ref[0, 1] = ref[0, 5], ref[1, 4], ref[0, 4]
    sdfg(q=q, N=7)
    assert np.array_equal(q, ref)


def test_a_copy_overwriting_what_an_earlier_one_reads_keeps_its_order():
    sdfg = dependent_copies.to_sdfg(simplify=True)
    assert FuseCopyChains().apply_pass(sdfg, {}) is None
    assert copy_edges(sdfg) == 2


if __name__ == "__main__":
    test_independent_copies_within_an_array_are_one_map()
    test_a_copy_overwriting_what_an_earlier_one_reads_keeps_its_order()
