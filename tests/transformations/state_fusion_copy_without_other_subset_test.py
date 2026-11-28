# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A copy edge that names only the OTHER container says nothing about the subset it reads here.

``p[0]`` is written in the first state and copied to ``a[k]`` in the second; that copy's memlet is ``a[k]`` with no
``other_subset``. Comparing ``p[0]`` against ``a[k]`` (the other container's subset) is meaningless, and with ``k``
known positive it is "provably disjoint" -- so state fusion took the read for independent of the write and merged it
into the wrong ``p`` node (a TSVC s118 miscompile: the write-back ran before the reduction).
"""

import dace
from dace.symbolic import Predicate
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended


def write_then_copy_out() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG("write_then_copy_out")
    sdfg.add_symbol("k", dace.int64, predicates=frozenset({Predicate.POSITIVE}))
    sdfg.add_array("a", [16], dace.float64)
    sdfg.add_scalar("p", dace.float64, transient=True)
    first = sdfg.add_state("first")
    tasklet = first.add_tasklet("produce", {}, {"out"}, "out = 1.0")
    first.add_edge(tasklet, "out", first.add_write("p"), None, dace.Memlet("p[0]"))
    second = sdfg.add_state_after(first, "second")
    second.add_nedge(second.add_read("p"), second.add_write("a"), dace.Memlet("a[k]"))
    return sdfg, first, second


def test_a_read_through_a_copy_without_other_subset_intersects_the_write():
    for fusion in (StateFusion, StateFusionExtended):
        sdfg, first, second = write_then_copy_out()
        written = next(n for n in first.data_nodes() if n.data == "p")
        read = next(n for n in second.data_nodes() if n.data == "p")
        assert fusion.memlets_intersect(first, [written], False, second, [read], True), fusion.__name__


def test_an_ordering_edge_is_no_access():
    """An empty Memlet into ``p`` only orders: it neither writes nor reads ``p``, so it cannot intersect ``p[0]``."""
    for fusion in (StateFusion, StateFusionExtended):
        sdfg, first, second = write_then_copy_out()
        written = next(n for n in first.data_nodes() if n.data == "p")
        ordered = second.add_access("p")
        second.add_nedge(second.add_read("a"), ordered, dace.Memlet())
        assert not fusion.memlets_intersect(first, [written], False, second, [ordered], False), fusion.__name__


if __name__ == "__main__":
    test_a_read_through_a_copy_without_other_subset_intersects_the_write()
    test_an_ordering_edge_is_no_access()
