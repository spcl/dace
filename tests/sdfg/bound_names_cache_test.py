# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFG.bound_names`` stays exact while additions extend its cache instead of dropping it."""

import dace
from dace.sdfg.sdfg import interstate_bound_names, scope_bound_names


def walked(sdfg: dace.SDFG) -> tuple[frozenset[str], frozenset[str]]:
    return frozenset(scope_bound_names(sdfg)), frozenset(interstate_bound_names(sdfg))


def test_the_cache_extended_by_additions_equals_a_fresh_walk():
    sdfg = dace.SDFG("bound_names_cache")
    sdfg.add_array("A", [8], dace.float64)
    first = sdfg.add_state("first")
    sdfg.bound_names()  # warm: every addition below goes through the cache
    first.add_mapped_tasklet(
        "scale", {"i": "0:8"}, {"a": dace.Memlet("A[i]")}, "b = 2 * a", {"b": dace.Memlet("A[i]")}, external_edges=True
    )
    assert "i" in sdfg.bound_names()[0]
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"k": "1"}))
    assert "k" in sdfg.bound_names()[1]
    loop = dace.sdfg.state.LoopRegion("loop", "j < 4", "j", "j = 0", "j = j + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(second, loop, dace.InterstateEdge())
    assert sdfg.bound_names() == walked(sdfg)


if __name__ == "__main__":
    test_the_cache_extended_by_additions_equals_a_fresh_walk()
