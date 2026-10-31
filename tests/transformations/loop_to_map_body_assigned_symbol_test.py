# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` in a nested SDFG moves a symbol the loop body assigns into the new loop body, so the enclosing nested
SDFG node must not map it: the parent defines no such symbol, and a binding that freezes the parent's signature sees a
free symbol nothing can supply."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

N = 8


def gather_through_an_assigned_symbol() -> tuple[dace.SDFG, dace.nodes.NestedSDFG]:
    inner = dace.SDFG("gather")
    inner.add_array("a", [N], dace.float64)
    inner.add_array("b", [N], dace.float64)
    inner.add_array("idx", [N], dace.int32)
    inner.add_symbol("k", dace.int32)
    loop = LoopRegion("loop", "i < 8", "i", "i = 0", "i = i + 1")
    inner.add_node(loop, is_start_block=True)
    load = loop.add_state("load", is_start_block=True)
    body = loop.add_state("body")
    loop.add_edge(load, body, dace.InterstateEdge(assignments={"k": "idx[i]"}))
    tasklet = body.add_tasklet("copy", {"x"}, {"y"}, "y = x")
    body.add_edge(body.add_read("a"), None, tasklet, "x", dace.Memlet("a[k]"))
    body.add_edge(tasklet, "y", body.add_write("b"), None, dace.Memlet("b[i]"))

    sdfg = dace.SDFG("gather_outer")
    for name, dtype in (("a", dace.float64), ("b", dace.float64), ("idx", dace.int32)):
        sdfg.add_array(name, [N], dtype)
    state = sdfg.add_state()
    node = state.add_nested_sdfg(inner, {"a", "idx"}, {"b"})
    state.add_edge(state.add_read("a"), None, node, "a", dace.Memlet("a[0:8]"))
    state.add_edge(state.add_read("idx"), None, node, "idx", dace.Memlet("idx[0:8]"))
    state.add_edge(node, "b", state.add_write("b"), None, dace.Memlet("b[0:8]"))
    return sdfg, node


def test_a_symbol_the_body_assigns_is_not_mapped_into_the_enclosing_nest():
    sdfg, node = gather_through_an_assigned_symbol()
    assert node.sdfg.apply_transformations_repeated(LoopToMap) == 1

    assert "k" not in node.symbol_mapping
    assert "k" not in sdfg.free_symbols
    sdfg.validate()
    a, idx, b = np.arange(N, dtype=np.float64), np.arange(N, dtype=np.int32)[::-1].copy(), np.zeros(N)
    sdfg(a=a, idx=idx, b=b)
    assert np.array_equal(b, a[idx])


if __name__ == "__main__":
    test_a_symbol_the_body_assigns_is_not_mapped_into_the_enclosing_nest()
