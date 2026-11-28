# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FillThroughCopies: a copy of a constant-filled array in the fill's state becomes a fill of its destination.

warpx_esirkepov_deposition writes ``i_new = np.zeros(n)`` and then ``i_old = np.copy(i_new)`` four more times; the
CPF form kept five copies of the zeros (SciComp review of 11-23).
"""

import numpy as np

import dace
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.fill_through_copies import FillThroughCopies


def fill_then_copies_sdfg(dst_dtype: dace.typeclass = dace.float64) -> dace.SDFG:
    """``src = 3.0``, then ``a = src`` and ``b = src``; ``out = src + 1`` keeps ``src`` read."""
    sdfg = dace.SDFG("fill_then_copies")
    sdfg.add_array("src", [16], dace.float64, transient=True)
    sdfg.add_array("a", [16], dst_dtype)
    sdfg.add_array("b", [16], dace.float64)
    sdfg.add_array("out", [16], dace.float64)
    state = sdfg.add_state("main")
    fill = FillLibraryNode("fill_src", value=3.0)
    src = state.add_access("src")
    state.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, src, None, dace.Memlet("src[0:16]"))
    state.add_edge(src, None, state.add_write("a"), None, dace.Memlet("a[0:16]"))
    state.add_edge(src, None, state.add_write("b"), None, dace.Memlet("b[0:16]"))
    state.add_mapped_tasklet(
        "inc",
        {"i": "0:16"},
        {"x": dace.Memlet("src[i]")},
        "y = x + 1.0",
        {"y": dace.Memlet("out[i]")},
        input_nodes={"src": src},
        external_edges=True,
    )
    return sdfg


def copies_between_access_nodes(sdfg: dace.SDFG) -> int:
    state = sdfg.start_state
    return sum(isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode) for e in state.edges())


def test_copies_of_a_fill_become_fills():
    sdfg = fill_then_copies_sdfg()
    assert FillThroughCopies().apply_pass(sdfg, {}) == 2
    assert copies_between_access_nodes(sdfg) == 0
    assert sum(isinstance(n, FillLibraryNode) for n in sdfg.start_state.nodes()) == 3
    sdfg.validate()
    a, b, out = np.zeros(16), np.zeros(16), np.zeros(16)
    sdfg(a=a, b=b, out=out)
    assert np.all(a == 3.0) and np.all(b == 3.0) and np.all(out == 4.0)


def test_a_converting_copy_stays():
    sdfg = fill_then_copies_sdfg(dst_dtype=dace.int64)
    assert FillThroughCopies().apply_pass(sdfg, {}) == 1
    assert copies_between_access_nodes(sdfg) == 1
    sdfg.validate()


if __name__ == "__main__":
    test_copies_of_a_fill_become_fills()
    test_a_converting_copy_stays()
