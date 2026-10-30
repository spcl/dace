# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InlineSDFG`` and ``InlineMultistateSDFG`` name an inlined transient apart from the connectors of the parent's code nodes, which validation
forbids data to share (an expanded copy's tasklet reads ``_cpy_in``, and so is named every copy nest's local view)."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import CopyLibraryNode
from dace.transformation.interstate import InlineMultistateSDFG, InlineSDFG

#: The connector names of an expanded copy's tasklet, which every copy nest also gives its local views.
CPY_IN, CPY_OUT = CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME


def parent_with_a_cpy_in_connector_and_a_nest() -> dace.SDFG:
    """A tasklet with the copy connectors, beside a nest computing through its own transient named like the input
    connector."""
    nest = dace.SDFG("nest")
    nest.add_array("x", [4], dace.float64)
    nest.add_array("y", [4], dace.float64)
    nest.add_transient(CPY_IN, [4], dace.float64)
    ns = nest.add_state("s")
    ns.add_mapped_tasklet(
        "double",
        dict(i="0:4"),
        {"a": dace.Memlet("x[i]")},
        "b = 2 * a",
        {"b": dace.Memlet(f"{CPY_IN}[i]")},
        external_edges=True,
    )
    tmp = next(n for n in ns.data_nodes() if n.data == CPY_IN)
    ns.add_mapped_tasklet(
        "copy_back",
        dict(i="0:4"),
        {"a": dace.Memlet(f"{CPY_IN}[i]")},
        "b = a",
        {"b": dace.Memlet("y[i]")},
        external_edges=True,
        input_nodes={CPY_IN: tmp},
    )

    sdfg = dace.SDFG("inline_beside_a_cpy_in_connector")
    for name in ("A", "B", "C", "D"):
        sdfg.add_array(name, [4], dace.float64)
    state = sdfg.add_state("s")
    state.add_mapped_tasklet(
        "copy_C_to_D",
        dict(i="0:4"),
        {CPY_IN: dace.Memlet("C[i]")},
        f"{CPY_OUT} = {CPY_IN}",
        {CPY_OUT: dace.Memlet("D[i]")},
        external_edges=True,
    )
    node = state.add_nested_sdfg(nest, {"x"}, {"y"})
    state.add_edge(state.add_read("A"), None, node, "x", dace.Memlet("A[0:4]"))
    state.add_edge(node, "y", state.add_write("B"), None, dace.Memlet("B[0:4]"))
    sdfg.validate()
    return sdfg


def test_an_inlined_transient_is_named_apart_from_a_connector():
    sdfg = parent_with_a_cpy_in_connector_and_a_nest()

    assert sdfg.apply_transformations_repeated(InlineSDFG) == 1
    sdfg.validate()
    assert CPY_IN not in sdfg.arrays

    a, c = np.arange(4.0), np.arange(4.0) + 10
    b, d = np.zeros(4), np.zeros(4)
    sdfg(A=a, B=b, C=c, D=d)
    assert np.allclose(b, 2 * a) and np.allclose(d, c)


@pytest.mark.parametrize("language", [dace.Language.Python, dace.Language.CPP])
@pytest.mark.parametrize("inliner", [InlineSDFG, InlineMultistateSDFG])
def test_inlined_tasklet_connectors_are_renamed_apart_from_parent_data_and_symbols(inliner, language):
    """A nested ``N = B * 2`` tasklet, inlined into a parent with an array ``B`` and a symbol ``N``."""
    nest = dace.SDFG("nest")
    nest.add_array("inp", [1], dace.float64)
    nest.add_array("out", [1], dace.float64)
    ns = nest.add_state("s", is_start_block=True)
    code = "N = B * 2" if language == dace.Language.Python else "N = B * 2;"
    tasklet = ns.add_tasklet("double", {"B": None}, {"N": None}, code, language=language)
    ns.add_edge(ns.add_read("inp"), None, tasklet, "B", dace.Memlet("inp[0]"))
    ns.add_edge(tasklet, "N", ns.add_write("out"), None, dace.Memlet("out[0]"))
    if inliner is InlineMultistateSDFG:
        nest.add_state_after(ns)

    sdfg = dace.SDFG(f"inline_connectors_named_like_parent_names_{inliner.__name__}_{language.name}")
    for name in ("X", "Y", "B"):
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_symbol("N", dace.int64)
    state = sdfg.add_state("call", is_start_block=True)
    node = state.add_nested_sdfg(nest, {"inp": None}, {"out": None}, symbol_mapping={})
    state.add_edge(state.add_read("X"), None, node, "inp", dace.Memlet("X[0]"))
    state.add_edge(node, "out", state.add_write("Y"), None, dace.Memlet("Y[0]"))
    after = sdfg.add_state_after(state, "use")
    tasklet = after.add_tasklet("read_symbol", {}, {"o": None}, "o = N")
    after.add_edge(tasklet, "o", after.add_write("B"), None, dace.Memlet("B[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(inliner) == 1
    sdfg.validate()

    x, y, b = np.full(1, 3.0), np.zeros(1), np.zeros(1)
    sdfg(X=x, Y=y, B=b, N=7)
    assert y[0] == 6 and b[0] == 7, (y, b)


if __name__ == "__main__":
    test_an_inlined_transient_is_named_apart_from_a_connector()
    for inliner in [InlineSDFG, InlineMultistateSDFG]:
        for language in [dace.Language.Python, dace.Language.CPP]:
            test_inlined_tasklet_connectors_are_renamed_apart_from_parent_data_and_symbols(inliner, language)
