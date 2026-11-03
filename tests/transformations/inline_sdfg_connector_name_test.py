# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InlineSDFG`` names an inlined transient apart from the connectors of the parent's code nodes, which validation
forbids data to share (an expanded copy's tasklet reads ``_cpy_in``, and so is named every copy nest's local view)."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import CopyLibraryNode
from dace.transformation.interstate import InlineSDFG
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

#: The connector names of an expanded copy's tasklet, which every copy nest also gives its local views.
CPY_IN, CPY_OUT = CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME


def test_a_multistate_inlined_transient_is_named_apart_from_a_parent_connector():
    """A multistate nest's transient ``tmp`` beside a parent tasklet that reads through a connector ``tmp``."""
    nest = dace.SDFG("nest")
    nest.add_array("out", [1], dace.float64)
    nest.add_transient("tmp", [1], dace.float64)
    first = nest.add_state("first", is_start_block=True)
    tasklet = first.add_tasklet("produce", {}, {"o"}, "o = 2")
    first.add_edge(tasklet, "o", first.add_write("tmp"), None, dace.Memlet("tmp[0]"))
    second = nest.add_state_after(first, "second")
    tasklet = second.add_tasklet("consume", {"i"}, {"o"}, "o = i + 1")
    second.add_edge(second.add_read("tmp"), None, tasklet, "i", dace.Memlet("tmp[0]"))
    second.add_edge(tasklet, "o", second.add_write("out"), None, dace.Memlet("out[0]"))

    sdfg = dace.SDFG("multistate_inline_beside_a_tmp_connector")
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_array("res", [1], dace.float64)
    state = sdfg.add_state("call", is_start_block=True)
    node = state.add_nested_sdfg(nest, {}, {"out"})
    state.add_edge(node, "out", state.add_write("out"), None, dace.Memlet("out[0]"))
    after = sdfg.add_state_after(state, "scale")
    tasklet = after.add_tasklet("scale", {"tmp"}, {"o"}, "o = tmp * 10")
    after.add_edge(after.add_read("out"), None, tasklet, "tmp", dace.Memlet("out[0]"))
    after.add_edge(tasklet, "o", after.add_write("res"), None, dace.Memlet("res[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(InlineMultistateSDFG) == 1
    sdfg.validate()

    out, res = np.zeros(1), np.zeros(1)
    sdfg(out=out, res=res)
    assert out[0] == 3 and res[0] == 30, (out, res)


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


def test_a_multistate_inlined_transient_is_named_apart_from_a_connector():
    sdfg = parent_with_a_cpy_in_connector_and_a_nest()

    assert sdfg.apply_transformations_repeated(InlineMultistateSDFG) == 1
    sdfg.validate()
    assert CPY_IN not in sdfg.arrays

    a, c = np.arange(4.0), np.arange(4.0) + 10
    b, d = np.zeros(4), np.zeros(4)
    sdfg(A=a, B=b, C=c, D=d)
    assert np.allclose(b, 2 * a) and np.allclose(d, c)


def parent_transient_named_like_a_nested_connector() -> dace.SDFG:
    """A parent computing through a transient named like the input connector of a tasklet inside a nest."""
    nest = dace.SDFG("nest")
    nest.add_array("x", [4], dace.float64)
    nest.add_array("y", [4], dace.float64)
    nest.add_state("s").add_mapped_tasklet(
        "copy_x_to_y",
        dict(i="0:4"),
        {CPY_IN: dace.Memlet("x[i]")},
        f"{CPY_OUT} = {CPY_IN}",
        {CPY_OUT: dace.Memlet("y[i]")},
        external_edges=True,
    )

    sdfg = dace.SDFG("inline_a_connector_beside_a_transient")
    for name in ("A", "B", "C", "D"):
        sdfg.add_array(name, [4], dace.float64)
    sdfg.add_transient(CPY_IN, [4], dace.float64)
    state = sdfg.add_state("s")
    tmp = state.add_access(CPY_IN)
    state.add_edge(state.add_read("C"), None, tmp, None, dace.Memlet(f"{CPY_IN}[0:4]"))
    state.add_edge(tmp, None, state.add_write("D"), None, dace.Memlet(f"{CPY_IN}[0:4]"))
    node = state.add_nested_sdfg(nest, {"x"}, {"y"})
    state.add_edge(state.add_read("A"), None, node, "x", dace.Memlet("A[0:4]"))
    state.add_edge(node, "y", state.add_write("B"), None, dace.Memlet("B[0:4]"))
    sdfg.validate()
    return sdfg


def test_an_inlined_connector_moves_aside_for_a_parent_transient():
    sdfg = parent_transient_named_like_a_nested_connector()

    assert sdfg.apply_transformations_repeated(InlineMultistateSDFG) == 1
    sdfg.validate()
    connectors = {
        c for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) for c in n.in_connectors
    }
    assert CPY_IN in sdfg.arrays and CPY_IN not in connectors

    a, c = np.arange(4.0), np.arange(4.0) + 10
    b, d = np.zeros(4), np.zeros(4)
    sdfg(A=a, B=b, C=c, D=d)
    assert np.allclose(b, a) and np.allclose(d, c)


def test_a_tasklet_naming_its_container_in_code_reads_it_under_the_outer_name():
    """CloudSC's flux nest binds connector ``zlneg_0`` to the outer ``zlneg``, and the readable generator's
    ``InlineTaskletConnectors`` wrote ``zlneg_0[1, ...]`` into the tasklet's code; inlined, the code reads ``zlneg``."""
    nest = dace.SDFG("nest_named_in_code")
    nest.add_symbol("i", dace.int64)
    nest.add_array("a_0", [2, 4], dace.float64)
    nest.add_array("y", [4], dace.float64)
    ns = nest.add_state("s")
    t = ns.add_tasklet("scale", {"__in1"}, {"__out"}, "__out = a_0[1, i] * 2.0")
    ns.add_edge(ns.add_read("a_0"), None, t, "__in1", dace.Memlet("a_0[1, i]"))
    ns.add_edge(t, "__out", ns.add_write("y"), None, dace.Memlet("y[i]"))

    sdfg = dace.SDFG("inline_code_named_container")
    sdfg.add_array("A", [2, 4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    state = sdfg.add_state("s")
    me, mx = state.add_map("m", dict(i="0:4"))
    node = state.add_nested_sdfg(nest, {"a_0"}, {"y"}, symbol_mapping={"i": "i"})
    state.add_memlet_path(state.add_read("A"), me, node, dst_conn="a_0", memlet=dace.Memlet("A[1, i]"))
    state.add_memlet_path(node, mx, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    sdfg.validate()

    assert sdfg.apply_transformations(InlineSDFG) == 1
    sdfg.validate()
    (tasklet,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert "a_0" not in tasklet.code.as_string and "A[1, i]" in tasklet.code.as_string
    a, b = np.arange(8.0).reshape(2, 4), np.zeros(4)
    sdfg(A=a, B=b)
    assert np.allclose(b, 2.0 * a[1])


def test_a_cpp_tasklet_local_named_like_a_renamed_container_stays_a_local():
    """ONNX Dropout's C++ tasklet declares a local ``ratio`` beside the container ``ratio`` it reads through
    ``__ratio``; inlining renames the container, and the local must not be shadowed by a second declaration."""
    nest = dace.SDFG("nest_cpp_local")
    nest.add_array("ratio", [1], dace.float64)
    nest.add_array("y", [1], dace.float64)
    ns = nest.add_state("s")
    t = ns.add_tasklet(
        "scale", {"__ratio"}, {"__out"}, "double ratio = __ratio;\n__out = 2.0 * ratio;", language=dace.Language.CPP
    )
    ns.add_edge(ns.add_read("ratio"), None, t, "__ratio", dace.Memlet("ratio[0]"))
    ns.add_edge(t, "__out", ns.add_write("y"), None, dace.Memlet("y[0]"))

    sdfg = dace.SDFG("inline_cpp_local")
    sdfg.add_array("R", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state("s")
    node = state.add_nested_sdfg(nest, {"ratio"}, {"y"})
    state.add_edge(state.add_read("R"), None, node, "ratio", dace.Memlet("R[0]"))
    state.add_edge(node, "y", state.add_write("B"), None, dace.Memlet("B[0]"))

    assert sdfg.apply_transformations(InlineSDFG) == 1
    b = np.zeros(1)
    sdfg(R=np.array([0.25]), B=b)
    assert b[0] == 0.5


def nest_with_a_tasklet_connector_named(clash: str, side: str, language: dace.Language) -> dace.SDFG:
    """``out = inp * 2`` in a single-state nest whose tasklet names its ``side`` connector ``clash``: the parent
    owns an array ``B`` and a symbol ``N``."""
    inner = dace.SDFG("clash_inner")
    inner.add_array("inp", [1], dace.float64)
    inner.add_array("out", [1], dace.float64)
    body = inner.add_state("body", is_start_block=True)
    i, o = (clash, "o") if side == "in" else ("i", clash)
    t = body.add_tasklet(
        "t", {i}, {o}, f"{o} = {i} * 2" + (";" if language is dace.Language.CPP else ""), language=language
    )
    body.add_edge(body.add_read("inp"), None, t, i, dace.Memlet("inp[0]"))
    body.add_edge(t, o, body.add_write("out"), None, dace.Memlet("out[0]"))

    sdfg = dace.SDFG("clash_outer")
    for name in "XYB":
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_symbol("N", dace.int64)
    state = sdfg.add_state("main", is_start_block=True)
    nest = state.add_nested_sdfg(inner, {"inp"}, {"out"})
    state.add_edge(state.add_read("X"), None, nest, "inp", dace.Memlet("X[0]"))
    state.add_edge(nest, "out", state.add_write("Y"), None, dace.Memlet("Y[0]"))
    use = sdfg.add_state_after(state, "use")
    u = use.add_tasklet("u", {}, {"o"}, "o = N")
    use.add_edge(u, "o", use.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg


@pytest.mark.parametrize("language", [dace.Language.Python, dace.Language.CPP])
@pytest.mark.parametrize("side", ["in", "out"])
@pytest.mark.parametrize("clash", ["B", "N"])
def test_a_single_state_inlined_tasklet_connector_named_like_a_parent_name_is_renamed(clash, side, language):
    """A connector named like the parent's array ``B`` or symbol ``N`` fails validation once inlined, unless it is
    renamed in its edges and code (as InlineMultistateSDFG does)."""
    sdfg = nest_with_a_tasklet_connector_named(clash, side, language)
    assert sdfg.apply_transformations(InlineSDFG) == 1
    sdfg.validate()
    x, y, b = np.array([3.0]), np.zeros(1), np.zeros(1)
    sdfg(X=x, Y=y, B=b, N=7)
    assert y[0] == 6.0 and b[0] == 7.0


if __name__ == "__main__":
    test_a_multistate_inlined_transient_is_named_apart_from_a_parent_connector()
    test_an_inlined_transient_is_named_apart_from_a_connector()
    test_a_multistate_inlined_transient_is_named_apart_from_a_connector()
    test_an_inlined_connector_moves_aside_for_a_parent_transient()
    test_a_tasklet_naming_its_container_in_code_reads_it_under_the_outer_name()
    test_a_cpp_tasklet_local_named_like_a_renamed_container_stays_a_local()
    for clash in "BN":
        for side in ("in", "out"):
            for language in (dace.Language.Python, dace.Language.CPP):
                test_a_single_state_inlined_tasklet_connector_named_like_a_parent_name_is_renamed(clash, side, language)
