# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ReadableNames gives generated containers short free names, also when their stem starts with a reserved prefix."""

import numpy as np
import pytest

import dace
from dace.ordered import OrderedSet
from dace.transformation.passes.readable_names import ReadableNames, fresh, is_free


@pytest.mark.parametrize("stem", ["cpf_tmp", "omp_buf", "dace_acc", "__x"])
def test_a_reserved_stem_gets_a_free_name(stem: str):
    taken = OrderedSet(["tmp"])
    name = fresh(stem, taken)
    assert name == "tmp1" and name in taken and is_free("tmp2", taken)


def test_a_transient_with_a_reserved_stem_is_renamed_and_computes_the_same():
    sdfg = dace.SDFG("reserved_stem")
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("b", [8], dace.float64)
    sdfg.add_transient("__cpf_stage", [8], dace.float64)
    state = sdfg.add_state()
    staged = state.add_access("__cpf_stage")
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:8"},
        {"x": dace.Memlet("a[i]")},
        "y = 2 * x",
        {"y": dace.Memlet("__cpf_stage[i]")},
        output_nodes={"__cpf_stage": staged},
        external_edges=True,
    )
    state.add_nedge(staged, state.add_write("b"), dace.Memlet("__cpf_stage[0:8]"))
    ReadableNames().apply_pass(sdfg, {})
    assert "__cpf_stage" not in sdfg.arrays and "tmp" in sdfg.arrays
    a, b = np.arange(8.0), np.zeros(8)
    sdfg(a=a, b=b)
    np.testing.assert_array_equal(b, 2 * a)


def test_a_renamed_symbol_reaches_the_cpp_code_of_a_library_expansion():
    """syrk's alpha is an interstate symbol the OpenBLAS expansion reads only in its C++ call; the rename must reach
    that call, or the rendered text names a symbol nothing declares (Polybench syrk, syr2k, symm)."""
    from dace.libraries.blas.nodes.syrk import Syrk

    sdfg = dace.SDFG("syrk_symbol_alpha")
    sdfg.add_array("A", [8, 4], dace.float64)
    sdfg.add_array("C", [8, 8], dace.float64)
    sdfg.add_array("alpha", [1], dace.float64)
    sdfg.add_symbol("__alpha_value", dace.float64)
    start = sdfg.add_state("start", is_start_block=True)
    state = sdfg.add_state_after(start, "call", assignments={"__alpha_value": "alpha[0]"})
    node = Syrk("syrk", alpha=dace.symbolic.symbol("__alpha_value", dace.float64), beta=1)
    node.implementation = "OpenBLAS"
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_A", dace.Memlet("A[0:8, 0:4]"))
    state.add_edge(state.add_read("C"), None, node, "_Cin", dace.Memlet("C[0:8, 0:8]"))
    state.add_edge(node, "_Cout", state.add_write("C"), None, dace.Memlet("C[0:8, 0:8]"))
    node.expand(state)
    ReadableNames().apply_pass(sdfg, {})
    calls = [
        n.code.as_string
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and "cblas_dsyrk" in n.code.as_string
    ]
    assert len(calls) == 1 and "__alpha_value" not in calls[0]
    assert "__alpha_value" not in sdfg.symbols


def test_an_iterator_read_after_its_loop_keeps_one_name():
    """A nest exports a loop iterator's final value from a state after the loop (Polybench trmm's ``np.dot`` row
    loop); renaming the iterator inside the loop only left that read naming an undefined symbol."""
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG("iterator_read_after_loop")
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("last", [1], dace.int64)
    sdfg.add_symbol("_it", dace.int64)
    loop = LoopRegion("loop", "_it < 8", "_it", "_it = 0", "_it = _it + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("set", {}, {"out"}, "out = _it")
    body.add_edge(tasklet, "out", body.add_write("a"), None, dace.Memlet("a[_it]"))
    after = sdfg.add_state_after(loop, "after")
    export = after.add_tasklet("export", {}, {"out"}, "out = _it")
    after.add_edge(export, "out", after.add_write("last"), None, dace.Memlet("last[0]"))
    ReadableNames().apply_pass(sdfg, {})
    assert export.code.as_string.strip() == f"out = {loop.loop_variable}"


def test_a_map_parameter_named_like_its_enclosing_loop_counter_is_renamed():
    """vexx_k: a map over ``i`` inside the ``for i`` loop rendered as ``for (int64_t i ...)`` inside ``for (i ...)``,
    shadowing the counter the rest of the loop body reads."""
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG("shadowed_counter")
    sdfg.add_array("a", ["N", "M"], dace.float64)
    loop = LoopRegion("rows", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_mapped_tasklet("w", {"i": "0:M"}, {}, "y = 1.0", {"y": dace.Memlet("a[0, i]")}, external_edges=True)
    ReadableNames().apply_pass(sdfg, {})
    (entry,) = [n for n in body.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert loop.loop_variable == "i" and entry.map.params[0] != "i", entry.map.params
    (edge,) = [e for e in body.edges() if isinstance(e.dst, dace.nodes.MapExit)]
    assert str(edge.data.subset) == f"0, {entry.map.params[0]}", edge.data.subset


if __name__ == "__main__":
    for stem in ("cpf_tmp", "omp_buf", "dace_acc", "__x"):
        test_a_reserved_stem_gets_a_free_name(stem)
    test_a_transient_with_a_reserved_stem_is_renamed_and_computes_the_same()
    test_a_renamed_symbol_reaches_the_cpp_code_of_a_library_expansion()
    test_an_iterator_read_after_its_loop_keeps_one_name()
    test_a_map_parameter_named_like_its_enclosing_loop_counter_is_renamed()
