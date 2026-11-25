# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ReadableNames renames a map parameter on both sides of a nested SDFG connector whose container it sizes.

QE ``vexx_bp_k_gpu``: the FFT view ``prhoc_d(1:nrxxs*jcurr)`` is sized by the band map's parameter ``_loop_it_59`` and
written by an FFT. Rendering renamed the parameter to ``l`` in the map scope but left the view's descriptor on
``_loop_it_59``; the FFT's expansion took ``l`` from the memlets, and validation failed ("the internal data descriptor ... is not equivalent").
"""

import numpy as np

import dace
from dace.transformation.passes.readable_names import ReadableNames

N = dace.symbol("N")
IT = "_loop_it_5"


def fill_prefix() -> dace.SDFG:
    """``out[j] = it`` for ``j < it``; ``out`` is sized by the imported iterator."""
    inner = dace.SDFG("fill_prefix")
    inner.add_symbol(IT, dace.int64)
    inner.add_array("out", [dace.symbol(IT)], dace.float64)
    state = inner.add_state("fill")
    state.add_mapped_tasklet(
        "fill", {"j": f"0:{IT}"}, {}, f"o = {IT}", {"o": dace.Memlet("out[j]")}, external_edges=True
    )
    return inner


def map_sdfg() -> dace.SDFG:
    """``for it in 1..N in parallel: v = A[it - 1, 0:it] (a view); fill_prefix(v)``: LoopToMap's map keeps the loop's
    ``_loop_it_<N>`` as its parameter, so the view's shape names a map parameter."""
    sdfg = dace.SDFG("readable_names_map_view")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_view("v", [dace.symbol(IT)], dace.float64)
    state = sdfg.add_state("body")
    entry, exit_ = state.add_map("rows", {IT: "1:N+1"})
    view = state.add_access("v")
    node = state.add_nested_sdfg(fill_prefix(), {}, {"out"}, symbol_mapping={IT: IT})
    state.add_nedge(entry, node, dace.Memlet())
    state.add_edge(node, "out", view, None, dace.Memlet(f"v[0:{IT}]"))
    exit_.add_in_connector("IN_A")
    exit_.add_out_connector("OUT_A")
    state.add_edge(view, "views", exit_, "IN_A", dace.Memlet(f"A[{IT} - 1, 0:{IT}]"))
    state.add_edge(exit_, "OUT_A", state.add_write("A"), None, dace.Memlet("A[0:N, 0:N]"))
    return sdfg


def test_a_map_parameter_sizing_a_view_into_a_nested_sdfg_is_renamed_on_both_sides():
    sdfg = map_sdfg()
    sdfg.validate()
    ReadableNames().apply_pass(sdfg, {})
    sdfg.validate()
    a = np.zeros((5, 5))
    sdfg(A=a, N=5)
    assert np.array_equal(a, np.tril(np.arange(1.0, 6.0)[:, None] * np.ones((5, 5))))


if __name__ == "__main__":
    test_a_map_parameter_sizing_a_view_into_a_nested_sdfg_is_renamed_on_both_sides()
