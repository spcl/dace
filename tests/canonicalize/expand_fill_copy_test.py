# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ExpandFillCopy``: fill and copy library nodes become maps before fusion, so fusion joins them with their
neighbours; a rank-changing copy stays a library call."""

import numpy as np

import dace
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.transformation.passes.canonicalize.expand_fill_copy import ExpandFillCopy
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")


def fill_then_accumulate() -> dace.SDFG:
    """``t[:] = 0; b[i] = 2 * (t[i] + a[i])``, the fill a library node."""
    sdfg = dace.SDFG("fill_then_accumulate")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    state = sdfg.add_state()
    filled = state.add_access("t")
    fill = FillLibraryNode("fill_t", value=0)
    state.add_node(fill)
    state.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, filled, None, dace.Memlet("t[0:N]"))
    state.add_mapped_tasklet(
        "use",
        {"i": "0:N"},
        {"x": dace.Memlet("t[i]"), "y": dace.Memlet("a[i]")},
        "o = 2 * (x + y)",
        {"o": dace.Memlet("b[i]")},
        external_edges=True,
        input_nodes={"t": filled},
    )
    sdfg.validate()
    return sdfg


def copy_between_maps(target_shape: tuple = (N,)) -> dace.SDFG:
    """``t[i] = 2 * a[i]; u = copy(t); b[...] = u[...] + 1``; ``target_shape`` other than ``(N,)`` reshapes."""
    sdfg = dace.SDFG(f"copy_between_maps_{len(target_shape)}")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", list(target_shape), dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    sdfg.add_transient("u", list(target_shape), dace.float64)
    state = sdfg.add_state()
    produced, copied = state.add_access("t"), state.add_access("u")
    state.add_mapped_tasklet(
        "produce",
        {"i": "0:N"},
        {"x": dace.Memlet("a[i]")},
        "o = 2 * x",
        {"o": dace.Memlet("t[i]")},
        external_edges=True,
        output_nodes={"t": produced},
    )
    copy = CopyLibraryNode("copy_t_u")
    state.add_node(copy)
    state.add_edge(produced, None, copy, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("t[0:N]"))
    full = ", ".join(f"0:{s}" for s in target_shape)
    state.add_edge(copy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, copied, None, dace.Memlet(f"u[{full}]"))
    params = {f"i{d}": f"0:{s}" for d, s in enumerate(target_shape)}
    index = ", ".join(params)
    state.add_mapped_tasklet(
        "consume",
        params,
        {"x": dace.Memlet(f"u[{index}]")},
        "o = x + 1",
        {"o": dace.Memlet(f"b[{index}]")},
        external_edges=True,
        input_nodes={"u": copied},
    )
    sdfg.validate()
    return sdfg


def count(sdfg: dace.SDFG, kind: type) -> int:
    return sum(isinstance(n, kind) for n, _ in sdfg.all_nodes_recursive())


def top_maps(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, dace.nodes.MapEntry) and s.entry_node(n) is None for s in sdfg.states() for n in s.nodes())


def test_fill_expands_and_fuses_with_its_reader():
    sdfg = fill_then_accumulate()
    assert ExpandFillCopy().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert count(sdfg, FillLibraryNode) == 0
    canonicalize(sdfg)
    assert count(sdfg, FillLibraryNode) == 0
    assert top_maps(sdfg) == 1
    a = np.random.default_rng(0).random(37)
    b = np.full(37, np.nan)
    sdfg(a=a, b=b, N=37)
    assert np.allclose(b, 2 * a)


def test_copy_expands_and_fuses_producer_and_consumer():
    sdfg = copy_between_maps()
    canonicalize(sdfg)
    assert count(sdfg, CopyLibraryNode) == 0
    assert top_maps(sdfg) == 1
    a = np.random.default_rng(1).random(37)
    b = np.full(37, np.nan)
    sdfg(a=a, b=b, N=37)
    assert np.allclose(b, 2 * a + 1)


def test_rank_changing_copy_stays_a_library_call():
    """A reshape walks one flattened index no neighbouring map shares: it stays a copy and stays correct."""
    sdfg = copy_between_maps(target_shape=(2, N // 2))
    assert ExpandFillCopy().apply_pass(sdfg, {}) is None
    assert count(sdfg, CopyLibraryNode) == 1
    canonicalize(sdfg)
    a = np.random.default_rng(2).random(36)
    b = np.full((2, 18), np.nan)
    sdfg(a=a, b=b, N=36)
    assert np.allclose(b, (2 * a + 1).reshape(2, 18))


def test_fill_inside_a_parallel_map_stays_a_library_call():
    """Inside a parallel map the fill's selection takes the single call, so the node stays."""
    sdfg = dace.SDFG("fill_in_map")
    sdfg.add_array("b", [N, 8], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:N"})
    fill = FillLibraryNode("fill_row", value=0)
    state.add_node(fill)
    state.add_nedge(entry, fill, dace.Memlet())
    state.add_memlet_path(fill, exit_node, state.add_write("b"), src_conn="_fill_out", memlet=dace.Memlet("b[i, 0:8]"))
    sdfg.validate()
    assert ExpandFillCopy().apply_pass(sdfg, {}) is None
    assert count(sdfg, FillLibraryNode) == 1


if __name__ == "__main__":
    test_fill_expands_and_fuses_with_its_reader()
    test_copy_expands_and_fuses_producer_and_consumer()
    test_rank_changing_copy_stays_a_library_call()
    test_fill_inside_a_parallel_map_stays_a_library_call()
