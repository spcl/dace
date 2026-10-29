# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pass that hoists the shared (Persistent, External, Global) transients of nested SDFGs."""

import copy
import json
import warnings

import numpy as np
import pytest

import dace
from dace import SDFG, InterstateEdge, Memlet
from dace.sdfg import nodes
from dace.transformation.passes import CanonicalizationPipeline, HoistSharedTransients
from dace.transformation.passes.hoist_shared_transients import has_nested_shared_transients


def _mode_sdfg(name: str, lifetime: dace.AllocationLifetime) -> SDFG:
    """
    A nested SDFG that stores its input ``a`` into the shared container ``p`` if ``mode == 0`` (and also returns
    ``a`` in ``b``), and returns the stored ``p`` in ``b`` otherwise.
    """
    sdfg = SDFG(name)
    sdfg.add_symbol("mode", dace.int32)
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_array("p", [1], dace.float64, transient=True, lifetime=lifetime)
    start = sdfg.add_state("start", is_start_block=True)
    store = sdfg.add_state("store")
    load = sdfg.add_state("load")
    end = sdfg.add_state("end")
    sdfg.add_edge(start, store, InterstateEdge("mode == 0"))
    sdfg.add_edge(start, load, InterstateEdge("mode != 0"))
    sdfg.add_edge(store, end, InterstateEdge())
    sdfg.add_edge(load, end, InterstateEdge())
    ra = store.add_read("a")
    store.add_nedge(ra, store.add_write("p"), Memlet("a[0]"))
    store.add_nedge(ra, store.add_write("b"), Memlet("a[0]"))
    load.add_nedge(load.add_read("p"), load.add_write("b"), Memlet("p[0]"))
    return sdfg


def _wrap(name: str, inner: SDFG) -> SDFG:
    """Wraps an SDFG with arguments ``a``/``b`` (one element each) and symbol ``mode`` in a nested SDFG node."""
    sdfg = SDFG(name)
    sdfg.add_symbol("mode", dace.int32)
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = state.add_nested_sdfg(inner, {"a"}, {"b"}, symbol_mapping={"mode": "mode"})
    state.add_edge(state.add_read("a"), None, node, "a", Memlet("a[0]"))
    state.add_edge(node, "b", state.add_write("b"), None, Memlet("b[0]"))
    return sdfg


def _nested_sdfgs(sdfg: SDFG) -> list[SDFG]:
    return [s for s in sdfg.all_sdfgs_recursive() if s is not sdfg]


def _run_twice(sdfg: SDFG) -> tuple[float, float]:
    """Stores 3.0 in one call and loads it in the next call of the same compiled program."""
    csdfg = sdfg.compile()
    b = np.zeros(1)
    csdfg(a=np.array([3.0]), b=b, mode=0)
    first = b[0]
    b = np.zeros(1)
    csdfg(a=np.array([5.0]), b=b, mode=1)
    return first, b[0]


def test_persistent_across_calls():
    sdfg = _wrap("hoist_persistent", _mode_sdfg("inner", dace.AllocationLifetime.Persistent))
    original = copy.deepcopy(sdfg)

    result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is not None and len(result) == 1
    sdfg.validate()
    assert not has_nested_shared_transients(sdfg)

    hoisted = next(iter(result.values()))
    assert sdfg.arrays[hoisted].transient
    assert sdfg.arrays[hoisted].lifetime == dace.AllocationLifetime.Persistent
    inner = _nested_sdfgs(sdfg)[0]
    assert not inner.arrays["p"].transient
    assert inner.arrays["p"].lifetime == dace.AllocationLifetime.Scope
    node = inner.parent_nsdfg_node
    assert "p" in node.in_connectors and "p" in node.out_connectors

    assert _run_twice(sdfg) == (3.0, 3.0)
    assert _run_twice(original) == (3.0, 3.0)


def test_external():
    sdfg = _wrap("hoist_external", _mode_sdfg("inner", dace.AllocationLifetime.External))

    result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is not None
    sdfg.validate()
    assert not has_nested_shared_transients(sdfg)
    hoisted = next(iter(result.values()))
    assert sdfg.arrays[hoisted].transient
    assert sdfg.arrays[hoisted].lifetime == dace.AllocationLifetime.External

    csdfg = sdfg.compile()
    a = np.array([3.0])
    b = np.zeros(1)
    csdfg.initialize(a=a, b=b, mode=0)
    assert csdfg.get_workspace_sizes() == {dace.StorageType.CPU_Heap: 8}
    workspace = np.zeros(1)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, workspace)
    csdfg(a=a, b=b, mode=0)
    # The caller-managed storage holds the stored value.
    assert workspace[0] == 3.0
    b = np.zeros(1)
    csdfg(a=np.array([5.0]), b=b, mode=1)
    assert b[0] == 3.0


def test_two_levels():
    middle = _wrap("middle", _mode_sdfg("inner", dace.AllocationLifetime.Persistent))
    sdfg = _wrap("hoist_two_levels", middle)
    original = copy.deepcopy(sdfg)

    result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is not None and len(result) == 1
    sdfg.validate()
    assert not has_nested_shared_transients(sdfg)

    hoisted = next(iter(result.values()))
    assert sdfg.arrays[hoisted].transient
    assert sdfg.arrays[hoisted].lifetime == dace.AllocationLifetime.Persistent
    middle = next(s for s in _nested_sdfgs(sdfg) if s.name == "middle")
    threaded = [name for name, desc in middle.arrays.items() if name not in ("a", "b")]
    assert len(threaded) == 1
    assert not middle.arrays[threaded[0]].transient
    assert threaded[0] in middle.parent_nsdfg_node.in_connectors
    assert threaded[0] in middle.parent_nsdfg_node.out_connectors

    assert _run_twice(sdfg) == (3.0, 3.0)
    assert _run_twice(original) == (3.0, 3.0)


def _map_sdfg(lifetime: dace.AllocationLifetime) -> SDFG:
    """
    A sequential map over ``i`` whose nested SDFG returns the value stored by the previous iteration (or, for a
    Persistent ``p``, the last iteration of the previous call) in ``p`` and then stores ``A[i]`` in ``p``.
    """
    inner = SDFG("inner")
    inner.add_symbol("i", dace.int64)
    inner.add_array("a", [8], dace.float64)
    inner.add_array("b", [8], dace.float64)
    inner.add_array("p", [1], dace.float64, transient=True, lifetime=lifetime)
    load = inner.add_state("load", is_start_block=True)
    store = inner.add_state_after(load, "store")
    load.add_nedge(load.add_read("p"), load.add_write("b"), Memlet("p[0] -> [i]"))
    store.add_nedge(store.add_read("a"), store.add_write("p"), Memlet("a[i] -> [0]"))

    sdfg = SDFG("hoist_in_map")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", {"i": "0:8"}, schedule=dace.ScheduleType.Sequential)
    node = state.add_nested_sdfg(inner, {"a"}, {"b"}, symbol_mapping={"i": "i"})
    state.add_memlet_path(state.add_read("A"), me, node, dst_conn="a", memlet=Memlet("A[0:8]"))
    state.add_memlet_path(node, mx, state.add_write("B"), src_conn="b", memlet=Memlet("B[0:8]"))
    return sdfg


@pytest.mark.parametrize("lifetime", [dace.AllocationLifetime.Persistent, dace.AllocationLifetime.Global])
def test_in_map(lifetime):
    sdfg = _map_sdfg(lifetime)
    sdfg.name = f"hoist_in_map_{lifetime.name.lower()}"
    original = copy.deepcopy(sdfg)

    result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is not None
    sdfg.validate()
    assert not has_nested_shared_transients(sdfg)
    hoisted = next(iter(result.values()))
    assert sdfg.arrays[hoisted].transient and sdfg.arrays[hoisted].lifetime == lifetime

    state = sdfg.start_state
    node = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    me = state.entry_node(node)
    mx = state.exit_node(me)
    # Both directions go through the map, with the whole container on every edge of the path.
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn == "p")
    out_edge = next(e for e in state.out_edges(node) if e.src_conn == "p")
    assert in_edge.src is me and out_edge.dst is mx
    for edge in state.memlet_path(in_edge) + state.memlet_path(out_edge):
        assert edge.data.data == hoisted
        assert edge.data.subset == dace.subsets.Range.from_array(sdfg.arrays[hoisted])

    A1 = np.arange(8, dtype=np.float64)
    A2 = np.arange(8, dtype=np.float64) + 100
    for program in (sdfg, original):
        csdfg = program.compile()
        B = np.zeros(8)
        csdfg(A=A1.copy(), B=B)
        assert np.array_equal(B[1:], A1[:-1])
        B = np.zeros(8)
        csdfg(A=A2.copy(), B=B)
        if lifetime == dace.AllocationLifetime.Persistent:
            assert B[0] == A1[-1]
        assert np.array_equal(B[1:], A2[:-1])


def test_symbolic_shape():
    inner = SDFG("inner")
    inner.add_symbol("M", dace.int64)
    inner.add_array("a", ["M"], dace.float64)
    inner.add_array("p", ["M"], dace.float64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    state = inner.add_state(is_start_block=True)
    state.add_nedge(state.add_read("a"), state.add_write("p"), Memlet("a[0:M]"))

    sdfg = SDFG("hoist_symbolic")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", ["2*N"], dace.float64)
    outer = sdfg.add_state(is_start_block=True)
    node = outer.add_nested_sdfg(inner, {"a"}, set(), symbol_mapping={"M": "2*N"})
    outer.add_edge(outer.add_read("A"), None, node, "a", Memlet("A[0:2*N]"))

    result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is not None
    sdfg.validate()
    hoisted = next(iter(result.values()))
    assert sdfg.arrays[hoisted].shape == (2 * dace.symbol("N"),)
    # Write-only: an output connector only.
    assert "p" in node.out_connectors and "p" not in node.in_connectors


def test_constant_shape():
    inner = SDFG("inner")
    inner.add_symbol("K", dace.int64)
    inner.add_constant("K", 4)
    inner.add_array("a", [1], dace.float64)
    inner.add_array("p", ["K"], dace.float64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    state = inner.add_state(is_start_block=True)
    state.add_nedge(state.add_read("a"), state.add_write("p"), Memlet("p[0]"))

    sdfg = SDFG("hoist_constant")
    sdfg.add_array("A", [1], dace.float64)
    outer = sdfg.add_state(is_start_block=True)
    node = outer.add_nested_sdfg(inner, {"a"}, set())
    outer.add_edge(outer.add_read("A"), None, node, "a", Memlet("A[0]"))

    result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is not None
    sdfg.validate()
    assert sdfg.arrays[next(iter(result.values()))].shape == (4,)
    assert inner.arrays["p"].shape == (4,)


def test_symbol_not_available():
    inner = SDFG("inner")
    inner.add_symbol("M", dace.int64)
    inner.add_array("a", [1], dace.float64)
    inner.add_array("p", ["M"], dace.float64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    state = inner.add_state(is_start_block=True)
    state.add_nedge(state.add_read("a"), state.add_write("p"), Memlet("p[0]"))

    sdfg = SDFG("hoist_map_symbol")
    sdfg.add_array("A", [8], dace.float64)
    outer = sdfg.add_state(is_start_block=True)
    me, mx = outer.add_map("m", {"i": "1:8"})
    node = outer.add_nested_sdfg(inner, {"a"}, set(), symbol_mapping={"M": "i"})
    outer.add_memlet_path(outer.add_read("A"), me, node, dst_conn="a", memlet=Memlet("A[i]"))
    outer.add_nedge(node, mx, Memlet())
    before = json.dumps(sdfg.to_json())

    with pytest.warns(UserWarning, match="not defined for the whole program"):
        result = HoistSharedTransients().apply_pass(sdfg, {})
    assert result is None
    assert json.dumps(sdfg.to_json()) == before
    assert inner.arrays["p"].transient


def test_nothing_to_hoist():
    inner = _mode_sdfg("inner", dace.AllocationLifetime.Scope)
    sdfg = _wrap("hoist_noop", inner)
    # A Persistent transient of the top-level SDFG itself stays where it is.
    sdfg.add_array("q", [1], dace.float64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    before = json.dumps(sdfg.to_json())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert HoistSharedTransients().apply_pass(sdfg, {}) is None
        assert CanonicalizationPipeline().apply_pass(sdfg, {}) is None
    assert json.dumps(sdfg.to_json()) == before
    sdfg.validate()


def test_idempotent():
    sdfg = _wrap("hoist_twice", _wrap("middle", _mode_sdfg("inner", dace.AllocationLifetime.Persistent)))
    assert CanonicalizationPipeline().apply_pass(sdfg, {}) is not None
    after_first = json.dumps(sdfg.to_json())
    assert CanonicalizationPipeline().apply_pass(sdfg, {}) is None
    assert json.dumps(sdfg.to_json()) == after_first
    sdfg.validate()


def test_frontend():
    N = dace.symbol("N")

    @dace.program
    def swap_with_last(x: dace.float64[N], y: dace.float64[N]):
        p = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.Persistent)
        y[:] = p
        p[:] = x

    @dace.program
    def caller(A: dace.float64[N], B: dace.float64[N]):
        for _ in range(2):
            swap_with_last(A, B)

    sdfg = caller.to_sdfg(simplify=False)
    sdfg.validate()
    assert not has_nested_shared_transients(sdfg)
    assert any(desc.transient and desc.lifetime == dace.AllocationLifetime.Persistent for desc in sdfg.arrays.values())

    A = np.random.rand(5)
    B = np.zeros(5)
    sdfg(A=A, B=B, N=5)
    # The second call of the nested program returns what the first one stored.
    assert np.array_equal(A, B)


if __name__ == "__main__":
    test_persistent_across_calls()
    test_external()
    test_two_levels()
    test_in_map(dace.AllocationLifetime.Persistent)
    test_in_map(dace.AllocationLifetime.Global)
    test_symbolic_shape()
    test_constant_shape()
    test_symbol_not_available()
    test_nothing_to_hoist()
    test_idempotent()
    test_frontend()
