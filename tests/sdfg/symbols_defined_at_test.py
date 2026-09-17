# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from unittest import mock

import numpy as np
import pytest

import dace
from dace.sdfg import nodes, propagation
from dace.sdfg.state import LoopRegion, SDFGState


def _make_sdfg(name: str, nested: bool = False) -> dace.SDFG:
    """An SDFG whose array shapes and map ranges are symbolic, with an optional nested SDFG."""
    sdfg = dace.SDFG(name)
    N = dace.symbol("N")
    for array in "ab":
        sdfg.add_array(array, shape=(N, 10), dtype=dace.float64, transient=False)
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "comp",
        map_ranges={
            "__i": "0:N",
            "__j": "0:10"
        },
        inputs={"__in": dace.Memlet("a[__i, __j]")},
        outputs={"__out": dace.Memlet("b[__i, __j]")},
        code="__out = __in + 1.0",
        external_edges=True,
    )
    if nested:
        inner = _make_sdfg(name + "_inner")
        inner_state = sdfg.add_state_after(state)
        nsdfg = inner_state.add_nested_sdfg(inner, {"a"}, {"b"}, symbol_mapping={"N": "N"})
        inner_state.add_edge(inner_state.add_access("a"), None, nsdfg, "a", dace.Memlet("a[0:N, 0:10]"))
        inner_state.add_edge(nsdfg, "b", inner_state.add_access("b"), None, dace.Memlet("b[0:N, 0:10]"))
    sdfg.validate()
    return sdfg


def test_state_symbols_give_the_same_result():
    sdfg = _make_sdfg("same_result")
    state = sdfg.states()[0]
    state_symbols = state.symbols_defined_at_state()

    for node in state.nodes():
        expected = state.symbols_defined_at(node)
        assert state.symbols_defined_at(node, state_symbols=state_symbols) == expected
        assert list(state.symbols_defined_at(node, state_symbols=state_symbols)) == list(expected)

    map_entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert "N" in state_symbols
    # The map parameters are what the node adds to the SDFG-wide symbols, inside the map only.
    assert set(state.symbols_defined_at(tasklet)) - set(state_symbols) == set(map_entry.map.params)
    assert set(state.symbols_defined_at(map_entry)) == set(state_symbols)


def test_propagation_resolves_the_state_symbols_once_per_state():
    """One resolution per state, however many Memlets that state holds."""
    sdfg = _make_sdfg("resolve_once", nested=True)
    states = [state for nested in sdfg.all_sdfgs_recursive() for state in nested.states()]
    assert sum(len(state.edges()) for state in states) > len(states)

    with mock.patch.object(SDFGState,
                           "symbols_defined_at_state",
                           autospec=True,
                           side_effect=SDFGState.symbols_defined_at_state) as spy:
        propagation.propagate_memlets_sdfg(sdfg)

    assert spy.call_count <= len(states)


def test_propagation_is_unchanged():
    sdfg = _make_sdfg("unchanged")
    state = sdfg.states()[0]
    map_entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    for edge in state.in_edges(map_entry):
        edge.data = dace.Memlet("a[0, 0]")

    propagation.propagate_memlets_sdfg(sdfg)

    assert str(state.in_edges(map_entry)[0].data.subset) == "0:N, 0:10"

    a = np.random.rand(16, 10)
    b = np.zeros_like(a)
    sdfg(a=a, b=b, N=16)
    assert np.allclose(b, a + 1.0)


def _make_sdfg_with_loop_region(name: str) -> tuple[dace.SDFG, SDFGState, SDFGState]:
    """A top-level state, then a `LoopRegion` whose state sees the loop iterator as well.

    Propagation visits the top-level state first, so anything it resolves per SDFG and reuses is
    missing the iterator by the time it reaches the loop body.
    """
    sdfg = dace.SDFG(name)
    N = dace.symbol("N")
    for array in "abc":
        sdfg.add_array(array, shape=(N, ), dtype=dace.float64, transient=False)
    top_level = sdfg.add_state("top_level", is_start_block=True)
    top_level.add_mapped_tasklet(
        "top_level_comp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("a[__i]")},
        outputs={"__out": dace.Memlet("c[__i]")},
        code="__out = __in + 1.0",
        external_edges=True,
    )
    loop = LoopRegion("loop", "it < N", "it", "it = 0", "it = it + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(top_level, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    body.add_mapped_tasklet(
        "body_comp",
        map_ranges={"__i": "0:it"},
        inputs={"__in": dace.Memlet("a[__i]")},
        outputs={"__out": dace.Memlet("b[__i]")},
        code="__out = __in + 1.0",
        external_edges=True,
    )
    sdfg.validate()
    return sdfg, top_level, body


def test_the_enclosing_regions_are_part_of_the_state_symbols():
    """A `LoopRegion` defines its iterator for its own states, not for every state of the SDFG."""
    sdfg, top_level, body = _make_sdfg_with_loop_region("regions")

    assert "it" in body.symbols_defined_at_state()
    assert "it" not in top_level.symbols_defined_at_state()


def test_propagation_keeps_the_loop_iterator():
    """The symbols of a `LoopRegion` belong to its states, not to every state of the SDFG."""
    sdfg, _, body = _make_sdfg_with_loop_region("regions_propagate")
    map_entry = next(n for n in body.nodes() if isinstance(n, nodes.MapEntry))

    propagation.propagate_memlets_sdfg(sdfg)

    outer_memlet = body.in_edges(map_entry)[0].data
    assert str(outer_memlet.subset) == "0:it"
    assert not outer_memlet.dynamic


def test_an_error_leaves_no_state_behind():
    sdfg = _make_sdfg("after_error")

    with mock.patch.object(SDFGState, "symbols_defined_at_state", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            propagation.propagate_memlets_sdfg(sdfg)

    # Nothing to clean up: the resolver is an ordinary object owned by the call.
    assert propagation.SymbolResolver()._per_state == {}


if __name__ == "__main__":
    test_state_symbols_give_the_same_result()
    test_propagation_resolves_the_state_symbols_once_per_state()
    test_propagation_is_unchanged()
    test_the_enclosing_regions_are_part_of_the_state_symbols()
    test_propagation_keeps_the_loop_iterator()
    test_an_error_leaves_no_state_behind()
