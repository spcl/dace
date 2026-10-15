# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.loop_to_state_scan.LoopToStateScan` and the finite-state analysis."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.state_scan import StateScan
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis.finite_state import finite_state
from dace.transformation.passes.canonicalize import pipeline
from dace.transformation.passes.loop_to_state_scan import LoopToStateScan

N, NS, NA, NI = (dace.symbol(name, dace.int64) for name in ("N", "NS", "NA", "NI"))


@dace.program
def dfa(trans: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    state = 0
    for i in range(N):
        state = trans[state, symbols[i]]
        counts[state] += 1


@dace.program
def huffman(child: dace.int64[NI, 2], bits: dace.int64[N], counts: dace.int64[NI + 1]):
    node = 0
    for i in range(N):
        nxt = child[node, bits[i]]
        if nxt >= NI:
            counts[nxt - NI] += 1
            node = 0
        else:
            node = nxt


@dace.program
def matches(delta: dace.int64[NS, 4], accept: dace.int64[NS], text: dace.int64[N], found: dace.int64[1]):
    state = 0
    total = 0
    for i in range(N):
        state = delta[state, text[i]]
        total = total + accept[state]
    found[0] = total


@dace.program
def shifted(trans: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    state = 1
    for i in range(N):
        state = trans[state - 1, symbols[i]]
        counts[state - 1] += 1


@dace.program
def strided_index(trans: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    state = 0
    for i in range(N):
        state = trans[2 * state, symbols[i]]
        counts[state] += 1


@dace.program
def rewritten_table(trans: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    state = 0
    for i in range(N):
        state = trans[state, symbols[i]]
        symbols[i] = state


@dace.program
def unbounded(values: dace.int64[N], out: dace.int64[1]):
    state = 0
    for i in range(N):
        state = state + values[i]
    out[0] = state


def reference_dfa(trans, symbols, counts):
    state = 0
    for symbol in symbols:
        state = trans[state, symbol]
        counts[state] += 1


def reference_huffman(child, bits, counts):
    inner = child.shape[0]
    node = 0
    for bit in bits:
        nxt = child[node, bit]
        if nxt >= inner:
            counts[nxt - inner] += 1
            node = 0
        else:
            node = nxt


def reference_matches(delta, accept, text, found):
    state = 0
    for symbol in text:
        state = delta[state, symbol]
        found[0] += accept[state]


def before_lifting(program) -> dace.SDFG:
    """``program`` canonicalized up to the stage that lifts loops to library nodes."""
    sdfg = program.to_sdfg(simplify=True)
    labels = pipeline.stage_labels()
    pipeline.canonicalize(sdfg, stages=labels[: labels.index(pipeline.LIFT_SEMANTICS)])
    return sdfg


def only_loop(sdfg: dace.SDFG) -> LoopRegion:
    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert len(loops) == 1
    return loops[0]


def state_scans(sdfg: dace.SDFG) -> list[StateScan]:
    return [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, StateScan)]


def automaton(rng: np.random.Generator, states: int, symbols: int, length: int):
    return rng.integers(0, states, (states, symbols)), rng.integers(0, symbols, length)


def test_a_table_driven_state_is_bounded_by_the_extent_it_indexes():
    sut = finite_state(only_loop(before_lifting(dfa)))

    assert sut is not None
    assert (sut.symbol, sut.base, sut.count) == ("state", 0, NS)
    assert sut.tables == {"trans", "symbols"}


def test_a_state_read_at_an_offset_starts_at_the_offset():
    sut = finite_state(only_loop(before_lifting(shifted)))

    assert sut is not None
    assert (sut.base, sut.count) == (1, NS)


@pytest.mark.parametrize("program", [strided_index, rewritten_table, unbounded])
def test_a_state_without_a_unit_stride_table_bound_or_with_a_written_table_is_not_finite(program):
    assert finite_state(only_loop(before_lifting(program))) is None


def test_the_huffman_step_keeps_the_branch_that_resets_the_node():
    sut = finite_state(only_loop(before_lifting(huffman)))

    assert sut is not None
    assert (sut.symbol, sut.count) == ("node", NI)
    assert {"nxt", "node"} <= sut.step_symbols


def test_a_dfa_scan_leaves_a_map_and_counts_the_states_the_loop_counted():
    rng = np.random.default_rng(7)
    trans, symbols = automaton(rng, 16, 4, 3000)
    expected = np.zeros(16, np.int64)
    reference_dfa(trans, symbols, expected)
    sdfg = dfa.to_sdfg(simplify=True)
    pipeline.canonicalize(sdfg)
    actual = np.zeros(16, np.int64)

    sdfg(trans=trans, symbols=symbols, counts=actual, N=3000, NS=16, NA=4)

    assert len(state_scans(sdfg)) == 1
    assert not any(isinstance(region, LoopRegion) for region in sdfg.all_control_flow_regions(recursive=True))
    np.testing.assert_array_equal(actual, expected)


def test_a_match_count_reduction_rides_along_the_lifted_state():
    rng = np.random.default_rng(11)
    delta, text = automaton(rng, 16, 4, 3000)
    accept = rng.integers(0, 3, 16)
    expected = np.zeros(1, np.int64)
    reference_matches(delta, accept, text, expected)
    sdfg = before_lifting(matches)

    lifted = LoopToStateScan().apply_pass(sdfg, {})
    actual = np.zeros(1, np.int64)
    sdfg(delta=delta, accept=accept, text=text, found=actual, N=3000, NS=16)

    assert lifted == 1
    np.testing.assert_array_equal(actual, expected)


def test_a_huffman_decode_through_the_scan_tallies_the_symbols_the_loop_tallied():
    rng = np.random.default_rng(3)
    child = np.array([[1, 2], [4, 5], [3, 6], [7, 8]], dtype=np.int64)
    bits = rng.integers(0, 2, 4000)
    expected = np.zeros(5, np.int64)
    reference_huffman(child, bits, expected)
    sdfg = before_lifting(huffman)

    lifted = LoopToStateScan().apply_pass(sdfg, {})
    actual = np.zeros(5, np.int64)
    sdfg(child=child, bits=bits, counts=actual, N=4000, NI=4)

    assert lifted == 1
    np.testing.assert_array_equal(actual, expected)


def test_a_loop_shorter_than_the_block_count_still_scans_exactly():
    rng = np.random.default_rng(5)
    trans, symbols = automaton(rng, 8, 3, 5)
    expected = np.zeros(8, np.int64)
    reference_dfa(trans, symbols, expected)
    sdfg = before_lifting(dfa)
    LoopToStateScan().apply_pass(sdfg, {})
    actual = np.zeros(8, np.int64)

    sdfg(trans=trans, symbols=symbols, counts=actual, N=5, NS=8, NA=3)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "implementation, schedule",
    [
        ("CPU", dace.ScheduleType.CPU_Multicore),
        ("CUDA", dace.ScheduleType.GPU_Device),
        ("pure", dace.ScheduleType.Sequential),
    ],
)
def test_each_expansion_runs_the_block_maps_on_its_device(implementation, schedule):
    sdfg = before_lifting(dfa)
    LoopToStateScan().apply_pass(sdfg, {})
    node = state_scans(sdfg)[0]
    node.implementation = implementation

    sdfg.expand_library_nodes(recursive=False)
    block_maps = [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.label.startswith("block_")
    ]

    assert len(block_maps) == 2
    assert {entry.map.schedule for entry in block_maps} == {schedule}
