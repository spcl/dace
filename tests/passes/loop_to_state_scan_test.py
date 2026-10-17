# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.loop_to_state_scan.LoopToStateScan` and the finite-state analysis."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.state_scan import StateScan, StateScanKind
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis.carried_state import gf2_state
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
def kmp(text: dace.int64[N], pattern: dace.int64[NA], fail: dace.int64[NA], found: dace.int64[1]):
    count = 0
    q = 0
    for i in range(N):
        while q > 0 and pattern[q] != text[i]:
            q = fail[q - 1]
        if pattern[q] == text[i]:
            q = q + 1
        if q == NA:
            count = count + 1
            q = fail[q - 1]
    found[0] = count


@dace.program
def crc16(data: dace.uint8[N], crc: dace.int64[1], polynomial: dace.int64):
    c = 0xFFFF
    for b in range(N):
        cur_byte = 0xFF & data[b]
        for bit in range(0, 8):
            if (c & 0x0001) ^ (cur_byte & 0x0001):
                c = (c >> 1) ^ polynomial
            else:
                c = c >> 1
            cur_byte = cur_byte >> 1
    crc[0] = (c ^ 0xFFFF) & 0xFFFF


@dace.program
def squared(values: dace.int64[N], out: dace.int64[1]):
    state = 1
    for i in range(N):
        state = (state * state) ^ values[i]
    out[0] = state


@dace.program
def classified(classes: dace.int64[256], trans: dace.int64[NS, NA], text: dace.int64[N], counts: dace.int64[2]):
    state = 1
    for i in range(N):
        state = trans[state, classes[text[i]]]
        if state <= 1:
            counts[state] += 1


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


def reference_kmp(text, pattern, fail, found):
    q = 0
    for symbol in text:
        while q > 0 and pattern[q] != symbol:
            q = fail[q - 1]
        if pattern[q] == symbol:
            q += 1
        if q == len(pattern):
            found[0] += 1
            q = fail[q - 1]


def failure_table(pattern):
    fail = np.zeros(len(pattern), np.int64)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[k] != pattern[i]:
            k = fail[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        fail[i] = k
    return fail


def reference_crc16(data, polynomial):
    c = 0xFFFF
    for byte in data:
        cur = int(byte)
        for bit in range(8):
            c = (c >> 1) ^ polynomial if (c ^ cur) & 1 else c >> 1
            cur >>= 1
    return (c ^ 0xFFFF) & 0xFFFF


def reference_classified(classes, trans, text, counts):
    state = 1
    for symbol in text:
        state = trans[state, classes[symbol]]
        if state <= 1:
            counts[state] += 1


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


def top_loop(sdfg: dace.SDFG) -> LoopRegion:
    """The one loop at the top of ``sdfg``; inner loops (a CRC's bit loop) are its body."""
    loops = [region for region in sdfg.nodes() if isinstance(region, LoopRegion)]
    assert len(loops) == 1
    return loops[0]


def state_scans(sdfg: dace.SDFG) -> list[StateScan]:
    return [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, StateScan)]


def automaton(rng: np.random.Generator, states: int, symbols: int, length: int):
    return rng.integers(0, states, (states, symbols)), rng.integers(0, symbols, length)


def test_a_table_driven_state_is_bounded_by_the_extent_it_indexes():
    sut = finite_state(only_loop(before_lifting(dfa)))

    assert sut is not None
    assert (sut.step.symbol, sut.base, sut.count) == ("state", 0, NS)
    assert sut.step.tables == {"trans", "symbols"}


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
    assert (sut.step.symbol, sut.count) == ("node", NI)
    assert {"nxt", "node"} <= sut.step.values


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


def test_a_kmp_search_whose_step_holds_a_while_loop_becomes_a_map_around_it():
    rng = np.random.default_rng(13)
    text = rng.integers(0, 2, 4000)
    pattern = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    fail = failure_table(pattern)
    expected = np.zeros(1, np.int64)
    reference_kmp(text, pattern, fail, expected)
    sdfg = kmp.to_sdfg(simplify=True)
    pipeline.canonicalize(sdfg)
    actual = np.zeros(1, np.int64)

    sdfg(text=text, pattern=pattern, fail=fail, found=actual, N=4000, NA=5)

    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert len(state_scans(sdfg)) == 1
    # The step's search loop, counted by WhileToFor, stays one sequential loop inside the scan's step
    assert len(loops) == 1
    assert all(loop.sdfg is not sdfg for loop in loops)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
def test_a_dfa_scan_offloaded_to_the_gpu_counts_the_states_the_loop_counted():
    import cupy  # Only present on GPU runners.

    from dace.transformation.passes.canonicalize import finalize

    rng = np.random.default_rng(17)
    trans, symbols = automaton(rng, 64, 4, 200000)
    expected = np.zeros(64, np.int64)
    reference_dfa(trans, symbols, expected)
    sdfg = dfa.to_sdfg(simplify=True)
    pipeline.canonicalize(sdfg, target="gpu")
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, "gpu")
    actual = cupy.zeros(64, cupy.int64)

    sdfg(trans=cupy.asarray(trans), symbols=cupy.asarray(symbols), counts=actual, N=200000, NS=64, NA=4)

    assert [node.schedule for node in state_scans(sdfg)] == [dace.ScheduleType.GPU_Device]
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)


def test_a_crc_register_is_proven_gf2_affine_with_the_polynomial_as_a_parameter():
    sut = gf2_state(top_loop(before_lifting(crc16)))

    assert sut is not None
    assert (sut.step.symbol, sut.bits) == ("c", 64)
    assert "polynomial" in sut.step.tables


@pytest.mark.parametrize("program", [squared, unbounded, dfa])
def test_a_step_squaring_adding_or_looking_up_its_state_is_not_gf2_affine(program):
    assert gf2_state(top_loop(before_lifting(program))) is None


@pytest.mark.parametrize("length", [0, 1, 7, 1600, 5000])
def test_a_crc_through_the_gf2_scan_equals_the_bytewise_crc(length):
    data = np.random.default_rng(length).integers(0, 256, length).astype(np.uint8)
    sdfg = crc16.to_sdfg(simplify=True)
    pipeline.canonicalize(sdfg)
    actual = np.zeros(1, np.int64)

    sdfg(data=data, crc=actual, polynomial=np.int64(0x8408), N=length)

    assert [node.kind for node in state_scans(sdfg)] == [StateScanKind.GF2_AFFINE]
    assert actual[0] == reference_crc16(data, 0x8408)


@pytest.mark.gpu
def test_a_crc_through_the_gf2_scan_on_the_gpu_equals_the_bytewise_crc():
    import cupy  # Only present on GPU runners.

    from dace.transformation.passes.canonicalize import finalize

    data = np.random.default_rng(19).integers(0, 256, 100000).astype(np.uint8)
    sdfg = crc16.to_sdfg(simplify=True)
    pipeline.canonicalize(sdfg, target="gpu")
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, "gpu")
    actual = cupy.zeros(1, cupy.int64)

    sdfg(data=cupy.asarray(data), crc=actual, polynomial=np.int64(0x8408), N=100000)

    assert [node.schedule for node in state_scans(sdfg)] == [dace.ScheduleType.GPU_Device]
    assert int(actual[0]) == reference_crc16(data, 0x8408)


@pytest.mark.parametrize("states, classes", [(9, 12), (5, 4)])
def test_a_byte_classifying_automaton_counts_its_accept_and_reject_states_as_the_loop_did(states, classes):
    """The shape of UTF-8 validation (9 states, 12 byte classes) and CSV tokenizing (5 states, 4 classes): a
    class table between the input and the transition table, a start state of 1, and a count only in states 0
    and 1."""
    rng = np.random.default_rng(states)
    byte_class = rng.integers(0, classes, 256)
    trans = rng.integers(0, states, (states, classes))
    text = rng.integers(0, 256, 6000)
    expected = np.zeros(2, np.int64)
    reference_classified(byte_class, trans, text, expected)
    sdfg = classified.to_sdfg(simplify=True)
    pipeline.canonicalize(sdfg)
    actual = np.zeros(2, np.int64)

    sdfg(classes=byte_class, trans=trans, text=text, counts=actual, N=6000, NS=states, NA=classes)

    assert [node.kind for node in state_scans(sdfg)] == [StateScanKind.FINITE]
    assert not any(isinstance(region, LoopRegion) for region in sdfg.all_control_flow_regions(recursive=True))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("chunks", [1, 3, 64, 5000])
def test_the_block_count_does_not_change_the_states_a_scan_records(chunks):
    rng = np.random.default_rng(chunks)
    trans, symbols = automaton(rng, 16, 4, 3000)
    expected = np.zeros(16, np.int64)
    reference_dfa(trans, symbols, expected)
    sdfg = before_lifting(dfa)
    LoopToStateScan().apply_pass(sdfg, {})
    state_scans(sdfg)[0].chunks = chunks
    actual = np.zeros(16, np.int64)

    sdfg(trans=trans, symbols=symbols, counts=actual, N=3000, NS=16, NA=4)

    np.testing.assert_array_equal(actual, expected)


def test_a_state_scan_survives_serialization_with_its_step():
    sdfg = before_lifting(crc16)
    LoopToStateScan().apply_pass(sdfg, {})

    sut = dace.SDFG.from_json(sdfg.to_json())

    node = state_scans(sut)[0]
    assert (node.kind, node.bits, node.state) == (StateScanKind.GF2_AFFINE, 64, "c")
    assert node.step is not None and "polynomial" in node.step.arrays
    sut.validate()


@pytest.mark.parametrize("chunks, length", [(1, 1000), (3, 1000), (256, 1600), (7, 50), (64, 6)])
def test_a_gf2_scan_whose_last_block_is_short_still_equals_the_bytewise_crc(chunks, length):
    """Lengths that do not divide into equal blocks leave one short block; its power of ``A`` is its own."""
    data = np.random.default_rng(chunks).integers(0, 256, length).astype(np.uint8)
    sdfg = before_lifting(crc16)
    LoopToStateScan().apply_pass(sdfg, {})
    state_scans(sdfg)[0].chunks = chunks
    actual = np.zeros(1, np.int64)

    sdfg(data=data, crc=actual, polynomial=np.int64(0x8408), N=length)

    assert actual[0] == reference_crc16(data, 0x8408)
