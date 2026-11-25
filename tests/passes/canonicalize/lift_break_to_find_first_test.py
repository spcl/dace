# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LiftBreakToFindFirst: an early-exit loop becomes FindFirst, a loop below the hit and the effect once."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.find_first import FindFirst
from dace.sdfg.state import BreakBlock, LoopRegion
from dace.transformation.passes.canonicalize.lift_break_to_find_first import LiftBreakToFindFirst

N = dace.symbol("N", dace.int64)


@dace.program
def capture(a: dace.float64[N], out_index: dace.int64[1], out_value: dace.float64[1]):
    out_index[0] = -1
    out_value[0] = -1.0
    for i in range(N):
        if a[i] > 0.9:
            out_index[0] = i
            out_value[0] = a[i]
            break


M = dace.symbol("M", dace.int64)


@dace.program
def capture_column(a: dace.float64[N, M], out_index: dace.int64[1]):
    out_index[0] = -1
    for i in range(N):
        if a[i, 2] > 0.9:
            out_index[0] = i
            break


@dace.program
def capture_into_symbols(a: dace.float64[N], result: dace.float64[1]):
    index = -2
    value = -1.0
    for i in range(N):
        if a[i] > 0.9:
            index = i
            value = a[i]
            break
    result[0] = value + index


@dace.program
def break_then_body(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(N):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]


@dace.program
def body_then_break(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break


@dace.program
def test_reads_what_the_body_writes(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i + 1] = a[i] + b[i]
        if a[i] > 2.0:
            break


@dace.program
def divides_by_data(a: dace.int64[N], out_index: dace.int64[1]):
    out_index[0] = -1
    for i in range(N):
        if 12 // a[i] > 3:
            out_index[0] = i
            break


def lifted(program, name: str, chunk: int = 64) -> tuple[dace.SDFG, int | None]:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    result = LiftBreakToFindFirst(chunk=chunk).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg, result


def has_break(sdfg: dace.SDFG) -> bool:
    return any(isinstance(b, BreakBlock) for b in sdfg.all_control_flow_blocks())


def find_firsts(sdfg: dace.SDFG) -> list[FindFirst]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, FindFirst)]


@pytest.mark.parametrize("hit", [17, 0, None])
def test_capture_finds_the_first_hit_and_runs_the_effect_once(hit):
    sdfg, result = lifted(capture, f"lift_ff_capture_{hit}")
    assert result == 1 and not has_break(sdfg) and len(find_firsts(sdfg)) == 1
    a = np.random.default_rng(0).uniform(0.0, 0.5, 1000)
    if hit is not None:
        a[hit] = 0.95
        a[hit + 300] = 0.99
    out_index, out_value = np.zeros(1, np.int64), np.zeros(1)
    sdfg(a=a, out_index=out_index, out_value=out_value, N=a.size)
    assert out_index[0] == (-1 if hit is None else hit)
    assert out_value[0] == (-1.0 if hit is None else 0.95)


@pytest.mark.parametrize("chunk", [64, 0])
@pytest.mark.parametrize("program, hit", [(break_then_body, 211), (break_then_body, None), (body_then_break, 211)])
def test_the_body_runs_exactly_below_the_hit(program, hit, chunk):
    """Chunked, the break survives on the chunk loop only; unchunked, no break is left."""
    sdfg, result = lifted(program, f"lift_ff_{program.name}_{hit}_{chunk}", chunk)
    assert result == 1 and has_break(sdfg) == (chunk > 0) and len(find_firsts(sdfg)) == 1
    rng = np.random.default_rng(1)
    a, b, c = rng.random(500), rng.random(500) + 2.0, rng.random(500)
    d = rng.random(500)
    if hit is not None:
        d[hit] = -1.0
        c[hit] = 5.0
    expected = a.copy()
    args = {"a": a, "b": b, "c": c, "N": a.size}
    if program is break_then_body:
        args["d"] = d
        stop = a.size if hit is None else hit
        expected[:stop] += b[:stop] * c[:stop]
    else:
        stop = a.size if hit is None else hit + 1
        expected[:stop] += b[:stop] * c[:stop]
    sdfg(**args)
    np.testing.assert_allclose(a, expected, rtol=1e-14, atol=0.0)


def test_a_test_that_reads_what_the_body_writes_is_refused():
    sdfg, result = lifted(test_reads_what_the_body_writes, "lift_ff_refused")
    assert result is None and has_break(sdfg) and not find_firsts(sdfg)


def test_a_test_dividing_by_data_is_refused():
    """``12 // a[i]`` past the hit may divide by zero where the sequential loop never tested."""
    sdfg, result = lifted(divides_by_data, "lift_ff_divides")
    assert result is None and has_break(sdfg) and not find_firsts(sdfg)


@pytest.mark.parametrize("program", [body_then_break, break_then_body])
def test_the_chunked_search_stays_parallel_inside_its_chunk_loop(program):
    """Canonicalize lifts the break from the unsimplified frontend graph, whose guard reads its condition through
    a preparation region (TSVC ``s481``), and the chunk loop hands the search a chunk of work per entry: the CPU
    tail must not serialize it."""
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f"lift_ff_parallel_chunk_{program.name}"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    (node,) = find_firsts(sdfg)
    assert node.schedule != dace.ScheduleType.Sequential


@pytest.mark.parametrize("hit", [33, None])
def test_symbols_only_the_effect_assigns_may_be_read_after_the_loop(hit):
    """TSVC ``s332``: the effect binds ``index`` / ``value``; with no hit they keep their pre-loop values."""
    sdfg, result = lifted(capture_into_symbols, f"lift_ff_symbols_{hit}")
    assert result == 1 and not has_break(sdfg)
    a = np.random.default_rng(2).uniform(0.0, 0.5, 300)
    if hit is not None:
        a[hit] = 0.95
    out = np.zeros(1)
    sdfg(a=a, result=out, N=a.size)
    assert out[0] == (-3.0 if hit is None else 0.95 + hit)


@pytest.mark.parametrize("implementation", ["CPU", "pure"])
@pytest.mark.parametrize("hit", [5, 0, None])
def test_a_two_dimensional_test_reads_the_element_it_names(implementation, hit):
    """The predicate keeps ``a[i, 2]`` per dimension; each lowering reads that element, never a flat neighbor."""
    sdfg, result = lifted(capture_column, f"lift_ff_column_{implementation}_{hit}")
    (node,) = find_firsts(sdfg)
    node.implementation = implementation
    a = np.random.default_rng(3).uniform(0.0, 0.5, (40, 7))
    a[:, 3] = 0.95  # the flat neighbor of column 2 fires everywhere
    if hit is not None:
        a[hit, 2] = 0.95
    out_index = np.zeros(1, np.int64)

    sdfg(a=a, out_index=out_index, N=40, M=7)

    assert result == 1
    assert "[__i, 2]" in node.predicate, node.predicate
    assert out_index[0] == (-1 if hit is None else hit)


@pytest.mark.parametrize("hit", [17, 0, None])
def test_the_pure_search_is_python_sdfg_components_and_finds_the_first_hit(hit):
    sdfg, result = lifted(capture, f"lift_ff_pure_{hit}")
    (node,) = find_firsts(sdfg)
    node.implementation = "pure"
    sdfg.expand_library_nodes()
    a = np.random.default_rng(0).uniform(0.0, 0.5, 1000)
    if hit is not None:
        a[hit] = 0.95
        a[hit + 300] = 0.99
    out_index, out_value = np.zeros(1, np.int64), np.zeros(1)

    sdfg(a=a, out_index=out_index, out_value=out_value, N=a.size)

    assert result == 1 and not find_firsts(sdfg)
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert all(t.language == dace.Language.Python for t in tasklets), [t.label for t in tasklets]
    assert out_index[0] == (-1 if hit is None else hit)
    assert out_value[0] == (-1.0 if hit is None else 0.95)


@dace.program
def capture_behind_a_sentinel(a: dace.float64[M], idx: dace.int64[N], out_index: dace.int64[1]):
    out_index[0] = -1
    for i in range(N):
        if a[idx[i]] > 0.5:
            out_index[0] = i
            break


def test_a_test_that_gathers_through_the_counter_keeps_the_loop():
    """``a[idx[i]]``: past the hit ``idx`` holds no valid index, and the search would read there."""
    sdfg, result = lifted(capture_behind_a_sentinel, "lift_ff_sentinel")
    a = np.zeros(16)
    a[7] = 1.0
    idx = np.full(4096, 2**40, np.int64)
    idx[:10], idx[10] = 3, 7
    out_index = np.zeros(1, np.int64)

    sdfg(a=a, idx=idx, out_index=out_index, N=idx.size, M=a.size)

    assert result is None and not find_firsts(sdfg)
    assert out_index[0] == 10


@dace.program
def last_match(a: dace.float64[N], out_index: dace.int64[1], out_value: dace.float64[1]):
    index = -1
    value = 0.0
    for i in range(N):
        if a[i] > 0.9:
            index = i
            value = a[i]
    out_index[0] = index
    out_value[0] = value


@dace.program
def last_match_into_memory(a: dace.float64[N], out_index: dace.int64[1]):
    out_index[0] = -1
    for i in range(N):
        if a[i] > 0.9:
            out_index[0] = i


@dace.program
def first_match_without_break(a: dace.float64[N], out_index: dace.int64[1]):
    index = -1
    for i in range(N):
        if a[i] > 0.9 and index < 0:
            index = i
    out_index[0] = index


@dace.program
def first_match_by_flag(a: dace.float64[N], out_index: dace.int64[1]):
    found = 0
    index = -1
    for i in range(N):
        if found == 0 and a[i] > 0.9:
            found = 1
            index = i
    out_index[0] = index


@dace.program
def counts_matches(a: dace.float64[N], out: dace.int64[1]):
    count = 0
    for i in range(N):
        if a[i] > 0.9:
            count = count + 1
    out[0] = count


@dace.program
def marks_every_match(a: dace.float64[N], marks: dace.int64[N]):
    for i in range(N):
        if a[i] > 0.9:
            marks[i] = 1


@dace.program
def sentinel_reopens(a: dace.float64[N], out_index: dace.int64[1]):
    index = -1
    for i in range(N):
        if a[i] > 0.9 and index < 0:
            index = i - 5
    out_index[0] = index


@dace.program
def record_feeds_the_test(a: dace.float64[N], out: dace.float64[1]):
    best = 0.5
    for i in range(N):
        if a[i] > best:
            best = a[i]
    out[0] = best


def hits_at(positions: list[int], size: int = 1000) -> np.ndarray:
    a = np.random.default_rng(2).uniform(0.0, 0.5, size)
    for position in positions:
        a[position] = 0.95 + position / (10 * size)
    return a


@pytest.mark.parametrize("positions", [[17, 400, 999], [0], []])
def test_a_last_match_is_a_search_from_the_end_and_records_once(positions):
    sdfg, result = lifted(last_match, f"lift_ff_last_{len(positions)}")
    assert result == 1 and len(find_firsts(sdfg)) == 1
    assert not any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions())
    a = hits_at(positions)
    out_index, out_value = np.zeros(1, np.int64), np.zeros(1)
    sdfg(a=a, out_index=out_index, out_value=out_value, N=a.size)
    assert out_index[0] == (positions[-1] if positions else -1)
    assert out_value[0] == (a[positions[-1]] if positions else 0.0)


@pytest.mark.parametrize("positions", [[17, 400], []])
def test_a_last_match_into_memory_keeps_the_value_from_before_the_loop_on_no_hit(positions):
    sdfg, result = lifted(last_match_into_memory, f"lift_ff_last_memory_{len(positions)}")
    assert result == 1
    a = hits_at(positions)
    out_index = np.zeros(1, np.int64)
    sdfg(a=a, out_index=out_index, N=a.size)
    assert out_index[0] == (positions[-1] if positions else -1)


@pytest.mark.parametrize("program", [first_match_without_break, first_match_by_flag])
@pytest.mark.parametrize("positions", [[17, 400], [999], []])
def test_a_first_match_without_break_is_a_find_first_guarded_by_its_sentinel(program, positions):
    sdfg, result = lifted(program, f"lift_ff_{program.name}_{len(positions)}")
    assert result == 1 and len(find_firsts(sdfg)) == 1
    a = hits_at(positions)
    out_index = np.zeros(1, np.int64)
    sdfg(a=a, out_index=out_index, N=a.size)
    assert out_index[0] == (positions[0] if positions else -1)


@pytest.mark.parametrize(
    "program", [counts_matches, marks_every_match, sentinel_reopens, record_feeds_the_test], ids=lambda p: p.name
)
def test_a_record_that_is_not_last_wins_is_refused(program):
    """Each record builds on the last, writes where the index says, reopens its sentinel, or feeds the test."""
    sdfg, result = lifted(program, f"lift_ff_refused_{program.name}")
    assert result is None and not find_firsts(sdfg)


if __name__ == "__main__":
    test_capture_finds_the_first_hit_and_runs_the_effect_once(17)
    test_capture_finds_the_first_hit_and_runs_the_effect_once(0)
    test_capture_finds_the_first_hit_and_runs_the_effect_once(None)
    for size in (64, 0):
        test_the_body_runs_exactly_below_the_hit(break_then_body, 211, size)
        test_the_body_runs_exactly_below_the_hit(break_then_body, None, size)
        test_the_body_runs_exactly_below_the_hit(body_then_break, 211, size)
    test_a_test_that_reads_what_the_body_writes_is_refused()
    test_the_chunked_search_stays_parallel_inside_its_chunk_loop(body_then_break)
    test_the_chunked_search_stays_parallel_inside_its_chunk_loop(break_then_body)
    test_symbols_only_the_effect_assigns_may_be_read_after_the_loop(33)
    test_symbols_only_the_effect_assigns_may_be_read_after_the_loop(None)
    for hit in (5, 0, None):
        test_a_two_dimensional_test_reads_the_element_it_names("CPU", hit)
        test_a_two_dimensional_test_reads_the_element_it_names("pure", hit)
    for hit in (17, 0, None):
        test_the_pure_search_is_python_sdfg_components_and_finds_the_first_hit(hit)
    test_a_test_that_gathers_through_the_counter_keeps_the_loop()
    test_a_test_dividing_by_data_is_refused()
    for positions in ([17, 400, 999], [0], []):
        test_a_last_match_is_a_search_from_the_end_and_records_once(positions)
    for positions in ([17, 400], []):
        test_a_last_match_into_memory_keeps_the_value_from_before_the_loop_on_no_hit(positions)
    for program in (first_match_without_break, first_match_by_flag):
        for positions in ([17, 400], [999], []):
            test_a_first_match_without_break_is_a_find_first_guarded_by_its_sentinel(program, positions)
    for program in (counts_matches, marks_every_match, sentinel_reopens, record_feeds_the_test):
        test_a_record_that_is_not_last_wins_is_refused(program)
