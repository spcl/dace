# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LiftBreakToFindFirst: an early-exit loop becomes FindFirst, a loop below the hit and the effect once."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.find_first import FindFirst
from dace.sdfg.state import BreakBlock
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
