# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Kernels canonicalization once changed the value of: each must compute what its unchanged SDFG computes."""

import copy

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


def _assert_canon_keeps(program, name: str, arrays: dict, **symbols) -> None:
    """Run ``program`` unchanged and through CPU canonicalization on copies of ``arrays``; every array must agree."""
    results = []
    for tag in ("ref", "canon"):
        sdfg = program.to_sdfg(simplify=False)
        sdfg.name = f"{name}_{tag}"
        if tag == "canon":
            sdfg = canonicalize(sdfg, target="cpu")
            finalize_for_target(sdfg, "cpu")
        args = copy.deepcopy(arrays)
        sdfg(**args, **symbols)
        results.append(args)
    for key, value in results[0].items():
        assert np.allclose(value, results[1][key]), key


@dace.program
def diagonal_update(A: dace.float64[120], C: dace.float64[120]):
    for i in range(N):
        for j in range(M):
            A[i + j] = A[i + j] * 2 + C[i]


def test_a_write_offset_by_an_inner_iterator_stays_ordered():
    """``A[i + j]`` overlaps across ``i``: ``i`` stays a loop, and a tile of ``j`` does not move outside it."""
    rng = np.random.default_rng(0)
    _assert_canon_keeps(diagonal_update, "diagonal_update", {"A": rng.random(120), "C": rng.random(120)}, N=100, M=20)


def test_loop_fusion_keeps_a_write_read_pair_at_unrelated_fixed_indices():
    """``b[K]`` and ``b[M]`` are one cell when ``K == M``: the running value must not reach the second loop's reads."""
    sdfg = dace.SDFG("fusion_fixed_indices")
    for name in "abc":
        sdfg.add_array(name, [8], dace.float64)
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_symbol("M", dace.int64)
    first = LoopRegion("first", "i < 8", "i", "i = 1", "i = i + 1")
    body = first.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("rec", {"p"}, {"o", "s"}, "o = p + 1\ns = o")
    body.add_edge(body.add_read("a"), None, tasklet, "p", dace.Memlet("a[i - 1]"))
    body.add_edge(tasklet, "o", body.add_write("a"), None, dace.Memlet("a[i]"))
    body.add_edge(tasklet, "s", body.add_write("b"), None, dace.Memlet("b[K]"))
    second = LoopRegion("second", "i < 8", "i", "i = 1", "i = i + 1")
    body = second.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("rec", {"p", "q"}, {"o"}, "o = p + q")
    body.add_edge(body.add_read("c"), None, tasklet, "p", dace.Memlet("c[i - 1]"))
    body.add_edge(body.add_read("b"), None, tasklet, "q", dace.Memlet("b[M]"))
    body.add_edge(tasklet, "o", body.add_write("c"), None, dace.Memlet("c[i]"))
    sdfg.add_node(first, is_start_block=True)
    sdfg.add_node(second)
    sdfg.add_edge(first, second, dace.InterstateEdge())
    assert sdfg.apply_transformations_repeated(LoopFusion) == 0


@dace.program
def seeded_max(m: dace.float64[N, N], out: dace.float64[3]):
    v = 100.0
    xi = -1
    yi = -1
    for i in range(N):
        for j in range(N):
            if m[i, j] > v:
                v = m[i, j]
                xi = i
                yi = j
    out[0] = v
    out[1] = xi
    out[2] = yi


def test_a_max_seeded_above_the_data_keeps_its_seed():
    m = np.random.default_rng(1).random((6, 6))
    _assert_canon_keeps(seeded_max, "seeded_max", {"m": m, "out": np.zeros(3)}, N=6)


@dace.program
def reseeded_max(a: dace.float64[N], out: dace.float64[1]):
    x = a[0]
    a[0] = 50.0
    for i in range(1, N):
        if a[i] > x:
            x = a[i]
    out[0] = x


def test_a_seed_read_before_a_store_is_not_the_stored_value():
    a = np.random.default_rng(2).random(8)
    _assert_canon_keeps(reseeded_max, "reseeded_max", {"a": a, "out": np.zeros(1)}, N=8)


def test_a_symbol_read_round_the_enclosing_loop_stays_live():
    """``O[t] = k`` reads what the inner loop assigned on the outer loop's previous trip."""
    sdfg = dace.SDFG("live_round_the_back_edge")
    sdfg.add_array("O", [8], dace.int64)
    sdfg.add_array("P", [4], dace.int64)
    sdfg.add_symbol("k", dace.int64)
    outer = LoopRegion("outer", "t < 8", "t", "t = 0", "t = t + 1")
    read = outer.add_state("read", is_start_block=True)
    tasklet = read.add_tasklet("use", {}, {"o"}, "o = k")
    read.add_edge(tasklet, "o", read.add_write("O"), None, dace.Memlet("O[t]"))
    inner = LoopRegion("inner", "i < 4", "i", "i = 0", "i = i + 1")
    outer.add_node(inner)
    outer.add_edge(read, inner, dace.InterstateEdge())
    first = inner.add_state("first", is_start_block=True)
    write = inner.add_state("write")
    inner.add_edge(first, write, dace.InterstateEdge(assignments={"k": "i + t"}))
    tasklet = write.add_tasklet("put", {}, {"o"}, "o = k")
    write.add_edge(tasklet, "o", write.add_write("P"), None, dace.Memlet("P[i]"))
    start = sdfg.add_state("start", is_start_block=True)
    sdfg.add_node(outer)
    sdfg.add_edge(start, outer, dace.InterstateEdge(assignments={"k": "100"}))
    ParallelizeLoops().apply_pass(sdfg, {})
    assert inner in list(outer.nodes())


if __name__ == "__main__":
    test_a_write_offset_by_an_inner_iterator_stays_ordered()
    test_a_symbol_read_round_the_enclosing_loop_stays_live()
    test_a_max_seeded_above_the_data_keeps_its_seed()
    test_a_seed_read_before_a_store_is_not_the_stored_value()
    test_loop_fusion_keeps_a_write_read_pair_at_unrelated_fixed_indices()
