# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``canonicalize(sdfg, stats=...)``: which stage settled each loop, and the time spent in total and in the solvers."""

import json

import dace
from dace.transformation.passes.canonicalize import canonicalize

N, K, NNZ = (dace.symbol(name) for name in ("N", "K", "NNZ"))


@dace.program
def hybrid_sparse(
    row_ptr: dace.int32[N + 1], col: dace.int32[NNZ], val: dace.float64[NNZ], x: dace.float64[N], y: dace.float64[N]
):
    for i in range(N):
        s = 0.0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            s += val[k] * x[col[k]]
        if i < K:
            y[i] = s
        else:
            y[i] = s + y[i - 1]


@dace.program
def carried_window(a: dace.float64[N], bb: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(i):
            a[i] += bb[j, i] * a[i - j - 1]


@dace.program
def wavefront(A: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            A[i, j] = A[i - 1, j] + A[i, j - 1]


def test_stats_name_the_stage_of_every_loop():
    stats = {}
    canonicalize(hybrid_sparse.to_sdfg(simplify=True), stats=stats)
    assert stats["kernel"].endswith("hybrid_sparse")
    assert stats["loops_in"] == 2
    assert stats["seconds"] > 0
    # The recurrence a split exposes is lifted to a scan; the rest maps at level one.
    assert stats["outcomes"]["lift:LoopToScan"] == 1, stats
    assert stats["outcomes"]["level1"] >= 1, stats
    assert not stats["remaining"], stats
    assert all(settled["stage"] and settled["unit"] for settled in stats["settled"])


def test_a_loop_left_sequential_names_its_proof():
    stats = {}
    canonicalize(carried_window.to_sdfg(simplify=True), stats=stats)
    assert [loop["outcome"] for loop in stats["remaining"]] == ["sequential_witness"], stats
    assert stats["z3_seconds"] > 0, stats


def test_a_skewed_nest_is_level_two_and_times_isl():
    stats = {}
    canonicalize(wavefront.to_sdfg(simplify=True), stats=stats)
    assert stats["outcomes"]["level2"] >= 1, stats
    assert {loop["outcome"] for loop in stats["remaining"]} == {"wavefront"}, stats
    assert stats["isl_seconds"] > 0, stats


def test_stats_append_one_json_line_and_change_nothing(tmp_path):
    path = tmp_path / "stats.jsonl"
    plain, recorded = (hybrid_sparse.to_sdfg(simplify=True) for _ in range(2))
    canonicalize(plain)
    canonicalize(recorded, stats=str(path))
    canonicalize(hybrid_sparse.to_sdfg(simplify=True), stats=str(path))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kernel"].endswith("hybrid_sparse")
    assert plain.generate_code()[0].clean_code == recorded.generate_code()[0].clean_code


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_stats_name_the_stage_of_every_loop()
    test_a_loop_left_sequential_names_its_proof()
    test_a_skewed_nest_is_level_two_and_times_isl()
    with tempfile.TemporaryDirectory() as directory:
        test_stats_append_one_json_line_and_change_nothing(pathlib.Path(directory))
