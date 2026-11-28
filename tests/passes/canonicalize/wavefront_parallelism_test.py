# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization exposes the anti-diagonal wavefront of a 2-D DP or Gauss-Seidel nest as a parallel map.

The programs are written the way HPCAgent-Bench emits them (``numpyto_c.dace_emit``), because that is the
shape the skew has to recognise: nussinov calls a helper (``match``) whose arguments become interstate
assignments, and its inner max-reduction is privatized into a scalar. Neither may keep the nest sequential:
the HPCAgent-Bench perf gate measured canon at 2.5x the compiled baseline while the diagonal stayed serial.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dtype=dace.int64, positive=True)
TSTEPS = dace.symbol("TSTEPS", dtype=dace.int64, positive=True)


@dace.program
def match(b1: dace.int32, b2: dace.int32, complement_sum: dace.int64, pair_bonus: dace.int64):
    if b1 + b2 == complement_sum:
        return pair_bonus
    else:
        return 0


@dace.program
def nussinov(seq: dace.int32[N], table: dace.int32[N, N], complement_sum: dace.int64, pair_bonus: dace.int64):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(
                        table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j], complement_sum, pair_bonus)
                    )
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])


def nussinov_reference(seq: np.ndarray, table: np.ndarray, complement_sum: int, pair_bonus: int) -> None:
    n = seq.shape[0]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < n:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < n:
                bonus = pair_bonus if seq[i] + seq[j] == complement_sum else 0
                table[i, j] = max(table[i, j], table[i + 1, j - 1] + (bonus if i < j - 1 else 0))
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])


@dace.program
def seidel_2d(A: dace.float64[N, N]):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            A[i, 1:-1] = A[i, 1:-1] + (
                A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]
            )
            for j in range(1, N - 1):
                A[i, j] = A[i, j] + A[i, j - 1]
                A[i, j] = A[i, j] / 9.0


def seidel_2d_reference(A: np.ndarray, tsteps: int) -> None:
    n = A.shape[0]
    for _ in range(tsteps):
        for i in range(1, n - 1):
            A[i, 1:-1] += (
                A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]
            )
            for j in range(1, n - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def canonical_cpu(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    return sdfg


def assert_parallel_diagonal(sdfg: dace.SDFG) -> None:
    """A sequential skewed diagonal loop, and inside it a multicore map whose range depends on the diagonal."""
    diagonals = {
        loop.loop_variable
        for loop in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(loop, LoopRegion) and loop.loop_variable.startswith("_skew_t")
    }
    assert diagonals, "the nest was not skewed: no diagonal loop"
    parallel = [
        entry
        for entry, _ in sdfg.all_nodes_recursive()
        if isinstance(entry, nodes.MapEntry)
        and entry.map.schedule == dace.ScheduleType.CPU_Multicore
        and {str(s) for s in entry.map.range.free_symbols} & diagonals
    ]
    assert parallel, "the diagonal is not a parallel map"


def test_nussinov_wavefront_is_a_parallel_diagonal():
    sdfg = canonical_cpu(nussinov)
    assert_parallel_diagonal(sdfg)
    n = 64
    rng = np.random.default_rng(20261015)
    seq = rng.integers(0, 4, n).astype(np.int32)
    table, want = np.zeros((n, n), np.int32), np.zeros((n, n), np.int32)
    sdfg(seq=seq, table=table, complement_sum=3, pair_bonus=1, N=n)
    nussinov_reference(seq, want, 3, 1)
    assert np.array_equal(table, want)


def test_seidel_2d_wavefront_is_a_parallel_diagonal():
    sdfg = canonical_cpu(seidel_2d)
    assert_parallel_diagonal(sdfg)
    n, tsteps = 40, 3
    A = np.random.default_rng(20261015).random((n, n))
    want = A.copy()
    sdfg(A=A, N=n, TSTEPS=tsteps)
    seidel_2d_reference(want, tsteps)
    assert np.allclose(A, want, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_nussinov_wavefront_is_a_parallel_diagonal()
    test_seidel_2d_wavefront_is_a_parallel_diagonal()
