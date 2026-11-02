# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map lifted on a proof ``LoopToMap`` cannot redo survives a second canonicalization."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64, positive=True)


@dace.program
def triangular_wavefront(a: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(i, N):
            a[i, j] = (a[i, j] + a[i - 1, j] + a[i, j - 1]) / 3.0


@dace.program
def threshold_pack(src: dace.float64[N], packed: dace.float64[N], count: dace.int64[1]):
    n = 0
    for i in range(N):
        if src[i] > 0.0:
            packed[n] = src[i] * 2.0
            n = n + 1
    count[0] = n


def map_count(sdfg: dace.SDFG) -> int:
    return sum(isinstance(node, nodes.MapEntry) for node, _ in sdfg.all_nodes_recursive())


def canonicalized_twice(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = name
    canonicalize(sdfg)
    first = map_count(sdfg)
    assert first >= 1
    canonicalize(sdfg)
    assert map_count(sdfg) == first
    return sdfg


def test_a_wavefront_map_survives_a_second_canonicalization():
    sdfg = canonicalized_twice(triangular_wavefront, "idempotent_wavefront")
    got = np.arange(36.0).reshape(6, 6).copy()
    expected = got.copy()
    for i in range(1, 6):
        for j in range(i, 6):
            expected[i, j] = (expected[i, j] + expected[i - 1, j] + expected[i, j - 1]) / 3.0
    sdfg(a=got, N=6)
    assert np.allclose(got, expected)


def test_a_stream_compaction_map_survives_a_second_canonicalization():
    sdfg = canonicalized_twice(threshold_pack, "idempotent_compaction")
    src = np.array([1.0, -1.0, 2.0, -2.0, 3.0, 0.0])
    packed, count = np.zeros(6), np.zeros(1, dtype=np.int64)
    sdfg(src=src, packed=packed, count=count, N=6)
    assert count[0] == 3 and np.array_equal(packed, [2.0, 4.0, 6.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    test_a_wavefront_map_survives_a_second_canonicalization()
    test_a_stream_compaction_map_survives_a_second_canonicalization()
