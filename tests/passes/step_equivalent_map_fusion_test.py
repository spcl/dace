# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Full map fusion treats ranges that differ only in step (``0:2N:2`` and ``0:N:1``) as the same domain."""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.fuse_maps import FuseMaps
from dace.transformation.passes.iteration_domain import equal_trip_counts, exact_trip_count, same_trip_count

N = dace.symbol('N', dtype=dace.int64, nonnegative=True)


@pytest.mark.parametrize('first, second, equal', [
    ((0, 2 * N - 1, 2), (0, N - 1, 1), True),
    ((0, 2 * N - 2, 2), (0, N - 1, 1), True),
    ((0, 3 * N - 1, 3), (0, N - 1, 1), True),
    ((0, 2 * N - 1, 2), (0, N, 1), False),
    ((0, 9, 2), (0, 4, 1), True),
])
def test_trip_counts_are_compared_exactly(first, second, equal):
    assert same_trip_count(exact_trip_count(*first), exact_trip_count(*second)) == equal
    assert equal_trip_counts([first], [second]) == equal


@dace.program
def strided_producer_unit_consumer(A: dace.float64[2 * N], B: dace.float64[2 * N], C: dace.float64[N]):
    for i in dace.map[0:2 * N:2]:
        B[i] = A[i] + 1.0
    for j in dace.map[0:N]:
        C[j] = B[2 * j] * 2.0


def count_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))


def test_step_equivalent_maps_fuse_into_one_unit_step_map():
    sdfg = strided_producer_unit_consumer.to_sdfg(simplify=True)
    assert count_maps(sdfg) == 2

    FuseMaps(validate=False, validate_all=False).apply_pass(sdfg, {})

    map_entries = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)]
    assert len(map_entries) == 1
    (begin, end, step), = map_entries[0].map.range.ranges
    assert (begin, step) == (0, 1)
    assert str(dace.symbolic.simplify(end)) == 'N - 1'
    sdfg.validate()


def test_fused_step_equivalent_maps_compute_the_same_values():
    sdfg = strided_producer_unit_consumer.to_sdfg(simplify=True)
    FuseMaps(validate=False, validate_all=False).apply_pass(sdfg, {})
    size = 16
    A = np.arange(2 * size, dtype=np.float64)
    B = np.zeros(2 * size)
    C = np.zeros(size)
    expected_B = np.zeros(2 * size)
    expected_B[0::2] = A[0::2] + 1.0

    sdfg(A=A, B=B, C=C, N=size)

    assert np.allclose(B, expected_B)
    assert np.allclose(C, expected_B[0::2] * 2.0)


@dace.program
def different_trip_counts(A: dace.float64[2 * N], B: dace.float64[2 * N], C: dace.float64[N + 1]):
    for i in dace.map[0:2 * N:2]:
        B[i] = A[i] + 1.0
    for j in dace.map[0:N + 1]:
        C[j] = A[j] * 2.0


def test_maps_with_different_trip_counts_keep_their_ranges():
    sdfg = different_trip_counts.to_sdfg(simplify=True)
    before = sorted(str(node.map.range) for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))

    FuseMaps(validate=False, validate_all=False).apply_pass(sdfg, {})

    after = sorted(str(node.map.range) for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))
    assert after == before
