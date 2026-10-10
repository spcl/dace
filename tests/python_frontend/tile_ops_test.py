# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.tile.add`` and ``dace.tile.masked_copy`` build the tile library nodes from a ``@dace.program``."""

import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError
from dace.libraries.tileops import MaskedCopyLibraryNode, TileBinop

N = dace.symbol("N")
Register = dace.StorageType.Register
LANES = 8


def count_nodes(sdfg: dace.SDFG, node_type: type) -> int:
    return sum(1 for node, state in sdfg.all_nodes_recursive() if isinstance(node, node_type))


def random_mask(length: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(length) > 0.5


@dace.program
def add_windows_in_a_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N:8]:
        C[i : i + 8] = dace.tile.add(A[i : i + 8], B[i : i + 8])


@dace.program
def add_whole_arrays(A: dace.float64[8], B: dace.float64[8], C: dace.float64[8]):
    C[:] = dace.tile.add(A, B)


@dace.program
def add_matrix_windows(A: dace.float64[16, 16], B: dace.float64[16, 16], C: dace.float64[16, 16]):
    for i, j in dace.map[0:16:4, 0:16:8]:
        C[i : i + 4, j : j + 8] = dace.tile.add(A[i : i + 4, j : j + 8], B[i : i + 4, j : j + 8])


@dace.program
def add_a_row_to_a_row(A: dace.float64[4, 16], B: dace.float64[4, 16], C: dace.float64[4, 16]):
    for i, j in dace.map[0:4, 0:16:8]:
        C[i, j : j + 8] = dace.tile.add(A[i, j : j + 8], B[i, j : j + 8])


@dace.program
def masked_window_copy(A: dace.float64[N], M: dace.bool_[N], C: dace.float64[N]):
    for i in dace.map[0:N:8]:
        dace.tile.masked_copy(C[i : i + 8], A[i : i + 8], M[i : i + 8])


@dace.program
def masked_load_into_a_tile(A: dace.float64[N], M: dace.bool_[N], C: dace.float64[N]):
    for i in dace.map[0:N:8]:
        tile = dace.define_local([8], dace.float64, storage=dace.StorageType.Register)
        dace.tile.masked_copy(tile, A[i : i + 8], M[i : i + 8])
        C[i : i + 8] = tile


@dace.program
def masked_store_from_a_tile(A: dace.float64[N], M: dace.bool_[N], C: dace.float64[N]):
    for i in dace.map[0:N:8]:
        tile = dace.define_local([8], dace.float64, storage=dace.StorageType.Register)
        tile[:] = A[i : i + 8]
        dace.tile.masked_copy(C[i : i + 8], tile, M[i : i + 8])


@dace.program
def masked_copy_under_a_register_mask(A: dace.float64[N], M: dace.bool_[N], C: dace.float64[N]):
    for i in dace.map[0:N:8]:
        mask = dace.define_local([8], dace.bool_, storage=dace.StorageType.Register)
        mask[:] = M[i : i + 8]
        dace.tile.masked_copy(C[i : i + 8], A[i : i + 8], mask)


@dace.program
def masked_copy_of_a_sum(A: dace.float64[N], B: dace.float64[N], M: dace.bool_[N], C: dace.float64[N]):
    for i in dace.map[0:N:8]:
        dace.tile.masked_copy(C[i : i + 8], dace.tile.add(A[i : i + 8], B[i : i + 8]), M[i : i + 8])


@dace.program
def masked_copy_outside_a_map(A: dace.float64[16], M: dace.bool_[16], C: dace.float64[16]):
    dace.tile.masked_copy(C[4:12], A[4:12], M[4:12])


@dace.program
def lanes_differ(A: dace.float64[16], M: dace.bool_[16], C: dace.float64[16]):
    dace.tile.masked_copy(C[0:8], A[0:4], M[0:8])


@dace.program
def types_differ(A: dace.float32[16], M: dace.bool_[16], C: dace.float64[16]):
    dace.tile.masked_copy(C[0:8], A[0:8], M[0:8])


@dace.program
def mask_is_no_bool(A: dace.float64[16], M: dace.float64[16], C: dace.float64[16]):
    dace.tile.masked_copy(C[0:8], A[0:8], M[0:8])


@dace.program
def tile_to_tile(A: dace.float64[16], M: dace.bool_[16], C: dace.float64[16]):
    first = dace.define_local([8], dace.float64, storage=dace.StorageType.Register)
    second = dace.define_local([8], dace.float64, storage=dace.StorageType.Register)
    dace.tile.masked_copy(second, first, M[0:8])


@dace.program
def sum_of_unequal_lanes(A: dace.float64[16], B: dace.float64[16], C: dace.float64[16]):
    C[0:8] = dace.tile.add(A[0:8], B[0:4])


def test_the_sum_of_two_windows_in_a_map():
    a, b, c = np.random.default_rng(1).random(32), np.random.default_rng(2).random(32), np.zeros(32)
    sdfg = add_windows_in_a_map.to_sdfg(simplify=False)
    assert count_nodes(sdfg, TileBinop) == 1
    assert count_nodes(sdfg, MaskedCopyLibraryNode) == 2
    sdfg(A=a, B=b, C=c, N=32)
    np.testing.assert_array_equal(c, a + b)


def test_the_sum_of_two_whole_arrays():
    a, b, c = np.arange(8.0), np.arange(8.0) * 3, np.zeros(8)
    add_whole_arrays(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a + b)


def test_the_sum_of_two_windows_of_a_matrix_is_a_two_dimensional_tile():
    a, b, c = (np.random.default_rng(seed).random((16, 16)) for seed in (3, 4, 5))
    sdfg = add_matrix_windows.to_sdfg(simplify=False)
    assert [tuple(node.widths) for node, state in sdfg.all_nodes_recursive() if isinstance(node, TileBinop)] == [(4, 8)]
    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a + b)


def test_a_window_of_extent_one_in_a_dim_is_no_lane_dim():
    a, b, c = (np.random.default_rng(seed).random((4, 16)) for seed in (6, 7, 8))
    sdfg = add_a_row_to_a_row.to_sdfg(simplify=False)
    assert [tuple(node.widths) for node, state in sdfg.all_nodes_recursive() if isinstance(node, TileBinop)] == [(8,)]
    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a + b)


def test_a_masked_copy_of_a_window_keeps_the_lanes_the_mask_switches_off():
    a, mask = np.random.default_rng(9).random(32), random_mask(32, 10)
    c = np.full(32, -1.0)
    sdfg = masked_window_copy.to_sdfg(simplify=False)
    assert count_nodes(sdfg, MaskedCopyLibraryNode) == 3
    sdfg(A=a, M=mask, C=c, N=32)
    np.testing.assert_array_equal(c, np.where(mask, a, -1.0))


def test_a_masked_copy_into_a_tile_zeroes_the_lanes_the_mask_switches_off():
    a, mask = np.random.default_rng(11).random(32) + 1, random_mask(32, 12)
    c = np.full(32, -1.0)
    masked_load_into_a_tile(A=a, M=mask, C=c, N=32)
    np.testing.assert_array_equal(c, np.where(mask, a, 0.0))


def test_a_masked_copy_from_a_tile_keeps_the_lanes_the_mask_switches_off():
    a, mask = np.random.default_rng(13).random(32), random_mask(32, 14)
    c = np.full(32, -1.0)
    masked_store_from_a_tile(A=a, M=mask, C=c, N=32)
    np.testing.assert_array_equal(c, np.where(mask, a, -1.0))


def test_a_mask_may_be_a_register_tile():
    a, mask = np.random.default_rng(15).random(32), random_mask(32, 16)
    c = np.full(32, -1.0)
    masked_copy_under_a_register_mask(A=a, M=mask, C=c, N=32)
    np.testing.assert_array_equal(c, np.where(mask, a, -1.0))


def test_a_masked_copy_takes_the_sum_of_two_windows():
    a, b, mask = np.random.default_rng(17).random(32), np.random.default_rng(18).random(32), random_mask(32, 19)
    c = np.full(32, -1.0)
    masked_copy_of_a_sum(A=a, B=b, M=mask, C=c, N=32)
    np.testing.assert_array_equal(c, np.where(mask, a + b, -1.0))


def test_a_masked_copy_outside_a_map():
    a, mask = np.random.default_rng(20).random(16), random_mask(16, 21)
    c = np.full(16, -1.0)
    masked_copy_outside_a_map(A=a, M=mask, C=c)
    expected = np.full(16, -1.0)
    expected[4:12] = np.where(mask[4:12], a[4:12], -1.0)
    np.testing.assert_array_equal(c, expected)


@dace.program
def elementwise_calls(
    A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], M: dace.bool_[N], D: dace.float64[N], S: dace.float64[N]
):
    for i in dace.map[0:N:8]:
        fused = dace.tile.fma(A[i : i + 8], B[i : i + 8], dace.tile.exp(C[i : i + 8]))
        larger = dace.tile.maximum(A[i : i + 8], dace.tile.sub(B[i : i + 8], C[i : i + 8]))
        chosen = dace.tile.where(M[i : i + 8], fused, larger)
        dace.tile.store(D[i : i + 8], chosen)
        S[i : i + 1] = dace.tile.sum(chosen)


def test_elementwise_calls_where_and_sum():
    rng = np.random.default_rng(30)
    a, b, c = rng.random(16), rng.random(16), rng.random(16)
    mask = random_mask(16, 31)
    d, s = np.zeros(16), np.zeros(16)
    elementwise_calls(A=a, B=b, C=c, M=mask, D=d, S=s)
    expected = np.where(mask, a * b + np.exp(c), np.maximum(a, b - c))
    np.testing.assert_allclose(d, expected, rtol=1e-12)
    np.testing.assert_allclose(s[::8], expected.reshape(2, 8).sum(axis=1), rtol=1e-12)


@pytest.mark.parametrize(
    "program,message",
    [
        (lanes_differ, "different lanes"),
        (types_differ, "different types"),
        (mask_is_no_bool, "must be of type bool"),
        (tile_to_tile, "between two tiles"),
        (sum_of_unequal_lanes, "different lanes"),
    ],
)
def test_a_call_that_does_not_fit_the_tiles_is_refused(program, message):
    with pytest.raises(DaceSyntaxError, match=message):
        program.to_sdfg(simplify=False)


if __name__ == "__main__":
    test_the_sum_of_two_windows_in_a_map()
    test_the_sum_of_two_whole_arrays()
    test_the_sum_of_two_windows_of_a_matrix_is_a_two_dimensional_tile()
    test_a_window_of_extent_one_in_a_dim_is_no_lane_dim()
    test_a_masked_copy_of_a_window_keeps_the_lanes_the_mask_switches_off()
    test_a_masked_copy_into_a_tile_zeroes_the_lanes_the_mask_switches_off()
    test_a_masked_copy_from_a_tile_keeps_the_lanes_the_mask_switches_off()
    test_a_mask_may_be_a_register_tile()
    test_a_masked_copy_takes_the_sum_of_two_windows()
    test_a_masked_copy_outside_a_map()
    test_elementwise_calls_where_and_sum()
    test_a_call_that_does_not_fit_the_tiles_is_refused(lanes_differ, "different lanes")
    test_a_call_that_does_not_fit_the_tiles_is_refused(types_differ, "different types")
    test_a_call_that_does_not_fit_the_tiles_is_refused(mask_is_no_bool, "must be of type bool")
    test_a_call_that_does_not_fit_the_tiles_is_refused(tile_to_tile, "between two tiles")
    test_a_call_that_does_not_fit_the_tiles_is_refused(sum_of_unequal_lanes, "different lanes")
