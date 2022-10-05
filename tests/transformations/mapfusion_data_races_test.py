# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace import nodes
from dace.transformation.dataflow import MapFusion
from dace.transformation.auto.auto_optimize import greedy_fuse

N = dace.symbol('N')


@dace.program
def rw_data_race(A: dace.float32[10, 10], B: dace.float32[10, 10]):
    tmp = A[:-1, :-1] + A[1:, 1:]
    B[1:-1, 1:-1] = tmp[:-1, :-1] + tmp[:-1, :-1]


@dace.program
def flip(A: dace.float64[N]):
    B = np.ndarray((N, ), dtype=np.float64)
    for i in dace.map[0:N]:
        B[i] = A[N - 1 - i]
    return B


@dace.program
def offset(A: dace.float64[N]):
    B = np.ndarray((N - 1, ), dtype=np.float64)
    for i in dace.map[0:N - 1]:
        B[i] = A[i + 1]
    return B


@dace.program
def rw_data_race_2(A: dace.float64[20], B: dace.float64[20]):
    A[:10] += 3.0 * flip(A[:10])


@dace.program
def rw_data_race_3(A: dace.float64[20], B: dace.float64[20]):
    A[:10] += 3.0 * offset(A[:11])


def test_rw_data_race():
    sdfg = rw_data_race.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MapFusion)
    map_entry_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert (len(map_entry_nodes) > 1)


def test_rw_data_race_2_mf():
    sdfg = rw_data_race_2.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MapFusion)
    map_entry_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert (len(map_entry_nodes) > 1)


def test_rw_data_race_2_sgf():
    sdfg = rw_data_race_2.to_sdfg(simplify=True)
    greedy_fuse(sdfg, True)
    map_entry_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert (len(map_entry_nodes) > 1)


def test_rw_data_race_3_sgf():
    sdfg = rw_data_race_3.to_sdfg(simplify=True)
    greedy_fuse(sdfg, True)
    map_entry_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert (len(map_entry_nodes) > 1)


if __name__ == "__main__":
    test_rw_data_race()
    test_rw_data_race_2_mf()
    test_rw_data_race_2_sgf()
    test_rw_data_race_3_sgf()
