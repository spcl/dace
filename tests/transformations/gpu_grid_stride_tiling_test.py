# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for GPU grid-strided tiling transformation."""
from numpy.random import default_rng
from typing import List, Tuple
from copy import deepcopy
import numpy as np
import cupy as cp
import dace
from dace.transformation.dataflow import MapInterchange, StripMining, MapReduceFusion, MapExpansion, MapToForLoop, TrivialTaskletElimination, GPUGridStridedTiling
from dace.transformation.interstate import GPUTransformSDFG


def copy_to_gpu(sdfg):
    for k, v in sdfg.arrays.items():
        if not v.transient and isinstance(v, dace.data.Array):
            v.storage = dace.dtypes.StorageType.GPU_Global


def find_map_entry(sdfg: dace.SDFG, map_name_list: List[str]) -> Tuple[dace.sdfg.nodes.MapEntry]:
    if isinstance(map_name_list, str):
        map_name_list = [
            map_name_list,
        ]
    ret_list = [None] * len(map_name_list)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                for i, map_name in enumerate(map_name_list):
                    if map_name == node.map.params[0]:
                        ret_list[i] = node
    # check if all map entries are found
    assert all([x is not None for x in ret_list])

    # unpack if only one map entry is found
    if len(ret_list) == 1:
        return ret_list[0]
    else:
        return tuple(ret_list)


def test_gpu_grid_stride_tiling():
    M = dace.symbol('M')
    N = dace.symbol('N')

    @dace.program
    def dummy(A: dace.float32[M, N], B: dace.float32[M, N]):
        for i in dace.map[1:M:2]:
            for j in dace.map[3:N:4]:
                A[i, j] = B[i, j] + 1.0

    sdfg = dummy.to_sdfg()
    sdfg.simplify()
    ime, jme = find_map_entry(sdfg, ["i", "j"])
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    copy_to_gpu(sdfg)
    GPUGridStridedTiling.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=jme)
    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.wcr:
            e.data.wcr_nonatomic = True

    sdfg.validate()


def test_gpu_grid_stride_tiling_with_indirection():

    M = dace.symbol('M')
    N = dace.symbol('N')
    K = dace.symbol('K')
    nnz_A = dace.symbol('nnz_A')
    nnz_D = dace.symbol('nnz_D')

    @dace.program
    def sddmm(D_vals: dace.float32[nnz_D], A2_crd: dace.int32[nnz_A], A2_pos: dace.int32[M + 1],
              A_vals: dace.float32[nnz_A], B: dace.float32[M, K], C: dace.float32[K, N]):
        for i in dace.map[0:M]:
            for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
                for k in dace.map[0:K]:
                    D_vals[j] += A_vals[j] * B[i, k] * C[k, A2_crd[j]]

    sdfg = sddmm.to_sdfg()
    sdfg.simplify()
    ime, jme, _ = find_map_entry(sdfg, ["i", "j", "k"])
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    copy_to_gpu(sdfg)
    GPUGridStridedTiling.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=jme)
    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.wcr:
            e.data.wcr_nonatomic = True

    sdfg.validate()


if __name__ == '__main__':
    test_gpu_grid_stride_tiling()
    test_gpu_grid_stride_tiling_with_indirection()
