# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for GPU grid-strided tiling transformation."""
from typing import List, Tuple
import pytest
import dace
from dace.transformation.dataflow import TrivialTaskletElimination, GPUGridStridedTiling
import numpy as np
import scipy.sparse as sparse


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


@pytest.mark.gpu
def test_gpu_grid_stride_tiling():
    M = 300
    N = 300

    @dace.program
    def dummy(A: dace.float32[M, N], B: dace.float32[M, N]):
        for i in dace.map[0:M]:
            for j in dace.map[0:N]:
                A[i, j] = B[i, j] + 1.0

    sdfg = dummy.to_sdfg()
    sdfg.simplify()
    ime, jme = find_map_entry(sdfg, ["i", "j"])
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    sdfg.apply_gpu_transformations()
    GPUGridStridedTiling.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=jme)

    sdfg.validate()

    B = np.random.rand(M, N).astype(np.float32)
    A_ref = np.zeros((M, N), dtype=np.float32)
    A_test = np.zeros((M, N), dtype=np.float32)
    A_ref = B + 1.0
    sdfg(A=A_test, B=B)
    assert np.allclose(A_ref, A_test)


@pytest.mark.gpu
def test_gpu_grid_stride_tiling_with_indirection():

    M = 300
    N = 300
    K = 300
    density = 0.01
    dtype = np.float32
    A = sparse.random(M, N, density=density, format='csr', dtype=dtype)
    nnz = A.nnz
    B = np.random.rand(M, K).astype(dtype)
    C = np.random.rand(K, N).astype(dtype)
    D_test = np.zeros_like(A.data)
    D_ref = np.zeros_like(A.data)

    @dace.program
    def sddmm(D_vals: dace.float32[nnz], A2_crd: dace.int32[nnz], A2_pos: dace.int32[M + 1], A_vals: dace.float32[nnz],
              B: dace.float32[M, K], C: dace.float32[K, N]):
        for i in dace.map[0:M]:
            for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
                for k in dace.map[0:K]:
                    D_vals[j] += A_vals[j] * B[i, k] * C[k, A2_crd[j]]

    sdfg = sddmm.to_sdfg()
    sdfg.simplify()
    ime, jme, _ = find_map_entry(sdfg, ["i", "j", "k"])
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)
    sdfg.apply_gpu_transformations()
    GPUGridStridedTiling.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=jme)
    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.wcr:
            e.data.wcr_nonatomic = True

    sdfg.validate()

    # reference
    for i in range(M):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            D_ref[j] += A.data[j] * (B[i, :] @ C[:, A.indices[j]])

    sdfg(A_vals=np.copy(A.data),
         A2_crd=np.copy(A.indices),
         A2_pos=A.indptr,
         B=B,
         C=C,
         D_vals=D_test)
    assert np.allclose(D_ref, D_test)


if __name__ == '__main__':
    test_gpu_grid_stride_tiling()
    test_gpu_grid_stride_tiling_with_indirection()
