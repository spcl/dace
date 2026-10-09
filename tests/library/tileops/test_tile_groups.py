# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The group of a tile node: ``dace.tile`` calls make ``BLOCK`` nodes, which a CPU runs as ``THREAD`` nodes and a GPU
kernel spreads over the threads of a block, with the tiles in shared memory."""

import copy

import numpy as np
import pytest

import dace
from dace.libraries.tileops.dispatch import TileGroup
from dace.libraries.tileops.lanes import LaneDistribution, distributed_lanes, nested_loops
from dace.libraries.tileops.nodes.tile_op import TileOp

M, N, K = (dace.symbol(name) for name in "MNK")
BM, BN, BK = 16, 16, 8


@dace.program
def blocked_gemm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M:BM, 0:N:BN]:
        acc = dace.define_local([BM, BN], dace.float64, storage=dace.StorageType.Register)
        dace.tile.fill(acc, 0.0)
        for k in range(0, K, BK):
            dace.tile.mma(A[i : i + BM, k : k + BK], B[k : k + BK, j : j + BN], acc)
        dace.tile.store(C[i : i + BM, j : j + BN], acc)


def tile_nodes(sdfg: dace.SDFG) -> list[TileOp]:
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, TileOp)]


def gemm_operands() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"A": rng.random((64, 48)), "B": rng.random((48, 32)), "C": np.zeros((64, 32))}


def test_tile_calls_make_block_nodes():
    sdfg = blocked_gemm.to_sdfg()
    assert {node.group for node in tile_nodes(sdfg)} == {TileGroup.BLOCK}


def test_a_block_node_lowers_as_a_thread_node_on_a_cpu():
    block = blocked_gemm.to_sdfg()
    block.name = "blocked_gemm_block_group"
    thread = copy.deepcopy(block)
    for node in tile_nodes(thread):
        node.group = TileGroup.THREAD
    code_of = lambda sdfg: next(obj.clean_code for obj in sdfg.generate_code() if obj.name == sdfg.name)
    assert code_of(block) == code_of(thread).replace(thread.name, block.name)


def test_blocked_gemm_with_mma_on_a_cpu():
    sdfg = blocked_gemm.to_sdfg()
    sdfg.name = "blocked_gemm_cpu"
    operands = gemm_operands()
    sdfg(**operands, M=64, N=32, K=48)
    np.testing.assert_allclose(operands["C"], operands["A"] @ operands["B"], rtol=1e-12)


def test_distributed_lanes_are_one_thread_strided_loop():
    body = "_c[(__l0 * 8) + __l1] = 1;"
    assert "for (std::size_t __l0" in nested_loops([4, 8], body)
    with distributed_lanes(LaneDistribution("__tile_t", 16)) as distribution:
        code = nested_loops([4, 8], body)
    assert distribution.loops == 1
    assert code.startswith("for (std::size_t __e = __tile_t; __e < 32; __e += 16) {")
    assert "const std::size_t __l0 = (__e / 8) % 4;" in code and "const std::size_t __l1 = (__e / 1) % 8;" in code


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["legacy", "experimental"])
def test_blocked_gemm_with_mma_on_a_gpu(implementation: str):
    sdfg = blocked_gemm.to_sdfg()
    sdfg.name = f"blocked_gemm_gpu_{implementation}"
    sdfg.apply_gpu_transformations()
    operands = gemm_operands()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value=implementation):
        sdfg(**operands, M=64, N=32, K=48)
        kernel = next(
            obj.clean_code for obj in sdfg.generate_code() if obj.target.target_name in ("cuda", "experimental_cuda")
        )
    np.testing.assert_allclose(operands["C"], operands["A"] @ operands["B"], rtol=1e-12)
    # The tiles every thread of a block reads live in shared memory
    for tile in ("acc[256]", "A_tile[128]", "B_tile[128]"):
        assert f"__shared__ double {tile}" in kernel


L = dace.symbol("L")


@dace.program
def fused_rows(A: dace.float64[L, 64], B: dace.float64[L, 64], D: dace.float64[L, 64], S: dace.float64[L]):
    for r in dace.map[0:L]:
        row = dace.tile.fma(A[r, :], B[r, :], dace.tile.exp(A[r, :]))
        dace.tile.store(D[r, :], row)
        S[r : r + 1] = dace.tile.sum(row)


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["legacy", "experimental"])
def test_elementwise_and_sum_on_a_gpu(implementation: str):
    sdfg = fused_rows.to_sdfg()
    sdfg.name = f"fused_rows_gpu_{implementation}"
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(1)
    a, b = rng.random((4, 64)), rng.random((4, 64))
    d, s = np.zeros((4, 64)), np.zeros(4)
    with dace.config.set_temporary("compiler", "cuda", "implementation", value=implementation):
        sdfg(A=a, B=b, D=d, S=s, L=4)
    expected = a * b + np.exp(a)
    np.testing.assert_allclose(d, expected, rtol=1e-12)
    np.testing.assert_allclose(s, expected.sum(axis=1), rtol=1e-12)


if __name__ == "__main__":
    test_tile_calls_make_block_nodes()
    test_a_block_node_lowers_as_a_thread_node_on_a_cpu()
    test_blocked_gemm_with_mma_on_a_cpu()
    test_distributed_lanes_are_one_thread_strided_loop()
    for implementation in ("legacy", "experimental"):
        test_blocked_gemm_with_mma_on_a_gpu(implementation)
        test_elementwise_and_sum_on_a_gpu(implementation)
