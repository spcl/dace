# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``NestToContraction``: a perfect loop/map nest around one contraction library node becomes one ``Einsum``."""

import numpy as np
import pytest

import dace
from dace.libraries.blas import nodes as blas
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

NB, N, K, M = (dace.symbol(s, dace.int64) for s in ("NB", "N", "K", "M"))
SIZES = dict(NB=5, N=6, K=7, M=8)
CONTRACTIONS = (blas.Einsum, blas.MatMul, blas.Gemm, blas.BatchedMatMul, blas.Gemv, blas.Dot, TensorDot)


def contraction_nodes(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CONTRACTIONS)]


def assert_one_lifted_einsum(sdfg: dace.SDFG, einsum_str: str, beta: float) -> None:
    found = contraction_nodes(sdfg)
    assert [type(n) for n in found] == [blas.Einsum], found
    assert found[0].einsum_str == einsum_str
    assert float(found[0].beta) == beta
    assert not any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))
    assert not any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


def assert_not_lifted(sdfg: dace.SDFG) -> None:
    assert not any(isinstance(n, blas.Einsum) for n in contraction_nodes(sdfg))


def canonical(program) -> dace.SDFG:
    return canonicalize(program.to_sdfg(simplify=False))


def random(*shape: int) -> np.ndarray:
    return np.random.default_rng(len(shape) * 31 + sum(shape)).random(shape)


def test_map_of_gemms_with_a_shared_operand_becomes_one_contraction():
    @dace.program
    def n2c_map_gemm(A: dace.float64[NB, N, K], B: dace.float64[K, M], C: dace.float64[NB, N, M]):
        for b in dace.map[0:NB]:
            C[b] = A[b] @ B

    sdfg = canonical(n2c_map_gemm)
    assert_one_lifted_einsum(sdfg, "aik,kj->aij", 0.0)
    A, B, C = random(5, 6, 7), random(7, 8), np.zeros((5, 6, 8))
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, A @ B)


def test_two_level_map_nest_of_gemms_becomes_one_contraction():
    @dace.program
    def n2c_map2_gemm(A: dace.float64[NB, M, N, K], B: dace.float64[K, M], C: dace.float64[NB, M, N, M]):
        for b, c in dace.map[0:NB, 0:M]:
            C[b, c] = A[b, c] @ B

    sdfg = canonical(n2c_map2_gemm)
    assert_one_lifted_einsum(sdfg, "abik,kj->abij", 0.0)
    A, B, C = random(5, 8, 6, 7), random(7, 8), np.zeros((5, 8, 6, 8))
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, A @ B)


def test_sequential_accumulating_loop_contracts_its_variable():
    @dace.program
    def n2c_loop_acc(A: dace.float64[NB, N, K], B: dace.float64[NB, K, M], C: dace.float64[N, M]):
        for k in range(NB):
            C[:] += A[k] @ B[k]

    sdfg = canonical(n2c_loop_acc)
    assert_one_lifted_einsum(sdfg, "aik,akj->ij", 1.0)
    A, B, C = random(5, 6, 7), random(5, 7, 8), random(6, 8)
    expected = C + np.einsum("bik,bkj->ij", A, B)
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, expected)


def test_loop_of_gemms_after_a_zero_init_takes_beta_zero():
    @dace.program
    def n2c_loop_init(A: dace.float64[NB, N, K], B: dace.float64[NB, K, M], C: dace.float64[N, M]):
        C[:] = 0
        for k in range(NB):
            C += A[k] @ B[k]

    sdfg = canonical(n2c_loop_init)
    assert_one_lifted_einsum(sdfg, "aik,akj->ij", 0.0)
    A, B, C = random(5, 6, 7), random(5, 7, 8), np.full((6, 8), np.nan)
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, np.einsum("bik,bkj->ij", A, B))


def test_loop_over_map_of_einsums_becomes_one_contraction():
    @dace.program
    def n2c_loop_map_einsum(A: dace.float64[K, NB, N, K], B: dace.float64[K, K, M], C: dace.float64[NB, N, M]):
        for k in range(K):
            for b in dace.map[0:NB]:
                C[b] += np.einsum("ij,jk->ik", A[k, b], B[k])

    sdfg = canonical(n2c_loop_map_einsum)
    assert_one_lifted_einsum(sdfg, "abik,akj->bij", 1.0)
    A, B, C = random(7, 5, 6, 7), random(7, 7, 8), random(5, 6, 8)
    expected = C + np.einsum("kbij,kjl->bil", A, B)
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, expected)


def test_strided_index_is_not_lifted():
    @dace.program
    def n2c_strided(A: dace.float64[2 * NB, N, K], B: dace.float64[K, M], C: dace.float64[NB, N, M]):
        for b in dace.map[0:NB]:
            C[b] = A[2 * b] @ B

    sdfg = canonical(n2c_strided)
    assert_not_lifted(sdfg)
    A, B, C = random(10, 6, 7), random(7, 8), np.zeros((5, 6, 8))
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, A[::2] @ B)


def test_second_tasklet_in_the_body_is_not_lifted():
    @dace.program
    def n2c_second_tasklet(
        A: dace.float64[NB, N, K], B: dace.float64[K, M], C: dace.float64[NB, N, M], D: dace.float64[NB]
    ):
        for b in dace.map[0:NB]:
            C[b] = A[b] @ B
            D[b] = A[b, 0, 0] * 2.0

    sdfg = canonical(n2c_second_tasklet)
    assert_not_lifted(sdfg)
    A, B, C, D = random(5, 6, 7), random(7, 8), np.zeros((5, 6, 8)), np.zeros(5)
    sdfg(A=A, B=B, C=C, D=D, **SIZES)
    assert np.allclose(C, A @ B) and np.allclose(D, 2 * A[:, 0, 0])


def test_triangular_nest_lifts_only_its_rectangular_inner_loop():
    @dace.program
    def n2c_triangular(A: dace.float64[NB, NB, N, K], B: dace.float64[K, M], C: dace.float64[NB, NB, N, M]):
        for i in range(NB):
            for j in range(i + 1):
                C[i, j] = A[i, j] @ B

    sdfg = canonical(n2c_triangular)
    found = contraction_nodes(sdfg)
    assert [type(n) for n in found] == [blas.Einsum] and found[0].einsum_str == "aik,kj->aij"
    assert any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True)) or any(
        isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()
    )
    A, B, C = random(5, 5, 6, 7), random(7, 8), np.zeros((5, 5, 6, 8))
    expected = np.tril(np.ones((5, 5)))[:, :, None, None] * (A @ B)
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, expected)


def test_overlapping_outputs_are_not_lifted():
    @dace.program
    def n2c_overlap(A: dace.float64[NB, N, K], B: dace.float64[K, M], C: dace.float64[NB, N, M]):
        for b in range(NB):
            C[b // 2] = A[b] @ B

    sdfg = canonical(n2c_overlap)
    assert_not_lifted(sdfg)
    A, B, C = random(5, 6, 7), random(7, 8), np.zeros((5, 6, 8))
    expected = C.copy()
    for b in range(5):
        expected[b // 2] = A[b] @ B
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, expected)


def test_operand_written_in_the_nest_is_not_lifted():
    @dace.program
    def n2c_operand_written(A: dace.float64[NB, N, N], B: dace.float64[N, N]):
        for b in range(NB - 1):
            A[b + 1] = A[b] @ B

    sdfg = canonical(n2c_operand_written)
    assert_not_lifted(sdfg)
    A, B = random(5, 6, 6), random(6, 6)
    expected = A.copy()
    for b in range(4):
        expected[b + 1] = expected[b] @ B
    sdfg(A=A, B=B, **SIZES)
    assert np.allclose(A, expected)


def test_non_accumulating_loop_whose_variable_is_not_in_the_output_is_not_lifted():
    @dace.program
    def n2c_last_writer(A: dace.float64[NB, N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
        for b in range(NB):
            C[:] = A[b] @ B

    sdfg = canonical(n2c_last_writer)
    assert_not_lifted(sdfg)
    A, B, C = random(5, 6, 7), random(7, 8), np.zeros((6, 8))
    sdfg(A=A, B=B, C=C, **SIZES)
    assert np.allclose(C, A[4] @ B)


@pytest.mark.gpu
def test_gpu_map_of_gemms_becomes_a_vendor_tensor_contraction():
    vendor = "hipTENSOR" if dace.Config.get("compiler", "cuda", "backend") == "hip" else "cuTENSOR"

    @dace.program
    def n2c_gpu_map_gemm(A: dace.float32[NB, N, K], B: dace.float32[K, M], C: dace.float32[NB, N, M]):
        for b in dace.map[0:NB]:
            C[b] = A[b] @ B

    sdfg = canonicalize(n2c_gpu_map_gemm.to_sdfg(simplify=False), target="gpu")
    assert_one_lifted_einsum(sdfg, "aik,kj->aij", 0.0)
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    assert [n.implementation for n in contraction_nodes(sdfg) if isinstance(n, TensorDot)] == [vendor]
    import cupy  # Only present on GPU runners.

    A, B = random(5, 6, 7).astype(np.float32), random(7, 8).astype(np.float32)
    C = cupy.zeros((5, 6, 8), dtype=cupy.float32)
    sdfg(A=cupy.asarray(A), B=cupy.asarray(B), C=C, **SIZES)
    assert np.allclose(cupy.asnumpy(C), A @ B, rtol=1e-5)


if __name__ == "__main__":
    test_map_of_gemms_with_a_shared_operand_becomes_one_contraction()
    test_two_level_map_nest_of_gemms_becomes_one_contraction()
    test_sequential_accumulating_loop_contracts_its_variable()
    test_loop_of_gemms_after_a_zero_init_takes_beta_zero()
    test_loop_over_map_of_einsums_becomes_one_contraction()
    test_strided_index_is_not_lifted()
    test_second_tasklet_in_the_body_is_not_lifted()
    test_triangular_nest_is_not_lifted_as_a_whole()
    test_overlapping_outputs_are_not_lifted()
    test_operand_written_in_the_nest_is_not_lifted()
    test_non_accumulating_loop_whose_variable_is_not_in_the_output_is_not_lifted()
    test_gpu_map_of_gemms_becomes_a_vendor_tensor_contraction()
