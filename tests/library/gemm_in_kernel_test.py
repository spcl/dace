# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``Gemm`` inside a GPU kernel runs on the kernel's thread block: shared-memory tiles, or one block reduction per
element of C when K is long and C small."""

import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm
from dace.libraries.standard.block_reduce import in_kernel_block_implementation

BATCH = 3


def gemm_per_block(name: str, m: int, n: int, k: int, trans_b: bool, alpha, beta, dtype=dace.float64):
    """``map b (GPU_Device) { C[b] = alpha * A[b] @ op(B[b]) + beta * C[b] }``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [BATCH, m, k], dtype, dace.StorageType.GPU_Global)
    sdfg.add_array("B", [BATCH, n, k] if trans_b else [BATCH, k, n], dtype, dace.StorageType.GPU_Global)
    sdfg.add_array("C", [BATCH, m, n], dtype, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, leave = state.add_map("batch", {"b": f"0:{BATCH}"}, schedule=dace.ScheduleType.GPU_Device)
    node = Gemm("gemm", transB=trans_b, alpha=alpha, beta=beta)
    b_subset = f"B[b, 0:{n}, 0:{k}]" if trans_b else f"B[b, 0:{k}, 0:{n}]"
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="_a", memlet=dace.Memlet(f"A[b, 0:{m}, 0:{k}]"))
    state.add_memlet_path(state.add_read("B"), entry, node, dst_conn="_b", memlet=dace.Memlet(b_subset))
    if "_c" in node.in_connectors:
        state.add_memlet_path(
            state.add_read("C"), entry, node, dst_conn="_c", memlet=dace.Memlet(f"C[b, 0:{m}, 0:{n}]")
        )
    state.add_memlet_path(node, leave, state.add_write("C"), src_conn="_c", memlet=dace.Memlet(f"C[b, 0:{m}, 0:{n}]"))
    return sdfg, state, node


def test_a_gemm_in_a_kernel_lowers_to_tiles_or_per_element_reductions_on_the_block():
    sdfg, state, node = gemm_per_block("gemm_block_lowering", 40, 56, 24, False, 1.0, 0.0)

    assert in_kernel_block_implementation(node, state) == "CUDA (block strided)"
    node.implementation = "CUDA (block strided)"
    sdfg.expand_library_nodes()
    sdfg.validate()
    lanes = [
        n.map
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_ThreadBlock
    ]
    assert [lane.range.size()[0] for lane in lanes] == [256]
    (code,) = [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert "__gsa" in code and "BlockReduce" in code


@pytest.mark.gpu
@pytest.mark.parametrize(
    "m, n, k, trans_b, alpha, beta, dtype",
    [
        pytest.param(40, 56, 24, True, 2.0, 0.5, dace.float64, id="tiles"),
        pytest.param(2, 3, 3000, False, 1.0, 0.0, dace.float64, id="per_element_reductions"),
        pytest.param(33, 17, 20, False, 1.0, 0.0, dace.complex128, id="complex_tiles"),
    ],
)
def test_a_gemm_on_the_block_matches_numpy(m, n, k, trans_b, alpha, beta, dtype):
    cupy = pytest.importorskip("cupy")
    sdfg, _, node = gemm_per_block(
        f"gemm_block_run_{m}_{n}_{k}_{dtype.to_string()}", m, n, k, trans_b, alpha, beta, dtype
    )
    node.implementation = "CUDA (block strided)"
    rng = np.random.default_rng(3)
    np_type = dtype.type

    def rand(*shape):
        values = rng.random(shape)
        return (values + 1j * rng.random(shape) if np_type == np.complex128 else values).astype(np_type)

    A = rand(BATCH, m, k)
    B = rand(BATCH, n, k) if trans_b else rand(BATCH, k, n)
    C = rand(BATCH, m, n)
    expected = alpha * A @ (B.transpose(0, 2, 1) if trans_b else B) + beta * C
    out = cupy.asarray(C)

    sdfg(A=cupy.asarray(A), B=cupy.asarray(B), C=out)

    np.testing.assert_allclose(cupy.asnumpy(out), expected, rtol=1e-12)


if __name__ == "__main__":
    test_a_gemm_in_a_kernel_lowers_to_tiles_or_per_element_reductions_on_the_block()
    test_a_gemm_on_the_block_matches_numpy(40, 56, 24, True, 2.0, 0.5, dace.float64)
    test_a_gemm_on_the_block_matches_numpy(2, 3, 3000, False, 1.0, 0.0, dace.float64)
    test_a_gemm_on_the_block_matches_numpy(33, 17, 20, False, 1.0, 0.0, dace.complex128)
