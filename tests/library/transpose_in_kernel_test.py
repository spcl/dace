# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``Transpose`` or ``TensorTranspose`` inside a GPU kernel runs on the kernel's thread block."""

import numpy as np
import pytest

import dace
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.libraries.standard.block_reduce import in_kernel_block_implementation


def transpose_per_block(
    node: dace.nodes.LibraryNode, name: str, inp: str = "_inp", out: str = "_out"
) -> tuple[dace.SDFG, dace.nodes.LibraryNode]:
    """``map k (GPU_Device) { B[k] = A[k]^T }`` over 4 slices of 300 x 200 through ``node``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [4, 300, 200], dace.float64, dace.StorageType.GPU_Global)
    sdfg.add_array("B", [4, 200, 300], dace.float64, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, leave = state.add_map("slices", {"k": "0:4"}, schedule=dace.ScheduleType.GPU_Device)
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn=inp, memlet=dace.Memlet("A[k, 0:300, 0:200]"))
    state.add_memlet_path(node, leave, state.add_write("B"), src_conn=out, memlet=dace.Memlet("B[k, 0:200, 0:300]"))
    return sdfg, node


def plain_transpose(name: str):
    return transpose_per_block(Transpose("transpose", dtype=dace.float64), name)


def tensor_transpose(name: str):
    return transpose_per_block(TensorTranspose("ttranspose", axes=[0, 2, 1]), name, "_inp_tensor", "_out_tensor")


@pytest.mark.parametrize("build", [plain_transpose, tensor_transpose])
def test_a_transpose_in_a_kernel_lowers_to_the_block(build):
    sdfg, node = build(f"{build.__name__}_lowering")
    state = sdfg.start_state

    assert in_kernel_block_implementation(node, state) == "CUDA (block)"
    node.implementation = "CUDA (block)"
    sdfg.expand_library_nodes()
    sdfg.validate()
    lanes = [
        n.map
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_ThreadBlock
    ]
    assert len(lanes) == 1 and lanes[0].gpu_force_syncthreads


@pytest.mark.gpu
@pytest.mark.parametrize("build", [plain_transpose, tensor_transpose])
def test_a_transpose_on_the_block_matches_numpy(build):
    cupy = pytest.importorskip("cupy")
    sdfg, node = build(f"{build.__name__}_run")
    node.implementation = "CUDA (block)"
    A = np.random.default_rng(2).random((4, 300, 200))
    B = cupy.zeros((4, 200, 300))

    sdfg(A=cupy.asarray(A), B=B)

    np.testing.assert_array_equal(cupy.asnumpy(B), A.transpose(0, 2, 1))
