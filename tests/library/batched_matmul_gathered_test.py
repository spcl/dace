# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul

N = dace.symbol("N")
PA, PB, M, K, NC = (dace.symbol(name) for name in ("PA", "PB", "M", "K", "NC"))


def gathered_product_sdfg(name: str, implementation: str, storage: dace.StorageType, shared_b: bool) -> dace.SDFG:
    """``C[p] = A[ia[p]] @ B[ib[p]]``, or ``A[ia[p]] @ B`` with ``shared_b``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [PA, M, K], dace.float64, storage=storage)
    sdfg.add_array("B", [K, NC] if shared_b else [PB, K, NC], dace.float64, storage=storage)
    sdfg.add_array("C", [N, M, NC], dace.float64, storage=storage)
    sdfg.add_array("ia", [N], dace.int64, storage=storage)
    if not shared_b:
        sdfg.add_array("ib", [N], dace.int64, storage=storage)
    state = sdfg.add_state()
    node = BatchedMatMul("gathered")
    node.implementation = implementation
    state.add_node(node)
    conns = {"_a": "A", "_b": "B", "_a_batch_index": "ia"} | ({} if shared_b else {"_b_batch_index": "ib"})
    for conn, data in conns.items():
        node.add_in_connector(conn)
        state.add_edge(state.add_read(data), None, node, conn, dace.Memlet.from_array(data, sdfg.arrays[data]))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def operands(shared_b: bool) -> dict:
    rng = np.random.default_rng(7)
    a, b = rng.random((6, 5, 3)), rng.random((3, 7) if shared_b else (4, 3, 7))
    ia = np.array([5, 0, 5, 2, 1, 3, 0, 4, 5], dtype=np.int64)
    args = {"A": a, "B": b, "ia": ia, "C": np.zeros((ia.size, 5, 7))}
    if not shared_b:
        args["ib"] = np.array([3, 3, 0, 1, 2, 0, 1, 2, 3], dtype=np.int64)
    return args


def reference(args: dict) -> np.ndarray:
    return args["A"][args["ia"]] @ (args["B"][args["ib"]] if "ib" in args else args["B"])


@pytest.mark.parametrize("shared_b", [False, True])
def test_gathered_batched_matmul_pure(shared_b: bool):
    sdfg = gathered_product_sdfg(f"gathered_pure_{shared_b}", "pure", dace.StorageType.Default, shared_b)
    args = operands(shared_b)
    sdfg(**args)
    assert np.allclose(args["C"], reference(args))


@pytest.mark.gpu
@pytest.mark.parametrize("shared_b", [False, True])
def test_gathered_batched_matmul_gpu_blas(shared_b: bool):
    """The vendor ``<T>gemmBatched`` over pointer arrays the device fills from the indices."""
    import cupy as cp

    implementation = "hipBLAS" if common.get_gpu_backend() == "hip" else "cuBLAS"
    sdfg = gathered_product_sdfg(f"gathered_gpu_{shared_b}", implementation, dace.StorageType.GPU_Global, shared_b)
    sdfg.expand_library_nodes()
    code = sdfg.generate_code()[0].clean_code
    assert "gemmBatched" in code and "StridedBatched" not in code
    args = operands(shared_b)
    ref = reference(args)
    device = {name: cp.asarray(value) for name, value in args.items()}
    sdfg(**device)
    assert np.allclose(cp.asnumpy(device["C"]), ref)


if __name__ == "__main__":
    test_gathered_batched_matmul_pure(False)
    test_gathered_batched_matmul_pure(True)
    test_gathered_batched_matmul_gpu_blas(False)
    test_gathered_batched_matmul_gpu_blas(True)
