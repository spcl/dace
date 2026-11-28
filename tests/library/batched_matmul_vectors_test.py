# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BatchedMatMul`` over stacks of vectors: a ``[B, M]`` output is a stack of matrix-vector products, a ``[B]`` one a
stack of dots; each lowers as the batched matrix product it is (``N = 1``, or ``M = N = 1``)."""

import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.libraries.blas.nodes.batched_matmul import BATCH_INDEX, BatchedMatMul

NB, P, M, K = (dace.symbol(name) for name in ("NB", "P", "M", "K"))

#: name: (_a shape, _b shape, _c shape, einsum of the reference, whether _a is gathered from a pool of P)
CASES = {
    "gemv": ([NB, M, K], [NB, K], [NB, M], "bmk,bk->bm", False),
    "gemv_shared_matrix": ([M, K], [NB, K], [NB, M], "mk,bk->bm", False),
    "gemv_shared_vector": ([NB, M, K], [K], [NB, M], "bmk,k->bm", False),
    "dot": ([NB, K], [NB, K], [NB], "bk,bk->b", False),
    "gemv_gathered": ([P, M, K], [NB, K], [NB, M], "bmk,bk->bm", True),
}


def vector_product_sdfg(name: str, case: str, implementation: str, storage: dace.StorageType) -> dace.SDFG:
    shape_a, shape_b, shape_c, _, gathered = CASES[case]
    sdfg = dace.SDFG(name)
    for data, shape in (("A", shape_a), ("X", shape_b), ("Y", shape_c)):
        sdfg.add_array(data, shape, dace.float64, storage=storage)
    state = sdfg.add_state()
    node = BatchedMatMul("vectors")
    node.implementation = implementation
    state.add_node(node)
    conns = {"_a": "A", "_b": "X"}
    if gathered:
        sdfg.add_array("ia", [NB], dace.int64, storage=storage)
        node.add_in_connector(BATCH_INDEX["_a"])
        conns[BATCH_INDEX["_a"]] = "ia"
    for conn, data in conns.items():
        state.add_edge(state.add_read(data), None, node, conn, dace.Memlet.from_array(data, sdfg.arrays[data]))
    state.add_edge(node, "_c", state.add_write("Y"), None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))
    return sdfg


def operands(case: str) -> tuple[dict, np.ndarray]:
    """Arguments with a batch of 9 (the gathered pool holds 4), and the numpy result."""
    shape_a, shape_b, shape_c, subscripts, gathered = CASES[case]
    sizes = {"NB": 9, "P": 4, "M": 5, "K": 3}
    rng = np.random.default_rng(11)
    concrete = [[sizes[str(d)] for d in shape] for shape in (shape_a, shape_b, shape_c)]
    args = {"A": rng.random(concrete[0]), "X": rng.random(concrete[1]), "Y": np.zeros(concrete[2])}
    a = args["A"]
    if gathered:
        args["ia"] = np.array([3, 0, 3, 2, 1, 1, 0, 2, 3], dtype=np.int64)
        a = a[args["ia"]]
    return args | {"NB": 9, "M": 5, "K": 3} | ({"P": 4} if gathered else {}), np.einsum(subscripts, a, args["X"])


@pytest.mark.parametrize("implementation", ["pure", "OpenBLAS"])
@pytest.mark.parametrize("case", list(CASES))
def test_vector_products_on_the_cpu(case: str, implementation: str):
    """The pure expansion, which the CPU BLAS lowerings fall back to for vector operands."""
    sdfg = vector_product_sdfg(f"vectors_cpu_{case}_{implementation}", case, implementation, dace.StorageType.Default)
    args, ref = operands(case)
    sdfg(**args)
    assert np.allclose(args["Y"], ref)


@pytest.mark.gpu
@pytest.mark.parametrize("case", list(CASES))
def test_vector_products_are_one_batched_vendor_call(case: str):
    """A strided stack is one ``gemmStridedBatched``; a gathered one is one ``gemmBatched`` over pointer arrays."""
    import cupy as cp

    implementation = "hipBLAS" if common.get_gpu_backend() == "hip" else "cuBLAS"
    sdfg = vector_product_sdfg(f"vectors_gpu_{case}", case, implementation, dace.StorageType.GPU_Global)
    sdfg.expand_library_nodes()
    code = sdfg.generate_code()[0].clean_code
    assert ("gemmBatched" if CASES[case][4] else "gemmStridedBatched") in code
    args, ref = operands(case)
    device = {name: cp.asarray(value) if isinstance(value, np.ndarray) else value for name, value in args.items()}
    sdfg(**device)
    assert np.allclose(cp.asnumpy(device["Y"]), ref)


if __name__ == "__main__":
    for case in CASES:
        test_vector_products_on_the_cpu(case, "pure")
        test_vector_products_on_the_cpu(case, "OpenBLAS")
        test_vector_products_are_one_batched_vendor_call(case)
