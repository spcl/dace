# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map whose body is one matrix-vector product becomes one ``BatchedMatMul`` over stacks of vectors."""

import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.libraries.blas.nodes.batched_matmul import BATCH_INDEX, BatchedMatMul
from dace.libraries.blas.nodes.gemv import Gemv
from dace.sdfg import nodes
from dace.transformation.passes.offloading.batch_map_products import BatchMapProducts

NB, P, M, K = (dace.symbol(name) for name in ("NB", "P", "M", "K"))

#: name: (shape of A, A's subset in iteration b, x shared, the reference from A, x)
GEMVS = {
    "strided": ([NB, M, K], "b, 0:M, 0:K", False, lambda a, x: np.einsum("bmk,bk->bm", a, x)),
    "shared_vector": ([NB, M, K], "b, 0:M, 0:K", True, lambda a, x: np.einsum("bmk,k->bm", a, x)),
    "gathered": ([P, M, K], "NB - 1 - b, 0:M, 0:K", False, lambda a, x: np.einsum("bmk,bk->bm", a[::-1], x)),
}


def map_of_gemv(name: str, case: str, storage: dace.StorageType) -> dace.SDFG:
    """``for b in map: y[b] = A[...] @ x[b]`` (or ``@ x``), its body one ``Gemv``."""
    shape_a, subset_a, shared_x, _ = GEMVS[case]
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", shape_a, dace.float64, storage=storage)
    sdfg.add_array("x", [K] if shared_x else [NB, K], dace.float64, storage=storage)
    sdfg.add_array("y", [NB, M], dace.float64, storage=storage)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("products", {"b": "0:NB"})
    gemv = Gemv("gemv")
    state.add_node(gemv)
    state.add_memlet_path(state.add_read("A"), entry, gemv, dst_conn="_A", memlet=dace.Memlet(f"A[{subset_a}]"))
    state.add_memlet_path(
        state.add_read("x"), entry, gemv, dst_conn="_x", memlet=dace.Memlet("x[0:K]" if shared_x else "x[b, 0:K]")
    )
    state.add_memlet_path(gemv, exit_node, state.add_write("y"), src_conn="_y", memlet=dace.Memlet("y[b, 0:M]"))
    return sdfg


def operands(case: str) -> tuple[dict, np.ndarray]:
    shape_a, _, shared_x, reference = GEMVS[case]
    rng = np.random.default_rng(5)
    a = rng.random((7, 4, 3))
    x = rng.random(3 if shared_x else (7, 3))
    sizes = {"NB": 7, "M": 4, "K": 3} | ({"P": 7} if shape_a[0] is P else {})
    return {"A": a, "x": x, "y": np.zeros((7, 4))} | sizes, reference(a, x)


def batched(sdfg: dace.SDFG) -> BatchedMatMul:
    """The one product left, after the pass made it from the map."""
    assert BatchMapProducts().apply_pass(sdfg, {}) == 1
    products = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (Gemv, BatchedMatMul))]
    assert len(products) == 1 and isinstance(products[0], BatchedMatMul)
    return products[0]


@pytest.mark.parametrize("case", list(GEMVS))
def test_a_map_of_gemv_is_one_batched_product(case: str):
    """Strided stacks need no index; a stack read at another expression of ``b`` is gathered through one."""
    sdfg = map_of_gemv(f"map_of_gemv_{case}", case, dace.StorageType.Default)
    product = batched(sdfg)
    assert (BATCH_INDEX["_a"] in product.in_connectors) == (case == "gathered")
    assert not any(isinstance(n, nodes.MapEntry) and n.map.label == "products" for n, _ in sdfg.all_nodes_recursive())
    product.implementation = "pure"
    args, ref = operands(case)
    sdfg(**args)
    assert np.allclose(args["y"], ref)


@pytest.mark.gpu
@pytest.mark.parametrize("case", list(GEMVS))
def test_a_map_of_gemv_runs_as_one_vendor_call(case: str):
    """On the device, over a symbolic batch: one ``gemmStridedBatched``, or one ``gemmBatched`` when gathered."""
    import cupy as cp

    sdfg = map_of_gemv(f"map_of_gemv_gpu_{case}", case, dace.StorageType.GPU_Global)
    product = batched(sdfg)
    product.implementation = "hipBLAS" if common.get_gpu_backend() == "hip" else "cuBLAS"
    sdfg.apply_gpu_transformations()
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert ("gemmBatched" if case == "gathered" else "gemmStridedBatched") in code
    args, ref = operands(case)
    device = {name: cp.asarray(v) if isinstance(v, np.ndarray) else v for name, v in args.items()}
    sdfg(**device)
    assert np.allclose(cp.asnumpy(device["y"]), ref)


if __name__ == "__main__":
    for case in GEMVS:
        test_a_map_of_gemv_is_one_batched_product(case)
        test_a_map_of_gemv_runs_as_one_vendor_call(case)
