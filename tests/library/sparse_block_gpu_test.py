# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CSRMV and CSRMM inside a GPU kernel lower to one thread block per product (CSR-scalar, outputs strided over the
lanes): a batch of independent sparse products, one per kernel iteration."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.sparse.nodes.csrmm import CSRMM
from dace.libraries.sparse.nodes.csrmv import CSRMV
from dace.libraries.standard.block_reduce import gpu_block_implementation
from dace.sdfg import nodes
from dace.transformation.passes.offloading.taskloop import is_kernel_refused_libnode

BATCH = dace.symbol("BATCH", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)
NC = dace.symbol("NC", dtype=dace.int64)
NNZ = dace.symbol("NNZ", dtype=dace.int64)

BLOCK = "CUDA (block strided)"
GLOBAL = dtypes.StorageType.GPU_Global


def batched_kernel(node: nodes.LibraryNode, dense: list, out: list, dense_subset: str, out_subset: str):
    """A ``GPU_Device`` map over ``p`` running ``node`` on the ``p``-th CSR matrix and dense operand."""
    sdfg = dace.SDFG(f"block_{node.label}")
    sdfg.add_array("rows", [BATCH, M + 1], dace.int32, storage=GLOBAL)
    sdfg.add_array("cols", [BATCH, NNZ], dace.int32, storage=GLOBAL)
    sdfg.add_array("vals", [BATCH, NNZ], dace.float64, storage=GLOBAL)
    sdfg.add_array("dense", dense, dace.float64, storage=GLOBAL)
    sdfg.add_array("out", out, dace.float64, storage=GLOBAL)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("problems", {"p": "0:BATCH"}, schedule=dtypes.ScheduleType.GPU_Device)
    reads = {"_a_rows": "rows[p, 0:M + 1]", "_a_cols": "cols[p, 0:NNZ]", "_a_vals": "vals[p, 0:NNZ]"}
    reads["_b"] = dense_subset
    if "_cin" in node.in_connectors:
        sdfg.add_array("initial", out, dace.float64, storage=GLOBAL)
        reads["_cin"] = out_subset.replace("out", "initial", 1)
    for conn, subset in reads.items():
        name = subset.split("[")[0]
        state.add_memlet_path(state.add_read(name), entry, node, dst_conn=conn, memlet=dace.Memlet(subset))
    state.add_memlet_path(node, exit_node, state.add_write("out"), src_conn="_c", memlet=dace.Memlet(out_subset))
    return sdfg, state


def spmv_kernel(beta: float = 0.0):
    node = CSRMV("spmv", alpha=2.0, beta=beta)
    return (*batched_kernel(node, [BATCH, K], [BATCH, M], "dense[p, 0:K]", "out[p, 0:M]"), node)


def spmm_kernel(trans_b: bool = False):
    node = CSRMM("spmm", transB=trans_b, alpha=1.5)
    dense = [BATCH, NC, K] if trans_b else [BATCH, K, NC]
    subset = "dense[p, 0:NC, 0:K]" if trans_b else "dense[p, 0:K, 0:NC]"
    return (*batched_kernel(node, dense, [BATCH, M, NC], subset, "out[p, 0:M, 0:NC]"), node)


def block_tasklet(sdfg: dace.SDFG) -> nodes.Tasklet:
    """Expand and return the product's one tasklet, checking it sits in a 256-lane map inside the device map."""
    sdfg.expand_library_nodes()
    lanes = [
        n
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert [lane.map.range.size()[0] for lane in lanes] == [256]
    (tasklet,) = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    return tasklet


@pytest.mark.parametrize("build", [spmv_kernel, spmm_kernel])
def test_a_csr_product_in_a_kernel_takes_the_block_lowering(build):
    sdfg, state, node = build()
    assert gpu_block_implementation(node, state, sdfg) == BLOCK
    assert not is_kernel_refused_libnode(node, state)


def test_the_block_spmv_strides_rows_over_the_lanes():
    sdfg, state, node = spmv_kernel(beta=0.5)
    node.implementation = BLOCK
    code = block_tasklet(sdfg).code.as_string
    assert "__t += 256" in code and "__syncthreads();" in code
    assert "__a_vals[__j * 1] * __b[((long long)__a_cols[__j * 1]) * (1)]" in code
    assert "double(0.5) * __cin[" in code


def test_the_block_spmm_reads_a_transposed_dense_operand_by_its_rows():
    sdfg, state, node = spmm_kernel(trans_b=True)
    node.implementation = BLOCK
    code = block_tasklet(sdfg).code.as_string
    assert "__b[(__k) * (K) + ((long long)__a_cols[__j * 1]) * (1)]" in code


def random_csr(rng: np.random.Generator, problems: int, m: int, k: int, nnz: int):
    """``problems`` CSR matrices of ``m x k`` with exactly ``nnz`` entries each, and their dense forms."""
    import scipy.sparse

    mats = [
        scipy.sparse.random(m, k, density=nnz / (m * k), format="csr", random_state=rng) for index in range(problems)
    ]
    assert all(mat.nnz == nnz for mat in mats)
    rows = np.stack([mat.indptr for mat in mats]).astype(np.int32)
    cols = np.stack([mat.indices for mat in mats]).astype(np.int32)
    vals = np.stack([mat.data for mat in mats])
    return rows, cols, vals, np.stack([mat.toarray() for mat in mats])


@pytest.mark.gpu
@pytest.mark.parametrize("m, k", [(64, 32), (300, 7)])
def test_batched_block_spmv_matches_numpy(m: int, k: int):
    import cupy

    sdfg, state, node = spmv_kernel(beta=0.5)
    node.implementation = BLOCK
    rng = np.random.default_rng(m)
    problems, nnz = 5, 3 * k
    rows, cols, vals, dense_a = random_csr(rng, problems, m, k, nnz)
    b, initial = rng.uniform(-1, 1, (problems, k)), rng.uniform(-1, 1, (problems, m))
    out = cupy.zeros((problems, m))
    gpu = {"rows": rows, "cols": cols, "vals": vals, "dense": b, "initial": initial}
    sdfg(**{name: cupy.asarray(value) for name, value in gpu.items()}, out=out, BATCH=problems, M=m, K=k, NNZ=nnz)
    expected = 2.0 * np.einsum("pmk,pk->pm", dense_a, b) + 0.5 * initial
    np.testing.assert_allclose(out.get(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("m, k, nc", [(64, 32, 16), (37, 9, 5)])
def test_batched_block_spmm_matches_numpy(trans_b: bool, m: int, k: int, nc: int):
    import cupy

    sdfg, state, node = spmm_kernel(trans_b)
    node.implementation = BLOCK
    rng = np.random.default_rng(m + nc)
    problems, nnz = 4, 2 * k
    rows, cols, vals, dense_a = random_csr(rng, problems, m, k, nnz)
    b = rng.uniform(-1, 1, (problems, nc, k) if trans_b else (problems, k, nc))
    out = cupy.zeros((problems, m, nc))
    gpu = {"rows": rows, "cols": cols, "vals": vals, "dense": b}
    sdfg(**{n: cupy.asarray(v) for n, v in gpu.items()}, out=out, BATCH=problems, M=m, K=k, NC=nc, NNZ=nnz)
    product = np.einsum("pmk,pnk->pmn", dense_a, b) if trans_b else np.einsum("pmk,pkn->pmn", dense_a, b)
    np.testing.assert_allclose(out.get(), 1.5 * product, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    for build in (spmv_kernel, spmm_kernel):
        test_a_csr_product_in_a_kernel_takes_the_block_lowering(build)
    test_the_block_spmv_strides_rows_over_the_lanes()
    test_the_block_spmm_reads_a_transposed_dense_operand_by_its_rows()
    for shape in ((64, 32), (300, 7)):
        test_batched_block_spmv_matches_numpy(*shape)
    for transposed in (False, True):
        for shape in ((64, 32, 16), (37, 9, 5)):
            test_batched_block_spmm_matches_numpy(transposed, *shape)
