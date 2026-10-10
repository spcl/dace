# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on the GPU runs a kernel one block per outer iteration and its inner work across the lanes.

An inner reduction folds with ``gpucub::BlockReduce``, an in-kernel ``Dot`` / ``Gemm`` / ``Reduce`` takes its
block collective, and a host map that only launched device work becomes the kernel. The structural checks
need no GPU; the numeric ones run on the device, at sizes that leave the last block partial.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.gpu_specialization.block_tile_kernels import BlockTileKernels

M, N, K, NB, NNZ = (dace.symbol(s) for s in ("M", "N", "K", "NB", "NNZ"))
SIZES = [(1, 1, 1, 1), (37, 300, 5, 3), (257, 513, 129, 2)]


@dace.program
def matvec(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[M]):
    for i in range(M):
        y[i] = 0.0
        for j in range(N):
            y[i] += A[i, j] * x[j]


@dace.program
def matvec_plus(A: dace.float64[M, N], x: dace.float64[N], b: dace.float64[M], y: dace.float64[M]):
    for i in range(M):
        y[i] = b[i]
        for j in range(N):
            y[i] += A[i, j] * x[j]


@dace.program
def gemm_loops(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    for i in range(M):
        for j in range(N):
            C[i, j] = 0.0
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


@dace.program
def rowsum(A: dace.float64[M, N], out: dace.float64[M]):
    for i in range(M):
        s = 0.0
        for j in range(N):
            s += A[i, j]
        out[i] = s


@dace.program
def batched_dot(X: dace.float64[NB, N], Y: dace.float64[NB, N], out: dace.float64[NB]):
    for b in dace.map[0:NB]:
        out[b] = np.dot(X[b], Y[b])


@dace.program
def batched_gemm(A: dace.float64[NB, M, K], B: dace.float64[NB, K, N], C: dace.float64[NB, M, N]):
    for b in dace.map[0:NB]:
        C[b] = A[b] @ B[b]


@dace.program
def subtract_rowsum(A: dace.float64[M, N], b: dace.float64[M]):
    for i in dace.map[0:M]:
        s = 0.0
        for j in dace.map[0:N]:
            s += A[i, j]
        b[i] -= s


@dace.program
def accumulate_outside(A: dace.float64[M, N], total: dace.float64[1]):
    for i in dace.map[0:M]:
        s = 0.0
        for j in dace.map[0:N]:
            s += A[i, j]
        total[0] += s


@dace.program
def spmv(
    A_data: dace.float64[NNZ],
    A_indices: dace.uint32[NNZ],
    A_indptr: dace.uint32[M + 1],
    x: dace.float64[N],
    y: dace.float64[M],
):
    for i in range(M):
        start = dace.define_local_scalar(dace.uint32)
        stop = dace.define_local_scalar(dace.uint32)
        start = A_indptr[i]
        stop = A_indptr[i + 1]
        cols = A_indices[start:stop]
        vals = A_data[start:stop]
        y[i] = vals @ x[cols]


def row_through_kernel_scratch() -> dace.SDFG:
    """``B[k, :] = 2 * A[k, :]`` per kernel thread, through a scratch row the kernel allocates per thread."""
    sdfg = dace.SDFG("row_through_kernel_scratch")
    sdfg.add_array("A", [K, M], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [K, M], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient("row", [M], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    a, b, row = state.add_read("A"), state.add_write("B"), state.add_access("row")
    kernel, kernel_exit = state.add_map("rows", {"k": "0:K"}, schedule=dtypes.ScheduleType.GPU_Device)
    inner, inner_exit = state.add_map("cols", {"j": "0:M"}, schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet("twice", {"x"}, {"y"}, "y = 2 * x")
    state.add_memlet_path(a, kernel, inner, tasklet, dst_conn="x", memlet=dace.Memlet("A[k, j]"))
    state.add_memlet_path(tasklet, inner_exit, row, src_conn="y", memlet=dace.Memlet("row[j]"))
    kernel_exit.add_in_connector("IN_B")
    kernel_exit.add_out_connector("OUT_B")
    state.add_edge(row, None, kernel_exit, "IN_B", dace.Memlet(data="B", subset="k, 0:M", other_subset="0:M"))
    state.add_edge(kernel_exit, "OUT_B", b, None, dace.Memlet("B[0:K, 0:M]"))
    return sdfg


def canonical_gpu(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    return sdfg


def maps_by_schedule(sdfg: dace.SDFG, schedule: dtypes.ScheduleType) -> list:
    return [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == schedule
    ]


def library_nodes(sdfg: dace.SDFG) -> dict:
    return {
        type(node).__name__: node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.LibraryNode)
    }


@pytest.mark.parametrize("program", [matvec, gemm_loops])
def test_an_inner_reduction_runs_across_the_lanes_of_one_block(program):
    sdfg = canonical_gpu(program)
    kernels = maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)
    lanes = maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)
    assert len(kernels) == 1 and kernels[0].map.gpu_block_size is None
    assert len(lanes) == 1 and lanes[0].map.range.size() == [256]
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "gpucub::BlockReduce" in code, "the lane partials must fold with the block collective"


def test_a_host_map_that_only_launches_device_work_is_the_kernel():
    sdfg = canonical_gpu(rowsum)
    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)) == 1
    assert library_nodes(sdfg)["Reduce"].implementation == "CUDA (block strided)"


@pytest.mark.parametrize(
    "program, node, implementation",
    [(batched_dot, "Dot", "CUDA (block strided)"), (batched_gemm, "Gemm", "CUDA (block strided)")],
)
def test_a_library_node_inside_a_kernel_takes_its_block_collective(program, node, implementation):
    sdfg = canonical_gpu(program)
    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)) == 1
    assert library_nodes(sdfg)[node].implementation == implementation


def test_a_dot_over_a_gathered_row_fuses_into_one_block_reduction():
    """spmv's ``vals @ x[cols]``: the gather has a per-row extent no kernel may allocate, so the dot expands and
    fuses with it, and the row loop is the kernel."""
    sdfg = canonical_gpu(spmv)
    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)) == 1
    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)) == 1
    assert not library_nodes(sdfg)
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "gpucub::BlockReduce" in code
    # Lanes start at ``__tid``; a ``stop - start - __tid`` bound underflows with unsigned row pointers.
    assert "- __tid" not in code


def test_an_in_place_update_outside_the_inner_map_runs_on_lane_zero_between_barriers():
    """``b[i] -= s`` read-modify-writes shared memory: once per lane would subtract ``s`` 256 times."""
    sdfg = canonical_gpu(subtract_rowsum)
    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)) == 1
    guards = [
        b
        for nested in sdfg.all_sdfgs_recursive()
        for b in nested.all_control_flow_blocks()
        if isinstance(b, dace.sdfg.state.ConditionalBlock)
    ]
    assert [c.as_string for g in guards for c, body in g.branches] == ["(__tid == 0)"]
    barriers = [
        n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) and n.label == "lane_barrier"
    ]
    assert len(barriers) == 2
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert code.count("__syncthreads();") >= 2


def test_a_body_that_accumulates_outside_the_inner_map_keeps_one_thread_per_iteration():
    """Every lane runs what lies outside the strided maps; ``total += s`` there would add ``s`` once per lane."""
    sdfg = canonical_gpu(accumulate_outside)
    assert not maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)


def test_a_per_thread_scratch_read_after_the_inner_map_keeps_one_thread_per_iteration():
    """Each lane would fill its share of its own copy of ``row`` and then copy all of it out."""
    sdfg = row_through_kernel_scratch()
    BlockTileKernels().apply_pass(sdfg, {})
    assert not maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)


@pytest.mark.gpu
@pytest.mark.parametrize("m, n, k, nb", SIZES)
def test_the_block_lowerings_compute_what_numpy_does(m, n, k, nb):
    import cupy as cp

    rng = np.random.default_rng(m + n)
    A, x, b = rng.random((m, n)), rng.random(n), rng.random(m)

    y = cp.zeros(m)
    canonical_gpu(matvec)(A=cp.asarray(A), x=cp.asarray(x), y=y, M=m, N=n)
    np.testing.assert_allclose(y.get(), A @ x, rtol=1e-12)

    # A non-identity initial value is counted once, not once per lane.
    y = cp.zeros(m)
    canonical_gpu(matvec_plus)(A=cp.asarray(A), x=cp.asarray(x), b=cp.asarray(b), y=y, M=m, N=n)
    np.testing.assert_allclose(y.get(), b + A @ x, rtol=1e-12)

    # The in-place update subtracts the row total once.
    y = cp.asarray(b)
    canonical_gpu(subtract_rowsum)(A=cp.asarray(A), b=y, M=m, N=n)
    np.testing.assert_allclose(y.get(), b - A.sum(axis=1), rtol=1e-12)

    out = cp.zeros(m)
    canonical_gpu(rowsum)(A=cp.asarray(A), out=out, M=m, N=n)
    np.testing.assert_allclose(out.get(), A.sum(axis=1), rtol=1e-12)

    L, R = rng.random((m, k)), rng.random((k, n))
    C = cp.zeros((m, n))
    canonical_gpu(gemm_loops)(A=cp.asarray(L), B=cp.asarray(R), C=C, M=m, N=n, K=k)
    np.testing.assert_allclose(C.get(), L @ R, rtol=1e-12)

    X, Y = rng.random((nb, n)), rng.random((nb, n))
    dots = cp.zeros(nb)
    canonical_gpu(batched_dot)(X=cp.asarray(X), Y=cp.asarray(Y), out=dots, NB=nb, N=n)
    np.testing.assert_allclose(dots.get(), (X * Y).sum(axis=1), rtol=1e-12)

    BA, BB = rng.random((nb, m, k)), rng.random((nb, k, n))
    BC = cp.zeros((nb, m, n))
    canonical_gpu(batched_gemm)(A=cp.asarray(BA), B=cp.asarray(BB), C=BC, NB=nb, M=m, N=n, K=k)
    np.testing.assert_allclose(BC.get(), BA @ BB, rtol=1e-12)

    import scipy.sparse

    mat = scipy.sparse.random(m, n, density=0.3, format="csr", random_state=m)
    y = cp.zeros(m)
    canonical_gpu(spmv)(
        A_data=cp.asarray(mat.data),
        A_indices=cp.asarray(mat.indices.astype(np.uint32)),
        A_indptr=cp.asarray(mat.indptr.astype(np.uint32)),
        x=cp.asarray(x),
        y=y,
        M=m,
        N=n,
        NNZ=mat.nnz,
    )
    np.testing.assert_allclose(y.get(), mat @ x, rtol=1e-12)


if __name__ == "__main__":
    test_an_inner_reduction_runs_across_the_lanes_of_one_block(matvec)
    test_an_inner_reduction_runs_across_the_lanes_of_one_block(gemm_loops)
    test_a_host_map_that_only_launches_device_work_is_the_kernel()
    test_a_library_node_inside_a_kernel_takes_its_block_collective(batched_dot, "Dot", "CUDA (block strided)")
    test_a_library_node_inside_a_kernel_takes_its_block_collective(batched_gemm, "Gemm", "CUDA (block strided)")
    test_a_dot_over_a_gathered_row_fuses_into_one_block_reduction()
    test_an_in_place_update_outside_the_inner_map_runs_on_lane_zero_between_barriers()
    test_a_body_that_accumulates_outside_the_inner_map_keeps_one_thread_per_iteration()
    test_a_per_thread_scratch_read_after_the_inner_map_keeps_one_thread_per_iteration()
    for sizes in SIZES:
        test_the_block_lowerings_compute_what_numpy_does(*sizes)
