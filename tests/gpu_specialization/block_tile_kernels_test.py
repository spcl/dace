# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on the GPU runs a kernel one block per outer iteration and its inner work across the lanes.

An inner reduction folds with ``gpucub::BlockReduce``, an in-kernel ``Dot`` / ``Gemm`` / ``Reduce`` takes its
block collective, and a host map that only launched device work becomes the kernel. The structural checks
need no GPU; the numeric ones run on the device, at sizes that leave the last block partial.
"""

import numpy as np
import pytest
from ordered_set import OrderedSet

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.gpu_specialization.block_tile_kernels import (
    BlockTileKernels,
    place_barriers,
    tile_per_cta,
    updates_shared,
)

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
def gathered_gemm(A: dace.float64[NB, M, K], B: dace.float64[NB, K, N], idx: dace.int64[NB], C: dace.float64[NB, M, N]):
    for b in dace.map[0:NB]:
        C[b] = A[idx[b]] @ B[b]


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


def row_staged_into_kernel_scratch() -> dace.SDFG:
    """``row = A[k, :]; B[k, j] = 2 * row[j]`` per kernel thread: icon_scatter's staging copy, filled outside the inner
    map and only read inside it."""
    sdfg = dace.SDFG("row_staged_into_kernel_scratch")
    sdfg.add_array("A", [K, M], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [K, M], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient("row", [M], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    a, b, row = state.add_read("A"), state.add_write("B"), state.add_access("row")
    kernel, kernel_exit = state.add_map("rows", {"k": "0:K"}, schedule=dtypes.ScheduleType.GPU_Device)
    inner, inner_exit = state.add_map("cols", {"j": "0:M"}, schedule=dtypes.ScheduleType.Sequential)
    state.add_memlet_path(a, kernel, row, memlet=dace.Memlet(data="A", subset="k, 0:M", other_subset="0:M"))
    tasklet = state.add_tasklet("twice", {"x"}, {"y"}, "y = 2 * x")
    state.add_memlet_path(row, inner, tasklet, dst_conn="x", memlet=dace.Memlet("row[j]"))
    state.add_memlet_path(tasklet, inner_exit, kernel_exit, b, src_conn="y", memlet=dace.Memlet("B[k, j]"))
    sdfg.validate()
    return sdfg


@dace.program
def blocked_columns(A: dace.float64[NB, M, N], B: dace.float64[NB, M, N]):
    for jb, jk, jc in dace.map[0:NB, 0:M, 0:N]:
        B[jb, jk, jc] = 2.0 * A[jb, jk, jc]


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
    [(batched_dot, "Dot", "CUDA (block strided)"), (gathered_gemm, "Gemm", "CUDA (block strided)")],
)
def test_a_library_node_inside_a_kernel_takes_its_block_collective(program, node, implementation):
    sdfg = canonical_gpu(program)
    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)) == 1
    assert library_nodes(sdfg)[node].implementation == implementation


def test_a_top_level_batched_gemm_is_one_vendor_batched_call():
    """``C[b] = A[b] @ B[b]`` over strided batches is a top-level BatchedMatMul: the vendor's strided-batched GEMM,
    no kernel of per-block products around it."""
    sdfg = canonical_gpu(batched_gemm)
    assert not maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)
    assert library_nodes(sdfg)["BatchedMatMul"].implementation in ("cuBLAS", "hipBLAS")


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


def test_a_tasklet_writing_a_plane_a_kernel_before_it_wrote_reads_nothing_it_updates():
    sdfg = dace.SDFG("plane_after_plane")
    sdfg.add_array("A", [4, 2], dtypes.float64)
    state = sdfg.add_state()
    _, _, first_exit = state.add_mapped_tasklet(
        "lo", {"i": "0:4"}, {}, "o = 1.0", {"o": dace.Memlet("A[i, 0]")}, external_edges=True
    )
    second = state.add_tasklet("hi", {}, {"o": None}, "o = 2.0")
    entry, exit_ = state.add_map("kernel", {"i": "0:4"})
    state.add_nedge(entry, second, dace.Memlet())
    written = state.out_edges(first_exit)[0].dst
    state.add_nedge(written, entry, dace.Memlet())
    state.add_memlet_path(second, exit_, state.add_write("A"), src_conn="o", memlet=dace.Memlet("A[i, 1]"))
    assert not updates_shared(second, state, OrderedSet())


def test_a_body_that_accumulates_outside_the_inner_map_keeps_one_thread_per_iteration():
    """Every lane runs what lies outside the strided maps; ``total += s`` there would add ``s`` once per lane."""
    sdfg = canonical_gpu(accumulate_outside)
    assert not maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)


def test_a_per_thread_scratch_read_after_the_inner_map_keeps_one_thread_per_iteration():
    """Each lane would fill its share of its own copy of ``row`` and then copy all of it out."""
    sdfg = row_through_kernel_scratch()
    BlockTileKernels().apply_pass(sdfg, {})
    assert not maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)


def test_a_per_thread_buffer_filled_outside_the_inner_map_keeps_one_thread_per_iteration():
    """Every lane would allocate and fill the whole row: 256 copies per block, which the device codegen hoists out of
    the kernel sized for every thread (icon_scatter's staging copies, ~54 GB at preset M)."""
    sdfg = row_staged_into_kernel_scratch()
    BlockTileKernels().apply_pass(sdfg, {})
    assert not maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)


@dace.program
def shuffled_rows(A: dace.float64[K, 512], P: dace.int64[512], B: dace.float64[K, 512]):
    for k in dace.map[0:K]:
        row = np.empty(512, dace.float64)
        for j in dace.map[0:512]:
            row[j] = 2 * A[k, j]
        for j in range(1, 512):
            row[j] = row[j] + row[P[j]]
        for j in dace.map[0:512]:
            B[k, j] = row[j]


def test_a_row_the_lanes_fill_and_a_sequential_loop_updates_is_the_blocks_and_the_loop_runs_on_lane_zero():
    """One CTA per row: the lanes fill one copy of ``row``, lane 0 walks the data-dependent loop over it."""
    sdfg = canonical_gpu(shuffled_rows)

    assert len(maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)) == 1
    rows = [desc for nested in sdfg.all_sdfgs_recursive() for name, desc in nested.arrays.items() if name == "row"]
    assert rows and all(len(desc.shape) == 2 and desc.storage == dtypes.StorageType.GPU_Global for desc in rows)
    guards = [
        block
        for nested in sdfg.all_sdfgs_recursive()
        for block in nested.all_control_flow_blocks()
        if isinstance(block, dace.sdfg.state.ConditionalBlock) and block.label.endswith("_lane_zero")
    ]
    assert any(isinstance(guard.branches[0][1].start_block, dace.sdfg.state.LoopRegion) for guard in guards)


@pytest.mark.gpu
def test_a_row_the_lanes_fill_and_lane_zero_updates_matches_numpy():
    cupy = pytest.importorskip("cupy")
    rng = np.random.default_rng(11)
    A, P = rng.random((40, 512)), rng.integers(0, 512, 512)
    expected = np.empty_like(A)
    for k in range(40):
        row = 2 * A[k]
        for j in range(1, 512):
            row[j] = row[j] + row[P[j]]
        expected[k] = row
    sdfg = canonical_gpu(shuffled_rows)
    B = cupy.zeros((40, 512))

    sdfg(A=cupy.asarray(A), P=cupy.asarray(P), B=B, K=40)

    np.testing.assert_allclose(cupy.asnumpy(B), expected)


def test_a_small_block_axis_is_not_strided_across_the_lanes():
    """The offload keeps a SMALL ``nblks`` axis out of the grid as a sequential map inside the kernel. Striding it
    across 256 lanes would make every grid point a block of which only ``nblks`` lanes work."""
    sdfg = blocked_columns.to_sdfg(simplify=True)
    sdfg.set_symbol_scale("NB", dace.symbolic.Scale.SMALL)
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    assert maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_Device)
    lanes = [p for e in maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock) for p in e.map.params]
    assert not [p for p in lanes if p.endswith("jb")], lanes


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


@dace.program
def bitonic_merge_rows(key: dace.int64[M, 1024]):
    for r in dace.map[0:M]:
        stride = 512
        while stride > 0:
            for idx in range(512):
                pos = 2 * idx - (idx % stride)
                left = key[r, pos]
                right = key[r, pos + stride]
                if left > right:
                    key[r, pos] = right
                    key[r, pos + stride] = left
            stride = stride // 2


@pytest.mark.gpu
def test_a_lane_strided_map_in_a_loop_waits_for_every_lane_before_its_next_trip():
    """Each bitonic stage reads elements other lanes swapped in the stage before: a barrier ends every trip."""
    cupy = pytest.importorskip("cupy")
    sdfg = canonical_gpu(bitonic_merge_rows)
    assert maps_by_schedule(sdfg, dtypes.ScheduleType.GPU_ThreadBlock)
    key = np.random.default_rng(0).integers(0, 1 << 20, size=(7, 1024))
    key[:, :512].sort(axis=1)
    key[:, 512:] = -np.sort(-key[:, 512:], axis=1)
    device = cupy.asarray(key)
    sdfg(key=device, M=7)
    np.testing.assert_array_equal(cupy.asnumpy(device), np.sort(key, axis=1))


T = dace.symbol("T")
TABLE = [3, 1, 0, 2]


def per_task_table_and_row() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    """Per task ``t``: four scalar stores fill ``tab``, a map fills ``row[j] = 2 * A[t, j]``, a map reads
    ``out[t, k] = row[tab[k]]``. ``tab`` and ``row`` become per-CTA arrays."""
    body = dace.SDFG("per_task_body")
    body.add_symbol("t", dace.int64)
    body.add_array("A", [T, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    body.add_array("out", [T, 4], dace.float64, storage=dtypes.StorageType.GPU_Global)
    body.add_transient("tab", [4], dace.int64)
    body.add_transient("row", [N], dace.float64)
    st = body.add_state()
    tab, row = st.add_access("tab"), st.add_access("row")
    for k, value in enumerate(TABLE):
        store = st.add_tasklet(f"store{k}", {}, {"o"}, f"o = {value}")
        st.add_edge(store, "o", tab, None, dace.Memlet(f"tab[{k}]"))
    st.add_mapped_tasklet(
        "fill",
        {"j": "0:N"},
        {"a": dace.Memlet("A[t, j]")},
        "r = 2 * a",
        {"r": dace.Memlet("row[j]")},
        input_nodes={"A": st.add_read("A")},
        output_nodes={"row": row},
        external_edges=True,
    )
    st.add_mapped_tasklet(
        "gather",
        {"k": "0:4"},
        {"tb": dace.Memlet("tab[k]"), "rw": dace.Memlet("row[0:N]")},
        "o = rw[tb]",
        {"o": dace.Memlet("out[t, k]")},
        input_nodes={"tab": tab, "row": row},
        external_edges=True,
    )
    sdfg = dace.SDFG("per_task_table_and_row")
    sdfg.add_array("A", [T, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("out", [T, 4], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    kernel, kernel_exit = state.add_map("tasks", {"t": "0:T"}, schedule=dtypes.ScheduleType.GPU_Device)
    nest = state.add_nested_sdfg(body, {"A": None}, {"out": None}, {"t": "t", "N": "N", "T": "T"})
    state.add_memlet_path(state.add_read("A"), kernel, nest, dst_conn="A", memlet=dace.Memlet("A[0:T, 0:N]"))
    state.add_memlet_path(
        nest, kernel_exit, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:T, 0:4]")
    )
    sdfg.validate()
    return sdfg, state, kernel


def barriers_in(sdfg: dace.SDFG) -> int:
    tasklets = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) and n.label == "lane_barrier"
    ]
    syncs = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapExit) and n.map.gpu_force_syncthreads
    ]
    return len(tasklets) + len(syncs)


def test_per_cta_lane_zero_stores_share_one_barrier_and_independent_steps_none():
    """The four lane-0 stores run back to back; the row fill touches nothing they do. Only the gather, which reads
    both, waits; the state ends at a barrier so the next task starts fenced."""
    sdfg, state, kernel = per_task_table_and_row()
    assert tile_per_cta(state, kernel)
    sdfg.validate()
    guards = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG) and n.label == "lane_zero"
    ]
    assert len(guards) == len(TABLE)
    assert barriers_in(sdfg) == 2


def test_a_step_reading_what_another_wrote_through_a_view_waits_at_a_barrier():
    """``V`` views ``G``: the second map reads what the first wrote, under another name."""
    sdfg = dace.SDFG("write_view_read_container")
    sdfg.add_array("G", [64], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("H", [64], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_view("V", [64], dace.float64)
    state = sdfg.add_state()
    view, container = state.add_access("V"), state.add_access("G")
    state.add_edge(view, "views", container, None, dace.Memlet("G[0:64]"))
    state.add_mapped_tasklet(
        "write", {"i": "0:64"}, {}, "v = i", {"v": dace.Memlet("V[i]")}, output_nodes={"V": view}, external_edges=True
    )
    state.add_mapped_tasklet(
        "read",
        {"i": "0:64"},
        {"g": dace.Memlet("G[63 - i]")},
        "h = g",
        {"h": dace.Memlet("H[i]")},
        input_nodes={"G": container},
        external_edges=True,
    )
    place_barriers(state, OrderedSet())
    sdfg.validate()
    assert barriers_in(sdfg) == 2


@pytest.mark.gpu
def test_per_cta_lane_zero_stores_and_lane_maps_match_numpy():
    cupy = pytest.importorskip("cupy")
    sdfg, state, kernel = per_task_table_and_row()
    tile_per_cta(state, kernel)
    A = np.random.default_rng(12).random((300, 70))
    out = cupy.zeros((300, 4))
    sdfg(A=cupy.asarray(A), out=out, T=300, N=70)
    np.testing.assert_allclose(cupy.asnumpy(out), 2 * A[:, TABLE])


if __name__ == "__main__":
    test_an_inner_reduction_runs_across_the_lanes_of_one_block(matvec)
    test_an_inner_reduction_runs_across_the_lanes_of_one_block(gemm_loops)
    test_a_host_map_that_only_launches_device_work_is_the_kernel()
    test_a_library_node_inside_a_kernel_takes_its_block_collective(batched_dot, "Dot", "CUDA (block strided)")
    test_a_library_node_inside_a_kernel_takes_its_block_collective(gathered_gemm, "Gemm", "CUDA (block strided)")
    test_a_top_level_batched_gemm_is_one_vendor_batched_call()
    test_a_tasklet_writing_a_plane_a_kernel_before_it_wrote_reads_nothing_it_updates()
    test_a_dot_over_a_gathered_row_fuses_into_one_block_reduction()
    test_an_in_place_update_outside_the_inner_map_runs_on_lane_zero_between_barriers()
    test_a_body_that_accumulates_outside_the_inner_map_keeps_one_thread_per_iteration()
    test_a_per_thread_scratch_read_after_the_inner_map_keeps_one_thread_per_iteration()
    test_a_per_thread_buffer_filled_outside_the_inner_map_keeps_one_thread_per_iteration()
    for sizes in SIZES:
        test_the_block_lowerings_compute_what_numpy_does(*sizes)
    test_a_lane_strided_map_in_a_loop_waits_for_every_lane_before_its_next_trip()
    test_per_cta_lane_zero_stores_share_one_barrier_and_independent_steps_none()
    test_a_step_reading_what_another_wrote_through_a_view_waits_at_a_barrier()
    test_per_cta_lane_zero_stores_and_lane_maps_match_numpy()
