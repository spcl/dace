# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reduce-shaped library nodes inside a GPU kernel lower to one thread-block CUB collective by default.

One kernel iteration per row: ``ArgReduce`` and ``FindFirst`` over that row become a ``GPU_ThreadBlock`` lane map
around a block-strided scan and one ``gpucub::BlockReduce``, never a per-thread serial loop and never a host-only
device-wide call.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.helper import select_implementation_by_schedule
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.find_first import FindFirst
from dace.sdfg import nodes

R = dace.symbol("R", dtype=dace.int64)
N = dace.symbol("N", dtype=dace.int64)


def row_kernel(
    name: str,
    library_node: nodes.LibraryNode,
    inputs: dict[str, str],
    outputs: dict[str, str],
    thread_level: bool = False,
):
    """A ``GPU_Device`` map over rows ``i`` holding ``library_node``, wired to global arrays by the given memlets;
    ``thread_level`` puts a ``GPU_ThreadBlock`` map between them."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [R, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("val", [R], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("idx", [R], dace.int64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    scopes = [state.add_map("rows", {"i": "0:R"}, schedule=dtypes.ScheduleType.GPU_Device)]
    if thread_level:
        scopes.append(state.add_map("lanes", {"j": "0:32"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock))
    state.add_node(library_node)
    for conn, subset in inputs.items():
        path = [state.add_read(subset.split("[")[0]), *(entry for entry, exit_node in scopes), library_node]
        state.add_memlet_path(*path, dst_conn=conn, memlet=dace.Memlet(subset))
    for conn, subset in outputs.items():
        path = [
            library_node,
            *(exit_node for entry, exit_node in reversed(scopes)),
            state.add_write(subset.split("[")[0]),
        ]
        state.add_memlet_path(*path, src_conn=conn, memlet=dace.Memlet(subset))
    return sdfg, state


def argmax_kernel(thread_level: bool = False):
    return row_kernel(
        "argmax_rows",
        ArgReduce("argmax", op="max"),
        {"_in": "A[i, 0:N]"},
        {"_out_val": "val[i]", "_out_idx": "idx[i]"},
        thread_level,
    )


def find_first_kernel():
    search = FindFirst("first_above", predicate="_a[i, __i] > 0.5", begin=0, end=N)
    search.symbols = ["i"]
    return row_kernel("find_first_rows", search, {"_a": "A[0:R, 0:N]"}, {"_out_idx": "idx[i]"})


def expanded_tasklet(sdfg: dace.SDFG) -> nodes.Tasklet:
    sdfg.expand_library_nodes()
    lanes = [
        n
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert [lane.map.range.size()[0] for lane in lanes] == [256], "one lane map of the collective's width"
    (tasklet,) = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    return tasklet


@pytest.mark.parametrize("build", [argmax_kernel, find_first_kernel])
def test_an_in_kernel_node_lowers_to_one_block_collective_by_default(build):
    sdfg, state = build()
    code = expanded_tasklet(sdfg).code.as_string
    assert "gpucub::BlockReduce<" in code and ", 256>" in code


def test_an_argmax_breaks_ties_with_the_shared_pair_fold():
    sdfg, state = argmax_kernel()
    assert "dace::arg_max_combine(__arl, __arr)" in expanded_tasklet(sdfg).code.as_string


def test_a_search_stops_at_the_first_block_step_with_a_hit():
    sdfg, state = find_first_kernel()
    code = expanded_tasklet(sdfg).code.as_string
    assert "if (__syncthreads_or(__ff_found))" in code and "break;" in code


def test_a_node_in_thread_level_code_keeps_its_per_thread_lowering():
    """Inside a thread-block map every lane would enter the collective on its own."""
    sdfg, state = argmax_kernel(thread_level=True)
    node = next(n for n in state.nodes() if isinstance(n, ArgReduce))
    node.schedule = dtypes.ScheduleType.Sequential
    sdfg.expand_library_nodes()
    assert node.implementation == "pure"


def test_asking_which_lowering_an_in_kernel_node_takes_keeps_the_kernel_block_size():
    """Only the expansion that commits to the block collective hands the block size over to its lane map."""
    sdfg, state = argmax_kernel()
    rows = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    rows.map.gpu_block_size = [128, 1, 1]
    node = next(n for n in state.nodes() if isinstance(n, ArgReduce))

    picked = select_implementation_by_schedule(node, state)

    assert picked == "CUDA (block)" and rows.map.gpu_block_size == [128, 1, 1]


@pytest.mark.gpu
def test_row_argmax_in_a_kernel_matches_numpy_with_ties():
    import cupy

    sdfg, state = argmax_kernel()
    rng = np.random.default_rng(7)
    rows, n = 5, 1000
    a = rng.integers(0, 9, (rows, n)).astype(np.float64)  # many ties: the first occurrence must win
    a[3, :] = -1.0
    a[4, 700] = 50.0
    val, idx = cupy.zeros(rows), cupy.zeros(rows, dtype=np.int64)
    sdfg(A=cupy.asarray(a), val=val, idx=idx, R=rows, N=n)
    assert np.array_equal(idx.get(), np.argmax(a, axis=1))
    assert np.array_equal(val.get(), np.max(a, axis=1))


@pytest.mark.gpu
def test_row_find_first_in_a_kernel_matches_numpy():
    import cupy

    sdfg, state = find_first_kernel()
    rows, n = 4, 1000
    a = np.zeros((rows, n))
    a[0, 0] = 1.0
    a[1, 257] = 1.0
    a[1, 900] = 1.0
    a[2, 999] = 1.0  # row 3 never fires: the answer is the end of the range
    idx = cupy.zeros(rows, dtype=np.int64)
    sdfg(A=cupy.asarray(a), val=cupy.zeros(rows), idx=idx, R=rows, N=n)
    assert idx.get().tolist() == [0, 257, 999, n]


def test_a_complex_block_reduce_starts_from_a_cpp_identity():
    """A complex row sum lowers to a block reduction whose accumulator starts at the identity spelled in C++:
    Python's ``0j`` is a GNU ``_Complex int`` literal there, which HIP refuses to convert (vexx_k)."""
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    n = dace.symbol("n", dtype=dace.int64, positive=True)

    @dace.program
    def row_sums(a: dace.complex128[n, n], out: dace.complex128[n]):
        for i in range(n):
            s = 0j
            for j in range(n):
                s = s + a[i, j]
            out[i] = s

    sdfg = row_sums.to_sdfg(simplify=False)
    canonicalize(sdfg, target="gpu", validate_all=False)
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, target="gpu")
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert "BlockReduce" in code or "block_reduce" in code.lower()
    assert "(0j)" not in code


if __name__ == "__main__":
    for build in (argmax_kernel, find_first_kernel):
        test_an_in_kernel_node_lowers_to_one_block_collective_by_default(build)
    test_an_argmax_breaks_ties_with_the_shared_pair_fold()
    test_a_search_stops_at_the_first_block_step_with_a_hit()
    test_a_node_in_thread_level_code_keeps_its_per_thread_lowering()
    test_row_argmax_in_a_kernel_matches_numpy_with_ties()
    test_row_find_first_in_a_kernel_matches_numpy()
    test_a_complex_block_reduce_starts_from_a_cpp_identity()
