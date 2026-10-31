# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A library expansion's transients take the storage of where the expansion runs, read off the scopes around it
and not only off the node's own schedule."""

import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.norm2 import Norm2

N = dace.symbol("N")


def argmax_rows(schedule: dtypes.ScheduleType, storage: dtypes.StorageType, thread_blocks: bool = False) -> dace.SDFG:
    """``idx[i] = argmax(a[i, :])`` by a ``Sequential`` ``pure`` ArgReduce under a map over rows."""
    sdfg = dace.SDFG("argmax_rows")
    sdfg.add_array("a", [N, 16], dace.float64, storage=storage)
    sdfg.add_array("idx", [N], dace.int64, storage=storage)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:N"}, schedule=schedule)
    if thread_blocks:
        inner_entry, inner_exit = state.add_map("lane", {"j": "0:1"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock)
        path_in, path_out = (entry, inner_entry), (inner_exit, exit_node)
    else:
        path_in, path_out = (entry,), (exit_node,)
    node = ArgReduce("argmax")
    node.implementation = "pure"
    node.schedule = dtypes.ScheduleType.Sequential
    node.remove_out_connector("_out_val")
    state.add_node(node)
    state.add_memlet_path(state.add_read("a"), *path_in, node, dst_conn="_in", memlet=dace.Memlet("a[i, 0:16]"))
    state.add_memlet_path(node, *path_out, state.add_write("idx"), src_conn="_out_idx", memlet=dace.Memlet("idx[i]"))
    return sdfg


def norm_rows(schedule: dtypes.ScheduleType, storage: dtypes.StorageType) -> dace.SDFG:
    """``n[i] = norm2(a[i, :])`` by a ``Sequential`` Norm2 under a map over rows."""
    sdfg = dace.SDFG("norm_rows")
    sdfg.add_array("a", [N, 16], dace.float64, storage=storage)
    sdfg.add_array("n", [N], dace.float64, storage=storage)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:N"}, schedule=schedule)
    node = Norm2("norm")
    node.schedule = dtypes.ScheduleType.Sequential
    state.add_node(node)
    state.add_memlet_path(state.add_read("a"), entry, node, dst_conn="_x", memlet=dace.Memlet("a[i, 0:16]"))
    state.add_memlet_path(node, exit_node, state.add_write("n"), src_conn="_out", memlet=dace.Memlet("n[i]"))
    return sdfg


def expansion_storages(sdfg: dace.SDFG, names: tuple[str, ...]) -> dict[str, dtypes.StorageType]:
    sdfg.expand_library_nodes(recursive=False)
    return {
        name: desc.storage
        for nested in sdfg.all_sdfgs_recursive()
        if nested is not sdfg
        for name, desc in nested.arrays.items()
        if name in names
    }


@pytest.mark.parametrize("thread_blocks", [False, True])
def test_an_argreduce_in_a_kernel_keeps_its_running_best_in_registers(thread_blocks):
    """One thread's running maximum and index. Inferred from the node's ``Sequential`` alone, they were
    ``CPU_Heap``: host memory, allocated outside the kernel."""
    sdfg = argmax_rows(dtypes.ScheduleType.GPU_Device, dtypes.StorageType.GPU_Global, thread_blocks)
    storages = expansion_storages(sdfg, ("best_val", "best_idx"))
    assert storages == dict.fromkeys(("best_val", "best_idx"), dtypes.StorageType.Register), storages


def test_a_norm_in_a_kernel_keeps_its_sum_of_squares_in_a_register():
    """A kernel without thread-block maps runs its body per thread, so the expansion's own maps are not
    thread-block maps and the sum is not block-shared memory."""
    storages = expansion_storages(norm_rows(dtypes.ScheduleType.GPU_Device, dtypes.StorageType.GPU_Global), ("_sumsq",))
    assert storages == {"_sumsq": dtypes.StorageType.Register}, storages


def test_an_expansion_in_a_parallel_cpu_map_is_thread_local():
    """Each iteration of a parallel map calls the expansion; its scalars live on that thread's stack, as the
    transients of any nested SDFG in such a map already do, not in a heap allocation per call."""
    sdfg = argmax_rows(dtypes.ScheduleType.CPU_Multicore, dtypes.StorageType.CPU_Heap)
    storages = expansion_storages(sdfg, ("best_val", "best_idx"))
    assert storages == dict.fromkeys(("best_val", "best_idx"), dtypes.StorageType.Register), storages


def test_a_host_level_expansion_keeps_its_host_storage():
    """No scope around the node: what its own schedule says, as before."""
    sdfg = argmax_rows(dtypes.ScheduleType.Sequential, dtypes.StorageType.CPU_Heap)
    storages = expansion_storages(sdfg, ("best_val", "best_idx"))
    assert storages == dict.fromkeys(("best_val", "best_idx"), dtypes.StorageType.CPU_Heap), storages


if __name__ == "__main__":
    test_an_argreduce_in_a_kernel_keeps_its_running_best_in_registers(False)
    test_an_argreduce_in_a_kernel_keeps_its_running_best_in_registers(True)
    test_a_norm_in_a_kernel_keeps_its_sum_of_squares_in_a_register()
    test_an_expansion_in_a_parallel_cpu_map_is_thread_local()
    test_a_host_level_expansion_keeps_its_host_storage()
