# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Consecutive device maps in a kernel, also in a nested SDFG of its body, become one kernel if each iteration of the
second reads only what the same iteration of the first wrote, and consecutive kernels otherwise, which order them
without a grid barrier.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""

import pytest

import dace

N = 32
W = 4


def _kernel_with_consecutive_maps(nested: bool, read: str = "T[i, j]") -> dace.SDFG:
    """A kernel over ``i`` whose two device maps over ``j`` write ``T[i, j]`` and then read ``read``."""
    name = f"grid_barrier_{'nested' if nested else 'flat'}_{'same' if read == 'T[i, j]' else 'other'}"
    sdfg = dace.SDFG(name)
    for name in ("A", "T", "B"):
        sdfg.add_array(name, [W, N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    kernel_entry, kernel_exit = state.add_map("kernel", dict(i=f"0:{W}"), dace.ScheduleType.GPU_Device)

    if nested:
        inner = dace.SDFG("inner")
        inner.add_symbol("i", dace.int64)
        for name in ("A", "T", "B"):
            inner.add_array(name, [W, N], dace.float64, storage=dace.StorageType.GPU_Global)
        body = inner.add_state()
    else:
        body = state

    first_entry, first_exit = body.add_map("first", dict(j=f"0:{N}"), dace.ScheduleType.GPU_Device)
    second_entry, second_exit = body.add_map("second", dict(j=f"0:{N}"), dace.ScheduleType.GPU_Device)
    first = body.add_tasklet("first", {"a"}, {"t"}, "t = a * 5")
    second = body.add_tasklet("second", {"t"}, {"b"}, "b = t + 1")
    tmp = body.add_access("T")
    if nested:
        body.add_memlet_path(body.add_read("A"), first_entry, first, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
        body.add_memlet_path(first, first_exit, tmp, src_conn="t", memlet=dace.Memlet("T[i, j]"))
        body.add_memlet_path(tmp, second_entry, second, dst_conn="t", memlet=dace.Memlet(read))
        body.add_memlet_path(second, second_exit, body.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i, j]"))
        node = state.add_nested_sdfg(inner, {"A"}, {"T", "B"}, {"i": "i"})
        state.add_memlet_path(
            state.add_read("A"), kernel_entry, node, dst_conn="A", memlet=dace.Memlet(f"A[0:{W}, 0:{N}]")
        )
        state.add_memlet_path(
            node, kernel_exit, state.add_write("T"), src_conn="T", memlet=dace.Memlet(f"T[0:{W}, 0:{N}]")
        )
        state.add_memlet_path(
            node, kernel_exit, state.add_write("B"), src_conn="B", memlet=dace.Memlet(f"B[0:{W}, 0:{N}]")
        )
    else:
        state.add_memlet_path(
            state.add_read("A"), kernel_entry, first_entry, first, dst_conn="a", memlet=dace.Memlet("A[i, j]")
        )
        state.add_memlet_path(first, first_exit, tmp, src_conn="t", memlet=dace.Memlet("T[i, j]"))
        state.add_memlet_path(tmp, second_entry, second, dst_conn="t", memlet=dace.Memlet(read))
        state.add_memlet_path(
            second, second_exit, kernel_exit, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i, j]")
        )
    sdfg.validate()
    return sdfg


def _cuda_code(sdfg: dace.SDFG) -> str:
    with dace.config.set_temporary("compiler", "cuda", "backend", value="cuda"):
        return next(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cu")


@pytest.mark.parametrize("nested", [False, True])
def test_maps_sharing_their_iterations_become_one_kernel(nested: bool):
    code = _cuda_code(_kernel_with_consecutive_maps(nested))
    assert code.count("__global__") == 1
    assert "this_grid().sync();" not in code


@pytest.mark.parametrize("nested", [False, True])
def test_maps_depending_across_iterations_become_consecutive_kernels(nested: bool):
    code = _cuda_code(_kernel_with_consecutive_maps(nested, read=f"T[i, {N} - 1 - j]"))
    assert code.count("__global__") == 2
    assert "this_grid().sync();" not in code


if __name__ == "__main__":
    test_maps_sharing_their_iterations_become_one_kernel(False)
    test_maps_sharing_their_iterations_become_one_kernel(True)
    test_maps_depending_across_iterations_become_consecutive_kernels(False)
    test_maps_depending_across_iterations_become_consecutive_kernels(True)
