# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sequential loops inside GPU kernels get ``#pragma unroll`` by default; CPU loops get ``#pragma omp unroll`` only
when asked to unroll."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion

N = dace.symbol("N")
M = dace.symbol("M")


def kernel_with_inner_map(unroll_factor: int = 0) -> dace.SDFG:
    """A GPU kernel over ``i`` whose body is a sequential map over ``j``."""
    sdfg = dace.SDFG(f"kernel_with_inner_map_{unroll_factor}")
    sdfg.add_array("A", [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    a_in, a_out = state.add_read("A"), state.add_write("A")
    outer_entry, outer_exit = state.add_map("outer", dict(i="0:N"), schedule=dace.ScheduleType.GPU_Device)
    outer_entry.map.gpu_block_size = [256, 1, 1]
    inner_entry, inner_exit = state.add_map("inner", dict(j="0:M"), schedule=dace.ScheduleType.Sequential)
    inner_entry.map.unroll_factor = unroll_factor
    tasklet = state.add_tasklet("double", {"a"}, {"b"}, "b = 2 * a")
    state.add_memlet_path(a_in, outer_entry, inner_entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, a_out, src_conn="b", memlet=dace.Memlet("A[i, j]"))
    return sdfg


def kernel_with_inner_loop() -> dace.SDFG:
    """A GPU kernel over ``i`` whose body is a nested SDFG with a loop region over ``j``."""
    inner = dace.SDFG("inner_loop")
    inner.add_symbol("i", dace.int64)
    inner.add_array("A", [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    loop = LoopRegion("jloop", "j < M", "j", "j = 0", "j = j + 1")
    inner.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("double", {"a"}, {"b"}, "b = 2 * a")
    body.add_edge(body.add_read("A"), None, tasklet, "a", dace.Memlet("A[i, j]"))
    body.add_edge(tasklet, "b", body.add_write("A"), None, dace.Memlet("A[i, j]"))

    sdfg = dace.SDFG("kernel_with_inner_loop")
    sdfg.add_array("A", [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("outer", dict(i="0:N"), schedule=dace.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [256, 1, 1]
    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"A"}, symbol_mapping={"N": "N", "M": "M", "i": "i"})
    state.add_memlet_path(state.add_read("A"), entry, nsdfg, dst_conn="A", memlet=dace.Memlet("A[i, 0:M]"))
    state.add_memlet_path(nsdfg, exit_, state.add_write("A"), src_conn="A", memlet=dace.Memlet("A[i, 0:M]"))
    return sdfg


def host_loop(unroll: bool = False, unroll_factor: int = 0) -> dace.SDFG:
    sdfg = dace.SDFG(f"host_loop_{int(unroll)}_{unroll_factor}")
    sdfg.add_array("A", [N], dace.float64)
    state = sdfg.add_state()
    tasklet, entry, exit_ = state.add_mapped_tasklet(
        "double",
        dict(i="0:N"),
        {"a": dace.Memlet("A[i]")},
        "b = 2 * a",
        {"b": dace.Memlet("A[i]")},
        schedule=dace.ScheduleType.Sequential,
        external_edges=True,
    )
    entry.map.unroll = unroll
    entry.map.unroll_factor = unroll_factor
    return sdfg


def generated_code(sdfg: dace.SDFG) -> str:
    with dace.config.set_temporary("compiler", "cuda", "backend", value="cuda"):
        return "\n".join(obj.clean_code for obj in sdfg.generate_code())


def lines_before_loop(code: str, loop_variable: str) -> str:
    """The line right before the ``for`` of ``loop_variable``."""
    lines = code.splitlines()
    index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("for (") and f" {loop_variable} = " in line
    )
    return lines[index - 1].strip()


def test_an_inner_map_of_a_gpu_kernel_unrolls_by_default():
    code = generated_code(kernel_with_inner_map())
    assert lines_before_loop(code, "j") == "#pragma unroll"


def test_an_unroll_factor_of_one_keeps_an_inner_map_rolled():
    code = generated_code(kernel_with_inner_map(unroll_factor=1))
    assert lines_before_loop(code, "j") == "#pragma unroll 1"


def test_a_loop_region_in_a_gpu_kernel_unrolls_by_default():
    code = generated_code(kernel_with_inner_loop())
    assert lines_before_loop(code, "j") == "#pragma unroll"


def test_a_host_loop_gets_no_unroll_pragma():
    code = generated_code(host_loop())
    assert "unroll" not in code


@pytest.mark.parametrize("unroll_factor, pragma", [(0, "#pragma omp unroll"), (4, "#pragma omp unroll partial(4)")])
def test_a_host_loop_asked_to_unroll_gets_an_omp_unroll_pragma_and_runs(unroll_factor, pragma):
    sdfg = host_loop(unroll=True, unroll_factor=unroll_factor)
    assert lines_before_loop(generated_code(sdfg), "i") == pragma
    data = np.arange(37, dtype=np.float64)
    sdfg(A=data, N=37)
    np.testing.assert_array_equal(data, 2 * np.arange(37, dtype=np.float64))


if __name__ == "__main__":
    test_an_inner_map_of_a_gpu_kernel_unrolls_by_default()
    test_an_unroll_factor_of_one_keeps_an_inner_map_rolled()
    test_a_loop_region_in_a_gpu_kernel_unrolls_by_default()
    test_a_host_loop_gets_no_unroll_pragma()
    test_a_host_loop_asked_to_unroll_gets_an_omp_unroll_pragma_and_runs(0, "#pragma omp unroll")
    test_a_host_loop_asked_to_unroll_gets_an_omp_unroll_pragma_and_runs(4, "#pragma omp unroll partial(4)")
