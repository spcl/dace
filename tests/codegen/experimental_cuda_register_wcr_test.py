# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A WCR copy out of a per-thread register is a thread-level write; only a shared buffer is reduced block-wide."""

import dace
from dace import dtypes


def register_wcr_sdfg() -> dace.SDFG:
    """``S += 2 * A[i]`` in a GPU_Device map without a thread-block map, through the register scalar ``t``."""
    sdfg = dace.SDFG("experimental_cuda_register_wcr")
    sdfg.add_array("A", [64], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("S", [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("t", dace.float64, storage=dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("sum", {"i": "0:64"}, schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet("double", {"a"}, {"o"}, "o = 2 * a")
    t = state.add_access("t")
    state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_edge(tasklet, "o", t, None, dace.Memlet("t[0]"))
    state.add_memlet_path(
        t, exit_node, state.add_write("S"), memlet=dace.Memlet("S[0]", wcr="lambda x, y: x + y", other_subset="0")
    )
    sdfg.validate()
    return sdfg


def test_a_register_wcr_source_is_a_thread_level_write():
    sdfg = register_wcr_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        device = next(code for code in sdfg.generate_code() if code.language == "cu").clean_code
    kernel = device[device.index("__global__") :]
    assert "SharedToGlobal1D" not in kernel
    assert "Accum" not in kernel


if __name__ == "__main__":
    test_a_register_wcr_source_is_a_thread_level_write()
