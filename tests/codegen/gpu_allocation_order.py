# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
from dace.codegen.exceptions import CompilationError


@pytest.mark.gpu
def test_allocation_requiring_map_variables():
    """
    The goal of this test is to ensure map variables are defined
    before allocating arrays in scopes, since the allocated array's
    size may depend on a map variable.
    """

    # Create SDFG and state
    sdfg = dace.SDFG("example")
    state = sdfg.add_state("main")

    # Add input and output array (same array) and corrsponding accessNodes
    sdfg.add_array("gpu_A", (128, ), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    gpu_a_acc_read = state.add_access("gpu_A")
    gpu_a_acc_write = state.add_access("gpu_A")

    # Add GPU_Device and GPU_ThreadBlock Maps
    gpu_map_entry, gpu_map_exit = state.add_map("gpu_map",
                                                dict(bi="0:128:32"),
                                                schedule=dace.dtypes.ScheduleType.GPU_Device)
    tb_map_entry, tb_map_exit = state.add_map(
        "tb",
        dict(i="bi:bi+32"),
        schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock,
    )

    # Add transient helper Array + a corresponding accessNode
    sdfg.add_transient("gpu_A_helper", ("i+128", ),
                       dace.uint32,
                       storage=dace.dtypes.StorageType.Register,
                       lifetime=dace.dtypes.AllocationLifetime.Scope)
    gpu_a_helper_acc = state.add_access("gpu_A_helper")

    # Add Edges & connectors
    state.add_edge(gpu_a_acc_read, None, gpu_map_entry, "IN_gpu_A", dace.Memlet("gpu_A[0:128]"))
    state.add_edge(gpu_map_entry, "OUT_gpu_A", tb_map_entry, "IN_gpu_A", dace.Memlet("gpu_A[0:bi+32]"))
    gpu_map_entry.add_in_connector("IN_gpu_A")
    gpu_map_entry.add_out_connector("OUT_gpu_A")
    tb_map_entry.add_in_connector("IN_gpu_A")

    # Weird copy, which triggers error if allocation happens to late during code generation
    state.add_edge(tb_map_entry, "OUT_gpu_A", gpu_a_helper_acc, None, dace.Memlet("gpu_A[0:i]"))
    tb_map_entry.add_out_connector("OUT_gpu_A")

    state.add_edge(gpu_a_helper_acc, None, tb_map_exit, "IN_1", dace.Memlet("gpu_A_helper[i]"))
    state.add_edge(tb_map_exit, "OUT_1", gpu_map_exit, "IN_1", dace.Memlet("gpu_A_helper[bi:bi+32]"))
    state.add_edge(gpu_map_exit, "OUT_1", gpu_a_acc_write, None, dace.Memlet("gpu_A[0:128]"))

    tb_map_exit.add_in_connector("IN_1")
    tb_map_exit.add_out_connector("OUT_1")
    gpu_map_exit.add_in_connector("IN_1")
    gpu_map_exit.add_out_connector("OUT_1")

    try:
        sdfg.compile()
    except CompilationError as e:
        pytest.fail(f"sdfg.compile() failed with: {e}")


if __name__ == '__main__':
    test_allocation_requiring_map_variables()
