# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU reduce expansion falls back to ``pure``, which must still be device code.

``ExpandReduceGPUAuto`` declines a node carrying no identity and hands it to ``ExpandReducePure``,
whose maps carry the default schedule. Schedule inference has already run by the time a library node
expands, so those maps reach codegen as host loops over ``GPU_Global`` memory (npbench nbody).
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS
from dace.libraries.standard.nodes.reduce import ExpandReducePure, Reduce

N = dace.symbol("N")


def gpu_reduce_without_identity() -> dace.SDFG:
    """Row sums of a device array, by a node whose identity nobody set."""
    sdfg = dace.SDFG("gpu_reduce_no_identity")
    sdfg.add_array("A", [8, 256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("out", [8], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Reduce("reduce_sum", wcr="lambda a, b: a + b", axes=[1], identity=None)
    node.implementation = "GPUAuto"
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_in", dace.Memlet("A[0:8, 0:256]"))
    state.add_edge(node, "_out", state.add_access("out"), None, dace.Memlet("out[0:8]"))
    node.add_in_connector("_in")
    node.add_out_connector("_out")
    sdfg.validate()
    return sdfg


def test_the_pure_fallback_is_scheduled_as_a_kernel():
    sdfg = gpu_reduce_without_identity()
    sdfg.expand_library_nodes()

    schedules = {
        node.map.label: (node.map.schedule, state.entry_node(node) is None)
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    }
    assert schedules, "the fallback emitted no map at all"
    outer = {label for label, (_, top) in schedules.items() if top}
    assert outer, "the fallback emitted no outermost map to launch"
    for label, (schedule, top) in schedules.items():
        expected = dtypes.ScheduleType.GPU_Device if top else dtypes.ScheduleType.Sequential
        assert schedule is expected, f"map {label} is {schedule.name}, not {expected.name}"
    sdfg.validate()


def gpu_product_without_identity(length) -> dace.SDFG:
    """``out[0] *= prod(A)``, tsvc s312's shape: the product accumulates onto whatever ``out`` holds."""
    sdfg = dace.SDFG("gpu_product_no_identity")
    sdfg.add_array("A", [length], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("out", [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Reduce("reduce_product", wcr="lambda a, b: a * b", axes=None, identity=None)
    node.implementation = "GPUAuto"
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_in", dace.Memlet(f"A[0:{length}]"))
    state.add_edge(node, "_out", state.add_access("out"), None, dace.Memlet("out[0]"))
    node.add_in_connector("_in")
    node.add_out_connector("_out")
    sdfg.validate()
    return sdfg


def test_a_whole_array_reduction_launches_a_bounded_grid():
    """One block commits one atomic, and a product's atomic is a CAS loop: the grid has to stay bounded
    however long the array is, with each thread striding over its share (s312: 2.5M blocks, 8.5 s)."""
    sdfg = gpu_product_without_identity(N)
    sdfg.expand_library_nodes()
    maps = {node.map.schedule: node.map for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)}
    assert set(maps) == {
        dtypes.ScheduleType.GPU_Device,
        dtypes.ScheduleType.GPU_ThreadBlock,
        dtypes.ScheduleType.Sequential,
    }, maps
    blocks = maps[dtypes.ScheduleType.GPU_Device].range.num_elements()
    assert blocks.subs(N, 322727372) == ExpandReducePure.GRID_STRIDE_BLOCKS, blocks
    assert blocks.subs(N, 1000) == 4, "a short array must not launch idle blocks"
    assert maps[dtypes.ScheduleType.GPU_ThreadBlock].range.num_elements() == BLOCK_COLLECTIVE_THREADS
    stride = maps[dtypes.ScheduleType.Sequential].range[0][2]
    assert (stride - blocks * BLOCK_COLLECTIVE_THREADS).simplify() == 0, "the threads must stride by the whole grid"
    sdfg.validate()


@pytest.mark.gpu
def test_a_whole_array_product_accumulates_onto_the_output():
    import cupy

    length = 3 * ExpandReducePure.GRID_STRIDE_BLOCKS * BLOCK_COLLECTIVE_THREADS + 7
    values = np.random.default_rng(20261006).uniform(0.9999, 1.0001, size=length)
    out = cupy.full(1, 2.0)
    gpu_product_without_identity(length)(A=cupy.asarray(values), out=out)
    np.testing.assert_allclose(cupy.asnumpy(out)[0], 2.0 * np.prod(values), rtol=1e-10)


if __name__ == "__main__":
    test_the_pure_fallback_is_scheduled_as_a_kernel()
    test_a_whole_array_reduction_launches_a_bounded_grid()
    test_a_whole_array_product_accumulates_onto_the_output()
