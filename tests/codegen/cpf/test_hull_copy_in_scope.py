# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A host-to-device copy inside a sequential host map writes into the map exit, not an access node: the hull
copy selector must refuse it instead of reading the exit as an access node (fv3_dycore's CUDA render)."""

import dace
from dace.libraries.standard.nodes.copy import CopyLibraryNode, select_copy_implementation


def test_host_to_device_copy_inside_map_is_not_hull_copy():
    n, planes = 12, 4
    sdfg = dace.SDFG("hull_copy_in_scope")
    sdfg.add_array("H", (n, n), dace.float64, storage=dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("D", (planes, n, n), dace.float64, storage=dace.dtypes.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map("planes", {"k": f"0:{planes}"}, schedule=dace.dtypes.ScheduleType.Sequential)
    node = CopyLibraryNode(name="cp_in")
    state.add_node(node)
    interior = f"1:{n - 1}, 1:{n - 1}"
    state.add_memlet_path(
        state.add_read("H"),
        entry,
        node,
        dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
        memlet=dace.Memlet(f"H[{interior}]"),
    )
    state.add_memlet_path(
        node,
        exit_node,
        state.add_write("D"),
        src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
        memlet=dace.Memlet(f"D[k, {interior}]"),
    )
    with dace.config.set_temporary("compiler", "cuda", "backend", value="cuda"):
        assert select_copy_implementation(node, state) != "MemcpyCUDAHull"


if __name__ == "__main__":
    test_host_to_device_copy_inside_map_is_not_hull_copy()
