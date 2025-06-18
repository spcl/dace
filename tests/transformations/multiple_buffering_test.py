import copy
import numpy as np
import dace
import pytest
import cupy as cp

from dace.transformation.dataflow.multiple_buffering import MultipleBuffering


def _add_shared_memory(sdfg: dace.SDFG, add_src_access_node: bool = False):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                next_map = None
                for n in state.bfs_nodes(node):
                    if isinstance(n, dace.sdfg.nodes.MapEntry) and n != node and n.map.schedule == dace.dtypes.ScheduleType.GPU_ThreadBlock:
                        next_map = n
                        break
                    elif isinstance(n, dace.nodes.MapExit):
                        break
                if next_map is None:
                    raise ValueError("No next map found for the GPU_Device map entry.")

                src_name_dst_name_offset = dict()
                edges_to_rm = set()
                for in_edge in state.in_edges(next_map):
                    if in_edge.data is not None:
                        in_arr_name = in_edge.data.data
                        copy_shape = [(0, (((e) - b)//s), 1) for b, e, s in in_edge.data.subset]
                        copied_shape = [(((e + 1) - b)//s) for b, e, s in in_edge.data.subset]
                        copy_offset = [b for b, _, _ in in_edge.data.subset]
                        shared_mem_name = "shr_" + in_arr_name
                        in_arr = sdfg.arrays[in_arr_name]
                        if shared_mem_name not in sdfg.arrays:
                            sdfg.add_array(shared_mem_name, copied_shape, in_arr.dtype, storage=dace.dtypes.StorageType.GPU_Shared, transient=True)

                        if add_src_access_node is True:
                            a1 = state.add_access(in_arr_name)
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(a1, None, a2, None, dace.Memlet(
                                data=in_arr_name,
                                subset=in_edge.data.subset,
                                other_subset=dace.subsets.Range(copy_shape),
                                wcr=None,
                            ))
                            e2 = state.add_edge(a2, None, next_map, in_edge.dst_conn,
                                                dace.Memlet.from_array(shared_mem_name,
                                                                    sdfg.arrays[shared_mem_name]))
                            e3 = state.add_edge(in_edge.src, in_edge.src_conn, a1, None,
                                                copy.deepcopy(in_edge.data))
                            edges_to_rm.add(in_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_offset)
                        else:
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(in_edge.src, in_edge.src_conn, a2, None, dace.Memlet(
                                data=in_arr_name,
                                subset=in_edge.data.subset,
                                other_subset=dace.subsets.Range(copy_shape),
                                wcr=None,
                            ))
                            e2 = state.add_edge(a2, None, next_map, in_edge.dst_conn,
                                                dace.Memlet.from_array(shared_mem_name,
                                                                    sdfg.arrays[shared_mem_name]))
                            edges_to_rm.add(in_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_offset)

                nodes = state.all_nodes_between(next_map, state.exit_node(next_map))
                for edge in state.all_edges(*nodes):
                    if edge.data is not None and edge.data.data in src_name_dst_name_offset:
                        dst_name, offset = src_name_dst_name_offset[edge.data.data]
                        edge.data.data = dst_name
                        old_subset = [(b,e,s) for b, e, s in edge.data.subset]
                        new_subset = [(b - offset[i], e - offset[i], s) for i, (b, e, s) in enumerate(old_subset)]
                        edge.data.subset = dace.subsets.Range(new_subset)

                for edge in edges_to_rm:
                    state.remove_edge(edge)

def test_standalone_execution():
    """Standalone test function that can be run without pytest."""

    # Setup
    dace.Config.set('cache', value='unique')

    # Create kernel
    N = dace.symbol("N", dtype=dace.int64)
    N_val = 1024

    @dace.program
    def kernel(
        A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        C: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
    ):
        for i in dace.map[0:N:512] @ dace.dtypes.ScheduleType.GPU_Device:
            for k in dace.map[0:2] @ dace.dtypes.ScheduleType.Sequential:
                for j in dace.map[0:256] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                    C[i + j + k * 256] = A[i + j + k * 256] + B[i + j + k * 256]

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify()
    original_sdfg.save("original_sdfg.sdfg")
    original_sdfg.validate()
    original_sdfg.validate()
    original_sdfg_w_shr_mem = copy.deepcopy(original_sdfg)
    _add_shared_memory(original_sdfg_w_shr_mem, add_src_access_node=False)
    original_sdfg_w_shr_mem.validate()

    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg2 = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = original_sdfg.name + "_double_buffered"
    transformed_sdfg2.name = original_sdfg.name + "_double_buffered_async"
    _add_shared_memory(transformed_sdfg, add_src_access_node=False)
    _add_shared_memory(transformed_sdfg2, add_src_access_node=False)

    # Apply transformations
    for state in transformed_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                #node.map.gpu_block_size = (256, 1, 1)  # Set GPU block size for the map entry
                # Apply multiple buffering transformation
                options_dict = {
                    "device_map_type":dace.dtypes.ScheduleType.GPU_Device,
                    "copy_src_type":dace.dtypes.StorageType.GPU_Global,
                    "copy_dst_type":dace.dtypes.StorageType.GPU_Shared,
                    "synchronous": True,
                }
                db_transform_can_be_applied = MultipleBuffering().can_be_applied_to(
                    sdfg=state.sdfg,
                    options=options_dict,
                    map_entry=node,
                    expr_index=state.node_id(node),
                    permissive=False,
                    where=node
                )
                assert db_transform_can_be_applied, f"MultipleBuffering transformation should be applicable to the map entry. Returned:{db_transform_can_be_applied}"

                MultipleBuffering().apply_to(
                    map_entry=node,
                    sdfg=transformed_sdfg,
                    options=options_dict,
                )
    for state in transformed_sdfg2.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                # Apply multiple buffering transformation
                options_dict = {
                    "device_map_type":dace.dtypes.ScheduleType.GPU_Device,
                    "copy_src_type":dace.dtypes.StorageType.GPU_Global,
                    "copy_dst_type":dace.dtypes.StorageType.GPU_Shared,
                    "synchronous": False,
                }
                db_transform_can_be_applied = MultipleBuffering().can_be_applied_to(
                    sdfg=state.sdfg,
                    options=options_dict,
                    map_entry=node,
                    expr_index=state.node_id(node),
                    permissive=False,
                    where=node
                )
                assert db_transform_can_be_applied, f"MultipleBuffering transformation should be applicable to the map entry. Returned:{db_transform_can_be_applied}"

                MultipleBuffering().apply_to(
                    map_entry=node,
                    sdfg=transformed_sdfg2,
                    options=options_dict,
                )

    # Validate SDFGs
    original_sdfg_w_shr_mem.validate()
    transformed_sdfg.validate()
    transformed_sdfg.save("double_buffered_sync.sdfg")
    transformed_sdfg2.save("double_buffered_async.sdfg")

    # Initialize data
    cp.random.seed(42)
    vals_A_orig = cp.fromfunction(lambda i, : i * (i + 2) / N_val, (N_val,), dtype=cp.float64)
    vals_B_orig = cp.fromfunction(lambda i, : i * (i + 3) / N_val, (N_val,), dtype=cp.float64)
    vals_C_orig = cp.fromfunction(lambda i, : i * 0 / N_val, (N_val,), dtype=cp.float64)

    vals_A_2 = vals_A_orig.copy()
    vals_B_2= vals_B_orig.copy()
    vals_C_2 = vals_C_orig.copy()

    # Execute SDFGs
    print("===RUN ORIGINAL SDFG WITH SHR MEM===")
    original_sdfg_w_shr_mem(A=vals_A_orig, B=vals_B_orig, C=vals_C_orig, N=N_val)
    print("====================================")
    print("==========RUN ORIGINAL SDFG=========")
    original_sdfg(A=vals_A_2, B=vals_B_2, C=vals_C_2, N=N_val)
    print("====================================")
    vals_C_close = cp.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)

    print(f"vals_C results match: {vals_C_close}")

    if vals_C_close:
        print("Test Fail: Naive ShrMem and Original SDFGs should preserve correctness")
    else:
        if not vals_C_close:
            print(f"vals_C max difference: {cp.max(cp.abs(vals_C_orig - vals_C_2))}")
            print(f"vals_C difference: {cp.abs(vals_C_orig - vals_C_2)}")
            print(f"vals_C orig: {vals_C_orig}")
            print(f"vals_C multiple buffered: {vals_C_2}")
            print("Sample of original values:", vals_C_orig[:64])
            print("Sample of buffered values:", vals_C_2[:64])
            print("Sample of differences:", cp.abs(vals_C_orig - vals_C_2)[:64])
    assert vals_C_close

    print("=====RUN MULTIPLE BUFFERED SDFG=====")
    transformed_sdfg(A=vals_A_2, B=vals_B_2, C=vals_C_2, N=N_val)
    print("====================================")

    # Check results
    vals_C_close = cp.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)

    print(f"vals_C results match: {vals_C_close}")

    if vals_C_close:
        print("Test Fail: Multiple Buffering transformations preserve correctness, but they should not be synchronized by the current codegen.")
    else:
        if not vals_C_close:
            print(f"vals_C max difference: {cp.max(cp.abs(vals_C_orig - vals_C_2))}")
            print(f"vals_C difference: {cp.abs(vals_C_orig - vals_C_2)}")
            print(f"vals_C orig: {vals_C_orig}")
            print(f"vals_C multiple buffered: {vals_C_2}")
            print("Sample of original values:", vals_C_orig[:64])
            print("Sample of buffered values:", vals_C_2[:64])
            print("Sample of differences:", cp.abs(vals_C_orig - vals_C_2)[:64])
    assert vals_C_close



if __name__ == "__main__":
    success = test_standalone_execution()
    exit(0 if success else 1)