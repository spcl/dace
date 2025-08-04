import copy
import dace
import dace.sdfg.utils as sdutils
import pytest


def _add_shared_memory(sdfg: dace.SDFG, add_src_access_node: bool = False):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                next_map = None
                for n in state.bfs_nodes(node):
                    if isinstance(n, dace.sdfg.nodes.MapEntry
                                  ) and n != node and n.map.schedule == dace.dtypes.ScheduleType.GPU_ThreadBlock:
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
                        copy_shape = [(0, (((e) - b) // s), 1) for b, e, s in in_edge.data.subset]
                        copied_shape = [(((e + 1) - b) // s) for b, e, s in in_edge.data.subset]
                        copy_offset = [b for b, _, _ in in_edge.data.subset]
                        shared_mem_name = "shr_" + in_arr_name
                        in_arr = sdfg.arrays[in_arr_name]
                        if shared_mem_name not in sdfg.arrays:
                            sdfg.add_array(shared_mem_name,
                                           copied_shape,
                                           in_arr.dtype,
                                           storage=dace.dtypes.StorageType.GPU_Shared,
                                           transient=True)

                        if add_src_access_node is True:
                            a1 = state.add_access(in_arr_name)
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(
                                a1, None, a2, None,
                                dace.Memlet(
                                    data=in_arr_name,
                                    subset=in_edge.data.subset,
                                    other_subset=dace.subsets.Range(copy_shape),
                                    wcr=None,
                                ))
                            e2 = state.add_edge(a2, None, next_map, in_edge.dst_conn,
                                                dace.Memlet.from_array(shared_mem_name, sdfg.arrays[shared_mem_name]))
                            e3 = state.add_edge(in_edge.src, in_edge.src_conn, a1, None, copy.deepcopy(in_edge.data))
                            edges_to_rm.add(in_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_offset)
                        else:
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(
                                in_edge.src, in_edge.src_conn, a2, None,
                                dace.Memlet(
                                    data=in_arr_name,
                                    subset=in_edge.data.subset,
                                    other_subset=dace.subsets.Range(copy_shape),
                                    wcr=None,
                                ))
                            e2 = state.add_edge(a2, None, next_map, in_edge.dst_conn,
                                                dace.Memlet.from_array(shared_mem_name, sdfg.arrays[shared_mem_name]))
                            edges_to_rm.add(in_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_offset)

                nodes = state.all_nodes_between(next_map, state.exit_node(next_map))
                for edge in state.all_edges(*nodes):
                    if edge.data is not None and edge.data.data in src_name_dst_name_offset:
                        dst_name, offset = src_name_dst_name_offset[edge.data.data]
                        edge.data.data = dst_name
                        old_subset = [(b, e, s) for b, e, s in edge.data.subset]
                        new_subset = [(b - offset[i], e - offset[i], s) for i, (b, e, s) in enumerate(old_subset)]
                        edge.data.subset = dace.subsets.Range(new_subset)

                for edge in edges_to_rm:
                    state.remove_edge(edge)


def _check_map_entries(state, include_symbols_for_offset_calculation, const_only, schedule, expected_data,
                       expected_symbols):
    map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry) and n.map.schedule == schedule]
    for me in map_entries:
        if const_only:
            const_data = sdutils.get_constant_data(scope=me, parent_state=state)
            const_symbols = sdutils.get_constant_symbols(
                scope=me,
                parent_state=state,
                include_symbols_for_offset_calculations=include_symbols_for_offset_calculation)
            assert expected_data == const_data, f"(Const Data) Expected {expected_data}, got {const_data} in map {me.label}"
            assert expected_symbols == const_symbols, f"(Const Symbols) Expected {expected_symbols}, got {const_symbols} in map {me.label}"
        else:
            used_data = sdutils.get_used_data(scope=me, parent_state=state)
            used_symbols = sdutils.get_used_symbols(
                scope=me,
                parent_state=state,
                include_symbols_for_offset_calculations=include_symbols_for_offset_calculation)
            assert expected_data == used_data, f"(Used Data) Expected {expected_data}, got {used_data} in map {me.label}"
            assert expected_symbols == used_symbols, f"(Used Symbols) Expected {expected_symbols}, got {used_symbols} in map {me.label}"


def _gen_sdfg_with_symbol_use_in_nsdfg(write_only: bool = True) -> dace.SDFG:
    sdfg = dace.SDFG(name="reassign_syms_in_nested_sdfg")
    sdfg.add_array(name="A", shape=(1, ), dtype=dace.int64, transient=False)
    sdfg.add_symbol(name="A_sym", stype=dace.int64)

    s0 = sdfg.add_state(label="state0", is_start_block=True)
    s1 = sdfg.add_state(label="state1")

    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"A_sym": "A[0]"}))

    inner_sdfg = dace.SDFG(name="inner_sdfg")
    A_sym = dace.symbolic.symbol("A_sym", dace.int64)
    nsdfg = s1.add_nested_sdfg(
        sdfg=inner_sdfg,
        inputs={},
        outputs={},
        symbol_mapping={"A_sym": A_sym},
    )
    assert "A_sym" in nsdfg.sdfg.symbols
    assert "A_sym" in nsdfg.sdfg.free_symbols
    if write_only:
        nsdfg.sdfg.add_symbol(name="_inner_sym", stype=dace.int64)

    s1_0 = inner_sdfg.add_state(label="i_state0", is_start_block=True)
    s1_1 = inner_sdfg.add_state(label="i_state1")
    s1_2 = inner_sdfg.add_state(label="i_state2")

    if write_only:
        inner_sdfg.add_edge(s1_0, s1_1, dace.InterstateEdge(assignments={"_inner_sym": "A_sym + 1"}))
        inner_sdfg.add_edge(s1_1, s1_2, dace.InterstateEdge(assignments={"A_sym": "_inner_sym"}))
    else:
        inner_sdfg.add_edge(s1_0, s1_1, dace.InterstateEdge(assignments={"A_sym": "5"}))
        inner_sdfg.add_edge(s1_1, s1_2, dace.InterstateEdge(assignments={}))

    s2: dace.SDFGState = sdfg.add_state(label="state2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    t0 = s2.add_tasklet(
        name="tasklet",
        inputs={"_in_A"},
        outputs={},
        code="printf(\"%ld\\n\", _in_A); printf(\"%ld\\n\", A_sym);",
        language=dace.Language.CPP,
        side_effects=True,
        code_global="#include <stdio.h>\n",
    )
    an0 = s2.add_access(array_or_stream_name="A")
    s2.add_edge(an0, None, t0, "_in_A", dace.Memlet(expr="A[0]"))
    return sdfg, s1, nsdfg


def test_const_utilities_case_non_const_input_not_present_in_output():
    """Standalone test function that can be run without pytest."""

    # Create kernel
    N = dace.symbol("N", dtype=dace.int64)
    K = 5

    @dace.program
    def kernel(
        A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        C: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
    ):
        for i in dace.map[0:N:256 * K] @ dace.dtypes.ScheduleType.GPU_Device:
            for k in dace.map[0:K] @ dace.dtypes.ScheduleType.Sequential:
                for j in dace.map[0:256] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                    C[i + j + k * 256] = A[i + j + k * 256] + B[i + j + k * 256]

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify()

    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    _add_shared_memory(transformed_sdfg, add_src_access_node=True)
    transformed_sdfg.validate()

    # Test cases
    original_state = next(iter(original_sdfg.all_states()))
    transformed_state = next(iter(transformed_sdfg.all_states()))
    assert original_state is not None
    assert transformed_state is not None

    all_data_names = set(node.data for node in original_state.data_nodes())
    transformed_sdfg_tmp_names = set(node.data for node in transformed_state.data_nodes()
                                     if transformed_sdfg.arrays[node.data].transient)
    original_sdfg_tmp_names = set(node.data for node in original_state.data_nodes()
                                  if original_sdfg.arrays[node.data].transient)

    # Original state tests
    _check_map_entries(original_state, True, False, dace.dtypes.ScheduleType.GPU_Device, all_data_names - {"C"},
                       {"i", "N"})
    _check_map_entries(original_state, True, False, dace.dtypes.ScheduleType.Sequential, all_data_names - {"C"},
                       {"i", "k", "N"})
    _check_map_entries(original_state, True, False, dace.dtypes.ScheduleType.GPU_ThreadBlock, all_data_names - {"C"},
                       {"i", "j", "k", "N"})

    # Transformed state tests
    _check_map_entries(transformed_state, True, False, dace.dtypes.ScheduleType.GPU_Device,
                       all_data_names - {"C"} | {"shr_A", "shr_B"}, {"i", "N"})
    _check_map_entries(transformed_state, True, False, dace.dtypes.ScheduleType.Sequential,
                       all_data_names - {"C"} | {"shr_A", "shr_B"}, {"i", "k", "N"})
    # Using only shr_a and shr_b means no need of N
    _check_map_entries(transformed_state, True, False, dace.dtypes.ScheduleType.GPU_ThreadBlock,
                       {"shr_A", "shr_B"} | transformed_sdfg_tmp_names, {"i", "j", "k"})

    # Original state tests
    _check_map_entries(original_state, True, True, dace.dtypes.ScheduleType.GPU_Device, {"A", "B"}, {"i", "N"})
    _check_map_entries(original_state, True, True, dace.dtypes.ScheduleType.Sequential, {"A", "B"}, {"i", "k", "N"})
    _check_map_entries(original_state, True, True, dace.dtypes.ScheduleType.GPU_ThreadBlock, {"A", "B"},
                       {"i", "j", "k", "N"})

    # Transformed state tests
    _check_map_entries(transformed_state, True, True, dace.dtypes.ScheduleType.GPU_Device, set(), {"i", "N"})
    _check_map_entries(transformed_state, True, True, dace.dtypes.ScheduleType.Sequential, set(), {"i", "k", "N"})
    # Using only shr_a and shr_b means no need of N
    _check_map_entries(transformed_state, True, True, dace.dtypes.ScheduleType.GPU_ThreadBlock, {"shr_A", "shr_B"},
                       {"i", "j", "k"})

    # Original state tests
    _check_map_entries(original_state, False, True, dace.dtypes.ScheduleType.GPU_Device, {"A", "B"}, {"i"})
    _check_map_entries(original_state, False, True, dace.dtypes.ScheduleType.Sequential, {"A", "B"}, {"i", "k"})
    _check_map_entries(original_state, False, True, dace.dtypes.ScheduleType.GPU_ThreadBlock, {"A", "B"},
                       {"i", "j", "k"})

    # Transformed state tests
    _check_map_entries(transformed_state, False, True, dace.dtypes.ScheduleType.GPU_Device, set(), {"i"})
    _check_map_entries(transformed_state, False, True, dace.dtypes.ScheduleType.Sequential, set(), {"i", "k"})
    # Using only shr_a and shr_b means no need of N
    _check_map_entries(transformed_state, False, True, dace.dtypes.ScheduleType.GPU_ThreadBlock, {"shr_A", "shr_B"},
                       {"i", "j", "k"})


def test_const_utilities_case_write_only_free_symbol_in_nsdfg():
    sdfg1, s1, nsdfg1 = _gen_sdfg_with_symbol_use_in_nsdfg(write_only=True)
    sdfg1.validate()

    const_data = sdutils.get_constant_data(nsdfg1)
    const_symbols = sdutils.get_constant_symbols(nsdfg1)
    assert set() == const_data
    assert set() == const_symbols

    sdfg2, s2, nsdfg2 = _gen_sdfg_with_symbol_use_in_nsdfg(write_only=False)
    sdfg2.validate()
    const_data = sdutils.get_constant_data(nsdfg2)
    const_symbols = sdutils.get_constant_symbols(nsdfg2)
    assert set() == const_data
    assert set() == const_symbols


if __name__ == "__main__":
    test_const_utilities_case_non_const_input_not_present_in_output()
    test_const_utilities_case_write_only_free_symbol_in_nsdfg()
