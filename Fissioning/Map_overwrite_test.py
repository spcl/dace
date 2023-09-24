from dace.sdfg.nodes import AccessNode
from dace.sdfg.state import SDFGState
import pytest

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.transformation.passes.array_fission import ArrayFission
from dace.transformation.passes.analysis import FindAccessNodes
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.subsets import Subset, Range
from typing import List, Optional, Sequence, Set, Union, Dict, Tuple

N = dace.symbol("N")

def assert_rename_sets(expected: Dict[str, List[Set[AccessNode]]],
                       access_nodes:Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]] 
                       ):
    
    rename_dict: Dict[str, Dict[str,Set[AccessNode]]] = {}
    for original_name, state_dict in access_nodes.items():
        if original_name not in rename_dict.keys():
            rename_dict[original_name] = {}
        for state, (reads, writes) in state_dict.items():
            access_nodes = reads.union(writes)
            for access_node in access_nodes:
                if access_node.data not in rename_dict[original_name].keys():
                    rename_dict[original_name][access_node.data] = {access_node}
                else:
                    rename_dict[original_name][access_node.data].add(access_node)
            

    for original_name, set_list in expected.items():
        for name_set in set_list:
            assert(name_set in rename_dict[original_name].values())


def test_simple_map_overwrite():
    "simple loop that overwrites a one-dimensional array"
    sdfg = dace.SDFG('branch_subscope_fission')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [2], dace.int32)
    sdfg.add_array('B', [N], dace.int32, transient=True)
    sdfg.add_array('C', [N], dace.int32)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    merge_1 = sdfg.add_state('merge_1')
    guard_after = sdfg.add_state('guard_after')
    after = sdfg.add_state('after')
    merge_after = sdfg.add_state('merge_after')
    first_assign = dace.InterstateEdge(assignments={'i': 'A[0]'})
    sdfg.add_edge(init_state, guard_1, first_assign)
    right_cond = dace.InterstateEdge(condition='i <= 0')
    sdfg.add_edge(guard_1, merge_1, right_cond)
    sdfg.add_edge(merge_1, guard_after, dace.InterstateEdge())
    sdfg.add_edge(guard_after, after, dace.InterstateEdge())
    sdfg.add_edge(after, merge_after, dace.InterstateEdge())
    a1 = guard_1.add_access('B')
    guard_1.add_mapped_tasklet("overwrite_1",
                map_ranges = {
                    '_i': f'0:N:2'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_i]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    src = guard_after.add_access("A")
    dst = guard_after.add_access("B")
    guard_after.add_mapped_tasklet("overwrite_2",
                {
                    '_i': f'0:N:1'
                },
                {
                    "a": dace.Memlet("A[_i]")
                },
                f"b = a",
                {
                    "b": dace.Memlet("B[_i]")
                },
                input_nodes={"A": src},
                output_nodes={"B": dst},
                external_edges=True

                )
    guard_after.add_mapped_tasklet("incomplete_overwrite",
                {
                    '_i': f'0:N:2'
                },
                {
                    "a": dace.Memlet("A[_i]")
                },
                f"b = a",
                {
                    "b": dace.Memlet("B[_i]")
                },
                input_nodes={"A": src},
                output_nodes={"B": dst},
                external_edges=True

                )
    a13 = merge_1.add_access('B')
    t8 = merge_1.add_tasklet('t8', {'b'}, {'c'}, 'c = b + 1')
    a14 = merge_1.add_access('C')
    merge_1.add_edge(a13, None, t8, 'b', dace.Memlet('B[0]'))
    merge_1.add_edge(t8, 'c', a14, None, dace.Memlet('C[0]'))
    a10 = after.add_access('B')
    t6 = after.add_tasklet('t6', {'b'}, {'c'}, 'c = b * 3')
    a11 = after.add_access('C')
    after.add_edge(a10, None, t6, 'b', dace.Memlet('B[0]'))
    after.add_edge(t6, 'c', a11, None, dace.Memlet('C[0]'))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    # make sets of accessnodes that should have the same name after the transformation
    name_sets:Dict[str, List[Set[AccessNode]]] = {}
    name_sets["B"] = [{a1, a13}, {dst, a10}]
    name_sets["A"] = [{src}]
    name_sets["C"] = [{a14, a11}]
    try:
        sdfg.validate()
    except:
        assert(False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                           ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    
def test_simple_overwrite_2():
    "first map does not overwrite the array"
    sdfg = dace.SDFG('branch_subscope_fission')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [2], dace.int32)
    sdfg.add_array('B', [N], dace.int32, transient=True)
    sdfg.add_array('C', [N], dace.int32)
    sdfg.add_array("tmp", [N], dace.int32)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    merge_1 = sdfg.add_state('merge_1')
    guard_after = sdfg.add_state('guard_after')
    after = sdfg.add_state('after')
    merge_after = sdfg.add_state('merge_after')
    first_assign = dace.InterstateEdge(assignments={'i': 'A[0]'})
    sdfg.add_edge(init_state, guard_1, first_assign)
    right_cond = dace.InterstateEdge(condition='i <= 0')
    sdfg.add_edge(guard_1, merge_1, right_cond)
    sdfg.add_edge(merge_1, guard_after, dace.InterstateEdge())
    sdfg.add_edge(guard_after, after, dace.InterstateEdge())
    sdfg.add_edge(after, merge_after, dace.InterstateEdge())
    a1 = guard_1.add_access('B')
    guard_1.add_mapped_tasklet("overwrite_1",
                map_ranges = {
                    '_i': f'0:N:2'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_i]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    src = guard_after.add_access("A")
    dst = guard_after.add_access("B")
    guard_after.add_mapped_tasklet("overwrite_2",
                {
                    '_i': f'0:N:1'
                },
                {
                    "a": dace.Memlet("A[_i]")
                },
                f"b = a",
                {
                    "b": dace.Memlet("B[_i]")
                },
                input_nodes={"A": src},
                output_nodes={"B": dst},
                external_edges=True

                )
    guard_after.add_mapped_tasklet("incomplete_overwrite",
                {
                    '_i': f'0:N:2'
                },
                {
                    "a": dace.Memlet("A[_i]")
                },
                f"b = a",
                {
                    "b": dace.Memlet("B[_i]")
                },
                input_nodes={"A": src},
                output_nodes={"B": dst},
                external_edges=True

                )
    a20 = init_state.add_access("B")
    a30 = init_state.add_access("tmp")
    read_init = init_state.add_tasklet("read", {"b"},{"t"}, "t = b")
    init_state.add_edge(a20, None, read_init, "b", dace.Memlet("B[1]"))
    init_state.add_edge(read_init, "t", a30, "b", dace.Memlet("tmp[1]"))
    a13 = merge_1.add_access('B')
    t8 = merge_1.add_tasklet('t8', {'b'}, {'c'}, 'c = b + 1')
    a14 = merge_1.add_access('C')
    merge_1.add_edge(a13, None, t8, 'b', dace.Memlet('B[0]'))
    merge_1.add_edge(t8, 'c', a14, None, dace.Memlet('C[0]'))
    a10 = after.add_access('B')
    t6 = after.add_tasklet('t6', {'b'}, {'c'}, 'c = b * 3')
    a11 = after.add_access('C')
    after.add_edge(a10, None, t6, 'b', dace.Memlet('B[0]'))
    after.add_edge(t6, 'c', a11, None, dace.Memlet('C[0]'))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    # make sets of accessnodes that should have the same name after the transformation
    name_sets:Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [{src}]
    name_sets["B"] = [{a1, a13, a20}, {dst, a10}]
    name_sets["C"] = [{a14, a11}]
    name_sets["tmp"] = [{a30}]
    try:
        sdfg.validate()
    except:
        assert(False)

    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                           ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)

def test_intrastate_overwrite():
    "array is overwritten and read from sequentially in the same state"
    sdfg = dace.SDFG('intrastate_overwrite')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32, transient=True)
    sdfg.add_array('C', [N], dace.int32)
    sdfg.add_array("tmp", [N], dace.int32)
    init = sdfg.add_state("init")
    overwrite = sdfg.add_state("overwrite")
    sdfg.add_edge(init, overwrite, dace.InterstateEdge())
    B_read_0 = overwrite.add_read("B")
    A_read_1 = overwrite.add_access("A")
    B_read_1 = overwrite.add_read("B")
    A_write_0 = overwrite.add_write("A")
    _, map_1_entry, map_1_exit = overwrite.add_mapped_tasklet("overwrite_1",
                {
                    '_i': f'0:N:1'
                },
                {
                    "b": dace.Memlet("B[_i]")
                },
                f"a = b",
                {
                    "a": dace.Memlet("A[_i]")
                },
                input_nodes={"B": B_read_0},
                output_nodes={"A": A_read_1},
                external_edges=True
                )

    _, map_2_entry, map_2_exit = overwrite.add_mapped_tasklet("overwrite_2",
                {
                    '_i': f'0:N:1'
                },
                {
                    "a": dace.Memlet("A[_i]")
                },
                f"b = a",
                {
                    "b": dace.Memlet("B[_i]")
                },
                input_nodes={"A": A_read_1},
                output_nodes={"B": B_read_1},
                external_edges=True

                )

    _, map_3_entry, map_3_exit = overwrite.add_mapped_tasklet("incomplete_overwrite_2",
                {
                    '_i': f'0:N:1'
                },
                {
                    "b": dace.Memlet("B[_i]")
                },
                f"a = b",
                {
                    "a": dace.Memlet("A[_i]")
                },
                input_nodes={"B": B_read_1},
                output_nodes={"A": A_write_0},
                external_edges=True
                )
    A_read_0 = init.add_read("A")
    tmp_write_0 = init.add_write("tmp")
    tasklet_1 = init.add_tasklet("copy", {"a"},{"t"},"t = a")
    init.add_edge(A_read_0,None,tasklet_1,"a", dace.Memlet("A[0]"))
    init.add_edge(tasklet_1, "t", tmp_write_0, None, dace.Memlet("tmp[0]"))
    
    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert(False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                           ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]


if __name__ == '__main__':
    test_simple_map_overwrite()
    test_simple_overwrite_2()
    test_intrastate_overwrite()

