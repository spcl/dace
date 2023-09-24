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
                       access_nodes: Dict[str, Dict[SDFGState,
                                                    Tuple[Set[AccessNode], Set[AccessNode]]]]
                       ):
    rename_dict: Dict[str, Dict[str, Set[AccessNode]]] = {}
    for original_name, state_dict in access_nodes.items():
        if original_name not in rename_dict.keys():
            rename_dict[original_name] = {}
        for state, (reads, writes) in state_dict.items():
            access_nodes = reads.union(writes)
            for access_node in access_nodes:
                if access_node.data not in rename_dict[original_name].keys():
                    rename_dict[original_name][access_node.data] = set([
                                                                       access_node])
                else:
                    rename_dict[original_name][access_node.data].add(
                        access_node)

    for original_name, set_list in expected.items():
        for name_set in set_list:
            assert (name_set in rename_dict[original_name].values())


def test_simple_loop_overwrite():
    "simple loop that overwrites a one-dimensional array"

    sdfg = dace.SDFG("two_loops_overwrite")
    sdfg.add_symbol("i", dace.int32)
    sdfg.add_array("A", [N], dace.int64, transient=True)
    sdfg.add_array("tmp", [1], dace.int64, transient=True)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body_1 = sdfg.add_state("loop_body_1")
    loop_body_2 = sdfg.add_state("loop_body_2")
    _, _, after_state = sdfg.add_loop(
        init, loop_body_1, None, "i", "0", "i < N", "i + 1")
    _, _, _ = sdfg.add_loop(
        after_state, loop_body_2, end, "i", "0", "i < N", "i + 1")
    a0 = loop_body_1.add_access("A")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet_1, "a", a0, None, dace.Memlet("A[i]"))
    a1 = loop_body_2.add_access("A")
    loop_tasklet_2 = loop_body_2.add_tasklet("overwrite1", {}, {"a"}, "a = 0")
    loop_body_2.add_edge(loop_tasklet_2, "a", a1, None, dace.Memlet("A[i]"))
    a2 = after_state.add_access("A")
    tmp1 = after_state.add_read("tmp")
    tasklet_1 = after_state.add_tasklet("read", {"a"}, {"t"}, "t = a")
    after_state.add_edge(a2, None, tasklet_1, "a", dace.Memlet("A[0]"))
    after_state.add_edge(tasklet_1, "t", tmp1, None, dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [{a0, a2}, {a1}]
    name_sets["tmp"] = [{tmp1}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    try:
        sdfg.validate()
    except:
        assert (False)


def test_simple_loop_overwrite2():
    "simple loop that overwrites a one-dimensional array"

    sdfg = dace.SDFG("two_loops_overwrite")
    sdfg.add_array("A", [N], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    init = sdfg.add_state("init")
    loop_body_1 = sdfg.add_state("loop_body_1")
    loop_body_2 = sdfg.add_state("loop_body_2")
    loop_end = sdfg.add_state("loop_end")
    _, _, after_state = sdfg.add_loop(
        init, loop_body_1, None, "i", "0", "i < N", "i + 1", loop_end)
    sdfg.add_edge(loop_body_1, loop_body_2,
                  dace.InterstateEdge(condition="i % 2 == 0"))
    sdfg.add_edge(loop_body_1, loop_end, dace.InterstateEdge())
    sdfg.add_edge(loop_body_2, loop_end, dace.InterstateEdge())
    a3 = loop_body_2.add_access("A")
    loop_tasklet_2 = loop_body_2.add_tasklet("overwrite", {}, {"a"}, "a = 1")
    loop_body_2.add_edge(loop_tasklet_2, "a", a3, None, dace.Memlet("A[i]"))
    a1 = init.add_access("A")
    tmp_0 = init.add_access("res")
    init.add_edge(a1, None, tmp_0, None, dace.Memlet("A[0]"))
    a0 = loop_body_1.add_access("A")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet_1, "a", a0, None, dace.Memlet("A[i]"))
    a2 = after_state.add_access("A")
    tmp1 = after_state.add_read("res")
    tasklet_1 = after_state.add_tasklet("read", {"a"}, {"t"}, "t = a")
    after_state.add_edge(a2, None, tasklet_1, "a", dace.Memlet("A[0]"))
    after_state.add_edge(tasklet_1, "t", tmp1, None, dace.Memlet("res[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    # make sets of accessnodes that should have the same name after the transformation
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [set([a0, a3, a2]), set([a1])]
    tmp_set_1 = set([tmp1, tmp_0])
    name_sets["res"] = []
    name_sets["res"].append(tmp_set_1)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    try:
        sdfg.validate()
    except:
        assert (False)


def test_loop_read_no_fission():
    """Full write by loop that also reads from the overwritten variable --> no Fission"""

    sdfg = dace.SDFG('scalar_isedge')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [N], dace.int32, transient=True)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    after = sdfg.add_state("after")
    end_state = sdfg.add_state('end')
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1, dace.InterstateEdge(
        condition='i < (N - 1)'))
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp[0]'})
    sdfg.add_edge(loop_1_1, loop_1_2, tmp1_edge)
    sdfg.add_edge(loop_1_2, guard_1, dace.InterstateEdge(
        assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, after, dace.InterstateEdge(
        condition='i >= (N - 1)'))
    sdfg.add_edge(after, end_state, dace.InterstateEdge())
    overwrite_0 = init_state.add_tasklet('overwrite_0', None, {'a'}, 'a = 0')
    tmp_access_0 = init_state.add_access("tmp")
    init_state.add_edge(overwrite_0, 'a', tmp_access_0,
                        None, dace.Memlet('tmp'))
    B_access_0 = loop_1_1.add_access("B")
    tmp_access_1 = loop_1_1.add_access("tmp")
    loop_1_1.add_edge(B_access_0, None, tmp_access_1, None, dace.Memlet('B'))
    loop1_tasklet_1 = loop_1_2.add_tasklet(
        'loop1_1', {'ap', 't'}, {'a'}, 'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet(
        'loop1_2', {'bp', 't'}, {'b'}, 'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1,
                      't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2,
                      't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1,
                      'ap', dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2,
                      'bp', dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a,
                      None, dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b,
                      None, dace.Memlet('B[i]'))
    after_tasklet = after.add_tasklet("after", {"tmp"}, {"b"}, "b = tmp")
    after_read_tmp = after.add_read("tmp")
    after_write_b = after.add_write("B")
    after.add_edge(after_read_tmp, None, after_tasklet,
                   "tmp", dace.Memlet("tmp[0]"))
    after.add_edge(after_tasklet, "b", after_write_b,
                   None, dace.Memlet("B[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [set([loop1_read_a, loop1_write_a])]
    name_sets["tmp"] = [
        set([tmp_access_0, tmp_access_1, loop1_read_tmp, after_read_tmp])]
    name_sets["B"] = [
        set([B_access_0, loop1_read_b, loop1_write_b, after_write_b])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    try:
        sdfg.validate()
    except:
        assert (False)


def test_loop_no_phi_node():
    """loop that has no phi node at the guard (after minimal SSA)"""
    sdfg = dace.SDFG('loop_no_phi_node')
    sdfg.add_array('A', [N], dace.int32, transient=True)
    sdfg.add_array('B', [1], dace.int32)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    branch_overwrite_1 = sdfg.add_state("branch_overwrite_1")
    loop_body_1 = sdfg.add_state('loop_body_1')
    branch_overwrite_2 = sdfg.add_state("branch_overwrite_2")
    loop_body_2 = sdfg.add_state('loop_body_2')
    sdfg.add_edge(init, branch_overwrite_1,
                  dace.InterstateEdge(condition="A[0] > 0", assignments={'i': '0'}))
    sdfg.add_edge(branch_overwrite_1, loop_body_1,
                  dace.InterstateEdge(condition='i < N'))
    sdfg.add_edge(loop_body_1, branch_overwrite_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(branch_overwrite_1, end,
                  dace.InterstateEdge(condition="not (i < N)"))
    sdfg.add_edge(init, branch_overwrite_2,
                  dace.InterstateEdge(condition="A[0] < 0", assignments={'i': '0'}))
    sdfg.add_edge(branch_overwrite_2, loop_body_2,
                  dace.InterstateEdge(condition='i < N'))
    sdfg.add_edge(loop_body_2, branch_overwrite_2,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(branch_overwrite_2, end,
                  dace.InterstateEdge(condition="not (i < N)"))
    A_0 = init.add_access("A")
    assign_0 = init.add_tasklet('assign_0', None, {'a'}, 'a = 0')
    init.add_edge(assign_0, 'a', A_0, None, dace.Memlet('A'))
    A_1 = loop_body_1.add_access('A')
    assign_1 = loop_body_1.add_tasklet('assign_1', None, {'a'}, 'a = 1')
    loop_body_1.add_edge(assign_1, 'a', A_1, None, dace.Memlet('A[i]'))
    A_2 = loop_body_2.add_access('A')
    assign_2 = loop_body_2.add_tasklet('assign_2', None, {'a'}, 'a = 2')
    loop_body_2.add_edge(assign_2, 'a', A_2, None, dace.Memlet('A[i]'))
    A_3 = end.add_access('A')
    B_1 = end.add_access('B')
    end.add_edge(A_3, None, B_1, None, dace.Memlet('A[0]'))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [
        set([A_0]), set([A_1, A_2, A_3])]
    name_sets["B"] = [set([B_1])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    sdfg.validate()
    try:
        sdfg.validate()
    except:
        assert (False)


def test_loop_read_before_write_no_fission():
    """Full write by loop that reads from the overwritten variable before it's overwritten--> no Fission"""

    sdfg = dace.SDFG('scalar_isedge')
    sdfg.add_array('A', [N], dace.int32, transient=True)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [N], dace.int32)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    after = sdfg.add_state("after")
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1, dace.InterstateEdge(condition='i < N'))
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp[0]'})
    sdfg.add_edge(loop_1_1, loop_1_2, tmp1_edge)
    sdfg.add_edge(loop_1_2, guard_1, dace.InterstateEdge(
        assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, after, dace.InterstateEdge(condition='i >= N - 1'))
    overwrite_0 = init_state.add_tasklet('overwrite_0', None, {'a'}, 'a = 0')
    tmp_access_0 = init_state.add_access("tmp")
    init_state.add_edge(overwrite_0, 'a', tmp_access_0,
                        None, dace.Memlet('tmp'))
    A_access_0 = loop_1_1.add_access("A")
    B_access_0 = loop_1_1.add_access("B")
    tmp_access_1 = loop_1_1.add_access("tmp")
    loop_1_1.add_mapped_tasklet("overwrite_loop",
                                {
                                    '_i': f'0:N'
                                },
                                {
                                    "a": dace.Memlet("A[_i]"),
                                    "b": dace.Memlet("B[_i]")
                                },
                                f"out = a*5",
                                {
                                    "out": dace.Memlet("tmp[_i]")
                                },
                                external_edges=True,
                                input_nodes={"A": A_access_0, "B": B_access_0},
                                output_nodes={"tmp": tmp_access_1}
                                )
    loop1_tasklet_1 = loop_1_2.add_tasklet(
        'loop1_1', {'t'}, {'a'}, 'a = ap + 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_write_a = loop_1_2.add_write('A')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1,
                      't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a,
                      None, dace.Memlet('A[i]'))
    after_read_A = after.add_read("A")
    after_write_tmp = after.add_write("tmp")
    after.add_edge(after_read_A, None, after_write_tmp,
                   None, dace.Memlet("A[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [
        set([A_access_0, loop1_write_a, after_read_A])]
    name_sets["tmp"] = [set(
        [tmp_access_0, tmp_access_1, loop1_read_tmp, after_write_tmp])]
    name_sets["B"] = [set([B_access_0])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    sdfg.validate()
    try:
        sdfg.validate()
    except:
        assert (False)


def test_loop_read_before_write_interstate():
    """Full write by loop that also reads from the overwritten variable in an interstate edge --> no Fission"""

    sdfg = dace.SDFG('scalar_isedge')
    sdfg.add_array('A', [N], dace.int32, transient=True)
    sdfg.add_array('tmp', [N], dace.int32, transient=True)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    loop_1_guard = sdfg.add_state("loop_1_guard")
    after = sdfg.add_state("after")
    end_state = sdfg.add_state('end')
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_guard,
                  dace.InterstateEdge(condition='i < N'))
    sdfg.add_edge(loop_1_guard, loop_1_1,
                  dace.InterstateEdge(condition='A[0] > 0'))
    sdfg.add_edge(loop_1_guard, loop_1_2,
                  dace.InterstateEdge(condition='not (A[0] > 0)'))
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    sdfg.add_edge(loop_1_2, guard_1, dace.InterstateEdge(
        assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, after, dace.InterstateEdge(condition='i >= N'))
    sdfg.add_edge(after, end_state, dace.InterstateEdge())
    tmp_access_0 = init_state.add_access("tmp")
    A_access_init = init_state.add_access("A")
    init_state.add_edge(A_access_init, None, tmp_access_0,
                        None, dace.Memlet('A'))
    loop1_tasklet_1 = loop_1_2.add_tasklet(
        'loop1_1', {'t'}, {'a'}, 'a = ap + 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_write_a = loop_1_2.add_write('A')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1,
                      't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a,
                      None, dace.Memlet('A[i]'))
    after_tasklet = after.add_tasklet("after", {"a"}, {"tmp"}, "tmp = a")
    after_read_A = after.add_read("A")
    after_write_tmp = after.add_write("tmp")
    after.add_edge(after_read_A, None, after_tasklet, "a", dace.Memlet("A[0]"))
    after.add_edge(after_tasklet, "tmp", after_write_tmp,
                   None, dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})
    sdfg.view()
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [set([loop1_write_a, after_read_A, A_access_init])]
    name_sets["tmp"] = [set([tmp_access_0, loop1_read_tmp, after_write_tmp])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    try:
        sdfg.validate()
    except:
        assert (False)


if __name__ == '__main__':
    test_loop_read_no_fission()
    test_simple_loop_overwrite()
    test_simple_loop_overwrite2()
    test_loop_no_phi_node()
    test_loop_read_before_write_no_fission()
    test_loop_read_before_write_interstate()
