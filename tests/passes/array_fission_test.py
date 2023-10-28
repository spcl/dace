from dace.sdfg.nodes import AccessNode
from dace.sdfg.state import SDFGState
import pytest

import dace
import numpy as np
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.transformation.passes.array_fission import ArrayFission
from dace.transformation.passes.analysis import FindAccessNodes
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.subsets import Subset, Range
from typing import List, Optional, Sequence, Set, Union, Dict, Tuple


def assert_rename_sets(expected: Dict[str, List[Set[AccessNode]]],
                       access_nodes: Dict[str, Dict[SDFGState,
                                                    Tuple[Set[AccessNode],
                                                          Set[AccessNode]]]]):
    rename_dict: Dict[str, Dict[str, Set[AccessNode]]] = {}
    for original_name, state_dict in access_nodes.items():
        if original_name not in rename_dict.keys():
            rename_dict[original_name] = {}
        for state, (reads, writes) in state_dict.items():
            access_nodes = reads.union(writes)
            for access_node in access_nodes:
                if access_node.data not in rename_dict[original_name].keys():
                    rename_dict[original_name][access_node.data] = set(
                        [access_node])
                else:
                    rename_dict[original_name][access_node.data].add(
                        access_node)

    for original_name, set_list in expected.items():
        for name_set in set_list:
            assert (name_set in rename_dict[original_name].values())


def test_simple_conditional_write2():
    "two conditional writes that overwrite a one-dimensional array"

    @dace.program
    def conditional_write(A: dace.float64[1], b: dace.bool):
        tmp = np.float64(0.0)
        if b:
            tmp = 1
        else:
            tmp = 2
        A = tmp

    sdfg = dace.SDFG("simple_conditional_nofission")
    sdfg.add_array("tmp", [1], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    sdfg.add_array("A", [1], dace.int64)
    init = sdfg.add_state("init")
    if_write = sdfg.add_state("if_write")
    else_write = sdfg.add_state("else_write")
    merge_1 = sdfg.add_state("merge_1")
    sdfg.add_edge(init, if_write, dace.InterstateEdge(condition="A[0]"))
    sdfg.add_edge(if_write, merge_1, dace.InterstateEdge())
    sdfg.add_edge(init, else_write, dace.InterstateEdge(condition="not A[0]"))
    sdfg.add_edge(else_write, merge_1, dace.InterstateEdge())
    tmp0 = init.add_access("tmp")
    t0 = init.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    init.add_edge(t0, "a", tmp0, None, dace.Memlet("tmp[0]"))
    tmp1 = if_write.add_access("tmp")
    t1 = if_write.add_tasklet("overwrite", {}, {"a"}, "a = 1")
    if_write.add_edge(t1, "a", tmp1, None, dace.Memlet("tmp[0]"))
    tmp2 = else_write.add_access("tmp")
    t2 = else_write.add_tasklet("overwrite", {}, {"a"}, "a = 2")
    else_write.add_edge(t2, "a", tmp2, None, dace.Memlet("tmp[0]"))
    tmp4 = merge_1.add_access("tmp")
    res0 = merge_1.add_access("res")
    merge_1.add_edge(tmp4, None, res0, None, dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["tmp"] = [set([tmp1, tmp2, tmp4, tmp0])]
    name_sets["res"] = [set([res0])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    try:
        sdfg.validate()
    except:
        assert (False)


def test_simple_conditional_write_no_fission():
    """two conditional writes that overwrite a one-dimensional array
    but the array is read before the second conditional assignment --> no fission"""
    N = dace.symbol("N")

    @dace.program
    def conditional_write(A: dace.float64[N]):
        tmp = np.zeros_like(A)
        if (A[0]):
            tmp[:] = 1
        A[:] = tmp
        if (not A[0]):
            tmp[:] = 2
        A[:] = tmp

    sdfg = dace.SDFG("simple_conditional_nofission")
    sdfg.add_array("tmp", [1], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    sdfg.add_array("A", [1], dace.int64)
    init = sdfg.add_state("init")
    if_write = sdfg.add_state("if_write")
    merge_0 = sdfg.add_state("merge_0")
    else_write = sdfg.add_state("else_write")
    merge_1 = sdfg.add_state("merge_1")
    sdfg.add_edge(init, if_write, dace.InterstateEdge(condition="A[0]"))
    sdfg.add_edge(if_write, merge_0, dace.InterstateEdge(condition="not A[0]"))
    sdfg.add_edge(init, merge_0, dace.InterstateEdge())
    sdfg.add_edge(merge_0, else_write,
                  dace.InterstateEdge(condition="not A[0]"))
    sdfg.add_edge(merge_0, merge_1, dace.InterstateEdge(condition="A[0]"))
    sdfg.add_edge(else_write, merge_1, dace.InterstateEdge())
    tmp0 = init.add_access("tmp")
    t0 = init.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    init.add_edge(t0, "a", tmp0, None, dace.Memlet("tmp[0]"))
    tmp1 = if_write.add_access("tmp")
    t1 = if_write.add_tasklet("overwrite", {}, {"a"}, "a = 1")
    if_write.add_edge(t1, "a", tmp1, None, dace.Memlet("tmp[0]"))
    tmp2 = else_write.add_access("tmp")
    t2 = else_write.add_tasklet("overwrite", {}, {"a"}, "a = 2")
    else_write.add_edge(t2, "a", tmp2, None, dace.Memlet("tmp[0]"))
    tmp4 = merge_1.add_access("tmp")
    res0 = merge_1.add_access("res")
    merge_1.add_edge(tmp4, None, res0, None, dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["tmp"] = [set([tmp0, tmp1, tmp2, tmp4])]
    name_sets["res"] = [set([res0])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    try:
        sdfg.validate()
    except:
        assert (False)
    assert_rename_sets(name_sets, access_nodes)


def test_multiple_conditions():
    "three conditional writes that overwrite a one-dimensional array"

    @dace.program
    def conditional_write_multiple_conditions(A: dace.float64[1],
                                              b1: dace.bool, b2: dace.bool):
        tmp = np.zeros_like(A)
        if (b1 and b2):
            tmp[:] = 1
        elif (not b1 and b2):
            tmp[:] = 2
        else:
            tmp[:] = 3
        A[:] = tmp

    sdfg = dace.SDFG("multiple_conditions")
    sdfg.add_array("tmp", [2], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    sdfg.add_array("A", [2], dace.int64)
    init = sdfg.add_state("init")
    if_write = sdfg.add_state("if_write")
    elif_write = sdfg.add_state("elif_write")
    else_write = sdfg.add_state("else_write")
    guard_0 = sdfg.add_state("guard_0")
    merge_0 = sdfg.add_state("merge_0")
    merge_1 = sdfg.add_state("merge_1")
    sdfg.add_edge(init, guard_0,
                  dace.InterstateEdge(condition="not (A[0] and A[1])"))
    sdfg.add_edge(guard_0, elif_write,
                  dace.InterstateEdge(condition="not A[0] and A[1]"))
    sdfg.add_edge(guard_0, else_write,
                  dace.InterstateEdge(condition="not A[0] and A[1]"))
    sdfg.add_edge(elif_write, merge_0, dace.InterstateEdge())
    sdfg.add_edge(else_write, merge_0, dace.InterstateEdge())
    sdfg.add_edge(merge_0, merge_1, dace.InterstateEdge())
    sdfg.add_edge(if_write, merge_1, dace.InterstateEdge())
    sdfg.add_edge(init, if_write,
                  dace.InterstateEdge(condition="A[0] and A[1]"))
    tmp0 = init.add_access("tmp")
    t0 = init.add_tasklet("overwrite", {}, {"a"}, "a = 0")
    init.add_edge(t0, "a", tmp0, None, dace.Memlet("tmp"))
    tmp1 = if_write.add_access("tmp")
    t1 = if_write.add_tasklet("overwrite", {}, {"a"}, "a = 1")
    if_write.add_edge(t1, "a", tmp1, None, dace.Memlet("tmp"))
    tmp2 = elif_write.add_access("tmp")
    t2 = elif_write.add_tasklet("overwrite", {}, {"a"}, "a = 2")
    elif_write.add_edge(t2, "a", tmp2, None, dace.Memlet("tmp"))
    tmp3 = else_write.add_access("tmp")
    t3 = else_write.add_tasklet("overwrite", {}, {"a"}, "a = 3")
    else_write.add_edge(t3, "a", tmp3, None, dace.Memlet("tmp"))
    tmp4 = merge_1.add_access("tmp")
    res0 = merge_1.add_access("res")
    merge_1.add_edge(tmp4, None, res0, None, dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["tmp"] = [set([tmp1, tmp2, tmp3, tmp4]), set([tmp0])]
    name_sets["res"] = [set([res0])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    try:
        sdfg.validate()
    except:
        assert (False)
    assert_rename_sets(name_sets, access_nodes)


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
    _, _, after_state = sdfg.add_loop(init, loop_body_1, None, "i", "0",
                                      "i < N", "i + 1")
    _, _, _ = sdfg.add_loop(after_state, loop_body_2, end, "i", "0", "i < N",
                            "i + 1")
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
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
    _, _, after_state = sdfg.add_loop(init, loop_body_1, None, "i", "0",
                                      "i < N", "i + 1", loop_end)
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
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
    sdfg.add_edge(guard_1, loop_1_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp[0]'})
    sdfg.add_edge(loop_1_1, loop_1_2, tmp1_edge)
    sdfg.add_edge(loop_1_2, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, after,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    sdfg.add_edge(after, end_state, dace.InterstateEdge())
    overwrite_0 = init_state.add_tasklet('overwrite_0', None, {'a'}, 'a = 0')
    tmp_access_0 = init_state.add_access("tmp")
    init_state.add_edge(overwrite_0, 'a', tmp_access_0, None,
                        dace.Memlet('tmp'))
    B_access_0 = loop_1_1.add_access("B")
    tmp_access_1 = loop_1_1.add_access("tmp")
    loop_1_1.add_edge(B_access_0, None, tmp_access_1, None, dace.Memlet('B'))
    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'ap', 't'}, {'a'},
                                           'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet('loop1_2', {'bp', 't'}, {'b'},
                                           'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't',
                      dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't',
                      dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap',
                      dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp',
                      dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None,
                      dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None,
                      dace.Memlet('B[i]'))
    after_tasklet = after.add_tasklet("after", {"tmp"}, {"b"}, "b = tmp")
    after_read_tmp = after.add_read("tmp")
    after_write_b = after.add_write("B")
    after.add_edge(after_read_tmp, None, after_tasklet, "tmp",
                   dace.Memlet("tmp[0]"))
    after.add_edge(after_tasklet, "b", after_write_b, None,
                   dace.Memlet("B[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [set([loop1_read_a, loop1_write_a])]
    name_sets["tmp"] = [
        set([tmp_access_0, tmp_access_1, loop1_read_tmp, after_read_tmp])
    ]
    name_sets["B"] = [
        set([B_access_0, loop1_read_b, loop1_write_b, after_write_b])
    ]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
    sdfg.add_edge(
        init, branch_overwrite_1,
        dace.InterstateEdge(condition="A[0] > 0", assignments={'i': '0'}))
    sdfg.add_edge(branch_overwrite_1, loop_body_1,
                  dace.InterstateEdge(condition='i < N'))
    sdfg.add_edge(loop_body_1, branch_overwrite_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(branch_overwrite_1, end,
                  dace.InterstateEdge(condition="not (i < N)"))
    sdfg.add_edge(
        init, branch_overwrite_2,
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
    name_sets["A"] = [set([A_0]), set([A_1, A_2, A_3])]
    name_sets["B"] = [set([B_1])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
    sdfg.add_edge(loop_1_2, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, after, dace.InterstateEdge(condition='i >= N'))
    overwrite_0 = init_state.add_tasklet('overwrite_0', None, {'a'}, 'a = 0')
    A_access_init = init_state.add_access('A')
    init_state.add_edge(overwrite_0, 'a', A_access_init, None,
                        dace.Memlet('A'))
    A_access_0 = loop_1_1.add_access("A")
    B_access_0 = loop_1_1.add_access("B")
    tmp_access_1 = loop_1_1.add_access("tmp")
    loop_1_1.add_mapped_tasklet("overwrite_loop", {'_i': f'0:N'}, {
        "a": dace.Memlet("A[_i]"),
        "b": dace.Memlet("B[_i]")
    },
                                f"out = a*5", {"out": dace.Memlet("tmp[_i]")},
                                external_edges=True,
                                input_nodes={
                                    "A": A_access_0,
                                    "B": B_access_0
                                },
                                output_nodes={"tmp": tmp_access_1})
    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'t'}, {'a'},
                                           'a = ap + 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_write_a = loop_1_2.add_write('A')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't',
                      dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None,
                      dace.Memlet('A[i]'))
    after_read_A = after.add_read("A")
    after_write_tmp = after.add_write("tmp")
    after.add_edge(after_read_A, None, after_write_tmp, None,
                   dace.Memlet("A[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [
        set([A_access_0, loop1_write_a, after_read_A, A_access_init])
    ]
    name_sets["tmp"] = [set([tmp_access_1, loop1_read_tmp, after_write_tmp])]
    name_sets["B"] = [set([B_access_0])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]

    assert_rename_sets(name_sets, access_nodes)
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
    sdfg.add_edge(loop_1_2, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, after, dace.InterstateEdge(condition='i >= N'))
    sdfg.add_edge(after, end_state, dace.InterstateEdge())
    tmp_access_0 = init_state.add_access("tmp")
    A_access_init = init_state.add_access("A")
    init_state.add_edge(A_access_init, None, tmp_access_0, None,
                        dace.Memlet('A'))
    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'t'}, {'a'},
                                           'a = ap + 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_write_a = loop_1_2.add_write('A')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't',
                      dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None,
                      dace.Memlet('A[i]'))
    after_tasklet = after.add_tasklet("after", {"a"}, {"tmp"}, "tmp = a")
    after_read_A = after.add_read("A")
    after_write_tmp = after.add_write("tmp")
    after.add_edge(after_read_A, None, after_tasklet, "a", dace.Memlet("A[0]"))
    after.add_edge(after_tasklet, "tmp", after_write_tmp, None,
                   dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [set([loop1_write_a, after_read_A, A_access_init])]
    name_sets["tmp"] = [set([tmp_access_0, loop1_read_tmp, after_write_tmp])]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    try:
        sdfg.validate()
    except:
        assert (False)


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
                               map_ranges={'_i': f'0:N:2'},
                               inputs={},
                               code=f"b = 5",
                               outputs={"b": dace.Memlet("B[_i]")},
                               output_nodes={"B": a1},
                               external_edges=True)
    src = guard_after.add_access("A")
    dst = guard_after.add_access("B")
    guard_after.add_mapped_tasklet("overwrite_2", {'_i': f'0:N:1'},
                                   {"a": dace.Memlet("A[_i]")},
                                   f"b = a", {"b": dace.Memlet("B[_i]")},
                                   input_nodes={"A": src},
                                   output_nodes={"B": dst},
                                   external_edges=True)
    guard_after.add_mapped_tasklet("incomplete_overwrite", {'_i': f'0:N:2'},
                                   {"a": dace.Memlet("A[_i]")},
                                   f"b = a", {"b": dace.Memlet("B[_i]")},
                                   input_nodes={"A": src},
                                   output_nodes={"B": dst},
                                   external_edges=True)
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
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["B"] = [{a1, a13}, {dst, a10}]
    name_sets["A"] = [{src}]
    name_sets["C"] = [{a14, a11}]
    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
                               map_ranges={'_i': f'0:N:2'},
                               inputs={},
                               code=f"b = 5",
                               outputs={"b": dace.Memlet("B[_i]")},
                               output_nodes={"B": a1},
                               external_edges=True)
    src = guard_after.add_access("A")
    dst = guard_after.add_access("B")
    guard_after.add_mapped_tasklet("overwrite_2", {'_i': f'0:N:1'},
                                   {"a": dace.Memlet("A[_i]")},
                                   f"b = a", {"b": dace.Memlet("B[_i]")},
                                   input_nodes={"A": src},
                                   output_nodes={"B": dst},
                                   external_edges=True)
    guard_after.add_mapped_tasklet("incomplete_overwrite", {'_i': f'0:N:2'},
                                   {"a": dace.Memlet("A[_i]")},
                                   f"b = a", {"b": dace.Memlet("B[_i]")},
                                   input_nodes={"A": src},
                                   output_nodes={"B": dst},
                                   external_edges=True)
    a20 = init_state.add_access("B")
    a30 = init_state.add_access("tmp")
    read_init = init_state.add_tasklet("read", {"b"}, {"t"}, "t = b")
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
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets["A"] = [{src}]
    name_sets["B"] = [{a1, a13, a20}, {dst, a10}]
    name_sets["C"] = [{a14, a11}]
    name_sets["tmp"] = [{a30}]
    try:
        sdfg.validate()
    except:
        assert (False)

    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
    _, map_1_entry, map_1_exit = overwrite.add_mapped_tasklet(
        "overwrite_1", {'_i': f'0:N:1'}, {"b": dace.Memlet("B[_i]")},
        f"a = b", {"a": dace.Memlet("A[_i]")},
        input_nodes={"B": B_read_0},
        output_nodes={"A": A_read_1},
        external_edges=True)

    _, map_2_entry, map_2_exit = overwrite.add_mapped_tasklet(
        "overwrite_2", {'_i': f'0:N:1'}, {"a": dace.Memlet("A[_i]")},
        f"b = a", {"b": dace.Memlet("B[_i]")},
        input_nodes={"A": A_read_1},
        output_nodes={"B": B_read_1},
        external_edges=True)

    _, map_3_entry, map_3_exit = overwrite.add_mapped_tasklet(
        "incomplete_overwrite_2", {'_i': f'0:N:1'},
        {"b": dace.Memlet("B[_i]")},
        f"a = b", {"a": dace.Memlet("A[_i]")},
        input_nodes={"B": B_read_1},
        output_nodes={"A": A_write_0},
        external_edges=True)
    A_read_0 = init.add_read("A")
    tmp_write_0 = init.add_write("tmp")
    tasklet_1 = init.add_tasklet("copy", {"a"}, {"t"}, "t = a")
    init.add_edge(A_read_0, None, tasklet_1, "a", dace.Memlet("A[0]"))
    init.add_edge(tasklet_1, "t", tmp_write_0, None, dace.Memlet("tmp[0]"))

    result = Pipeline([ArrayFission()]).apply_pass(sdfg, {})
    assert (False)

    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = result[FindAccessNodes.__name__][sdfg.sdfg_id]


def test_scalar_write_shadow_split():
    """
    Test the scalar write shadow scopes pass with writes dominating reads across state.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_split')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [2], dace.int32, transient=True)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    intermediate = sdfg.add_state('intermediate')
    end = sdfg.add_state("end")
    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None,
                        dace.Memlet('tmp'))
    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'},
                                        'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None,
                      dace.Memlet('tmp'))
    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'ap', 't'}, {'a'},
                                           'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet('loop1_2', {'bp', 't'}, {'b'},
                                           'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't',
                      dace.Memlet('tmp'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't',
                      dace.Memlet('tmp'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap',
                      dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp',
                      dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None,
                      dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None,
                      dace.Memlet('B[i]'))
    tmp_1 = intermediate.add_access("tmp")
    intermediate_tasklet = intermediate.add_tasklet("overwrite", {}, {"t"},
                                                    "t = 0")
    intermediate.add_edge(intermediate_tasklet, "t", tmp_1, None,
                          dace.Memlet("tmp"))
    tmp_2 = end.add_access("tmp")
    B_0 = end.add_access("B")
    endtasklet = end.add_tasklet("read", {"t"}, {"b"}, "b = t")
    end.add_edge(tmp_2, None, endtasklet, "t", dace.Memlet("tmp"))
    end.add_edge(endtasklet, "b", B_0, None, dace.Memlet("B[0]"))
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    sdfg.add_edge(loop_1_2, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    sdfg.add_edge(intermediate, end, dace.InterstateEdge())

    results = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{a1_read, loop1_read_a, loop1_write_a}]
    name_sets['B'] = [{b1_read, loop1_read_b, loop1_write_b, B_0}]
    name_sets["tmp"] = [{init_write}, {tmp1_write, loop1_read_tmp},
                        {tmp_1, tmp_2}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_scalar_write_shadow_fused():
    """
    Test the scalar write shadow scopes pass with writes dominating reads in the same state.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_fused')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [2], dace.int32, transient=True)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1 = sdfg.add_state('loop_1')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2 = sdfg.add_state('loop_2')
    end_state = sdfg.add_state('end')
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_1, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    sdfg.add_edge(intermediate, guard_2,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_2, guard_2,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None,
                        dace.Memlet('tmp'))
    tmp1_tasklet = loop_1.add_tasklet('tmp1', {'a', 'b'}, {'out'},
                                      'out = a * b')
    loop1_tasklet_1 = loop_1.add_tasklet('loop1_1', {'ap', 't'}, {'a'},
                                         'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1.add_tasklet('loop1_2', {'bp', 't'}, {'b'},
                                         'b = bp - 2 * t')
    tmp1_read_write = loop_1.add_access('tmp')
    a1_read = loop_1.add_read('A')
    b1_read = loop_1.add_read('B')
    a1_write = loop_1.add_write('A')
    b1_write = loop_1.add_write('B')
    loop_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1.add_edge(tmp1_tasklet, 'out', tmp1_read_write, None,
                    dace.Memlet('tmp'))
    loop_1.add_edge(tmp1_read_write, None, loop1_tasklet_1, 't',
                    dace.Memlet('tmp'))
    loop_1.add_edge(tmp1_read_write, None, loop1_tasklet_2, 't',
                    dace.Memlet('tmp'))
    loop_1.add_edge(a1_read, None, loop1_tasklet_1, 'ap',
                    dace.Memlet('A[i + 1]'))
    loop_1.add_edge(b1_read, None, loop1_tasklet_2, 'bp',
                    dace.Memlet('B[i + 1]'))
    loop_1.add_edge(loop1_tasklet_1, 'a', a1_write, None, dace.Memlet('A[i]'))
    loop_1.add_edge(loop1_tasklet_2, 'b', b1_write, None, dace.Memlet('B[i]'))
    tmp2_tasklet = loop_2.add_tasklet('tmp2', {'a', 'b'}, {'out'},
                                      'out = a / b')
    loop2_tasklet_1 = loop_2.add_tasklet('loop2_1', {'ap', 't'}, {'a'},
                                         'a = ap + t * t')
    loop2_tasklet_2 = loop_2.add_tasklet('loop2_2', {'bp', 't'}, {'b'},
                                         'b = bp - t * t')
    tmp2_read_write = loop_2.add_access('tmp')
    a2_read = loop_2.add_read('A')
    b2_read = loop_2.add_read('B')
    a2_write = loop_2.add_write('A')
    b2_write = loop_2.add_write('B')
    loop_2.add_edge(a2_read, None, tmp2_tasklet, 'a', dace.Memlet('A[i + 1]'))
    loop_2.add_edge(b2_read, None, tmp2_tasklet, 'b', dace.Memlet('B[i + 1]'))
    loop_2.add_edge(tmp2_tasklet, 'out', tmp2_read_write, None,
                    dace.Memlet('tmp'))
    loop_2.add_edge(tmp2_read_write, None, loop2_tasklet_1, 't',
                    dace.Memlet('tmp'))
    loop_2.add_edge(tmp2_read_write, None, loop2_tasklet_2, 't',
                    dace.Memlet('tmp'))
    loop_2.add_edge(a2_read, None, loop2_tasklet_1, 'ap', dace.Memlet('A[i]'))
    loop_2.add_edge(b2_read, None, loop2_tasklet_2, 'bp', dace.Memlet('B[i]'))
    loop_2.add_edge(loop2_tasklet_1, 'a', a2_write, None,
                    dace.Memlet('A[i + 1]'))
    loop_2.add_edge(loop2_tasklet_2, 'b', b2_write, None,
                    dace.Memlet('B[i + 1]'))

    results = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{a1_read, a1_write, a2_read, a2_write}]
    name_sets['B'] = [{b1_read, b1_write, b2_read, b2_write}]
    name_sets["tmp"] = [{init_write}, {tmp1_read_write}, {tmp2_read_write}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_scalar_write_shadow_interstate_self():
    """
    Tests the scalar write shadow pass with interstate edge reads being shadowed by the state they're originating from.
    """
    # TODO: test the interstate-edges somehow
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_isedge')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [2], dace.int32, transient=True)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2_1 = sdfg.add_state('loop_2_1')
    loop_2_2 = sdfg.add_state('loop_2_2')
    end_state = sdfg.add_state('end')
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_1_1, loop_1_2, tmp1_edge)
    sdfg.add_edge(loop_1_2, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    sdfg.add_edge(intermediate, guard_2,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    tmp2_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_2_1, loop_2_2, tmp2_edge)
    sdfg.add_edge(loop_2_2, guard_2,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    sdfg.add_edge(init_state, guard_2, dace.InterstateEdge())
    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None,
                        dace.Memlet('tmp'))
    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'},
                                        'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None,
                      dace.Memlet('tmp'))
    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'ap', 't'}, {'a'},
                                           'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet('loop1_2', {'bp', 't'}, {'b'},
                                           'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't',
                      dace.Memlet('tmp'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't',
                      dace.Memlet('tmp'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap',
                      dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp',
                      dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None,
                      dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None,
                      dace.Memlet('B[i]'))
    tmp2_tasklet = loop_2_1.add_tasklet('tmp2', {'a', 'b'}, {'out'},
                                        'out = a / b')
    tmp2_write = loop_2_1.add_write('tmp')
    a2_read = loop_2_1.add_read('A')
    b2_read = loop_2_1.add_read('B')
    loop_2_1.add_edge(a2_read, None, tmp2_tasklet, 'a',
                      dace.Memlet('A[i + 1]'))
    loop_2_1.add_edge(b2_read, None, tmp2_tasklet, 'b',
                      dace.Memlet('B[i + 1]'))
    loop_2_1.add_edge(tmp2_tasklet, 'out', tmp2_write, None,
                      dace.Memlet('tmp'))
    loop2_tasklet_1 = loop_2_2.add_tasklet('loop2_1', {'ap', 't'}, {'a'},
                                           'a = ap + t * t')
    loop2_tasklet_2 = loop_2_2.add_tasklet('loop2_2', {'bp', 't'}, {'b'},
                                           'b = bp - t * t')
    loop2_read_tmp = loop_2_2.add_read('tmp')
    loop2_read_a = loop_2_2.add_read('A')
    loop2_read_b = loop_2_2.add_read('B')
    loop2_write_a = loop_2_2.add_write('A')
    loop2_write_b = loop_2_2.add_write('B')
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_1, 't',
                      dace.Memlet('tmp'))
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_2, 't',
                      dace.Memlet('tmp'))
    loop_2_2.add_edge(loop2_read_a, None, loop2_tasklet_1, 'ap',
                      dace.Memlet('A[i]'))
    loop_2_2.add_edge(loop2_read_b, None, loop2_tasklet_2, 'bp',
                      dace.Memlet('B[i]'))
    loop_2_2.add_edge(loop2_tasklet_1, 'a', loop2_write_a, None,
                      dace.Memlet('A[i + 1]'))
    loop_2_2.add_edge(loop2_tasklet_2, 'b', loop2_write_b, None,
                      dace.Memlet('B[i + 1]'))

    results = Pipeline([ArrayFission()]).apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{
        a1_read, a2_read, loop1_read_a, loop1_write_a, loop2_read_a,
        loop2_write_a
    }]
    name_sets['B'] = [{
        b1_read, b2_read, loop1_read_b, loop1_write_b, loop2_read_b,
        loop2_write_b
    }]
    name_sets["tmp"] = [{init_write}, {tmp1_write, loop1_read_tmp},
                        {tmp2_write, loop2_read_tmp}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)
    assert (tmp1_write.data in tmp1_edge.assignments.values())
    assert (tmp2_write.data in tmp2_edge.assignments.values())


def test_scalar_write_shadow_interstate_pred():
    """
    Tests the scalar write shadow pass with interstate edge reads being shadowed by a predecessor state.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_isedge')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [2], dace.int32, transient=True)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    loop_1_3 = sdfg.add_state('loop_1_3')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2_1 = sdfg.add_state('loop_2_1')
    loop_2_2 = sdfg.add_state('loop_2_2')
    loop_2_3 = sdfg.add_state('loop_2_3')
    end_state = sdfg.add_state('end')
    sdfg.add_edge(init_state, guard_1,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_1_2, loop_1_3, tmp1_edge)
    sdfg.add_edge(loop_1_3, guard_1,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    sdfg.add_edge(intermediate, guard_2,
                  dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2_1,
                  dace.InterstateEdge(condition='i < (N - 1)'))
    sdfg.add_edge(loop_2_1, loop_2_2, dace.InterstateEdge())
    tmp2_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_2_2, loop_2_3, tmp2_edge)
    sdfg.add_edge(loop_2_3, guard_2,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state,
                  dace.InterstateEdge(condition='i >= (N - 1)'))
    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None,
                        dace.Memlet('tmp[0]'))
    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'},
                                        'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None,
                      dace.Memlet('tmp'))
    loop1_tasklet_1 = loop_1_3.add_tasklet('loop1_1', {'ap', 't'}, {'a'},
                                           'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_3.add_tasklet('loop1_2', {'bp', 't'}, {'b'},
                                           'b = bp - 2 * t')
    loop1_read_tmp = loop_1_3.add_read('tmp')
    loop1_read_a = loop_1_3.add_read('A')
    loop1_read_b = loop_1_3.add_read('B')
    loop1_write_a = loop_1_3.add_write('A')
    loop1_write_b = loop_1_3.add_write('B')
    loop_1_3.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't',
                      dace.Memlet('tmp'))
    loop_1_3.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't',
                      dace.Memlet('tmp'))
    loop_1_3.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap',
                      dace.Memlet('A[i + 1]'))
    loop_1_3.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp',
                      dace.Memlet('B[i + 1]'))
    loop_1_3.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None,
                      dace.Memlet('A[i]'))
    loop_1_3.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None,
                      dace.Memlet('B[i]'))
    tmp2_tasklet = loop_2_1.add_tasklet('tmp2', {'a', 'b'}, {'out'},
                                        'out = a / b')
    tmp2_write = loop_2_1.add_write('tmp')
    a2_read = loop_2_1.add_read('A')
    b2_read = loop_2_1.add_read('B')
    loop_2_1.add_edge(a2_read, None, tmp2_tasklet, 'a',
                      dace.Memlet('A[i + 1]'))
    loop_2_1.add_edge(b2_read, None, tmp2_tasklet, 'b',
                      dace.Memlet('B[i + 1]'))
    loop_2_1.add_edge(tmp2_tasklet, 'out', tmp2_write, None,
                      dace.Memlet('tmp'))
    loop2_tasklet_1 = loop_2_3.add_tasklet('loop2_1', {'ap', 't'}, {'a'},
                                           'a = ap + t * t')
    loop2_tasklet_2 = loop_2_3.add_tasklet('loop2_2', {'bp', 't'}, {'b'},
                                           'b = bp - t * t')
    loop2_read_tmp = loop_2_3.add_read('tmp')
    loop2_read_a = loop_2_3.add_read('A')
    loop2_read_b = loop_2_3.add_read('B')
    loop2_write_a = loop_2_3.add_write('A')
    loop2_write_b = loop_2_3.add_write('B')
    loop_2_3.add_edge(loop2_read_tmp, None, loop2_tasklet_1, 't',
                      dace.Memlet('tmp'))
    loop_2_3.add_edge(loop2_read_tmp, None, loop2_tasklet_2, 't',
                      dace.Memlet('tmp'))
    loop_2_3.add_edge(loop2_read_a, None, loop2_tasklet_1, 'ap',
                      dace.Memlet('A[i]'))
    loop_2_3.add_edge(loop2_read_b, None, loop2_tasklet_2, 'bp',
                      dace.Memlet('B[i]'))
    loop_2_3.add_edge(loop2_tasklet_1, 'a', loop2_write_a, None,
                      dace.Memlet('A[i + 1]'))
    loop_2_3.add_edge(loop2_tasklet_2, 'b', loop2_write_b, None,
                      dace.Memlet('B[i + 1]'))

    name_sets: Dict[str, List[Set[AccessNode]]] = {}

    name_sets['A'] = [{
        a1_read, a2_read, loop1_read_a, loop1_write_a, loop2_read_a,
        loop2_write_a
    }]
    name_sets['B'] = [{
        b1_read, b2_read, loop1_read_b, loop1_write_b, loop2_read_b,
        loop2_write_b
    }]
    name_sets["tmp"] = [{init_write}, {tmp1_write, loop1_read_tmp},
                        {tmp2_write, loop2_read_tmp}]

    # Test the pass.
    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_loop_fake_shadow_symbolic():
    """variable is overwritten in loop with symbolic range and read afterwards so the last
    definition in the loop should write to the same variable as the first write"""
    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array('A', [2], dace.float64, transient=True)
    sdfg.add_array('B', [2], dace.float64, transient=True)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')
    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A'))
    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_1 = loop.add_tasklet('loop_1', {}, {'a'}, 'a = 1')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_tasklet_1, 'a', loop_access, None, dace.Memlet('A'))
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B'))
    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None,
                   dace.Memlet('B'))
    end_access = end.add_access('A')
    end_access_b = end.add_access('B')
    end_tasklet = end.add_tasklet('end', {'a'}, {'b'}, 'b = a')
    end.add_edge(end_access, None, end_tasklet, 'a', dace.Memlet('A'))
    end.add_edge(end_tasklet, 'b', end_access_b, None, dace.Memlet('B'))
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < N'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= N'))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{init_access, end_access, loop2_access}, {loop_access}]
    name_sets["B"] = [{loop_access_b}, {loop2_access_b}, {end_access_b}]
    assert_rename_sets(name_sets, access_nodes)


def test_loop_fake_shadow():
    """variable is overwritten in loop with symbolic range and read afterwards so the last
    definition in the loop should write to the same variable as the first write"""

    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_array('A', [2], dace.float64, transient=True)
    sdfg.add_array('B', [2], dace.float64, transient=True)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')
    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A'))
    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_1 = loop.add_tasklet('loop_1', {}, {'a'}, 'a = 1')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_tasklet_1, 'a', loop_access, None, dace.Memlet('A'))
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B'))
    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None,
                   dace.Memlet('B'))
    end_access = end.add_access('A')
    end_access_b = end.add_access('B')
    end_tasklet = end.add_tasklet('end', {'a'}, {'b'}, 'b = a')
    end.add_edge(end_access, None, end_tasklet, 'a', dace.Memlet('A'))
    end.add_edge(end_tasklet, 'b', end_access_b, None, dace.Memlet('B'))
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{init_access}, {loop_access}, {loop2_access, end_access}]
    name_sets["B"] = [{loop_access_b}, {loop2_access_b}, {end_access_b}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_loop_fake_complex_shadow():
    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_array('A', [2], dace.float64, transient=True)
    sdfg.add_array('B', [2], dace.float64, transient=True)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')
    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A'))
    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B'))
    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None,
                   dace.Memlet('B'))
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{init_access, loop_access, loop2_access}]
    name_sets["B"] = [{loop_access_b}, {loop2_access_b}]
    assert_rename_sets(name_sets, access_nodes)


def test_loop_real_shadow():
    sdfg = dace.SDFG('loop_fake_shadow')
    sdfg.add_array('A', [2], dace.float64, transient=True)
    sdfg.add_array('B', [2], dace.float64, transient=True)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    loop = sdfg.add_state('loop')
    loop2 = sdfg.add_state('loop2')
    end = sdfg.add_state('end')
    init_access = init.add_access('A')
    init_tasklet = init.add_tasklet('init', {}, {'a'}, 'a = 0')
    init.add_edge(init_tasklet, 'a', init_access, None, dace.Memlet('A'))
    loop_access = loop.add_access('A')
    loop_access_b = loop.add_access('B')
    loop_tasklet_1 = loop.add_tasklet('loop_1', {}, {'a'}, 'a = 1')
    loop_tasklet_2 = loop.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop.add_edge(loop_tasklet_1, 'a', loop_access, None, dace.Memlet('A'))
    loop.add_edge(loop_access, None, loop_tasklet_2, 'a', dace.Memlet('A'))
    loop.add_edge(loop_tasklet_2, 'b', loop_access_b, None, dace.Memlet('B'))
    loop2_access = loop2.add_access('A')
    loop2_access_b = loop2.add_access('B')
    loop2_tasklet_1 = loop2.add_tasklet('loop_1', {}, {'a'}, 'a = 2')
    loop2_tasklet_2 = loop2.add_tasklet('loop_2', {'a'}, {'b'}, 'b = a')
    loop2.add_edge(loop2_tasklet_1, 'a', loop2_access, None, dace.Memlet('A'))
    loop2.add_edge(loop2_access, None, loop2_tasklet_2, 'a', dace.Memlet('A'))
    loop2.add_edge(loop2_tasklet_2, 'b', loop2_access_b, None,
                   dace.Memlet('B'))
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(loop, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, guard,
                  dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{init_access}, {loop_access}, {loop2_access}]
    name_sets["B"] = [{loop_access_b}, {loop2_access_b}]
    assert_rename_sets(name_sets, access_nodes)


def test_dominationless_write_branch1():
    sdfg = dace.SDFG('dominationless_write_branch')
    sdfg.add_array('A', [2], dace.float64, transient=True)
    sdfg.add_array('B', [2], dace.float64, transient=True)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    left = sdfg.add_state('left')
    merge = sdfg.add_state('merge')
    init_a = init.add_access('A')
    init_b = init.add_access('B')
    init_t1 = init.add_tasklet('init_1', {}, {'a'}, 'a = 0')
    init_t2 = init.add_tasklet('init_1', {'a'}, {'b'}, 'b = a + 1')
    init.add_edge(init_t1, 'a', init_a, None, dace.Memlet('A'))
    init.add_edge(init_a, None, init_t2, 'a', dace.Memlet('A'))
    init.add_edge(init_t2, 'b', init_b, None, dace.Memlet('B'))
    guard_a = guard.add_access('A')
    guard_t1 = guard.add_tasklet('guard_1', {}, {'a'}, 'a = 1')
    guard.add_edge(guard_t1, 'a', guard_a, None, dace.Memlet('A'))
    left_a = left.add_access('A')
    left_t1 = left.add_tasklet('left_1', {}, {'a'}, 'a = 2')
    left.add_edge(left_t1, 'a', left_a, None, dace.Memlet('A'))
    merge_a = merge.add_access('A')
    merge_b = merge.add_access('B')
    merge_t1 = merge.add_tasklet('merge_1', {'a'}, {'b'}, 'b = a + 1')
    merge.add_edge(merge_a, None, merge_t1, 'a', dace.Memlet('A'))
    merge.add_edge(merge_t1, 'b', merge_b, None, dace.Memlet('B'))
    sdfg.add_edge(init, guard, dace.InterstateEdge())
    sdfg.add_edge(guard, left, dace.InterstateEdge(condition='B[0] < 10'))
    sdfg.add_edge(guard, merge, dace.InterstateEdge(condition='B[0] >= 10'))
    sdfg.add_edge(left, merge, dace.InterstateEdge())

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{init_a}, {guard_a, left_a, merge_a}]
    name_sets["B"] = [{init_b}, {merge_b}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_dominationless_write_branch2():
    sdfg = dace.SDFG('dominationless_write_branch')
    sdfg.add_array('A', [2], dace.float64, transient=True)
    sdfg.add_array('B', [2], dace.float64, transient=True)
    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    left = sdfg.add_state('left')
    merge = sdfg.add_state('merge')
    init_a = init.add_access('A')
    init_b = init.add_access('B')
    init_t1 = init.add_tasklet('init_1', {}, {'a'}, 'a = 0')
    init_t2 = init.add_tasklet('init_1', {'a'}, {'b'}, 'b = a + 1')
    init.add_edge(init_t1, 'a', init_a, None, dace.Memlet('A'))
    init.add_edge(init_a, None, init_t2, 'a', dace.Memlet('A'))
    init.add_edge(init_t2, 'b', init_b, None, dace.Memlet('B'))
    guard_a = guard.add_access('A')
    guard_t1 = guard.add_tasklet('guard_1', {}, {'a'}, 'a = 1')
    guard.add_edge(guard_t1, 'a', guard_a, None, dace.Memlet('A'))
    left_a = left.add_access('A')
    left_t1 = left.add_tasklet('left_1', {}, {'a'}, 'a = 2')
    left.add_edge(left_t1, 'a', left_a, None, dace.Memlet('A'))
    merge_a = merge.add_access('A')
    merge_a_1 = merge.add_access('A')
    merge_t1 = merge.add_tasklet('merge_1', {'a'}, {'b'}, 'b = a + 1')
    merge.add_edge(merge_a, None, merge_t1, 'a', dace.Memlet('A'))
    merge.add_edge(merge_t1, 'b', merge_a_1, None, dace.Memlet('A'))
    sdfg.add_edge(init, guard, dace.InterstateEdge())
    sdfg.add_edge(guard, left, dace.InterstateEdge(condition='B[0] < 10'))
    sdfg.add_edge(guard, merge, dace.InterstateEdge(condition='B[0] >= 10'))
    sdfg.add_edge(left, merge, dace.InterstateEdge())

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{init_a}, {guard_a, left_a, merge_a}, {merge_a_1}]
    name_sets["B"] = [{init_b}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_nested_loops():
    """three nested loops that overwrite two dimensional array. 
    the innermost loop is surrounded by a loop that doesn't iterate over array range but over an empty constant range.
    Therefore the loop nest as a whole does not overwrite the array"""
    sdfg = dace.SDFG("nested_loops")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("i", dace.int32)
    sdfg.add_symbol("j", dace.int32)
    sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [2], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body_0 = sdfg.add_state("loop_body_0")
    loop_body_1 = sdfg.add_state("loop_body_1")
    guard_0 = sdfg.add_state("guard_0")
    guard_1 = sdfg.add_state("guard_1")
    guard_2 = sdfg.add_state("guard_2")
    guard_3 = sdfg.add_state("guard_3")
    a0 = loop_body_0.add_access("A")
    res0 = loop_body_0.add_access("res")
    loop_tasklet_0 = loop_body_0.add_tasklet("overwrite_0", {}, {"a"}, "a = 0")
    loop_body_0.add_edge(loop_tasklet_0, "a", a0, None, dace.Memlet("A"))
    loop_body_0.add_edge(a0, None, res0, None, dace.Memlet("A[0]"))
    a1 = loop_body_1.add_access("A")
    res1 = loop_body_1.add_access("res")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite_1", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet_1, "a", a1, None, dace.Memlet("A"))
    loop_body_1.add_edge(a1, None, res1, None, dace.Memlet("A[0]"))
    sdfg.add_edge(guard_0, loop_body_0, dace.InterstateEdge(condition="i < N"))
    sdfg.add_edge(loop_body_0, guard_0,
                  dace.InterstateEdge(assignments={"i": "i+1"}))
    sdfg.add_edge(
        guard_1, guard_0,
        dace.InterstateEdge(condition="j < N", assignments={"i": "i + 1"}))
    sdfg.add_edge(
        guard_0, guard_1,
        dace.InterstateEdge(condition="not (i < N)",
                            assignments={"j": "j + 1"}))
    sdfg.add_edge(
        guard_1, guard_2,
        dace.InterstateEdge(condition="not(j < N)", assignments={"k": "0"}))
    sdfg.add_edge(guard_2, loop_body_1, dace.InterstateEdge(condition="k < N"))
    sdfg.add_edge(loop_body_1, guard_2,
                  dace.InterstateEdge(assignments={"k": "k + 1"}))
    sdfg.add_edge(
        guard_2, guard_3,
        dace.InterstateEdge(condition="not(k < N)", assignments={"l":
                                                                 "l + 1"}))
    sdfg.add_edge(init, guard_3, dace.InterstateEdge(assignments={"l": "0"}))
    sdfg.add_edge(guard_3, guard_1, dace.InterstateEdge(condition="l < N"))
    sdfg.add_edge(guard_3, end, dace.InterstateEdge(condition="not (l < N)", ))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    sdfg.validate()
    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{a0}, {a1}]
    name_sets["res"] = [{res0, res1}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_nested_loops2_fission():
    """three nested loops that overwrite two dimensional array. 
    the innermost loop is surrounded by a loop that doesn't iterate over array range but over an empty constant range.
    Therefore the loop nest as a whole does not overwrite the array"""
    sdfg = dace.SDFG("nested_loops")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("i", dace.int32)
    sdfg.add_symbol("j", dace.int32)
    sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [2], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body_0 = sdfg.add_state("loop_body_0")
    loop_body_1 = sdfg.add_state("loop_body_1")
    guard_1 = sdfg.add_state("guard_1")
    guard_2 = sdfg.add_state("guard_2")
    guard_3 = sdfg.add_state("guard_3")
    a0 = loop_body_0.add_access("A")
    res0 = loop_body_0.add_access("res")
    loop_tasklet_0 = loop_body_0.add_tasklet("overwrite_0", {}, {"a"}, "a = 0")
    loop_body_0.add_edge(loop_tasklet_0, "a", a0, None, dace.Memlet("A"))
    loop_body_0.add_edge(a0, None, res0, None, dace.Memlet("A[0]"))
    a1 = loop_body_1.add_access("A")
    res1 = loop_body_1.add_access("res")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite_1", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet_1, "a", a1, None, dace.Memlet("A"))
    loop_body_1.add_edge(a1, None, res1, None, dace.Memlet("A[0]"))
    init_tasklet = init.add_tasklet("overwrite_2", {}, {"a"}, "a = 0")
    a2 = init.add_access("A")
    init.add_edge(init_tasklet, "a", a2, None, dace.Memlet("A"))
    sdfg.add_edge(guard_1, loop_body_0, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(loop_body_0, guard_1,
                  dace.InterstateEdge(assignments={"j": "j + 1"}))
    sdfg.add_edge(
        guard_1, guard_2,
        dace.InterstateEdge(condition="not(j < N)", assignments={"k": "0"}))
    sdfg.add_edge(guard_2, loop_body_1, dace.InterstateEdge(condition="k < N"))
    sdfg.add_edge(loop_body_1, guard_2,
                  dace.InterstateEdge(assignments={"k": "k + 1"}))
    sdfg.add_edge(
        guard_2, guard_3,
        dace.InterstateEdge(condition="not(k < N)", assignments={"l":
                                                                 "l + 1"}))
    sdfg.add_edge(init, guard_3, dace.InterstateEdge(assignments={"l": "0"}))
    sdfg.add_edge(
        guard_3, guard_1,
        dace.InterstateEdge(condition="l < N", assignments={"j": "0"}))
    sdfg.add_edge(guard_3, end, dace.InterstateEdge(condition="not (l < N)", ))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    sdfg.validate()
    try:
        sdfg.validate()
    except:
        assert (False)
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{a0}, {a1}, {a2}]
    name_sets["res"] = [{res0, res1}]
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    assert_rename_sets(name_sets, access_nodes)


def test_nested_loops2_no_fission():
    """three nested loops that overwrite two dimensional array. 
    the innermost loop is surrounded by a loop that doesn't iterate over array range but over an empty constant range.
    Therefore the loop nest as a whole does not overwrite the array"""
    sdfg = dace.SDFG("nested_loops")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("i", dace.int32)
    sdfg.add_symbol("j", dace.int32)
    sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [2], dace.int64, transient=True)
    sdfg.add_array("res", [1], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body_0 = sdfg.add_state("loop_body_0")
    loop_body_1 = sdfg.add_state("loop_body_1")
    guard_1 = sdfg.add_state("guard_1")
    guard_2 = sdfg.add_state("guard_2")
    guard_3 = sdfg.add_state("guard_3")
    a0 = loop_body_0.add_access("A")
    res0 = loop_body_0.add_access("res")
    loop_tasklet_0 = loop_body_0.add_tasklet("overwrite_0", {}, {"a"}, "a = 0")
    loop_body_0.add_edge(loop_tasklet_0, "a", a0, None, dace.Memlet("A"))
    loop_body_0.add_edge(a0, None, res0, None, dace.Memlet("A[0]"))
    a1 = loop_body_1.add_access("A")
    res1 = loop_body_1.add_access("res")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite_1", {}, {"a"}, "a = 0")
    loop_body_1.add_edge(loop_tasklet_1, "a", a1, None, dace.Memlet("A"))
    loop_body_1.add_edge(a1, None, res1, None, dace.Memlet("A[0]"))
    init_tasklet = init.add_tasklet("overwrite_2", {}, {"a"}, "a = 0")
    a2 = init.add_access("A")
    init.add_edge(init_tasklet, "a", a2, None, dace.Memlet("A"))
    res3 = end.add_access("res")
    a3 = end.add_access("A")
    end.add_edge(a3, None, res3, None, dace.Memlet("A[0]"))
    sdfg.add_edge(guard_1, loop_body_0, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(loop_body_0, guard_1,
                  dace.InterstateEdge(assignments={"j": "j + 1"}))
    sdfg.add_edge(
        guard_1, guard_2,
        dace.InterstateEdge(condition="not(j < N)", assignments={"k": "0"}))
    sdfg.add_edge(guard_2, loop_body_1, dace.InterstateEdge(condition="k < N"))
    sdfg.add_edge(loop_body_1, guard_2,
                  dace.InterstateEdge(assignments={"k": "k + 1"}))
    sdfg.add_edge(
        guard_2, guard_3,
        dace.InterstateEdge(condition="not(k < N)", assignments={"l":
                                                                 "l + 1"}))
    sdfg.add_edge(init, guard_3, dace.InterstateEdge(assignments={"l": "0"}))
    sdfg.add_edge(
        guard_3, guard_1,
        dace.InterstateEdge(condition="l < N", assignments={"j": "0"}))
    sdfg.add_edge(guard_3, end, dace.InterstateEdge(condition="not (l < N)", ))

    pipeline = Pipeline([ArrayFission()])
    results = pipeline.apply_pass(sdfg, {})

    sdfg.validate()
    try:
        sdfg.validate()
    except:
        assert (False)
    access_nodes: Dict[str, Dict[SDFGState, Tuple[
        Set[AccessNode],
        Set[AccessNode]]]] = results[FindAccessNodes.__name__][sdfg.sdfg_id]
    name_sets: Dict[str, List[Set[AccessNode]]] = {}
    name_sets['A'] = [{a0, a1, a2, a3}]
    name_sets["res"] = [{res0, res1, res3}]
    assert_rename_sets(name_sets, access_nodes)


if __name__ == '__main__':
    test_simple_conditional_write2()
    test_simple_conditional_write_no_fission()
    test_multiple_conditions()
    test_loop_read_no_fission()
    test_simple_loop_overwrite()
    test_simple_loop_overwrite2()
    test_loop_no_phi_node()
    test_loop_read_before_write_no_fission()
    test_loop_read_before_write_interstate()
    test_simple_map_overwrite()
    test_simple_overwrite_2()
    test_intrastate_overwrite()
    test_nested_loops2_no_fission()
    test_nested_loops2_fission()
    test_scalar_write_shadow_split()
    test_scalar_write_shadow_fused()
    test_scalar_write_shadow_interstate_self()
    test_scalar_write_shadow_interstate_pred()
    test_loop_fake_shadow()
    test_loop_fake_shadow_symbolic()
    test_loop_fake_complex_shadow()
    test_loop_real_shadow()
    test_dominationless_write_branch1()
    test_dominationless_write_branch2()
    test_nested_loops()
