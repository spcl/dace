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
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
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
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    try:
        sdfg.validate()
    except:
        assert (False)
    assert_rename_sets(name_sets, access_nodes)


def test_multiple_conditions():
    "three conditional writes that overwrite a one-dimensional array"
    @dace.program
    def conditional_write_multiple_conditions(A: dace.float64[1], b1: dace.bool, b2: dace.bool):
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
    sdfg.add_edge(init, guard_0, dace.InterstateEdge(
        condition="not (A[0] and A[1])"))
    sdfg.add_edge(guard_0, elif_write, dace.InterstateEdge(
        condition="not A[0] and A[1]"))
    sdfg.add_edge(guard_0, else_write, dace.InterstateEdge(
        condition="not A[0] and A[1]"))
    sdfg.add_edge(elif_write, merge_0, dace.InterstateEdge())
    sdfg.add_edge(else_write, merge_0, dace.InterstateEdge())
    sdfg.add_edge(merge_0, merge_1, dace.InterstateEdge())
    sdfg.add_edge(if_write, merge_1, dace.InterstateEdge())
    sdfg.add_edge(init, if_write, dace.InterstateEdge(
        condition="A[0] and A[1]"))
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
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[AccessNode], Set[AccessNode]]]
                       ] = result[FindAccessNodes.__name__][sdfg.sdfg_id]
    try:
        sdfg.validate()
    except:
        assert (False)
    assert_rename_sets(name_sets, access_nodes)


if __name__ == '__main__':
    test_simple_conditional_write()
    test_simple_conditional_write2()
    test_simple_conditional_write_no_fission()
    test_multiple_conditions()
