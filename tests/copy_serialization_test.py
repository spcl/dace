import sys

import dace
from dace.sdfg import nodes as dace_nodes
from dace.sdfg import graph as dace_graph
from dace.symbolic import pystr_to_symbolic as s2s
from dace.codegen.targets import cpp as dace_cpp


def _make_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_graph.MultiConnectorEdge[dace.Memlet], dace_nodes.AccessNode]:
    sdfg = dace.SDFG("copy_sdfg")
    state = sdfg.add_state()

    for sname in ["__out_IDim_range_0", "__out_IDim_range_1", "__out_IDim_stride", "__out_JDim_range_0", "__out_JDim_range_1", "__out_JDim_stride", "__out_KDim_range_0", "__out_KDim_range_1", "__out_KDim_stride"]:
        sdfg.add_symbol(sname, dace.int32)

    sdfg.add_array(
            "a",
            dtype=dace.float64,
            transient=False,
            shape=(
                s2s("max(0, -__out_IDim_range_0 + __out_IDim_range_1)"),
                s2s("max(0, -__out_JDim_range_0 + __out_JDim_range_1)"),
                s2s("max(0, -max(0, __out_KDim_range_0) + __out_KDim_range_1)"),
            ),
            strides=(
                1,
                s2s("max(0, -__out_IDim_range_0 + __out_IDim_range_1)"),
                s2s("max(0, -__out_IDim_range_0 + __out_IDim_range_1)*max(0, -__out_JDim_range_0 + __out_JDim_range_1)"),
            ),
    )
    sdfg.add_array(
            "b",
            dtype=dace.float64,
            transient=False,
            shape=(
                s2s("max(0, -__out_IDim_range_0 + __out_IDim_range_1)"),
                s2s("max(0, -__out_JDim_range_0 + __out_JDim_range_1)"),
                s2s("max(0, -__out_KDim_range_0 + __out_KDim_range_1)"),
            ),
            strides=(
                1,
                s2s("max(0, -__out_IDim_range_0 + __out_IDim_range_1)"),
                s2s("max(0, -__out_IDim_range_0 + __out_IDim_range_1)*max(0, -__out_JDim_range_0 + __out_JDim_range_1)"),
            ),
    )

    a = state.add_access("a")
    b = state.add_access("b")
    e = state.add_nedge(
            a,
            b,
            dace.Memlet(
                data=a.data,
                subset="0:max(0, __out_IDim_range_1 - __out_IDim_range_0), 0:max(0, __out_JDim_range_1 - __out_JDim_range_0), 0:max(0, __out_KDim_range_0, __out_KDim_range_1) - max(0, __out_KDim_range_0)",
                other_subset="0:max(0, __out_IDim_range_1 - __out_IDim_range_0), 0:max(0, __out_JDim_range_1 - __out_JDim_range_0), -__out_KDim_range_0 + max(0, __out_KDim_range_0):max(0, __out_KDim_range_0, __out_KDim_range_1) - __out_KDim_range_0",
            )
    )

    sdfg.validate()

    return sdfg, state, a, e, b


def test_copy_before_and_after_serialization():
    sdfg1, state1, a1, e1, b1 = _make_sdfg()
    assert sdfg1.number_of_nodes() == 1
    assert state1.number_of_edges() == 1

    copy_shape1, src_strides1, dst_strides1, _, _ = dace_cpp.memlet_copy_to_absolute_strides(
                None, sdfg1, state1, e1, a1, b1
    )
    print(f"|{copy_shape1}| = {len(copy_shape1)}")
    assert len(copy_shape1) == 1

    json = sdfg1.to_json()

    sdfg2 = dace.SDFG.from_json(json)
    state2 = sdfg2.states()[0]
    assert sdfg2.number_of_nodes() == 1
    assert state2.number_of_edges() == 1

    e2 = state2.edges()[0]
    a2 = e2.src
    b2 = e2.dst
    assert a2.data == a1.data
    assert b2.data == b1.data

    copy_shape2, src_strides2, dst_strides2, _, _ = dace_cpp.memlet_copy_to_absolute_strides(
                None, sdfg2, state2, e2, a2, b2
    )

    print(f"|{copy_shape2}| = {len(copy_shape2)}")
    assert len(copy_shape2) == 1

    return 0


def test_memlet_copy_shape_roundtrip():
    sdfg1, state1, a1, e1, b1 = _make_sdfg()

    copy_shape1, *_ = dace_cpp.memlet_copy_to_absolute_strides(
        None, sdfg1, state1, e1, a1, b1
    )

    sdfg2 = dace.SDFG.from_json(sdfg1.to_json())
    state2 = sdfg2.states()[0]
    e2 = state2.edges()[0]

    copy_shape2, *_ = dace_cpp.memlet_copy_to_absolute_strides(
        None, sdfg2, state2, e2, e2.src, e2.dst
    )

    assert copy_shape1 == copy_shape2

if __name__ == "__main__":
    test_memlet_copy_shape_roundtrip()
    test_copy_before_and_after_serialization()