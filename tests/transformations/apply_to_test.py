# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the `apply_to` transformation API. """
import dace
import numpy as np
import pytest
from dace.sdfg import utils as sdutil
from dace.transformation.dataflow import MapFusionVertical
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.passes.pattern_matching import enumerate_matches


@dace.function
def dbladd(A: dace.float64[100, 100], B: dace.float64[100, 100]):
    """Test function of two maps that can be fused."""
    dbl = B
    return A + dbl * B


@dace.program
def unfusable(A: dace.float64[100], B: dace.float64[100, 100]):
    """Test function of two maps that can not be fused."""
    tmp = np.empty_like(A)
    ret = np.empty_like(B)
    for k in dace.map[0:100]:
        tmp[k] = A[k] + 3
    for i, j in dace.map[0:100, 0:100]:
        ret[i, j] = B[i, j] * tmp[i]
    return ret


def test_applyto_enumerate():
    sdfg = dbladd.to_sdfg()
    sdfg.simplify()

    # Construct subgraph pattern
    pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)
    for subgraph in enumerate_matches(sdfg, pattern):
        MapFusionVertical.apply_to(sdfg,
                                   first_map_exit=subgraph.source_nodes()[0],
                                   array=next(n for n in subgraph.nodes() if isinstance(n, dace.nodes.AccessNode)),
                                   second_map_entry=subgraph.sink_nodes()[0])


def test_applyto_pattern():
    sdfg = dbladd.to_sdfg()
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1

    state = sdfg.node(0)

    # The multiplication map is called "_Mult__map" (see above graph), we can
    # query it
    mult_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == '_Mult__map')
    # Same goes for the addition entry
    add_entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.label == '_Add__map')
    # Since all redundant arrays have been removed by simplification pass,
    # we can get the only transient array that remains in the graph
    transient = next(aname for aname, desc in sdfg.arrays.items() if desc.transient)
    access_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == transient)

    assert MapFusionVertical.can_be_applied_to(sdfg,
                                               first_map_exit=mult_exit,
                                               array=access_node,
                                               second_map_entry=add_entry)
    MapFusionVertical.apply_to(sdfg, first_map_exit=mult_exit, array=access_node, second_map_entry=add_entry)

    assert len([node for node in state.nodes() if isinstance(node, dace.nodes.MapEntry)]) == 1


def test_applyto_pattern_2():
    """Tests if the `can_be_applied_to()` also returns negative results."""
    sdfg: dace.SDFG = unfusable.to_sdfg()
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1

    state: dace.SDFGState = sdfg.node(0)

    # We identify the maps my looking for the `tmp` node.
    tmp: dace.nodes.AccessNode = next(n for n in state.data_nodes() if n.data == "tmp")
    assert state.in_degree(tmp) == 1 and state.out_degree(tmp) == 1
    assert tmp.desc(sdfg).transient

    # Now get the two maps.
    map_exit_1 = next(e.src for e in state.in_edges(tmp) if isinstance(e.src, dace.nodes.MapExit))
    map_entry_2 = next(e.dst for e in state.out_edges(tmp) if isinstance(e.dst, dace.nodes.MapEntry))

    assert not MapFusionVertical.can_be_applied_to(
        sdfg, first_map_exit=map_exit_1, array=tmp, second_map_entry=map_entry_2)
    with pytest.raises(
            ValueError,
            match=r'Transformation cannot be applied on the given subgraph \("can_be_applied" failed\)',
    ):
        MapFusionVertical.apply_to(sdfg,
                                   verify=True,
                                   first_map_exit=map_exit_1,
                                   array=tmp,
                                   second_map_entry=map_entry_2)


def test_applyto_subgraph():
    sdfg = dbladd.to_sdfg()
    sdfg.simplify()
    state = sdfg.node(0)

    # Apply to subgraph
    assert SubgraphFusion.can_be_applied_to(sdfg, *state.nodes())
    SubgraphFusion.apply_to(sdfg, state.nodes())


def test_applyto_subgraph_2():
    """Tests if the `can_be_applied_to()` also returns negative results."""
    sdfg = unfusable.to_sdfg()
    sdfg.simplify()
    state = sdfg.node(0)

    # Apply to subgraph
    assert not SubgraphFusion.can_be_applied_to(sdfg, state.nodes())

    with pytest.raises(
            ValueError,
            match=r'Transformation cannot be applied on the given subgraph \("can_be_applied" failed\)',
    ):
        SubgraphFusion.apply_to(sdfg, state.nodes())


if __name__ == '__main__':
    test_applyto_enumerate()
    test_applyto_pattern()
    test_applyto_pattern_2()
    test_applyto_subgraph()
    test_applyto_subgraph_2()
