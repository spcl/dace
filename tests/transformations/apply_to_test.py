# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the `apply_to` transformation API. """
import dace
from dace.sdfg import utils as sdutil
from dace.transformation.dataflow import MapFusion
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.pattern_matching import enumerate_matches


@dace.function
def dbladd(A: dace.float64[100, 100], B: dace.float64[100, 100]):
    dbl = B
    return A + dbl * B


def test_applyto_pattern():
    sdfg = dbladd.to_sdfg()
    sdfg.coarsen_dataflow()

    # Since there is only one state (thanks to StateFusion), we can use the
    # first one in the SDFG
    state = sdfg.node(0)

    # The multiplication map is called "_Mult__map" (see above graph), we can
    # query it
    mult_exit = next(
        n for n in state.nodes()
        if isinstance(n, dace.nodes.MapExit) and n.label == '_Mult__map')
    # Same goes for the addition entry
    add_entry = next(
        n for n in state.nodes()
        if isinstance(n, dace.nodes.MapEntry) and n.label == '_Add__map')
    # Since all redundant arrays have been removed by dataflow coarsening,
    # we can get the only transient array that remains in the graph
    transient = next(aname for aname, desc in sdfg.arrays.items()
                     if desc.transient)
    access_node = next(
        n for n in state.nodes()
        if isinstance(n, dace.nodes.AccessNode) and n.data == transient)

    MapFusion.apply_to(sdfg,
                       first_map_exit=mult_exit,
                       array=access_node,
                       second_map_entry=add_entry)


def test_applyto_enumerate():
    sdfg = dbladd.to_sdfg()
    sdfg.coarsen_dataflow()

    # Construct subgraph pattern
    pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode,
                                     dace.nodes.MapEntry)
    for subgraph in enumerate_matches(sdfg, pattern):
        MapFusion.apply_to(sdfg,
                           first_map_exit=subgraph.source_nodes()[0],
                           array=next(n for n in subgraph.nodes()
                                      if isinstance(n, dace.nodes.AccessNode)),
                           second_map_entry=subgraph.sink_nodes()[0])


def test_applyto_subgraph():
    sdfg = dbladd.to_sdfg()
    sdfg.coarsen_dataflow()
    state = sdfg.node(0)
    # Apply to subgraph
    SubgraphFusion.apply_to(sdfg, state.nodes())


if __name__ == '__main__':
    test_applyto_pattern()
    test_applyto_enumerate()
    test_applyto_subgraph()