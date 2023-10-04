import sympy

import dace
from dace import nodes
from dace.memlet import Memlet
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers as xfsh


def k_caching_simple():
    """
    Simple example of two maps of different size being fused while A is being resized and used as a circular buffer
    """
    sdfg = dace.SDFG('differentliy_sized_maps')
    sdfg.add_array('A', [5], dace.float64, transient=True)
    sdfg.add_array('B', [5], dace.float64)
    state = sdfg.add_state()

    task1, mentry1, mexit1 = state.add_mapped_tasklet(
            name="map1",
            map_ranges={"i": "0:5"},
            inputs={},
            outputs={'a': Memlet(data='A', subset='i')},
            code='a = i',
            external_edges=True,
            propagate=True)

    task2, mentry2, mexit2 = state.add_mapped_tasklet(
            name="map2",
            map_ranges={"i": "1:5"},
            inputs={'a1': Memlet(data='A', subset='i'), 'a2': Memlet(data='A', subset='i-1')},
            outputs={'b': Memlet(data='B', subset='i')},
            code='b = a1 - a2',
            external_edges=True)

    to_remove_access = state.in_edges(mentry2)[0]
    state.add_memlet_path(state.out_edges(mexit1)[0].dst, mentry2, memlet=Memlet(data='A', subset='0:5'), dst_conn='IN_A')
    state.remove_edge(to_remove_access)
    state.remove_node(to_remove_access.src)
    subgraph = xfsh.subgraph_from_maps(sdfg, state, [mentry1, mentry2])
    cf = CompositeFusion()
    cf.setup_match(subgraph)
    # Without the K-caching enabled the maps can not be fused
    assert not cf.can_be_applied(sdfg, subgraph)
    cf.subgraph_fusion_properties = {
             'max_difference_start': 1,
             'max_difference_end': 0
             }
    assert cf.can_be_applied(sdfg, subgraph)
    cf.apply(sdfg)
    sdfg.validate()

    # Check that A has been shrunk to a circular buffer
    assert sdfg.data('A').shape == (2,)

    # Check edge writing to A -> should write to (i mod 2)
    edge = state.out_edges(task1)[0]
    assert edge.data.data == 'A'
    rng = edge.data.subset.ranges.ranges[0]
    # It only writes one element
    assert rng[0] == rng[1]
    # It writes to (i mod 2)
    assert isinstance(rng[0], sympy.Mod)
    assert str(rng[0].args[0]) == 'i' and rng[0].args[1] == 2
    # The edge goes to an access node
    dst = state.out_edges(task1)[0].dst
    assert isinstance(dst, nodes.AccessNode) and dst.data == 'A'

    # The edge going away from the AccessNode should go to the nested SDFG and transport the whole of A
    edge = state.out_edges(dst)[0]
    assert edge.data.data == 'A'
    rng = edge.data.subset.ranges.ranges[0]
    # memlet transports whole A
    assert rng[0] == 0 and rng[1] == 1
    # edge goes to nested sdfg
    nsdfg = edge.dst
    assert isinstance(nsdfg, nodes.NestedSDFG)

    # Check that the nested SDFG contains a start state with an interstate edge going away from it protecting it from
    # being executed for i==0
    interstate_edge = nsdfg.sdfg.out_edges(nsdfg.sdfg.start_state)
    assert len(interstate_edge) == 1
    interstate_edge = interstate_edge[0]
    assert interstate_edge.data.condition.as_string == '(i >= 1)'

    # The nsdfg should have two states, the second one should contain the work of the second map
    assert len(nsdfg.sdfg.states()) == 2
    work_state = interstate_edge.dst
    nsdfg_nodes = work_state.nodes()
    assert task2 in nsdfg_nodes
    # The edges going from A to the tasklet should also have the modulo operation
    for iedge in work_state.in_edges(task2):
        assert iedge.src.data == 'A'
        rng = iedge.data.subset.ranges.ranges[0]
        assert rng[0] == rng[1]
        assert isinstance(rng[0], sympy.Mod)
        assert str(rng[0].args[0]) == 'i' or str(rng[0].args[0]) == 'i + 1' or str(rng[0].args[0]) == 'i - 1'
        assert rng[0].args[1] == 2


def main():
    k_caching_simple()


if __name__ == '__main__':
    main()
