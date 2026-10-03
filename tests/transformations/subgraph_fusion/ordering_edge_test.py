# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Ordering edges into a fused map entry must not be anchored on the fused entry when they come
from a node the fused map itself produces -- that closes a cycle through the entry. """

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import SubgraphFusion

N = dace.symbol('N')


def build_waw_ordered_sdfg() -> dace.SDFG:
    """ Two same-range maps writing disjoint columns of A, sequenced by an ordering edge. """
    sdfg = dace.SDFG('subgraph_fusion_ordering_edge')
    sdfg.add_array('A', [N, 2], dace.float64)
    state = sdfg.add_state()

    _, _, mx1 = state.add_mapped_tasklet('first', {'i': '0:N'}, {},
                                         'out = 1.0', {'out': dace.Memlet('A[i, 0]')},
                                         external_edges=True)
    _, me2, _ = state.add_mapped_tasklet('second', {'i': '0:N'}, {},
                                         'out = 2.0', {'out': dace.Memlet('A[i, 1]')},
                                         external_edges=True)

    intermediate = state.out_edges(mx1)[0].dst
    state.add_edge(intermediate, None, me2, None, dace.Memlet())
    sdfg.validate()
    return sdfg


def test_ordering_edge_from_intermediate_node():
    sdfg = build_waw_ordered_sdfg()
    state = sdfg.states()[0]
    intermediate = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0][0]
    second = [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'second'][0]

    subgraph = SubgraphView(state, state.nodes())
    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert sf.can_be_applied(sdfg, subgraph)
    sf.apply(sdfg)

    sdfg.validate()
    assert not list(state.find_cycles())

    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert len(entries) == 1
    entry = entries[0]
    # The fused entry sits at global scope. Anchoring the ordering edge on it made it its own
    # scope ancestor, which is what `symbols_defined_at` walked into forever.
    scope = state.scope_dict()
    assert scope[entry] is None
    assert all(scope[n] is not n for n in state.nodes())

    # The sequencing the edge encoded survives per iteration, inside the fused body.
    assert scope[intermediate] is entry
    assert scope[second] is entry
    assert second in set(state.bfs_nodes(intermediate))

    A = np.zeros((32, 2), dtype=np.float64)
    sdfg(A=A, N=32)
    assert np.allclose(A[:, 0], 1.0)
    assert np.allclose(A[:, 1], 2.0)


if __name__ == '__main__':
    test_ordering_edge_from_intermediate_node()
