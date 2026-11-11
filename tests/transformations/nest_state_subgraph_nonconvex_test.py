# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``nest_state_subgraph`` must nest a subgraph that leaves and re-enters through an access node as one body.

A subgraph is not convex when a path leaves it and comes back: here the tasklets of one statement write an array
element to a global access node and read it back through that node (a read-after-write chain through ``A``), so the
access nodes sit between the tasklets but outside the node set the caller names. Nesting only the named nodes
folded the boundary edges of every ``A`` access node onto one connector and restated the happens-before of each
folded edge as an ordering edge: ``access node -> nest`` for the reads, then ``nest -> nest`` for the writes. The
state came out cyclic (``NetworkXUnfeasible`` in the canonicalization passes that sort it) and the write-then-read
order inside the nest was lost. The nest must include the access nodes the path crosses.
"""

import networkx as nx
import numpy as np
import pytest

import dace
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import nest_state_subgraph


def chain_through_a_global_array():
    """``a[0] -> t1 -> A[1] -> t2 -> A[2] -> t3 -> b[0]``, each hop through its own access node of the global ``A``.

    The three tasklets are one statement group; the two ``A`` access nodes lie between them, outside the node set.
    An independent tasklet ``other`` (``c[0] = a[1]``) shares the state and must stay outside the nest.
    """
    sdfg = dace.SDFG("nonconvex_chain")
    for name in ("a", "A", "b", "c"):
        sdfg.add_array(name, [4], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    t1 = state.add_tasklet("t1", {"v"}, {"o"}, "o = v * 2")
    t2 = state.add_tasklet("t2", {"v"}, {"o"}, "o = v + 1")
    t3 = state.add_tasklet("t3", {"v"}, {"o"}, "o = v * 3")
    first, second = state.add_access("A"), state.add_access("A")
    state.add_edge(state.add_access("a"), None, t1, "v", dace.Memlet("a[0]"))
    state.add_edge(t1, "o", first, None, dace.Memlet("A[1]"))
    state.add_edge(first, None, t2, "v", dace.Memlet("A[1]"))
    state.add_edge(t2, "o", second, None, dace.Memlet("A[2]"))
    state.add_edge(second, None, t3, "v", dace.Memlet("A[2]"))
    state.add_edge(t3, "o", state.add_access("b"), None, dace.Memlet("b[0]"))
    other = state.add_tasklet("other", {"v"}, {"o"}, "o = v")
    state.add_edge(state.add_access("a"), None, other, "v", dace.Memlet("a[1]"))
    state.add_edge(other, "o", state.add_access("c"), None, dace.Memlet("c[0]"))
    return sdfg, state, (t1, t2, t3), other


def nests_of(state):
    return [n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)]


def test_achain_through_a_global_array_nests_without_a_cycle():
    sdfg, state, group, outside = chain_through_a_global_array()

    nest_state_subgraph(sdfg, state, SubgraphView(state, list(group)), name="statement")

    assert nx.is_directed_acyclic_graph(state.nx), "nesting must not leave a cyclic state"
    (nest,) = nests_of(state)
    assert not any(e.src is nest and e.dst is nest for e in state.edges()), "an edge from the nest to itself"
    sdfg.validate()


def test_the_nest_keeps_the_write_then_read_order_and_the_values():
    sdfg, state, group, outside = chain_through_a_global_array()
    nest_state_subgraph(sdfg, state, SubgraphView(state, list(group)), name="statement")

    a = np.array([5.0, 7.0, 0.0, 0.0])
    A, b, c = np.zeros(4), np.zeros(4), np.zeros(4)
    sdfg(a=a, A=A, b=b, c=c)

    assert A[1] == 10.0 and A[2] == 11.0, "the nest writes A[1] and A[2] as the tasklets did"
    assert b[0] == 33.0, "t3 reads the A[2] that t2 wrote: ((5 * 2) + 1) * 3"
    assert c[0] == 7.0


def test_a_node_off_every_path_stays_outside_the_nest():
    sdfg, state, group, other = chain_through_a_global_array()

    nest_state_subgraph(sdfg, state, SubgraphView(state, list(group)), name="statement")

    assert other in state.nodes(), "the independent tasklet is not on a path that leaves and re-enters the subgraph"
    assert len(nests_of(state)) == 1


def test_two_statements_nest_one_after_the_other():
    """The order ``group_statements`` uses: each statement group in turn, the second reading what the first wrote."""
    sdfg, state, group, other = chain_through_a_global_array()

    nest_state_subgraph(sdfg, state, SubgraphView(state, list(group)), name="first")
    nest_state_subgraph(sdfg, state, SubgraphView(state, [other]), name="second")

    assert nx.is_directed_acyclic_graph(state.nx)
    sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
