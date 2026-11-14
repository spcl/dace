# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The zeros a contraction sums onto: the store of ``0`` that leaves an accumulator at the sum's identity just before
an ``Einsum`` adds into it.

``LiftEinsum`` and ``LoopToEinsum`` choose ``beta``, the weight of the output's prior value, from whether anything
else writes the output. A store of zeros is such a write, yet the einsum overwriting (``beta=0``) computes the same
thing, and once nothing else reads the zeros and the einsum writes every element they cover, the store is dead. Three
spellings reach the lifts:

* the reduction init ``MapReduceFusion`` puts ahead of the sum it fuses, as a map of its own or joined into an
  unrelated map by fusion (gemver's ``beta * y``);
* a ``FillLibraryNode`` of ``0`` (``C[:] = 0`` ahead of an ``ijk`` accumulate loop);
* either of them in a state before the nest, with only states that do not touch the array in between.

A zero store is the array's write by a ``FillLibraryNode`` of constant ``0``, or by a tasklet of a top-level map that
reads nothing and stores the literal ``0``. It is the sum's prior value when it runs right before it: in the sum's
state, ordered before the sum's scope, with every other access of the array in that state either before the zeros or
after the sum; or in an earlier state of the same region whose last access of the array it is, reached through a
chain of single unconditional edges over states that do not name the array.
"""

import ast
from dataclasses import dataclass

from dace import Memlet, dtypes, nodes, subsets
from dace.sdfg.state import ControlFlowBlock, SDFGState


@dataclass(slots=True)
class ZeroStore:
    """A store of ``0`` into ``node``, the access node it lands in, over ``subset``; ``observed`` when anything but
    the sum reads the zeros."""

    state: SDFGState
    node: nodes.AccessNode
    writer: nodes.Node
    subset: subsets.Subset
    observed: bool = False

    def remove(self) -> None:
        """Delete the store, keeping the ordering it gave the access node's successors."""
        state, node = self.state, self.node
        (edge,) = [e for e in state.in_edges(node) if not e.data.is_empty()]
        succs = [e.dst for e in state.out_edges(node)]
        if isinstance(self.writer, nodes.LibraryNode):
            before = [e.src for e in state.in_edges(self.writer)]
            state.remove_node(self.writer)
        else:
            path = state.memlet_path(edge)
            outer = state.entry_node(path[-1].src)
            scope = state.scope_subgraph(outer)
            if [n for n in scope.nodes() if not isinstance(n, (nodes.EntryNode, nodes.ExitNode))] == [self.writer]:
                before = [e.src for e in state.in_edges(outer)]
                state.remove_nodes_from(scope.nodes())
            else:
                before = [path[-1].src]
                for hop in path[1:]:
                    state.remove_edge(hop)
                    hop.src.remove_in_connector(hop.src_conn.replace("OUT_", "IN_", 1))
                    hop.src.remove_out_connector(hop.src_conn)
                state.remove_node(self.writer)
        before += [e.src for e in state.in_edges(node)]
        state.remove_node(node)
        for src in before:
            for dst in succs:
                if src in state.nodes() and not any(e.dst is dst for e in state.out_edges(src)):
                    state.add_nedge(src, dst, Memlet())


def zero_store(state: SDFGState, node: nodes.AccessNode) -> ZeroStore | None:
    """The store of ``0`` that is ``node``'s only write, if it is one (module docstring)."""
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode  # Avoid import loop

    writes = [e for e in state.in_edges(node) if not e.data.is_empty()]
    if len(writes) != 1 or state.entry_node(node) is not None or writes[0].data.data != node.data:
        return None
    edge = writes[0]
    observed = any(not e.data.is_empty() for e in state.out_edges(node))
    if isinstance(edge.src, FillLibraryNode):
        if edge.src.value_edge(state) is not None or edge.src.value != 0 or isinstance(edge.src.value, bool):
            return None
        return ZeroStore(state, node, edge.src, edge.data.subset, observed)
    if not isinstance(edge.src, nodes.MapExit):
        return None
    tasklet = state.memlet_path(edge)[0].src
    if not isinstance(tasklet, nodes.Tasklet) or tasklet.code.language != dtypes.Language.Python:
        return None
    if state.out_degree(tasklet) != 1 or any(not e.data.is_empty() for e in state.in_edges(tasklet)):
        return None
    code = tasklet.code.code
    if len(code) != 1 or not isinstance(code[0], ast.Assign) or not isinstance(code[0].value, ast.Constant):
        return None
    value = code[0].value.value
    if isinstance(value, bool) or value != 0:
        return None
    return ZeroStore(state, node, tasklet, edge.data.subset, observed)


def zero_before_scope(state: SDFGState, out_node: nodes.AccessNode, entry: nodes.Node) -> ZeroStore | None:
    """The zeros ``out_node``'s array holds when the scope ``entry`` (or node) summing into ``out_node`` starts, stored
    in this state: ordered before ``entry``, every other access of the array before the zeros or after the sum."""
    name = out_node.data
    others = [n for n in state.data_nodes() if n.data == name and n is not out_node]
    for node in others:
        found = zero_store(state, node)
        if found is None:
            continue
        after = set(state.bfs_nodes(node))
        if entry not in after:
            continue
        downstream = set(state.bfs_nodes(out_node))
        if all(n is node or n in downstream or node in set(state.bfs_nodes(n)) for n in others):
            return found
    return None


def zero_before_block(block: ControlFlowBlock, name: str) -> ZeroStore | None:
    """The zeros array ``name`` holds when ``block`` starts, stored by an earlier state of ``block``'s region and not
    touched since (module docstring)."""
    region = block.parent_graph
    current = block
    while True:
        edges = region.in_edges(current)
        if len(edges) != 1 or not edges[0].data.is_unconditional() or name in edges[0].data.free_symbols:
            return None
        current = edges[0].src
        if not isinstance(current, SDFGState):
            return None
        accesses = [n for n in current.data_nodes() if n.data == name]
        if not accesses:
            continue
        last = [n for n in accesses if not any(m is not n and m in current.bfs_nodes(n) for m in accesses)]
        if len(last) != 1:
            return None
        found = zero_store(current, last[0])
        if found is None or current.entry_node(last[0]) is not None:
            return None
        return found
