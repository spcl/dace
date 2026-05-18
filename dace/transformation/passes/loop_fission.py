# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop fission (distribution): the LoopRegion equivalent of MapFission.

Splits a ``LoopRegion`` whose body is a single ``SDFGState`` into one loop
per independent node group, replicating the loop header. Components that
share a written data container (a RAW/WAW/WAR dependency) stay in the same
loop; only data-independent groups are separated, so the result is always
value-preserving. A no-op when the body has a single group.
"""
import copy
from typing import Any, Dict, List, Optional, Set

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation import pass_pipeline as ppl, transformation


def _independent_groups(state: SDFGState) -> List[List[nodes.Node]]:
    """Partition ``state``'s nodes into data-independent groups.

    A *pure input* is an AccessNode with no in-edges whose data is never
    written in the state -- a read-only loop input. Such nodes do not connect
    their consumers (each fissioned loop re-reads the input). Non-input nodes
    are grouped by dataflow connectivity, then groups touching a common
    container that is *written* in the state are merged (RAW/WAW/WAR). Each
    returned group also carries the input nodes feeding it, so cloning then
    pruning to a group keeps a self-contained body.

    :param state: The loop body state.
    :returns: A list of node lists, one per independent group, deterministic.
    """
    order = {n: i for i, n in enumerate(state.nodes())}
    written = {
        n.data
        for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0
    }
    is_input = {
        n
        for n in state.nodes()
        if isinstance(n, nodes.AccessNode) and state.in_degree(n) == 0 and n.data not in written
    }
    core = [n for n in state.nodes() if n not in is_input]
    parent: Dict[nodes.Node, nodes.Node] = {n: n for n in core}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    # Connect non-input nodes by dataflow.
    for e in state.edges():
        if e.src not in is_input and e.dst not in is_input:
            union(e.src, e.dst)

    # Merge groups that touch a container written in the state. A re-read
    # through an input node binds to the group(s) of its consumers.
    def group_of(node):
        if node in is_input:
            return [find(e.dst) for e in state.out_edges(node) if e.dst not in is_input]
        return [find(node)]

    for data in written:
        reps = []
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == data:
                reps += group_of(n)
        for r in reps[1:]:
            union(reps[0], r)

    classes: Dict[nodes.Node, List[nodes.Node]] = {}
    for n in core:
        classes.setdefault(find(n), []).append(n)
    groups = []
    for members in classes.values():
        member_set = set(members)
        feeders = [
            n for n in is_input if any(e.dst in member_set for e in state.out_edges(n))
        ]
        groups.append(sorted(members + feeders, key=lambda n: order[n]))
    return sorted(groups, key=lambda g: order[g[0]])


@transformation.explicit_cf_compatible
class LoopFission(ppl.Pass):
    """Distribute a single-body-state loop into one loop per independent group."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Fission every qualifying loop in ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of loops fissioned, or ``None`` if none.
        """
        count = 0
        for loop in list(sdfg.all_control_flow_regions(recursive=True)):
            if not isinstance(loop, LoopRegion):
                continue
            body = list(loop.nodes())
            if len(body) != 1 or not isinstance(body[0], SDFGState):
                continue  # Conservative: only the simple single-state body.
            if len(_independent_groups(body[0])) < 2:
                continue
            self._fission(loop)
            count += 1
        return count or None

    @staticmethod
    def _fission(loop: LoopRegion):
        """Replace ``loop`` with one header-replicated loop per group."""
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop
        ngroups = len(_independent_groups(loop.nodes()[0]))

        clones: List[LoopRegion] = []
        for gi in range(ngroups):
            clone = copy.deepcopy(loop)
            clone.label = f"{loop.label}_fis{gi}"
            parent.add_node(clone)
            cstate = clone.nodes()[0]
            keep = set(_independent_groups(cstate)[gi])
            for n in [n for n in cstate.nodes() if n not in keep]:
                cstate.remove_node(n)
            clones.append(clone)

        # Re-thread interstate edges: pred -> clone0 -> ... -> cloneN -> succ.
        for e in in_edges:
            parent.add_edge(e.src, clones[0], copy.deepcopy(e.data))
        for a, b in zip(clones, clones[1:]):
            parent.add_edge(a, b, InterstateEdge())
        for e in out_edges:
            parent.add_edge(clones[-1], e.dst, copy.deepcopy(e.data))
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(loop)
        if is_start:
            parent.start_block = parent.node_id(clones[0])
