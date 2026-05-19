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
from dace.sdfg.utils import set_nested_sdfg_parent_references
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


def _single_compute_state(loop: LoopRegion) -> Optional[SDFGState]:
    """The loop body's unique non-empty ``SDFGState`` if the body is that
    state plus only empty states joined by unconditional edges; else
    ``None``.

    The dace frontend commonly emits ``empty-state --(idx_index=idx[i])-->
    compute-state`` for an indirect (gather/scatter) loop body: the
    interstate-edge assignments are the indirect-access index symbols (by
    convention these body edges only ever carry indirection symbols, never
    computation, and -- being structured control flow -- never a condition).
    They are loop-body-local and side-effect-free, so node-group fission
    still applies to the single compute state -- the empty states and the
    symbol-defining edges ride along unchanged in every clone (``_fission``
    deep-copies the whole loop), porting each tasklet together with the
    symbols it needs.

    :param loop: The loop whose body shape is inspected.
    :returns: The sole compute ``SDFGState``, or ``None`` if the body is not
        of that shape.
    """
    blocks = list(loop.nodes())
    if any(not isinstance(b, SDFGState) for b in blocks):
        return None
    nonempty = [s for s in blocks if s.nodes()]
    return nonempty[0] if len(nonempty) == 1 else None


def _block_rw(block):
    """Recursively collect (reads, writes) data containers of a CFG block.

    :param block: An ``SDFGState`` or control-flow region.
    :returns: ``(reads, writes)`` sets of data-container names.
    """
    reads: Set[str] = set()
    writes: Set[str] = set()
    states = [block] if isinstance(block, SDFGState) else list(block.all_states())
    for st in states:
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode):
                if st.in_degree(n) > 0:
                    writes.add(n.data)
                if st.out_degree(n) > 0 or st.in_degree(n) == 0:
                    reads.add(n.data)
    return reads, writes


def _linear_blocks(loop: LoopRegion) -> Optional[List]:
    """Return ``loop``'s body blocks in execution order if it is a simple
    linear chain of unconditional, assignment-free edges; else ``None``.

    :param loop: The loop whose body CFG is inspected.
    :returns: The ordered block list, or ``None`` if not a plain chain.
    """
    blocks = list(loop.nodes())
    edges = list(loop.edges())
    if len(edges) != len(blocks) - 1:
        return None
    for e in edges:
        if e.data.assignments or e.data.condition.as_string not in ('1', 'True', '(1)'):
            return None
    succ = {e.src: e.dst for e in edges}
    order = [loop.start_block]
    while order[-1] in succ:
        order.append(succ[order[-1]])
    return order if len(order) == len(blocks) else None


def _independent_block_groups(loop: LoopRegion) -> Optional[List[List]]:
    """Partition ``loop``'s body blocks into data-independent groups.

    Only a plain linear chain of >= 2 blocks qualifies. Blocks touching a
    common written container are merged (a real dependency); read-only
    sharing does not merge. This realizes perfect-loop-nesting for loops:
    distribute the parent loop over its independent inner blocks.

    :param loop: The parent loop.
    :returns: Ordered list of block groups, or ``None`` if not applicable.
    """
    order = _linear_blocks(loop)
    if order is None or len(order) < 2:
        return None
    pos = {b: i for i, b in enumerate(order)}
    rw = {b: _block_rw(b) for b in order}
    parent: Dict = {b: b for b in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    written: Set[str] = set()
    for _r, w in rw.values():
        written |= w
    for data in written:
        touch = [b for b in order if data in rw[b][0] or data in rw[b][1]]
        for b in touch[1:]:
            parent[find(b)] = find(touch[0])

    classes: Dict = {}
    for b in order:
        classes.setdefault(find(b), []).append(b)
    groups = sorted((sorted(g, key=lambda b: pos[b]) for g in classes.values()),
                    key=lambda g: pos[g[0]])
    return groups if len(groups) >= 2 else None


@transformation.explicit_cf_compatible
class LoopFission(ppl.Pass):
    """Distribute a loop into one loop per independent group.

    Two shapes: a single-body-``SDFGState`` loop split by independent node
    groups, and a multi-block linear body split by independent blocks
    (perfect-loop-nesting for loops -- the LoopRegion analogue of how
    map-side ``PerfLoopNesting`` delegates to ``MapFission``).
    """
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
            compute = _single_compute_state(loop)
            if compute is not None:
                if len(_independent_groups(compute)) < 2:
                    continue
                self._fission(loop, compute)
                count += 1
            else:
                groups = _independent_block_groups(loop)
                if groups is None:
                    continue
                self._fission_blocks(loop, groups)
                count += 1
        if count:
            # The per-group ``copy.deepcopy(loop)`` clones leave any nested
            # SDFG inside the body with a stale ``parent_sdfg`` (it still
            # points away from this root); reattach all nested-SDFG parents.
            set_nested_sdfg_parent_references(sdfg)
        return count or None

    @staticmethod
    def _fission_blocks(loop: LoopRegion, groups: List[List]):
        """Distribute ``loop`` over independent body-block groups."""
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop
        orig_order = _linear_blocks(loop)
        keep_idx = [sorted(orig_order.index(b) for b in g) for g in groups]

        clones: List[LoopRegion] = []
        for gi, idxs in enumerate(keep_idx):
            clone = copy.deepcopy(loop)
            clone.label = f"{loop.label}_fis{gi}"
            parent.add_node(clone)
            corder = _linear_blocks(clone)
            keep = [corder[i] for i in idxs]
            for b in [b for b in clone.nodes() if b not in keep]:
                clone.remove_node(b)
            for e in list(clone.edges()):
                clone.remove_edge(e)
            for a, b in zip(keep, keep[1:]):
                clone.add_edge(a, b, InterstateEdge())
            clone.start_block = clone.node_id(keep[0])
            clones.append(clone)

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

    @staticmethod
    def _fission(loop: LoopRegion, compute: SDFGState):
        """Replace ``loop`` with one header-replicated loop per independent
        node group of its single compute state.

        :param loop: The loop to distribute.
        :param compute: ``loop``'s sole non-empty body state (any empty no-op
            states ride along unchanged in every clone).
        """
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop
        cidx = list(loop.nodes()).index(compute)
        ngroups = len(_independent_groups(compute))

        clones: List[LoopRegion] = []
        for gi in range(ngroups):
            clone = copy.deepcopy(loop)
            clone.label = f"{loop.label}_fis{gi}"
            parent.add_node(clone)
            cstate = list(clone.nodes())[cidx]
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
