# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop fission (distribution): the LoopRegion equivalent of MapFission.

Splits a ``LoopRegion`` whose body is a single ``SDFGState`` into one loop
per independent node group, replicating the loop header. Components that
share a written data container (a RAW/WAW/WAR dependency) stay in the same
loop; only data-independent groups are separated, so the result is always
value-preserving. A no-op when the body has a single group.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion, SDFGState
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation


def _is_per_iter_subset(subset, loop_var: Optional[str]) -> bool:
    """``True`` iff every dimension of ``subset`` is a single-point access at
    the loop variable with offset zero (or loop-invariant).

    A ``write-then-read`` chain through a non-transient AccessNode that obeys
    this is *per-iteration* -- the produced and consumed values coincide on
    the same loop index. Sequential loop fission preserves the value because
    the producer loop finishes (every ``a[i]`` updated) before the consumer
    loop starts (each ``a[i]`` read sees the just-updated value, exactly as
    in the original interleaved order).

    Cross-iteration subsets like ``a[i - 1]`` break this property and force
    the producer / consumer to stay in the same fission group.
    """
    if loop_var is None or subset is None:
        return False
    import sympy as sp
    try:
        loop_sym = sp.Symbol(loop_var)
    except Exception:
        return False
    saw_loop_var = False
    for rb, re_, _ in subset.ndrange():
        if rb != re_:
            return False
        try:
            expr = sp.sympify(str(rb))
        except Exception:
            return False
        if loop_sym in expr.free_symbols:
            offset = sp.simplify(expr - loop_sym)
            if not (getattr(offset, 'is_number', False) and offset == 0):
                return False
            saw_loop_var = True
    # A subset that NEVER references the loop variable is a constant slot in
    # this loop's scope (e.g. ``a[i]`` inside an inner ``for j``). Constant
    # slots are NOT per-iteration: a write-then-read in the body forms an
    # intra-iteration dependence (stmt2 reads the value stmt1 wrote in the
    # SAME iteration), and fissioning the loop would let the consumer see
    # only the final write instead of the per-iter intermediates. TSVC s257
    # ``a[i] = aa[j,i] - a[i-1]; aa[j,i] = a[i] + bb[j,i]`` inside the
    # inner ``j`` loop is the canonical regression.
    return saw_loop_var


def _container_per_iter_only(state: SDFGState, data: str, loop_var: Optional[str]) -> bool:
    """``True`` iff every memlet referencing ``data`` in ``state`` is per-iter."""
    for n in state.nodes():
        if not (isinstance(n, nodes.AccessNode) and n.data == data):
            continue
        for e in list(state.in_edges(n)) + list(state.out_edges(n)):
            if e.data is None:
                continue
            sub = e.data.get_dst_subset(e, state) if e.data.subset is None else e.data.subset
            if not _is_per_iter_subset(sub, loop_var):
                return False
    return True


def _rewrite_per_iter_bridges(state: SDFGState, loop_var: Optional[str]):
    """In-place: replace writer-side AccessNodes whose out-edges feed
    downstream consumers (the textbook fission bridge) with a fresh reader
    AccessNode for the same data.

    For a non-transient ``a`` that's *only* accessed per-iteration in this
    state, the value the writer just produced equals ``a[loop_var]`` -- which
    is exactly what a fresh reader on the same array would load. The pre-
    rewrite makes the producer and consumer naturally appear as separate
    dataflow components, and downstream loops in the parent CFG will see
    the producer's loop finish before the consumer's loop starts, so each
    reader genuinely sees the just-written value.

    Has no effect if ``loop_var`` is ``None`` or if no per-iter shared
    container exists.
    """
    if loop_var is None:
        return
    written = {n.data for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0}
    for data in list(written):
        desc = state.sdfg.arrays.get(data)
        if desc is None or getattr(desc, 'transient', False):
            continue
        if not _container_per_iter_only(state, data, loop_var):
            continue
        # Find writer-side AccessNodes with downstream consumers.
        for n in list(state.nodes()):
            if not (isinstance(n, nodes.AccessNode) and n.data == data and state.in_degree(n) > 0
                    and state.out_degree(n) > 0):
                continue
            out_edges = list(state.out_edges(n))
            if not out_edges:
                continue
            for oe in out_edges:
                fresh = state.add_access(data)
                state.add_edge(fresh, oe.src_conn, oe.dst, oe.dst_conn, oe.data)
                state.remove_edge(oe)


def _independent_groups(state: SDFGState, loop_var: Optional[str] = None) -> List[List[nodes.Node]]:
    """Partition ``state``'s nodes into data-independent groups.

    A *pure input* is an AccessNode with no in-edges whose data is never
    written in the state -- a read-only loop input. Such nodes do not connect
    their consumers (each fissioned loop re-reads the input). Non-input nodes
    are grouped by dataflow connectivity, then groups touching a common
    container that is *written* in the state are merged (RAW/WAW/WAR). Each
    returned group also carries the input nodes feeding it, so cloning then
    pruning to a group keeps a self-contained body.

    When ``loop_var`` is provided and a non-transient container is accessed
    *only* per-iteration (``a[loop_var]`` everywhere) the producer/consumer
    bridge through that container is severed in both the dataflow union and
    the container-shared merge: sequential loop fission preserves the value
    in that case. TSVC s221 (``a[i] = a[i] + c[i] * d[i]; b[i] = b[i-1] +
    a[i] + d[i]``) fissions into two loops under this rule.

    :param state: The loop body state.
    :param loop_var: The enclosing loop's iteration variable. ``None`` keeps
        the legacy strict-merge behaviour.
    :returns: A list of node lists, one per independent group, deterministic.
    """
    order = {n: i for i, n in enumerate(state.nodes())}
    written = {n.data for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0}
    is_input = {
        n
        for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) == 0 and n.data not in written
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
        # Skip per-iter non-transient containers: write-then-read at the same
        # loop index can be safely sequenced into sibling loops (the producer
        # finishes all writes before the consumer reads). The bridges are
        # rewritten in :func:`_fission` to give each clone its own reader.
        if loop_var is not None:
            desc = state.sdfg.arrays.get(data)
            if (desc is not None and not getattr(desc, 'transient', False)
                    and _container_per_iter_only(state, data, loop_var)):
                continue
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
        feeders = [n for n in is_input if any(e.dst in member_set for e in state.out_edges(n))]
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

    Refuses (returns ``None``) when any body iedge carries a **stateful**
    assignment -- one whose RHS references the LHS, like
    ``k := k + 1`` (a counter recurrence). Cloning the loop for fission
    would duplicate the increment in every sibling, so a body that
    semantically does ``k += 1`` per iter would do ``k += 1`` × N_siblings
    per iter and produce wrong values for any downstream consumer reading
    ``k`` (TSVC ``s126``). Side-effect-free derivations like
    ``idx_index := idx[i]`` do not reference their own LHS and remain
    fissionable -- each sibling rederives the same value from arrays.

    :param loop: The loop whose body shape is inspected.
    :returns: The sole compute ``SDFGState``, or ``None`` if the body is not
        of that shape.
    """
    blocks = list(loop.nodes())
    if any(not isinstance(b, SDFGState) for b in blocks):
        return None
    nonempty = [s for s in blocks if s.nodes()]
    if len(nonempty) != 1:
        return None
    # Refuse stateful (self-referencing) body iedge assignments. Clone-
    # duplicating ``k := k + 1`` across siblings would multiply the
    # increment per outer iter.
    for e in loop.edges():
        for lhs, rhs in (e.data.assignments or {}).items():
            try:
                from dace import symbolic
                rhs_free = set(str(s) for s in symbolic.pystr_to_symbolic(rhs).free_symbols)
            except Exception:
                rhs_free = {lhs}  # conservative: assume self-reference on parse failure
            if lhs in rhs_free:
                return None
    return nonempty[0]


def _block_rw(block: ControlFlowBlock) -> Tuple[Set[str], Set[str]]:
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
    groups = sorted((sorted(g, key=lambda b: pos[b]) for g in classes.values()), key=lambda g: pos[g[0]])
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
        # Maximal fission (perfect loop nesting): fissioning an inner loop
        # turns its parent's body from a single block into several independent
        # blocks, which then makes the parent itself fissionable. A single
        # outer-to-inner sweep distributes only the inner loop -- the parent is
        # visited before its body splits -- so re-sweep until nothing more
        # fissions. The goal is that every leaf computation ends up enclosed by
        # its own complete loop nest. Each fission rebuilds the CFG, so restart
        # the scan after applying one.
        count = 0
        changed = True
        while changed:
            changed = False
            for loop in list(sdfg.all_control_flow_regions(recursive=True)):
                if not isinstance(loop, LoopRegion):
                    continue
                compute = _single_compute_state(loop)
                if compute is not None:
                    _rewrite_per_iter_bridges(compute, loop.loop_variable)
                    if len(_independent_groups(compute, loop.loop_variable)) < 2:
                        continue
                    self._fission(loop, compute)
                else:
                    groups = _independent_block_groups(loop)
                    if groups is None:
                        continue
                    self._fission_blocks(loop, groups)
                count += 1
                changed = True
                # The per-group ``copy.deepcopy(loop)`` clones leave any nested
                # SDFG inside the body with a stale ``parent_sdfg`` (it still
                # points away from this root); reattach all nested-SDFG parents
                # before the next scan re-reads the CFG.
                set_nested_sdfg_parent_references(sdfg)
                break
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
        loop_var = loop.loop_variable
        ngroups = len(_independent_groups(compute, loop_var))

        clones: List[LoopRegion] = []
        for gi in range(ngroups):
            clone = copy.deepcopy(loop)
            clone.label = f"{loop.label}_fis{gi}"
            parent.add_node(clone)
            cstate = list(clone.nodes())[cidx]
            keep = set(_independent_groups(cstate, loop_var)[gi])
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
