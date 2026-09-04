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

from dace.ordered import OrderedSet

from dace import SDFG
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion, SDFGState
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis, smt_dependence


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
    try:
        loop_sym = symbolic.pystr_to_symbolic(loop_var)
    except Exception:
        return False
    saw_loop_var = False
    for rb, re_, _ in subset.ndrange():
        if rb != re_:
            return False
        try:
            expr = symbolic.pystr_to_symbolic(str(rb))
        except Exception:
            return False
        if loop_sym in expr.free_symbols:
            offset = symbolic.simplify(expr - loop_sym)
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


def _subsets_at_node(node: nodes.AccessNode, state: SDFGState):
    """Yield every (subset, is_write) pair that touches ``node`` on ``node``'s side.

    Copy edges and view edges may carry the other container's name as the
    memlet's ``.data``; the src/dst subset helpers still return the slice that
    belongs to ``node.data``.
    """
    for e in list(state.in_edges(node)) + list(state.out_edges(node)):
        if e.data is None or e.data.is_empty():
            continue
        if e.src is node:
            sub = e.data.get_src_subset(e, state)
            is_write = False
        else:
            sub = e.data.get_dst_subset(e, state)
            is_write = True
        if sub is not None:
            yield sub, is_write


def _accesses_interfere_across_iterations(loop: LoopRegion, subset_a, subset_b) -> bool:
    """``True`` unless z3 proves the two loop-body subsets never alias across iterations.

    Conservative: any exception, missing bound, or inconclusive solver result is
    treated as a real dependence, so the caller keeps the two accesses in the same
    fission group.
    """
    if loop is None or not loop.loop_variable or not smt_dependence.has_z3():
        return True
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    step = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or step is None:
        return True
    nd_a = list(subset_a.ndrange())
    nd_b = list(subset_b.ndrange())
    if len(nd_a) != len(nd_b) or not nd_a:
        return True
    try:
        box_a = [(symbolic.pystr_to_symbolic(str(rb)), symbolic.pystr_to_symbolic(str(re_))) for rb, re_, _ in nd_a]
        box_b = [(symbolic.pystr_to_symbolic(str(rb)), symbolic.pystr_to_symbolic(str(re_))) for rb, re_, _ in nd_b]
    except Exception:
        return True
    try:
        disjoint = smt_dependence.prove_disjoint_access_boxes(box_a, box_b, loop.loop_variable, start, end, step)
    except Exception:
        return True
    return disjoint is not True


def _fissions_after_bridge_rewrite(loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``loop``'s body splits into >= 2 independent groups once its per-iter bridges are rewritten.

    Decided on a DEEPCOPY. The rewrite has to happen before the grouping -- severing a bridge is what makes
    a producer and its consumer look independent (TSVC s221 goes 1 -> 2 groups only because of it) -- but it
    is only SOUND once the fission puts them in separate loops. Answering the question on a throwaway copy
    keeps that conditional mutation off the real graph entirely: a loop that does not fission is never
    touched, so the pass reporting "nothing applied" is the truth.
    """
    probe = copy.deepcopy(loop)  # detached: its states carry no .sdfg, hence the explicit one below
    # Restore the parent reference so that grouping can see sibling loops (e.g. the
    # accumulator loop that consumes a split-out side-writes loop). The probe is
    # never written back, so this is safe.
    probe.parent_graph = loop.parent_graph
    probe_compute = _single_compute_state(probe)
    if probe_compute is None:
        return False
    _rewrite_per_iter_bridges(probe_compute, probe.loop_variable, sdfg)
    return len(_independent_groups(probe_compute, probe, sdfg)) >= 2


def _container_per_iter_only(state: SDFGState, data: str, loop_var: Optional[str]) -> bool:
    """``True`` iff every memlet referencing ``data`` in ``state`` is per-iter."""
    for n in state.nodes():
        if not (isinstance(n, nodes.AccessNode) and n.data == data):
            continue
        for sub, _ in _subsets_at_node(n, state):
            if not _is_per_iter_subset(sub, loop_var):
                return False
    return True


def _rewrite_per_iter_bridges(state: SDFGState, loop_var: Optional[str], sdfg: SDFG):
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

    That last sentence is a PRECONDITION, not a description: the reader only
    sees the written value once the fission has actually put producer and
    consumer in separate loops. Until then the rewrite has merely deleted the
    edge that ordered the write before the read *within one iteration*, and the
    state says nothing about which runs first -- so the value would depend on
    the order codegen happens to pick. CALL THIS ONLY WHEN THE FISSION IS GOING
    TO BE APPLIED; to ask whether it would, use
    :func:`_fissions_after_bridge_rewrite`, which decides on a copy.

    Has no effect if ``loop_var`` is ``None`` or if no per-iter shared
    container exists.

    :param sdfg: The SDFG owning ``state``'s data descriptors, passed in rather
        than read off ``state``: a detached copy of a loop (what the fission
        probe reasons about) has no ``.sdfg``.
    """
    if loop_var is None:
        return
    written = OrderedSet(n.data for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0)
    for data in list(written):
        desc = sdfg.arrays.get(data)
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


def _has_self_path(state: SDFGState, data: str) -> bool:
    """True if the state's dataflow graph contains a directed path from an
    :class:`AccessNode` of ``data`` to another (or the same) :class:`AccessNode`
    of ``data``. Such a self-path captures a live-in/live-out recurrence like a
    scalar accumulator update.
    """
    starts = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == data]
    if len(starts) < 2 and not any(state.out_degree(n) > 0 and state.in_degree(n) > 0 for n in starts):
        return False
    for start in starts:
        seen = OrderedSet()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur is not start and isinstance(cur, nodes.AccessNode) and cur.data == data:
                return True
            for e in state.out_edges(cur):
                stack.append(e.dst)
    return False


def _is_accumulator_group(group: List[nodes.Node], state: SDFGState, sdfg: SDFG) -> bool:
    """True if ``group`` contains a transient scalar that is both read and written
    in the same state, indicating a loop-carried accumulator update. Per-iteration
    transient temporaries (produced and consumed within one iteration) are excluded
    because they have no self-path.
    """
    data_nodes = [n for n in group if isinstance(n, nodes.AccessNode)]
    written_data = {n.data for n in data_nodes if state.in_degree(n) > 0}
    for data in written_data:
        desc = sdfg.arrays.get(data)
        if desc is None or getattr(desc, 'transient', False) is False:
            continue
        if _has_self_path(state, data):
            return True
    return False


def _written_data(groups: List[List[nodes.Node]], state: SDFGState) -> Set[str]:
    """Containers written by AccessNodes in ``groups``."""
    return {n.data for g in groups for n in g if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0}


def _consumed_by_sibling_loop(state: SDFGState, data: Set[str]) -> bool:
    """True if a sibling :class:`LoopRegion` in the same parent CFG reads any
    container in ``data``. Used after fission to keep per-element producer loops
    together when a downstream consumer loop reads them, avoiding an unnecessary
    further split.
    """
    loop = state.parent_graph
    if loop is None or not hasattr(loop, 'parent_graph'):
        return False
    parent = loop.parent_graph
    if parent is None:
        return False
    for block in parent.nodes():
        if not isinstance(block, LoopRegion):
            continue
        # Skip the loop itself, and skip the original loop when this state is a
        # deepcopy probe (the probe shares the parent's nodes, so the original loop
        # would otherwise look like a sibling of itself).
        if block is loop or getattr(block, 'label', None) == getattr(loop, 'label', None):
            continue
        reads, _ = _block_rw(block)
        if reads & data:
            return True
    return False


def _merge_side_write_groups(groups: List[List[nodes.Node]],
                             state: SDFGState,
                             loop_var: Optional[str],
                             sdfg: SDFG,
                             sibling_check: bool = True) -> List[List[nodes.Node]]:
    """Merge side-write groups when they are part of a compound reduction body
    (one scalar accumulator group plus multiple per-element side writes) or when
    they feed a sibling consumer loop. Both cases are value-preserving: keeping the
    side-write statements in a single loop preserves their original order, and any
    consumer sees the fully-produced per-element arrays after that loop completes.
    """
    if loop_var is None or len(groups) < 2:
        return groups
    acc_idxs = {i for i, g in enumerate(groups) if _is_accumulator_group(g, state, sdfg)}
    side_idxs = [i for i in range(len(groups)) if i not in acc_idxs]
    if len(side_idxs) <= 1:
        return groups
    side_writes = _written_data([groups[i] for i in side_idxs], state)
    # Merge side writes when an accumulator is present in the same body (the
    # compound-reduction pattern) or when a sibling loop consumes the side writes
    # (so the producer loop should not be split further after fission).
    consumed = _consumed_by_sibling_loop(state, side_writes) if sibling_check else False
    if not acc_idxs and not consumed:
        return groups
    order = {n: i for i, n in enumerate(state.nodes())}
    merged: List[nodes.Node] = []
    kept: List[List[nodes.Node]] = []
    for i, g in enumerate(groups):
        if i in acc_idxs:
            kept.append(g)
        else:
            merged.extend(g)
    if merged:
        seen = set()
        deduped = [n for n in merged if not (n in seen or seen.add(n))]
        kept.append(sorted(deduped, key=lambda n: order[n]))
    return sorted(kept, key=lambda g: order[g[0]])


def _independent_groups(state: SDFGState,
                        loop: Optional[LoopRegion],
                        sdfg: SDFG,
                        sibling_check: bool = True) -> List[List[nodes.Node]]:
    """Partition ``state``'s nodes into data-independent groups.

    A *pure input* is an AccessNode with no in-edges whose data is never
    written in the state -- a read-only loop input. Such nodes do not connect
    their consumers (each fissioned loop re-reads the input). Non-input nodes
    are grouped by dataflow connectivity, then groups touching a common
    container that is *written* in the state are merged (RAW/WAW/WAR). Each
    returned group also carries the input nodes feeding it, so cloning then
    pruning to a group keeps a self-contained body.

    When the loop variable is known and a non-transient container is accessed
    *only* per-iteration (``a[loop_var]`` everywhere) the producer/consumer
    bridge through that container is severed in both the dataflow union and
    the container-shared merge: sequential loop fission preserves the value
    in that case. TSVC s221 (``a[i] = a[i] + c[i] * d[i]; b[i] = b[i-1] +
    a[i] + d[i]``) fissions into two loops under this rule.

    Two chains that touch the same written container are merged only when the
    SMT oracle cannot prove their live ranges are disjoint across loop
    iterations. Renaming a chain (giving it a different container name) is no
    longer the only way to make it fissionable.

    :param state: The loop body state.
    :param loop: The enclosing ``LoopRegion``. ``None`` keeps the legacy
        strict-merge behaviour.
    :param sdfg: The SDFG owning ``state``'s data descriptors, passed in rather
        than read off ``state``: a detached copy of a loop (what the fission
        probe reasons about) has no ``.sdfg``.
    :returns: A list of node lists, one per independent group, deterministic.
    """
    order = {n: i for i, n in enumerate(state.nodes())}
    written = OrderedSet(n.data for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0)
    is_input = OrderedSet(n for n in state.nodes()
                          if isinstance(n, nodes.AccessNode) and state.in_degree(n) == 0 and n.data not in written)
    core = [n for n in state.nodes() if n not in is_input]
    parent: Dict[nodes.Node, nodes.Node] = {n: n for n in core}
    loop_var = loop.loop_variable if loop is not None else None

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
        # Skip per-iter non-transient containers for the dependence-aware merge:
        # write-then-read at the same loop index can be safely sequenced into
        # sibling loops (the producer finishes all writes before the consumer
        # reads). The bridges are rewritten in :func:`_fission` to give each clone
        # its own reader. Keep the legacy name-based merge for the *write* nodes
        # of such a container: two overwrites to ``A[i]`` in the same body must
        # stay in one group even after the bridge rewrite severs their ordering
        # edge.
        per_iter_non_transient = False
        if loop_var is not None:
            desc = sdfg.arrays.get(data)
            per_iter_non_transient = (desc is not None and not getattr(desc, 'transient', False)
                                      and _container_per_iter_only(state, data, loop_var))
        if per_iter_non_transient:
            write_nodes = [
                n for n in state.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == data and state.in_degree(n) > 0
            ]
            reps = []
            for n in write_nodes:
                reps += group_of(n)
            for r in reps[1:]:
                union(reps[0], r)
            continue

        access_nodes = sorted((n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == data),
                              key=lambda n: order[n])
        for i, n1 in enumerate(access_nodes):
            reps1 = group_of(n1)
            if not reps1:
                continue
            subs1 = list(_subsets_at_node(n1, state))
            for n2 in access_nodes[i + 1:]:
                reps2 = group_of(n2)
                if not reps2:
                    continue
                subs2 = list(_subsets_at_node(n2, state))
                dependent = False
                for sub1, is_w1 in subs1:
                    for sub2, is_w2 in subs2:
                        if is_w1 or is_w2:
                            if _accesses_interfere_across_iterations(loop, sub1, sub2):
                                dependent = True
                                break
                    if dependent:
                        break
                if dependent:
                    for r in reps2:
                        union(reps1[0], r)

    classes: Dict[nodes.Node, List[nodes.Node]] = {}
    for n in core:
        classes.setdefault(find(n), []).append(n)
    groups = []
    for members in classes.values():
        member_set = OrderedSet(members)
        feeders = [n for n in is_input if any(e.dst in member_set for e in state.out_edges(n))]
        groups.append(sorted(members + feeders, key=lambda n: order[n]))
    groups = sorted(groups, key=lambda g: order[g[0]])
    return _merge_side_write_groups(groups, state, loop_var, sdfg, sibling_check=sibling_check)


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
                    # Decide on a copy, then mutate: the bridge rewrite is only sound once the fission
                    # separates producer and consumer into their own loops, so a loop that turns out not to
                    # fission must come out exactly as it went in.
                    if not _fissions_after_bridge_rewrite(loop, sdfg):
                        continue
                    _rewrite_per_iter_bridges(compute, loop.loop_variable, sdfg)
                    self._fission(loop, compute, sdfg)
                else:
                    groups = _independent_block_groups(loop)
                    if groups is None:
                        continue
                    self._fission_blocks(loop, groups)
                count += 1
                changed = True
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
            parent.add_node(clone, ensure_unique_name=True)  # derived label; wired by object ref
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
    def _fission(loop: LoopRegion, compute: SDFGState, sdfg: SDFG):
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
        ngroups = len(_independent_groups(compute, loop, sdfg, sibling_check=False))

        clones: List[LoopRegion] = []
        for gi in range(ngroups):
            clone = copy.deepcopy(loop)
            clone.label = f"{loop.label}_fis{gi}"
            parent.add_node(clone, ensure_unique_name=True)  # derived label; wired by object ref
            cstate = list(clone.nodes())[cidx]
            keep = set(_independent_groups(cstate, clone, sdfg, sibling_check=False)[gi])
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
