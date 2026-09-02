# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement fission: split a loop/map body into one perfect nest per
independent output *statement*.

This is the single statement-fission pass of the canonicalization design. It
DISTRIBUTES a straight-line loop itself -- one loop per independent output --
rather than handing that to ``LoopFission`` / ``MapFission``; those passes stay
responsible for the shapes this one refuses. It handles the shapes ordinary
in-place fission cannot separate:

  * a **straight-line multi-output body** -- ``for i: a[i]=a[i-1]+x[i]; b[i]=y[i]*2``
    -- one output carries a sequential recurrence, the other is data-parallel;
  * **statements inside an if** -- ``for i: if c: A[i]=..; B[i]=..`` -- the body
    is a ``NestedSDFG`` holding a ``ConditionalBlock`` (the branch cannot be
    split in place);
  * **indirect (gather/scatter) access** -- ``for i: A[i]=B[idx[i]]; C[i]=D[idx[i]]``
    -- where ``idx[i]`` is an iterator-dependent interstate-edge symbol
    assignment that ``MapFission`` cannot hoist.

WHEN A SPLIT IS LEGAL
---------------------

A split is legal when running each group's statements as a WHOLE LOOP, one loop after the other,
computes what the fused loop computed. There are two ways that holds, and the pass implements both.

**1. Unordered siblings.** No group reads an array another group writes. The clones are independent,
go in ONE state, and their execution order does not matter. ``rmw_confined`` proves it;
``_split`` builds it. This is the original policy and the only one that admits three or more groups.

**2. Ordered loops.** A group DOES read an array another group writes, but one order of the two
loops reproduces the fused loop exactly. ``split_order`` decides which, ``_split_ordered`` builds it
by putting each clone in its own state, because an SDFG spells order as states.

The order follows from the DIRECTION of the cross-group access, per array and per read, comparing
the read's subset against the write's as a constant iteration offset (``access_offset``):

===================================  ==================  ==========================
cross-group access                   dependence          which loop runs first
===================================  ==================  ==========================
reader reads ``X[i + k]``, k > 0     anti (WAR)          the READER
reader reads ``X[i - k]``, k > 0     true, carried       the WRITER
reader reads ``X[i]``, same index    decided by the access node it reads from:
                                     one with producers is the post-write value (WRITER first),
                                     one without is the pre-write value (READER first)
===================================  ==================  ==========================

Every read of every shared array must agree on one order. Two arrays pulling opposite ways, an
offset that is not a constant, or a subset that is not a single point means no order is provably
right and the loop stands as it was.

Worked, both from TSVC::

    for i: a[i] = a[i] * c[i]              # A writes a[i]
           b[i] = b[i] + a[i+1] * d[i]     # B reads a[i+1] -- AHEAD

``s212``: the only cross-group touch of ``a`` is a read one ahead of the write, so B wants the
ORIGINAL values and B's loop goes first. Two parallel maps, no copies -- where the fused loop
otherwise pays a seam buffer and a sequential in-chunk sweep (``ChunkAntiDependence``)::

    for i: a[i] = a[i] + c[i] * d[i]       # A writes a[i]
           b[i] = b[i-1] + a[i] + d[i]     # B reads a[i] -- SAME index, post-write

``s221``: B reads exactly what A wrote, so A's loop goes first. A becomes a parallel map and B a
prefix-sum ``Scan`` -- from a wholly sequential loop with no map at all.

WHAT A CLONE MUST HAVE REMOVED FROM IT
--------------------------------------

A clone starts as a copy of the WHOLE body, so everything the group does not own has to go, and the
order of removal matters:

1. ``suppress_writes`` cuts the STORES to the other groups' arrays. Cutting the store rather than
   renaming the node is deliberate: one access node commonly carries a write and an unrelated read
   at disjoint subsets (``s212`` writes ``a[i]`` and reads ``a[i + 1]`` through the same node), and
   renaming would take the read with it.
2. ``drop_dataless_access_nodes`` splices out what that leaves holding only empty memlets. Those are
   ORDERING edges, so the node is spliced (predecessors reconnected to successors), not dropped.
3. ``SimplifyPass`` prunes the now-dead computation.
4. Only then is the connector set decided, from what the clone still moves a VALUE through -- not
   from which descriptors survive, because a connector array stays non-transient through
   ``SimplifyPass`` however dead it is.

Step 4 is not tidiness. An input connector nothing reads makes the inliner materialise an access
node for it in the parent, held by an ordering edge alone; ``LoopToMap`` then derives its body
SDFG's arrays from memlets, does not see it, and dies looking the node up. ``PruneConnectors`` runs
after ``LoopToMap`` in the pipeline as the general cleanup for the same class of leftover.

DISTRIBUTION IS ALL THIS PASS DOES. Breaking a dependence is a separate concern
and belongs to
:class:`~dace.transformation.passes.break_anti_dependence.BreakAntiDependence`:
a forward-read anti-dependence (``for i: A[i]=..; d[i]=A[i]+A[i+1]``, TSVC s1244)
binds two otherwise-independent statements, and that pass -- run with
``forward_reads=True`` -- snapshots the array so the read-ahead stops binding
them. The pipeline schedules the two one after the other; this module holds no
part of that rewrite.

A straight-line body reaches the split by two different routes. When the frontend
already wrapped it in a ``NestedSDFG`` the node is cloned in place. When it is a
FLAT ``SDFGState`` -- what a plain loop looks like at the 'prep' stage, with no
``NestedSDFG`` anywhere -- the whole ``LoopRegion`` is outlined into a
``NestedSDFG`` first (:meth:`SplitStatements._split_loop_bodies`), cloned per
output group, and each clone inlined back, so ONE loop becomes exactly TWO -- the
sequential recurrences and the data-parallel work -- without any later pass being
involved. The canonical form always prefers the parallel version, so the split is
never refused on cost grounds; it is declined only when there is no parallel work
to peel (:meth:`SplitStatements._minimal_loops`).

On the ``NestedSDFG`` route the node is cloned once per independent output group,
deep-copying the shared condition / index-symbol interstate assignments into every
clone (this subsumes the former ``ConditionalComponentFission``). A guard is
DUPLICABLE, so it is a shape the split handles rather than a precondition for it.
An in-place read-modify-write output is admitted only while the group that WRITES
the array is the only one that READS it -- see :func:`rmw_stays_in_writer_group`.

Parallelization of the resulting statements is done by the passes that follow
(``LoopToMap`` / ``MapFission`` on the shapes still nested); ``MapFusion``
re-fuses whatever should recombine.
"""
import copy
from typing import Any

from dace import SDFG, Memlet, dtypes, properties, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


def states_touch_view(states, arrays: dict) -> bool:
    """Whether any state names a ``View``: it owns no storage, so a clone cut from its binding
    edge is unbound, and the split refuses rather than try to carry the binding."""
    return any(isinstance(arrays[n.data], dt.View) for st in states for n in st.data_nodes() if n.data in arrays)


def neighbours_touch_view(state: SDFGState, node) -> bool:
    """Whether an AccessNode adjacent to ``node`` is a ``View`` (the split rewires those edges)."""
    arrays = state.sdfg.arrays
    return any(
        isinstance(arrays[n.data], dt.View) for e in state.all_edges(node) for n in (e.src, e.dst)
        if isinstance(n, nodes.AccessNode) and n.data in arrays)


def is_opaque_code(node, sdfg: SDFG) -> bool:
    """Whether ``node``'s effects are invisible to statement splitting.

    Splitting clones a body once per independent output group, so anything a node does
    beyond writing its out-memlets happens once per clone. Two black boxes qualify: a node
    that declares (or is detected to have) side effects, and a non-Python tasklet -- the
    detection behind :meth:`Tasklet.has_side_effects` walks a Python AST, so a C++ / MLIR
    body answers ``False`` for lack of anything to look at, not for lack of effects.

    :param node: The node to classify.
    :param sdfg: The SDFG owning ``node`` (for the side-effect query).
    """
    if not isinstance(node, nodes.CodeNode) or isinstance(node, nodes.NestedSDFG):
        return False
    if isinstance(node, nodes.Tasklet) and node.language != dtypes.Language.Python:
        return True
    return node.has_side_effects(sdfg)


def has_opaque_code(sdfg: SDFG) -> bool:
    """Whether any node anywhere in ``sdfg`` is opaque to splitting (:func:`is_opaque_code`)."""
    return any(is_opaque_code(node, parent.sdfg) for node, parent in sdfg.all_nodes_recursive())


def _has_conditional(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` contains a ``ConditionalBlock`` (recursively)."""
    return any(isinstance(cfg, ConditionalBlock) for cfg in sdfg.all_control_flow_regions(recursive=True))


def _has_interstate_assignments(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` has any interstate edge carrying an assignment.

    By the canonicalization convention these encode indirect-access index
    symbols (``__sym = idx[i]``); ``MapFission`` refuses to split a map whose
    NestedSDFG body has such a map-iterator-dependent assignment, so the
    NestedSDFG must be replicated per output group first.
    """
    return any(e.data.assignments for e in sdfg.all_interstate_edges())


def value_edges(edges) -> list:
    """The edges of ``edges`` that carry a VALUE, dropping the empty-memlet ordering edges.

    :param edges: The edges to filter.
    """
    return [e for e in edges if e.data is not None and not e.data.is_empty()]


def producer_edges(state, node) -> list:
    """``node``'s in-edges that carry a VALUE, dropping the empty-memlet ordering edges.

    An empty memlet constrains execution ORDER, not dataflow, so it must not make one statement
    look like a producer of another -- the frontend leaves such an edge between the two chains of
    ``s = x[i]; a[i] = a[i-1] + s; s = y[i]; b[i] = s * 2.0`` (the WAR on the reused name ``s``),
    and following it merges two independent outputs into one group and the split silently
    never fires.

    Dropping the constraint from the ANALYSIS is sound because of what the pass already refuses:
    :func:`has_opaque_code` bars any node whose effect is not exactly its out-memlets, so an
    ordering edge can only be protecting a memory hazard, never a side effect. ``_split`` then
    clones the WHOLE body per group, so a hazard with both endpoints in one clone keeps its edge
    verbatim; a hazard SPANNING two clones is either on a body transient (each clone owns a
    private copy -- the hazard dissolves) or on a connector array that is both read and written,
    which :func:`rmw_stays_in_writer_group` already confines to a single group.

    :param state: The state holding ``node``.
    :param node: The node whose producers are wanted.
    """
    return value_edges(state.in_edges(node))


def body_compute_states(loop: LoopRegion) -> list | None:
    """The loop body's non-empty states, or ``None`` when the body is not a plain chain of them.

    One state is what the frontend leaves for a simple body. A body that hands a value on through a
    transient (TSVC ``s3251``) is several states joined by unconditional, assignment-free edges --
    the same straight-line program written across more blocks, and splittable for the same reasons.
    A branch, an interstate assignment or a nested region is a different program and is refused.

    :param loop: The loop whose body is inspected.
    """
    from dace.sdfg import utils as sdutil
    if any(not isinstance(blk, SDFGState) for blk in loop.nodes()):
        return None
    for e in loop.edges():
        if e.data.assignments or e.data.condition.as_string not in ('1', 'True', '(1)'):
            return None
    # Execution order, not insertion order: which state wrote a value before another read it is
    # what :func:`split_order` reads off this list.
    states = [blk for blk in sdutil.dfs_topological_sort(loop, [loop.start_block]) if blk.nodes()]
    return states or None


def staged_producer_edges(body, state, node, input_names: dict[str, None]) -> list:
    """``(state, edge)`` for every VALUE producer of ``node``, following one staged transient back
    into the state that wrote it.

    A body of several states hands a value on through a transient: TSVC ``s3251`` computes
    ``c_index_0`` in one state and multiplies it in the next, so a cone walk that only looks inside
    one state stops on an access node with no in-edges and reads two dependent statements as
    independent. Continuing from the transient's stores elsewhere in the body is what makes the
    walk see the whole chain. A name in ``input_names`` is where the cone is MEANT to stop -- its
    value comes from outside the body -- so it is never followed.

    :param body: The body being walked.
    :param state: The state holding ``node``.
    :param node: The node whose producers are wanted.
    :param input_names: The names whose values enter ``body`` from outside.
    """
    own = [(state, e) for e in producer_edges(state, node)]
    if own or not isinstance(node, nodes.AccessNode) or node.data in input_names:
        return own
    return [(st, e) for st in body.all_states() if st is not state for n in st.data_nodes() if n.data == node.data
            for e in producer_edges(st, n)]


def loop_local_transients(loop: LoopRegion, sdfg: SDFG) -> dict[str, None]:
    """``sdfg``'s transients that nothing outside ``loop`` observes -- the loop's own temporaries.

    Read as a value by anything outside (an access node, an interstate-edge or region condition
    referring to it as a promoted scalar) disqualifies a name, so what remains is exactly what may
    be turned into a per-clone private copy.

    :param loop: The loop whose temporaries are wanted.
    :param sdfg: The SDFG owning ``loop``.
    """
    inner = dict.fromkeys(id(s) for s in loop.all_states())
    outside: dict[str, None] = {}
    for state in sdfg.all_states():
        if id(state) in inner:
            continue
        for n in state.data_nodes():
            outside[n.data] = None
    for e in sdfg.all_interstate_edges():
        outside.update(dict.fromkeys(s for s in e.data.free_symbols if s in sdfg.arrays))
    for cfr in sdfg.all_control_flow_regions(recursive=True):
        conditions = []
        if isinstance(cfr, ConditionalBlock):
            conditions = [c for c, _ in cfr.branches if c is not None]
        elif isinstance(cfr, LoopRegion) and cfr is not loop:
            conditions = [cfr.loop_condition]
        for cond in conditions:
            outside.update(dict.fromkeys(s for s in cond.get_free_symbols() if s in sdfg.arrays))
    return dict.fromkeys(nm for nm, desc in sdfg.arrays.items() if desc.transient and nm not in outside)


def _output_dependency(sdfg: SDFG, out_name: str, input_names: dict[str, None]) -> dict[str, None]:
    """Inner array names that feed ``out_name``, excluding pure shared inputs."""
    deps: dict[str, None] = {}
    for state in sdfg.all_states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen: dict = {}
        stack = [(state, w) for w in writers]
        while stack:
            st, node = stack.pop()
            if (st, node) in seen:
                continue
            seen[(st, node)] = None
            if isinstance(node, nodes.AccessNode):
                if node.data in input_names:
                    continue
                deps[node.data] = None
            stack.extend((s2, e.src) for s2, e in staged_producer_edges(sdfg, st, node, input_names))
    return deps


def output_input_reads(sdfg: SDFG, out_name: str, input_names: dict[str, None]) -> dict[str, None]:
    """Input-connector arrays whose values feed ``out_name``'s computation.

    The complement of :func:`_output_dependency`, which drops exactly these. Seeded from the
    writers' IN-EDGES rather than the writer nodes themselves, so an in-place output reports the
    read of its own array instead of stopping on it.

    :param sdfg: The nested-SDFG body to walk.
    :param out_name: The output connector whose producers are traced.
    :param input_names: The body's input connector names.
    """
    reads: dict[str, None] = {}
    for state in sdfg.all_states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen: dict = dict.fromkeys((state, w) for w in writers)
        stack = [(s2, e.src) for w in writers for s2, e in staged_producer_edges(sdfg, state, w, input_names)]
        while stack:
            st, node = stack.pop()
            if (st, node) in seen:
                continue
            seen[(st, node)] = None
            if isinstance(node, nodes.AccessNode) and node.data in input_names:
                reads[node.data] = None
                continue
            stack.extend((s2, e.src) for s2, e in staged_producer_edges(sdfg, st, node, input_names))
    return reads


def rmw_confined(body, in_names: dict[str, None], rmw: list[str], groups: list[dict[str, None]]) -> bool:
    """Whether every read-modify-write array is read ONLY by the group that writes it.

    The split clones the whole body once per group and the clones are unordered siblings, so an
    RMW array's carry survives only while its read and its write stay inside ONE clone. A read
    from another group runs against the array either before or after the writing clone's store.

    :param body: The body being cloned -- anything answering ``all_states`` /
                 ``all_control_flow_regions`` / ``all_interstate_edges``, i.e. a nested SDFG on the
                 NestedSDFG path and the ``LoopRegion`` itself on the loop path.
    :param in_names: The names whose values enter ``body`` from outside.
    :param rmw: Names that ``body`` both reads and writes.
    :param groups: The independent output groups.
    """
    # A branch guard / index-symbol assignment is duplicated into EVERY clone and re-evaluated
    # there against already-updated data, and it is not dataflow, so the walk below cannot see
    # which array it reads. Measured on TSVC s2710: ``a[i] = a[i] + b[i]*d[i]`` flips the guard
    # ``a[i] > b[i]``, and the else-arm's store to ``b`` then fires on if-arm lanes.
    if _has_conditional(body) or _has_interstate_assignments(body):
        return False
    # ``output_input_reads`` walks a single state at a time, so a producer chain that crosses
    # states is invisible to it -- only a one-state body is fully analyzable.
    if sum(1 for _ in body.all_states()) != 1:
        return False
    for grp in groups:
        reads: dict[str, None] = {}
        for oc in grp:
            reads.update(output_input_reads(body, oc, in_names))
        if any(r in reads for r in rmw if r not in grp):
            return False
    return True


def rmw_stays_in_writer_group(node: nodes.NestedSDFG, rmw: list[str], groups: list[dict[str, None]]) -> bool:
    """:func:`rmw_confined` for a NestedSDFG body, whose inputs are its input connectors.

    :param node: The NestedSDFG about to be cloned per group.
    :param rmw: Connector names that are both an input and an output of ``node``.
    :param groups: The independent output-connector groups.
    """
    return rmw_confined(node.sdfg, dict.fromkeys(node.in_connectors), rmw, groups)


def edge_subset(edge, name: str):
    """The subset ``edge`` touches of array ``name``, whichever end of the memlet carries it."""
    if edge.data is None or edge.data.is_empty():
        return None
    return edge.data.subset if edge.data.data == name else edge.data.other_subset


def subset_point(subset) -> list | None:
    """The single element ``subset`` addresses, one expression per dimension, else ``None``.

    Only a point access has a direction to compare: a slice covers several iterations at once and
    the read/write offset that decides the split order is not defined for it.
    """
    if subset is None:
        return None
    point = []
    for rb, re, _ in subset.ndrange():
        rb = rb.expr if isinstance(rb, symbolic.SymExpr) else rb
        re = re.expr if isinstance(re, symbolic.SymExpr) else re
        if symbolic.simplify(re - rb) != 0:
            return None
        point.append(rb)
    return point


def access_offset(read_subset, write_subset) -> int | None:
    """Sign of ``read - write`` as a constant iteration offset, or ``None`` when undecidable.

    Positive means the read is AHEAD of the write -- it wants an element a LATER iteration
    overwrites -- and negative means it is behind, wanting one an earlier iteration produced.
    Multi-dimensional accesses compare lexicographically, which is the order the iteration space is
    swept in. A difference that is not a plain integer leaves the direction unknown, and the caller
    must refuse rather than guess.
    """
    read, write = subset_point(read_subset), subset_point(write_subset)
    if read is None or write is None or len(read) != len(write):
        return None
    for r, w in zip(read, write):
        try:
            # Same-named symbols from two memlets can be different INSTANCES (the dtype is not part
            # of symbol identity), and then ``i + 1 - i`` stays an unevaluated Add instead of 1 --
            # the offset reads as unknown and every split is refused. Equalize before subtracting.
            r, w = symbolic.equalize_symbols(r, w)
            diff = symbolic.simplify(r - w)
        except (TypeError, ValueError, AttributeError):
            return None
        if not getattr(diff, 'is_Integer', False):
            return None
        if diff != 0:
            return 1 if diff > 0 else -1
    return 0


def output_read_edges(sdfg: SDFG, out_name: str, input_names: dict[str, None]) -> list:
    """``(state, access_node, edge)`` for every input array read in ``out_name``'s producer cone.

    :func:`output_input_reads` with the edges kept, because the SUBSET on the edge is what decides
    which way a cross-group dependence runs.

    :param sdfg: The body to walk.
    :param out_name: The output whose producers are traced.
    :param input_names: The names whose values enter the body from outside.
    """
    found = []
    for state in sdfg.all_states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen: dict = dict.fromkeys((state, w) for w in writers)
        stack = [pair for w in writers for pair in staged_producer_edges(sdfg, state, w, input_names)]
        while stack:
            st, edge = stack.pop()
            node = edge.src
            if isinstance(node, nodes.AccessNode) and node.data in input_names:
                found.append((st, node, edge))
                continue
            if (st, node) in seen:
                continue
            seen[(st, node)] = None
            stack.extend(staged_producer_edges(sdfg, st, node, input_names))
    return found


def write_subsets(body, name: str) -> list:
    """Every subset of ``name`` that ``body`` STORES to."""
    out = []
    for state in body.all_states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            out.extend(sub for sub in (edge_subset(e, name) for e in producer_edges(state, node)) if sub is not None)
    return out


def suppress_writes(body: SDFG, name: str) -> None:
    """Delete the stores to ``name`` inside ``body``, keeping every load of it.

    A clone that does not own ``name`` must not write it: the array is a connector, so a store
    reaches the parent's memory and clobbers what the owning clone produces. Deleting the producer
    edges (and the chain that dangles behind them) removes the STATEMENT while leaving reads of the
    same array untouched -- which matters because one access node commonly carries both, at disjoint
    subsets: TSVC ``s212`` writes ``a[i]`` and reads ``a[i + 1]`` through the same node, so renaming
    the node instead of cutting its in-edges would take the read with it.

    :param body: The clone to prune.
    :param name: The array whose stores go away.
    """
    from dace.sdfg import utils as sdutil
    for state in body.all_states():
        for node in [n for n in state.data_nodes() if n.data == name]:
            for e in producer_edges(state, node):
                sdutil.remove_edge_and_dangling_path(state, e)
            if node in state.nodes() and state.degree(node) == 0:
                state.remove_node(node)


def drop_dataless_access_nodes(body: SDFG) -> None:
    """Splice out what the pruning left with no live value traffic, keeping the order it carried.

    Cutting the other group's stores can leave an access node attached by nothing but empty
    memlets. Those are ORDERING edges, so the node is not dead weight to be dropped blindly -- it
    may sit between two nodes that must stay ordered, and the ordering is reconnected here rather
    than lost. The node itself has to go: it names an array while contributing no memlet, so a
    later pass that derives its read/write sets from memlets builds a body SDFG without that
    descriptor and then dies looking the node up (``LoopToMap`` on TSVC ``s211``, ``KeyError: 'e'``).

    A node whose STORED VALUE nothing reads any more is the same case one step later. The pruning
    can leave the other group's staging transient still written -- ``c[i - 1] -> c_index`` in TSVC
    ``s261`` -- with only ordering edges downstream. Its producer keeps a read of ANOTHER
    iteration's element alive, which is what makes the clone look loop-carried and costs it the
    map, and the ordering that read was under is vacuous once the read is gone. So the store goes
    too, its producer follows it, and only a TRANSIENT qualifies: a connector array's value leaves
    the clone and is read by definition.

    :param body: The clone to clean.
    """
    while True:
        removed = False
        read_for_value = {
            n.data
            for state in body.all_states()
            for n in state.data_nodes() if value_edges(state.out_edges(n))
        }
        for state in body.all_states():
            for node in [n for n in state.data_nodes() if not value_edges(state.out_edges(n))]:
                if (value_edges(state.in_edges(node))
                        and not (body.arrays[node.data].transient and node.data not in read_for_value)):
                    continue
                # Only the EMPTY in-edges carry order worth keeping; a value in-edge comes from the
                # dead producer swept below.
                preds = [e.src for e in state.in_edges(node) if e.data.is_empty()]
                succs = [e.dst for e in state.out_edges(node)]
                state.remove_node(node)
                for src in preds:
                    for dst in succs:
                        state.add_nedge(src, dst, Memlet())
                removed = True
            # A CodeNode whose out-edges are all EMPTY computes a value nobody reads: the node is
            # dead but its ORDERING is not, so it is spliced like an access node rather than
            # dropped. Keeping it is not an option -- the store that used its out-connector is what
            # was just cut, so the connector is left dangling and validation rejects the clone
            # (the frontend leaves exactly this edge between the two chains of
            # ``a[i] = a[i-1] + s1; b[i] = y[i] * 2``). Deleting the computation is sound because
            # :func:`has_opaque_code` already barred anything whose effect is not its out-memlets.
            for producer in [
                    n for n in state.nodes() if isinstance(n, nodes.CodeNode) and not value_edges(state.out_edges(n))
            ]:
                preds = [e.src for e in state.in_edges(producer) if e.data is not None and e.data.is_empty()]
                succs = [e.dst for e in state.out_edges(producer)]
                state.remove_node(producer)
                for src in preds:
                    for dst in succs:
                        state.add_nedge(src, dst, Memlet())
                removed = True
        if not removed:
            return


def rmw_analyzable(body) -> bool:
    """Whether the cross-group read/write picture of ``body`` can be read off its dataflow at all.

    Shared precondition of both split policies: a branch guard or an index-symbol assignment is
    re-evaluated inside every clone against already-updated data and is not dataflow. Several
    states are fine -- :func:`staged_producer_edges` follows a producer chain across them.
    """
    return not _has_conditional(body) and not _has_interstate_assignments(body)


def iteration_distinct(subsets, loop_var: str) -> bool:
    """Whether ``subsets`` name a DIFFERENT element on every iteration of ``loop_var``.

    A store the loop variable does not reach is the same element every iteration -- a value the
    loop CARRIES. Its reader cannot be put on either side of the writer: run the writer's loop
    first and the reader sees the final value in every iteration, run it second and it sees the
    initial one, while the fused loop saw the running one. So no order is legal, and the
    same-index rule in :func:`split_order` must not be allowed to pick one -- a carried scalar
    compares EQUAL to its own store, which is precisely the input that rule reads as "same
    element, same iteration" (``s = s + a[i]; b[i] = s`` is the shape).
    """
    return bool(subsets) and all(sub is not None and loop_var in (str(s) for s in sub.free_symbols) for sub in subsets)


def carries_across_iterations(body, name: str, in_names: dict[str, None]) -> bool:
    """Whether ``name``'s read-modify-write actually crosses iterations.

    A name the loop both reads and writes is not automatically a recurrence: ``a[i] = a[i] + x[i]``
    reads exactly the element it writes, one per iteration, and is data-parallel. Only a read of a
    DIFFERENT element -- ``e[i] = e[i - 1] * e[i - 1]`` -- is the carry that forces the loop to stay
    sequential. An unreadable offset, or a write the loop variable does not reach, answers ``True``:
    this only decides which statements are worth peeling into their own loop, so an over-cautious
    answer costs parallelism, never correctness.

    :param body: The body being split (the ``LoopRegion`` on the loop path).
    :param name: A name ``body`` both reads and writes.
    :param in_names: The names whose values enter ``body`` from outside.
    """
    stores = write_subsets(body, name)
    if not iteration_distinct(stores, body.loop_variable):
        return True
    for _state, node, edge in output_read_edges(body, name, in_names):
        if node.data == name and {access_offset(edge_subset(edge, name), w) for w in stores} != {0}:
            return True
    return False


def sees_written_value(states: list, name: str, state, node) -> bool:
    """Whether the read at ``node`` sees ``name``'s store from the SAME iteration.

    Within one state the answer is dataflow: an access node with producers is the post-write value,
    one without is what the iteration started with. Across states it is block order -- a store in an
    EARLIER state has already run, so a reader that has no producer of its own still sees the
    updated value (TSVC ``s2251`` writes ``a[i]`` in one state and reads it in the next).

    :param states: The body's states in execution order (:func:`body_compute_states`).
    :param name: The array in question.
    :param state: The state holding ``node``.
    :param node: The reading access node.
    """
    if producer_edges(state, node):
        return True
    if state not in states:
        return False
    before = states[:states.index(state)]
    return any(producer_edges(st, n) for st in before for n in st.data_nodes() if n.data == name)


def split_order(body, in_names: dict[str, None], rmw: list[str], groups: list[dict[str, None]]):
    """``groups`` re-ordered so the split is legal, or ``None`` when no order is.

    :func:`rmw_confined` decides whether the groups can run as UNORDERED siblings, which needs every
    read-modify-write array confined to the group that writes it. Ordering the two loops answers a
    weaker question and admits far more: a group that reads an array the other one writes is fine
    as long as it runs wholly on one side of that write.

    Which side is fixed by the direction of the access, per array and per read:

    * read AHEAD of the write (``a[i + 1]`` against a write to ``a[i]``) -- an anti-dependence. The
      reader wants the ORIGINAL values, so the reader's loop goes FIRST (TSVC ``s212``).
    * read BEHIND the write (``a[i - 1]``) -- a carried true dependence. The reader wants the
      finished values, so the writer's loop goes first.
    * read at the SAME index -- decided by which access node it comes from: a node with producers is
      the post-write value (writer first, TSVC ``s221``), one without is the pre-write value.

    All three rules read the shared array as one element PER ITERATION, so they only apply while the
    writes actually are per-iteration -- :func:`iteration_distinct` is the precondition, and a value
    the loop carries in a scalar fails it and refuses.

    Each read yields one PAIRWISE constraint -- this group before that one -- and the split is
    legal exactly when the constraints admit a total order, which is a topological sort of the
    groups. Two groups is the one-bit case (TSVC ``s212`` / ``s221``); three (``s3251``) is the
    same question asked twice. Constraints that disagree form a cycle, and a read whose direction
    is not a constant offset yields no constraint at all -- both leave the loop as it was.

    :param body: The body being split (the ``LoopRegion`` on the loop path).
    :param in_names: The names whose values enter ``body`` from outside.
    :param rmw: Names ``body`` both reads and writes.
    :param groups: The independent output groups.
    """
    if len(groups) < 2 or not rmw_analyzable(body):
        return None
    states = body_compute_states(body) or []
    # ``before[i]`` are the groups that must run after group ``i``; dicts, not sets, because the
    # sort below walks them and its result is the emitted state order.
    after: list[dict[int, None]] = [{} for _ in groups]
    constrained = False
    for name in rmw:
        writer = next((i for i, grp in enumerate(groups) if name in grp), None)
        if writer is None:
            return None
        stores: list | None = None
        for reader, grp in enumerate(groups):
            if reader == writer:
                continue
            for state, node, edge in [r for oc in grp for r in output_read_edges(body, oc, in_names)]:
                if node.data != name:
                    continue
                if stores is None:
                    # Asked here rather than per name: a name no OTHER group reads imposes no
                    # order, so what its writes look like never comes up. A value the loop carries
                    # is exactly that case once :func:`merge_carried_groups` has put its reader
                    # and its writer in one group.
                    stores = write_subsets(body, name)
                    if not iteration_distinct(stores, body.loop_variable):
                        return None
                read = edge_subset(edge, name)
                offsets = {access_offset(read, w) for w in stores}
                if len(offsets) != 1 or None in offsets:
                    return None
                offset = offsets.pop()
                if offset == 0:
                    # Same element, same iteration: whether the read already sees the write decides.
                    first = writer if sees_written_value(states, name, state, node) else reader
                else:
                    first = reader if offset > 0 else writer
                after[first][writer if first == reader else reader] = None
                constrained = True
    if not constrained:
        return None
    order = topological_group_order(after)
    return None if order is None else [groups[i] for i in order]


def topological_group_order(after: list[dict[int, None]]) -> list[int] | None:
    """``after[i]`` naming the groups that must follow group ``i``, the groups in a legal order,
    or ``None`` when the constraints contradict each other.

    Ties keep the groups' own order, so the emitted states do not depend on which read produced a
    constraint first.
    """
    pending = [0] * len(after)
    for succs in after:
        for j in succs:
            pending[j] += 1
    order: list[int] = []
    ready = [i for i, n in enumerate(pending) if n == 0]
    while ready:
        i = min(ready)
        ready.remove(i)
        order.append(i)
        for j in after[i]:
            pending[j] -= 1
            if pending[j] == 0:
                ready.append(j)
    return order if len(order) == len(after) else None


def merge_carried_groups(body, groups: list[dict[str, None]], rmw: list[str],
                         in_names: dict[str, None]) -> list[dict[str, None]]:
    """``groups`` with the producer of a CARRIED value and everything that reads it merged into one.

    A value the loop carries -- ``s`` in TSVC ``s2251``, written every iteration at the same element
    -- cannot be put on either side of its reader: run the writer's loop first and the reader sees
    the final value in every iteration, run it second and it sees the initial one, while the fused
    loop saw the running one. No order is legal, so the only way to split a body holding one is not
    to split around it. Merging its groups costs nothing and leaves the REST of the body free to
    distribute, which is the difference between one sequential loop and one sequential loop beside a
    map.

    :param body: The body being split.
    :param groups: The independent output groups.
    :param rmw: Names ``body`` both reads and writes.
    :param in_names: The names whose values enter ``body`` from outside.
    """
    merged = list(groups)
    for name in rmw:
        if iteration_distinct(write_subsets(body, name), body.loop_variable):
            continue
        writer = next((i for i, grp in enumerate(merged) if name in grp), None)
        if writer is None:
            continue
        readers = [
            i for i, grp in enumerate(merged)
            if i != writer and any(n.data == name for oc in grp for _s, n, _e in output_read_edges(body, oc, in_names))
        ]
        if not readers:
            continue
        for i in readers:
            merged[writer].update(merged[i])
        merged = [grp for i, grp in enumerate(merged) if i not in readers]
    return merged


def independent_groups(body, out_names: list[str], in_names: dict[str, None]) -> list[dict[str, None]]:
    """Partition ``out_names`` so two outputs share a group iff their producer cones overlap.

    :param body: The body to walk (see :func:`rmw_confined`).
    :param out_names: The output names, in a deterministic order.
    :param in_names: The names whose values enter ``body`` from outside.
    """
    dep = {oc: _output_dependency(body, oc, in_names) for oc in out_names}
    parent = {oc: oc for oc in out_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(out_names):
        for b in out_names[i + 1:]:
            if any(s in dep[b] for s in dep[a]):
                parent[find(a)] = find(b)
    groups: dict[str, dict[str, None]] = {}
    for oc in out_names:
        groups.setdefault(find(oc), {})[oc] = None
    return list(groups.values())


@transformation.explicit_cf_compatible
@properties.make_properties
class SplitStatements(ppl.Pass):
    """Split a loop/map body into one perfect nest per independent output
    statement -- including statements inside ifs and gather/scatter accesses
    (per-output NestedSDFG replication). Subsumes ConditionalComponentFission.
    Distribution only: breaking a dependence belongs to ``BreakAntiDependence``.

    ``split_loops`` peels the data-parallel statements of a STRAIGHT-LINE loop out
    of the sequential recurrences into a loop of their own, by outlining the
    ``LoopRegion``, cloning it per group and inlining the clones back. This is the
    flat-body route: at the 'prep' stage a plain loop body is an ``SDFGState`` with
    no ``NestedSDFG`` at all, so nothing else in this pass would ever see it. A
    loop with nothing to peel is left alone -- see :meth:`_minimal_loops`.

    ``split_maps`` additionally fissions a STRAIGHT-LINE map that writes several
    global outputs into one map per output (the map analogue of the loop split);
    a shared local temp is recomputed in each, never materialized. It is OFF by
    default so the canonicalization pipeline (which lowers maps to loops and
    fissions there) is byte-identical; the nest-forge agent path turns it on to
    fission at map granularity without lowering."""

    CATEGORY: str = 'Canonicalization'

    split_maps = properties.Property(
        dtype=bool, default=False, desc="Also fission a straight-line multi-global-output map into one map per output.")

    split_loops = properties.Property(dtype=bool,
                                      default=True,
                                      desc="Distribute a straight-line multi-output loop into one loop per output.")

    def __init__(self, split_maps: bool = False, split_loops: bool = True) -> None:
        super().__init__()
        self.split_maps = split_maps
        self.split_loops = split_loops

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        # (1) Statements inside ifs + gather/scatter: replicate the blocking
        #     NestedSDFG once per independent output group so MapFission splits it.
        count += self._replicate_components(sdfg)
        # (1b) Straight-line map with >=2 global outputs -> one map per output
        #      (opt-in; leaves the canon pipeline unchanged).
        if self.split_maps:
            count += self._split_map_bodies(sdfg)
        # (1c) Straight-line LOOP with >=2 independent outputs -> one loop per output. The flat-body
        #      route: no NestedSDFG exists yet, so the loop is outlined into one and split there.
        if self.split_loops:
            count += self._split_loop_bodies(sdfg)
        return count or None

    # ------------------------------------------------------------------
    # (1) Per-output replication of a MapFission-blocking NestedSDFG
    #     (conditional / indirection-symbol). Formerly ConditionalComponentFission.
    # ------------------------------------------------------------------

    def _replicate_components(self, sdfg: SDFG) -> int:
        from dace.transformation.passes.simplify import SimplifyPass

        count = 0
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for state in list(nsdfg.states()):
                for node in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    groups = self._independent_output_groups(state, node)
                    if groups is None or len(groups) < 2:
                        continue
                    self._split(nsdfg, state, node, groups, SimplifyPass, rmw_read_is_dead=True)
                    count += 1
        return count

    @staticmethod
    def _independent_output_groups(state, node: nodes.NestedSDFG):
        """Partition ``node``'s output connectors into independent groups, or ``None`` to refuse.

        A STRAIGHT-LINE body qualifies like a conditional / gather-scatter one: the guard and the
        ``sym = idx[i]`` assignment are duplicable per statement (``_split`` deep-copies them into
        every clone), not a precondition for splitting.
        """
        out_conns = list(node.out_connectors)
        if len(out_conns) < 2:
            return None
        if states_touch_view(node.sdfg.all_states(), node.sdfg.arrays) or neighbours_touch_view(state, node):
            return None
        # No WCR on the boundary (it would not be replicable per group).
        for e in state.out_edges(node):
            if e.data is None or e.data.wcr is not None:
                return None
        # In-place read-modify-write: output arrays that are ALSO input connectors. The carry has
        # to stay inside one statement, which :func:`rmw_stays_in_writer_group` decides once the
        # groups are known; the connector test here is the cheap way to skip it entirely.
        rmw = [c for c in out_conns if c in node.in_connectors]
        # A black-box body is not analyzable: ``_output_dependency`` reads the memlets, which
        # do not describe an opaque node's effects, and ``_split`` would then duplicate them.
        if has_opaque_code(node.sdfg):
            return None
        in_names = dict.fromkeys(node.in_connectors)
        result = independent_groups(node.sdfg, out_conns, in_names)
        if rmw and not rmw_stays_in_writer_group(node, rmw, result):
            return None
        return result

    @staticmethod
    def _split(parent_sdfg: SDFG,
               state,
               node: nodes.NestedSDFG,
               groups,
               simplify_cls,
               rmw_read_is_dead: bool = False,
               cut_other_stores: bool = False):
        """Clone ``node`` once per group, prune each, rewire, drop original.

        :param rmw_read_is_dead: The caller established (via :func:`rmw_stays_in_writer_group`) that a
                                 read-modify-write array is read ONLY by the group that writes it. Only
                                 then may a non-writing group drop the connector; without that proof the
                                 array is a real dependency between the clones and must stay one.
        :param cut_other_stores: Delete the other groups' STORES before simplifying, the way
                                 :meth:`_split_ordered` does. Flipping their array to a transient is
                                 not enough to make dead-code elimination take a recurrence away: the
                                 store to ``a[i]`` is what the read of ``a[i - 1]`` consumes, so the
                                 pair keeps each other live and the clone that owns neither still
                                 carries the whole chain -- and stays sequential for it.
        """
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        for grp in groups:
            # Both the read and the write of an RMW array are dead in a group that does not write it.
            # Dropping the connector lets the flip below make it transient so SimplifyPass prunes them
            # -- kept non-transient it stays a write with no output connector, which validation rejects.
            if rmw_read_is_dead:
                kept_in = [c for c in node.in_connectors if c not in node.out_connectors or c in grp]
            else:
                kept_in = list(node.in_connectors)
            clone_sdfg = copy.deepcopy(node.sdfg)
            # COMPENSATES FOR AN ATTACHMENT GAP, and is not the root fix. ``SDFG.__deepcopy__``
            # leaves a NESTED sdfg's cfg list empty, and ``add_nested_sdfg`` only propagates to the
            # regions that list already holds -- it never walks a subtree it has not seen. So a
            # deep-copied ``LoopRegion`` is never registered and the next ``parent_graph.cfg_id``
            # dies on it, arbitrarily far away (the vectorizer's re-run of canonicalize on polybench
            # ``deriche``). Rebuilding here is the cheapest place to restore the invariant from this
            # side; once ``add_nested_sdfg`` REGISTERS an unseen subtree rather than only
            # propagating, this call goes away.
            clone_sdfg.reset_cfg_list()
            if cut_other_stores:
                for other in [c for c in node.out_connectors if c not in grp]:
                    suppress_writes(clone_sdfg, other)
                drop_dataless_access_nodes(clone_sdfg)
                simplify_cls().apply_pass(clone_sdfg, {})
                # Cutting a store leaves the inputs only that statement read with no reader, and an
                # input connector nothing reads makes the inliner materialise an access node the next
                # pass to derive read sets from memlets does not know about (``LoopToMap`` on TSVC
                # ``s211``). Keep exactly what the clone still moves a value through.
                used = {
                    n.data
                    for st in clone_sdfg.all_states()
                    for n in st.data_nodes() if value_edges(st.in_edges(n) + st.out_edges(n))
                }
                kept_in = [c for c in kept_in if c in clone_sdfg.arrays and c in used]
            # dicts/sorted, not sets: these become the clone's connector dicts, which are
            # observable in validation and codegen order.
            clone = state.add_nested_sdfg(clone_sdfg,
                                          inputs=dict.fromkeys(kept_in),
                                          outputs=dict.fromkeys(sorted(grp)),
                                          symbol_mapping=dict(node.symbol_mapping))
            for e in in_edges:
                if e.dst_conn is not None and e.dst_conn not in kept_in:
                    continue
                state.add_edge(e.src, e.src_conn, clone, e.dst_conn, copy.deepcopy(e.data))
            for e in out_edges:
                # ``src_conn is None`` is an ORDERING edge, not an output: every clone inherits it.
                if e.src_conn is not None and e.src_conn not in grp:
                    continue
                state.add_edge(clone, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
            for arr in [c for c in clone_sdfg.arrays if c not in grp and c not in kept_in]:
                desc = clone_sdfg.arrays[arr]
                if not desc.transient:
                    desc.transient = True
            simplify_cls().apply_pass(clone_sdfg, {})
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)

    # ------------------------------------------------------------------
    # (1b) Straight-line map statement fission (opt-in via split_maps).
    # ------------------------------------------------------------------

    def _split_map_bodies(self, sdfg: SDFG) -> int:
        """Map analogue of the loop statement split: fission a STRAIGHT-LINE map writing >=2 global
        outputs into one map per output, so each is a self-contained statement.

        Reuses :meth:`_split`: nest the flat body into a NestedSDFG so the map reads as
        ``MapEntry -> NestedSDFG(out = the global outputs) -> MapExit``, then clone that NestedSDFG once
        per output with STRICT per-output groups. ``SimplifyPass`` (inside ``_split``) drops each clone's
        dead compute while keeping a shared local producer that still feeds the kept output -- so a temp
        feeding two outputs is RECOMPUTED in each map, never materialized to an array (the over-split
        MapFission would otherwise promote it). The downstream ``MapFission`` then separates the cloned
        bodies into distinct maps.

        Only PLAIN leaf maps: a body holding a NestedSDFG (a conditional / gather-scatter index symbol)
        is left to :meth:`_replicate_components`, which already duplicates the guard / ``sym = idx[i]``
        assignment per output. A WCR output (reduction) is never split.
        """
        from dace.transformation.passes.simplify import SimplifyPass
        from dace.transformation import helpers
        from dace.transformation.interstate import InlineSDFG
        from dace.sdfg.graph import SubgraphView

        count = 0
        for cfg in list(sdfg.all_sdfgs_recursive()):
            for state in list(cfg.states()):
                entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
                for entry in entries:
                    if entry not in state.nodes():  # a prior split in this state removed/replaced it
                        continue
                    if self._split_one_map(cfg, state, entry, SimplifyPass, helpers, InlineSDFG, SubgraphView):
                        count += 1
        return count

    @staticmethod
    def _split_one_map(cfg, state, entry, simplify_cls, helpers, inline_cls, subgraph_cls) -> bool:
        """Split one straight-line multi-output map into one FLAT map per output; return whether it fired.

        Nest the WHOLE scope (entry..exit) into a NestedSDFG, clone it per output (``_split`` duplicates a
        shared local + prunes the dead output), then inline each clone back so the result is flat maps --
        not maps buried in NestedSDFGs. The inline is confined to the clones this call made (captured by
        diff), so unrelated NestedSDFGs are untouched.
        """
        xit = state.exit_node(entry)
        # Global outputs = distinct arrays written through the exit; bail on a WCR (reduction) edge.
        out_names: list[str] = []
        for e in state.in_edges(xit):
            if e.data is None or e.data.data is None:
                continue
            if e.data.wcr is not None:
                return False
            if e.data.data not in out_names:
                out_names.append(e.data.data)
        if len(out_names) < 2:
            return False
        # In-place read-modify-write: a GLOBAL array read into the scope is also written out of it
        # (``t = A[i]; A[i] = t + B[i]; C[i] = t*2``). Cloning the scope per output would let one clone
        # recompute ``t = A[i]`` AFTER another clone overwrote ``A[i]`` (the read/modify/write order is
        # lost) and duplicate the write -- both silently wrong, and it still validates. Mirror the RMW
        # guard SplitTasklets uses (split_tasklets.py) and leave such a map unsplit. Global
        # (non-transient) arrays only: a shared local temp is meant to be recomputed per output.
        read_arrays = dict.fromkeys(
            e.data.data for e in state.in_edges(entry)
            if e.data is not None and e.data.data is not None and not cfg.arrays[e.data.data].transient)
        write_arrays = dict.fromkeys(
            e.data.data for e in state.in_edges(xit)
            if e.data is not None and e.data.data is not None and not cfg.arrays[e.data.data].transient)
        if any(a in write_arrays for a in read_arrays):
            return False
        scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        if any(
                isinstance(cfg.arrays[n.data], dt.View) for n in scope.nodes()
                if isinstance(n, nodes.AccessNode) and n.data in cfg.arrays):
            return False
        inner = [n for n in scope.nodes() if n not in (entry, xit)]
        # A PLAIN dataflow NestedSDFG body (a dependent read-after-write the frontend wrapped, e.g.
        # ``A[i]=..; B[i]=A[i]*2``) is inlined FIRST, so the split sees flat tasklets and a shared local
        # is duplicated like any other. A CONDITIONAL / indirection-symbol body is NOT inlined -- it
        # cannot live directly in a map scope and is _replicate_components' job.
        if (len(inner) == 1 and isinstance(inner[0], nodes.NestedSDFG) and not _has_conditional(inner[0].sdfg)
                and not _has_interstate_assignments(inner[0].sdfg)):
            inline_cls.apply_to(cfg, nested_sdfg=inner[0], save=False, verify=False)
            scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            inner = [n for n in scope.nodes() if n not in (entry, xit)]
        # PLAIN leaf map only: no nested map / NestedSDFG in the body (those go to _replicate_components).
        if not inner or any(isinstance(n, (nodes.NestedSDFG, nodes.MapEntry, nodes.MapExit)) for n in inner):
            return False
        # Same black-box refusal as the NestedSDFG path: the clones below recompute the whole
        # body per output, which is only sound while every node's effect is its out-memlets.
        if any(is_opaque_code(n, cfg) for n in inner):
            return False
        before = dict.fromkeys(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
        nsdfg_node = helpers.nest_state_subgraph(cfg, state, subgraph_cls(state, list(scope.nodes())))
        groups = [dict.fromkeys([o]) for o in nsdfg_node.out_connectors if o in out_names]
        if len(groups) < 2:  # nesting coalesced the outputs onto one connector -- nothing to split
            return False
        SplitStatements._split(cfg, state, nsdfg_node, groups, simplify_cls)
        for clone in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]:
            inline_cls.apply_to(cfg, nested_sdfg=clone, save=False, verify=False)
        return True

    # ------------------------------------------------------------------
    # (1c) Straight-line LOOP statement fission: one loop per output.
    # ------------------------------------------------------------------

    def _split_loop_bodies(self, sdfg: SDFG) -> int:
        """Loop analogue of :meth:`_split_map_bodies`: fission a STRAIGHT-LINE loop writing >=2
        outputs into one LOOP per independent output.

        The map path only has to nest the scope, because ``nest_state_subgraph`` swallows entry and
        exit together and cloning the result already yields two maps. A loop has no such scope node,
        so the mechanism is a different one: outline the whole ``LoopRegion`` into a ``NestedSDFG``
        (:func:`~dace.transformation.helpers.nest_sdfg_subgraph`), clone THAT per output group with
        the same :meth:`_split`, and inline every clone back with ``InlineMultistateSDFG`` -- each
        clone carries its own copy of the loop, so one loop comes out as one loop per statement.

        Everything the split refuses is decided BEFORE the outlining
        (:meth:`_loop_output_groups`), because the outlining is not free to undo: a refusal must
        leave the SDFG byte-identical.
        """
        from dace.transformation.passes.simplify import SimplifyPass
        from dace.transformation import helpers
        from dace.transformation.interstate import InlineMultistateSDFG
        from dace.sdfg.graph import SubgraphView

        count = 0
        for cfg in list(sdfg.all_sdfgs_recursive()):
            for loop in [r for r in cfg.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                local = loop_local_transients(loop, cfg)
                decision = self._loop_output_groups(loop, cfg, local)
                if decision is None:
                    continue
                groups, ordered = decision
                if self._split_one_loop(cfg, loop, groups, ordered, SimplifyPass, helpers, InlineMultistateSDFG,
                                        SubgraphView):
                    count += 1
        return count

    @staticmethod
    def _loop_output_groups(loop: LoopRegion, sdfg: SDFG, local: dict[str, None]):
        """``(groups, ordered)`` for the loop, or ``None`` to refuse -- computed WITHOUT touching it.

        ``ordered`` says the clones must run one after the other, in the order ``groups`` gives,
        rather than as the unordered siblings :meth:`_split` produces.

        :param loop: The loop to classify.
        :param sdfg: The SDFG owning ``loop``.
        :param local: The loop's own temporaries (:func:`loop_local_transients`); they become
                      private to each clone, so they are neither inputs nor outputs of the split.
        """
        from dace.transformation.passes.analysis import loop_analysis

        states = body_compute_states(loop)
        if states is None:
            return None
        body_nodes = [n for state in states for n in state.nodes()]
        # A NestedSDFG body is _replicate_components' shape: it splits in place, no outlining needed.
        if any(isinstance(n, nodes.NestedSDFG) for n in body_nodes):
            return None
        # Same black-box refusal as the other two paths: the clones recompute the whole body per
        # output, which is only sound while every node's effect is exactly its out-memlets.
        if any(is_opaque_code(n, sdfg) for n in body_nodes):
            return None
        if states_touch_view(states, sdfg.arrays):
            return None
        # Both clones must sweep the SAME iteration space, so the trip count may not depend on
        # anything the body writes: a counting loop whose bounds are pure symbols.
        if not loop.loop_variable or not loop.init_statement:
            return None
        if any(s in sdfg.arrays for s in loop.loop_condition.get_free_symbols()):
            return None
        init = loop_analysis.get_init_assignment(loop)
        if init is None or loop.loop_variable in symbolic.free_symbols_and_functions(init):
            return None
        # A counter something outside the loop reads is EXPORTED by the outlining as an extra
        # scalar output, which the per-output clones would each have to write. Leave those alone.
        if loop_analysis.counter_used_outside_loop(loop.loop_variable, loop, sdfg):
            return None

        # Empty memlets are ORDERING edges, so they neither read nor write: counting one as a write
        # turns a plain input into a read-modify-write output and refuses the whole split.
        reads: dict[str, None] = {}
        writes: dict[str, None] = {}
        for state in states:
            for n in state.data_nodes():
                if value_edges(state.out_edges(n)):
                    reads[n.data] = None
                stores = producer_edges(state, n)
                if not stores:
                    continue
                writes[n.data] = None
                # A WCR store is a reduction; it is not replicable per group.
                if any(e.data.wcr is not None for e in stores):
                    return None
        # A private temp read BEFORE anything writes it holds the PREVIOUS iteration's value (a
        # scalar rotation). Each clone gets its own private copy, so a clone that only reads such a
        # temp would read one that was never written. "Before" spans the whole body: a temp written
        # in one state and read in the NEXT is an ordinary staged value, not a carry.
        for state in states:
            for n in state.data_nodes():
                if (n.data in local and value_edges(state.out_edges(n))
                        and not sees_written_value(states, n.data, state, n)):
                    return None

        out_names = [w for w in writes if w not in local]
        in_names = dict.fromkeys(r for r in reads if r not in local)
        if len(out_names) < 2:
            return None
        groups = independent_groups(loop, out_names, in_names)
        if len(groups) < 2:
            return None
        rmw = [o for o in out_names if o in in_names]
        groups = merge_carried_groups(loop, groups, rmw, in_names)
        if len(groups) < 2:
            return None
        if not rmw or rmw_confined(loop, in_names, rmw, groups):
            minimal = SplitStatements._minimal_loops(loop, groups, rmw, in_names)
            return None if minimal is None else (minimal, False)
        # The clones cannot stand as unordered siblings, but ONE order of the two loops may still
        # reproduce the fused loop exactly -- see :func:`split_order`. That is the whole of TSVC
        # s212 (reader first) and s221 (writer first), neither of which the sibling split reaches.
        ordered = split_order(loop, in_names, rmw, groups)
        return None if ordered is None else (ordered, True)

    @staticmethod
    def _minimal_loops(body, groups: list[dict[str, None]], rmw: list[str],
                       in_names: dict[str, None]) -> list[dict[str, None]] | None:
        """Coalesce the output groups into the FEWEST loops that free the parallel work: TWO.

        What the split is FOR is peeling the data-parallel statements out of a loop that carries a
        recurrence. So there are only ever two kinds of group: CARRIED (it writes an array the loop
        also reads, and that recurrence is what forces the loop to stay sequential) and FREE (it
        reads nothing the loop writes -- ``rmw_confined`` has already proved no other group reads a
        carried array). Every FREE group is parallel together with every other, and every CARRIED
        group is already sequential, so one loop of each is the whole benefit; anything finer just
        adds full-length passes over the data. A reduction hand-unrolled into eleven accumulators is
        the case that makes this concrete -- eleven carried groups, eleven sweeps, no parallelism
        gained, and the re-roll that would have lifted it to ONE ``Reduce`` no longer matches.

        Returns ``None`` when one side is empty: nothing to peel, and the loop stands as it was.

        CARRIED is decided per NAME by :func:`carries_across_iterations`, not by membership in
        ``rmw``: ``a[i] = a[i] + b[i] * c[i]`` reads and writes ``a`` yet carries nothing, and
        reading it as carried puts the loop's only parallel statement on the sequential side and
        refuses the split for want of a free group (TSVC ``s222``).

        :param body: The body being split.
        :param groups: The independent output groups.
        :param rmw: Names the loop both reads and writes.
        :param in_names: The names whose values enter ``body`` from outside.
        """
        recurrences = [n for n in rmw if carries_across_iterations(body, n, in_names)]
        carried: dict[str, None] = {}
        free: dict[str, None] = {}
        for grp in groups:
            (carried if any(o in recurrences for o in grp) else free).update(grp)
        if not carried or not free:
            return None
        return [carried, free]

    @staticmethod
    def _split_one_loop(sdfg: SDFG, loop: LoopRegion, groups, ordered: bool, simplify_cls, helpers, inline_cls,
                        subgraph_cls) -> bool:
        """Outline ``loop``, clone it per group, inline the clones back; return whether it fired.

        ``nest_sdfg_subgraph`` moves the loop's own temporaries INSIDE the nest (they are its
        ``unique_set``), so each clone gets a private copy and a shared temp is recomputed per group
        rather than handed between the clones through one outer array.
        """
        outer = helpers.nest_sdfg_subgraph(sdfg, subgraph_cls(loop.parent_graph, [loop]))
        sdfg.reset_cfg_list()
        node = next(n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG))
        # The outlining decides the connector set itself; only split when it agrees with what the
        # refusal above was computed from, otherwise put the loop back untouched.
        covered = dict.fromkeys(o for grp in groups for o in grp)
        if dict.fromkeys(node.out_connectors) != covered:
            inline_cls.apply_to(sdfg, nested_sdfg=node, save=False, verify=False)
            sdfg.reset_cfg_list()
            return False
        before = dict.fromkeys(n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG))
        if ordered:
            clones = SplitStatements._split_ordered(outer, node, groups, simplify_cls)
            if clones is None:
                inline_cls.apply_to(sdfg, nested_sdfg=node, save=False, verify=False)
                sdfg.reset_cfg_list()
                return False
        else:
            # ``rmw_read_is_dead``: the sibling split only runs once ``rmw_confined`` has proved a
            # read-modify-write array is read by nobody but its writer, so the read is dead in
            # every other clone. The ordered split makes no such claim -- there the read is real
            # and the order is what keeps it correct -- so it keeps every input connector.
            SplitStatements._split(sdfg,
                                   outer,
                                   node,
                                   groups,
                                   simplify_cls,
                                   rmw_read_is_dead=True,
                                   cut_other_stores=True)
            clones = [n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]
        for clone in clones:
            inline_cls.apply_to(sdfg, nested_sdfg=clone, save=False, verify=False)
        sdfg.reset_cfg_list()
        return True

    @staticmethod
    def _split_ordered(state, node: nodes.NestedSDFG, groups, simplify_cls):
        """Clone ``node`` once per group into CONSECUTIVE states; the clones, or ``None`` to refuse.

        :meth:`_split` puts every clone in ONE state, where they are unordered siblings -- which is
        exactly why it may only separate outputs no other group reads. Giving the clones an order
        is what makes the other split legal: a group that reads an array the other one writes runs
        wholly before that write (seeing the original values) or wholly after it (seeing the
        updated ones). An SDFG spells order as states, so each clone after the first gets its own.

        :param state: The state holding ``node`` -- the FIRST group's clone stays here.
        :param node: The outlined loop to clone.
        :param groups: The output groups, already in execution order.
        :param simplify_cls: ``SimplifyPass``, to prune each clone down to its own group.
        """
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        # A later group's clone needs its own access nodes, which can only be rebuilt from an
        # AccessNode neighbour; anything else and the caller puts the loop back untouched.
        if any(not isinstance(e.src, nodes.AccessNode) for e in in_edges):
            return None
        if any(not isinstance(e.dst, nodes.AccessNode) for e in out_edges):
            return None
        # An ORDERING edge cannot be re-homed: the clones land in CONSECUTIVE states, so a copy in
        # a later state constrains nothing about the endpoint left behind in the first. Refuse.
        if any(e.data is not None and e.data.is_empty() for e in in_edges + out_edges):
            return None
        parent = state.parent_graph
        clones = []
        target = state
        for idx, grp in enumerate(groups):
            if idx:
                target = parent.add_state_after(target, label=f'{state.label}_split_{idx}')
            # Prune the clone BEFORE it is wired: dropping the other groups' stores leaves inputs
            # only they read, and SimplifyPass deletes those descriptors. A connector whose inner
            # array is gone is not a valid NestedSDFG, so the surviving descriptors decide the
            # connector set rather than the other way round.
            clone_sdfg = copy.deepcopy(node.sdfg)
            for other in [c for c in node.out_connectors if c not in grp]:
                suppress_writes(clone_sdfg, other)
            drop_dataless_access_nodes(clone_sdfg)
            for arr in [c for c in clone_sdfg.arrays if c not in grp and c not in node.in_connectors]:
                desc = clone_sdfg.arrays[arr]
                if not desc.transient:
                    desc.transient = True
            # The clone is still detached, so its cfg list is empty and SimplifyPass cannot find a
            # root; the unordered split gets one for free by adding the node before simplifying.
            clone_sdfg.reset_cfg_list()
            simplify_cls().apply_pass(clone_sdfg, {})
            # A CONNECTOR array stays non-transient through SimplifyPass even once the pruning has
            # left nothing reading it, so descriptor presence is the wrong test. An input connector
            # nothing reads is not merely untidy: the inliner materialises an access node for it in
            # the parent, held by an ordering edge alone, and the next pass to derive read sets from
            # memlets builds a body without that descriptor and dies on the node (``LoopToMap`` on
            # TSVC ``s211``). Keep exactly what the clone still moves a value through.
            used = {
                n.data
                for st in clone_sdfg.all_states()
                for n in st.data_nodes() if value_edges(st.in_edges(n) + st.out_edges(n))
            }
            kept_in = [c for c in node.in_connectors if c in clone_sdfg.arrays and c in used]
            if any(o not in clone_sdfg.arrays for o in grp):
                return None
            for leftover in [c for c in clone_sdfg.arrays if c not in kept_in and c not in grp]:
                clone_sdfg.arrays[leftover].transient = True
            # dicts/sorted, not sets: these become the clone's connector dicts, which are
            # observable in validation and codegen order.
            clone = target.add_nested_sdfg(clone_sdfg,
                                           inputs=dict.fromkeys(kept_in),
                                           outputs=dict.fromkeys(sorted(grp)),
                                           symbol_mapping=dict(node.symbol_mapping))
            for e in in_edges:
                if e.dst_conn not in kept_in:
                    continue
                src = e.src if target is state else target.add_access(e.src.data)
                target.add_edge(src, e.src_conn, clone, e.dst_conn, copy.deepcopy(e.data))
            for e in out_edges:
                if e.src_conn not in grp:
                    continue
                dst = e.dst if target is state else target.add_access(e.dst.data)
                target.add_edge(clone, e.src_conn, dst, e.dst_conn, copy.deepcopy(e.data))
            clones.append(clone)
        endpoints = dict.fromkeys([e.src for e in in_edges] + [e.dst for e in out_edges])
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)
        # The first group's clone stays in ``state`` and rewires only ITS OWN edges, so an access
        # node that served only a later group is left behind with nothing attached.
        for n in endpoints:
            if n in state.nodes() and state.degree(n) == 0:
                state.remove_node(n)
        return clones


__all__ = ['SplitStatements']
