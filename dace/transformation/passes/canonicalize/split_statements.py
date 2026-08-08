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

from dace import SDFG, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.loop_fission import _single_compute_state


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
        stack = list(writers)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen[node] = None
            if isinstance(node, nodes.AccessNode):
                if node.data in input_names:
                    continue
                deps[node.data] = None
            for e in producer_edges(state, node):
                stack.append(e.src)
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
        seen: dict = dict.fromkeys(writers)
        stack = [e.src for w in writers for e in producer_edges(state, w)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen[node] = None
            if isinstance(node, nodes.AccessNode) and node.data in input_names:
                reads[node.data] = None
                continue
            stack.extend(e.src for e in producer_edges(state, node))
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
    def _split(parent_sdfg: SDFG, state, node: nodes.NestedSDFG, groups, simplify_cls, rmw_read_is_dead: bool = False):
        """Clone ``node`` once per group, prune each, rewire, drop original.

        :param rmw_read_is_dead: The caller established (via :func:`rmw_stays_in_writer_group`) that a
                                 read-modify-write array is read ONLY by the group that writes it. Only
                                 then may a non-writing group drop the connector; without that proof the
                                 array is a real dependency between the clones and must stay one.
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
                if e.src_conn in grp:
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
                groups = self._loop_output_groups(loop, cfg, local)
                if groups is None:
                    continue
                if self._split_one_loop(cfg, loop, groups, SimplifyPass, helpers, InlineMultistateSDFG, SubgraphView):
                    count += 1
        return count

    @staticmethod
    def _loop_output_groups(loop: LoopRegion, sdfg: SDFG, local: dict[str, None]):
        """The loop's independent output groups, or ``None`` to refuse -- computed WITHOUT touching it.

        :param loop: The loop to classify.
        :param sdfg: The SDFG owning ``loop``.
        :param local: The loop's own temporaries (:func:`loop_local_transients`); they become
                      private to each clone, so they are neither inputs nor outputs of the split.
        """
        from dace.transformation.passes.analysis import loop_analysis

        state = _single_compute_state(loop)
        if state is None:
            return None
        # A NestedSDFG body is _replicate_components' shape: it splits in place, no outlining needed.
        if any(isinstance(n, nodes.NestedSDFG) for n in state.nodes()):
            return None
        # Same black-box refusal as the other two paths: the clones recompute the whole body per
        # output, which is only sound while every node's effect is exactly its out-memlets.
        if any(is_opaque_code(n, sdfg) for n in state.nodes()):
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
        # A private temp read before anything writes it holds the PREVIOUS iteration's value (a
        # scalar rotation). Each clone gets its own copy and its own dead-code pruning, so the
        # producer of that value can be pruned out of the clone that consumes it.
        for n in state.data_nodes():
            if n.data in local and not producer_edges(state, n) and value_edges(state.out_edges(n)):
                return None

        out_names = [w for w in writes if w not in local]
        in_names = dict.fromkeys(r for r in reads if r not in local)
        if len(out_names) < 2:
            return None
        groups = independent_groups(loop, out_names, in_names)
        if len(groups) < 2:
            return None
        rmw = [o for o in out_names if o in in_names]
        if rmw and not rmw_confined(loop, in_names, rmw, groups):
            return None
        return SplitStatements._minimal_loops(groups, rmw)

    @staticmethod
    def _minimal_loops(groups: list[dict[str, None]], rmw: list[str]) -> list[dict[str, None]] | None:
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

        :param groups: The independent output groups.
        :param rmw: Names the loop both reads and writes.
        """
        carried: dict[str, None] = {}
        free: dict[str, None] = {}
        for grp in groups:
            (carried if any(o in rmw for o in grp) else free).update(grp)
        if not carried or not free:
            return None
        return [carried, free]

    @staticmethod
    def _split_one_loop(sdfg: SDFG, loop: LoopRegion, groups, simplify_cls, helpers, inline_cls, subgraph_cls) -> bool:
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
        SplitStatements._split(sdfg, outer, node, groups, simplify_cls, rmw_read_is_dead=True)
        for clone in [n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]:
            inline_cls.apply_to(sdfg, nested_sdfg=clone, save=False, verify=False)
        sdfg.reset_cfg_list()
        return True


__all__ = ['SplitStatements']
