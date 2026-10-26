# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift every parallelizable loop of an SDFG to a Map, outermost-first."""

import contextlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dace import properties, subsets, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import SDFG, nodes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlets_state
from dace.sdfg.sdfg import forget_facts, memoize_facts
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.loop_to_map import (
    UNCOMPUTED,
    LiftContext,
    LiftInvariants,
    LoopFacts,
    LoopLift,
    LoopRefusal,
    Refusal,
    build_lift_context,
    build_lift_invariants,
    index_block_symbols,
    spans_small_symbols,
)
from dace.transformation.passes.analysis import array_facts, loop_analysis


class ProbeSession:
    """What the loop probes of one analysis share while the graph stands: one symbol resolver and, per SDFG, the
    lift invariants and context (:class:`LiftContext`). Valid until the graph changes: a pass that changes an SDFG
    calls :meth:`forget` on it before probing it again."""

    def __init__(self) -> None:
        self.resolver = SymbolResolver()
        self.invariants: dict[SDFG, LiftInvariants] = {}
        self.contexts: dict[SDFG, LiftContext] = {}
        self.loop_facts: dict[LoopRegion, LoopFacts] = {}

    def bind(self, xform: LoopLift, sdfg: SDFG) -> LoopLift:
        """``xform`` set up to probe the loops of ``sdfg`` with what this session holds."""
        inv = self.invariants.get(sdfg)
        if inv is None:
            inv = self.invariants[sdfg] = build_lift_invariants(sdfg)
        ctx = self.contexts.get(sdfg)
        if ctx is None:
            ctx = self.contexts[sdfg] = build_lift_context(sdfg, inv, self.loop_facts)
        xform.lift_context = ctx
        xform.symbol_resolver = self.resolver
        return xform

    def forget(self, sdfg: SDFG) -> None:
        """Drop what this session derived for ``sdfg`` and the SDFGs nested in it, after a change to ``sdfg``."""
        within = set(sdfg.all_sdfgs_recursive())
        for nested in within:
            self.invariants.pop(nested, None)
            self.contexts.pop(nested, None)
        for loop in [loop for loop in self.loop_facts if loop.sdfg in within]:
            del self.loop_facts[loop]
        self.resolver.forget_within(sdfg)


#: The ``pipeline_results`` key under which :class:`ParallelizeLoops` leaves ``{loop: LoopRefusal}`` for every loop it
#: probed and kept a loop, in probe order; the passes after it in a pipeline act on these loops only.
REFUSALS = "ParallelizeLoops.refusals"

#: Added to the hint of a map whose stores were proven disjoint through array facts.
FACTS_NOTE = "stores proven disjoint by array facts: {facts}"


def candidate_loops(sdfg: SDFG) -> list[LoopRegion]:
    """Every loop in ``sdfg`` and its nested SDFGs that carries an iteration variable."""
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable]


def edge_assignment_counts(region: ControlFlowRegion) -> Counter[str]:
    """How many interstate edges of ``region`` and its sub-regions (not across nested SDFGs) assign each name."""
    return Counter(k for e in region.all_interstate_edges() for k in e.data.assignments)


@dataclass(slots=True)
class LiftSite:
    """What a lift is about to change, read off the graph just before ``LoopLift.rewrite``: the loop's
    region is the only one whose own blocks and edges change, and the loop's subtree moves as a whole."""

    #: the region's own blocks, in block order
    level_order: list[Any]
    #: the loop's own entry plus every block of its subtree in the SDFG's block order
    subtree_size: int
    loop_states: OrderedSet
    region_edges: Counter
    loop_edges: Counter
    inner_loops: list[LoopRegion]

    @staticmethod
    def capture(loop: LoopRegion) -> "LiftSite":
        region = loop.parent_graph
        return LiftSite(
            level_order=list(cfg_analysis.blockorder_topological_sort(region, recursive=False)),
            subtree_size=1 + sum(1 for _ in cfg_analysis.blockorder_topological_sort(loop)),
            loop_states=OrderedSet(loop.states()),
            region_edges=Counter(k for e in region.edges() for k in e.data.assignments),
            loop_edges=edge_assignment_counts(loop),
            inner_loops=[r for r in candidate_loops(loop) if r is not loop],
        )


def last_iteration(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """The iteration variable's value in the last iteration: the loop's inclusive end moved onto its stride."""
    end = loop_analysis.get_loop_end(loop)
    start, step = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_stride(loop)
    if end is None or start is None or step is None or step in (1, -1):
        return end
    return symbolic.simplify(start + symbolic.int_floor(end - start, step) * step)


def lifted_blocks(site: LiftSite, loop: LoopRegion, region: ControlFlowRegion) -> list[SDFGState] | None:
    """The blocks the lift put in ``loop``'s place, if the region's block order is the old one with the loop
    replaced by them -- the one case in which the SDFG's block order can be patched instead of rebuilt."""
    level_after = list(cfg_analysis.blockorder_topological_sort(region, recursive=False))
    before = {id(b) for b in site.level_order}
    added = [b for b in level_after if id(b) not in before]
    if not added or not all(isinstance(b, SDFGState) for b in added):
        return None
    k = next(i for i, b in enumerate(site.level_order) if b is loop)
    expected = site.level_order[:k] + added + site.level_order[k + 1 :]
    if len(expected) != len(level_after) or any(a is not b for a, b in zip(expected, level_after)):
        return None
    return added


def patch_context(
    ctx: LiftContext, sd: SDFG, loop: LoopRegion, region: ControlFlowRegion, site: LiftSite, added: list[SDFGState]
) -> bool:
    """Bring ``sd``'s context up to date after ``loop`` was lifted, instead of rebuilding it from the whole SDFG.

    Every field is what :func:`build_lift_context` would compute now: the loop's subtree leaves the
    block order and the access index, the added states take its place, and the free symbols are the ones
    the lift handed over (or walked again, when it had none).

    :returns: ``False`` if the context cannot be patched and must be rebuilt.
    """
    start = ctx.block_index.get(loop)
    if start is None:
        return False
    ctx.block_order = ctx.block_order[:start] + added + ctx.block_order[start + site.subtree_size :]
    ctx.block_index = {block: i for i, block in enumerate(ctx.block_order)}
    for state in site.loop_states:
        for node in state.data_nodes():
            holders = ctx.access_states.get(node.data)
            if holders is not None:
                holders.discard(state)
    for state in added:
        for node in state.data_nodes():
            ctx.access_states[node.data].add(state)
    fresh = ctx.post_lift_free_symbols
    ctx.sdfg_free_symbols = set(sd.free_symbols) if fresh is None else fresh
    ctx.post_lift_free_symbols = None
    ctx.internalized_data = None
    if ctx.edge_assignments is not None:
        after = Counter(k for e in region.edges() for k in e.data.assignments)
        ctx.edge_assignments = ctx.edge_assignments - (site.region_edges + site.loop_edges) + after
    if ctx.symbol_uses is not None:
        for state in added:
            index_block_symbols(ctx, state)
    return True


def loop_order_key(loop: LoopRegion) -> tuple[int, int]:
    """Sort key placing OUTERMOST loops first: nested-SDFG level, then control-flow nesting depth."""
    sdfg_level = 0
    sd = loop.sdfg
    while sd is not None and sd.parent_sdfg is not None:
        sdfg_level += 1
        sd = sd.parent_sdfg

    depth = 0
    graph = loop.parent_graph
    while graph is not None and not isinstance(graph, SDFG):
        depth += 1
        graph = graph.parent_graph
    return (sdfg_level, depth)


@properties.make_properties
@transformation.explicit_cf_compatible
class ParallelizeLoops(ppl.Pass):
    """Lift every loop ``LoopToMap`` accepts into a Map, top-down, once per level.

    Each SDFG's loops are probed once, outermost first; a lifted loop's body becomes a nested SDFG, which the next
    level sweeps with the SDFGs already nested in maps. Outermost first because deepest-first lifts an inner loop
    into a nested SDFG whose outer memlet propagates wider, and the enclosing loop then fails ``LoopToMap``'s
    ``a*i+b`` write check. Every ``apply`` follows a full probe of the loop against the current graph; the one
    exception is :meth:`parallelize_loop` with ``proven=True``, for a caller whose own proof replaces the probe.

    The probes of one SDFG share its :class:`LiftContext` (types, access index, block order), patched after each
    lift. Memlets are propagated once, at the end. Code that lifts ONE given loop calls :meth:`parallelize_loop`.
    """

    CATEGORY: str = "Optimization Preparation"

    permissive = properties.Property(
        dtype=bool, default=False, desc="Probe loops in permissive mode, skipping the conservative refusals."
    )
    expand_buffers = properties.Property(
        dtype=bool,
        default=False,
        desc="Lift a refused loop whose refusal an expansion of its scratch buffers by one iteration-indexed "
        "dimension removes (BufferExpansion), expanding them",
    )
    propagate = properties.Property(
        dtype=bool,
        default=True,
        desc="Propagate memlets over the whole SDFG once, after the last lift. "
        "Set False only if the caller propagates itself.",
    )

    def __init__(self, permissive: bool = False, propagate: bool = True):
        self.permissive = permissive
        self.propagate = propagate

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Lift every liftable loop, top-down: the loops of each SDFG once, outermost first, then the SDFGs nested in
        it, which hold the bodies of its lifted loops; memlets are propagated once, at the end.

        :returns: The number of loops lifted, or ``None`` if none were.
        """
        applied = 0
        # Two lifetimes. The invariants are what a lift cannot change -- the symbol/array types, the
        # StructureView flag, and every block's free symbols -- so they are built once per SDFG and
        # never rebuilt. The contexts hold what a lift does change (the access-node index, the block
        # order, the cfg ids) and are patched or dropped after every lift.
        invariants: dict[SDFG, LiftInvariants] = {}
        contexts: dict[SDFG, LiftContext] = {}
        # loop -> the facts derived from its own body. Only a lift INSIDE a loop invalidates them.
        loop_facts: dict[Any, LoopFacts] = {}
        # LoopToMap and propagation re-parse bounds from strings; parse every name at the dtype some
        # SDFG of the tree declares, as canonicalize does, or a bound and a shape stop cancelling. An
        # enclosing authority (canonicalize's) already does, and keeps what a lift declares.
        scope = contextlib.nullcontext()
        if not symbolic.symbol_dtype_authority_active():
            scope = symbolic.serialization_symbol_dtypes(
                {name: dtype for nested in sdfg.all_sdfgs_recursive() for name, dtype in nested.symbols.items()}
            )
        with scope:
            # Every probe reads the facts of its loop's SDFG and of each SDFG nesting it; only a lift changes them,
            # and ``lift`` drops the lifted SDFG's.
            with memoize_facts():
                # One instance, reused: setting ``loop`` and probing overwrites every field a probe reads.
                xform = LoopLift()
                xform.symbol_resolver = SymbolResolver()
                level = [sdfg]
                while level:
                    deeper: list[SDFG] = []
                    for sd in level:
                        applied += self.sweep(sd, xform, pipeline_results, contexts, invariants, loop_facts)
                        deeper += [
                            node.sdfg
                            for state in sd.states()
                            for node in state.nodes()
                            if isinstance(node, nodes.NestedSDFG) and node.sdfg
                        ]
                    level = deeper
            if applied:
                self.finish(sdfg)
        return applied or None

    def parallelize_loop(self, sdfg: SDFG, loop: LoopRegion, proven: bool = False) -> bool:
        """Lift ONE loop the way :meth:`apply_pass` lifts each candidate, then run its post-lift steps.

        :param sdfg: the SDFG to prune and propagate after the lift; it must contain ``loop``.
        :param loop: the loop to lift, in ``sdfg`` or one of its nested SDFGs.
        :param proven: skip the ``LoopToMap`` probe. Only for a caller holding an independence proof
                       the probe cannot reconstruct: a wavefront legality proof, or a runtime guard
                       that rules out colliding writes. A loop over ``SMALL`` symbols still stays a loop.
        :returns: whether ``loop`` was lifted.
        """
        if not self.lift(LoopLift(), loop, {}, proven):
            return False
        self.finish(sdfg)
        return True

    def lift(
        self,
        xform: LoopLift,
        loop: LoopRegion,
        pipeline_results: dict[str, Any],
        proven: bool = False,
        before_apply: Callable[[], None] | None = None,
    ) -> bool:
        """Probe ``loop`` against the CURRENT graph and lift it if ``LoopToMap`` accepts it.

        :param xform: the instance to match; its ``lift_context``, when set, supplies the shared analysis.
        :param proven: skip the probe, see :meth:`parallelize_loop`.
        :param before_apply: called once the probe accepted, just before the graph changes.
        :returns: whether ``loop`` was lifted.
        """
        sd = loop.sdfg
        graph = loop.parent_graph
        xform.loop = loop
        facts: list[array_facts.ArrayFact] = []
        expanded: list[str] | None = None
        if not proven and not self.probe(xform, loop, facts):
            expanded = self.rescuing_buffers(xform, loop, facts)
            if expanded is None:
                return False
        # A known-parallel loop is mapped as it is: no sibling chain holds it, and its scale is the CPU specialization's
        # to settle.
        known_parallel = loop_analysis.is_known_parallel(loop)
        if not proven and not known_parallel and self.holds_back(xform, loop):
            return False
        if expanded:
            # avoid an import cycle
            from dace.transformation.passes.buffer_expansion import expand_iteration_private

            expand_iteration_private(sd, loop, expanded)
            # The expansion rewrote the loop's subsets and the buffers' shapes: what was derived from them is stale.
            if xform.lift_context is not None:
                xform.lift_context.loop_facts.pop(loop, None)
            if xform.symbol_resolver is not None:
                xform.symbol_resolver.forget_within(sd)
        # A proof of independence replaces the dependence test, not the scale rule: a SMALL loop stays sequential.
        if proven and not known_parallel and xform.keep_small_loops_sequential and spans_small_symbols(loop):
            return False
        if before_apply is not None:
            before_apply()
        end, last = loop_analysis.get_loop_end(loop), last_iteration(loop)
        itervar = loop.loop_variable
        existing = {n for st in graph.states() for n in st.nodes() if isinstance(n, nodes.MapEntry)} if facts else set()
        # The lifted states are propagated below, whole: the rewrite need not propagate each memlet it routes.
        propagate_edges, xform.propagate_edges = xform.propagate_edges, False
        try:
            with memoize_facts(active=False):
                xform.rewrite()
        finally:
            xform.propagate_edges = propagate_edges
        # The lift edits ``sd``'s descriptors, symbols and interstate edges, and builds a new nested SDFG. The one
        # edit outside ``sd``, the symbol mapping of the node nesting it, is not a fact of its parent.
        forget_facts(sd)
        # So the probes' resolver, once it forgets ``sd`` and what is nested in it, still holds what every enclosing
        # SDFG resolved, and the propagation below need not derive their facts again.
        resolver = xform.symbol_resolver
        if resolver is not None:
            resolver.forget_within(sd)
        for state in graph.states():
            over_itervar = [
                node for node in state.nodes() if isinstance(node, nodes.MapEntry) and node.map.params == [itervar]
            ]
            lifted = [node for node in over_itervar if node.map.range[0][1] == end]
            for node in lifted:
                if last is not None and last != end:
                    # LoopToMap ends the map at the loop bound; a strided map ends at its last iterate, or the memlets
                    # propagated over it reach past the array (s351: ``range(0, 4 * N, 4)`` reading ``b[i + 3]``).
                    rb, _, rs = node.map.range[0]
                    node.map.range = subsets.Range([(rb, last, rs)])
            for node in lifted:
                if facts and node not in existing:
                    note = FACTS_NOTE.format(facts="; ".join(f.describe() for f in facts))
                    node.specialization_hint = "\n".join(h for h in (node.specialization_hint, note) if h)
            if over_itervar:
                # LoopToMap connects the body's nested SDFG through whole arrays; its per-iteration accesses
                # (``T[k]``) are what lets a later fusion see one element per iteration. Only this state, so the
                # cost stays independent of ``propagate``.
                propagate_memlets_state(sd, state, resolver)
        return True

    def rescuing_buffers(
        self, xform: LoopLift, loop: LoopRegion, facts: list[array_facts.ArrayFact]
    ) -> list[str] | None:
        """After :meth:`probe` refused ``loop``: the buffers whose expansion makes the probe accept it, each one a
        refusal named and :func:`~dace.transformation.passes.buffer_expansion.rescues` admits, when
        :attr:`expand_buffers` is set; ``None`` otherwise. Changes nothing."""
        if not self.expand_buffers:
            return None
        # avoid an import cycle
        from dace.transformation.passes import buffer_expansion

        assumed = xform.assume_private
        buffers: list[str] = []
        try:
            while True:
                refusal = xform.last_refusal
                name = None if refusal is None else refusal.container
                if name is None or name in buffers or not buffer_expansion.rescues(loop.sdfg, loop, name):
                    return None
                buffers.append(name)
                xform.assume_private = assumed | frozenset(buffers)
                if self.probe(xform, loop, facts):
                    return buffers
        finally:
            xform.assume_private = assumed

    def probe(self, xform: LoopLift, loop: LoopRegion, facts: list[array_facts.ArrayFact]) -> bool:
        """``LoopToMap``'s probe of the matched ``loop``, retried with each container it refuses on for non-unique
        stores that the array facts prove conflict-free (:func:`array_facts.conflict_free_under_facts`) vouched
        for. The facts used are appended to ``facts``."""
        sd = loop.sdfg
        if xform.probe(self.permissive):
            if xform.solver_facts is not None:
                facts.extend(xform.solver_facts.used)
            return True
        vouched = xform.contract_arrays
        used: list[array_facts.ArrayFact] = []
        try:
            while xform.last_refusal is not None and xform.last_refusal.kind is Refusal.NON_UNIQUE_WRITE:
                name = xform.last_refusal.container
                if name is None or name in xform.contract_arrays:
                    return False
                proof = array_facts.conflict_free_under_facts(sd, loop, name)
                if proof is None:
                    return False
                used += [f for f in proof if f not in used]
                xform.contract_arrays = xform.contract_arrays | {name}
                if xform.probe(self.permissive):
                    facts.extend(used)
                    if xform.solver_facts is not None:
                        facts.extend(f for f in xform.solver_facts.used if f not in facts)
                    return True
            return False
        finally:
            xform.contract_arrays = vouched

    def accepts(
        self,
        loop: LoopRegion,
        xform: LoopLift | None = None,
        assume_private: frozenset[str] = frozenset(),
        closed_forms: dict[str, symbolic.SymbolicType] | None = None,
        session: ProbeSession | None = None,
    ) -> bool:
        """Whether a sweep would lift ``loop`` as the graph stands: :meth:`probe` with its array-facts retry, then
        :meth:`holds_back`. Changes nothing; the one entry for a pass asking whether a loop parallelizes.

        :param xform: the instance to probe with, when the caller supplies the shared analysis (``lift_context``) or
                      reads the refusal (``last_refusal``) afterwards.
        :param assume_private: containers the caller would make iteration-private before the lift
                               (:attr:`LoopLift.assume_private`); the verdict is the one for the rewritten loop.
        :param closed_forms: symbols the caller would rewrite into closed forms of the iteration
                             (:attr:`LoopLift.closed_forms`); the verdict is the one for the rewritten loop.
        :param session: the analysis the caller's probes share (:class:`ProbeSession`), bound to ``xform``.
        """
        xform = LoopLift() if xform is None else xform
        if session is not None:
            session.bind(xform, loop.sdfg)
        xform.loop = loop
        assumed, closed = xform.assume_private, xform.closed_forms
        xform.assume_private, xform.closed_forms = assume_private, closed_forms
        try:
            if not self.probe(xform, loop, []):
                return False
            return loop_analysis.is_known_parallel(loop) or not self.holds_back(xform, loop)
        finally:
            xform.assume_private, xform.closed_forms = assumed, closed

    def holds_back(self, xform: LoopLift, loop: LoopRegion) -> bool:
        """Whether to keep ``loop`` a loop although its probe accepted it; asked only after that probe, which refuses
        most candidates, so a subclass's own check runs on the few that pass.

        :param xform: the probe that accepted ``loop``, with the shared analysis it was given.
        """
        return False

    def finish(self, sdfg: SDFG) -> None:
        """The post-lift steps: drop stale nested-SDFG symbol mappings, then propagate memlets if asked."""
        prune_stale_symbol_mappings(sdfg)
        if self.propagate:
            propagate_memlets_sdfg(sdfg)

    def sweep(
        self,
        sd: SDFG,
        xform: LoopLift,
        pipeline_results: dict[str, Any],
        contexts: dict[SDFG, LiftContext],
        invariants: dict[SDFG, LiftInvariants],
        loop_facts: dict[Any, LoopFacts],
    ) -> int:
        """Probe each loop of ``sd`` once, outermost first, and lift the accepted ones; a lifted loop's inner loops
        move with its body into a nested SDFG, which the next level sweeps. A loop refused on an access of an inner
        loop lifted after it is probed once more.

        :param contexts: per-SDFG :class:`LiftContext` cache; a lift patches or drops the lifted SDFG's entry.
        :param invariants: per-SDFG facts a lift cannot change; built on first use, never rebuilt.
        :param loop_facts: per-loop body facts; dropped for the ancestors of each lift.
        :returns: the number of loops lifted.
        """
        refusals: dict[LoopRegion, LoopRefusal | None] = pipeline_results.setdefault(REFUSALS, {})
        # ``sd``'s free symbols as of the end of the last lift in it, handed over by ``apply`` rather than walked again
        # by the next ``build_lift_context``
        fresh_free_symbols = None
        loops = [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

        def visit(loop: LoopRegion) -> bool:
            """Probe ``loop`` and lift it if accepted; whether it was lifted."""
            nonlocal fresh_free_symbols
            # Moved into the body of an enclosing loop lifted before it: the next level probes it there
            if loop.sdfg is not sd:
                return False
            graph = loop.parent_graph
            inv = invariants.get(sd)
            if inv is None:
                inv = invariants[sd] = build_lift_invariants(sd)
            ctx = contexts.get(sd)
            if ctx is None:
                ctx = contexts[sd] = build_lift_context(sd, inv, loop_facts, fresh_free_symbols)
                fresh_free_symbols = None
            xform.lift_context = ctx
            # Read before ``lift``, which applies: a lift that edits this mapping moves the parent's free symbols.
            pnode = sd.parent_nsdfg_node
            mapping_keys = None if pnode is None else tuple(pnode.symbol_mapping.keys())
            sites: list[LiftSite] = []
            if not self.lift(xform, loop, pipeline_results, before_apply=lambda: sites.append(LiftSite.capture(loop))):
                # ``None``: accepted, then held back (a SMALL range under a proof, a sibling chain)
                refusals[loop] = xform.last_refusal
                return False
            refusals.pop(loop, None)
            site = sites[0]
            added = lifted_blocks(site, loop, graph)
            # Only ``sd``'s context is stale. A lift rewrites ``sd``'s own graph and builds a new nested SDFG; the one
            # thing it touches outside is the ``symbol_mapping`` of the node nesting ``sd``, which only the parent's
            # free symbols read. The lifted SDFG's context is patched in place when its region's block order allows
            # it; every enclosing SDFG's context loses the iterator from the loops it counts.
            if added is None or not patch_context(ctx, sd, loop, graph, site, added):
                del contexts[sd]
                fresh_free_symbols = ctx.post_lift_free_symbols
            if mapping_keys is not None and mapping_keys != tuple(pnode.symbol_mapping.keys()):
                contexts.pop(sd.parent_sdfg, None)
            owner = sd
            while owner is not None:
                kept = contexts.get(owner)
                if kept is not None and kept.loop_variables is not None:
                    kept.loop_variables = kept.loop_variables - Counter({loop.loop_variable: 1})
                owner = owner.parent_sdfg
            loop_facts.pop(loop, None)
            # An enclosing region's read/write sets are PATCHED, not dropped: a lift adds no access, and the only
            # names that leave are the body-local containers it internalized. The region's block order and its
            # loop-local-transient analysis genuinely change under it.
            gone = ctx.internalized_data
            region = graph
            while region is not None and not isinstance(region, SDFG):
                facts = loop_facts.get(region)
                if facts is not None:
                    if gone and facts.read_write is not UNCOMPUTED:
                        read_set, write_set = facts.read_write
                        facts.read_write = (read_set - gone, write_set - gone)
                    facts.block_order = UNCOMPUTED
                    facts.carried = UNCOMPUTED
                region = region.parent_graph
            return True

        order = sorted(loops, key=loop_order_key)
        # loop -> the iteration variables of the loops lifted inside it
        lifted_inside: dict[LoopRegion, set[str]] = {}

        def lift_and_note(loop: LoopRegion) -> int:
            enclosing = enclosing_loops(loop)
            if not visit(loop):
                return 0
            for outer in enclosing:
                lifted_inside.setdefault(outer, set()).add(loop.loop_variable)
            return 1

        applied = sum(lift_and_note(loop) for loop in order)
        # A loop refused on an access of an inner loop lifted since sees that access as the range the lift
        # propagated: probed once more, innermost first, so a chain of such loops lifts bottom-up.
        for loop in reversed(order):
            refusal = refusals.get(loop) if loop.sdfg is sd else None
            if refusal is not None and names_any(refusal, lifted_inside.get(loop, set())):
                applied += lift_and_note(loop)
        return applied


def enclosing_loops(loop: LoopRegion) -> list[LoopRegion]:
    """The loops enclosing ``loop`` within its SDFG, innermost first."""
    found = []
    region = loop.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            found.append(region)
        region = region.parent_graph
    return found


def names_any(refusal: LoopRefusal, names: set[str]) -> bool:
    """Whether a subset ``refusal`` names holds one of ``names``."""
    return any(
        str(sym) in names
        for subset in (*refusal.writes, *refusal.reads)
        if subset is not None
        for sym in subset.free_symbols
    )


def prune_stale_symbol_mappings(sdfg: SDFG) -> int:
    """Drop nested-SDFG ``symbol_mapping`` entries the callee no longer needs.

    A lift moves the loop's interstate assignments INTO the new body, so a symbol the enclosing
    SDFG used to receive from outside can end up produced inside it and stop being free there. The
    node wrapping that SDFG still maps it -- and a mapping's VALUE is what keeps a name alive in the
    CALLER -- so the entry holds a symbol one level up that nothing defines any more. It sits
    harmless until that scope is itself nested, which then has no value to pass for it: azimint_naive
    fails validation with ``Missing symbols on nested SDFG: ['__map_fusion___tmp0', ...]``, one lift
    making the entry stale and a later one tripping over it.

    ``free_symbols`` is the exact test -- it already excludes a symbol the callee assigns before use.
    Bottom-up, because pruning a node's mapping can retire its parent's entry too.

    :param sdfg: the SDFG to clean, recursively.
    :returns: how many entries were dropped.
    """
    removed = 0
    for block in sdfg.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        for node in block.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            removed += prune_stale_symbol_mappings(node.sdfg)
            keep = {str(sym) for sym in node.sdfg.free_symbols} | set(node.sdfg.arrays)
            # A connector's descriptor is compared in the caller's symbols, so its symbols stay mapped even when
            # the callee holds them as compile-time constants (``SDFG.specialize``), which ``free_symbols`` omits.
            for conn in node.in_connectors.keys() | node.out_connectors.keys():
                if conn in node.sdfg.arrays:
                    keep |= {str(sym) for sym in node.sdfg.arrays[conn].free_symbols}
            for key in [k for k in node.symbol_mapping if k not in keep]:
                del node.symbol_mapping[key]
                removed += 1
    return removed
