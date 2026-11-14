# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse two consecutive loops with identical bodies over adjacent index ranges.

A hand-tiled loop nest arrives split into a *main-body* loop over the largest
multiple-of-``K`` prefix and a step-1 *remainder* loop over the trailing
``< K`` elements::

    for i in range(0, (N // K) * K):   # main body (after re-roll: unit stride)
        acc += a[i]
    for i in range((N // K) * K, N):   # remainder
        acc += a[i]

Both loops run the *same* body over *adjacent* index ranges, so together they
sweep ``range(0, N)``. Left split, a reduction like the one above lifts to two
separate ``Reduce`` library nodes writing the same accumulator -- a shape whose
seed does not chain across the two nodes (the second re-seeds the accumulator
and drops the first's partial sum). Fusing the two loops back into one
``for i in range(0, N)`` removes the split so a single ``Reduce`` is lifted and
the result is exact.

Concatenating two adjacent iteration ranges ``[A, B)`` and ``[B, C)`` under the
*same* stride and the *same* body executes the body for the same index sequence
in the same order as a single loop over their union, so every loop-carried value
is identical. That union is ``[A, C)`` only for ``A <= B <= C``; an empty side
makes it ``[min(A, B), max(B, C))``, which holds in every case and is what the
fused loop runs when the order is not provable. The match is deliberately
conservative -- both loops must be
single-state, unit-stride, directly consecutive (one plain interstate edge
between them, nothing else), and structurally identical up to their iteration
variable -- so it fires only on the re-rolled tile/remainder shape and its kin,
never on unrelated adjacent loops.
"""

import re
from collections.abc import Callable, Container
from typing import Any

import sympy

import dace
from dace import symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import Edge
from dace.sdfg.narrowing import as_expr
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

#: Per-SDFG context: data name -> states referencing an AccessNode for it. A merge deletes a
#: whole body state, which changes this, so it is rebuilt once per sweep rather than cached
#: across an ``apply`` (see ``apply_pass``); never rebuilt per candidate pair.
ScratchIndex = dict[str, OrderedSet[SDFGState]]

#: Iterator- and scratch-name-independent key of a body node: ``(kind, payload)``.
NodeKey = tuple[str, str]
#: Interned value keys: two nodes with one id compute one value, in any body they appear in.
ValueIds = dict[tuple, int]

#: Placeholder the iteration variable is normalised to when comparing two loop
#: bodies, so ``a[_loop_it_0]`` and ``a[_loop_it_1]`` compare equal.
ITER_PLACEHOLDER = "__lv__"

#: Placeholder every body-local scratch transient name is normalised to, so two
#: bodies differing only in a frontend-generated intermediate name (e.g.
#: ``s0_plus_a_slice`` vs ``s0_plus_a_slice_0``) compare equal. The carried
#: accumulator and the read/written arrays -- which are referenced OUTSIDE the
#: body and so are not body-local -- keep their real names and must match.
SCRATCH_PLACEHOLDER = "__scratch__"


def _int_floor_to_sympy(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Rewrite DaCe ``int_floor(a, b)`` sub-terms as SymPy ``floor(a / b)``.

    ``int_floor`` is opaque to :func:`sympy.simplify`, so a genuine identity
    such as ``int_floor(N - 11, 11) + 1 == int_floor(N, 11)`` is left unproven.
    SymPy's ``floor`` knows ``floor(x + k) == floor(x) + k`` for integer ``k``,
    which is exactly what an adjacency check between a tile bound and a
    remainder start needs.
    """
    return expr.replace(
        lambda x: x.func.__name__ == "int_floor" and len(x.args) == 2, lambda x: sympy.floor(x.args[0] / x.args[1])
    )


def _symbolically_equal(a: symbolic.SymbolicType, b: symbolic.SymbolicType) -> bool:
    """Whether two symbolic expressions are provably equal, floor-aware.

    :param a: First expression.
    :param b: Second expression.
    :returns: ``True`` iff ``a - b`` simplifies to zero (after rewriting
              ``int_floor`` to SymPy ``floor``), else ``False``.
    """
    try:
        # The two expressions come from different sources (two memlets, a subset against a reparsed
        # bound), so a shared name can carry two symbol instances with different dtypes. Subtraction
        # goes through identity, not name, and would leave ``i - i`` uncancelled.
        pa, pb = symbolic.equalize_symbols_across(symbolic.pystr_to_symbolic(a), symbolic.pystr_to_symbolic(b))
        diff = symbolic.simplify(as_expr(pa) - as_expr(pb))
        if diff == 0:
            return True
        return symbolic.simplify(_int_floor_to_sympy(diff)) == 0
    except Exception:
        return False


def value_keys(
    state: SDFGState,
    node_key: Callable[[nodes.Node], Any],
    edge_key: Callable[[dace.Memlet], tuple],
    ids: ValueIds,
    stored: dict[str, int],
    inputs_of: Container[nodes.Node] | None = None,
) -> dict[nodes.Node, int] | None:
    """An interned key per node of ``state``, equal only for nodes computing one value: it folds in the keys of the
    node's sources, wired by connector, so one set of tasklets and memlets connected differently differs.

    :param node_key: What a node is by itself; ``None`` refuses.
    :param edge_key: What a memlet reads.
    :param ids: The interning table, shared by everything compared.
    :param stored: Per container, the key of its last write; carries values into the next state.
    :param inputs_of: The nodes whose sources count, all if ``None``; the key of any other is ``node_key`` alone.
    :returns: The keys, or ``None`` if ``node_key`` refused.
    """
    keys: dict[nodes.Node, int] = {}
    for node in sdutil.dfs_topological_sort(state):
        payload = node_key(node)
        if payload is None:
            return None
        inputs = []
        if inputs_of is None or node in inputs_of:
            # A memlet-path edge has None for a connector, a View's 'views'.
            inputs = [
                (keys[e.src], e.src_conn or "", e.dst_conn or "", *edge_key(e.data))
                for e in state.in_edges(node)
                if not e.data.is_empty()
            ]
        if not inputs and isinstance(node, nodes.AccessNode) and node.data in stored:
            keys[node] = stored[node.data]  # a read of what an access node before it wrote
            continue
        keys[node] = ids.setdefault((payload, tuple(sorted(inputs))), len(ids))
        if inputs and isinstance(node, nodes.AccessNode):
            stored[node.data] = keys[node]
    return keys


def provably_ordered(low: symbolic.SymbolicType, high: symbolic.SymbolicType, facts: symbolic.Facts) -> bool:
    """Whether ``low <= high`` is provable, floor-aware."""
    low, high = symbolic.equalize_symbols_across(as_expr(low), as_expr(high))
    return symbolic.provably_le(low, high, facts) or bool(
        symbolic.simplify(_int_floor_to_sympy(high - low)).is_nonnegative
    )


def _normalize(text: str, loop_var: str) -> str:
    """Replace whole-word occurrences of ``loop_var`` in ``text`` with the
    canonical placeholder so two bodies differing only in iterator name match."""
    return re.sub(r"\b%s\b" % re.escape(loop_var), ITER_PLACEHOLDER, text)


def _canon_data(name: str, local_scratch: dict[str, None]) -> str:
    """Map a body-local scratch transient to the canonical placeholder; leave
    carried / external names (accumulator, arrays) untouched."""
    return SCRATCH_PLACEHOLDER if name in local_scratch else name


def _node_key(node: nodes.Node, loop_var: str, local_scratch: dict[str, None]) -> NodeKey:
    """A structural key for a body node, iterator- and scratch-name-independent."""
    if isinstance(node, nodes.AccessNode):
        return ("access", _canon_data(node.data, local_scratch))
    return ("tasklet", _normalize(node.code.as_string.strip(), loop_var))


@explicit_cf_compatible
class FuseConsecutiveLoops(ppl.Pass):
    """Fuse two directly-consecutive, identical-bodied, unit-stride loops whose
    iteration ranges are adjacent (``[A, B)`` followed by ``[B, C)``) into a
    single loop over ``[A, C)``."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Fuse every matching consecutive-loop pair in ``sdfg`` and its nested
        SDFGs, repeating until no pair matches (a chain of tiles collapses one
        adjacency per sweep).

        :param sdfg: SDFG to mutate in place.
        :returns: The number of fusions performed, or ``None`` if none.
        """
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                # Rebuilt once per sweep, not per candidate pair: a merge deletes a body
                # state, which is exactly what this index tracks, so it goes stale the
                # moment one fires -- the next sweep iteration rebuilds it fresh.
                scratch_index = self._build_scratch_index(sd)
                for cfg in list(sd.all_control_flow_regions()):
                    if self._fuse_one(cfg, scratch_index):
                        fused += 1
                        changed = True
                        break
        return fused or None

    def _build_scratch_index(self, sd: ControlFlowRegion) -> ScratchIndex:
        """Map every AccessNode data name in ``sd`` to the states referencing it."""
        index: ScratchIndex = {}
        for st in sd.states():
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode):
                    index.setdefault(n.data, OrderedSet()).add(st)
        return index

    def _fuse_one(self, cfg: ControlFlowRegion, scratch_index: ScratchIndex) -> bool:
        """Find and fuse one consecutive-loop pair inside ``cfg``.

        :param cfg: The control-flow region to search (one level; loops nested
                    deeper are reached via ``all_control_flow_regions``).
        :param scratch_index: Current sweep's data-name -> states map (see ``apply_pass``).
        :returns: ``True`` if a pair was fused.
        """
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            link = out_edges[0]
            second = link.dst
            if not isinstance(second, LoopRegion) or second is first:
                continue
            if len(cfg.in_edges(second)) != 1:
                continue
            # The connecting edge must be pure sequencing: no assignments and a
            # trivial condition, so nothing runs (or is decided) between them.
            if link.data.assignments:
                continue
            if link.data.condition is not None and link.data.condition.as_string not in ("1", "True"):
                continue
            if self._adjacent_identical(first, second, scratch_index):
                self._merge(cfg, first, second, link)
                return True
        return False

    def _adjacent_identical(self, first: LoopRegion, second: LoopRegion, scratch_index: ScratchIndex) -> bool:
        """Whether ``first`` then ``second`` are unit-stride, structurally identical, cover adjacent index ranges
        ``[A, B)`` and ``[B, C)``, and nothing outside reads the iterator value the fusion changes."""
        for loop in (first, second):
            if not loop.loop_variable:
                return False
            stride = loop_analysis.get_loop_stride(loop)
            if stride is None or symbolic.simplify(stride) != 1:
                return False
            start = loop_analysis.get_init_assignment(loop)
            end = loop_analysis.get_loop_end(loop)
            if start is None or end is None:
                return False
        # Adjacency: first's exclusive end (last value + 1) == second's start.
        first_end_excl = symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(first))) + 1)
        second_start = loop_analysis.get_init_assignment(second)
        if not _symbolically_equal(first_end_excl, second_start):
            return False
        if not self._bodies_match(first, second, scratch_index):
            return False
        # A shared name exits with the value it had after ``second``, which the fused loop keeps.
        names = (first, second) if first.loop_variable != second.loop_variable else ()
        return not any(loop_analysis.counter_used_outside_loop(l.loop_variable, l, l.sdfg) for l in names)

    def _single_body_state(self, loop: LoopRegion) -> SDFGState | None:
        """The loop's one non-empty body state, or ``None`` if the body is not a
        single compute state (empty connective states are tolerated, their edges
        must not decide or assign anything: those are not compared)."""
        blocks = list(loop.nodes())
        if not all(isinstance(b, SDFGState) for b in blocks):
            return None
        non_empty = [b for b in blocks if b.nodes()]
        if len(non_empty) != 1:
            return None
        if any(e.data.assignments or not e.data.is_unconditional() for e in loop.edges()):
            return None
        return non_empty[0]

    def _bodies_match(self, first: LoopRegion, second: LoopRegion, scratch_index: ScratchIndex) -> bool:
        """Whether the two loops' single body states compute the same values into the same places,
        up to their iteration variable and body-local scratch names."""
        s1 = self._single_body_state(first)
        s2 = self._single_body_state(second)
        if s1 is None or s2 is None:
            return False
        ids: ValueIds = {}
        sig1 = self._state_signature(s1, first.loop_variable, self._local_scratch(first, s1, scratch_index), ids)
        sig2 = self._state_signature(s2, second.loop_variable, self._local_scratch(second, s2, scratch_index), ids)
        return sig1 is not None and sig1 == sig2

    def _local_scratch(self, loop: LoopRegion, body_state: SDFGState, scratch_index: ScratchIndex) -> dict[str, None]:
        """Transient data names used ONLY inside ``body_state`` -- i.e. not
        referenced by any other block of the owning SDFG (not carried across
        iterations, not read/written outside the loop). These are frontend
        scratch intermediates whose names are irrelevant to what the body
        computes and so are normalised away when comparing two bodies."""
        root = loop
        while root.parent_graph is not None:
            root = root.parent_graph
        local: dict[str, None] = {}
        for n in body_state.nodes():
            if isinstance(n, nodes.AccessNode) and len(scratch_index.get(n.data, ())) == 1:
                desc = root.arrays.get(n.data)
                if desc is not None and desc.transient:
                    local[n.data] = None
        return local

    def _state_signature(
        self, state: SDFGState, loop_var: str, local_scratch: dict[str, None], ids: ValueIds
    ) -> list[int] | None:
        """The sorted value keys of ``state``'s sinks, ``None`` if a node's meaning is not in its key:
        scope, nested SDFG and library nodes carry it in properties and contents."""

        def node_key(node: nodes.Node) -> NodeKey | None:
            if not isinstance(node, (nodes.AccessNode, nodes.Tasklet)):
                return None
            if isinstance(node, nodes.AccessNode) and node.data in local_scratch:
                desc = state.sdfg.arrays[node.data]
                return ("scratch", f"{desc.dtype} {desc.shape}")
            return _node_key(node, loop_var, local_scratch)

        def edge_key(memlet: dace.Memlet) -> tuple:
            subsets = ("" if s is None else _normalize(str(s), loop_var) for s in (memlet.subset, memlet.other_subset))
            return (_canon_data(memlet.data, local_scratch) if memlet.data else "", *subsets, str(memlet.wcr))

        keys = value_keys(state, node_key, edge_key, ids, {})
        return None if keys is None else sorted(keys[n] for n in state.sink_nodes())

    def _merge(self, cfg: ControlFlowRegion, first: LoopRegion, second: LoopRegion, link: Edge[InterstateEdge]) -> None:
        """Extend ``first`` over the union range and splice ``second`` out.

        ``first`` keeps its body and iterator; only its exclusive upper bound is
        widened to ``second``'s. ``second``'s successor edges are re-homed onto
        ``first`` and ``second`` is removed.

        :param cfg: The region owning both loops.
        :param first: The surviving loop (its body is kept).
        :param second: The loop to remove (its body is a duplicate of ``first``).
        :param link: The ``first -> second`` sequencing edge (removed with ``second``).
        """
        var = first.loop_variable
        facts = SymbolResolver().facts_at(first.parent_graph)
        start = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(first))
        adjacent = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(second))
        new_end_excl = symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(second))) + 1)
        if not provably_ordered(start, adjacent, facts):
            first.init_statement = dace.properties.CodeBlock(f"{var} = {symbolic.symstr(sympy.Min(start, adjacent))}")
        if not provably_ordered(adjacent, new_end_excl, facts):
            new_end_excl = sympy.Max(adjacent, new_end_excl)
        first.loop_condition = dace.properties.CodeBlock(f"{var} < ({symbolic.symstr(new_end_excl)})")
        # Independent within each range says nothing across them (``A[i + K] = A[i]`` over ``[0, K)`` and ``[K, 2K)``).
        first.known_parallel = False

        out_edges = list(cfg.out_edges(second))
        cfg.remove_node(second)  # also drops the ``first -> second`` link edge
        for e in out_edges:
            cfg.add_edge(first, e.dst, e.data)


__all__ = ["FuseConsecutiveLoops", "GuardedFusionPlan", "plan_guarded_fusion", "commit_guarded_fusion"]

# #
# Guarded fusion: the same adjacent-range rewrite for loops whose bodies DIFFER.
# #
#
# ``FuseConsecutiveLoops`` above fuses a chain only when the bodies are identical, because then
# the merge is free -- widen one bound, drop the twin. When the bodies differ the index sequence
# argument is unchanged (concatenating ``[A, B)`` and ``[B, C)`` under the same stride visits the
# same indices in the same order), but the merged loop has to pick a body per iteration, so the
# bodies move into a ``ConditionalBlock`` keyed on the iterator.
#
# That rewrite is NOT unconditionally desirable, which is why it is a plan/commit pair rather
# than part of the pass above: if one sibling is a recurrence and the other is DOALL, fusing them
# makes the whole range a recurrence and LOSES the parallel half, and nothing downstream splits
# that back apart. Only a caller that has already proved it wins something -- ``WavefrontSkew``,
# which needs the merged 2-D space to find a diagonal -- may commit it.


class GuardedFusionPlan:
    """Sibling loops that fuse into one guarded loop over the union of their ranges.

    ``guards[k]`` are the constraints (each meant ``>= 0``) that hold inside sibling ``k`` --
    exactly the branch conditions :func:`commit_guarded_fusion` will emit, so an analysis run
    against the plan sees the same iteration space as the committed graph.

    ``var`` is the iterator the fused loop uses -- the first sibling's. Siblings normalized
    apart (``normalize_loops_and_maps`` gives every loop a PRIVATE ``_loop_it_N``) keep their own
    names in ``loop_vars`` and are renamed onto ``var`` at commit.
    """

    __slots__ = ("loops", "loop_vars", "var", "lo", "hi", "bounds", "guards")

    def __init__(
        self,
        loops: list[LoopRegion],
        loop_vars: list[str],
        var: str,
        lo: symbolic.SymbolicType,
        hi: symbolic.SymbolicType,
        bounds: list[tuple[symbolic.SymbolicType, symbolic.SymbolicType]],
        guards: list[list[symbolic.SymbolicType]],
    ) -> None:
        self.loops = loops
        self.loop_vars = loop_vars
        self.var = var
        self.lo = lo
        self.hi = hi
        self.bounds = bounds
        self.guards = guards


def plan_guarded_fusion(region: ControlFlowRegion) -> GuardedFusionPlan | None:
    """The maximal chain of adjacent-range sibling loops in ``region``, or ``None``.

    Refuses unless the chain is everything ``region`` holds (bar empty states): the callers
    reason about a two-level nest, and a stray computation beside the chain would be swept into
    a branch it does not belong to. Does not mutate.
    """
    loops = [b for b in region.nodes() if isinstance(b, LoopRegion)]
    if len(loops) < 2:
        return None
    for b in region.nodes():
        if b not in loops and isinstance(b, SDFGState) and len(list(b.nodes())) > 0:
            return None

    # Order the chain by its sequencing edges, starting from the loop nothing else points at.
    heads = [l for l in loops if not any(isinstance(e.src, LoopRegion) for e in region.in_edges(l))]
    if len(heads) != 1:
        return None
    chain: list[LoopRegion] = [heads[0]]
    while True:
        out = region.out_edges(chain[-1])
        if len(out) != 1:
            break
        link, nxt = out[0], out[0].dst
        if not isinstance(nxt, LoopRegion) or nxt in chain:
            break
        if len(region.in_edges(nxt)) != 1:
            break
        # Pure sequencing only: nothing may run, or be decided, between two fused siblings.
        if link.data.assignments:
            return None
        if link.data.condition is not None and link.data.condition.as_string not in ("1", "True"):
            return None
        chain.append(nxt)
    if len(chain) != len(loops):
        return None

    var = chain[0].loop_variable
    loop_vars = [l.loop_variable for l in chain]
    # Renaming a later sibling onto ``var`` must not capture: refuse if ``var`` already means
    # something inside that sibling.
    if any(var in l.free_symbols for l in chain[1:]):
        return None
    bounds: list[tuple[symbolic.SymbolicType, symbolic.SymbolicType]] = []
    for loop in chain:
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None or symbolic.simplify(stride) != 1:
            return None
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None:
            return None
        bounds.append((symbolic.pystr_to_symbolic(start), symbolic.pystr_to_symbolic(end)))
    for (_, prev_end), (nxt_start, _) in zip(bounds, bounds[1:]):
        if not _symbolically_equal(symbolic.simplify(prev_end + 1), nxt_start):
            return None
    # The guards partition the union only when no sibling's range is inverted.
    facts = SymbolResolver().facts_at(region)
    if not all(provably_ordered(lo, hi + 1, facts) for lo, hi in bounds):
        return None

    v = symbolic.pystr_to_symbolic(var)
    guards = [[symbolic.simplify(v - lo), symbolic.simplify(hi - v)] for (lo, hi) in bounds]
    return GuardedFusionPlan(chain, loop_vars, var, bounds[0][0], bounds[-1][1], bounds, guards)


def commit_guarded_fusion(plan: GuardedFusionPlan, region: ControlFlowRegion) -> LoopRegion:
    """Replace ``plan``'s chain in ``region`` with one loop over the union range whose body is a
    ``ConditionalBlock`` selecting the original body per iteration. Returns the new loop.

    The branch conditions are the chain's own sub-ranges, tested in order, with the last sibling
    as the ``else`` -- so the partition is total by construction and no iteration falls through.
    """
    var = plan.var
    sdfg = region.sdfg
    # Read before any mutation: if the chain led the region, the fused loop has to inherit that,
    # and ``add_node`` is the only sanctioned way to say so.
    was_start = region.start_block is plan.loops[0]
    merged = LoopRegion(
        f"{plan.loops[0].label}_fused",
        condition_expr=f"{var} < ({symbolic.symstr(plan.hi + 1)})",
        loop_var=var,
        initialize_expr=f"{var} = {symbolic.symstr(plan.lo)}",
        update_expr=f"{var} = {var} + 1",
        sdfg=sdfg,
    )
    selector = ConditionalBlock(f"{var}_range_select", sdfg=sdfg)
    for k, loop in enumerate(plan.loops):
        if plan.loop_vars[k] != var:
            loop.replace_dict({plan.loop_vars[k]: var})
        branch = ControlFlowRegion(f"{loop.label}_range", sdfg=sdfg)
        # Snapshot before moving: ``add_node`` re-homes each block's ``parent_graph``, so the
        # edge list has to be read off the loop while it still owns them.
        start, blocks, edges = loop.start_block, list(loop.nodes()), list(loop.edges())
        for blk in blocks:
            branch.add_node(blk, is_start_block=blk is start)
        for e in edges:
            branch.add_edge(e.src, e.dst, e.data)
        last = k == len(plan.loops) - 1
        selector.add_branch(None if last else f"{var} < ({symbolic.symstr(plan.bounds[k][1] + 1)})", branch)
    merged.add_node(selector, is_start_block=True)

    in_edges = list(region.in_edges(plan.loops[0]))
    out_edges = list(region.out_edges(plan.loops[-1]))
    region.add_node(merged, is_start_block=was_start, ensure_unique_name=True)
    for e in in_edges:
        region.add_edge(e.src, merged, e.data)
    for e in out_edges:
        region.add_edge(merged, e.dst, e.data)
    for loop in plan.loops:
        region.remove_node(loop)
    return merged
