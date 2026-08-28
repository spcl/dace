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

The rewrite is unconditionally value-preserving: concatenating two disjoint,
adjacent iteration ranges ``[A, B)`` and ``[B, C)`` under the *same* stride and
the *same* body executes the body for exactly the same index sequence in the
same order as a single ``[A, C)`` loop, so every loop-carried value is
identical. The match is deliberately conservative -- both loops must be
single-state, unit-stride, directly consecutive (one plain interstate edge
between them, nothing else), and structurally identical up to their iteration
variable -- so it fires only on the re-rolled tile/remainder shape and its kin,
never on unrelated adjacent loops.
"""
import re
from typing import List, Optional, Tuple

import sympy

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.analysis import loop_analysis

#: Placeholder the iteration variable is normalised to when comparing two loop
#: bodies, so ``a[_loop_it_0]`` and ``a[_loop_it_1]`` compare equal.
_ITER_PLACEHOLDER = '__lv__'

#: Placeholder every body-local scratch transient name is normalised to, so two
#: bodies differing only in a frontend-generated intermediate name (e.g.
#: ``s0_plus_a_slice`` vs ``s0_plus_a_slice_0``) compare equal. The carried
#: accumulator and the read/written arrays -- which are referenced OUTSIDE the
#: body and so are not body-local -- keep their real names and must match.
_SCRATCH_PLACEHOLDER = '__scratch__'


def _int_floor_to_sympy(expr):
    """Rewrite DaCe ``int_floor(a, b)`` sub-terms as SymPy ``floor(a / b)``.

    ``int_floor`` is opaque to :func:`sympy.simplify`, so a genuine identity
    such as ``int_floor(N - 11, 11) + 1 == int_floor(N, 11)`` is left unproven.
    SymPy's ``floor`` knows ``floor(x + k) == floor(x) + k`` for integer ``k``,
    which is exactly what an adjacency check between a tile bound and a
    remainder start needs.
    """
    return expr.replace(lambda x: x.func.__name__ == 'int_floor' and len(x.args) == 2,
                        lambda x: sympy.floor(x.args[0] / x.args[1]))


def _symbolically_equal(a, b) -> bool:
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
        diff = symbolic.simplify(pa - pb)
        if diff == 0:
            return True
        return symbolic.simplify(_int_floor_to_sympy(diff)) == 0
    except Exception:
        return False


def _normalize(text: str, loop_var: str) -> str:
    """Replace whole-word occurrences of ``loop_var`` in ``text`` with the
    canonical placeholder so two bodies differing only in iterator name match."""
    return re.sub(r'\b%s\b' % re.escape(loop_var), _ITER_PLACEHOLDER, text)


def _canon_data(name: str, local_scratch: dict) -> str:
    """Map a body-local scratch transient to the canonical placeholder; leave
    carried / external names (accumulator, arrays) untouched."""
    return _SCRATCH_PLACEHOLDER if name in local_scratch else name


def _node_key(node, loop_var: str, local_scratch: dict) -> Tuple:
    """A structural key for a body node, iterator- and scratch-name-independent."""
    if isinstance(node, nodes.AccessNode):
        return ('access', _canon_data(node.data, local_scratch))
    if isinstance(node, nodes.Tasklet):
        return ('tasklet', _normalize(node.code.as_string.strip(), loop_var))
    return ('other', type(node).__name__)


@explicit_cf_compatible
class FuseConsecutiveLoops(ppl.Pass):
    """Fuse two directly-consecutive, identical-bodied, unit-stride loops whose
    iteration ranges are adjacent (``[A, B)`` followed by ``[B, C)``) into a
    single loop over ``[A, C)``."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
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
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if self._fuse_one(cfg):
                        fused += 1
                        changed = True
                        break
        return fused or None

    def _fuse_one(self, cfg: ControlFlowRegion) -> bool:
        """Find and fuse one consecutive-loop pair inside ``cfg``.

        :param cfg: The control-flow region to search (one level; loops nested
                    deeper are reached via ``all_control_flow_regions``).
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
            if link.data.condition is not None and link.data.condition.as_string not in ('1', 'True'):
                continue
            if self._adjacent_identical(first, second):
                self._merge(cfg, first, second, link)
                return True
        return False

    def _adjacent_identical(self, first: LoopRegion, second: LoopRegion) -> bool:
        """Whether ``first`` then ``second`` are unit-stride, structurally
        identical, and cover adjacent index ranges ``[A, B)`` and ``[B, C)``."""
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
        first_end_excl = symbolic.simplify(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(first)) + 1)
        second_start = loop_analysis.get_init_assignment(second)
        if not _symbolically_equal(first_end_excl, second_start):
            return False
        return self._bodies_match(first, second)

    def _single_body_state(self, loop: LoopRegion) -> Optional[SDFGState]:
        """The loop's one non-empty body state, or ``None`` if the body is not a
        single compute state (empty connective states are tolerated)."""
        blocks = list(loop.nodes())
        if not all(isinstance(b, SDFGState) for b in blocks):
            return None
        non_empty = [b for b in blocks if b.nodes()]
        if len(non_empty) != 1:
            return None
        return non_empty[0]

    def _bodies_match(self, first: LoopRegion, second: LoopRegion) -> bool:
        """Whether the two loops' single body states are identical up to their
        iteration variable and body-local scratch names (same nodes, same edges,
        same memlets)."""
        s1 = self._single_body_state(first)
        s2 = self._single_body_state(second)
        if s1 is None or s2 is None:
            return False
        sig1 = self._state_signature(s1, first.loop_variable, self._local_scratch(first, s1))
        sig2 = self._state_signature(s2, second.loop_variable, self._local_scratch(second, s2))
        return sig1 == sig2

    def _local_scratch(self, loop: LoopRegion, body_state: SDFGState) -> dict:
        """Transient data names used ONLY inside ``body_state`` -- i.e. not
        referenced by any other block of the owning SDFG (not carried across
        iterations, not read/written outside the loop). These are frontend
        scratch intermediates whose names are irrelevant to what the body
        computes and so are normalised away when comparing two bodies."""
        root = loop
        while root.parent_graph is not None:
            root = root.parent_graph
        external: dict = {}
        for st in root.all_states():
            if st is body_state:
                continue
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode):
                    external[n.data] = None
        local: dict = {}
        for n in body_state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data not in external:
                desc = root.arrays.get(n.data)
                if desc is not None and desc.transient:
                    local[n.data] = None
        return local

    def _state_signature(self, state: SDFGState, loop_var: str, local_scratch: dict) -> Tuple:
        """An iterator- and scratch-name-independent structural signature of a
        body state: its sorted node keys and its sorted edge descriptors
        (endpoints, connectors, memlet data / both subsets / wcr)."""
        node_sig = sorted(_node_key(n, loop_var, local_scratch) for n in state.nodes())
        edge_sig = []
        for e in state.edges():
            subset = _normalize(str(e.data.subset), loop_var) if (e.data and e.data.subset is not None) else ''
            # A copy memlet indexes its DESTINATION in ``other_subset``; two bodies differing only
            # there are two different statements, and the loser's body is deleted by ``_merge``.
            other = _normalize(str(e.data.other_subset), loop_var) if (e.data
                                                                       and e.data.other_subset is not None) else ''
            data_name = _canon_data(e.data.data, local_scratch) if (e.data is not None and e.data.data) else ''
            wcr = str(e.data.wcr) if e.data is not None else ''
            # Connectors are the only raw fields here, and a memlet-path edge carries None while a
            # View's carries 'views'. Two edges whose endpoints canonicalize alike then reach a
            # None-vs-str comparison in the sort below, so spell an absent connector like the rest.
            src_key = _node_key(e.src, loop_var, local_scratch)
            dst_key = _node_key(e.dst, loop_var, local_scratch)
            edge_sig.append((src_key, e.src_conn or '', dst_key, e.dst_conn or '', data_name, subset, other, wcr))
        return (tuple(node_sig), tuple(sorted(edge_sig)))

    def _merge(self, cfg: ControlFlowRegion, first: LoopRegion, second: LoopRegion, link) -> None:
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
        new_end_excl = symbolic.simplify(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(second)) + 1)
        first.loop_condition = dace.properties.CodeBlock(f"{var} < ({symbolic.symstr(new_end_excl)})")

        out_edges = list(cfg.out_edges(second))
        cfg.remove_node(second)  # also drops the ``first -> second`` link edge
        for e in out_edges:
            cfg.add_edge(first, e.dst, e.data)


__all__ = ['FuseConsecutiveLoops', 'GuardedFusionPlan', 'plan_guarded_fusion', 'commit_guarded_fusion']

# --------------------------------------------------------------------------- #
# Guarded fusion: the same adjacent-range rewrite for loops whose bodies DIFFER.
# --------------------------------------------------------------------------- #
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

    __slots__ = ('loops', 'loop_vars', 'var', 'lo', 'hi', 'bounds', 'guards')

    def __init__(self, loops: List[LoopRegion], loop_vars: List[str], var: str, lo, hi, bounds: List[Tuple],
                 guards: List[List]):
        self.loops = loops
        self.loop_vars = loop_vars
        self.var = var
        self.lo = lo
        self.hi = hi
        self.bounds = bounds
        self.guards = guards


def plan_guarded_fusion(region: ControlFlowRegion) -> Optional[GuardedFusionPlan]:
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
    chain: List[LoopRegion] = [heads[0]]
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
        if link.data.condition is not None and link.data.condition.as_string not in ('1', 'True'):
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
    bounds: List[Tuple] = []
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
    merged = LoopRegion(f'{plan.loops[0].label}_fused',
                        condition_expr=f'{var} < ({symbolic.symstr(plan.hi + 1)})',
                        loop_var=var,
                        initialize_expr=f'{var} = {symbolic.symstr(plan.lo)}',
                        update_expr=f'{var} = {var} + 1',
                        sdfg=sdfg)
    selector = ConditionalBlock(f'{var}_range_select', sdfg=sdfg)
    for k, loop in enumerate(plan.loops):
        if plan.loop_vars[k] != var:
            loop.replace_dict({plan.loop_vars[k]: var})
        branch = ControlFlowRegion(f'{loop.label}_range', sdfg=sdfg)
        # Snapshot before moving: ``add_node`` re-homes each block's ``parent_graph``, so the
        # edge list has to be read off the loop while it still owns them.
        start, blocks, edges = loop.start_block, list(loop.nodes()), list(loop.edges())
        for blk in blocks:
            branch.add_node(blk, is_start_block=blk is start)
        for e in edges:
            branch.add_edge(e.src, e.dst, e.data)
        last = k == len(plan.loops) - 1
        selector.add_branch(None if last else f'{var} < ({symbolic.symstr(plan.bounds[k][1] + 1)})', branch)
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
    region.reset_cfg_list()
    return merged
