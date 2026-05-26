# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone parallelization-preparation passes.

These rewrite loops so that ``LoopToMap`` can parallelize more of them. They are
plain :class:`~dace.transformation.pass_pipeline.Pass` objects so the
``parallelize`` pipeline (and anyone else) can just compose them:

- :class:`ShortLoopUnroll` -- fully unroll constant-trip loops with at most
  ``unroll_limit`` iterations, turning small recurrence / reduction loops into
  inline straight-line code instead of atomically-parallelized maps.
- :class:`BestEffortLoopPeeling` -- search front/back/both peels of 1..``peel_limit``
  boundary iterations, keep the one that unblocks the most maps, revert if none
  helps, and prune the now-dead boundary guard from the remainder.

Transformation classes are imported lazily inside the methods: importing them at
module load would cycle (this package is imported by the transformations those
imports pull in).
"""
import copy
from typing import Any, Dict, Optional

from dace import properties, symbolic
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl

#: Default trip-count threshold below which a constant-trip loop is unrolled.
DEFAULT_UNROLL_LIMIT = 8
#: Default maximum number of iterations peeled (per side) when searching for a
#: peel that unblocks parallelization.
DEFAULT_PEEL_LIMIT = 8


def _loops(sdfg: SDFG):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _num_maps(sdfg: SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _unique_block_label(sdfg: SDFG, base: str) -> str:
    """A control-flow-block label not currently used anywhere in ``sdfg``."""
    import itertools
    existing = {b.label for b in sdfg.all_control_flow_blocks()}
    for n in itertools.count():
        cand = f'{base}_p{n}'
        if cand not in existing:
            return cand


def _constant_trip_count(loop: LoopRegion, sdfg: SDFG) -> Optional[int]:
    """The exact iteration count of ``loop`` if it is constant, else ``None``
    (matches ``len(range(0, end - start + 1, stride))``, i.e. LoopUnroll's count)."""
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or not loop.loop_variable:
        return None
    if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
        return None
    try:
        stride_val = int(symbolic.evaluate(stride, sdfg.constants))
        diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
    except (TypeError, ValueError):
        return None
    if stride_val <= 0 or diff <= 0:
        return None
    return len(range(0, diff, stride_val))


@properties.make_properties
class ShortLoopUnroll(ppl.Pass):
    """Fully unroll every constant-trip loop with at most ``unroll_limit`` iterations."""

    CATEGORY: str = 'Optimization Preparation'

    unroll_limit = properties.Property(
        dtype=int,
        default=DEFAULT_UNROLL_LIMIT,
        desc='Fully unroll constant-trip loops with at most this many iterations (0 disables).')

    def __init__(self, unroll_limit: int = DEFAULT_UNROLL_LIMIT):
        self.unroll_limit = unroll_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Unroll short constant-trip loops; returns the number unrolled or None.

        Re-collects after each unroll since unrolling rewrites the control-flow
        structure (and may expose newly-constant inner loops).
        """
        if self.unroll_limit <= 0:
            return None
        from dace.transformation.interstate.loop_unroll import LoopUnroll
        unrolled = 0
        changed = True
        while changed:
            changed = False
            for loop in _loops(sdfg):
                trip = _constant_trip_count(loop, sdfg)
                if trip is None or trip > self.unroll_limit:
                    continue
                try:
                    LoopUnroll().apply_to(sdfg=loop.sdfg, loop=loop)
                except Exception:
                    continue  # not unrollable in this context; leave it for LoopToMap
                unrolled += 1
                changed = True
                break
        return unrolled or None


@properties.make_properties
class BestEffortLoopPeeling(ppl.Pass):
    """Best-effort loop peeling that unblocks parallelization.

    For each of front / back / both and each peel count ``k`` in
    ``1..peel_limit``, peel ``k`` boundary iterations off the loops and run a
    cheap candidate check (scalar fission -> symbol propagation -> constant
    propagation -> LoopToMap) counting the resulting maps. Keep the single peel
    that yields the most maps; if none beats the no-peel baseline, leave the SDFG
    unpeeled. The search runs on ``copy.deepcopy`` copies (revertible by
    construction); only the winning peel is applied to the real SDFG.
    """

    CATEGORY: str = 'Optimization Preparation'

    peel_limit = properties.Property(
        dtype=int,
        default=DEFAULT_PEEL_LIMIT,
        desc='Try peeling 1..peel_limit iterations from the front, the back, and both; keep the '
        'peel that produces the most maps and revert if none beats the no-peel baseline (0 disables).')

    def __init__(self, peel_limit: int = DEFAULT_PEEL_LIMIT):
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _peel_loops(self, sdfg: SDFG, count: int, direction: str) -> int:
        """Peel ``count`` iterations off every peelable loop and return how many
        loops were peeled. ``direction`` is ``'front'``, ``'back'`` or ``'both'``.

        Uses ``verify=False`` so loops with symbolic bounds (where peeling a
        boundary is most useful) are not rejected by ``LoopUnroll``'s
        constant-size gate; infeasible loops simply raise and are skipped.
        """
        from dace.transformation.interstate.loop_peeling import LoopPeeling
        sides = {'front': [True], 'back': [False], 'both': [True, False]}[direction]
        peeled = 0
        for loop in _loops(sdfg):
            # A loop short enough to be fully consumed by the peel is the unroll
            # pass's job, not peeling's.
            trip = _constant_trip_count(loop, sdfg)
            if trip is not None and trip <= count * len(sides):
                continue
            did = False
            for idx, begin in enumerate(sides):
                try:
                    if idx > 0:
                        # LoopPeeling names the peeled iteration regions after the
                        # loop label; a second peel on the same loop would reuse
                        # those names. Relabel the remainder loop first so the
                        # front and back peels produce distinct region names.
                        loop.label = _unique_block_label(loop.sdfg, loop.label)
                    # Properties must go through ``options=`` -- bare kwargs are not
                    # applied, leaving ``count`` at LoopUnroll's default 0 (a no-op).
                    LoopPeeling().apply_to(sdfg=loop.sdfg,
                                           loop=loop,
                                           verify=False,
                                           options={'count': count, 'begin': begin})
                    did = True
                except Exception:
                    continue
            peeled += int(did)
        if peeled:
            # Peeling moves the boundary iteration out but leaves the remainder
            # loop body unchanged; a guard like ``if i == N-1`` is now dead over
            # the remainder's range. Pruning it is what actually frees LoopToMap.
            self._prune_dead_loop_branches(sdfg)
        return peeled

    def _cond_dead_over_range(self, cond, ivar: str, start, end, sdfg: SDFG) -> bool:
        """Whether the guard ``cond`` is provably false for every ``ivar`` in
        ``[start, end]``. Handles ``ivar <cmp> C`` / ``C <cmp> ivar`` with a
        loop-invariant ``C`` and ``cmp`` in ``==, <, <=, >, >=``."""
        import ast
        try:
            node = cond.code[0]
        except (AttributeError, IndexError, TypeError):
            return False
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return False
        left, op, right = node.left, node.ops[0], node.comparators[0]

        def is_ivar(n):
            return isinstance(n, ast.Name) and n.id == ivar

        if is_ivar(left) and not is_ivar(right):
            other, flip = right, False
        elif is_ivar(right) and not is_ivar(left):
            other, flip = left, True
        else:
            return False
        try:
            c = symbolic.pystr_to_symbolic(ast.unparse(other))
        except Exception:
            return False
        if symbolic.pystr_to_symbolic(ivar) in c.free_symbols:
            return False  # C must be loop-invariant

        opname = type(op).__name__
        if flip:
            opname = {'Lt': 'Gt', 'LtE': 'GtE', 'Gt': 'Lt', 'GtE': 'LtE'}.get(opname, opname)

        def is_pos(x):
            s = symbolic.simplify(x)
            return s.is_number and s > 0

        def is_nonneg(x):
            s = symbolic.simplify(x)
            return s.is_number and s >= 0

        # ``ivar`` ranges over [start, end].
        if opname == 'Eq':  # i == C : false for all iff C < start or C > end
            return is_pos(start - c) or is_pos(c - end)
        if opname == 'Lt':  # i < C  : false for all iff start >= C
            return is_nonneg(start - c)
        if opname == 'LtE':  # i <= C : false for all iff start > C
            return is_pos(start - c)
        if opname == 'Gt':  # i > C  : false for all iff end <= C
            return is_nonneg(c - end)
        if opname == 'GtE':  # i >= C : false for all iff end < C
            return is_pos(c - end)
        return False

    def _prune_dead_loop_branches(self, sdfg: SDFG) -> bool:
        """Remove single-branch (no-else) conditionals in a loop body whose guard
        is provably false over the loop's iteration range -- the boundary guard a
        peel leaves behind (e.g. ``if i == N-1`` once the remainder is ``[0,N-2]``).
        ``PruneEmptyConditionalBranches`` then drops the now-empty conditional,
        leaving a clean affine body for LoopToMap."""
        from dace.sdfg.state import ConditionalBlock
        from dace.transformation.passes.analysis import loop_analysis
        from dace.transformation.passes.simplification.prune_empty_conditional_branches import \
            PruneEmptyConditionalBranches
        changed = False
        for loop in _loops(sdfg):
            start = loop_analysis.get_init_assignment(loop)
            end = loop_analysis.get_loop_end(loop)
            if start is None or end is None:
                continue
            ivar = loop.loop_variable
            for cb in [b for b in loop.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]:
                if len(cb.branches) != 1:
                    continue  # only the simple boundary-guard shape
                cond, region = cb.branches[0]
                if cond is None:
                    continue
                if self._cond_dead_over_range(cond, ivar, start, end, sdfg):
                    # Empty the dead branch's body (it is never taken in the
                    # remainder); PruneEmptyConditionalBranches then drops the
                    # now-empty branch and the conditional itself.
                    region.remove_nodes_from(list(region.nodes()))
                    changed = True
        if changed:
            PruneEmptyConditionalBranches().apply_pass(sdfg, {})
        return changed

    def _l2m_candidate_maps(self, candidate: SDFG) -> int:
        """Cheap proxy: does the peel unblock more ``LoopToMap``? Runs scalar
        fission -> symbol propagation -> constant propagation -> LoopToMap (no
        reduction passes) and returns the map count. Mutates ``candidate``."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.constant_propagation import ConstantPropagation
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        PrivatizeScalars().apply_pass(candidate, {})
        SymbolPropagation().apply_pass(candidate, {})
        ConstantPropagation().apply_pass(candidate, {})
        PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(candidate, {})
        return _num_maps(candidate)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Apply the best peel found (or none); returns the peel count applied or None."""
        if self.peel_limit <= 0 or not _loops(sdfg):
            return None

        baseline = self._l2m_candidate_maps(copy.deepcopy(sdfg))
        best_count, best = baseline, None
        for direction in ('front', 'back', 'both'):
            for count in range(1, self.peel_limit + 1):
                candidate = copy.deepcopy(sdfg)
                if self._peel_loops(candidate, count, direction) == 0:
                    continue
                try:
                    candidate.validate()  # only a peel that stays valid is a working parameter
                    n_maps = self._l2m_candidate_maps(candidate)
                except Exception:
                    continue
                if n_maps > best_count:
                    best_count, best = n_maps, (count, direction)

        if best is None:
            return None
        self._peel_loops(sdfg, *best)
        return best[0]
