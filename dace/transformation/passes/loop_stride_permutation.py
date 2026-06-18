# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Permute perfectly-nested loops to maximize unit-stride array accesses.

The ``LoopRegion`` analogue of :class:`MinimizeStridePermutation`, slotted
*before* every ``LoopTo*`` lifting pass (LoopToScan / LoopToReduce / LoopToMap
/ ...) so it sees the raw loop nest -- once a recurrence has been lifted to a
``Scan`` libnode there are no loops left to interchange.

Motivating shape (TSVC ``s231`` and friends)::

    for i in 0:N:            # i indexes the UNIT-STRIDE column of aa[j, i]
        for j in 1:N:        # j carries the recurrence aa[j, i] = aa[j-1, i] + ...
            aa[j, i] = aa[j - 1, i] + bb[j, i]

Here ``i`` is data-parallel and unit-stride, ``j`` carries the dependence. With
``i`` outermost the parallel axis is the *strided* one: ``LoopToMap`` would map
the strided ``i`` (or the recurrence forces a ``Scan`` over a strided apply).
Interchanging to ``for j: for i:`` puts the unit-stride parallel ``i`` innermost
-- it becomes a contiguous vectorizable ``Map`` and ``j`` stays a plain
sequential loop. No ``Scan`` libnode is needed and the inner map's loads/stores
are unit-stride.

Soundness (no hand-rolled dependence analysis -- user direction 2026-06-18):
a fully data-parallel (DOALL) loop carries no loop-carried dependence and may be
moved to any position in a perfect nest. DOALL-ness is a property of the loop's
body, independent of its nest depth, so checking it *after* a speculative
interchange proves the interchange was legal. The check reuses the existing,
proven ``LoopToMap.can_be_applied_to`` oracle (it accepts exactly the DOALL,
single-state-body loops). The pass therefore only interchanges to move a
unit-stride loop inward when that loop is DOALL once innermost; otherwise it
reverts and leaves a TODO. This also implements the "do not reduce the number of
parallelizable maps" guard -- a reverted candidate keeps whatever mappable inner
loop existed before.

The interchange itself is a swap of the two regions' loop control metadata
(``loop_variable`` / ``init_statement`` / ``loop_condition`` /
``update_statement`` / ``inverted``): the body references the loop variables by
name, so swapping which region iterates which variable *is* the interchange,
with no node movement and no memlet rewrite.

Scope (v1): perfect two-level nests where the outer loop's body is exactly the
inner loop. Deeper nests and multi-statement bodies (which need
:class:`LoopFission` to run first) are left as a TODO.
"""
from typing import Dict, List, Optional, Set, Tuple

import sympy

import dace
from dace import SDFG
from dace import data as dt
from dace.sdfg.state import LoopRegion, SDFGState
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.loop_to_map import LoopToMap

#: Loop-control properties swapped to realize an interchange.
_LOOP_META_ATTRS = ('loop_variable', 'init_statement', 'loop_condition', 'update_statement', 'inverted')


@transformation.explicit_cf_compatible
class LoopStridePermutation(ppl.Pass):
    """Interchange perfect loop nests so a unit-stride DOALL loop is innermost.

    See the module docstring for the full design, the DOALL-based legality
    argument, and the v1 scope. Degrades gracefully: a nest it cannot prove
    safe to reorder is left untouched (logged as a TODO).
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, object]) -> Optional[int]:
        """Interchange every eligible perfect two-level loop nest in ``sdfg``.

        :param sdfg: The SDFG to canonicalize.
        :returns: The number of interchanges applied, or ``None`` if none.
        """
        applied = 0
        for outer, inner in self._perfect_two_nests(sdfg):
            applied += self._maybe_interchange(sdfg, outer, inner)
        return applied or None

    def _perfect_two_nests(self, sdfg: SDFG) -> List[Tuple[LoopRegion, LoopRegion]]:
        """Collect ``(outer, inner)`` LoopRegion pairs forming a perfect nest.

        A pair qualifies when ``outer``'s body is exactly one block, that block
        is the ``inner`` LoopRegion, and ``inner`` is itself innermost (its body
        contains no further LoopRegion). Both loops must be plain forward
        for-loops (a resolvable loop variable, not inverted).
        """
        pairs: List[Tuple[LoopRegion, LoopRegion]] = []
        for region in sdfg.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            blocks = list(region.nodes())
            if len(blocks) != 1 or not isinstance(blocks[0], LoopRegion):
                continue
            inner = blocks[0]
            # inner must be innermost: no nested LoopRegion in its body.
            if any(isinstance(b, LoopRegion) for b in inner.nodes()):
                continue
            if not region.loop_variable or not inner.loop_variable or region.inverted or inner.inverted:
                continue
            pairs.append((region, inner))
        return pairs

    def _maybe_interchange(self, sdfg: SDFG, outer: LoopRegion, inner: LoopRegion) -> int:
        """Interchange ``outer``/``inner`` if it moves a unit-stride loop inward
        and the moved loop is DOALL once innermost.

        :returns: 1 if an interchange was committed, else 0.
        """
        outer_var, inner_var = outer.loop_variable, inner.loop_variable
        # Rectangular iteration space: a metadata swap only realizes the
        # interchange when neither loop's bounds reference the other's loop
        # variable. A triangular nest (``for j: for i in 1:j+1``) changes its
        # iteration SET under interchange and needs a bound transformation, not
        # a swap -- reject it here.
        if not self._bounds_independent(outer, inner):
            return 0
        unit_vars = self._unit_stride_loop_vars(sdfg, inner, {outer_var, inner_var})
        # Only act when the unit-stride loop is the OUTER one and the inner is
        # not unit-stride -- that is the case interchange improves. (When both
        # or neither are unit-stride there is nothing unambiguous to gain.)
        if not (outer_var in unit_vars and inner_var not in unit_vars):
            return 0

        # Speculatively interchange (swap loop-control metadata) so the
        # unit-stride loop becomes the inner region, then verify with the
        # LoopToMap DOALL oracle that the now-inner loop is data-parallel. A
        # DOALL loop is freely interchangeable, so DOALL-after proves the swap
        # was legal; it also guarantees the moved loop becomes a map (the
        # parallelizable-map count does not drop).
        self._swap_loop_metadata(outer, inner)
        if self._is_doall(sdfg, inner):
            return 1
        # Not provably safe/beneficial -- revert and leave a TODO.
        self._swap_loop_metadata(outer, inner)
        # TODO(loop-stride-permutation): unit-stride loop ``outer_var`` is not
        # DOALL once innermost (carries a dependence, or its body is not a
        # single state pre-fission). Handle via dependence analysis / running
        # after LoopFission for multi-statement nests.
        return 0

    @staticmethod
    def _bounds_independent(outer: LoopRegion, inner: LoopRegion) -> bool:
        """``True`` iff neither loop's bounds reference the other's loop variable
        (a rectangular iteration space). A triangular nest fails this test."""
        import re

        def meta_tokens(lr: LoopRegion) -> Set[str]:
            toks: Set[str] = set()
            for attr in ('init_statement', 'loop_condition', 'update_statement'):
                cb = getattr(lr, attr)
                if cb is None:
                    continue
                toks |= set(re.findall(r"\b[A-Za-z_]\w*\b", cb.as_string))
            return toks

        return (inner.loop_variable not in meta_tokens(outer)) and (outer.loop_variable not in meta_tokens(inner))

    @staticmethod
    def _swap_loop_metadata(a: LoopRegion, b: LoopRegion) -> None:
        """Swap the loop-control metadata of two regions -- this realizes the
        interchange because the body references the loop variables by name."""
        for attr in _LOOP_META_ATTRS:
            av, bv = getattr(a, attr), getattr(b, attr)
            setattr(a, attr, bv)
            setattr(b, attr, av)

    def _is_doall(self, sdfg: SDFG, loop: LoopRegion) -> bool:
        """``True`` iff ``loop`` is a DOALL loop ``LoopToMap`` would parallelize.

        Uses the existing transformation oracle so the parallelism judgement
        matches exactly what the downstream ``parallelize`` stage accepts.
        """
        try:
            return LoopToMap.can_be_applied_to(sdfg, loop=loop)
        except Exception:  # noqa: BLE001 -- oracle refuses exotic shapes -> not provably DOALL
            return False

    def _unit_stride_loop_vars(self, sdfg: SDFG, inner: LoopRegion, loop_vars: Set[str]) -> Set[str]:
        """Loop variables that index a stride-1 array axis with a unit coefficient.

        Walks every memlet inside the (innermost) loop body and, per the user's
        method, collects the parameters appearing in the *unit-stride dimension*
        of each access. Only literal stride-1 axes count, so the test is
        concrete (no symbolic stride comparison needed).
        """
        result: Set[str] = set()
        for state in inner.all_states():
            if not isinstance(state, SDFGState):
                continue
            for edge in state.edges():
                memlet = edge.data
                if memlet is None or memlet.data is None:
                    continue
                desc = sdfg.arrays.get(memlet.data)
                if not isinstance(desc, dt.Array):
                    continue
                subset = memlet.subset
                if subset is None or len(subset) != len(desc.strides):
                    continue
                for rng, stride in zip(subset.ndrange(), desc.strides):
                    try:
                        is_unit = bool(dace.symbolic.simplify(sympy.sympify(stride) - 1) == 0)
                    except Exception:  # noqa: BLE001
                        is_unit = False
                    if not is_unit:
                        continue
                    index_expr = sympy.sympify(rng[0])
                    free = {str(s) for s in index_expr.free_symbols}
                    for v in loop_vars:
                        if v not in free:
                            continue
                        coeff = index_expr.coeff(pystr_to_symbolic(v), 1)
                        if sympy.Abs(coeff) == 1:
                            result.add(v)
        return result
