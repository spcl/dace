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
a loop that carries no dependence *at its own level* may be swapped with the loop
directly inside it. Every dependence is then either carried outside the pair (its
leading ``<`` is untouched by the swap) or ``=`` at that level (so the swap moves
a ``=`` inward and the leading ``<`` outward), and neither case can turn a
lexicographically positive direction vector negative. The moved loop still
carries nothing at its new level, so the argument re-applies and the whole bubble
to innermost is legal by induction.

The DOALL judgement must therefore be taken **before** the bubble, on the loop at
its ORIGINAL position: "DOALL once innermost" is a strictly weaker property and
does NOT imply the interchange was legal. The classic ``(<, >)`` counter-example
is ``a[j, i] = a[j + 1, i - 1] + b[j, i]`` with ``i`` outermost -- moving ``i``
innermost makes it parallel *there* while reversing the dependence.

The check reuses the existing, proven ``LoopToMap.can_be_applied_to`` oracle (it
accepts exactly the DOALL loops). The pass only interchanges to move a unit-stride
loop inward when that loop is DOALL where it stands *and* still DOALL once
innermost; otherwise it reverts and leaves a TODO. The second half also
implements the "do not reduce the number of parallelizable maps" guard -- a
reverted candidate keeps whatever mappable inner loop existed before.

The interchange itself is a swap of the two regions' loop control metadata
(``loop_variable`` / ``init_statement`` / ``loop_condition`` /
``update_statement`` / ``inverted``): the body references the loop variables by
name, so swapping which region iterates which variable *is* the interchange,
with no node movement and no memlet rewrite.

Scope: perfect N-level nests (each level's body is exactly the next loop). The
unit-stride DOALL axis is bubbled inward one adjacent swap at a time; every swap
must keep a rectangular iteration space and the final innermost loop must be
DOALL (verified via the ``LoopToMap`` oracle), otherwise the whole bubble
reverts. A DOALL loop is freely interchangeable, so proving the moved loop is
DOALL once innermost certifies every adjacent swap along the way. Multi-statement
bodies (which need :class:`LoopFission` to run first) are still out of scope --
the innermost body must be a single statement for the oracle.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

import sympy

import dace
from dace import SDFG
from dace import data as dt
from dace.sdfg.state import LoopRegion, SDFGState
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl, transformation
from dace.properties import CodeBlock
from dace.symbolic import symstr
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.analysis import loop_analysis

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
        """Interchange every eligible perfect loop nest in ``sdfg`` so a
        unit-stride DOALL axis becomes innermost.

        :param sdfg: The SDFG to canonicalize.
        :returns: The number of nests interchanged, or ``None`` if none.
        """
        applied = 0
        for chain in self._perfect_nests(sdfg):
            applied += self._maybe_interchange_nest(sdfg, chain)
        return applied or None

    def _perfect_nests(self, sdfg: SDFG) -> List[List[LoopRegion]]:
        """Collect maximal perfect loop-nest chains ``[L0, .., L_{n-1}]`` (outer
        to inner), ``n >= 2``.

        A chain descends while each level's body is exactly one block that is the
        next ``LoopRegion``, ending at the innermost loop (whose body is not a
        single LoopRegion). Only chain TOPS are seeded -- a loop that is itself
        the sole body of a parent loop belongs to that parent's chain, not its
        own -- so each maximal nest is returned exactly once. Every loop must be
        a plain forward for-loop (resolvable loop variable, not inverted).
        """
        # A region is the "inner" of a perfect pair iff it is the single block of
        # a parent LoopRegion; chain tops are the loops that are NOT such inners.
        inners: Set[int] = set()
        for region in sdfg.all_control_flow_regions():
            if isinstance(region, LoopRegion):
                blocks = list(region.nodes())
                if len(blocks) == 1 and isinstance(blocks[0], LoopRegion):
                    inners.add(id(blocks[0]))

        chains: List[List[LoopRegion]] = []
        for region in sdfg.all_control_flow_regions():
            if not isinstance(region, LoopRegion) or id(region) in inners:
                continue
            chain = [region]
            cur = region
            while True:
                blocks = list(cur.nodes())
                if len(blocks) == 1 and isinstance(blocks[0], LoopRegion):
                    cur = blocks[0]
                    chain.append(cur)
                else:
                    break
            if len(chain) < 2:
                continue
            if any((not lr.loop_variable) or lr.inverted for lr in chain):
                continue
            chains.append(chain)
        return chains

    def _maybe_interchange_nest(self, sdfg: SDFG, chain: List[LoopRegion]) -> int:
        """Bubble a unit-stride DOALL loop in ``chain`` to the innermost slot.

        :param chain: A perfect nest ``[L0, .., L_{n-1}]`` (outer to inner).
        :returns: 1 if an interchange was committed, else 0.
        """
        loop_vars = [lr.loop_variable for lr in chain]
        innermost = chain[-1]
        unit_vars = self._unit_stride_loop_vars(sdfg, innermost, set(loop_vars))
        # Nothing unambiguous to gain when the innermost axis is already
        # unit-stride (or no axis is): interchange only helps by moving an OUTER
        # unit-stride axis inward.
        if not unit_vars or loop_vars[-1] in unit_vars:
            return 0
        # Bubble a non-innermost unit-stride axis to innermost. Prefer the axis
        # closest to the inner slot (fewest swaps); fall back to outer ones.
        for pos in range(len(chain) - 2, -1, -1):
            if loop_vars[pos] in unit_vars and self._try_bubble_to_inner(sdfg, chain, pos):
                return 1
        return 0

    def _try_bubble_to_inner(self, sdfg: SDFG, chain: List[LoopRegion], pos: int) -> bool:
        """Bubble ``chain[pos]``'s loop variable to the innermost slot via adjacent
        metadata swaps, verify, and revert the whole bubble on failure.

        Each adjacent interchange must keep a rectangular iteration space
        (``_bounds_independent``). Legality comes from the moved loop being DOALL
        at its ORIGINAL level (via the ``LoopToMap`` oracle): a loop carrying
        nothing at its own level may be swapped inward, and stays that way, so the
        whole bubble is certified. The post-bubble DOALL re-check is the benefit
        guard, not the legality one.
        """
        # "DOALL once innermost" does not imply the swap was legal -- a (<, >)
        # dependence is parallel innermost yet its interchange reverses it.
        if not self._is_doall(sdfg, chain[pos]):
            return False

        # How many levels of the nest are parallel before the move, to compare against after.
        doall_before = sum(1 for lr in chain if self._is_doall(sdfg, lr))

        done: List[Tuple[LoopRegion, LoopRegion]] = []
        snapshot = [{attr: getattr(lr, attr) for attr in _LOOP_META_ATTRS} for lr in chain]

        def revert() -> None:
            # Restore rather than re-swap: a trapezoid rewrite is not its own inverse.
            for lr, meta in zip(chain, snapshot):
                for attr, value in meta.items():
                    setattr(lr, attr, value)

        for k in range(pos, len(chain) - 1):
            # After prior swaps the target var sits at chain[k]; the pair being
            # interchanged is (target=chain[k], chain[k + 1]). A metadata swap only
            # realizes the interchange when the iteration space stays rectangular;
            # a triangular pair changes the iteration SET and is rejected here.
            if self._bounds_independent(chain[k], chain[k + 1]):
                self._swap_loop_metadata(chain[k], chain[k + 1])
            elif not self._swap_trapezoid(chain[k], chain[k + 1]):
                revert()
                return False
            done.append((chain[k], chain[k + 1]))
        if not self._is_doall(sdfg, chain[-1]):
            # Not provably safe/beneficial -- revert and leave a TODO.
            revert()
            # TODO(loop-stride-permutation): the unit-stride loop is not DOALL once
            # innermost (carries a dependence, or a multi-statement body pre-fission).
            return False
        # A LEGAL interchange can still cost a parallel level: `_swap_trapezoid` leaves the new
        # outer loop a `min`/`int_floor` bound that `LoopToMap` will not take, trading two Maps
        # for one Map plus a sequential loop. Unit stride is not worth that.
        if sum(1 for lr in chain if self._is_doall(sdfg, lr)) < doall_before:
            revert()
            return False
        return True

    @staticmethod
    def _bounds_independent(outer: LoopRegion, inner: LoopRegion) -> bool:
        """``True`` iff neither loop's bounds reference the other's loop variable
        (a rectangular iteration space). A triangular nest fails this test."""
        return (inner.loop_variable not in LoopStridePermutation._meta_tokens(outer)
                and outer.loop_variable not in LoopStridePermutation._meta_tokens(inner))

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        """Identifiers appearing in ``text``."""
        return set(re.findall(r"\b[A-Za-z_]\w*\b", text))

    @staticmethod
    def _meta_tokens(lr: LoopRegion) -> Set[str]:
        """Identifiers appearing anywhere in ``lr``'s loop control."""
        toks: Set[str] = set()
        for attr in ('init_statement', 'loop_condition', 'update_statement'):
            block = getattr(lr, attr)
            if block is not None:
                toks |= LoopStridePermutation._tokens(block.as_string)
        return toks

    @staticmethod
    def _swap_trapezoid(outer: LoopRegion, inner: LoopRegion) -> bool:
        """Interchange a trapezoidal pair by rebuilding both loops' bounds.

        For an inner loop STARTING at an affine function of the outer variable (TSVC ``s1232``),
        ``{j0 <= j <= j1, c*j + d <= i <= i1}`` with ``c > 0`` is exactly
        ``{c*j0 + d <= i <= i1, j0 <= j <= min(j1, (i - d) // c)}``. Anything else is refused.

        :param outer: The outer loop of the adjacent pair.
        :param inner: The inner loop.
        :returns: ``True`` if the pair was interchanged.
        """
        ovar, ivar = outer.loop_variable, inner.loop_variable
        if ivar in LoopStridePermutation._meta_tokens(outer):
            return False
        for attr in ('loop_condition', 'update_statement'):
            block = getattr(inner, attr)
            if block is not None and ovar in LoopStridePermutation._tokens(block.as_string):
                return False
        if loop_analysis.get_loop_stride(outer) != 1 or loop_analysis.get_loop_stride(inner) != 1:
            return False

        j0 = loop_analysis.get_init_assignment(outer)
        j1 = loop_analysis.get_loop_end(outer)
        i0 = loop_analysis.get_init_assignment(inner)
        i1 = loop_analysis.get_loop_end(inner)
        if any(b is None for b in (j0, j1, i0, i1)):
            return False

        osym = pystr_to_symbolic(ovar)
        c = i0.coeff(osym, 1)
        d = dace.symbolic.simplify(i0 - c * osym)
        if osym in d.free_symbols or not c.is_integer or not c.is_positive:
            return False
        # ``int_floor`` is a floor division: a negative numerator would round the wrong way, so the
        # rewrite is only taken where the new outer loop starts at or above ``d``.
        if (c * j0).is_nonnegative is not True:
            return False

        floor_expr = symstr(dace.symbolic.int_floor(pystr_to_symbolic(ivar) - d, c))
        outer.loop_variable, inner.loop_variable = ivar, ovar
        outer.init_statement = CodeBlock(ivar + ' = ' + symstr(c * j0 + d))
        outer.loop_condition = CodeBlock(ivar + ' <= ' + symstr(i1))
        outer.update_statement = CodeBlock(ivar + ' = ' + ivar + ' + 1')
        inner.init_statement = CodeBlock(ovar + ' = ' + symstr(j0))
        inner.loop_condition = CodeBlock(ovar + ' <= min(' + symstr(j1) + ', ' + floor_expr + ')')
        inner.update_statement = CodeBlock(ovar + ' = ' + ovar + ' + 1')
        return True

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
                        is_unit = bool(dace.symbolic.simplify(pystr_to_symbolic(stride) - 1) == 0)
                    except Exception:  # noqa: BLE001
                        is_unit = False
                    if not is_unit:
                        continue
                    index_expr = pystr_to_symbolic(rng[0])
                    free = {str(s) for s in index_expr.free_symbols}
                    for v in loop_vars:
                        if v not in free:
                            continue
                        coeff = index_expr.coeff(pystr_to_symbolic(v), 1)
                        if sympy.Abs(coeff) == 1:
                            result.add(v)
        return result
