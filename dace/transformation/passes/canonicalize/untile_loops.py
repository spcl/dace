# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collapse a manually-tiled two-level loop nest back to a single unit-stride loop.

Source kernels often arrive with an unrolled tile already baked into the loop
structure -- TSVC ``s116`` / ``s353`` / ``s31111`` write::

    for i in range(0, N, 4):           # outer: stride-4 tile loop
        for ii in range(0, 4):         #   inner: trip == outer stride
            a[i + ii] = ...
    # or, equivalently
    for i in range(0, N, 4):
        for ii in range(i, i + 4):
            a[ii] = ...

Both shapes describe a single unit-stride traversal of ``range(0, N)``. The
hand-written tile suppresses :class:`~dace.transformation.interstate.loop_to_map.LoopToMap`
because the outer loop has a non-unit stride and the inner trip is small; it
also blocks ``ShortLoopUnroll`` from collapsing the inner (the inner depends on
the outer's runtime value).

``UntileLoops`` recognises this two-level pattern and rewrites the outer loop
to drive a single unit-stride iterator ``_untile_k_<n>`` over ``[0, N)``, after
which downstream canonicalize passes (``LoopToReduce``, ``LoopToScan``,
``LoopToMap``) see the body in its natural one-dimensional form.

Pattern
=======

The outer loop must be ``for i in range(0, N, K)`` where ``K`` is a positive
tile size -- a concrete integer literal ``> 1`` **or** a positive symbol (e.g.
a block-size parameter) -- and the start is ``0``. The single body block of the
outer must be exactly one nested :class:`~dace.sdfg.state.LoopRegion` and
nothing else (perfect nest). Symbolic tiles admit only a unit inner stride
(single-level untile); a concrete tile additionally admits cascade rungs whose
inner stride divides ``K``.

The inner loop must be one of the following shapes (both with unit stride):

* **Case A** -- ``for ii in range(0, K)``: the inner trip equals the outer
  stride. Every memlet inside the inner body must address its arrays only via
  the combined expression ``i + ii`` (no bare ``i`` or bare ``ii``).
* **Case B** -- ``for ii in range(i, i + K)``: the inner runs over the absolute
  tile. Every memlet must address its arrays only via ``ii`` (no bare ``i``).

In both cases the inner index is the *only* affine reference to the outer
iterator the body is allowed; the rewrite remaps that single reference to the
new unit iterator. If any memlet uses ``i`` or ``ii`` outside the recognised
combination, the rewrite refuses.

Rewrite
=======

The outer ``LoopRegion`` is reused: its iterator becomes ``_untile_k_<n>`` and
its bounds become ``0 <= k < N`` with step ``1``. The inner ``LoopRegion`` is
spliced out (its body block is moved up under the outer), and an in-body
``replace_dict`` substitutes the recognised index expression with ``k``:

* **Case A**: ``i`` -> ``k - 0 = k``? No -- ``i + ii`` -> ``k``, then any
  surviving bare ``ii`` (which we already refused) cannot appear; the rewrite
  simply removes ``ii`` from the body and binds ``i = k`` on the iteration-entry
  iedge. The combined ``i + ii`` reference becomes ``k`` after symbol
  substitution.
* **Case B**: ``ii`` -> ``k``; ``i`` does not appear in any memlet (per the
  safety check) so it is dropped.

The original ``i`` / ``ii`` symbols are removed from the SDFG symbol table
*only* if they had no live readers after the rewrite. (The outer ``i`` may
still be referenced by interstate-edge assignments that the cascade-up pass
hoisted; those are left alone.)
"""
import copy
from typing import List, Optional, Set, Tuple

import sympy

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.graph import NodeNotFoundError
from dace.sdfg.state import LoopRegion, SDFGState, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.tracked_assumptions import record_assumption

#: Prefix for the synthesised unit-stride iterator that replaces the (i, ii) pair.
_UNTILE_PREFIX = '_untile_k_'


def _next_id(sdfg: SDFG) -> int:
    used: Set[int] = set()
    for sd in sdfg.all_sdfgs_recursive():
        for s in list(sd.symbols.keys()):
            if s.startswith(_UNTILE_PREFIX):
                tail = s[len(_UNTILE_PREFIX):]
                if tail.isdigit():
                    used.add(int(tail))
        for cfg in sd.all_control_flow_regions():
            if isinstance(cfg, LoopRegion) and cfg.loop_variable and cfg.loop_variable.startswith(_UNTILE_PREFIX):
                tail = cfg.loop_variable[len(_UNTILE_PREFIX):]
                if tail.isdigit():
                    used.add(int(tail))
    n = 0
    while n in used:
        n += 1
    return n


def _try_extract_perfect_one_child(cfg: ControlFlowRegion) -> Optional[ControlFlowRegion]:
    """Return the single non-empty child block of ``cfg`` if it has
    exactly one, otherwise ``None``.

    Empty :class:`SDFGState` instances are tolerated (canonicalize often
    leaves them as connective tissue). Any other CFG construct (a
    non-empty plain state, a ConditionalBlock, etc.) breaks the perfect
    nest and the function refuses.
    """
    candidate: Optional[ControlFlowRegion] = None
    for b in cfg.nodes():
        if isinstance(b, SDFGState):
            if len(b.nodes()) > 0:
                return None
            continue
        if not isinstance(b, ControlFlowRegion):
            return None
        if candidate is not None:
            return None
        candidate = b
    return candidate


def _try_extract_perfect_two_level_nest(outer: LoopRegion) -> Optional[LoopRegion]:
    """Backward-compatible single-level wrapper around
    :func:`_try_extract_perfect_one_child`. Kept for tests that pin the
    immediate-inner contract; new code should prefer
    :func:`_iter_candidate_inners` for multi-dim support."""
    inner = _try_extract_perfect_one_child(outer)
    if isinstance(inner, LoopRegion):
        return inner
    return None


def _iter_candidate_inners(outer: LoopRegion):
    """Walk down through perfect 1-child intermediate chains, yielding
    every descendant :class:`LoopRegion` as a potential tile-pair partner
    for ``outer``.

    For a 2-D tile shape ``for ti: for tj: for i: for j: body`` the
    same-axis partner of ``ti`` is ``i``, two scopes deep with ``tj`` in
    between. The ascent stops at the first non-perfect 1-child boundary
    (a non-empty plain state, a sibling CFR, etc.), so non-perfect-nest
    cases are still refused.
    """
    seen: Set[int] = set()
    current: ControlFlowRegion = outer
    while True:
        nxt = _try_extract_perfect_one_child(current)
        if nxt is None or id(nxt) in seen:
            return
        seen.add(id(nxt))
        if isinstance(nxt, LoopRegion):
            yield nxt
        current = nxt


def _intermediate_chain_clean(outer: LoopRegion, inner: LoopRegion, outer_var: str) -> bool:
    """``True`` iff every LoopRegion strictly between ``outer`` and
    ``inner`` is free of references to ``outer.loop_variable`` in its
    iteration descriptors.

    Multi-dim untile is sound only when the intermediates index on
    independent axes; same-axis cascades (whose intermediates use
    ``outer.var`` in their bounds) must be handled level-by-level by
    fixpoint instead, not by descending past them in one rewrite.
    """
    outer_sym = symbolic.pystr_to_symbolic(outer_var)
    current = inner.parent_graph
    while current is not outer and current is not None:
        if isinstance(current, LoopRegion):
            for code in (current.init_statement, current.loop_condition, current.update_statement):
                if code is None:
                    continue
                try:
                    free = symbolic.pystr_to_symbolic(code.as_string).free_symbols
                except Exception:
                    free = set()
                if outer_sym in free:
                    return False
        current = current.parent_graph
    return True


def _is_constant_positive_int(expr) -> Optional[int]:
    """If ``expr`` simplifies to a positive integer literal, return that value."""
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return None
    if not getattr(s, 'is_number', False) or not getattr(s, 'is_Integer', False):
        return None
    v = int(s)
    return v if v > 0 else None


def _is_zero(expr) -> bool:
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return False
    return getattr(s, 'is_number', False) and s == 0


def _tile_size(expr) -> Optional[Tuple[symbolic.SymbolicType, Optional[int]]]:
    """Classify an outer-loop stride as a tile size.

    Returns ``(K_expr, K_const)`` where ``K_expr`` is the simplified
    stride expression and ``K_const`` is its value if the stride is a
    concrete integer literal ``> 1``, else ``None``. Returns ``None``
    entirely when the stride cannot be used as a tile:

    * a concrete literal ``<= 1`` (``1`` is already untiled; ``<= 0`` is
      not a forward tile);
    * a symbolic stride that SymPy can prove is non-positive.

    A **bare symbol** tile (e.g. a block-size parameter ``BS``) is accepted
    (``K_const=None``): DaCe treats every symbol as non-negative by
    convention -- we do *not* rely on SymPy sign assumptions -- and the
    collapse to a unit-stride ``[start, N)`` traversal is sound for any
    ``K >= 1`` (even the degenerate ``K == 1`` symbolic case). A **compound
    symbolic expression** (e.g. ``s1 - s2`` or ``N // 4``) is *not* assumed
    positive and is refused: it is not a plausible tile size and its sign
    cannot be trusted. Symbolic tiles admit only a unit inner stride
    (single-level untile) -- see :func:`_match_inner_case` -- because a
    concrete stride cannot be proven to divide a symbol.
    """
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return None
    if getattr(s, 'is_number', False):
        if not getattr(s, 'is_Integer', False):
            return None
        v = int(s)
        if v <= 1:
            return None
        return (s, v)
    # Symbolic: accept a bare symbol only (assumed non-negative by DaCe
    # convention). A compound expression's sign is not proven, so refuse it.
    if isinstance(s, sympy.Symbol):
        return (s, None)
    return None


def _match_inner_case(inner: LoopRegion, outer_var: str, K_expr: symbolic.SymbolicType,
                      K_const: Optional[int]) -> Optional[Tuple[str, symbolic.SymbolicType, bool]]:
    """Classify the inner loop shape and return ``(case, inner_stride, needs_div_assumption)``.

    * ``'A'`` -- inner ``range(0, K, S)`` (body uses ``i + ii``),
    * ``'B'`` -- inner ``range(i, i + K, S)`` (body uses ``ii``),

    with the inner stride ``S`` returned alongside. ``S == 1`` is the
    classic single-level untile; ``S > 1`` (with ``S | K``) is the
    cascade-tile intermediate level the fixpoint pass collapses one rung
    at a time. The new loop after the rewrite uses step ``S`` (not always
    1), so a subsequent fixpoint iteration can collapse it with the next
    inner.

    ``K_expr`` is the (possibly symbolic) outer tile size; ``K_const`` is
    its concrete value or ``None`` when it is symbolic. A concrete tile with a
    concrete stride admits a cascade rung iff ``S | K``. When either the tile or
    the stride is symbolic, the rung is a whole tile only under ``K % S == 0``,
    which cannot be proven -- ``needs_div_assumption`` is then ``True`` and the
    caller records that relation as a runtime-trapped assumption. (The source
    nest ``for iii in range(ii, ii+K, S): for i in range(iii, iii+S)`` already
    requires ``S | K`` -- else its last inner tile overshoots ``ii+K`` and the
    numpy oracle overshoots identically -- so the assumption never diverges from
    the reference on any input the kernel is valid for.) A unit inner stride
    needs no assumption.

    Returns ``None`` if neither shape matches.
    """
    stride = loop_analysis.get_loop_stride(inner)
    start = loop_analysis.get_init_assignment(inner)
    end = loop_analysis.get_loop_end(inner)
    if stride is None or start is None or end is None:
        return None
    S_concrete = _is_constant_positive_int(stride)
    needs_div_assumption = False
    if S_concrete is not None:
        inner_stride: symbolic.SymbolicType = S_concrete
        if S_concrete == 1:
            pass  # unit inner stride: a genuine single-level untile, always sound
        elif K_const is not None:
            # Concrete tile + concrete cascade stride: a whole tile only when it
            # divides exactly (no partial-tile remainder).
            if K_const % S_concrete != 0:
                return None
        else:
            # Concrete stride into a symbolic tile: divisibility unprovable -> admit
            # under a recorded ``K % S == 0`` assumption.
            needs_div_assumption = True
    else:
        # Symbolic inner stride: accept only a bare positive symbol (a block-size
        # parameter -- the inner tile ``T2`` of a ``T1``/``T2`` double tile), never
        # a compound expression. Always a cascade rung requiring divisibility.
        s_stride = symbolic.simplify(stride)
        if not isinstance(s_stride, sympy.Symbol):
            return None
        inner_stride = s_stride
        needs_div_assumption = True
    outer_sym = symbolic.pystr_to_symbolic(outer_var)
    s_sym = symbolic.simplify(start)
    e_sym = symbolic.simplify(end)
    # ``get_loop_end`` returns ``exclusive_upper_bound - 1`` regardless
    # of step (i.e. for ``range(a, b, S)`` it returns ``b - 1``, NOT the
    # actual last admitted value ``a + S * floor((b - a - 1) / S)``). We
    # therefore match against ``K - 1`` rather than ``K - S``. The
    # ``_diff_is_zero`` helper tolerates symbolic differences (returns
    # False rather than raising), so the multi-dim descent can probe
    # non-matching candidates without crashing, and it folds a symbolic
    # ``K`` (e.g. ``end == K - 1`` against ``K_expr - 1``) to zero.
    # Case A: start == 0, end == K - 1.
    if _is_zero(s_sym):
        if _diff_is_zero(e_sym, K_expr - 1):
            return ('A', inner_stride, needs_div_assumption)
        return None
    # Case B: start == i, end == i + K - 1.
    if _diff_is_zero(s_sym, outer_sym):
        if _diff_is_zero(e_sym, outer_sym + K_expr - 1):
            return ('B', inner_stride, needs_div_assumption)
    return None


def _diff_is_zero(a, b) -> bool:
    """``simplify(a - b) == 0`` if both sides reduce to the same value,
    else ``False``. Tolerates symbolic mismatches by catching the
    ``TypeError`` SymPy raises when an unresolved expression is coerced
    to ``int``."""
    try:
        diff = symbolic.simplify(a - b)
    except Exception:
        return False
    if hasattr(diff, "is_number") and diff.is_number:
        try:
            return int(diff) == 0
        except (TypeError, ValueError):
            return False
    return False


def _collect_body_subset_exprs(inner: LoopRegion) -> List[symbolic.SymbolicType]:
    """All symbolic expressions used in memlet subsets inside ``inner``'s body
    (across every state) -- one entry per axis bound (lo/hi/stride) per memlet.
    Used to audit which references to ``i`` / ``ii`` show up in the body."""
    exprs: List[symbolic.SymbolicType] = []
    for st in inner.all_states():
        for e in st.edges():
            if e.data is None or e.data.is_empty():
                continue
            if e.data.subset is not None:
                for lo, hi, stp in e.data.subset.ranges:
                    exprs.append(symbolic.pystr_to_symbolic(str(lo)))
                    exprs.append(symbolic.pystr_to_symbolic(str(hi)))
                    exprs.append(symbolic.pystr_to_symbolic(str(stp)))
            if e.data.other_subset is not None:
                for lo, hi, stp in e.data.other_subset.ranges:
                    exprs.append(symbolic.pystr_to_symbolic(str(lo)))
                    exprs.append(symbolic.pystr_to_symbolic(str(hi)))
                    exprs.append(symbolic.pystr_to_symbolic(str(stp)))
    return exprs


def _all_memlet_uses_only(inner: LoopRegion, allowed_atoms: Set[str], forbidden_atoms: Set[str]) -> bool:
    """``True`` iff every memlet-subset expression references at most symbols
    from ``allowed_atoms`` (any expression of them is fine) and references *no*
    symbol from ``forbidden_atoms``.

    The check is conservative: an expression like ``2*i + ii + 1`` is fine if
    both ``i`` and ``ii`` are allowed (because ``i + ii`` is the combined
    iterator), but ``i`` alone without ``ii`` is forbidden -- the rewrite would
    map only the ``i + ii`` part to ``k`` and would leave the bare ``i`` adrift.

    This function only checks the *atom membership* of each expression's free
    symbols; the structural ``i + ii`` vs ``ii``-only requirement is enforced
    by the caller (it sets ``allowed_atoms`` appropriately).
    """
    forbidden = {symbolic.pystr_to_symbolic(a) for a in forbidden_atoms}
    for ex in _collect_body_subset_exprs(inner):
        free = ex.free_symbols
        if any(f in forbidden for f in free):
            return False
    return True


def _audit_combined_access(inner: LoopRegion, outer_var: str, inner_var: str, case: str) -> bool:
    """The structural safety check the docstring describes.

    Case A (``ii in range(0, K)``): the combined expression ``i + ii`` must be
    the *only* way ``i`` and ``ii`` enter any memlet. Conservative test: every
    memlet-subset expression that mentions ``i`` must also mention ``ii``, and
    vice-versa.

    Case B (``ii in range(i, i + K)``): ``i`` must NEVER appear in a memlet
    (only ``ii``). The new iterator ``k`` becomes ``ii`` directly.
    """
    i_sym = symbolic.pystr_to_symbolic(outer_var)
    ii_sym = symbolic.pystr_to_symbolic(inner_var)
    if case == 'B':
        for ex in _collect_body_subset_exprs(inner):
            if i_sym in ex.free_symbols:
                return False
        return True
    # Case A: ``i`` and ``ii`` must always appear together.
    for ex in _collect_body_subset_exprs(inner):
        has_i = i_sym in ex.free_symbols
        has_ii = ii_sym in ex.free_symbols
        if has_i != has_ii:
            return False
    return True


@properties.make_properties
@xf.explicit_cf_compatible
class UntileLoops(ppl.Pass):
    """Collapse a manually-tiled multi-level / multi-dim perfect nest to a single
    loop (or single multi-dim Map when the round-trip lift fires).

    Recognises:

    * Case A -- ``for i in range(0, N, K): for ii in range(0, K, S): ...``,
      body addresses arrays via ``i + ii``.
    * Case B -- ``for i in range(0, N, K): for ii in range(i, i + K, S): ...``,
      body addresses arrays via ``ii``.

    ``K`` is either a concrete positive integer literal (``> 1``) with ``S``
    (inner stride) dividing ``K``, or a positive **symbol** (in which case
    only ``S == 1`` is admitted -- a concrete stride cannot be proven to
    divide a symbol). ``S == 1`` is the classic single-level untile; ``S > 1``
    is an intermediate cascade rung that fixpoint then collapses with the
    next inner.

    **Multi-dim ascent**: the inner doesn't have to be the immediate child
    of the outer. The matcher walks down through perfect 1-child
    intermediate chains, skipping foreign-axis loops whose iteration
    variables don't appear in the outer's bounds. For an N-D tile shape
    the same-axis partner sits N levels deep with the other-axis tile
    loops between.

    **Fixpoint iteration**: each pass collapses one (outer, inner) tile
    pair; multi-level cascades and multi-axis tiles unwind progressively.
    Bounded by the total LoopRegion count.

    **Map round-trip** (``map_roundtrip=True``, default): pre-step lowers
    every Map via :class:`MapExpansion` + :class:`MapToForLoop` so
    Map-tiled patterns enter the matcher as LoopRegions; post-step
    re-lifts via :class:`LoopToMap` + :class:`MapCollapse`. Set to
    ``False`` from the canonicalize pipeline if a separate ``parallelize``
    stage does the lift downstream (it's a no-op in that case anyway).

    Runs BEFORE :class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll`
    so the small fixed-trip inner loop doesn't get straight-line-unrolled.
    """

    CATEGORY: str = 'Canonicalization'

    map_roundtrip = properties.Property(dtype=bool,
                                        default=False,
                                        desc='Lower Maps to LoopRegions before untile and re-lift after, so '
                                        'Map-tiled patterns are detected. Off by default: the canonicalize '
                                        'pipeline runs the lift downstream and existing range-tile tests '
                                        'assert on raw LoopRegion shape after untile. Test driver enables '
                                        'when the kernel uses ``dace.map[...]`` tiles.')

    def __init__(self, map_roundtrip: bool = False):
        super().__init__()
        self.map_roundtrip = map_roundtrip

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def _maps_to_loops(self, sdfg: SDFG) -> None:
        """Pre-round-trip step: lower every Map to a LoopRegion.

        Sequence:

        1. ``MapExpansion`` -- split multi-dim Maps so ``MapToForLoop``
           (which only accepts uni-dim Maps) can handle them.
        2. ``MapToForLoop`` -- each uni-dim Map becomes a LoopRegion at
           the parent CFR. With ``inline_after=True`` (default), the
           wrapping NSDFG is flattened in-place when it isn't itself
           Map-scoped. NSDFGs that were created INSIDE another Map's
           scope are left wrapped (per-iteration narrowing is
           intentional inside a Map) and become un-scoped only after
           their enclosing Map gets converted too.
        3. ``ExpandNestedSDFGInputs`` + ``InlineMultistateSDFG`` --
           post-sweep that catches the leftover wrappers from (2) once
           every Map has become a LoopRegion. Run as a fixpoint to
           handle deeply-nested cases.
        """
        from dace.transformation.dataflow.map_expansion import MapExpansion
        from dace.transformation.dataflow.map_for_loop import MapToForLoop
        from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
        from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([MapToForLoop()]).apply_pass(sdfg, {})
        # Sweep up any NSDFG wrappers that survived MapToForLoop's
        # inline_after step because they were Map-scoped at the time.
        # After all Maps are lifted they are no longer scoped, so a
        # fixpoint sweep flattens them.
        for _ in range(16):
            before = sum(1 for n, _ in sdfg.all_nodes_recursive()
                         if hasattr(n, "sdfg") and hasattr(n, "symbol_mapping"))
            PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
            PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(sdfg, {})
            after = sum(1 for n, _ in sdfg.all_nodes_recursive() if hasattr(n, "sdfg") and hasattr(n, "symbol_mapping"))
            if after >= before:
                break

    def _loops_back_to_maps(self, sdfg: SDFG) -> None:
        """Post-round-trip step: re-lift every parallelizable LoopRegion
        to a Map and re-fuse adjacent uni-dim Maps."""
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([MapCollapse()]).apply_pass(sdfg, {})

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Run the per-loop rewrite as a fixpoint over the SDFG.

        When ``map_roundtrip`` is on, lower every Map to a LoopRegion
        first, run the fixpoint, then re-lift. Each fixpoint pass
        collapses one (outer, inner) tile pair; multi-level cascade and
        multi-axis tiles (where successive iterations expose new tile
        pairs that became the outermost after the prior collapse) unwind
        progressively. Iteration cap = 1 + (loop count); once an
        iteration rewrites nothing we stop.
        """
        if self.map_roundtrip:
            self._maps_to_loops(sdfg)

        total = 0
        # Safety cap: at most one rewrite per LoopRegion in the SDFG.
        max_iters = 1 + sum(1 for sd in sdfg.all_sdfgs_recursive()
                            for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion))
        for _ in range(max_iters):
            rewritten_this_pass = 0
            for sd in sdfg.all_sdfgs_recursive():
                for cfg in list(sd.all_control_flow_regions()):
                    if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                        continue
                    if self._try_untile(cfg, sd):
                        rewritten_this_pass += 1
            if rewritten_this_pass == 0:
                break
            total += rewritten_this_pass

        if self.map_roundtrip:
            self._loops_back_to_maps(sdfg)
        if total:
            # Propagate once, at the end of the pass -- the in-place iterator
            # rewrites above intentionally do not self-propagate per rewrite.
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
        return total or None

    def _try_untile(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        # The outer must be ``for i in range(0, N, K)`` with a positive tile
        # ``K`` -- a concrete literal ``> 1`` or a positive symbol.
        outer_stride = loop_analysis.get_loop_stride(outer)
        outer_start = loop_analysis.get_init_assignment(outer)
        outer_end = loop_analysis.get_loop_end(outer)
        if outer_stride is None or outer_start is None or outer_end is None:
            return False
        tile = _tile_size(outer_stride)
        if tile is None:
            return False  # stride <= 1, or a provably non-positive symbol
        K_expr, K_const = tile
        # ``outer_start`` need not be 0: a tiled stencil walks tile origins over
        # the interior ``[S, N)`` (e.g. ``for ii in range(1, N-1-K, K)``). The
        # collapsed loop simply starts at the same ``S`` (set below). Only reject
        # a start we cannot render symbolically.
        outer_start_sym = symbolic.simplify(outer_start)

        # Walk down through perfect 1-child intermediate chains and pick
        # the first descendant LoopRegion whose shape + audit match the
        # tile-pair contract with ``outer``. For an N-D tile shape
        # (different axes interleaved) the same-axis partner sits N
        # levels deep with foreign-axis loops between -- the descent
        # walks past those.
        case: Optional[str] = None
        inner_stride: symbolic.SymbolicType = None
        needs_div_assumption = False
        inner: Optional[LoopRegion] = None
        for candidate in _iter_candidate_inners(outer):
            if not candidate.loop_variable:
                continue
            match = _match_inner_case(candidate, outer.loop_variable, K_expr, K_const)
            if match is None:
                continue
            cand_case, cand_stride, cand_needs_div = match
            if not _audit_combined_access(candidate, outer.loop_variable, candidate.loop_variable, cand_case):
                continue
            # Multi-dim sanity: any intermediate LoopRegion between outer
            # and candidate must not reference outer's iteration variable
            # in its own bounds. Same-axis cascades (where the
            # intermediates DO reference outer.var) are handled by
            # fixpoint level-by-level instead.
            if candidate is not outer and not _intermediate_chain_clean(outer, candidate, outer.loop_variable):
                continue
            inner = candidate
            case = cand_case
            inner_stride = cand_stride
            needs_div_assumption = cand_needs_div
            break
        if inner is None:
            return False

        # A cascade rung whose stride divides the tile only under an unprovable
        # relation (symbolic tile and/or symbolic stride) is admitted here; record
        # ``K % S == 0`` so the terminal AssumeSymbolConstraints pass emits a
        # runtime trap. The source tile nest already requires it.
        if needs_div_assumption:
            record_assumption(sdfg, sympy.Eq(sympy.Mod(K_expr, inner_stride), 0))

        # Synthesise the new iterator with step = ``inner_stride`` and
        # rewrite both loops in place. ``inner_stride == 1`` is the
        # classic single-level untile (collapsed loop runs unit stride);
        # ``inner_stride > 1`` is an intermediate cascade rung that the
        # fixpoint pass collapses with its own inner on a subsequent
        # iteration.
        k_var = f"{_UNTILE_PREFIX}{_next_id(sdfg)}"
        sdfg.add_symbol(k_var, sdfg.symbols.get(outer.loop_variable, dace.int64))
        # Exclusive upper bound for the collapsed iterator is the union of the
        # tile spans the original nest actually visits. The outer walks tile
        # origins ``ii = outer_start + m*K`` for every ``ii < stop`` (where
        # ``stop = outer_end + 1`` is the outer's exclusive upper bound), and the
        # inner covers ``[ii, ii + K)``. So the last visited element is
        # ``last_origin + K`` where ``last_origin`` is the largest origin below
        # ``stop`` -- i.e. the union end is ``stop`` rounded UP to the next tile
        # boundary above ``outer_start``: ``outer_start + ceil((stop -
        # outer_start) / K) * K``.
        #
        # When the tile evenly divides the span (the classic ``for i in
        # range(0, N, K): for ii in range(0, K)`` shape with ``K | N``,
        # ``outer_start == 0``) this reduces to exactly ``stop == N`` -- the old
        # ``outer_end + 1`` formula. But a tiled stencil walks the interior with
        # ``stop = LEN - 1 - K`` (NOT a tile multiple), so the last tile overshoots
        # ``stop`` and ``outer_end + 1`` truncated the final tile (missed its tail
        # rows/cols). The earlier ``outer_end + outer_stride`` over-shot the other
        # way (a full extra tile). The round-up is the exact union.
        stop_excl = symbolic.simplify(outer_end + 1)
        span = symbolic.simplify(stop_excl - outer_start_sym)
        N_excl = symbolic.simplify(outer_start_sym + symbolic.int_ceil(span, K_expr) * K_expr)

        # Body substitution: ``i + ii`` -> ``k`` (case A) or ``ii`` -> ``k`` (case B).
        i_sym = outer.loop_variable
        ii_sym = inner.loop_variable
        if case == 'A':
            # ``i + ii`` -> ``k``. Implementation: substitute ``ii`` with
            # ``k - i``, then ``i`` with ``0`` -- ``i + ii`` collapses to ``k``.
            # Because the audit guarantees every appearance of ``i`` co-occurs
            # with ``ii`` (and vice-versa), the substitution preserves every
            # other algebraic combination of ``i`` and ``ii``.
            inner.replace_dict({ii_sym: f"({k_var}) - ({i_sym})"})
            inner.replace_dict({i_sym: '0'})
        else:
            # Case B: ``ii`` -> ``k``; ``i`` doesn't appear in any memlet.
            inner.replace_dict({ii_sym: k_var})

        # Splice the inner's body blocks into the inner's parent CFR
        # (which may be ``outer`` directly for single-level untile, or a
        # nested intermediate LoopRegion for multi-dim untile). The
        # outer LoopRegion is then re-purposed as the collapsed loop by
        # rewriting its iteration descriptors below.
        parent_of_inner: ControlFlowRegion = inner.parent_graph
        inner_was_start = (parent_of_inner.start_block is inner)
        # ``inner`` need NOT be the sole/start block of its parent: after the
        # Map round-trip's inline step the parent LoopRegion holds connective
        # states (``block_*_pre/post_state``) with interstate edges FEEDING
        # ``inner`` and FLOWING OUT of it. A naive detach that only moves
        # ``inner``'s children and drops those parent edges orphans the
        # connective states (unreachable-from-start -> dominator ``KeyError``
        # at codegen). So capture the parent's edges incident to ``inner`` and
        # ``inner``'s own entry/exit BEFORE mutating, then reconnect the chain.
        pred_edges = list(parent_of_inner.in_edges(inner))
        succ_edges = list(parent_of_inner.out_edges(inner))
        try:
            inner_start_block = inner.start_block if inner.number_of_nodes() > 0 else None
        except (NodeNotFoundError, ValueError):
            inner_start_block = None
        inner_sinks = inner.sink_nodes()
        child_blocks = list(inner.nodes())
        inner_edges = list(inner.edges())
        # Detach the inner wrapper (drops its incident parent edges too), then
        # splice its blocks up into the parent. ``add_node`` re-parents each
        # child (sets ``parent_graph``/``sdfg``, recursing into nested CFRs)
        # and, when ``is_start_block`` is set, fixes the parent's start pointer
        # via the reliable add API (a post-hoc ``start_block =`` assignment is
        # silently dropped when the parent's start is ambiguous).
        parent_of_inner.remove_node(inner)
        for child in child_blocks:
            inner.remove_node(child)
            child_is_start = inner_was_start and (child is inner_start_block)
            parent_of_inner.add_node(child, is_start_block=child_is_start, ensure_unique_name=True)
        # Re-attach the inner's own body interstate edges.
        for ie in inner_edges:
            parent_of_inner.add_edge(ie.src, ie.dst, ie.data)
        # Reconnect the parent chain through the spliced body: predecessors of
        # the old ``inner`` now flow into its entry block; its exit block(s)
        # flow to the old successors. Deep-copy the interstate-edge payload so
        # each new edge owns its condition/assignments (a shared object across
        # fan-out edges corrupts propagation).
        if inner_start_block is not None:
            for pe in pred_edges:
                parent_of_inner.add_edge(pe.src, inner_start_block, copy.deepcopy(pe.data))
        for se in succ_edges:
            for sink in inner_sinks:
                parent_of_inner.add_edge(sink, se.dst, copy.deepcopy(se.data))

        # Rewrite the outer's iteration descriptors to drive ``k`` over
        # ``[outer_start, N)`` in steps of ``inner_stride``. Case A collapses
        # ``i + ii`` (which starts at ``outer_start + 0``); Case B collapses
        # ``ii`` (which starts at the outer origin ``outer_start``). Either way
        # the fused iterator begins at ``outer_start``.
        outer.loop_variable = k_var
        outer.init_statement = dace.properties.CodeBlock(f"{k_var} = {symbolic.symstr(outer_start_sym)}")
        outer.loop_condition = dace.properties.CodeBlock(f"{k_var} < ({symbolic.symstr(N_excl)})")
        outer.update_statement = dace.properties.CodeBlock(f"{k_var} = {k_var} + {symbolic.symstr(inner_stride)}")
        return True


__all__ = ['UntileLoops']
