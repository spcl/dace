# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collapse a manually-tiled loop nest back to a single unit-stride loop per axis.

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
a block-size parameter). The single body block of the outer must be exactly one
nested :class:`~dace.sdfg.state.LoopRegion` and nothing else (perfect nest).
A cascade rung (inner stride ``S > 1``) needs ``S | K``: proven outright when
both are concrete, otherwise admitted under a recorded divisibility assumption,
which is what lets a double- or triple-tiled nest with symbolic tiles unwind
level by level.

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
from typing import Dict, List, Optional, Tuple

import sympy

import dace
from dace import SDFG, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.graph import NodeNotFoundError
from dace.sdfg.state import LoopRegion, SDFGState, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.tracked_assumptions import record_assumption

#: Prefix for the synthesised unit-stride iterator that replaces the (i, ii) pair.
_UNTILE_PREFIX = '_untile_k_'


def count_applied(result) -> int:
    """Number of transformations a ``PatternMatchAndApplyRepeated`` run applied.

    It returns ``{transformation name: [applied, ...]}``, or ``None`` when it matched nothing.
    Callers need the count to report their own modification honestly.
    """
    if not result:
        return 0
    return sum(len(applied) for applied in result.values())


def _next_id(sdfg: SDFG) -> int:
    used: Dict[int, None] = {}
    for sd in sdfg.all_sdfgs_recursive():
        for s in list(sd.symbols.keys()):
            if s.startswith(_UNTILE_PREFIX):
                tail = s[len(_UNTILE_PREFIX):]
                if tail.isdigit():
                    used[int(tail)] = None
        for cfg in sd.all_control_flow_regions():
            if isinstance(cfg, LoopRegion) and cfg.loop_variable and cfg.loop_variable.startswith(_UNTILE_PREFIX):
                tail = cfg.loop_variable[len(_UNTILE_PREFIX):]
                if tail.isdigit():
                    used[int(tail)] = None
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
    seen: Dict[int, None] = {}
    current: ControlFlowRegion = outer
    while True:
        nxt = _try_extract_perfect_one_child(current)
        if nxt is None or id(nxt) in seen:
            return
        seen[id(nxt)] = None
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
                    free = {}
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
    if not s.is_number or not s.is_Integer:
        return None
    v = int(s)
    return v if v > 0 else None


def _is_zero(expr) -> bool:
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return False
    return s.is_number and s == 0


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
    if s.is_number:
        if not s.is_Integer:
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


def map_tile_partner(outer: nodes.Map, inner: nodes.Map) -> bool:
    """``True`` iff ``inner`` spans exactly one tile of ``outer`` on some axis.

    Map-range mirror of :func:`_match_inner_case`: an outer dimension with a tile stride ``K``
    against an inner dimension running ``[0, K-1]`` (case A) or ``[p, p + K - 1]`` (case B) for
    that outer dimension's parameter ``p``. Subset ranges carry an INCLUSIVE end, which is the
    same convention ``loop_analysis.get_loop_end`` returns, so the comparisons match the
    LoopRegion matcher's.
    """
    for p, (_, _, step) in zip(outer.params, outer.range):
        tile = _tile_size(step)
        if tile is None:
            continue
        k_expr = tile[0]
        p_sym = symbolic.pystr_to_symbolic(p)
        for begin, end, _ in inner.range:
            if _is_zero(begin) and _diff_is_zero(end, k_expr - 1):
                return True
            if _diff_is_zero(begin, p_sym) and _diff_is_zero(end, p_sym + k_expr - 1):
                return True
    return False


def map_tile_pattern_present(sdfg: SDFG) -> bool:
    """``True`` iff the SDFG holds a hand-tiled **Map** nest, i.e. a Map whose stride is a tile
    size and another Map covering one such tile.

    The matcher itself only reads LoopRegions, so a Map-tiled nest is invisible to it unless the
    round trip lowers the Maps first; this is the trigger that decides whether that round trip is
    worth running (see :meth:`UntileLoops.apply_pass`). Tile-strided Maps are rare, so the scan
    bails on the first sweep for almost every SDFG. GPU-scheduled Maps are skipped: lowering
    device parallelism to a sequential loop is not a canonicalization.
    """
    tiled: List[nodes.Map] = []
    every: List[nodes.Map] = []
    for n, _ in sdfg.all_nodes_recursive():
        if not isinstance(n, nodes.MapEntry) or n.map.schedule in dtypes.GPU_SCHEDULES:
            continue
        every.append(n.map)
        if any(_tile_size(step) is not None for _, _, step in n.map.range):
            tiled.append(n.map)
    if not tiled:
        return False
    return any(map_tile_partner(outer, inner) for outer in tiled for inner in every if inner is not outer)


def tiles_a_parent_window(outer: LoopRegion, start, span) -> bool:
    """``True`` iff ``outer`` walks a fixed-width window opened by an enclosing loop.

    The witness is the pair (start depends on an enclosing loop's iterator, width does not): that
    is the cascade rung ``for iiii in range(iii, iii + T2, T3)``, whose union is the window
    ``[iii, iii + T2)`` itself. A loop over a global extent (``range(1, LEN - 1 - T, T)``) fails
    the test and keeps the round-up bound, which is what makes its intended overshoot past the
    last origin survive.
    """
    enclosing: Dict[str, None] = {}
    graph = outer.parent_graph
    while graph is not None:
        if isinstance(graph, LoopRegion) and graph.loop_variable:
            enclosing[graph.loop_variable] = None
        graph = graph.parent_graph
    if not enclosing or not any(str(s) in enclosing for s in start.free_symbols):
        return False
    return not any(str(s) in enclosing for s in span.free_symbols)


def count_loops(sdfg: SDFG) -> int:
    """Number of iterating LoopRegions anywhere in ``sdfg`` (nested SDFGs included)."""
    return sum(1 for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
               if isinstance(r, LoopRegion) and r.loop_variable)


def clamp_is_the_parent_limit(end_plus_one, outer_sym, K_expr, outer_limit) -> bool:
    """``True`` iff ``end_plus_one`` is ``Min(parent limit, i + K)`` -- the remainder clamp.

    A strip-mined nest whose last tile would overrun writes its inner bound as
    ``min(i + K, <the parent's own limit>)``, and that clamp is exactly what makes the union of
    the tiles equal the original range rather than exceed it. Untiling is therefore EXACT here,
    which is the whole reason the clamp may be accepted.

    The clamp has to be the parent's limit and nothing else. ``min(i + K, something_smaller)``
    would visit fewer points than the flat loop, so collapsing it would invent iterations; the
    check refuses anything it cannot match to ``outer_limit`` term for term.
    """
    if not isinstance(end_plus_one, sympy.Min):
        return False
    tile_reach = symbolic.simplify(outer_sym + K_expr)
    rest = []
    saw_tile_reach = False
    for arg in end_plus_one.args:
        if not saw_tile_reach and _diff_is_zero(arg, tile_reach):
            saw_tile_reach = True
            continue
        rest.append(arg)
    if not saw_tile_reach or not rest:
        return False
    # The remaining terms are compared TOGETHER, not one by one. A correctly strip-mined cascade
    # clamps each level to every window enclosing it -- ``min(i3 + W3, i2 + W2, i1 + W1, N - 1)``
    # -- so no single term equals the parent's limit; their Min does, and that is the parent's
    # limit spelled out. Term-by-term comparison would refuse exactly the well-formed nest.
    return _diff_is_zero(sympy.Min(*rest), outer_limit)


def _match_inner_case(inner: LoopRegion,
                      outer_var: str,
                      K_expr: symbolic.SymbolicType,
                      K_const: Optional[int],
                      outer_limit=None) -> Optional[Tuple[str, symbolic.SymbolicType, bool, bool]]:
    """Classify the inner shape: ``(case, inner_stride, needs_div_assumption, clamped)``.

    ``clamped`` says the inner bound carried the remainder clamp. The caller MUST honour it:
    an unclamped tile overshoots its last span and the collapsed bound has to round up to the
    tile boundary, while a clamped one covers the parent range exactly and rounding up walks
    off the end of the array.

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
            return ('A', inner_stride, needs_div_assumption, False)
        return None
    # Case B: start == i, end == i + K - 1 -- or the same with the remainder clamp,
    # end == min(<parent limit>, i + K) - 1, which every hand-tiled stencil writes.
    if _diff_is_zero(s_sym, outer_sym):
        if _diff_is_zero(e_sym, outer_sym + K_expr - 1):
            return ('B', inner_stride, needs_div_assumption, False)
        if outer_limit is not None and clamp_is_the_parent_limit(symbolic.simplify(e_sym + 1), outer_sym, K_expr,
                                                                 outer_limit):
            return ('B', inner_stride, needs_div_assumption, True)
    return None


def _diff_is_zero(a, b) -> bool:
    """``simplify(a - b) == 0`` if both sides reduce to the same value,
    else ``False``. Tolerates symbolic mismatches by catching the
    ``TypeError`` SymPy raises when an unresolved expression is coerced
    to ``int``."""
    try:
        # Sides reach here from different sources -- a memlet bound, a reparsed loop expression, a
        # descriptor shape -- so one name arrives as several sympy instances that never cancel.
        if isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic):
            a, b = symbolic.equalize_symbols_across(a, b)
        diff = symbolic.simplify(a - b)
    except Exception:
        return False
    if diff.is_number:
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


def depends_only_on_sum(ex: sympy.Basic, i_sym: sympy.Symbol, ii_sym: sympy.Symbol) -> bool:
    """``True`` iff ``ex`` reads ``i`` and ``ii`` only through the sum ``i + ii``.

    The case-A rewrite substitutes ``ii -> k - i`` and then ``i -> 0``, which preserves a
    subexpression's value exactly when that subexpression is a function of ``i + ii``: any such
    function has equal partials in both. Co-occurrence is NOT sufficient -- ``2*i + ii`` and
    ``i**2 + ii**2`` both mention the two together yet collapse to ``k`` and ``k**2``. Anything
    sympy cannot differentiate (``int_floor`` and friends) refuses, so the audit fails closed.
    """
    try:
        # diff goes through symbol IDENTITY: a loop var minted from its name differentiates to 0
        # against an expression carrying another instance, which reads as "equal partials" and
        # lets the rewrite through. Equalize so the partials are taken w.r.t. what ex holds.
        ex, i_sym, ii_sym = symbolic.equalize_symbols_across(ex, i_sym, ii_sym)
        return symbolic.simplify(sympy.diff(ex, i_sym) - sympy.diff(ex, ii_sym)) == 0
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return False


def _audit_combined_access(inner: LoopRegion, outer_var: str, inner_var: str, case: str) -> bool:
    """The structural safety check the docstring describes.

    Case A (``ii in range(0, K)``): ``i`` and ``ii`` must enter every memlet only as ``i + ii``.

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
    return all(depends_only_on_sum(ex, i_sym, ii_sym) for ex in _collect_body_subset_exprs(inner))


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

    ``K`` is a concrete positive integer literal (``> 1``) or a positive
    **symbol**. ``S == 1`` is the classic single-level untile; ``S > 1`` is an
    intermediate cascade rung that fixpoint then collapses with the next inner,
    and needs ``S | K`` -- decided exactly when both are concrete, recorded as
    a tracked assumption when either is symbolic.

    **Multi-dim ascent**: the inner doesn't have to be the immediate child
    of the outer. The matcher walks down through perfect 1-child
    intermediate chains, skipping foreign-axis loops whose iteration
    variables don't appear in the outer's bounds. For an N-D tile shape
    the same-axis partner sits N levels deep with the other-axis tile
    loops between.

    **Fixpoint iteration**: each pass collapses one (outer, inner) tile
    pair; multi-level cascades and multi-axis tiles unwind progressively.
    Bounded by the total LoopRegion count.

    **Map round-trip**: pre-step lowers every Map via :class:`MapExpansion` +
    :class:`MapToForLoop` so Map-tiled patterns enter the matcher as
    LoopRegions; post-step re-lifts via :class:`LoopToMap` +
    :class:`MapCollapse`. It runs when ``map_roundtrip=True`` forces it, and
    otherwise whenever :func:`map_tile_pattern_present` finds a hand-tiled Map
    nest AND :meth:`roundtrip_recovers_maps` confirms on a copy that the trip
    both untiles something and gives every lowered Map back as a Map. A
    ``dace.map``-tiled kernel is therefore recovered on the pass's defaults --
    without the auto-trigger the matcher never sees it and the hand-written
    tiling survives into codegen.

    Runs BEFORE :class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll`
    so the small fixed-trip inner loop doesn't get straight-line-unrolled.
    """

    CATEGORY: str = 'Canonicalization'

    map_roundtrip = properties.Property(dtype=bool,
                                        default=False,
                                        desc='Force the Map -> LoopRegion -> Map round trip that exposes '
                                        'Map-tiled patterns to the matcher. Off by default, in which case the '
                                        'trip is taken only for SDFGs that hold a Map tile nest and only when '
                                        'a probe on a copy shows it pays off; forcing it lowers and re-lifts '
                                        'every Map unconditionally.')

    def __init__(self, map_roundtrip: bool = False):
        super().__init__()
        self.map_roundtrip = map_roundtrip

    def modifies(self) -> ppl.Modifies:
        # Nodes: the Map round trip creates and removes Map scopes (and the NSDFGs around them).
        return ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def _maps_to_loops(self, sdfg: SDFG) -> int:
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
        applied = count_applied(PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {}))
        lower_maps = MapToForLoop()
        lower_maps.keep_reductions_parallel = True  # canon preference, off in the transformation's default contract
        applied += count_applied(PatternMatchAndApplyRepeated([lower_maps]).apply_pass(sdfg, {}))
        # Sweep up any NSDFG wrappers that survived MapToForLoop's
        # inline_after step because they were Map-scoped at the time.
        # After all Maps are lifted they are no longer scoped, so a
        # fixpoint sweep flattens them.
        for _ in range(16):
            before = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
            applied += count_applied(PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {}))
            applied += count_applied(PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(sdfg, {}))
            after = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
            if after >= before:
                break
        return applied

    def _loops_back_to_maps(self, sdfg: SDFG) -> int:
        """Post-round-trip step: re-lift every parallelizable LoopRegion
        to a Map and re-fuse adjacent uni-dim Maps."""
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        applied = count_applied(PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(sdfg, {}))
        applied += count_applied(PatternMatchAndApplyRepeated([MapCollapse()]).apply_pass(sdfg, {}))
        return applied

    def untile_fixpoint(self, sdfg: SDFG) -> int:
        """Collapse tile pairs until none is left, and return how many were collapsed.

        Each sweep collapses one (outer, inner) tile pair per nest; multi-level cascade and
        multi-dim tiles (where successive sweeps expose the pair that became outermost after the
        prior collapse) unwind progressively. Iteration cap = 1 + (loop count); once a sweep
        rewrites nothing we stop.
        """
        total = 0
        max_iters = 1 + count_loops(sdfg)
        for _ in range(max_iters):
            rewritten_this_pass = 0
            for sd in sdfg.all_sdfgs_recursive():
                # INNERMOST FIRST. A cascade is collapsed one rung at a time, and the rung nearest
                # the body is the one whose partner is unambiguous: an outer tile loop in a
                # multi-level nest has every deeper loop as a candidate, and matching it against
                # the wrong rung fixes a pairing the levels below then cannot undo. Working up
                # from the body, each level meets a nest that has already been flattened beneath
                # it, so the next pair to collapse is the only pair left.
                for cfg in reversed(list(sd.all_control_flow_regions())):
                    if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                        continue
                    if self._try_untile(cfg, sd):
                        rewritten_this_pass += 1
            if rewritten_this_pass == 0:
                break
            total += rewritten_this_pass
        return total

    def roundtrip_recovers_maps(self, sdfg: SDFG) -> bool:
        """Whether taking the Map round trip on ``sdfg`` is worth it, decided on a copy.

        Two conditions, both necessary. The trip must actually untile something -- otherwise it is
        pure churn. And it must not leave a Map behind as a LoopRegion: the lowering is applied to
        EVERY Map, and a Map that ``LoopToMap`` cannot re-derive (a scatter the user asserted was
        parallel, say) would come back sequential, trading the hand-written tiling for lost
        parallelism. Loop count is the exact witness -- untiling only ever removes loops, so any
        growth is a Map that failed to return.
        """
        probe = copy.deepcopy(sdfg)
        loops_before = count_loops(probe)
        self._maps_to_loops(probe)
        untiled = self.untile_fixpoint(probe)
        self._loops_back_to_maps(probe)
        return untiled > 0 and count_loops(probe) <= loops_before

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Run the per-loop rewrite as a fixpoint over the SDFG, around the Map round trip when
        that is forced or when a Map tile nest makes it pay off."""
        # The round trip rewrites the graph even when no tile pair is found, so its edits have to
        # be reported too -- returning None after lowering and re-lifting every map would tell the
        # caller nothing changed and let it reuse stale analyses.
        roundtrip = 0
        take_roundtrip = self.map_roundtrip or (map_tile_pattern_present(sdfg) and self.roundtrip_recovers_maps(sdfg))
        if take_roundtrip:
            roundtrip += self._maps_to_loops(sdfg)

        total = self.untile_fixpoint(sdfg)

        if take_roundtrip:
            roundtrip += self._loops_back_to_maps(sdfg)
        if total:
            # Propagate once, at the end of the pass -- the in-place iterator
            # rewrites above intentionally do not self-propagate per rewrite.
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
        # ``total`` counts untiled loops; the round trip's edits do not add to that number, but
        # they still mean "modified", hence 0 rather than None.
        if total:
            return total
        return 0 if roundtrip else None

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
        clamped = False
        inner: Optional[LoopRegion] = None
        for candidate in _iter_candidate_inners(outer):
            if not candidate.loop_variable:
                continue
            match = _match_inner_case(candidate,
                                      outer.loop_variable,
                                      K_expr,
                                      K_const,
                                      outer_limit=symbolic.simplify(outer_end + 1))
            if match is None:
                continue
            cand_case, cand_stride, cand_needs_div, cand_clamped = match
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
            clamped = cand_clamped
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
        #
        # One shape rounds up to a WRONG bound: a rung that walks a fixed-width window carved out
        # by an enclosing loop (``for iiii in range(iii, iii + T2, T3)`` inside the ``T2`` tile).
        # There the window IS the union, and the source nest is only well formed when the rung
        # divides it -- otherwise its own last tile overshoots the window, exactly as for the
        # cascade-stride rung above. With a symbolic width the round-up cannot fold, so it leaves
        # ``T3*int_ceil(T2, T3)`` where the enclosing rung expects ``T2``, and the next fixpoint
        # sweep no longer recognises the pair (measured on ``jacobi2d_triple_tiled_sym``: the
        # cascade stalled with two of three levels collapsed). Take the window as the union and
        # record the divisibility, same contract as the stride rung.
        stop_excl = symbolic.simplify(outer_end + 1)
        span = symbolic.simplify(stop_excl - outer_start_sym)
        if clamped:
            # ``min(i + K, stop)`` is precisely what stops the last tile overshooting, so the union
            # IS ``stop`` and there is nothing to round up. Rounding up anyway walks the collapsed
            # loop past the end of the array -- an out-of-bounds write, not a slow kernel.
            N_excl = stop_excl
        else:
            tiles_end = symbolic.simplify(symbolic.int_ceil(span, K_expr) * K_expr)
            if _diff_is_zero(tiles_end,
                             span) or tiles_end.is_number or not tiles_a_parent_window(outer, outer_start_sym, span):
                N_excl = symbolic.simplify(outer_start_sym + tiles_end)
            else:
                record_assumption(sdfg, sympy.Eq(sympy.Mod(span, K_expr), 0))
                N_excl = stop_excl

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
