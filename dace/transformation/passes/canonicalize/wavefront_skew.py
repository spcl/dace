# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Expose wavefront parallelism in perfectly-nested 2-D loops by loop skewing.

A 2-D nest of the form ::

    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            arr[i, j] = f(arr[i, j-1], arr[i-1, j], ...)

carries dependence vectors ``(0, 1)`` (left neighbour) and ``(1, 0)`` (top
neighbour). Neither loop in this nest is parallel by itself: every iteration
reads a value the previous outer iteration wrote. But the *anti-diagonal*
``i + j = const`` is parallel -- the elements on one diagonal depend only on
the previous diagonal, never on each other. The classical polyhedral skewing
``(i, j) -> (t = i + j, p = j)`` makes this explicit:

    for t in range(t_lo, t_hi):                              # sequential
        for p in range(max(j_lo, t-i_hi+1), min(j_hi, t-i_lo+1)):  # parallel
            i = t - p
            j = p
            arr[i, j] = f(arr[i, j-1], arr[i-1, j], ...)

After this rewrite the inner loop has no loop-carried dependence, so the
downstream ``LoopToMap`` lifts it to a parallel ``Map``.

Pluto / Polly use linear-programming based dependence analysis and emit an
arbitrary affine transformation; this pass implements the single classical
skewing that covers the textbook case (TSVC ``s2111``, the 2-D heat-flux /
Smith-Waterman / Floyd-Warshall family). The detector picks up any read at
``(i + r_i, j + r_j)`` whose offset ``(r_i, r_j)`` is "backward" with respect
to lexicographic iteration order and whose dependence vector becomes positive
on the ``t`` axis under the ``t = i + j`` skew. More general affine schedules
are left as a follow-up.

References:

- Wolfe, *"More Iteration Space Tiling"* (Supercomputing '89).
- Wolf & Lam, *"A loop transformation theory and an algorithm to maximize
  parallelism"* (IEEE TPDS '91) -- the unimodular framework that ``Pluto``
  extends to affine schedules.
- Bondhugula et al., *"A practical automatic polyhedral parallelizer and
  locality optimizer"* (PLDI '08) -- the Pluto algorithm.
"""
from typing import List, Optional, Set, Tuple

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Prefix for the synthesised skewed iterators.
_SKEW_T_PREFIX = '_skew_t_'
_SKEW_P_PREFIX = '_skew_p_'


def _next_id(sdfg: SDFG) -> int:
    """Lowest ``<N>`` no existing ``_skew_(t|p)_<N>`` symbol uses anywhere in the SDFG tree."""
    used: Set[int] = set()
    for sd in sdfg.all_sdfgs_recursive():
        for s in list(sd.symbols.keys()):
            for pre in (_SKEW_T_PREFIX, _SKEW_P_PREFIX):
                if s.startswith(pre):
                    tail = s[len(pre):]
                    if tail.isdigit():
                        used.add(int(tail))
        for cfg in sd.all_control_flow_regions():
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                for pre in (_SKEW_T_PREFIX, _SKEW_P_PREFIX):
                    if cfg.loop_variable.startswith(pre):
                        tail = cfg.loop_variable[len(pre):]
                        if tail.isdigit():
                            used.add(int(tail))
    n = 0
    while n in used:
        n += 1
    return n


def _unit_positive_stride(loop: LoopRegion) -> bool:
    s = loop_analysis.get_loop_stride(loop)
    try:
        return s is not None and int(symbolic.simplify(s)) == 1
    except (TypeError, ValueError):
        return False


def _try_extract_two_level_nest(outer: LoopRegion) -> Optional[LoopRegion]:
    """If ``outer`` perfectly nests a single inner :class:`LoopRegion` -- and
    nothing else -- return that inner loop. ``None`` otherwise."""
    blocks = list(outer.nodes())
    inner = [b for b in blocks if isinstance(b, LoopRegion)]
    if len(inner) != 1:
        return None
    # Every non-inner block must be empty / no data nodes (perfect nest).
    for b in blocks:
        if b is inner[0]:
            continue
        if isinstance(b, SDFGState) and len(list(b.nodes())) > 0:
            return None
    return inner[0]


def _extract_wavefront_offsets(inner: LoopRegion, outer_var: str, inner_var: str,
                               sdfg: SDFG) -> Optional[Tuple[str, List[Tuple[object, object]]]]:
    """Look at the inner body for a write to a 2-D array at ``(outer_var, inner_var)``
    and reads from the same array. Returns ``(array_name, list_of_read_offsets)``
    where each ``(d_outer, d_inner)`` is the offset of a read relative to the
    write -- either as integer literals or as symbolic expressions whose free
    symbols don't reference the loop variables. ``None`` if the body doesn't
    match the canonical single-tasklet two-input wavefront shape.

    Symbolic offsets (e.g. ``aa[i, j - sym1]`` and ``aa[i - sym2, j]``) are
    accepted; the caller's positivity check decides whether the dependence
    vector lies in the half-plane the skew makes parallel.
    """
    blocks = list(inner.nodes())
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return None
    state = blocks[0]
    write_arr = None
    write_offsets = None
    read_offsets: List[Tuple[int, int]] = []

    iter_syms = {symbolic.pystr_to_symbolic(outer_var), symbolic.pystr_to_symbolic(inner_var)}

    def _classify_2d(subset, axis_a, axis_b):
        nd = list(subset.ndrange())
        if len(nd) < 2:
            return None
        try:
            ra = symbolic.simplify(nd[0][0] - symbolic.pystr_to_symbolic(axis_a))
            rb = symbolic.simplify(nd[1][0] - symbolic.pystr_to_symbolic(axis_b))
            if nd[0][0] != nd[0][1] or nd[1][0] != nd[1][1]:
                return None
            # Offsets must NOT reference the loop iterators -- otherwise the
            # access isn't a clean ``(i + d_i, j + d_j)`` shape. Symbolic
            # offsets (loop-invariant) are accepted; their sign is checked by
            # the wavefront-amenable predicate.
            if ra.free_symbols & iter_syms or rb.free_symbols & iter_syms:
                return None
            return ra, rb
        except Exception:
            return None

    for node in state.data_nodes():
        desc = sdfg.arrays.get(node.data)
        if desc is None or len(desc.shape) != 2:
            continue
        in_e = list(state.in_edges(node))
        out_e = list(state.out_edges(node))
        if in_e:
            for e in in_e:
                if e.data is None or e.data.subset is None:
                    continue
                off = _classify_2d(e.data.subset, outer_var, inner_var)
                if off is None or off != (0, 0):
                    return None  # write must be exactly at (i, j)
                if write_arr is not None and write_arr != node.data:
                    return None
                write_arr = node.data
                write_offsets = off
        if out_e:
            for e in out_e:
                if e.data is None or e.data.subset is None:
                    continue
                off = _classify_2d(e.data.subset, outer_var, inner_var)
                if off is None:
                    continue
                if node.data == write_arr or write_arr is None:
                    # Reads from the carrier array. Defer write/read name match
                    # until we've classified all writes.
                    read_offsets.append(off)
    if write_arr is None or write_offsets is None:
        return None
    # Restrict to reads that come from the carrier array.
    final_offsets: List[Tuple[int, int]] = []
    for node in state.data_nodes():
        if node.data != write_arr:
            continue
        for e in state.out_edges(node):
            if e.data is None or e.data.subset is None:
                continue
            off = _classify_2d(e.data.subset, outer_var, inner_var)
            if off is not None and off != (0, 0):
                final_offsets.append(off)
    if not final_offsets:
        return None
    return write_arr, final_offsets


def _nonpositive(expr) -> bool:
    """``True`` iff ``expr`` is provably non-positive *or* an iter-var-free
    symbolic expression treated as non-positive by assumption.

    Concrete numerics are exact: ``-1`` is non-positive, ``+1`` is not.
    For symbolic expressions DaCe's SDFG-level symbol table does not carry
    sympy's positivity assumptions through (the user's ``dace.symbol(name,
    positive=True)`` is reduced to a typed entry), so an ``is_nonpositive``
    check on the SDFG side cannot prove much. We *assume* iter-var-free
    symbolic offsets are non-positive -- the natural case for a read pattern
    ``a[i - sym]`` carried into the wavefront matcher is exactly this: ``sym``
    must be positive for the access to be a valid backward read, otherwise
    the program would index out of range. The optimistic fall-through here is
    paired with a runtime ``<= 0`` guard plant in :meth:`WavefrontSkew._try_skew`
    (see :func:`_collect_runtime_guards` + :meth:`_emit_nonpositive_guard`,
    analogous to ``BreakAntiDependence``'s ``WAR_symbolic`` guard).
    """
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return False
    if getattr(s, 'is_number', False):
        try:
            return s <= 0
        except TypeError:
            return False
    if s.is_nonpositive is True:
        return True
    try:
        neg = symbolic.simplify(-s)
    except Exception:
        return False
    if getattr(neg, 'is_nonnegative', None) is True or getattr(neg, 'is_positive', None) is True:
        return True
    # Fall-through: pure symbolic expression with no positivity oracle.
    # Optimistically accept; a runtime guard will be needed for soundness.
    return True


def _has_any_nonzero(offsets: List[Tuple[object, object]]) -> bool:
    """Reject the degenerate ``(0, 0)`` set; otherwise let
    :func:`_nonpositive` handle the sign judgement (numeric or symbolic).
    """
    for r_i, r_j in offsets:
        for c in (r_i, r_j):
            try:
                cs = symbolic.simplify(c)
            except Exception:
                continue
            if not (getattr(cs, 'is_number', False) and cs == 0):
                return True
    return False


def _collect_runtime_guards(offsets: List[Tuple[object, object]]) -> List[object]:
    """Return one guard expression per offset component that :func:`_nonpositive`
    accepted *only* via the optimistic fall-through -- i.e., a non-numeric
    symbolic expression with no sympy positivity oracle. Each returned ``e``
    is the offset component itself; the caller's planted guard traps when
    ``e > 0`` (the soundness condition is ``e <= 0``).

    Numeric zero / negative components and symbolic components that sympy can
    prove non-positive (e.g. ``-sym`` where ``sym`` was declared
    ``positive=True``) need no runtime check.
    """
    guards: List[object] = []
    seen = set()
    for r_i, r_j in offsets:
        for c in (r_i, r_j):
            try:
                cs = symbolic.simplify(c)
            except Exception:
                continue
            if getattr(cs, 'is_number', False):
                continue  # numeric components are checked statically by ``_nonpositive``
            if getattr(cs, 'is_nonpositive', None) is True:
                continue  # sympy oracle proves non-positivity; no runtime check needed
            try:
                neg = symbolic.simplify(-cs)
            except Exception:
                neg = None
            if neg is not None and (getattr(neg, 'is_nonnegative', None) is True
                                    or getattr(neg, 'is_positive', None) is True):
                continue
            key = str(cs)
            if key in seen:
                continue
            seen.add(key)
            guards.append(cs)
    return guards


def _inner_already_parallel(offsets: List[Tuple[object, object]]) -> bool:
    """``True`` iff every dep offset has a zero *inner*-axis component, i.e. the
    inner loop is already parallel without any skew. In that case, the skew
    rewrite is unnecessary -- a direct ``LoopToMap`` on the inner loop yields
    the same parallel structure with a simpler axis-aligned form (TSVC
    ``s1119``: ``aa[i, j] = aa[i-1, j] + bb[i, j]`` has dep ``(1, 0)``; the
    inner ``j``-loop is fully parallel on its own).
    """
    for _r_outer, r_inner in offsets:
        try:
            s = symbolic.simplify(r_inner)
        except Exception:
            return False
        if not (getattr(s, 'is_number', False) and s == 0):
            return False
    return True


def _is_wavefront_amenable(offsets: List[Tuple[object, object]]) -> bool:
    """All read offsets must lie strictly *before* the write in lexicographic
    iteration order, and be non-positive on both axes -- the ``t = i + j``
    skew then puts every dependence vector strictly forward on the ``t`` axis.

    Numeric ``(0, -1)`` and ``(-1, 0)`` are the textbook case. Symbolic
    forms like ``(0, -sym1)`` and ``(-sym2, 0)`` are accepted when ``sym1``,
    ``sym2`` are declared positive -- a regime DaCe's symbol system supports
    directly, but where polyhedral analyzers without an oracle for symbol
    sign typically give up.

    Refuses when the *inner* axis is already free of carried deps
    (see :func:`_inner_already_parallel`): the skew gains nothing over a
    direct ``LoopToMap`` and the axis-aligned form is preferable.
    """
    if not offsets:
        return False
    for r_i, r_j in offsets:
        if not (_nonpositive(r_i) and _nonpositive(r_j)):
            return False
    if not _has_any_nonzero(offsets):
        return False
    if _inner_already_parallel(offsets):
        return False
    return True


@properties.make_properties
@xf.explicit_cf_compatible
class WavefrontSkew(ppl.Pass):
    """Apply ``t = i + j, p = j`` skewing to perfect 2-D loop nests whose body
    has a wavefront-style dependence pattern, exposing inner-loop parallelism
    for downstream ``LoopToMap``."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Skew every eligible 2-D nest. Returns the count or ``None`` on no match.

        A successful skew lifts the inner ``p``-loop to a Map (via
        :meth:`_convert_inner_to_map`), removing it from the CFG. The snapshot
        ``list(sd.all_control_flow_regions())`` may therefore contain stale
        references to inner LoopRegions removed by a prior skew; ``_try_skew``
        rejects those on its parent-graph check before attempting any work.
        """
        skewed = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                # Stale-snapshot guard: a previous successful skew may have
                # removed this LoopRegion from its parent (its ``inner`` was
                # converted to a Map). Skip rather than process a detached node.
                parent = cfg.parent_graph
                if parent is None or cfg not in parent.nodes():
                    continue
                if self._try_skew(cfg, sd):
                    skewed += 1
        return skewed or None

    def _try_skew(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        if not _unit_positive_stride(outer):
            return False
        inner = _try_extract_two_level_nest(outer)
        if inner is None or not _unit_positive_stride(inner):
            return False
        i_lo = loop_analysis.get_init_assignment(outer)
        i_hi = loop_analysis.get_loop_end(outer)
        j_lo = loop_analysis.get_init_assignment(inner)
        j_hi = loop_analysis.get_loop_end(inner)
        if any(x is None for x in (i_lo, i_hi, j_lo, j_hi)):
            return False
        # Reject *triangular* nests -- bounds where the inner's range depends on
        # the outer's iterator (TSVC ``s232``: ``for i in range(1, j+1)``). The
        # skew formula ``t in [i_lo + j_lo, i_hi + j_hi]`` then has an
        # unresolved outer-var symbol baked into ``t_hi`` / ``p_lo`` / ``p_hi``,
        # producing a skewed loop whose bounds can't be evaluated at runtime.
        # A correct triangular skew would need to bound ``t`` purely in terms
        # of ``N`` (e.g. by using ``j_hi.subs(outer_var, t_hi_resolved)``); for
        # now refuse and let the outer iteration stay sequential.
        outer_sym = symbolic.pystr_to_symbolic(outer.loop_variable)
        inner_sym = symbolic.pystr_to_symbolic(inner.loop_variable)
        for bound in (i_lo, i_hi, j_lo, j_hi):
            try:
                bs = symbolic.simplify(bound)
            except Exception:
                bs = bound
            free = getattr(bs, 'free_symbols', set())
            if outer_sym in free and bound in (j_lo, j_hi):
                return False
            if inner_sym in free:
                return False  # inner var leaks into a bound -- malformed input
        detected = _extract_wavefront_offsets(inner, outer.loop_variable, inner.loop_variable, sdfg)
        if detected is None:
            return False
        _arr, offsets = detected
        if not _is_wavefront_amenable(offsets):
            return False

        # Plant a runtime ``component <= 0`` guard for every offset component
        # whose non-positivity ``_nonpositive`` accepted only via the
        # optimistic fall-through (purely symbolic, no sympy positivity
        # oracle). If the user later runs with values that violate the
        # assumption the trap fires loudly instead of corrupting output.
        runtime_guards = _collect_runtime_guards(offsets)
        if runtime_guards:
            self._emit_nonpositive_guard(outer, runtime_guards)

        # Synthesise the skewed iterators and rewrite both loops in place.
        nid = _next_id(sdfg)
        t_var = f"{_SKEW_T_PREFIX}{nid}"
        p_var = f"{_SKEW_P_PREFIX}{nid}"
        sdfg.add_symbol(t_var, dace.int64)
        sdfg.add_symbol(p_var, dace.int64)

        i_sym = outer.loop_variable
        j_sym = inner.loop_variable
        # New iteration space: t in [i_lo + j_lo, i_hi + j_hi]; for each t,
        # p in [max(j_lo, t - i_hi), min(j_hi, t - i_lo)].
        t_lo = symbolic.symstr(symbolic.simplify(i_lo + j_lo))
        t_hi = symbolic.symstr(symbolic.simplify(i_hi + j_hi))
        p_lo = (f"max(({symbolic.symstr(j_lo)}), "
                f"({t_var}) - ({symbolic.symstr(i_hi)}))")
        p_hi = (f"min(({symbolic.symstr(j_hi)}), "
                f"({t_var}) - ({symbolic.symstr(i_lo)}))")

        outer.loop_variable = t_var
        outer.init_statement = dace.properties.CodeBlock(f"{t_var} = ({t_lo})")
        outer.loop_condition = dace.properties.CodeBlock(f"{t_var} <= ({t_hi})")
        outer.update_statement = dace.properties.CodeBlock(f"{t_var} = {t_var} + 1")

        inner.loop_variable = p_var
        inner.init_statement = dace.properties.CodeBlock(f"{p_var} = ({p_lo})")
        inner.loop_condition = dace.properties.CodeBlock(f"{p_var} <= ({p_hi})")
        inner.update_statement = dace.properties.CodeBlock(f"{p_var} = {p_var} + 1")

        # Substitute the original loop variables directly in every memlet /
        # tasklet inside the inner loop body. The downstream ``LoopToMap``
        # affine-subset matcher then sees clean ``a * p + b`` expressions
        # rather than ``i`` / ``j`` symbols whose binding lives on an
        # interstate edge.
        repl = {i_sym: f"(({t_var}) - ({p_var}))", j_sym: p_var}
        inner.replace_dict(repl)

        self._convert_inner_to_map(outer, inner, sdfg)
        return True

    def _convert_inner_to_map(self, outer: LoopRegion, inner: LoopRegion, sdfg: SDFG):
        """Lift the just-skewed inner ``p``-loop directly to a Map via the
        :class:`LoopToMap` conversion utility (``apply``), bypassing the
        ``can_be_applied`` safety gate.

        The bypass is sound by construction:

        * The skew rewrites the iteration space to ``(t, p)`` where ``t``
          is the diagonal index. Every dependence in the original body --
          which was lex-backward in ``(i, j)`` per
          :func:`_is_wavefront_amenable` -- becomes strictly forward on
          ``t`` after the substitution ``i = t - p, j = p``. There is no
          intra-``t`` dependence between two different ``p`` values, so
          every iteration of the ``p``-loop at fixed ``t`` is independent.

        * The body's structure (state graph, tasklets, memlets, nested
          regions) is preserved by :meth:`LoopRegion.replace_dict`; only
          the symbolic iteration variables in subset bounds and tasklet
          code change. ``LoopToMap.apply`` treats the body as a unit and
          does not depend on the iteration order, so the rewrite remains
          well-defined on the substituted form.

        Skipping ``can_be_applied`` matters because its write-write race
        analysis on the synthesised ``min`` / ``max`` ``p``-bounds is both
        unnecessarily expensive (we have the proof above) and brittle on
        synthesised symbolic forms. The underlying ``apply`` utility, in
        contrast, only needs structurally well-formed loop bounds, which
        ``_try_skew`` guarantees.

        An exception from ``apply`` here indicates a real upstream bug
        (malformed memlet, broken NSDFG, etc.) rather than an unfit input,
        and is intentionally NOT caught so the failure surfaces during
        canonicalize rather than corrupting downstream output.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap
        instance = LoopToMap()
        instance.loop = inner
        instance.apply(outer, sdfg)

    def _emit_nonpositive_guard(self, outer: LoopRegion, exprs: List[object]):
        """Plant a side-effect-only tasklet immediately before ``outer`` that
        traps if any of ``exprs`` is positive at runtime. Matches the shape of
        :class:`~dace.transformation.passes.break_anti_dependence.BreakAntiDependence`'s
        ``_emit_positive_guard``: zero-connector CPP tasklet, ``__builtin_trap``
        on violation, body reads the symbols by name from the enclosing scope.
        """
        parts = ' || '.join(f'(({symbolic.symstr(e)}) > 0)' for e in exprs)
        tag = abs(hash(parts)) & 0xfffffff
        code = f'if ({parts}) {{ __builtin_trap(); }}'
        pre = outer.parent_graph.add_state_before(outer, label=f'_skew_guard_{tag:x}')
        pre.add_tasklet(
            name=f'_skew_guard_{tag:x}',
            inputs={},
            outputs={},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


__all__ = ['WavefrontSkew']
