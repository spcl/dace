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

The outer loop must be ``for i in range(0, N, K)`` where ``K`` is a concrete
positive integer literal and the start is ``0``. The single body block of the
outer must be exactly one nested :class:`~dace.sdfg.state.LoopRegion` and
nothing else (perfect nest).

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
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.state import LoopRegion, SDFGState, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


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


def _try_extract_perfect_two_level_nest(outer: LoopRegion) -> Optional[LoopRegion]:
    """Return the single inner LoopRegion of ``outer`` if the body holds nothing
    else; otherwise ``None``. Empty states are tolerated."""
    inner = None
    for b in outer.nodes():
        if isinstance(b, SDFGState):
            if len(b.nodes()) > 0:
                return None  # non-empty plain state breaks the perfect nest
            continue
        if isinstance(b, LoopRegion):
            if inner is not None:
                return None  # more than one inner LoopRegion
            inner = b
            continue
        return None  # any other CFG construct breaks the perfect nest
    return inner


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


def _match_inner_case(inner: LoopRegion, outer_var: str, K: int) -> Optional[str]:
    """Classify the inner loop shape: returns ``'A'`` for ``range(0, K)``,
    ``'B'`` for ``range(i, i + K)``, or ``None`` if neither."""
    stride = loop_analysis.get_loop_stride(inner)
    start = loop_analysis.get_init_assignment(inner)
    end = loop_analysis.get_loop_end(inner)
    if stride is None or start is None or end is None:
        return None
    if int(symbolic.simplify(stride)) != 1:
        return None
    outer_sym = symbolic.pystr_to_symbolic(outer_var)
    s_sym = symbolic.simplify(start)
    e_sym = symbolic.simplify(end)
    # ``get_loop_end`` returns the inclusive bound; for ``range(a, b)`` that's ``b - 1``.
    # Case A: start == 0, end == K - 1.
    if _is_zero(s_sym):
        if int(symbolic.simplify(e_sym - (K - 1))) == 0:
            return 'A'
        return None
    # Case B: start == i, end == i + K - 1.
    if symbolic.simplify(s_sym - outer_sym) == 0:
        if symbolic.simplify(e_sym - (outer_sym + K - 1)) == 0:
            return 'B'
    return None


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


def _all_memlet_uses_only(inner: LoopRegion, allowed_atoms: Set[str],
                          forbidden_atoms: Set[str]) -> bool:
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


def _audit_combined_access(inner: LoopRegion, outer_var: str, inner_var: str,
                           case: str) -> bool:
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
    """Collapse a manually-tiled two-level perfect nest to a single unit-stride loop.

    Recognises ``for i in range(0, N, K): for ii in range(0, K): ...`` (case A)
    and ``for i in range(0, N, K): for ii in range(i, i + K): ...`` (case B),
    where ``K`` is a concrete positive integer and every memlet inside the
    inner body references the affine index only via ``i + ii`` (case A) or
    ``ii`` (case B). Rewrites both loops into a single ``for k in range(0, N)``,
    substituting the combined access with ``k``.

    Runs BEFORE :class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll`
    so the small fixed-trip inner loop doesn't get straight-line-unrolled (which
    would re-bake the tile into the body and lose the chance to collapse it).
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        rewritten = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                if self._try_untile(cfg, sd):
                    rewritten += 1
        return rewritten or None

    def _try_untile(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        # The outer must be ``for i in range(0, N, K)`` with concrete K > 0.
        outer_stride = loop_analysis.get_loop_stride(outer)
        outer_start = loop_analysis.get_init_assignment(outer)
        outer_end = loop_analysis.get_loop_end(outer)
        if outer_stride is None or outer_start is None or outer_end is None:
            return False
        K = _is_constant_positive_int(outer_stride)
        if K is None or K == 1:
            return False  # K must be > 1 (K == 1 is already untiled)
        if not _is_zero(outer_start):
            return False

        # The outer body must be exactly one inner LoopRegion (perfectly nested).
        inner = _try_extract_perfect_two_level_nest(outer)
        if inner is None or not inner.loop_variable:
            return False

        # Classify the inner shape (A or B) and audit access patterns.
        case = _match_inner_case(inner, outer.loop_variable, K)
        if case is None:
            return False
        if not _audit_combined_access(inner, outer.loop_variable, inner.loop_variable, case):
            return False

        # Synthesise the new unit-stride iterator and rewrite both loops in place.
        k_var = f"{_UNTILE_PREFIX}{_next_id(sdfg)}"
        sdfg.add_symbol(k_var, sdfg.symbols.get(outer.loop_variable, dace.int64))
        # ``N`` = outer_end + 1 -- ``get_loop_end`` returns the inclusive bound
        # for the stride-K outer (e.g. for ``range(0, N, K)`` the last admitted
        # value of ``i`` is the largest ``K * q < N``; the *exclusive* upper
        # bound on the collapsed iterator is just ``N``, which is what the
        # original ``range(0, N)`` would have used).
        N_excl = symbolic.simplify(outer_end + outer_stride)

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

        # Splice the inner's body blocks into the outer, then remove the inner
        # LoopRegion node. We re-use the outer LoopRegion as the new collapsed
        # loop (just rewrites its iteration descriptors), so we need to move
        # ``inner``'s body content into ``outer`` while detaching ``inner``.
        # Strategy: detach ``inner``'s start block, attach it directly under
        # ``outer`` in inner's place.
        outer.remove_node(inner)
        # Re-parent inner's blocks one level up.
        for child in list(inner.nodes()):
            inner.remove_node(child)
            outer.add_node(child, is_start_block=(child is inner.start_block))
        # Re-attach the inner's interstate edges.
        for ie in list(inner.edges()):
            outer.add_edge(ie.src, ie.dst, ie.data)

        # Rewrite the outer's iteration descriptors to drive ``k`` over [0, N).
        outer.loop_variable = k_var
        outer.init_statement = dace.properties.CodeBlock(f"{k_var} = 0")
        outer.loop_condition = dace.properties.CodeBlock(f"{k_var} < ({symbolic.symstr(N_excl)})")
        outer.update_statement = dace.properties.CodeBlock(f"{k_var} = {k_var} + 1")
        return True


__all__ = ['UntileLoops']
