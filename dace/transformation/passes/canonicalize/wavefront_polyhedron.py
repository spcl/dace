# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ISL-backed polyhedral engine for :class:`WavefrontSkew`.

Two operations power the general wavefront transform:

* **Schedule legality** -- given the dependence distance vectors of a 2-D loop
  nest (each possibly *parametric* in the iterators of an enclosing reduction
  loop) and a candidate skew ``tau = (a, b)``, decide whether ``tau`` makes
  every dependence strictly forward on the sequential axis
  (``tau . delta < 0`` for all deltas over the whole iteration domain). This is
  the classical Lamport / Feautrier legality test; here it is an exact integer
  emptiness query on ``{ domain and delta-range and tau.delta >= 0 }``. It is
  decided by :func:`~dace.sdfg.analysis.polyhedral_isl.is_domain_empty`.

* **Skewed loop bounds** -- apply the unimodular map ``(u, v) -> (t = a*u+b*v,
  p = v)`` to the (possibly *triangular*) iteration domain and read back the new
  loop bounds: the ``t`` range (project ``p`` out) and the parametric ``p``
  range at fixed ``t``. Fourier-Motzkin over the reals loses integer tightness
  and keeps redundant constraints; ISL coalesces to a minimal exact form, which
  is exactly the ``max(...) <= p <= min(...)`` shape a LoopRegion bound wants.
  This is :func:`skew_bounds`.

The general, SDFG-type-free ISL layer these two build on -- quasi-affine
expression <-> ISL rendering, integer set construction, emptiness queries, and
constraint extraction -- lives in
:mod:`dace.sdfg.analysis.polyhedral_isl` so any pass can reuse it. This module
re-exports the names :class:`WavefrontSkew` reaches through the ``poly`` alias.
``islpy`` is an optional dependency: when it is absent :data:`HAVE_ISL` is
``False`` and the pass degrades to a no-op (loops stay sequential; the
``pinned_sequential`` safety net preserves the never-slower-than-auto_optimize
guarantee).
"""
from typing import List, Optional, Sequence, Tuple

import sympy as sp

from dace import symbolic
# noqa: F401 -- re-exports reached as poly.<name> from wavefront_skew; ruff cannot see that use.
from dace.sdfg.analysis.polyhedral_isl import HAVE_ISL, is_domain_empty  # noqa: F401
from dace.sdfg.analysis.polyhedral_isl import (isl, classify_dim, collect_basic_sets, constraint_to_sympy, dedupe_terms,
                                               make_set, pwaff_bound, subs_by_name)


def constraints_from_condition(cond) -> Optional[List]:
    """``cond`` rendered as a list of expressions each meant ``>= 0``, or ``None``.

    A read that only executes under a branch guard carries a dependence only where that guard
    holds, so the guard belongs in the dependence's polyhedron exactly like the loop bounds do.
    Adding it can only SHRINK the region a schedule must order, so a caller that drops a guard
    this refuses stays conservative.

    Strict relations are tightened by one (``a < b`` becomes ``b - a - 1 >= 0``), which is exact
    on the integer index space and would be wrong on a rational one. ``Or`` has no conjunctive
    form and ``Ne`` no convex one, so both are refused rather than approximated.
    """
    if cond is None or cond is sp.true or cond is True:
        return []
    if cond is sp.false or cond is False:
        return None  # an unsatisfiable guard is a caller bug, not a constraint to render
    # ``AND``/``OR`` also arrive as DaCe's own function nodes: ``pystr_to_symbolic`` builds a
    # parsed condition with ``evaluate=False`` and keeps them verbatim rather than folding to
    # sympy's connectives, so matching only ``sp.And`` would miss every parsed guard.
    func = str(cond.func) if isinstance(cond, sp.Basic) else ''
    if isinstance(cond, sp.And) or func == 'AND':
        out: List = []
        for arg in cond.args:
            part = constraints_from_condition(arg)
            if part is None:
                return None
            out += part
        return out
    if func == 'NOT':
        return constraints_from_condition(symbolic.refold_booleans(sp.Not(cond.args[0])))
    if isinstance(cond, sp.StrictLessThan):  # a < b
        return [symbolic.simplify(cond.rhs - cond.lhs - 1)]
    if isinstance(cond, sp.LessThan):  # a <= b
        return [symbolic.simplify(cond.rhs - cond.lhs)]
    if isinstance(cond, sp.StrictGreaterThan):  # a > b
        return [symbolic.simplify(cond.lhs - cond.rhs - 1)]
    if isinstance(cond, sp.GreaterThan):  # a >= b
        return [symbolic.simplify(cond.lhs - cond.rhs)]
    if isinstance(cond, sp.Equality):  # a == b, as the two inequalities
        return [symbolic.simplify(cond.lhs - cond.rhs), symbolic.simplify(cond.rhs - cond.lhs)]
    return None


class SkewBounds:
    """The bound *terms* extracted for a valid skew: ``t`` in
    ``[max(t_lo_terms), min(t_hi_terms)]`` and, at fixed ``t``, ``p`` in
    ``[max(p_lo_terms), min(p_hi_terms)]``. The pass renders these to loop
    bounds."""

    def __init__(self, t_lo_terms: List, t_hi_terms: List, p_lo_terms: List, p_hi_terms: List):
        self.t_lo_terms = t_lo_terms
        self.t_hi_terms = t_hi_terms
        self.p_lo_terms = p_lo_terms
        self.p_hi_terms = p_hi_terms


def skew_bounds(dims: Tuple[str, str],
                params: Sequence[str],
                domain_constraints,
                tau: Tuple[int, int],
                t_name: str,
                p_name: str,
                t_range: Optional[Tuple[object, object]] = None) -> Optional[SkewBounds]:
    """Project the domain through the unimodular skew ``t = a*u + b*v`` and read
    back bound terms. ``dims`` are ``(u, v)``; ``tau = (a, b)``. The parallel axis
    ``p`` is the coordinate whose complement inverts over the integers:

    * ``|a| == 1``: ``p = v``, ``u = a*(t - b*p)`` -- the shallow ``(1, +-1)`` /
      ``(1, +-2)`` family.
    * ``|b| == 1``: ``p = u``, ``v = b*(t - a*p)`` -- the steep ``(2, +-1)``
      Gauss-Seidel family, where the diagonal is steeper than 45 degrees.

    When both hold either works; when neither does (``(2, 3)``) there is no
    single-coordinate unimodular complement and the skew is refused. Returns a
    :class:`SkewBounds` or ``None`` if a bound is not expressible as a simple
    (possibly int-division) loop bound.

    ``t_range`` supplies the diagonal range instead of projecting for it, and may be any SUPERSET
    of the true one. The projection is exact but not always renderable -- a tiled interior's
    diagonal extent is genuinely piecewise (interior tile / last row / last column / corner), which
    is four affine pieces and no single loop bound. Handing in the enclosing rectangle's diagonal
    range costs nothing, because the ``p`` range at fixed ``t`` is read exactly: a diagonal outside
    the real region comes out EMPTY and runs no iterations."""
    u, v = dims
    a, b = tau
    tsym = symbolic.pystr_to_symbolic(t_name)
    psym = symbolic.pystr_to_symbolic(p_name)
    if abs(a) == 1:
        subs = {u: a * (tsym - b * psym), v: psym}  # a in {1,-1} => 1/a == a
    elif abs(b) == 1:
        subs = {u: psym, v: b * (tsym - a * psym)}  # b in {1,-1} => 1/b == b
    else:
        return None
    skewed = [subs_by_name(c, subs) for c in domain_constraints]
    sdims = (t_name, p_name)

    s_set, nmap = make_set(sdims, params, skewed)
    s_set = s_set.coalesce()
    inv = {safe: orig for orig, safe in nmap.items()}
    safe_dims = [nmap[t_name], nmap[p_name]]
    safe_params = [nmap[p] for p in params]

    # p-range at fixed t (parametric in t): read directly from the skewed set. A
    # steep skew scales p by |a| > 1, which ``classify_dim`` turns into an exact
    # int_ceil / int_floor bound.
    p_lo_terms: List = []
    p_hi_terms: List = []
    for b_set in collect_basic_sets(s_set):
        for c in b_set.get_constraints():
            e = constraint_to_sympy(c, safe_dims, safe_params, inv)
            lo, hi, ok = classify_dim(e, psym)
            if not ok:
                return None
            p_lo_terms += lo
            p_hi_terms += hi
            if c.is_equality():
                lo2, hi2, ok2 = classify_dim(symbolic.simplify(-e), psym)
                if not ok2:
                    return None
                p_lo_terms += lo2
                p_hi_terms += hi2

    # t-range: project the parallel axis out and take the exact integer min / max
    # of the remaining diagonal dim. Per-constraint reading is unsound here --
    # projecting p out of a slanted (non-unit-scaled) domain leaves an ISL
    # existential (a divisibility on t) that ``classify_dim`` cannot read -- so
    # ``dim_min`` / ``dim_max`` resolve the integer shadow exactly.
    if t_range is not None:
        t_lo_terms, t_hi_terms = [t_range[0]], [t_range[1]]
    else:
        t_set = s_set.project_out(isl.dim_type.set, 1, 1).coalesce()
        t_lo = pwaff_bound(t_set.dim_min(0), inv)
        t_hi = pwaff_bound(t_set.dim_max(0), inv)
        if t_lo is None or t_hi is None:
            return None
        t_lo_terms = [t_lo]
        t_hi_terms = [t_hi]

    p_lo_terms = dedupe_terms(p_lo_terms)
    p_hi_terms = dedupe_terms(p_hi_terms)
    if not (t_lo_terms and t_hi_terms and p_lo_terms and p_hi_terms):
        return None
    return SkewBounds(t_lo_terms, t_hi_terms, p_lo_terms, p_hi_terms)
