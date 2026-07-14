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

from dace import symbolic
from dace.sdfg.analysis.polyhedral_isl import (HAVE_ISL, isl, classify_dim, collect_basic_sets, constraint_to_sympy,
                                               dedupe_terms, is_domain_empty, make_set, pwaff_bound, subs_by_name)


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


def skew_bounds(dims: Tuple[str, str], params: Sequence[str], domain_constraints, tau: Tuple[int, int],
                t_name: str, p_name: str) -> Optional[SkewBounds]:
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
    (possibly int-division) loop bound."""
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
