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
                                               dedupe_terms, is_domain_empty, make_set, subs_by_name)


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
    """Project the domain through ``(t = a*u + b*v, p = v)`` and read back bound
    terms. ``dims`` are ``(u, v)``; ``tau = (a, b)`` with ``a in {1, -1}`` so the
    map is unimodular (``p = v``). Returns a :class:`SkewBounds` or ``None`` if a
    bound is non-unit in ``p`` (an integer-division bound ISL expresses with an
    existential -- outside the simple loop-bound form we splice back)."""
    u, v = dims
    a, b = tau
    tsym = symbolic.pystr_to_symbolic(t_name)
    psym = symbolic.pystr_to_symbolic(p_name)
    # Skewed set S(t, p): substitute u = a*(t - b*p), v = p  (a in {1,-1} => a == 1/a).
    usub = a * (tsym - b * psym)
    skewed = [subs_by_name(c, {u: usub, v: psym}) for c in domain_constraints]
    sdims = (t_name, p_name)

    s_set, nmap = make_set(sdims, params, skewed)
    s_set = s_set.coalesce()
    inv = {safe: orig for orig, safe in nmap.items()}
    safe_dims = [nmap[t_name], nmap[p_name]]
    safe_params = [nmap[p] for p in params]

    # p-range at fixed t.
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

    # t-range: project p out.
    t_set = s_set.project_out(isl.dim_type.set, 1, 1).coalesce()
    t_lo_terms: List = []
    t_hi_terms: List = []
    for b_set in collect_basic_sets(t_set):
        for c in b_set.get_constraints():
            e = constraint_to_sympy(c, [nmap[t_name]], safe_params, inv)
            lo, hi, ok = classify_dim(e, tsym)
            if not ok:
                return None
            t_lo_terms += lo
            t_hi_terms += hi
            if c.is_equality():
                lo2, hi2, ok2 = classify_dim(symbolic.simplify(-e), tsym)
                if not ok2:
                    return None
                t_lo_terms += lo2
                t_hi_terms += hi2

    t_lo_terms = dedupe_terms(t_lo_terms)
    t_hi_terms = dedupe_terms(t_hi_terms)
    p_lo_terms = dedupe_terms(p_lo_terms)
    p_hi_terms = dedupe_terms(p_hi_terms)
    if not (t_lo_terms and t_hi_terms and p_lo_terms and p_hi_terms):
        return None
    return SkewBounds(t_lo_terms, t_hi_terms, p_lo_terms, p_hi_terms)
