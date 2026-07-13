# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ISL-backed polyhedral engine for :class:`WavefrontSkew`.

Two operations power the general wavefront transform:

* **Schedule legality** -- given the dependence distance vectors of a 2-D loop
  nest (each possibly *parametric* in the iterators of an enclosing reduction
  loop) and a candidate skew ``tau = (a, b)``, decide whether ``tau`` makes
  every dependence strictly forward on the sequential axis
  (``tau . delta < 0`` for all deltas over the whole iteration domain). This is
  the classical Lamport / Feautrier legality test; here it is an exact integer
  emptiness query on ``{ domain and delta-range and tau.delta >= 0 }``.

* **Skewed loop bounds** -- apply the unimodular map ``(u, v) -> (t = a*u+b*v,
  p = v)`` to the (possibly *triangular*) iteration domain and read back the new
  loop bounds: the ``t`` range (project ``p`` out) and the parametric ``p``
  range at fixed ``t``. Fourier-Motzkin over the reals loses integer tightness
  and keeps redundant constraints; ISL coalesces to a minimal exact form, which
  is exactly the ``max(...) <= p <= min(...)`` shape a LoopRegion bound wants.

The module is deliberately free of DaCe SDFG types -- inputs are symbolic
expressions (built through :mod:`dace.symbolic`) plus variable-name strings,
outputs are symbolic expressions -- so the polyhedral core can be unit-tested
without building an SDFG. ``islpy`` is an optional dependency: when it is absent
:data:`HAVE_ISL` is ``False`` and the pass degrades to a no-op (loops stay
sequential; the ``pinned_sequential`` safety net preserves the
never-slower-than-auto_optimize guarantee).
"""
from typing import Dict, List, Optional, Sequence, Tuple

import sympy

from dace import symbolic
from dace.symbolic import int_ceil, int_floor

try:
    import islpy as isl
    HAVE_ISL = True
except ImportError:  # pragma: no cover - environment without islpy
    isl = None
    HAVE_ISL = False


def safe_name(prefix: str, index: int) -> str:
    """A canonical ISL-safe identifier. DaCe symbols (``_loop_it_0``) and offset
    symbols carry leading underscores / arbitrary characters we do not want to
    feed into ISL's parser verbatim; every symbol is remapped to ``<prefix><i>``
    and mapped back after the query."""
    return f"{prefix}{index}"


def build_name_map(dims: Sequence[str], params: Sequence[str]) -> Dict[str, str]:
    """Bijection ``original -> ISL-safe`` covering the iteration dims and params."""
    mp: Dict[str, str] = {}
    for i, d in enumerate(dims):
        mp[d] = safe_name('d', i)
    for i, p in enumerate(params):
        mp[p] = safe_name('P', i)
    return mp


def subs_by_name(expr, mapping):
    """``expr.subs`` matching the expression's *actual* free symbols by name, so
    it is immune to symbol assumption mismatches (two ``symbol('u')`` with
    different assumptions are unequal under a plain ``subs`` dict)."""
    e = symbolic.simplify(expr)
    sub = {}
    for s in e.free_symbols:
        nm = getattr(s, 'name', None)
        if nm in mapping:
            sub[s] = mapping[nm]
    return symbolic.simplify(e.subs(sub))


def isl_divisor(b) -> str:
    """Render a ``floor`` / ``ceil`` / ``mod`` divisor. ISL only admits integer
    division, so the divisor must be a positive integer constant; a symbolic or
    non-positive divisor is refused (``ValueError``)."""
    if b.is_Integer and int(b) > 0:
        return str(int(b))
    raise ValueError(f"non-constant / non-positive divisor {b}")


def to_isl(e) -> str:
    """Render a quasi-affine expression -- affine plus integer ``floor`` / ``ceil``
    / ``mod`` -- over already-ISL-safe symbol names into an ISL constraint string.

    ``int_floor`` / ``int_ceil`` (DaCe's integer-division functions) and ``Mod`` map
    to ISL's native ``floor(x/d)`` / ``ceil(x/d)`` / ``(x mod d)``, which are exact in
    Presburger arithmetic -- so ISL reasons about tiled / strided domains (a tile
    bound like ``int_floor(N, 8)``) directly rather than the query being rejected.

    Raises ``ValueError`` on anything ISL cannot represent exactly -- a nonlinear
    term (variable*variable, a power), a non-integer coefficient, or a symbolic
    div/mod divisor -- so the caller refuses the query rather than emitting an
    unsound (or unparseable) one."""
    if e.is_Integer:
        return str(int(e))
    if e.is_Symbol:
        return e.name
    if isinstance(e, int_floor):
        return f"floor(({to_isl(e.args[0])})/{isl_divisor(e.args[1])})"
    if isinstance(e, int_ceil):
        return f"ceil(({to_isl(e.args[0])})/{isl_divisor(e.args[1])})"
    if isinstance(e, sympy.Mod):
        return f"(({to_isl(e.args[0])}) mod {isl_divisor(e.args[1])})"
    if e.is_Add:
        return "(" + " + ".join(to_isl(t) for t in e.args) + ")"
    if e.is_Mul:
        coeff, rest = e.as_coeff_Mul()
        if not coeff.is_Integer:
            raise ValueError(f"non-integer coefficient {coeff} in {e}")
        if rest == sympy.Integer(1):
            return str(int(coeff))
        # ``rest`` is the product of the non-constant factors. Affine => it is a single
        # atomic factor (symbol / floor / ceil / mod) or a parenthesised sum to
        # distribute over; a residual product (var*var) or a power is nonlinear.
        if not (rest.is_Symbol or rest.is_Add or isinstance(rest, (int_floor, int_ceil, sympy.Mod))):
            raise ValueError(f"nonlinear term {rest} in {e}")
        inner = to_isl(rest)
        c = int(coeff)
        if c == 1:
            return inner
        if c == -1:
            return f"-({inner})"
        return f"{c}*({inner})"
    raise ValueError(f"non-affine / unsupported term {e}")


def render_affine(expr, name_map: Dict[str, str]) -> str:
    """Render a quasi-affine expression into an ISL constraint string under
    ``name_map`` (see :func:`to_isl`). Raises ``ValueError`` on a form ISL cannot
    represent exactly, which the caller treats as "cannot reason, refuse" rather
    than risking an unsound query."""
    e = subs_by_name(expr, {orig: symbolic.pystr_to_symbolic(safe) for orig, safe in name_map.items()})
    safe_syms = {symbolic.pystr_to_symbolic(s) for s in name_map.values()}
    if not e.free_symbols <= safe_syms:
        raise ValueError(f"unmapped symbol in {expr} (free={e.free_symbols - safe_syms})")
    return to_isl(e)


def constraints_str(constraints, name_map: Dict[str, str]) -> str:
    """`` and ``-joined ISL constraint body from exprs each meaning ``>= 0``."""
    parts = [f"({render_affine(c, name_map)}) >= 0" for c in constraints]
    return ' and '.join(parts) if parts else 'true'


def make_set(dims: Sequence[str], params: Sequence[str], constraints):
    """Build an ``isl.Set`` over ``dims`` parametrised by ``params`` from a list
    of exprs (each ``>= 0``). Returns ``(set, name_map)``."""
    name_map = build_name_map(dims, params)
    pstr = ', '.join(name_map[p] for p in params)
    dstr = ', '.join(name_map[d] for d in dims)
    body = constraints_str(constraints, name_map)
    text = f"[{pstr}] -> {{ [{dstr}] : {body} }}"
    return isl.Set(text), name_map


def is_domain_empty(dims: Sequence[str], params: Sequence[str], constraints) -> bool:
    """Exact integer emptiness of ``{ constraints }``. Raises on a bad render."""
    s, _ = make_set(dims, params, constraints)
    return s.is_empty()


def value_to_int(v) -> int:
    return int(v.to_python()) if hasattr(v, 'to_python') else int(str(v))


def constraint_to_sympy(c, safe_dims: Sequence[str], safe_params: Sequence[str], inv_map: Dict[str, str]):
    """Reconstruct the affine expr (meaning ``>= 0``) of an ``isl.Constraint``,
    mapping ISL-safe names back to originals via ``inv_map``."""
    expr = symbolic.pystr_to_symbolic(value_to_int(c.get_constant_val()))
    for i, nm in enumerate(safe_dims):
        co = value_to_int(c.get_coefficient_val(isl.dim_type.set, i))
        if co:
            expr = expr + co * symbolic.pystr_to_symbolic(inv_map[nm])
    for i, nm in enumerate(safe_params):
        co = value_to_int(c.get_coefficient_val(isl.dim_type.param, i))
        if co:
            expr = expr + co * symbolic.pystr_to_symbolic(inv_map[nm])
    return symbolic.simplify(expr)


def collect_basic_sets(s) -> List:
    """``isl.Set`` -> list of basic sets. (ISL callbacks abort the process on any
    Python exception, so the callback only appends -- never computes.)"""
    out: List = []
    s.foreach_basic_set(lambda b: out.append(b))
    return out


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


def classify_dim(e, dsym) -> Tuple[List, List, bool]:
    """Split an ``e >= 0`` constraint into lower/upper bound terms for ``dsym``.

    coeff == 0  -> not a bound on this dim (ignore).
    coeff == 1  -> dsym >= -rest.
    coeff == -1 -> dsym <= rest.
    |coeff| > 1 -> integer-division bound; unsupported (ok=False).
    """
    coeff = e.coeff(dsym, 1)
    rest = symbolic.simplify(e - coeff * dsym)
    if coeff == 0:
        return [], [], True
    if coeff == 1:
        return [symbolic.simplify(-rest)], [], True
    if coeff == -1:
        return [], [symbolic.simplify(rest)], True
    return [], [], False


def dedupe_terms(terms: List) -> List:
    """Drop syntactically duplicate bound terms (ISL may repeat across basic sets)."""
    seen = set()
    out = []
    for t in terms:
        st = symbolic.simplify(t)
        key = str(st)
        if key not in seen:
            seen.add(key)
            out.append(st)
    return out
