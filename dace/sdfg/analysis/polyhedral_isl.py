# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A reusable exact-integer ISL polyhedral layer.

Provides quasi-affine expression <-> ISL rendering (affine plus integer
``floor`` / ``ceil`` / ``mod``), integer ``isl.Set`` construction, exact
emptiness queries, and ``isl.Constraint`` -> symbolic extraction. The layer is
free of DaCe SDFG types -- its inputs are :mod:`dace.symbolic` expressions plus
variable-name strings and its outputs are symbolic expressions -- so it can be
unit-tested without building an SDFG and reused by any pass that needs
dependence / domain reasoning (:class:`WavefrontSkew` today; a future
``LoopToMap`` dependence analysis).

``islpy`` is an optional dependency: when it is absent :data:`HAVE_ISL` is
``False`` and callers are expected to degrade gracefully.
"""
from typing import Dict, List, Sequence, Tuple

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
