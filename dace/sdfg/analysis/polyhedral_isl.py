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

``islpy`` is a hard dependency of dace (see ``pyproject.toml``).
"""

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import islpy as isl
import sympy

from dace.sdfg.narrowing import SymbolicLike, as_expr, coeff_of, free_symbols, simplified
from dace.symbolic import int_ceil, int_floor


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
        mp[d] = safe_name("d", i)
    for i, p in enumerate(params):
        mp[p] = safe_name("P", i)
    return mp


def subs_by_name(expr: SymbolicLike, mapping: Mapping[str, sympy.Expr]) -> sympy.Expr:
    """``expr.subs`` matching the expression's *actual* free symbols by name, so
    it is immune to symbol assumption mismatches (two ``symbol('u')`` with
    different assumptions are unequal under a plain ``subs`` dict)."""
    e = simplified(expr)
    sub: Dict[sympy.Basic | complex, sympy.Basic | complex] = {}
    for s in free_symbols(e):
        if s.name in mapping:
            sub[s] = mapping[s.name]
    return simplified(e.subs(sub))


def isl_divisor(b: sympy.Expr) -> str:
    """Render a ``floor`` / ``ceil`` / ``mod`` divisor. ISL only admits integer
    division, so the divisor must be a positive integer constant; a symbolic or
    non-positive divisor is refused (``ValueError``)."""
    if b.is_Integer and int(b) > 0:
        return str(int(b))
    raise ValueError(f"non-constant / non-positive divisor {b}")


def to_isl(e: sympy.Expr) -> str:
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
    if isinstance(e, sympy.Symbol):
        return e.name
    if isinstance(e, int_floor):
        return f"floor(({to_isl(as_expr(e.args[0]))})/{isl_divisor(as_expr(e.args[1]))})"
    if isinstance(e, int_ceil):
        return f"ceil(({to_isl(as_expr(e.args[0]))})/{isl_divisor(as_expr(e.args[1]))})"
    if isinstance(e, sympy.Mod):
        return f"(({to_isl(as_expr(e.args[0]))}) mod {isl_divisor(as_expr(e.args[1]))})"
    if e.is_Add:
        return "(" + " + ".join(to_isl(as_expr(t)) for t in e.args) + ")"
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


def render_affine(expr: SymbolicLike, name_map: Dict[str, str]) -> str:
    """Render a quasi-affine expression into an ISL constraint string under
    ``name_map`` (see :func:`to_isl`). Raises ``ValueError`` on a form ISL cannot
    represent exactly, which the caller treats as "cannot reason, refuse" rather
    than risking an unsound query."""
    e = subs_by_name(expr, {orig: as_expr(safe) for orig, safe in name_map.items()})
    safe_syms = {as_expr(s) for s in name_map.values()}
    if not free_symbols(e) <= safe_syms:
        raise ValueError(f"unmapped symbol in {expr} (free={free_symbols(e) - safe_syms})")
    return to_isl(e)


def constraints_str(constraints: Iterable[SymbolicLike], name_map: Dict[str, str]) -> str:
    """`` and ``-joined ISL constraint body from exprs each meaning ``>= 0``."""
    parts = [f"({render_affine(c, name_map)}) >= 0" for c in constraints]
    return " and ".join(parts) if parts else "true"


def make_set(
    dims: Sequence[str], params: Sequence[str], constraints: Iterable[SymbolicLike]
) -> Tuple[isl.Set, Dict[str, str]]:
    """Build an ``isl.Set`` over ``dims`` parametrised by ``params`` from a list
    of exprs (each ``>= 0``). Returns ``(set, name_map)``."""
    name_map = build_name_map(dims, params)
    pstr = ", ".join(name_map[p] for p in params)
    dstr = ", ".join(name_map[d] for d in dims)
    body = constraints_str(constraints, name_map)
    text = f"[{pstr}] -> {{ [{dstr}] : {body} }}"
    return isl.Set(text), name_map


def is_domain_empty(dims: Sequence[str], params: Sequence[str], constraints: Iterable[SymbolicLike]) -> bool:
    """Exact integer emptiness of ``{ constraints }``. Raises on a bad render."""
    s, _ = make_set(dims, params, constraints)
    return s.is_empty()


def value_to_int(v: isl.Val) -> int:
    return int(v.to_python())


def constraint_to_sympy(
    c: isl.Constraint, safe_dims: Sequence[str], safe_params: Sequence[str], inv_map: Dict[str, str]
) -> sympy.Expr:
    """Reconstruct the affine expr (meaning ``>= 0``) of an ``isl.Constraint``,
    mapping ISL-safe names back to originals via ``inv_map``."""
    expr = as_expr(value_to_int(c.get_constant_val()))
    for i, nm in enumerate(safe_dims):
        co = value_to_int(c.get_coefficient_val(isl.dim_type.set, i))
        if co:
            expr = expr + co * as_expr(inv_map[nm])
    for i, nm in enumerate(safe_params):
        co = value_to_int(c.get_coefficient_val(isl.dim_type.param, i))
        if co:
            expr = expr + co * as_expr(inv_map[nm])
    return simplified(expr)


def collect_basic_sets(s: isl.Set) -> List[isl.BasicSet]:
    """``isl.Set`` -> list of basic sets. (ISL callbacks abort the process on any
    Python exception, so the callback only appends -- never computes.)"""
    out: List[isl.BasicSet] = []
    s.foreach_basic_set(lambda b: out.append(b))
    return out


def classify_dim(e: sympy.Expr, dsym: sympy.Expr) -> Tuple[List[sympy.Expr], List[sympy.Expr], bool]:
    """Split an ``e >= 0`` constraint into lower/upper bound terms for ``dsym``.

    coeff == 0  -> not a bound on this dim (ignore).
    coeff == 1  -> dsym >= -rest.
    coeff == -1 -> dsym <= rest.
    |coeff| > 1 -> an integer-division bound: ``c*dsym + rest >= 0`` gives, for
    ``c > 0``, ``dsym >= ceil(-rest / c)`` (a lower bound); for ``c < 0``,
    ``dsym <= floor(rest / -c)`` (an upper bound). This is exact over the
    integers -- it is the shape a steep skew (``tau = (2, 1)``, Gauss-Seidel)
    leaves on the parallel ``p`` axis, where the complementary coordinate scales
    ``p`` by ``|a|`` (here ``2``). ``int_ceil`` / ``int_floor`` render to DaCe's
    native integer-division intrinsics, so the resulting loop / Map bound is
    codegen-legal.
    """
    coeff = coeff_of(e, dsym)
    rest = simplified(e - coeff * dsym)
    if coeff == 0:
        return [], [], True
    if coeff == 1:
        return [simplified(-rest)], [], True
    if coeff == -1:
        return [], [simplified(rest)], True
    if coeff > 0:
        return [int_ceil(simplified(-rest), coeff)], [], True
    return [], [int_floor(simplified(rest), -coeff)], True


def pwaff_bound(pw: isl.PwAff, inv_map: Dict[str, str]) -> sympy.Expr | None:
    """A single-piece ``isl.PwAff`` (as returned by ``Set.dim_min`` / ``dim_max``)
    rendered to a symbolic expression, with ISL-safe names mapped back through
    ``inv_map``. The piece's parameter guard (e.g. ``N >= 3``, a mere
    nonemptiness condition) is ignored -- an out-of-range outer index just yields
    an empty inner range. Returns ``None`` when the bound is genuinely piecewise
    (several affine pieces over disjoint parameter regions) or carries an integer
    ``div`` term, neither of which is a single loop-bound expression.

    This is the exact-integer route for the ``t`` (sequential-diagonal) range of a
    steep skew: projecting the parallel axis out of a slanted (non-unit-scaled)
    domain leaves an ISL *existential* that per-constraint reading cannot turn
    into a clean bound, but ``dim_min`` / ``dim_max`` resolve it exactly.
    """
    pieces: List[isl.Aff] = []
    pw.foreach_piece(lambda st, aff: pieces.append(aff))
    if len(pieces) != 1:
        return None
    aff = pieces[0]
    if aff.dim(isl.dim_type.div) != 0:
        return None
    expr = as_expr(value_to_int(aff.get_constant_val()))
    for kind in (isl.dim_type.param, isl.dim_type.in_):
        for i in range(aff.dim(kind)):
            co = value_to_int(aff.get_coefficient_val(kind, i))
            if co:
                nm = aff.get_dim_name(kind, i)
                assert nm is not None  # every ISL dim of these sets is named by make_set
                expr = expr + co * as_expr(inv_map.get(nm, nm))
    return simplified(expr)


def dedupe_terms(terms: Sequence[SymbolicLike]) -> List[sympy.Expr]:
    """Drop syntactically duplicate bound terms (ISL may repeat across basic sets)."""
    seen: set[str] = set()
    out: List[sympy.Expr] = []
    for t in terms:
        st = simplified(t)
        key = str(st)
        if key not in seen:
            seen.add(key)
            out.append(st)
    return out
