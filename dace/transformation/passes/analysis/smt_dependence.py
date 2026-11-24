# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SMT-backed dependence oracle for non-affine loop-carried dependences.

The oracle is a *conservative* fallback: it asks z3 to prove the absence of a
loop-carried dependence (or the direction of the only possible dependence).
When z3 is unavailable, every query returns ``None`` and callers fall back to
their existing safe refusal.
"""

from collections.abc import Mapping
from typing import Any, NamedTuple

import sympy as sp

from dace import symbolic
from dace.config import Config
from dace.optionals import required
from dace.sdfg.analysis.cfg import collect_enclosing_conditions  # noqa: F401 -- re-exported

try:
    import z3

    _HAS_Z3 = True
    # A collector run on another thread (pytest-xdist's receiver) finalizes terms while this thread is inside z3 with
    # the GIL released; z3's reference counts are not thread-safe, so that aborted the process. Deferred instead.
    z3.Z3_enable_concurrent_dec_ref(z3.main_ctx().ref())
except Exception:
    z3 = None  # type: ignore
    _HAS_Z3 = False


def has_z3() -> bool:
    """Whether the solver may be asked: z3 is importable and ``optimizer.canonicalization.smt`` holds."""
    return _HAS_Z3 and Config.get_bool("optimizer", "canonicalization", "smt")


# Solver budget in z3 resource units, not milliseconds: a wall-clock limit lets machine load flip
# a verdict. 50M units is ~8 CPU seconds at the ~6M units/s measured here, above the old 5 s
# budget; the largest oracle query measured on CloudSC ParallelizeLoops used 505 units.
DEFAULT_RLIMIT = 50_000_000


def bounded_solver(rlimit: int = DEFAULT_RLIMIT) -> Any:
    """A z3 solver limited to ``rlimit`` resource units, with no wall-clock limit."""
    solver = required(z3).Solver()
    solver.set("rlimit", rlimit)
    return solver


def _z3_int(expr):
    """Wrap an expression so z3 treats it as an integer."""
    if isinstance(expr, int):
        return required(z3).IntVal(expr)
    return expr


def _pow_to_mul(base, exp):
    """Turn an integer power into a product so z3's integer solver can see it."""
    if exp == 0:
        return required(z3).IntVal(1)
    if exp < 0:
        return None  # a fraction, not an integer term
    out = base
    for _ in range(exp - 1):
        out = out * base
    return out


def _power(base: Any, exp: Any) -> Any:
    """``base ** exp`` for a symbolic ``exp``: uninterpreted, but at least 1 for ``base >= 1`` and ``exp >= 0``."""
    if base is None or exp is None or not required(z3).is_int(_z3_int(exp)):
        return None
    base, exp = _z3_int(base), _z3_int(exp)
    value = required(z3).Function("fn_pow", *[required(z3).IntSort()] * 3)(base, exp)
    return required(z3).If(required(z3).And(base >= 1, exp >= 0, value < 1), 1, value)


def _array_rank(arr: Any) -> int:
    """How many indices ``arr`` must be given before it yields a scalar."""
    rank = 0
    sort = arr.sort()
    # ``z3.is_array_sort`` takes an EXPRESSION and reads its sort; handed a sort it raises
    # ``ast is not an expression``. The sort itself is what has to be classified here.
    while isinstance(sort, required(z3).ArraySortRef):
        rank += 1
        sort = sort.range()
    return rank


def _sympy_to_z3(expr: sp.Basic, sym_cache: dict[str, Any], arr_cache: dict[str, Any]) -> Any:
    """Translate a sympy expression into a z3 integer term.

    Symbols become integer variables; array subscripts ``A[i]`` become
    ``Select(A, i)``; common integer operations are mapped directly.  Anything
    that cannot be translated returns ``None``.
    """
    if expr is None:
        return None

    if isinstance(expr, (int, sp.Integer)):
        return required(z3).IntVal(int(expr))

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name not in sym_cache:
            sym_cache[name] = required(z3).Int(name)
        return sym_cache[name]

    if isinstance(expr, sp.Indexed):
        base = expr.base
        if not isinstance(base, sp.Symbol):
            return None
        arr_name = str(base)
        if arr_name not in arr_cache:
            arr_cache[arr_name] = required(z3).Array(arr_name, required(z3).IntSort(), required(z3).IntSort())
        arr = arr_cache[arr_name]
        if len(expr.indices) != 1:
            return None
        idx = _sympy_to_z3(expr.indices[0], sym_cache, arr_cache)
        if idx is None:
            return None
        return required(z3).Select(arr, _z3_int(idx))

    if isinstance(expr, symbolic.Subscript):
        arr_name = str(expr.args[0])
        rank = len(expr.args) - 1
        if rank < 1:
            return None
        if arr_name not in arr_cache:
            # Build a nested array type of the right rank; rank 1 is the common case.
            range_sort = required(z3).IntSort()
            for _ in range(rank - 1):
                range_sort = required(z3).ArraySort(required(z3).IntSort(), range_sort)
            arr_cache[arr_name] = required(z3).Array(arr_name, required(z3).IntSort(), range_sort)
        arr = arr_cache[arr_name]
        # The cache is keyed by NAME, so the same array read at two different ranks would apply
        # this rank's Select chain to the other's sort. z3 does not reject the ill-sorted term --
        # it SEGFAULTS inside ``Z3_solver_assert``, taking the interpreter with it. Refuse instead:
        # the two reads cannot be related here anyway.
        if _array_rank(arr) != rank:
            return None
        # Nested Select for multi-dimensional subscripts.
        cur = arr
        for idx_expr in expr.args[1:]:
            idx = _sympy_to_z3(idx_expr, sym_cache, arr_cache)
            if idx is None:
                return None
            cur = required(z3).Select(cur, _z3_int(idx))
        return cur

    if isinstance(expr, sp.Pow):
        base = _sympy_to_z3(expr.base, sym_cache, arr_cache)
        if base is None:
            return None
        exp = expr.exp
        if isinstance(exp, (int, sp.Integer)):
            return _pow_to_mul(base, int(exp))
        return _power(base, _sympy_to_z3(exp, sym_cache, arr_cache))

    if isinstance(expr, sp.Add):
        terms = [_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(t is None for t in terms):
            return None
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total

    if isinstance(expr, sp.Mul):
        factors = [_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(f is None for f in factors):
            return None
        total = factors[0]
        for f in factors[1:]:
            total = total * f
        return total

    if isinstance(expr, sp.Max):
        args = [_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = required(z3).If(a > out, a, out)
        return out

    if isinstance(expr, sp.Min):
        args = [_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = required(z3).If(a < out, a, out)
        return out

    if isinstance(expr, (sp.floor, sp.ceiling)):
        # ``i // 2`` arrives as ``floor(i / 2)``: an integer numerator over a positive integer denominator floors
        # with SMT-LIB ``div``; an argument that is already an integer term is its own floor and ceiling.
        num, den = sp.fraction(sp.together(expr.args[0]))
        if den != 1:
            a = _sympy_to_z3(num, sym_cache, arr_cache)
            if a is None or not den.is_Integer or den <= 0 or not required(z3).is_int(_z3_int(a)):
                return None
            a = _z3_int(a)
            return a / int(den) if isinstance(expr, sp.floor) else -((-a) / int(den))
        return _sympy_to_z3(expr.args[0], sym_cache, arr_cache)

    if isinstance(expr, (symbolic.IfExpr, symbolic.ITE)) and len(expr.args) == 3:
        cond = _bool_to_z3(expr.args[0], sym_cache, arr_cache)
        then, other = (_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args[1:])
        if cond is None or cond is True or then is None or other is None:
            return then if cond is True else None
        return required(z3).If(cond, _z3_int(then), _z3_int(other))

    if not isinstance(expr, sp.Function):
        return None
    args = [_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args]
    if any(a is None or not required(z3).is_int(_z3_int(a)) for a in args):
        return None
    args = [_z3_int(a) for a in args]
    name = type(expr).__name__
    If = required(z3).If
    if isinstance(expr, sp.Abs):
        return If(args[0] < 0, -args[0], args[0])
    if len(args) == 2:
        a, b = args
        # SMT-LIB ``div``/``mod`` floor for a positive divisor only; the negative one goes through ``-a / -b``
        floor = If(b < 0, (-a) / (-b), a / b)
        if isinstance(expr, symbolic.int_floor):
            return floor
        if isinstance(expr, symbolic.int_ceil):
            return -If(b < 0, a / (-b), (-a) / b)
        # Python's floored modulo (``py_mod`` is its spelling for the code generator): the sign of the divisor
        if isinstance(expr, sp.Mod) or name == "py_mod":
            return If(b < 0, -((-a) % (-b)), a % b)
        # C's truncated modulo: the sign of the dividend
        if isinstance(expr, symbolic.CMod):
            return If(a < 0, -((-a) % If(b < 0, -b, b)), a % If(b < 0, -b, b))
        if isinstance(expr, symbolic.ipow):
            return _power(a, b)
    # Any other function is uninterpreted: equal arguments give equal values, which is all a proof for every
    # interpretation may rely on
    return required(z3).Function(f"fn_{name}", *[required(z3).IntSort()] * (len(args) + 1))(*args)

    return None


def _bool_to_z3(expr: sp.Basic, sym_cache: dict[str, Any], arr_cache: dict[str, Any]) -> Any:
    """Translate a sympy boolean into a z3 boolean."""
    if expr is None or expr is True:
        return True

    if isinstance(expr, sp.Rel):
        lhs = _sympy_to_z3(expr.lhs, sym_cache, arr_cache)
        rhs = _sympy_to_z3(expr.rhs, sym_cache, arr_cache)
        if lhs is None or rhs is None:
            return None
        if isinstance(expr, sp.StrictLessThan):
            return lhs < rhs
        if isinstance(expr, sp.LessThan):
            return lhs <= rhs
        if isinstance(expr, sp.StrictGreaterThan):
            return lhs > rhs
        if isinstance(expr, sp.GreaterThan):
            return lhs >= rhs
        if isinstance(expr, sp.Equality):
            return lhs == rhs
        if isinstance(expr, sp.Ne):
            return lhs != rhs
        return None

    # DaCe's own connectives, which a PARSED condition is built from: ``pystr_to_symbolic``
    # keeps ``AND`` / ``OR`` / ``NOT`` as function nodes rather than folding them to sympy's,
    # so a guard collected off a ConditionalBlock arrives in this form and used to fall through
    # to ``None`` here -- silently disarming every guard-dependent proof.
    func = str(expr.func) if isinstance(expr, sp.Basic) else ""
    if func in ("AND", "OR"):
        args = [_bool_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        joiner = required(z3).And if func == "AND" else required(z3).Or
        out = args[0]
        for a in args[1:]:
            out = joiner(out, a)
        return out

    if func == "NOT":
        arg = _bool_to_z3(expr.args[0], sym_cache, arr_cache)
        return None if arg is None else required(z3).Not(arg)

    if isinstance(expr, sp.And):
        args = [_bool_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = required(z3).And(out, a)
        return out

    if isinstance(expr, sp.Or):
        args = [_bool_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = required(z3).Or(out, a)
        return out

    if isinstance(expr, sp.Not):
        arg = _bool_to_z3(expr.args[0], sym_cache, arr_cache)
        if arg is None:
            return None
        return required(z3).Not(arg)

    return None


def _iter_bounds(i: Any, start: Any, end: Any, step: Any, widen: bool = False) -> list[Any]:
    """z3 constraints that put ``i`` inside the strided iteration domain.

    :param widen: Leave out a bound z3 cannot express (``32 * 2**b``) instead of giving up: for a caller proving a
                  property of EVERY pair of iterations, a larger domain only makes the proof harder, never wrong.
    """
    if isinstance(start, (int, sp.Integer)):
        start_z = required(z3).IntVal(int(start))
    else:
        start_z = _sympy_to_z3(start, {}, {})
    if isinstance(end, (int, sp.Integer)):
        end_z = required(z3).IntVal(int(end))
    else:
        end_z = _sympy_to_z3(end, {}, {})
    # Both bounds must be INTEGER terms. A bound that translated to an array read (or anything
    # else non-arithmetic) builds an ill-sorted ``i >= start``, and z3 answers that by aborting the
    # process inside ``Z3_solver_assert`` rather than raising -- so it has to be caught here.
    start_z, end_z = (None if b is None or not required(z3).is_int(b) else b for b in (start_z, end_z))
    if (start_z is None or end_z is None) and not (widen and (start_z is not None or end_z is not None)):
        return []
    try:
        step_v = int(symbolic.evaluate(step, {}))
    except Exception:
        step_v = 0
    if step_v == 0:
        # The direction is unknown, and guessing it is not safe in either direction: a
        # ``start <= i <= end`` box is EMPTY for a backward loop, an empty antecedent makes every
        # implication vacuously valid, and every caller here reads validity as "provably disjoint".
        # Refuse to bound instead -- callers treat the absence of bounds as inconclusive.
        return []
    # DaCe loop bounds are inclusive and stated in ITERATION order, so a negative step counts DOWN
    # from ``start`` to ``end`` and the interval is the other way round.
    lo, hi = (start_z, end_z) if step_v > 0 else (end_z, start_z)
    cons = [*([] if lo is None else [i >= lo]), *([] if hi is None else [i <= hi])]
    if abs(step_v) != 1 and start_z is not None:
        # (i - start) % |step| == 0, so the solver knows two distinct iterations are a whole
        # stride apart rather than adjacent. ``z3.SRem`` is bitvector-only and raises on an
        # Int term, so it silently lost this constraint and every strided loop was reasoned about
        # as if it stepped by one. ``%`` is SMT-LIB ``mod``; the magnitude is what is asked for
        # because divisibility does not care which way the loop travels.
        cons.append((i - start_z) % required(z3).IntVal(abs(step_v)) == 0)
    return cons


def prove_unsat(antecedent: Any, consequent: Any, rlimit: int = DEFAULT_RLIMIT) -> bool | None:
    """Return ``True`` if ``antecedent => consequent`` is valid (consequent holds
    for every model of antecedent), ``False`` if a counter-model exists, and
    ``None`` if the solver gives up or the encoding failed."""
    # An ill-sorted term reaches z3 as a process ABORT, not an exception, so every query is
    # sort-checked before the solver sees it. Callers build these from translated sympy, where a
    # single untranslatable subterm can leave a non-boolean behind.
    if not (required(z3).is_bool(antecedent) and required(z3).is_bool(consequent)):
        return None
    s = bounded_solver(rlimit)
    s.add(antecedent)
    s.add(required(z3).Not(consequent))
    try:
        r = s.check()
        if r == required(z3).unsat:
            return True
        if r == required(z3).sat:
            return False
        return None
    except Exception:
        return None


def assumed(antecedent: Any, domain_assumptions: sp.Basic | None, facts: Any, terms: list[Any]) -> Any:
    """``antecedent`` with the caller's ``domain_assumptions`` and what ``facts`` (a
    :class:`~dace.transformation.passes.parallelization.facts.SolverFacts`) let the solver assume about ``terms``."""
    extra = []
    if domain_assumptions is not None:
        dom = _bool_to_z3(domain_assumptions, {}, {})
        if dom is not None:
            extra.append(dom)
    if facts is not None:
        extra += facts.axioms(terms)
    return required(z3).And(antecedent, *extra) if extra else antecedent


def prove_assuming(
    antecedent: Any,
    consequent: Any,
    domain_assumptions: sp.Basic | None,
    facts: Any,
    terms: list[Any],
    rlimit: int = DEFAULT_RLIMIT,
) -> bool | None:
    """:func:`prove_unsat` under :func:`assumed`; a proof tells ``facts`` that the facts it was offered held."""
    result = prove_unsat(assumed(antecedent, domain_assumptions, facts, terms), consequent, rlimit)
    if result is True and facts is not None:
        facts.confirm()
    return result


def prove_injective_write(
    write_expr: sp.Basic,
    itervar: str,
    start: Any,
    end: Any,
    step: Any = 1,
    domain_assumptions: sp.Basic | None = None,
    rlimit: int = DEFAULT_RLIMIT,
    facts: Any = None,
) -> bool | None:
    """Prove that distinct iterations write to distinct locations.

    :param write_expr: The write-index expression in terms of ``itervar``.
    :returns: ``True`` if z3 proves injectivity; ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None

    sym_cache: dict[str, Any] = {}
    arr_cache: dict[str, Any] = {}

    i1 = required(z3).Int(f"{itervar}_1")
    i2 = required(z3).Int(f"{itervar}_2")
    sym_cache[itervar] = i1
    w1 = _sympy_to_z3(write_expr, sym_cache, arr_cache)
    sym_cache[itervar] = i2
    w2 = _sympy_to_z3(write_expr, sym_cache, arr_cache)
    if w1 is None or w2 is None:
        return None

    bounds = _iter_bounds(i1, start, end, step, widen=True) + _iter_bounds(i2, start, end, step, widen=True)
    if not bounds:
        return None

    antecedent = required(z3).And(i1 != i2, *bounds)
    return prove_assuming(antecedent, w1 != w2, domain_assumptions, facts, [w1, w2, *bounds], rlimit)


def prove_disjoint_write_ranges(
    lo_expr: sp.Basic,
    hi_expr: sp.Basic,
    itervar: str,
    start: Any,
    end: Any,
    step: Any = 1,
    domain_assumptions: sp.Basic | None = None,
    rlimit: int = DEFAULT_RLIMIT,
    facts: Any = None,
) -> bool | None:
    """Prove that distinct iterations write to non-overlapping RANGES.

    The range form of :func:`prove_injective_write`: an iteration writes the inclusive interval
    ``[lo(i), hi(i)]`` rather than the single element ``write_expr(i)``, and the property that
    makes the loop parallel is that two distinct iterations' intervals never intersect. Chunked
    rewrites produce exactly this shape -- anti-dependence chunking writes
    ``[i, Min(N - 2, i + 4095)]`` with ``step = 4096``, where the tail clamp defeats the affine
    ``a*i+b`` matcher even though the chunks are plainly disjoint.

    Two inclusive intervals intersect iff ``lo1 <= hi2 and lo2 <= hi1``; an empty interval
    (``lo > hi``) makes that conjunction false, so an iteration that writes nothing is handled by
    the same formula.

    :param lo_expr: Lower bound of the written interval, in terms of ``itervar``.
    :param hi_expr: Upper (inclusive) bound of the written interval, in terms of ``itervar``.
    :returns: ``True`` if z3 proves disjointness; ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None

    sym_cache: dict[str, Any] = {}
    arr_cache: dict[str, Any] = {}

    i1 = required(z3).Int(f"{itervar}_1")
    i2 = required(z3).Int(f"{itervar}_2")
    sym_cache[itervar] = i1
    lo1 = _sympy_to_z3(lo_expr, sym_cache, arr_cache)
    hi1 = _sympy_to_z3(hi_expr, sym_cache, arr_cache)
    sym_cache[itervar] = i2
    lo2 = _sympy_to_z3(lo_expr, sym_cache, arr_cache)
    hi2 = _sympy_to_z3(hi_expr, sym_cache, arr_cache)
    if lo1 is None or hi1 is None or lo2 is None or hi2 is None:
        return None

    bounds = _iter_bounds(i1, start, end, step, widen=True) + _iter_bounds(i2, start, end, step, widen=True)
    if not bounds:
        return None

    antecedent = required(z3).And(i1 != i2, *bounds)
    consequent = required(z3).Not(required(z3).And(lo1 <= hi2, lo2 <= hi1))
    return prove_assuming(antecedent, consequent, domain_assumptions, facts, [lo1, hi1, lo2, hi2, *bounds], rlimit)


def prove_disjoint_access_boxes(
    box1: list[Any],
    box2: list[Any],
    itervar: str,
    start: Any,
    end: Any,
    step: Any = 1,
    domain_assumptions: sp.Basic | None = None,
    rlimit: int = DEFAULT_RLIMIT,
    facts: Any = None,
) -> bool | None:
    """Prove that two MULTI-DIMENSIONAL range accesses never touch the same element on
    two different iterations.

    The multi-dimensional generalization of :func:`prove_disjoint_write_ranges`, which compares one
    interval expression against itself. Here each access is its own box -- a list of one inclusive
    ``(lo, hi)`` interval per dimension, both in terms of ``itervar`` -- so the two can index the
    iteration variable in DIFFERENT dimensions. That is the shape a mirrored triangular access has:
    polybench covariance writes ``cov[i, i:M]`` and ``cov[i:M, i]`` and reads the first back, and
    those boxes intersect only on the diagonal of a single iteration. No dimension is disjoint on
    its own and neither box is a point, so the per-dimension and affine-certificate tests both
    abstain.

    A ``True`` verdict says the two accesses are independent ACROSS iterations, which serves both a
    write/write pair (neither ordering is observable) and a read/write pair (any alias has distance
    zero, so it lives inside one iteration whose body order is preserved).

    SOUNDNESS: every symbol but ``itervar`` becomes ONE z3 constant shared by both boxes, which
    describes the loop only while that symbol holds a single value for its whole execution. A symbol
    that varies INSIDE the body takes independent values in the two iterations and sharing it
    manufactures a bogus proof, so the caller must screen those out first (see
    ``loop_varying_symbols``).

    Two boxes intersect iff EVERY dimension's intervals intersect, and two inclusive intervals
    intersect iff ``lo1 <= hi2 and lo2 <= hi1``; an empty interval (``lo > hi``) falsifies its
    dimension, so an iteration that writes nothing is covered by the same formula. An inner stride
    only thins an interval, so proving the boxes disjoint proves the written sets disjoint whatever
    the strides are.

    :param box1: Inclusive ``(lo, hi)`` bounds of the first access, one pair per dimension.
    :param box2: The same for the second access; must have the same rank.
    :returns: ``True`` if z3 proves disjointness; ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None
    if len(box1) != len(box2) or not box1:
        return None

    # One cache pair for both boxes, so every symbol and array read outside ``itervar`` is the SAME
    # z3 constant on both sides -- a per-call cache would compare two unrelated uninterpreted terms.
    sym_cache: dict[str, Any] = {}
    arr_cache: dict[str, Any] = {}
    i1 = required(z3).Int(f"{itervar}_1")
    i2 = required(z3).Int(f"{itervar}_2")
    intersects = []
    for (lo1, hi1), (lo2, hi2) in zip(box1, box2):
        sym_cache[itervar] = i1
        a1 = _sympy_to_z3(lo1, sym_cache, arr_cache)
        b1 = _sympy_to_z3(hi1, sym_cache, arr_cache)
        sym_cache[itervar] = i2
        a2 = _sympy_to_z3(lo2, sym_cache, arr_cache)
        b2 = _sympy_to_z3(hi2, sym_cache, arr_cache)
        # Every bound must be an INTEGER term. A partially-applied array read comes back as an
        # array, and comparing one with ``<=`` builds an ill-sorted term that segfaults z3 rather
        # than raising, so the check has to happen before the term is built, not inside the solver.
        if any(t is None or not required(z3).is_int(t) for t in (a1, b1, a2, b2)):
            return None
        intersects += [a1 <= b2, a2 <= b1]

    bounds = _iter_bounds(i1, start, end, step, widen=True) + _iter_bounds(i2, start, end, step, widen=True)
    if not bounds:
        return None

    antecedent = required(z3).And(i1 != i2, *bounds)
    consequent = required(z3).Not(required(z3).And(*intersects))
    return prove_assuming(antecedent, consequent, domain_assumptions, facts, [*intersects, *bounds], rlimit)


def _overlap_pair(
    write_expr: sp.Basic,
    read_expr: sp.Basic,
    read_guard: sp.Basic | None,
    itervar: str,
    start: Any,
    end: Any,
    step: Any,
    order: str,
    domain_assumptions: sp.Basic | None = None,
    facts: Any = None,
) -> bool | None:
    """Prove/disprove an overlap of the requested order.

    ``order`` is ``'raw'`` (a write iteration precedes a read iteration),
    ``'raw_le'`` (a write iteration precedes *or equals* it) or ``'war'`` (a write
    iteration follows a read iteration).  The read access is optionally guarded by
    ``read_guard``.
    """
    sym_cache: dict[str, Any] = {}
    arr_cache: dict[str, Any] = {}

    i_w = required(z3).Int(f"{itervar}_w")
    i_r = required(z3).Int(f"{itervar}_r")
    sym_cache[itervar] = i_w
    wz = _sympy_to_z3(write_expr, sym_cache, arr_cache)
    sym_cache[itervar] = i_r
    rz = _sympy_to_z3(read_expr, sym_cache, arr_cache)
    gz = _bool_to_z3(read_guard, sym_cache, arr_cache) if read_guard is not None else True
    if wz is None or rz is None or gz is None:
        return None

    bounds = _iter_bounds(i_w, start, end, step) + _iter_bounds(i_r, start, end, step)
    if not bounds:
        return None

    # Bounds exist only for a numeric step; a negative one runs the larger iterator first
    w, r = (i_w, i_r) if int(symbolic.evaluate(step, {})) > 0 else (-i_w, -i_r)
    if order == "raw":
        order_cons = w < r
    elif order == "raw_le":
        order_cons = w <= r
    else:
        order_cons = w > r
    terms = [wz, rz, *bounds] + ([] if gz is True else [gz])
    antecedent = required(z3).And(order_cons, *bounds, gz)
    return prove_assuming(antecedent, wz != rz, domain_assumptions, facts, terms)


def prove_read_ahead(
    read_expr: sp.Basic,
    write_expr: sp.Basic,
    itervar: str,
    start: Any,
    end: Any,
    step: Any = 1,
    read_guard: sp.Basic | None = None,
    domain_assumptions: sp.Basic | None = None,
    facts: Any = None,
) -> bool | None:
    """Prove that a read only ever touches elements no iteration up to and including its own
    has written -- the precondition for breaking an anti-dependence by snapshotting.

    Stricter than a ``'WAR'`` verdict from :func:`classify_read_write_pair`, which excludes only
    a STRICTLY earlier write. A same-iteration alias must be excluded too: redirecting such a
    read to a pre-loop snapshot would hand it the stale original instead of the value its own
    iteration just wrote.

    :returns: ``True`` when the solver proves it, ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None
    return _overlap_pair(
        write_expr, read_expr, read_guard, itervar, start, end, step, "raw_le", domain_assumptions, facts
    )


def prove_no_write_after_read(
    read_expr: sp.Basic,
    write_expr: sp.Basic,
    itervar: str,
    start: Any,
    end: Any,
    step: Any = 1,
    read_guard: sp.Basic | None = None,
    domain_assumptions: sp.Basic | None = None,
    facts: Any = None,
) -> bool | None:
    """Prove that no iteration AFTER the reader's own writes the element it reads.

    The other half of :func:`prove_read_ahead`, which covers the iterations up to and including
    the reader's own. Both True means the read never aliases the write in any iteration at all --
    there is no dependence of either direction to break, so a snapshot-rename would buy nothing
    and only add a copy.

    :returns: ``True`` when the solver proves it, ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None
    return _overlap_pair(write_expr, read_expr, read_guard, itervar, start, end, step, "war", domain_assumptions, facts)


def classify_read_write_pair(
    read_expr: sp.Basic,
    write_expr: sp.Basic,
    itervar: str,
    start: Any,
    end: Any,
    step: Any = 1,
    read_guard: sp.Basic | None = None,
    domain_assumptions: sp.Basic | None = None,
    facts: Any = None,
) -> str | None:
    """Classify a single read/write pair as ``'WAR'``, ``'RAW'``, ``'none'``,
    or ``None`` (inconclusive).

    The classification is *one* of:

    * ``'none'`` -- z3 proves the two accesses never alias across iterations;
    * ``'WAR'``  -- no RAW can occur, so any alias is a write-after-read that a
      snapshot-rename can break;
    * ``'RAW'``  -- a true read-after-write alias exists (must stay sequential);
    * ``None``   -- the solver gave up or the expressions could not be encoded.
    """
    if not _HAS_Z3:
        return None

    no_overlap_raw = _overlap_pair(
        write_expr, read_expr, read_guard, itervar, start, end, step, "raw", domain_assumptions, facts
    )
    no_overlap_war = _overlap_pair(
        write_expr, read_expr, read_guard, itervar, start, end, step, "war", domain_assumptions, facts
    )

    if no_overlap_raw is True and no_overlap_war is True:
        return "none"
    if no_overlap_raw is False:
        return "RAW"
    if no_overlap_raw is True:
        # No RAW can occur, so any remaining alias is at worst a WAR.
        return "WAR"
    return None


#: One iterator an access sits under: ``(name, start, end, step)``, bounds inclusive and in iteration order.
IterRange = tuple[str, Any, Any, Any]

#: The branch conditions an access runs under, all of which hold: ``(condition, holds)``, where ``holds`` is
#: ``False`` for an earlier branch's condition that must fail.
Guard = tuple[tuple[sp.Basic, bool], ...]


def is_plain_index(expr: sp.Basic) -> bool:
    """Whether ``expr`` is integer arithmetic over symbols: no array read, no opaque function."""
    return not expr.atoms(sp.Indexed, symbolic.Subscript, sp.Function) - expr.atoms(sp.Min, sp.Max, sp.Mod)


def admissible(exprs: list[sp.Basic], allowed: set[str]) -> bool:
    """Whether every expression is integer arithmetic over ``allowed`` names only."""
    return all(is_plain_index(e) and {s.name for s in e.free_symbols} <= allowed for e in exprs)


def encode_side(suffix: str, index: tuple[sp.Basic, ...], iters: tuple[IterRange, ...], sym_cache: dict[str, Any]):
    """``(bounds, terms)`` for one access with its own copy of every iterator (named with ``suffix``), else None."""
    names = {name for name, *_ in iters}

    def local(expr):
        expr = sp.sympify(expr)
        return expr.xreplace({s: sp.Symbol(s.name + suffix) for s in expr.free_symbols if s.name in names})

    bounds = []
    for name, start, end, step in iters:
        domain = _iter_bounds(required(z3).Int(name + suffix), local(start), local(end), step)
        if not domain:
            return None
        bounds += domain
    terms = [_sympy_to_z3(local(e), sym_cache, {}) for e in index]
    if any(t is None or not required(z3).is_int(t) for t in terms):
        return None
    return bounds, terms


class NeedsComputed(NamedTuple):
    """A conflict exists, but only for values of ``values`` (``iphase[1]=7``): elements of arrays the program computes
    before the loop, which the solver could not show are reachable."""

    values: str


def selects(term: Any, found: dict[str, list[Any]]) -> None:
    """Collect every ``Select(array, index)`` in ``term`` by array name."""
    if required(z3).is_select(term):
        found.setdefault(str(term.arg(0)), []).append(term.arg(1))
    for child in term.children():
        selects(child, found)


def table_term(values: tuple[int, ...]) -> Any:
    """The array that holds ``values`` at ``0..len - 1``."""
    table = required(z3).K(required(z3).IntSort(), required(z3).IntVal(0))
    for index, value in enumerate(values):
        table = required(z3).Store(table, index, int(value))
    return table


def element_range(fact: Any, sym_cache: dict[str, Any]) -> tuple[Any, Any] | None:
    """``(low, high)`` z3 bounds (``None`` for an open end) every element of the array of ``fact`` lies in, else None
    for a fact that is not a range."""
    if fact.kind is symbolic.DataFactKind.NONNEGATIVE:
        return required(z3).IntVal(0), None
    if fact.kind is not symbolic.DataFactKind.BOUNDED:
        return None
    bounds = [
        None if b is None else _sympy_to_z3(symbolic.pystr_to_symbolic(str(b)), sym_cache, {})
        for b in (fact.lo, fact.hi)
    ]
    if any(b is not None and not required(z3).is_int(b) for b in bounds):
        return None
    return bounds[0], bounds[1]


def find_conflict_witness(
    first: tuple[sp.Basic, ...],
    second: tuple[sp.Basic, ...],
    first_iters: tuple[IterRange, ...],
    second_iters: tuple[IterRange, ...],
    sizes: set[str],
    rlimit: int = DEFAULT_RLIMIT,
    outer: tuple[IterRange, ...] = (),
    shape: tuple[sp.Basic | None, ...] = (),
    fixed: set[str] = frozenset(),
    kills: tuple[tuple[tuple[sp.Basic, ...], tuple[IterRange, ...], bool, Guard], ...] = (),
    guards: tuple[Guard, Guard] = ((), ()),
    tables: Mapping[str, tuple[int, ...]] | None = None,
    computed: frozenset[str] = frozenset(),
    facts: Any = None,
) -> dict[str, int] | NeedsComputed | None:
    """Values under which ``first``, at an earlier iteration of the carrying loop, and ``second``, at a later one,
    touch the same element; ``None`` when there are none, the solver gives up, or something cannot be encoded.

    The iterator lists run outermost first, and both start with the carrying loop. Each side gets its own copy of
    every iterator. The only other symbols the witness may choose are ``sizes``, at least 1: a symbol outside
    them is a fixed value nobody knows, and a conflict for one chosen value of it proves nothing about the
    program, so its presence refuses the query, unless it is in ``fixed``. ``outer`` are the counters of the loops around the carrying one,
    outermost first: both accesses share one copy, an iteration of those loops in which the carrying loop runs.
    ``shape`` is the accessed array's extent per dimension (``None`` where it is not in ``sizes``): both accesses
    must lie inside it, so no witness rests on an index no valid input reaches. ``fixed`` are parameters the
    program never changes while the carrying loop runs but whose value nobody knows (a column index a caller
    passes in): one may appear only if, for EVERY value of it under which the loops run with both accesses inside
    ``shape``, some iterations conflict (:func:`conflicts_for_every`), so the witness rests on no chosen value of it. ``kills`` are the writes ``(index, iters,
    same, guard)`` of the array in one iteration: none of them, at the later iteration, may store the element ``second``
    reads, except the very instance of the read's own statement when ``same`` (it writes after it reads).
    ``guards`` are the branch conditions ``first`` and ``second`` run under, each at its own iteration; an array they
    read is input the witness may choose, and its chosen elements are part of the witness (``flag[1]``).
    ``tables`` are guard arrays whose every element is known (constants of the SDFG): the guard reads them as they
    are. ``computed`` are guard arrays the program fills before the loop: the witness may not choose their elements,
    so the conflict must hold for every content the ``facts`` (:class:`SolverFacts`: the ranges of those arrays and of
    the parameters) allow, else a :class:`NeedsComputed` names the values the one found rests on.
    """
    exprs = [sp.sympify(e) for e in first + second]
    exprs += [sp.sympify(b) for _, start, end, _ in outer + first_iters + second_iters for b in (start, end)]
    kill_iters = tuple(it for kill in kills for it in kill[1])
    exprs += [sp.sympify(e) for kill in kills for e in kill[0]]
    exprs += [sp.sympify(b) for it in kill_iters for b in it[1:3]]
    iterators = {it[0] for it in outer + first_iters + second_iters + kill_iters}
    allowed = iterators | sizes | set(fixed)
    conditions = [sp.sympify(term[0]) for guard in (*guards, *(kill[3] for kill in kills)) for term in guard]
    if not _HAS_Z3 or len(first) != len(second) or not admissible(exprs, allowed):
        return None
    if any({s.name for s in e.free_symbols} - allowed for e in conditions):
        return None
    params = sorted({s.name for e in exprs + conditions for s in e.free_symbols} - iterators - sizes)
    sym_cache: dict[str, Any] = {}
    arr_cache: dict[str, Any] = {}
    sides = [encode_side("__a", first, first_iters, sym_cache), encode_side("__b", second, second_iters, sym_cache)]
    shared = encode_side("", (), outer, sym_cache)
    held = [
        encode_guard("__a", guards[0], first_iters, sym_cache, arr_cache),
        encode_guard("__b", guards[1], second_iters, sym_cache, arr_cache),
    ]
    if None in sides or shared is None or None in held:
        return None
    carrier, _, _, step = first_iters[0]
    earlier, later = required(z3).Int(carrier + "__a"), required(z3).Int(carrier + "__b")
    # Two ordered iterations belong to the domain: a parameter value under which the loop runs once orders nothing
    domain = [*shared[0], *sides[0][0], *sides[1][0]]
    domain.append(earlier < later if int(symbolic.evaluate(step, {})) > 0 else earlier > later)
    conflict = [*(a == b for a, b in zip(sides[0][1], sides[1][1])), *held[0], *held[1]]
    for number, (index, iters, same, guard) in enumerate(kills):
        suffix = f"__k{number}"
        side = encode_side(suffix, index, iters, sym_cache)
        runs = encode_guard(suffix, guard, iters, sym_cache, arr_cache)
        if side is None or runs is None or len(index) != len(second):
            return None
        hit = [*side[0], *runs, *(a == b for a, b in zip(side[1], sides[1][1]))]
        if same:
            hit.append(
                required(z3).Or(*(required(z3).Int(it[0] + suffix) != required(z3).Int(it[0] + "__b") for it in iters))
            )
        bound = [required(z3).Int(it[0] + suffix) for it in iters]
        hit = required(z3).And(required(z3).Int(carrier + suffix) == later, *hit)
        conflict.append(required(z3).Not(required(z3).Exists(bound, hit)))
    in_bounds = []
    for dim, extent in enumerate(shape):
        if extent is None or not admissible([sp.sympify(extent)], sizes):
            continue
        bound = _sympy_to_z3(sp.sympify(extent), sym_cache, {})
        if bound is None:
            continue
        in_bounds += [term >= 0 for term in (sides[0][1][dim], sides[1][1][dim])]
        in_bounds += [term < bound for term in (sides[0][1][dim], sides[1][1][dim])]
    tables = tables or {}
    reads: dict[str, list[Any]] = {}
    for term in (t for guard in held for t in guard):
        selects(term, reads)
    universal = []
    for name in sorted(tables.keys() & reads.keys()):
        table = table_term(tables[name])
        for x in reads[name]:
            conflict += [
                0 <= x,
                x < len(tables[name]),
                required(z3).Select(arr_cache[name], x) == required(z3).Select(table, x),
            ]
    for fact in facts.array_facts(set(computed) & reads.keys()) if facts is not None else ():
        bounds = element_range(fact, sym_cache)
        if bounds is None or fact.array not in reads:
            continue
        everywhere = required(z3).Int("__element")
        for x in (*reads[fact.array], everywhere):
            element = required(z3).Select(arr_cache[fact.array], x)
            held_range = [
                c
                for c in (bounds[0] is None or bounds[0] <= element, bounds[1] is None or element <= bounds[1])
                if c is not True
            ]
            if x is everywhere:
                universal.append(required(z3).ForAll([everywhere], required(z3).And(*held_range)))
            else:
                conflict += held_range
    quantified = sorted(set(computed) & reads.keys())
    # What the caller knows of the parameters' values: a loop runs only under them
    domain += facts.symbol_axioms(set(params)) if facts is not None and params else []
    solver = bounded_solver(rlimit)
    domain += in_bounds
    solver.add(*domain, *conflict, *(required(z3).Int(name) >= 1 for name in sorted(sizes)))
    try:
        if solver.check() != required(z3).sat:
            return None
    except Exception:
        return None
    model = solver.model()
    if (params or quantified) and not conflicts_for_every(
        domain, conflict, model, params, sizes, rlimit, quantified, universal
    ):
        if quantified and conflicts_for_every(domain, conflict, model, params, sizes, rlimit):
            return NeedsComputed(
                ", ".join(
                    f"{k}={v}"
                    for k, v in guard_elements(
                        model, guards, first_iters, second_iters, sym_cache, arr_cache, computed
                    ).items()
                )
            )
        return None
    shown = [name for name, *_ in outer] + [carrier + "__a", carrier + "__b"]
    shown += sorted({s.name for e in exprs for s in e.free_symbols} & sizes)
    # A parameter that bounds a loop decides which iterations exist, so the example names the value it rests on
    bounds = [sp.sympify(b) for it in outer + first_iters + second_iters for b in it[1:3]]
    shown += sorted({s.name for e in bounds for s in e.free_symbols} & set(params))
    witness = {name: model.eval(required(z3).Int(name), model_completion=True).as_long() for name in shown}
    witness.update(
        guard_elements(model, guards, first_iters, second_iters, sym_cache, arr_cache, set(), computed | tables.keys())
    )
    return witness


def guard_elements(
    model: Any,
    guards: tuple[Guard, Guard],
    first_iters: tuple[IterRange, ...],
    second_iters: tuple[IterRange, ...],
    sym_cache: dict[str, Any],
    arr_cache: dict[str, Any],
    only: Any,
    skip: Any = (),
) -> dict[str, int]:
    """The elements of the arrays the guards read, as the ``model`` chose them (``flag[1]``): those named in ``only``
    when it is not empty, else all but those named in ``skip``."""
    found = {}
    for suffix, guard, iters in (("__a", guards[0], first_iters), ("__b", guards[1], second_iters)):
        for term in guard:
            for element in sorted(local_names(term[0], iters, suffix).atoms(symbolic.Subscript), key=str):
                array = str(element.args[0])
                if (array not in only) if only else (array in skip):
                    continue
                index = [
                    model.eval(_sympy_to_z3(i, sym_cache, arr_cache), model_completion=True) for i in element.args[1:]
                ]
                value = model.eval(_sympy_to_z3(element, sym_cache, arr_cache), model_completion=True)
                found[f"{array}[{', '.join(str(i) for i in index)}]"] = value.as_long()
    return found


def local_names(expr: sp.Basic, iters: tuple[IterRange, ...], suffix: str) -> sp.Basic:
    """``expr`` with every iterator of ``iters`` renamed to its copy ``name + suffix``."""
    names = {it[0] for it in iters}
    expr = sp.sympify(expr)
    return expr.xreplace({s: sp.Symbol(s.name + suffix) for s in expr.free_symbols if s.name in names})


def encode_guard(
    suffix: str, guard: Guard, iters: tuple[IterRange, ...], sym_cache: dict[str, Any], arr_cache: dict[str, Any]
) -> list[Any] | None:
    """z3 terms for ``guard`` at the iteration named by ``suffix``, else None when a condition cannot be encoded. A
    condition that is a number holds when it is not zero, as in C."""
    terms = []
    for expr, holds in guard:
        expr = local_names(expr, iters, suffix)
        if expr in (sp.true, sp.false):
            term = required(z3).BoolVal(bool(expr))
        elif isinstance(expr, (sp.logic.boolalg.Boolean, sp.Rel)) or str(expr.func) in ("AND", "OR", "NOT"):
            term = _bool_to_z3(expr, sym_cache, arr_cache)
            term = None if term is True else term
        else:
            value = _sympy_to_z3(expr, sym_cache, arr_cache)
            term = None if value is None or not required(z3).is_int(value) else value != 0
        if term is None:
            return None
        terms.append(term if holds else required(z3).Not(term))
    return terms


def conflicts_for_every(
    domain: list[Any],
    conflict: list[Any],
    model: Any,
    params: list[str],
    sizes: set[str],
    rlimit: int,
    arrays: list[str] = (),
    premises: list[Any] = (),
) -> bool:
    """Whether, with the sizes of ``model``, every value of ``params`` and every content of ``arrays`` that satisfies
    ``premises``, under which the loops run in bounds (``domain`` holds for some pair of iterations, the carrying loop
    running at least twice), also has iterations that ``conflict``."""
    z3m = required(z3)
    variables = {str(v): v for v in z3m.z3util.get_vars(z3m.And(*domain, *conflict))}
    names = set(variables)
    # Iterators and the input arrays a branch condition reads: chosen per value of the parameters
    iters = [variables[name] for name in sorted(names - set(params) - set(arrays) - sizes)]
    solver = bounded_solver(rlimit)
    for name in sorted(names & sizes):
        solver.add(z3m.Int(name) == model.eval(z3m.Int(name), model_completion=True))
    runs = z3m.Exists(iters, z3m.And(*domain)) if iters else z3m.And(*domain)
    meets = z3m.Exists(iters, z3m.And(*domain, *conflict)) if iters else z3m.And(*domain, *conflict)
    solver.add(runs, *premises, z3m.Not(meets))
    try:
        return solver.check() == z3m.unsat
    except Exception:
        return False


__all__ = [
    "find_conflict_witness",
    "has_z3",
    "prove_injective_write",
    "prove_disjoint_write_ranges",
    "prove_read_ahead",
    "prove_no_write_after_read",
    "classify_read_write_pair",
    # Re-exported from ``sdfg.analysis.cfg``: the branch guards a read executes under are a
    # plain control-flow fact, shared with the polyhedral engine in ``wavefront_skew``.
    "collect_enclosing_conditions",
]
