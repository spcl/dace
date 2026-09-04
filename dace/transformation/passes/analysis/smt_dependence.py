# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SMT-backed dependence oracle for non-affine loop-carried dependences.

The oracle is a *conservative* fallback: it asks z3 to prove the absence of a
loop-carried dependence (or the direction of the only possible dependence).
When z3 is unavailable, every query returns ``None`` and callers fall back to
their existing safe refusal.
"""
from typing import Any, Dict, List, Optional

import sympy as sp

from dace import symbolic
from dace.sdfg.analysis.cfg import collect_enclosing_conditions  # noqa: F401 -- re-exported

try:
    import z3
    _HAS_Z3 = True
except Exception:
    z3 = None  # type: ignore
    _HAS_Z3 = False


def has_z3() -> bool:
    return _HAS_Z3


# Default SMT timeout in milliseconds. Keep short: this runs inside the
# canonicalize pipeline, not a verification bench.
DEFAULT_TIMEOUT_MS = 5000


def _z3_int(expr):
    """Wrap an expression so z3 treats it as an integer."""
    if isinstance(expr, int):
        return z3.IntVal(expr)
    return expr


def _pow_to_mul(base, exp):
    """Turn an integer power into a product so z3's integer solver can see it."""
    if exp == 0:
        return z3.IntVal(1)
    if exp < 0:
        return z3.IntVal(0)  # unsupported negative exponent; conservative
    out = base
    for _ in range(exp - 1):
        out = out * base
    return out


def _array_rank(arr: Any) -> int:
    """How many indices ``arr`` must be given before it yields a scalar."""
    rank = 0
    sort = arr.sort()
    # ``z3.is_array_sort`` takes an EXPRESSION and reads its sort; handed a sort it raises
    # ``ast is not an expression``. The sort itself is what has to be classified here.
    while isinstance(sort, z3.ArraySortRef):
        rank += 1
        sort = sort.range()
    return rank


def _sympy_to_z3(expr: sp.Basic, sym_cache: Dict[str, Any], arr_cache: Dict[str, Any]) -> Any:
    """Translate a sympy expression into a z3 integer term.

    Symbols become integer variables; array subscripts ``A[i]`` become
    ``Select(A, i)``; common integer operations are mapped directly.  Anything
    that cannot be translated returns ``None``.
    """
    if expr is None:
        return None

    if isinstance(expr, (int, sp.Integer)):
        return z3.IntVal(int(expr))

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name not in sym_cache:
            sym_cache[name] = z3.Int(name)
        return sym_cache[name]

    if isinstance(expr, sp.Indexed):
        base = expr.base
        if not isinstance(base, sp.Symbol):
            return None
        arr_name = str(base)
        if arr_name not in arr_cache:
            arr_cache[arr_name] = z3.Array(arr_name, z3.IntSort(), z3.IntSort())
        arr = arr_cache[arr_name]
        if len(expr.indices) != 1:
            return None
        idx = _sympy_to_z3(expr.indices[0], sym_cache, arr_cache)
        if idx is None:
            return None
        return z3.Select(arr, _z3_int(idx))

    if isinstance(expr, symbolic.Subscript):
        arr_name = str(expr.args[0])
        rank = len(expr.args) - 1
        if rank < 1:
            return None
        if arr_name not in arr_cache:
            # Build a nested array type of the right rank; rank 1 is the common case.
            range_sort = z3.IntSort()
            for _ in range(rank - 1):
                range_sort = z3.ArraySort(z3.IntSort(), range_sort)
            arr_cache[arr_name] = z3.Array(arr_name, z3.IntSort(), range_sort)
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
            cur = z3.Select(cur, _z3_int(idx))
        return cur

    if isinstance(expr, sp.Pow):
        base = _sympy_to_z3(expr.base, sym_cache, arr_cache)
        if base is None:
            return None
        exp = expr.exp
        if isinstance(exp, (int, sp.Integer)):
            return _pow_to_mul(base, int(exp))
        return None

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
            out = z3.If(a > out, a, out)
        return out

    if isinstance(expr, sp.Min):
        args = [_sympy_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = z3.If(a < out, a, out)
        return out

    if isinstance(expr, sp.floor):
        # Rational inner expressions fail translation above, so the argument is already an
        # integer term and floor is the identity on it.
        return _sympy_to_z3(expr.args[0], sym_cache, arr_cache)

    if isinstance(expr, sp.Mod):
        a = _sympy_to_z3(expr.args[0], sym_cache, arr_cache)
        b = _sympy_to_z3(expr.args[1], sym_cache, arr_cache)
        if a is None or b is None:
            return None
        # z3's `%` is SMT-LIB `mod`, which matches sympy's Mod for positive divisors --
        # the only divisors that appear in subscripts. SRem would diverge on negative
        # dividends and let the solver certify a false disjointness.
        return _z3_int(a) % _z3_int(b)

    if isinstance(expr, sp.Function):
        # Opaque functions are not modeled.
        return None

    return None


def _bool_to_z3(expr: sp.Basic, sym_cache: Dict[str, Any], arr_cache: Dict[str, Any]) -> Any:
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
    func = str(expr.func) if isinstance(expr, sp.Basic) else ''
    if func in ('AND', 'OR'):
        args = [_bool_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        joiner = z3.And if func == 'AND' else z3.Or
        out = args[0]
        for a in args[1:]:
            out = joiner(out, a)
        return out

    if func == 'NOT':
        arg = _bool_to_z3(expr.args[0], sym_cache, arr_cache)
        return None if arg is None else z3.Not(arg)

    if isinstance(expr, sp.And):
        args = [_bool_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = z3.And(out, a)
        return out

    if isinstance(expr, sp.Or):
        args = [_bool_to_z3(a, sym_cache, arr_cache) for a in expr.args]
        if any(a is None for a in args):
            return None
        out = args[0]
        for a in args[1:]:
            out = z3.Or(out, a)
        return out

    if isinstance(expr, sp.Not):
        arg = _bool_to_z3(expr.args[0], sym_cache, arr_cache)
        if arg is None:
            return None
        return z3.Not(arg)

    return None


def _iter_bounds(i: Any, start: Any, end: Any, step: Any) -> List[Any]:
    """z3 constraints that put ``i`` inside the strided iteration domain."""
    if isinstance(start, (int, sp.Integer)):
        start_z = z3.IntVal(int(start))
    else:
        start_z = _sympy_to_z3(start, {}, {})
    if isinstance(end, (int, sp.Integer)):
        end_z = z3.IntVal(int(end))
    else:
        end_z = _sympy_to_z3(end, {}, {})
    # Both bounds must be INTEGER terms. A bound that translated to an array read (or anything
    # else non-arithmetic) builds an ill-sorted ``i >= start``, and z3 answers that by aborting the
    # process inside ``Z3_solver_assert`` rather than raising -- so it has to be caught here.
    if start_z is None or end_z is None or not z3.is_int(start_z) or not z3.is_int(end_z):
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
    cons = [i >= lo, i <= hi]
    if abs(step_v) != 1:
        # (i - start) % |step| == 0, so the solver knows two distinct iterations are a whole
        # stride apart rather than adjacent. ``z3.SRem`` is bitvector-only and raises on an
        # Int term, so it silently lost this constraint and every strided loop was reasoned about
        # as if it stepped by one. ``%`` is SMT-LIB ``mod``; the magnitude is what is asked for
        # because divisibility does not care which way the loop travels.
        cons.append((i - start_z) % z3.IntVal(abs(step_v)) == 0)
    return cons


def _prove_unsat(antecedent: Any, consequent: Any, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Optional[bool]:
    """Return ``True`` if ``antecedent => consequent`` is valid (consequent holds
    for every model of antecedent), ``False`` if a counter-model exists, and
    ``None`` if the solver gives up or the encoding failed."""
    # An ill-sorted term reaches z3 as a process ABORT, not an exception, so every query is
    # sort-checked before the solver sees it. Callers build these from translated sympy, where a
    # single untranslatable subterm can leave a non-boolean behind.
    if not (z3.is_bool(antecedent) and z3.is_bool(consequent)):
        return None
    s = z3.Solver()
    s.set('timeout', timeout_ms)
    s.add(antecedent)
    s.add(z3.Not(consequent))
    try:
        r = s.check()
        if r == z3.unsat:
            return True
        if r == z3.sat:
            return False
        return None
    except Exception:
        return None


def _domain_assumptions(start: Any, end: Any, step: Any) -> sp.Basic:
    """Default domain assumptions for a loop: start <= end, step > 0.

    These are the invariants the canonicalization pipeline maintains after
    ``NormalizeNegativeStride``.
    """
    try:
        step_v = int(symbolic.evaluate(step, {}))
    except Exception:
        step_v = None
    if step_v is not None:
        step_expr = sp.Integer(step_v)
    else:
        step_expr = sp.Symbol(str(step))
    return sp.And(sp.LessThan(start, end + 1), sp.StrictGreaterThan(step_expr, 0))


def prove_injective_write(write_expr: sp.Basic,
                          itervar: str,
                          start: Any,
                          end: Any,
                          step: Any = 1,
                          domain_assumptions: Optional[sp.Basic] = None,
                          timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Optional[bool]:
    """Prove that distinct iterations write to distinct locations.

    :param write_expr: The write-index expression in terms of ``itervar``.
    :returns: ``True`` if z3 proves injectivity; ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None

    sym_cache: Dict[str, Any] = {}
    arr_cache: Dict[str, Any] = {}

    i1 = z3.Int(f'{itervar}_1')
    i2 = z3.Int(f'{itervar}_2')
    sym_cache[itervar] = i1
    w1 = _sympy_to_z3(write_expr, sym_cache, arr_cache)
    sym_cache[itervar] = i2
    w2 = _sympy_to_z3(write_expr, sym_cache, arr_cache)
    if w1 is None or w2 is None:
        return None

    bounds = _iter_bounds(i1, start, end, step) + _iter_bounds(i2, start, end, step)
    if not bounds:
        return None

    antecedent = z3.And(i1 != i2, *bounds)
    if domain_assumptions is not None:
        dom = _bool_to_z3(domain_assumptions, {}, {})
        if dom is not None:
            antecedent = z3.And(antecedent, dom)

    consequent = w1 != w2
    return _prove_unsat(antecedent, consequent, timeout_ms)


def prove_disjoint_write_ranges(lo_expr: sp.Basic,
                                hi_expr: sp.Basic,
                                itervar: str,
                                start: Any,
                                end: Any,
                                step: Any = 1,
                                domain_assumptions: Optional[sp.Basic] = None,
                                timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Optional[bool]:
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

    sym_cache: Dict[str, Any] = {}
    arr_cache: Dict[str, Any] = {}

    i1 = z3.Int(f'{itervar}_1')
    i2 = z3.Int(f'{itervar}_2')
    sym_cache[itervar] = i1
    lo1 = _sympy_to_z3(lo_expr, sym_cache, arr_cache)
    hi1 = _sympy_to_z3(hi_expr, sym_cache, arr_cache)
    sym_cache[itervar] = i2
    lo2 = _sympy_to_z3(lo_expr, sym_cache, arr_cache)
    hi2 = _sympy_to_z3(hi_expr, sym_cache, arr_cache)
    if lo1 is None or hi1 is None or lo2 is None or hi2 is None:
        return None

    bounds = _iter_bounds(i1, start, end, step) + _iter_bounds(i2, start, end, step)
    if not bounds:
        return None

    antecedent = z3.And(i1 != i2, *bounds)
    if domain_assumptions is not None:
        dom = _bool_to_z3(domain_assumptions, {}, {})
        if dom is not None:
            antecedent = z3.And(antecedent, dom)

    consequent = z3.Not(z3.And(lo1 <= hi2, lo2 <= hi1))
    return _prove_unsat(antecedent, consequent, timeout_ms)


def prove_disjoint_access_boxes(box1: List[Any],
                                box2: List[Any],
                                itervar: str,
                                start: Any,
                                end: Any,
                                step: Any = 1,
                                domain_assumptions: Optional[sp.Basic] = None,
                                timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Optional[bool]:
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
    sym_cache: Dict[str, Any] = {}
    arr_cache: Dict[str, Any] = {}
    i1 = z3.Int(f'{itervar}_1')
    i2 = z3.Int(f'{itervar}_2')
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
        if any(t is None or not z3.is_int(t) for t in (a1, b1, a2, b2)):
            return None
        intersects += [a1 <= b2, a2 <= b1]

    bounds = _iter_bounds(i1, start, end, step) + _iter_bounds(i2, start, end, step)
    if not bounds:
        return None

    antecedent = z3.And(i1 != i2, *bounds)
    if domain_assumptions is not None:
        dom = _bool_to_z3(domain_assumptions, {}, {})
        if dom is not None:
            antecedent = z3.And(antecedent, dom)

    return _prove_unsat(antecedent, z3.Not(z3.And(*intersects)), timeout_ms)


def _overlap_pair(write_expr: sp.Basic,
                  read_expr: sp.Basic,
                  read_guard: Optional[sp.Basic],
                  itervar: str,
                  start: Any,
                  end: Any,
                  step: Any,
                  order: str,
                  domain_assumptions: Optional[sp.Basic] = None) -> Optional[bool]:
    """Prove/disprove an overlap of the requested order.

    ``order`` is ``'raw'`` (a write iteration precedes a read iteration),
    ``'raw_le'`` (a write iteration precedes *or equals* it) or ``'war'`` (a write
    iteration follows a read iteration).  The read access is optionally guarded by
    ``read_guard``.
    """
    sym_cache: Dict[str, Any] = {}
    arr_cache: Dict[str, Any] = {}

    i_w = z3.Int(f'{itervar}_w')
    i_r = z3.Int(f'{itervar}_r')
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

    if order == 'raw':
        order_cons = i_w < i_r
    elif order == 'raw_le':
        order_cons = i_w <= i_r
    else:
        order_cons = i_w > i_r
    antecedent = z3.And(order_cons, *bounds, gz)
    if domain_assumptions is not None:
        dom = _bool_to_z3(domain_assumptions, {}, {})
        if dom is not None:
            antecedent = z3.And(antecedent, dom)

    consequent = wz != rz
    return _prove_unsat(antecedent, consequent, timeout_ms=DEFAULT_TIMEOUT_MS)


def prove_read_ahead(read_expr: sp.Basic,
                     write_expr: sp.Basic,
                     itervar: str,
                     start: Any,
                     end: Any,
                     step: Any = 1,
                     read_guard: Optional[sp.Basic] = None,
                     domain_assumptions: Optional[sp.Basic] = None) -> Optional[bool]:
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
    return _overlap_pair(write_expr, read_expr, read_guard, itervar, start, end, step, 'raw_le', domain_assumptions)


def prove_no_write_after_read(read_expr: sp.Basic,
                              write_expr: sp.Basic,
                              itervar: str,
                              start: Any,
                              end: Any,
                              step: Any = 1,
                              read_guard: Optional[sp.Basic] = None,
                              domain_assumptions: Optional[sp.Basic] = None) -> Optional[bool]:
    """Prove that no iteration AFTER the reader's own writes the element it reads.

    The other half of :func:`prove_read_ahead`, which covers the iterations up to and including
    the reader's own. Both True means the read never aliases the write in any iteration at all --
    there is no dependence of either direction to break, so a snapshot-rename would buy nothing
    and only add a copy.

    :returns: ``True`` when the solver proves it, ``False``/``None`` otherwise.
    """
    if not _HAS_Z3:
        return None
    return _overlap_pair(write_expr, read_expr, read_guard, itervar, start, end, step, 'war', domain_assumptions)


def classify_read_write_pair(read_expr: sp.Basic,
                             write_expr: sp.Basic,
                             itervar: str,
                             start: Any,
                             end: Any,
                             step: Any = 1,
                             read_guard: Optional[sp.Basic] = None,
                             domain_assumptions: Optional[sp.Basic] = None) -> Optional[str]:
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

    no_overlap_raw = _overlap_pair(write_expr, read_expr, read_guard, itervar, start, end, step, 'raw',
                                   domain_assumptions)
    no_overlap_war = _overlap_pair(write_expr, read_expr, read_guard, itervar, start, end, step, 'war',
                                   domain_assumptions)

    if no_overlap_raw is True and no_overlap_war is True:
        return 'none'
    if no_overlap_raw is False:
        return 'RAW'
    if no_overlap_raw is True:
        # No RAW can occur, so any remaining alias is at worst a WAR.
        return 'WAR'
    return None


__all__ = [
    'has_z3',
    'prove_injective_write',
    'prove_disjoint_write_ranges',
    'prove_read_ahead',
    'prove_no_write_after_read',
    'classify_read_write_pair',
    # Re-exported from ``sdfg.analysis.cfg``: the branch guards a read executes under are a
    # plain control-flow fact, shared with the polyhedral engine in ``wavefront_skew``.
    'collect_enclosing_conditions',
]
