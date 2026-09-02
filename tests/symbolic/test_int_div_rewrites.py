# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``int_floor`` / ``int_ceil`` rewrite rules.

``int_ceil`` used to fold only when BOTH operands were numbers, unlike ``int_floor`` which already
folded a unit denominator. That asymmetry leaked: ``strides_from_layout`` pads with the default
``alignment=1``, so every symbolic descriptor came back carrying an ``int_ceil(N, 1)`` that never
folded back to ``N``. Both functions now share the unit-denominator and exact-division rules.
"""
import numpy as np
import pytest
import sympy

import dace
from dace import dtypes
from dace.symbolic import (deserialize_symbolic, int_ceil, int_floor, pystr_to_symbolic, symbol, symstr,
                           sympy_intdiv_fix)

N = pystr_to_symbolic('N')
M = pystr_to_symbolic('M')
#: A stride and a trip count, carrying the assumptions an extent normally arrives with.
P = symbol('P', dtype=dtypes.int64, positive=True)
K = symbol('K', dtype=dtypes.int64, nonnegative=True)


def test_unit_denominator_folds_away():
    """Dividing by 1 rounds nothing up, and int_floor already folds it."""
    assert int_ceil(N, 1) == N
    assert int_floor(N, 1) == N
    assert int_ceil(N * M + 3, 1) == N * M + 3


@pytest.mark.parametrize('fn', [int_floor, int_ceil])
def test_exact_division_yields_the_quotient(fn):
    """Exact division is not a rounding operation, so neither node should survive it.

    Returning the quotient rather than the other rounding function keeps the result comparable and
    simplifiable -- ``int_floor(4*N, 4)`` and ``N`` are the same number and should compare equal.
    """
    assert fn(4 * N, 4) == N
    assert fn(8 * N + 16, 8) == N + 2
    assert fn(12 * N + 6, 3) == 4 * N + 2


@pytest.mark.parametrize('fn', [int_floor, int_ceil])
def test_inexact_division_is_left_symbolic(fn):
    """A numerator that may leave a remainder must keep the rounding node."""
    assert fn(N, 8).func is fn
    assert fn(4 * N + 2, 4).func is fn
    # A symbolic denominator cannot be shown to divide, so it is left alone too.
    assert fn(N, M).func is fn


@pytest.mark.parametrize('rounding', [sympy.floor, sympy.ceiling])
def test_rounding_of_a_non_integer_expression_is_kept(rounding):
    """A rounding call with nothing to divide by must survive the rewrite."""
    x = sympy.Symbol('x')
    expr = rounding(sympy.sin(x))
    assert sympy_intdiv_fix(expr) == expr


@pytest.mark.parametrize('rounding,call', [(sympy.floor, 'floor'), (sympy.ceiling, 'ceil')])
def test_rounding_of_a_non_integer_expression_lowers_to_the_math_call(rounding, call):
    """The kept rounding must reach C++ as the matching math-library call."""
    x = pystr_to_symbolic('x')
    assert symstr(rounding(sympy.sin(x)), cpp_mode=True) == '(%s(sin(x)))' % call


def test_ceiling_of_index_arithmetic_is_integer_typed():
    """A ceiling over integer symbols is an extent, and must reach C++ with an integer type.

    ``ceil`` returns a double. As a map's range end that makes the emitted loop compare an integer
    induction variable against a double, which is not OpenMP's canonical form, and the compiler
    rejects the enclosing ``omp for`` outright with "invalid controlling predicate" -- npbench's
    ``stockham_fft``, whose radix extent is ``R ** (K - 1)``.
    """
    r = symbol('R', dtype=dtypes.int64)
    k = symbol('K', dtype=dtypes.int64)
    emitted = symstr(sympy.ceiling(r**(k - 1)), cpp_mode=True)
    assert '(int64_t)' in emitted, emitted
    assert 'ceil(' in emitted, emitted


def test_ceiling_of_a_float_valued_call_on_integer_symbols_stays_floating():
    """Integrality is decided by the OPERATIONS, not the operands.

    Every atom of ``sin(n)`` is an integer symbol and the value is a double, so reading the atoms
    would type this one as an index and truncate it.
    """
    n = symbol('n', dtype=dtypes.int64)
    assert symstr(sympy.ceiling(sympy.sin(n)), cpp_mode=True) == '(ceil(sin(n)))'


def test_ceiling_of_an_integer_prints_as_its_argument():
    """A ``ceiling`` over a known-integer argument must print as the argument itself.

    Deserialization rebuilds an application through ``Basic.__new__`` and bypasses the
    ``eval`` that would have folded the wrapper, so a stored ``ceiling`` reaches the
    printer unevaluated. Neither call is acceptable there: libm ``ceil`` returns a
    ``double``, and the runtime's ``ceiling`` is only overloaded for ``int``, ``float``
    and ``double``, which leaves a 64-bit or unsigned argument ambiguous.
    """
    stored = deserialize_symbolic('ceiling(__int_floor($N, 2))')
    assert isinstance(stored, sympy.ceiling)
    assert stored.args[0].is_integer

    assert symstr(stored, cpp_mode=True) == '(((N) / (2)))'


@pytest.mark.parametrize('rounding,name', [(sympy.floor, 'int_floor'), (sympy.ceiling, 'int_ceil')])
def test_division_by_an_integer_still_rewrites(rounding, name):
    """The integer-division rewrite is unchanged where a real denominator is present."""
    assert sympy_intdiv_fix(rounding(N / 8)).func.__name__ == name


@pytest.mark.parametrize('x,y,expected', [(17, 8, 3), (16, 8, 2), (1, 8, 1), (0, 8, 0)])
def test_numeric_operands_fold(x, y, expected):
    assert int_ceil(x, y) == expected


@pytest.mark.parametrize('fn', [int_floor, int_ceil])
def test_a_divisor_that_may_be_zero_is_not_folded(fn):
    """Folding ``i // i`` to 1 asserts ``i != 0``, which nothing established.

    It also costs the expression its only symbol, and the autodiff store identifies a loop by the
    index still being IN the dimension it parsed -- so the fold does not merely over-claim, it
    breaks a consumer that has no way to see what was lost.
    """
    i = pystr_to_symbolic('i')
    assert fn(i, i).func is fn
    assert fn(i, i).free_symbols == {i}
    # The same numerator over a divisor that cannot be zero is exact, and does fold.
    assert fn(P, P) == 1


@pytest.mark.parametrize('fn', [int_floor, int_ceil])
def test_exact_division_reaches_a_symbolic_divisor(fn):
    """sympy cancels a monomial against a monomial but never a sum, so the terms are divided one at
    a time. Exactness needs no assumption beyond a nonzero divisor: nothing is rounded."""
    whole = (P * N + P) / P
    assert whole.is_integer is None, whole
    assert fn(P * N + P, P) == N + 1
    assert fn(P * (K - 1) + P, P) == K


@pytest.mark.parametrize('fn,folded', [(int_floor, K), (int_ceil, K + 1)])
def test_an_inexact_split_needs_the_assumptions_that_make_truncation_flooring(fn, folded):
    """Pulling the divisible terms out and leaving the remainder is an identity about FLOOR, and
    these emit C ``/``, which truncates toward zero. The two agree on a nonnegative numerator over a
    positive divisor -- which is what an extent normally is -- so the split is taken exactly there
    and refused where the assumptions are missing."""
    assert fn(4 * K + 2, 4) == folded
    assert fn(4 * N + 2, 4).func is fn


def test_a_numerator_that_can_go_negative_is_never_split():
    """The hazard the assumptions rule out. ``int_floor(3 - 2*P, P)`` splits to ``int_floor(3, P) -
    2``, which answers -2 where the program, at ``P = 5``, computes ``-7/5 == -1`` -- so a positive
    divisor alone does not earn the split."""
    assert int_floor(3 - 2 * P, P).func is int_floor
    assert int_ceil(3 - 2 * P, P).func is int_ceil


def test_a_strided_slice_extent_folds_to_its_trip_count():
    """``a[0:(K - 1)*P + 1:P]`` has extent ``int_ceil(P*(K - 1) + 1, P)``, which is ``K``.

    The biased numerator is what makes it reachable: ``dace::math::int_ceil`` IS ``(x + y - 1) / y``,
    and biasing turns this one into ``P*K`` over ``P``. So the fold is the value the emitted program
    computes, not an appeal to the mathematical ceiling.
    """
    assert int_ceil(P * (K - 1) + 1, P) == K
    assert int_ceil(P * K + 1, P) == K + 1


def divide_in_a_program(name: str, numerator: str, p: int) -> int:
    """The value the GENERATED program computes for ``int_floor(numerator, P)``."""
    sdfg = dace.SDFG(f'intdiv_{name}')
    sdfg.add_symbol('P', dace.int64)
    sdfg.add_symbol('r', dace.int64)
    sdfg.add_array('out', [1], dace.int64)
    entry, body = sdfg.add_state(), sdfg.add_state()
    sdfg.add_edge(entry, body, dace.InterstateEdge(assignments={'r': f'int_floor({numerator}, P)'}))
    tasklet = body.add_tasklet('write', {}, {'o'}, 'o = r')
    body.add_edge(tasklet, 'o', body.add_write('out'), None, dace.Memlet('out[0]'))

    out = np.zeros(1, dtype=np.int64)
    sdfg(out=out, P=p)
    return int(out[0])


def test_int_floor_truncates_toward_zero_in_the_generated_code():
    """The semantics every rule above is measured against. C integer division truncates, so a
    negative numerator does not round down -- which is why the split is gated on the numerator being
    nonnegative rather than taken unconditionally."""
    assert divide_in_a_program('negative', '3 - 2*P', 5) == -1, 'int_floor is C truncation, not floor'
    # Positive control: where truncation and flooring agree, the same path must still be right.
    assert divide_in_a_program('positive', '2*P + 3', 5) == 2


if __name__ == '__main__':
    test_unit_denominator_folds_away()
    for fn in (int_floor, int_ceil):
        test_exact_division_yields_the_quotient(fn)
        test_inexact_division_is_left_symbolic(fn)
    for args in [(17, 8, 3), (16, 8, 2), (1, 8, 1), (0, 8, 0)]:
        test_numeric_operands_fold(*args)
    test_a_numerator_that_can_go_negative_is_never_split()
    test_a_strided_slice_extent_folds_to_its_trip_count()
    test_int_floor_truncates_toward_zero_in_the_generated_code()
    test_ceiling_of_index_arithmetic_is_integer_typed()
    test_ceiling_of_a_float_valued_call_on_integer_symbols_stays_floating()
