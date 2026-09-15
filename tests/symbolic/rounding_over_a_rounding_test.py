# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A rounding whose DENOMINATOR is itself a rounding call.

``sympy_intdiv_fix`` normalizes a sympy ``ceiling`` / ``floor`` into DaCe's ``int_ceil`` /
``int_floor``, whose C++ spelling is an integer division that rounds the way the node names. The
arms that recognize a rounding call inside the expression have to be built from the REAL
``dace.symbolic.int_ceil`` class: ``sympy.Function('int_ceil')`` is an undefined function that
prints identically and reports the same ``.func`` name, but is a different class, so a pattern
built from it matches nothing an SDFG actually carries.

What a miss costs is not cosmetic. A surviving ``ceiling`` is printed as libm ``ceil()`` wrapped
around an argument whose ``/`` is C INTEGER division, so the rounding it names never happens -- the
division already truncated -- and the call's ``double`` result makes a map's range end a
non-integer loop predicate: ``for (auto i = 0; i < ceil(...); ...)``, which gcc rejects with
"invalid controlling predicate" instead of accepting the ``omp for``.
"""
import math

import numpy as np
import pytest
import sympy

import dace
from dace.subsets import Range
from dace.symbolic import int_ceil, int_floor, symbol, symstr, sympy_intdiv_fix

N = symbol('N')
M = symbol('M')
T = symbol('T')

#: All four combinations of an outer rounding over an inner one, with a bare symbol on top. The
#: first two are what the deleted ``ceiling(a / int_ceil(c, d))`` and ``floor(a / int_floor(c, d))``
#: arms used to carry; the last two never had an arm at all, because every dedicated arm named the
#: SAME head on both sides.
SYMBOL_NUMERATOR = [
    (sympy.ceiling(N / int_ceil(M, T)), int_ceil(N, int_ceil(M, T))),
    (sympy.floor(N / int_floor(M, T)), int_floor(N, int_floor(M, T))),
    (sympy.ceiling(N / int_floor(M, T)), int_ceil(N, int_floor(M, T))),
    (sympy.floor(N / int_ceil(M, T)), int_floor(N, int_ceil(M, T))),
]

#: The same four with a composite numerator -- a range chunked over a block size. Neither the
#: symbol-numerator shape nor the symbol-denominator shape, so these went through the deleted
#: ``e / int_ceil(c, d)`` and ``e / int_floor(c, d)`` arms, or through nothing at all.
COMPOSITE_NUMERATOR = [
    (sympy.ceiling((N - 2) / int_ceil(N - 2, T)), int_ceil(N - 2, int_ceil(N - 2, T))),
    (sympy.floor((N - 2) / int_floor(M, T)), int_floor(N - 2, int_floor(M, T))),
    (sympy.ceiling((N - 2) / int_floor(M, T)), int_ceil(N - 2, int_floor(M, T))),
    (sympy.floor((N - 2) / int_ceil(M, T)), int_floor(N - 2, int_ceil(M, T))),
]

#: Names the outer rounding then the inner one, so a failure says which pair broke.
PAIR_IDS = ['ceil_over_ceil', 'floor_over_floor', 'ceil_over_floor', 'floor_over_ceil']


@pytest.mark.parametrize('expr,expected', SYMBOL_NUMERATOR, ids=PAIR_IDS)
def test_a_rounding_denominator_is_recognized_by_class_not_by_name(expr, expected):
    """A bare symbol over a rounding call normalizes, whichever rounding sits on either side.

    A rounding call IS an integer division, so it is a denominator the normalizer can divide by
    like any symbol. Naming one head on both sides instead -- what the dedicated arms did -- covers
    only the matching pair and leaves the mixed one to survive.
    """
    normalized = sympy_intdiv_fix(expr)
    assert normalized == expected, normalized
    assert not normalized.find(sympy.ceiling), normalized
    assert not normalized.find(sympy.floor), normalized


@pytest.mark.parametrize('expr,expected', COMPOSITE_NUMERATOR, ids=PAIR_IDS)
def test_a_composite_numerator_over_a_rounding_denominator_normalizes(expr, expected):
    """The same four pairs with a numerator the symbol-only Wild cannot take."""
    normalized = sympy_intdiv_fix(expr)
    assert normalized == expected, normalized
    assert not normalized.find(sympy.ceiling), normalized
    assert not normalized.find(sympy.floor), normalized


@pytest.mark.parametrize('expr', [expr for expr, _ in SYMBOL_NUMERATOR + COMPOSITE_NUMERATOR],
                         ids=[f'{i}_{n}' for n in ('sym', 'composite') for i in PAIR_IDS])
def test_no_libm_rounding_call_reaches_cpp_for_an_integer_division(expr):
    """The emitted C++ is integer arithmetic end to end -- no ``ceil()`` / ``floor()`` call.

    Both spellings return a ``double``, and both would be applied to an argument C has already
    divided with truncation, so their presence is simultaneously a wrong value and a loop predicate
    OpenMP refuses.
    """
    emitted = symstr(sympy_intdiv_fix(expr), cpp_mode=True)
    assert 'ceil(' not in emitted.replace('int_ceil(', ''), emitted
    assert 'floor(' not in emitted.replace('int_floor(', ''), emitted


@pytest.mark.parametrize('expr', [sympy.floor(sympy.sin(N)), sympy.ceiling(sympy.sin(N))], ids=['floor', 'ceiling'])
def test_a_rounding_with_nothing_to_divide_by_is_not_an_integer_division(expr):
    """The ``b != 1`` guard on the denominator, which widening the Wild must not cost.

    A Wild that admits ``1`` matches ``floor(e / b)`` against ``floor(sin(N))`` with ``b = 1`` and
    rewrites it to ``int_floor(sin(N), 1)``, which prints as a division by one -- the rounding
    silently dropped (spcl/dace#2524). ``sin`` is a genuinely floating quantity, so the rounding
    here is real work, not a normalization artefact.
    """
    normalized = sympy_intdiv_fix(expr)
    assert normalized == expr, normalized
    assert not normalized.find(int_ceil), normalized
    assert not normalized.find(int_floor), normalized


#: Denominators the normalizer still gets wrong, with a witness point for each. Recorded rather
#: than fixed: neither has a safe ``int_ceil`` lowering as things stand. An ``Add`` denominator
#: needs its sign to be known before ``(x + y - 1) / y`` is the ceiling, and a product denominator
#: is split into a numerator that C then truncates on its own. Both are pre-existing.
KNOWN_WRONG_DENOMINATORS = [
    (sympy.ceiling(N / (M - 1)), 'Add denominator: N=5, M=4 counts 1 where the ceiling is 2'),
    (sympy.ceiling(N / (M * T)), 'product denominator: N=3, M=2, T=1 counts 1 where the ceiling is 2'),
]


@pytest.mark.parametrize('expr,witness', KNOWN_WRONG_DENOMINATORS, ids=['add_denominator', 'product_denominator'])
@pytest.mark.xfail(strict=True, reason='known-wrong denominator shapes, see KNOWN_WRONG_DENOMINATORS')
def test_a_composite_denominator_ceiling_reaches_cpp_as_integer_arithmetic(expr, witness):
    """What ``_print_ceiling`` documents as its precondition, and does not get.

    Its docstring states that only a ceiling with nothing to divide by reaches it, because
    ``sympy_intdiv_fix`` turned every real integer division into ``int_ceil``. These two shapes
    break that: one arrives as libm ``ceil()`` around a C integer division that already truncated,
    the other as an ``int_ceil`` whose numerator truncates. Both compile and run the wrong trip
    count -- silent, unlike the loud ``ceil()`` of a ``double``. Marked strict so a future fix
    reports XPASS instead of passing unnoticed.
    """
    emitted = symstr(sympy_intdiv_fix(expr), cpp_mode=True)
    assert 'ceil(' not in emitted.replace('int_ceil(', ''), f'{emitted} -- {witness}'
    assert '/' not in emitted.split('int_ceil(', 1)[-1].split(',', 1)[0], f'{emitted} -- {witness}'


def trip_count_sdfg(inner) -> dace.SDFG:
    """An SDFG summing a 1 per iteration of a map whose range end is ``ceil(N / inner(M, T))``.

    The map's end is the normalizer's OUTPUT, so what the loop counts is exactly what code
    generation made of the expression -- a value assertion on the rewrite rather than on its text.
    """
    end = sympy_intdiv_fix(sympy.ceiling(N / inner(M, T)))
    sdfg = dace.SDFG(f'trip_count_over_{inner.__name__}')
    sdfg.add_array('out', [1], dace.int64)
    for name in ('N', 'M', 'T'):
        sdfg.add_symbol(name, dace.int64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': Range([(0, end - 1, 1)])})
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1')
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, 'o', exit_node, 'IN_out', dace.Memlet('out[0]', wcr='lambda a, b: a + b'))
    exit_node.add_in_connector('IN_out')
    exit_node.add_out_connector('OUT_out')
    state.add_edge(exit_node, 'OUT_out', state.add_access('out'), None, dace.Memlet('out[0]', wcr='lambda a, b: a + b'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('inner', [int_ceil, int_floor], ids=['over_ceil', 'over_floor'])
@pytest.mark.parametrize('n,m,t', [(13, 4, 2), (100, 7, 3), (5, 4, 2), (17, 5, 2)])
def test_a_rounding_over_a_rounding_range_end_compiles_and_counts(inner, n, m, t):
    """End to end: the loop compiles as an ``omp for`` and runs the arithmetic trip count.

    A range end the normalizer did not reach arrives as ``ceil((N / int_ceil(M, T)))`` -- libm
    ``ceil`` over an argument C already truncated. gcc refuses that loop outright, "invalid
    controlling predicate", so this fails at compilation before any value is compared.
    """
    csdfg = trip_count_sdfg(inner).compile()
    out = np.zeros(1, dtype=np.int64)
    csdfg(out=out, N=n, M=m, T=t)
    divisor = -(-m // t) if inner is int_ceil else m // t
    assert out[0] == math.ceil(n / divisor)


if __name__ == '__main__':
    for case in SYMBOL_NUMERATOR:
        test_a_rounding_denominator_is_recognized_by_class_not_by_name(*case)
    for case in COMPOSITE_NUMERATOR:
        test_a_composite_numerator_over_a_rounding_denominator_normalizes(*case)
    for case in SYMBOL_NUMERATOR + COMPOSITE_NUMERATOR:
        test_no_libm_rounding_call_reaches_cpp_for_an_integer_division(case[0])
    for expr in (sympy.floor(sympy.sin(N)), sympy.ceiling(sympy.sin(N))):
        test_a_rounding_with_nothing_to_divide_by_is_not_an_integer_division(expr)
    for inner in (int_ceil, int_floor):
        test_a_rounding_over_a_rounding_range_end_compiles_and_counts(inner, 13, 4, 2)
