# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The symbolic printer under both dialects, and the memoization that sits under it.

``symstr`` and ``_sym2cpp`` are both ``lru_cache``d. If the dialect did not reach their keys, a
``RUNTIME`` call would warm the cache and a later ``STANDALONE`` call for the same expression would
read that entry back -- emitting ``dace::math::pow`` into a translation unit that includes no DaCe
header. The failure is order-dependent and silent, so the tests here drive the same expression in
BOTH orders and assert the two dialects never collapse onto one answer.
"""
import dace
import sympy

import pytest

from dace import mpr_lowering, symbolic
from dace.codegen.common import sym2cpp
from dace.mpr_lowering import Dialect
from tests.codegen.mpr.conftest import assert_standalone

X = sympy.Symbol('x')
Y = sympy.Symbol('y')

#: ``(label, expression)`` whose printed form MUST differ between the two dialects.
DIVERGING = [
    ('pow', X**Y),
    ('sqrt', sympy.sqrt(X)),
    ('abs', sympy.Abs(X)),
    ('max', sympy.Max(X, Y)),
    ('min', sympy.Min(X, Y)),
    ('reciprocal', X**-2),
    ('floor_of_sin', sympy.floor(sympy.sin(X))),
    ('typecast', symbolic.int32(X)),
    ('ite', symbolic.ITE(X > 0, X, Y)),
    ('conj', symbolic.conj(X)),
]


def printed(expression, dialect):
    """``expression`` in C++ under ``dialect``."""
    return symbolic.symstr(expression, cpp_mode=True, dialect=dialect)


@pytest.mark.parametrize('label,expression', DIVERGING, ids=[label for label, _ in DIVERGING])
def test_standalone_output_is_free_of_runtime_names(label, expression):
    """Whatever the runtime dialect emits, the standalone one names only the standard library."""
    assert_standalone(printed(expression, Dialect.STANDALONE), label=label)


@pytest.mark.parametrize('label,expression', DIVERGING, ids=[label for label, _ in DIVERGING])
def test_the_two_dialects_do_not_collapse(label, expression):
    """The dialects genuinely differ here, so a cache that mixed them would be observable."""
    assert printed(expression, Dialect.RUNTIME) != printed(expression, Dialect.STANDALONE)


@pytest.mark.parametrize('first', [Dialect.RUNTIME, Dialect.STANDALONE], ids=['runtime-first', 'standalone-first'])
def test_memoization_does_not_mix_dialects(first):
    """Priming the cache with one dialect must not change what the other returns.

    Driven in both orders: a missing key shows up as the SECOND call echoing the first, and which
    call is wrong depends on which ran first.
    """
    second = Dialect.STANDALONE if first is Dialect.RUNTIME else Dialect.RUNTIME
    expected = {dialect: printed(sympy.Abs(X * 3 + 1), dialect) for dialect in (first, second)}
    symbolic.symstr.cache_clear()
    primed_first = printed(sympy.Abs(X * 3 + 1), first)
    primed_second = printed(sympy.Abs(X * 3 + 1), second)
    assert primed_first == expected[first], f'{first} changed after clearing the cache'
    assert primed_second == expected[second], (f'{second} came back as {primed_second!r} after {first} primed the '
                                               f'cache, but should be {expected[second]!r}; the dialect is not in '
                                               'the memoization key')


@pytest.mark.parametrize('first', [Dialect.RUNTIME, Dialect.STANDALONE], ids=['runtime-first', 'standalone-first'])
def test_memoization_does_not_mix_dialects_taken_from_the_scope(first):
    """The same, for the callers that do NOT name a dialect -- which is nearly all of them.

    ``symstr``'s several hundred call sites in the code generators pass no dialect and take the
    ambient one from :func:`~dace.mpr_lowering.dialect_scope`. That value has to be resolved in
    FRONT of the memoized body: resolved inside it, every ambient-dialect caller shares the one key
    ``dialect=None``, and an MPR rendering earlier in the process serves ``std::exp`` back to the
    runtime printer -- the exact spelling that is ambiguous for a 16-bit float. The explicit-dialect
    test above cannot see this: it never uses the key that collides.
    """
    second = Dialect.STANDALONE if first is Dialect.RUNTIME else Dialect.RUNTIME
    expression = sympy.Abs(X * 5 + 2)
    expected = {dialect: printed(expression, dialect) for dialect in (first, second)}
    symbolic.symstr.cache_clear()
    with mpr_lowering.dialect_scope(first):
        primed_first = symbolic.symstr(expression, cpp_mode=True)
    with mpr_lowering.dialect_scope(second):
        primed_second = symbolic.symstr(expression, cpp_mode=True)
    assert primed_first == expected[first], f'{first} changed when taken from the scope'
    assert primed_second == expected[second], (f'{second} came back as {primed_second!r} after {first} primed the '
                                               f'cache from the ambient dialect, but should be '
                                               f'{expected[second]!r}; the ambient dialect is resolved inside the '
                                               'memoized call rather than in front of it')


@pytest.mark.parametrize('first', [Dialect.RUNTIME, Dialect.STANDALONE], ids=['runtime-first', 'standalone-first'])
def test_sym2cpp_memoization_does_not_mix_dialects(first):
    """The same, one level up: ``sym2cpp`` has its own cache in front of ``symstr``."""
    second = Dialect.STANDALONE if first is Dialect.RUNTIME else Dialect.RUNTIME
    expression = sympy.Max(X, Y)
    primed_first = sym2cpp(expression, dialect=first)
    primed_second = sym2cpp(expression, dialect=second)
    assert primed_first != primed_second, (f'sym2cpp returned {primed_first!r} for both dialects with {first} first; '
                                           'the dialect is not in the _sym2cpp key')
    runtime, standalone = (primed_first, primed_second) if first is Dialect.RUNTIME else (primed_second, primed_first)
    # Max is an UNQUALIFIED runtime global, so the runtime spelling carries no ``dace::`` marker
    # at all -- which is why the harness checks these by name rather than by namespace.
    assert runtime.startswith('Max('), runtime
    assert standalone.startswith('mpr_max('), standalone


def test_sym2cpp_defaults_to_the_runtime_dialect():
    """An unannotated call keeps today's behaviour, so no existing caller changes meaning."""
    assert sym2cpp(sympy.Abs(X)) == sym2cpp(sympy.Abs(X), dialect=Dialect.RUNTIME)


#: ``(dace type, C++ spelling, C spelling)``. Only the complex widths differ between the dialects:
#: C++ has a class template, C a type qualifier, and nothing else in the table has two spellings.
CTYPES = [
    ('dace::complex64', 'std::complex<float>', 'float _Complex'),
    ('dace::complex128', 'std::complex<double>', 'double _Complex'),
    ('dace::uint', 'uint32_t', 'uint32_t'),
    ('dace::int64', 'int64_t', 'int64_t'),
    ('double', 'double', 'double'),
    ('int32_t', 'int32_t', 'int32_t'),
]


@pytest.mark.parametrize('ctype,cxx,c', CTYPES, ids=[ctype for ctype, _, _ in CTYPES])
def test_ctype_lowering(ctype, cxx, c):
    """Only the few ``dace::``-namespaced spellings move; the already-plain ones are untouched."""
    assert mpr_lowering.ctype_for(ctype, Dialect.STANDALONE) == cxx
    assert mpr_lowering.ctype_for(ctype, Dialect.STANDALONE_C) == c


def test_a_complex_type_has_no_cpp_spelling_left_in_c():
    """``std::complex<T>`` reaches the C tables too: a library body may already have written it."""
    for cxx, c in (('std::complex<float>', 'float _Complex'), ('std::complex<double>', 'double _Complex')):
        assert mpr_lowering.ctype_for(cxx, Dialect.STANDALONE_C) == c


@pytest.mark.parametrize('dialect', [Dialect.STANDALONE, Dialect.STANDALONE_C], ids=['c++', 'c'])
@pytest.mark.parametrize('ctype', sorted(mpr_lowering.UNSUPPORTED_CTYPES))
def test_unsupported_ctypes_refuse_loudly(ctype, dialect):
    """fp16/bfloat16/fp8 have no portable spelling in either language, so they are refused."""
    with pytest.raises(NotImplementedError, match='cannot emit the type'):
        mpr_lowering.ctype_for(ctype, dialect)


@pytest.mark.parametrize('dialect', [Dialect.STANDALONE, Dialect.STANDALONE_C], ids=['c++', 'c'])
def test_helpers_used_finds_calls_and_ignores_definitions(dialect):
    """The helper scan sees a call, and is not confused by a substring of a longer name."""
    assert mpr_lowering.helpers_used('y = int_ceil(a, b) + mod(c, d);', dialect) == {'int_ceil', 'mod'}
    assert mpr_lowering.helpers_used('y = my_mod(c, d) + a.mod(e);', dialect) == set()


def test_helpers_used_finds_the_c_dispatch_macros():
    """The C macros go through the same scan, which is what makes the preamble carry them."""
    assert mpr_lowering.helpers_used('y = mpr_sqrt(x);', Dialect.STANDALONE_C) == {'mpr_sqrt'}
    assert mpr_lowering.helpers_used('y = mpr_sqrt(x);', Dialect.STANDALONE) == set()


@pytest.mark.parametrize('dialect', [Dialect.STANDALONE, Dialect.STANDALONE_C], ids=['c++', 'c'])
def test_helpers_used_drives_the_definitions_a_unit_needs(dialect):
    """Scanning emitted text yields exactly the definitions that text requires, callees included."""
    code = printed(symbolic.mod(X, Y), dialect)
    assert mpr_lowering.definitions_for(mpr_lowering.helpers_used(code, dialect), dialect)


#: ``(label, expression, inferred ctype)``. A sympy ``Rational`` is an exact fraction of two
#: INTEGERS, and C++ truncates integer division, so ``x + 1/2`` reaches the compiler as ``x + 0``
#: unless the surrounding floating type is known. The type is read off the expression's own atoms.
INFERENCE = [
    ('int_symbol', symbolic.symbol('n_i') // 8, None),
    ('int_floor', sympy.floor(symbolic.symbol('n_j') / 8), None),
    ('fp32_symbol', symbolic.symbol('x_f', dace.float32) + sympy.Rational(1, 2), 'float'),
    ('fp64_symbol', symbolic.symbol('x_d', dace.float64) + sympy.Rational(1, 2), 'double'),
    ('float_literal', X * sympy.Float(2.5), 'double'),
    ('no_typed_atom', X + Y, None),
]


@pytest.mark.parametrize('label,expression,expected', INFERENCE, ids=[label for label, _, _ in INFERENCE])
def test_fp_type_is_inferred_from_the_expression(label, expression, expected):
    """The floating type comes from the expression's own atoms, with no symbol table threaded in."""
    assert symbolic.infer_fp_ctype(expression) == expected


def test_integer_index_arithmetic_keeps_integer_division():
    """An all-integer expression must NOT become floating: this is what subscripts are made of."""
    n = symbolic.symbol('n_k')
    for expression in (n // 8, sympy.floor(n / 8), symbolic.int_floor(n, 8)):
        emitted = printed(expression, Dialect.STANDALONE)
        assert '/ (8)' in emitted or '/(8)' in emitted, emitted
        assert 'float' not in emitted and 'double' not in emitted, emitted


@pytest.mark.parametrize('ctype,dace_type', [('float', dace.float32), ('double', dace.float64)], ids=['fp32', 'fp64'])
def test_rational_is_emitted_in_the_expressions_own_precision(ctype, dace_type):
    """fp32 stays fp32: a double literal inside a float kernel widens the whole computation."""
    x = symbolic.symbol('x_%s' % ctype, dace_type)
    emitted = printed(x + sympy.Rational(1, 2), Dialect.STANDALONE)
    assert '%s(1) / %s(2)' % (ctype, ctype) in emitted, emitted
    other = 'double' if ctype == 'float' else 'float'
    assert other not in emitted, emitted


def test_explicit_fp_ctype_overrides_the_inference():
    """A caller that knows the surrounding type better than the expression does still wins."""
    emitted = symbolic.symstr(X + sympy.Rational(1, 2), cpp_mode=True, fp_ctype='float')
    assert 'float(1) / float(2)' in emitted, emitted


@pytest.mark.parametrize('first', ['float', 'double'], ids=['fp32-first', 'fp64-first'])
def test_memoization_does_not_mix_precisions(first):
    """``fp_ctype`` reaches the cache key too, or one precision would serve the other's answer."""
    second = 'double' if first == 'float' else 'float'
    expression = X + sympy.Rational(1, 3)
    primed_first = symbolic.symstr(expression, cpp_mode=True, fp_ctype=first)
    primed_second = symbolic.symstr(expression, cpp_mode=True, fp_ctype=second)
    assert first in primed_first and second not in primed_first, primed_first
    assert second in primed_second and first not in primed_second, primed_second
