# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Round-trip tests for symbolic expressions through ``pystr_to_symbolic`` and ``symstr``
(and through interstate-edge serialization), covering subscripts, operators, infinities,
NaN, booleans and float precision. """
import sympy

import dace
from dace import subsets, symbolic
from dace.symbolic import pystr_to_symbolic, symstr, arrays, free_symbols_and_functions, bitwise_or


def _roundtrip(s, cpp_mode=False):
    return symstr(pystr_to_symbolic(s), cpp_mode=cpp_mode)


def _idempotent(s):
    once = symstr(pystr_to_symbolic(s))
    return once == symstr(pystr_to_symbolic(once))


def test_operator_roundtrip_renders_operator():
    # The Python operators round-trip back to operators, not to ``func(a, b)``.
    cases = {'~C': '~', 'a & b': '&', 'a | b': '|', 'a ^ b': '^', 'a << b': '<<', 'a >> b': '>>'}
    for expr, op in cases.items():
        out = _roundtrip(expr)
        assert op in out
        assert 'bitwise' not in out and 'shift' not in out
        assert _idempotent(expr)


def test_ternary_roundtrip():
    assert _roundtrip('b if c else d') == '(((b) if (c) else (d)))'
    assert _idempotent('b if c else d')


def test_floordiv_roundtrip():
    # ``//`` round-trips to ``//`` in Python and lowers to ``/`` in C++; an explicit
    # int_floor(a, b) keeps its function spelling in Python.
    assert _roundtrip('a // b') == '(((a) // (b)))'
    assert _roundtrip('a // b', cpp_mode=True) == '(((a) / (b)))'
    assert _idempotent('a // b')
    assert _roundtrip('int_floor(a, b)') == '(int_floor(a, b))'
    assert _roundtrip('int_floor(a, b)', cpp_mode=True) == '(((a) / (b)))'


def test_cpp_lowering():
    # The operators must lower to the identical C++ operators.
    assert _roundtrip('a << b', cpp_mode=True) == '(((a) << (b)))'
    assert _roundtrip('a >> b', cpp_mode=True) == '(((a) >> (b)))'
    assert _roundtrip('a & b', cpp_mode=True) == '(((a) & (b)))'
    assert _roundtrip('a | b', cpp_mode=True) == '(((a) | (b)))'
    assert _roundtrip('~c', cpp_mode=True) == '((~(c)))'
    assert _roundtrip('b if c else d', cpp_mode=True) == '(((c) ? (b) : (d)))'


def test_operator_uses_internal_variant():
    # ``|`` parses to the ``__``-prefixed internal variant (still an instance of the
    # bare class) so it round-trips to the operator.
    op = pystr_to_symbolic('a | b')
    assert str(op.func) == '__bitwise_or'
    assert isinstance(op, bitwise_or)


def test_explicit_function_name_preserved():
    # An explicit ``bitwise_or(a, b)`` keeps its spelling in Python but still lowers to
    # the operator in C++.
    assert symstr(pystr_to_symbolic('bitwise_or(a, b)')) == '(bitwise_or(a, b))'
    assert symstr(pystr_to_symbolic('bitwise_or(a, b)'), cpp_mode=True) == '(((a) | (b)))'


def test_subscript_roundtrip():
    assert _roundtrip('A[i]') == '(A[i])'
    assert _roundtrip('sizes[i, j]') == '(sizes[i, j])'
    assert _idempotent('A[i]') and _idempotent('sizes[i, j]')


def test_subscript_free_symbols_excludes_container():
    # Only the indices are free symbols; the container is a data access.
    assert {str(s) for s in pystr_to_symbolic('A[i]').free_symbols} == {'i'}
    assert {str(s) for s in pystr_to_symbolic('A[i, j]').free_symbols} == {'i', 'j'}
    # Regression guard: a numeric-indexed access must not leak the container as a symbol
    # (otherwise it gets mapped to its data descriptor and ``sympify`` raises).
    assert pystr_to_symbolic('num[0]').free_symbols == set()


def test_arrays_helper():
    assert arrays('A[i] + sin(x)') == {'A'}
    assert arrays('a[i, j, k]') == {'a'}
    assert arrays('A[B[i]]') == {'A', 'B'}
    assert arrays('a + b') == set()


def test_struct_member_subscript():
    # A struct-member array access (``a.b[i]``, nested ``A.B.C[0]``): only the indices
    # are free symbols, the full member path is reported by ``arrays``, and the struct
    # root must NOT leak into ``Range.free_symbols`` (which previously caused
    # ``SympifyError`` when parsing map ranges).
    e = pystr_to_symbolic('a.b[i]')
    assert {str(s) for s in e.free_symbols} == {'i'}
    assert arrays('a.b[i]') == {'a.b'}
    assert {str(s) for s in subsets.Range([(0, e - 1, 1)]).free_symbols} == {'i'}

    # A is a struct, B a member struct, C a member array.
    assert pystr_to_symbolic('A.B.C[0]').free_symbols == set()
    assert arrays('A.B.C[0]') == {'A.B.C'}
    assert {str(s) for s in pystr_to_symbolic('A.B.C[i]').free_symbols} == {'i'}
    assert {str(s) for s in subsets.Range([(0, pystr_to_symbolic('A.B.C[i]'), 1)]).free_symbols} == {'i'}


def test_free_symbols_and_functions_excludes_arrays():
    # Clean split: ``arrays`` reports the container, ``free_symbols_and_functions`` does not.
    assert free_symbols_and_functions('A[i] + sin(x)') == {'i', 'x'}


def test_scalar_versus_array_needs_descriptors():
    # A subscripted access is unambiguously a container; a bare name (possibly a rank-0
    # scalar) is indistinguishable from a symbol, so ``arrays`` reports only the former.
    # Consumers recover scalars by intersecting ``free_symbols_and_functions`` with the
    # SDFG's data descriptors.
    assert arrays('A[i]') == {'A'}
    assert arrays('s') == set()  # ``s`` could be a scalar or a symbol; not decidable here
    descriptors = {'A', 's'}  # both are data containers in this hypothetical SDFG
    referenced = (free_symbols_and_functions('A[i] + s') | arrays('A[i] + s')) & descriptors
    assert referenced == {'A', 's'}  # arrays() -> A; fsf & descriptors -> s


def test_contains_sympy_functions_subscript():
    assert symbolic.contains_sympy_functions(pystr_to_symbolic('A[i]')) is True
    assert symbolic.contains_sympy_functions(pystr_to_symbolic('a + b')) is False
    assert symbolic.contains_sympy_functions(pystr_to_symbolic('a | b')) is False


def test_float_precision_preserved():
    # Floats print at the shortest round-tripping precision.
    assert _roundtrip('3.14') == '3.14'
    assert _roundtrip('0.1') == '0.1'
    # A near-max double (Fortran ``HUGE``) stays finite and at full precision.
    for huge in ('1.79769313486232e+308', '1.7976931348623157e+308', '-1.7976931348623157e+308'):
        assert _roundtrip(huge) == huge
        assert pystr_to_symbolic(huge) not in (sympy.oo, -sympy.oo)


def test_infinity_roundtrip():
    assert pystr_to_symbolic('inf') == sympy.oo
    assert pystr_to_symbolic('-inf') == -sympy.oo
    assert _roundtrip('inf') == 'inf'
    assert _roundtrip('-inf') == '-inf'
    # Infinity inside an expression resolves to ``oo`` (and ``a - inf`` to ``-oo``).
    assert pystr_to_symbolic('a + inf').has(sympy.oo)
    assert pystr_to_symbolic('a - inf').has(-sympy.oo)


def test_infinity_cpp_lowering():
    assert _roundtrip('inf', cpp_mode=True) == 'INFINITY'
    assert _roundtrip('-inf', cpp_mode=True) == '-INFINITY'


def test_nan_roundtrip():
    assert pystr_to_symbolic('nan') is sympy.nan
    assert _roundtrip('nan') == 'nan'
    assert _roundtrip('nan', cpp_mode=True) == 'NAN'


def test_boolean_preserved_and_distinct_from_int():
    # Booleans round-trip as booleans and stay distinct from the integers 1/0.
    assert _roundtrip('True') == 'True'
    assert _roundtrip('False') == 'False'
    assert _roundtrip('1') == '1'
    assert _roundtrip('0') == '0'
    assert isinstance(pystr_to_symbolic('True'), sympy.logic.boolalg.BooleanTrue)
    assert _roundtrip('True', cpp_mode=True) == 'true'


def test_interstate_edge_assignment_roundtrip():
    # Assignments survive save -> load -> save unchanged (idempotent serialization).
    sdfg = dace.SDFG('iedge_roundtrip')
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(
        s0, s1, dace.InterstateEdge(assignments={
            'p': 'a | b',
            'q': 'True',
            'r': '1',
            's': '1.79769313486232e+308',
        }))

    reloaded = dace.SDFG.from_json(sdfg.to_json())
    twice = dace.SDFG.from_json(reloaded.to_json())
    assert reloaded.edges()[0].data.assignments == twice.edges()[0].data.assignments

    assigns = reloaded.edges()[0].data.assignments
    assert '|' in assigns['p'] and 'bitwise_or' not in assigns['p']
    assert assigns['q'] == 'True'  # boolean preserved, not collapsed to '1'
    assert assigns['r'] == '1'
    assert assigns['s'] == '1.79769313486232e+308'  # finite, full precision


if __name__ == '__main__':
    test_operator_roundtrip_renders_operator()
    test_ternary_roundtrip()
    test_floordiv_roundtrip()
    test_cpp_lowering()
    test_operator_uses_internal_variant()
    test_explicit_function_name_preserved()
    test_subscript_roundtrip()
    test_subscript_free_symbols_excludes_container()
    test_arrays_helper()
    test_struct_member_subscript()
    test_free_symbols_and_functions_excludes_arrays()
    test_scalar_versus_array_needs_descriptors()
    test_contains_sympy_functions_subscript()
    test_float_precision_preserved()
    test_infinity_roundtrip()
    test_infinity_cpp_lowering()
    test_nan_roundtrip()
    test_boolean_preserved_and_distinct_from_int()
    test_interstate_edge_assignment_roundtrip()
