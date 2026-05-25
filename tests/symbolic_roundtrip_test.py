# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Round-trip tests for symbolic expressions through ``pystr_to_symbolic`` and ``symstr``
(and through interstate-edge serialization), covering subscripts, operators, infinities,
NaN, booleans and float precision. """
import sympy

import dace
from dace import subsets, symbolic
from dace.symbolic import pystr_to_symbolic, symstr, arrays, free_symbols_and_functions, bitwise_or
from dace.frontend.python.newast import _subset_has_indirection


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


def test_array_rename_through_subscript():
    # replace / replace_dict / subs must rename an array referenced via Subscript -- the
    # GPU offloading pass relies on this (e.g. A_row -> gpu_A_row).
    e = dace.InterstateEdge(assignments={'x': 'A_row[i + 1] - A_row[i]'})
    e.replace('A_row', 'gpu_A_row')
    assert arrays(e.assignments['x']) == {'gpu_A_row'}

    e2 = dace.InterstateEdge(assignments={'x': 'A_row[i]'})
    e2.replace_dict({'A_row': 'gpu_A_row'})
    assert e2.assignments['x'] == dace.symbolic.pystr_to_symbolic('gpu_A_row[i]')

    renamed = pystr_to_symbolic('A_row[i]').subs(symbolic.symbol('A_row'), symbolic.symbol('gpu_A_row'))
    assert arrays(renamed) == {'gpu_A_row'}


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


def test_scalar_to_symbol_promotion_moves_scalar_out_of_scalars():
    # A scalar read on an interstate edge is reported by scalars(); after scalar->symbol
    # promotion it is a symbol instead, so scalars() no longer reports it (expression unchanged).
    from dace.transformation.passes import scalar_to_symbol

    sdfg = dace.SDFG('promote_scalar')
    sdfg.add_scalar('s', dace.int32, transient=True)
    sdfg.add_array('A', [20], dace.int32)
    init = sdfg.add_state('init', is_start_block=True)
    init.add_edge(init.add_tasklet('set', {}, {'o'}, 'o = 7'), 'o', init.add_access('s'), None, dace.Memlet('s[0]'))
    body = sdfg.add_state('body')
    sdfg.add_edge(init, body, dace.InterstateEdge(assignments={'i': 's'}))
    body.add_edge(body.add_tasklet('use', {}, {'o'}, 'o = i'), 'o', body.add_access('A'), None, dace.Memlet('A[0]'))
    sdfg.add_symbol('i', dace.int32)

    assign = sdfg.edges()[0].data.assignments['i']
    assert symbolic.scalars(assign, sdfg.arrays) == {'s'}  # 's' is a scalar read on the edge
    assert 's' not in sdfg.symbols

    scalar_to_symbol.ScalarToSymbolPromotion().apply_pass(sdfg, {})

    assert sdfg.edges()[0].data.assignments['i'] == dace.symbol('s')  # expression unchanged
    assert symbolic.scalars('s', sdfg.arrays) == set()  # no longer a scalar descriptor
    assert 's' in sdfg.symbols  # now a symbol


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


def test_map_range_array_bound_is_hoisted():
    # An array access in a map-range bound must be hoisted to a dynamic-map-input symbol:
    # the range then carries a symbol (not a raw Subscript) and the array is wired as a
    # read. Without this, GPU offloading can't see the array and loop_to_map refuses it.
    N = dace.symbol('N')

    @dace.program
    def maprange(ptr: dace.uint32[N + 1], data: dace.float32[64], out: dace.float32[N]):
        for i in dace.map[0:N]:

            @dace.map(_[ptr[i]:ptr[i + 1]])
            def inner(j):
                d << data[j]
                o >> out(1, lambda x, y: x + y)[i]
                o = d

    sdfg = maprange.to_sdfg()
    inner = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.map.params == ['j']]
    assert inner
    for me in inner:
        assert 'ptr[' not in str(me.map.range)  # bound hoisted to a symbol, not a raw Subscript
        assert me.map.range.free_symbols  # the range is symbolic


def test_subset_indirection_detects_index_subscript():
    # An index-position array access (a gather, ``data[A_col[j]]``) is indirection;
    # a plain index (``data[j]``) is not. With the head excluded from a Subscript's
    # free_symbols, this detection depends on contains_sympy_functions being
    # Subscript-aware.
    gather = subsets.Range.from_indices([pystr_to_symbolic('A_col[j]')])
    plain = subsets.Range.from_indices([pystr_to_symbolic('j')])
    assert _subset_has_indirection(gather)
    assert not _subset_has_indirection(plain)
    assert pystr_to_symbolic('A_col[j]').free_symbols == {symbolic.symbol('j')}


def test_subset_indirection_detects_nested_and_multidim_subscript():
    # Nested gathers (``data[A[B[j]]]``) and a per-dimension gather (``data[i, A[j]]``)
    # are both indirection; a fully plain multi-dim subset is not.
    assert _subset_has_indirection(subsets.Range.from_indices([pystr_to_symbolic('A[B[j]]')]))
    assert _subset_has_indirection(subsets.Range.from_indices([pystr_to_symbolic('i'), pystr_to_symbolic('A[j]')]))
    assert not _subset_has_indirection(subsets.Range.from_indices([pystr_to_symbolic('i'), pystr_to_symbolic('j')]))


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
    test_array_rename_through_subscript()
    test_struct_member_subscript()
    test_free_symbols_and_functions_excludes_arrays()
    test_scalar_versus_array_needs_descriptors()
    test_scalar_to_symbol_promotion_moves_scalar_out_of_scalars()
    test_contains_sympy_functions_subscript()
    test_float_precision_preserved()
    test_infinity_roundtrip()
    test_infinity_cpp_lowering()
    test_nan_roundtrip()
    test_boolean_preserved_and_distinct_from_int()
    test_map_range_array_bound_is_hoisted()
    test_subset_indirection_detects_index_subscript()
    test_subset_indirection_detects_nested_and_multidim_subscript()
    test_interstate_edge_assignment_roundtrip()
