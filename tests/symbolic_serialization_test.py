# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

import dace
from dace import subsets, symbolic
from dace.codegen.common import sym2cpp
from dace.properties import DictProperty, ListProperty
from dace.sdfg.infer_types import infer_connector_types
import sympy
import json


def test_symbolic_serialization_roundtrip_preserves_metadata():
    typed_sym = symbolic.symbol('test_sym', dtype=dace.uint64, nonnegative=True)
    expr = symbolic.SymExpr(typed_sym + symbolic.TypedConstant(np.int16(2)),
                            typed_sym + symbolic.TypedConstant(np.int16(4)))

    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    assert serialized.startswith('SymExpr(')
    assert '$test_sym' in serialized
    assert isinstance(restored, symbolic.SymExpr)

    restored_sym = next(iter(restored.expr.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'test_sym'
    assert restored_sym.dtype == dace.uint64
    assert restored_sym.is_nonnegative

    constants = [atom for atom in restored.expr.atoms() if isinstance(atom, symbolic.TypedConstant)]
    assert len(constants) == 1
    assert constants[0].dtype == dace.int16
    assert int(constants[0]) == 2


def test_range_json_roundtrip_uses_symbolic_deserializer():
    sym = symbolic.symbol('N', dtype=dace.uint64, nonnegative=True)
    rng = subsets.Range([(symbolic.TypedConstant(np.int16(2)), sym + symbolic.TypedConstant(np.int16(6)),
                          symbolic.TypedConstant(np.int16(2)), symbolic.TypedConstant(np.uint8(4)))])

    restored = subsets.Range.from_json(rng.to_json(), {"version": dace.__version__})
    start, end, step = restored.ranges[0]
    tile = restored.tile_sizes[0]

    assert isinstance(start, symbolic.TypedConstant)
    assert start.dtype == dace.int16
    assert int(start) == 2

    restored_sym = next(iter(end.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.dtype == dace.uint64
    assert restored_sym.is_nonnegative

    assert isinstance(step, symbolic.TypedConstant)
    assert step.dtype == dace.int16
    assert isinstance(tile, symbolic.TypedConstant)
    assert tile.dtype == dace.uint8


def test_symstr_codegen_for_typed_constants():
    expr = symbolic.deserialize_symbolic('2i16 + $N')

    python_expr = symbolic.symstr(expr)
    cpp_expr = symbolic.symstr(expr, cpp_mode=True)

    assert 'dace.int16(2)' in python_expr
    assert 'int16_t(2)' in cpp_expr


def test_symbol_name_clash_roundtrip():
    expr = symbolic.symbol('sin', dtype=dace.uint64, nonnegative=True) + symbolic.TypedConstant(np.uint64(1))

    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    restored_sym = next(iter(restored.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'sin'
    assert restored_sym.dtype == dace.uint64


def test_complex_symbol_roundtrip_preserves_dtype():
    expr = symbolic.symbol('c', dtype=dace.complex128)

    restored = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(expr))

    restored_sym = next(iter(restored.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'c'
    assert restored_sym.dtype == dace.complex128


@pytest.mark.parametrize('dtype', [dace.complex64, dace.complex128])
def test_suffixless_dtype_constant_roundtrips_via_cast_form(dtype):
    """A constant whose dtype has no literal suffix (complex) serializes
    via the parseable ``dace.<dtype>(value)`` form and round-trips."""
    tc = symbolic.deserialize_symbolic(f'dace.{dtype.to_string()}(5)')
    assert isinstance(tc, symbolic.TypedConstant) and tc.dtype == dtype

    serialized = symbolic.serialize_symbolic(tc)
    restored = symbolic.deserialize_symbolic(serialized)
    assert isinstance(restored, symbolic.TypedConstant)
    assert restored.dtype == dtype and int(restored.value) == 5
    assert symbolic.serialize_symbolic(restored) == serialized


def test_complex_literal_parses_as_complex128():
    tc = symbolic.deserialize_symbolic('4j')
    assert isinstance(tc, symbolic.TypedConstant) and tc.dtype == dace.complex128
    assert complex(sympy.re(tc.value), sympy.im(tc.value)) == complex(0, 4)


def test_complex_constant_parse_save_roundtrip():
    tc = symbolic.deserialize_symbolic('complex(3.0, 4.2)')
    assert isinstance(tc, symbolic.TypedConstant) and tc.dtype == dace.complex128
    assert complex(sympy.re(tc.value), sympy.im(tc.value)) == complex(3.0, 4.2)

    serialized = symbolic.serialize_symbolic(tc)
    assert serialized == '(3.0 + 4.2j)c128'
    restored = symbolic.deserialize_symbolic(serialized)
    assert restored.dtype == dace.complex128
    assert symbolic.serialize_symbolic(restored) == serialized

    c64 = symbolic.TypedConstant(complex(1, 2), dace.complex64)
    s64 = symbolic.serialize_symbolic(c64)
    assert s64 == '(1.0 + 2.0j)c64'
    r64 = symbolic.deserialize_symbolic(s64)
    assert r64.dtype == dace.complex64 and symbolic.serialize_symbolic(r64) == s64


def test_sym2cpp_emits_uint64_literals():
    expr = symbolic.TypedConstant(np.uint64(1)) + symbolic.symbol('N', dtype=dace.uint64)

    cpp_expr = sym2cpp(expr)

    assert '1ULL' in cpp_expr


def test_sym2cpp_emits_float32_literals():
    expr = symbolic.TypedConstant(np.float32(1.5)) + symbolic.symbol('N', dtype=dace.float32)

    cpp_expr = sym2cpp(expr)

    assert '1.5f' in cpp_expr


def test_undefined_symbol_serialization_uses_dollar_prefix():
    undefined = symbolic.UndefinedSymbol()

    serialized = symbolic.serialize_symbolic(undefined)
    restored = symbolic.deserialize_symbolic(serialized)

    assert serialized == '$?'
    assert isinstance(restored, symbolic.UndefinedSymbol)


def test_symbol_assumption_roundtrip_preserves_bool_metadata():
    serialized = 'symbol($i, nonnegative=True)'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_same_name_symbols_with_different_dtypes_serialize_independently():
    typed = symbolic.symbol('i', dtype=dace.int16)
    default = symbolic.symbol('i')

    assert symbolic.serialize_symbolic(typed) == 'symbol($i, dtype=dace.int16)'
    assert symbolic.serialize_symbolic(default) == '$i'


def test_typed_symbol_deserialization_does_not_strip_dtype():
    # Input may not be in canonical order, but dtype must survive a round-trip
    original = '2*(1 + symbol($M, dtype=dace.int16))*symbol($M, dtype=dace.int16)'
    restored = symbolic.deserialize_symbolic(original)
    reserialized = symbolic.serialize_symbolic(restored)

    # Dtype still appears in the output
    assert 'symbol($M, dtype=dace.int16)' in reserialized

    # The round-trip is now a fixed point (deterministic)
    assert symbolic.serialize_symbolic(symbolic.deserialize_symbolic(reserialized)) == reserialized

    # Also verify the symbol object itself has the correct dtype
    from dace import int16
    syms = {s.name: s for s in restored.free_symbols}
    assert 'M' in syms and syms['M'].dtype == int16


def test_rational_addition_roundtrip_preserves_serialization():
    serialized = '-10/3 + 1/3*$i'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_pystr_to_symbolic_preserves_typed_symbols():
    # Prime the parser cache with an equal untyped SymPy expression.
    cache_seed_expr = symbolic.deserialize_symbolic('-1 + $N')
    cache_seed_restored = symbolic.pystr_to_symbolic(cache_seed_expr)
    typed_expr = symbolic.deserialize_symbolic('-1 + symbol($N, dtype=dace.int16)')

    typed_restored = symbolic.pystr_to_symbolic(typed_expr)

    assert cache_seed_restored is cache_seed_expr
    assert symbolic.serialize_symbolic(cache_seed_restored) == '-1 + $N'
    assert typed_restored is typed_expr
    assert symbolic.serialize_symbolic(typed_restored) == '-1 + symbol($N, dtype=dace.int16)'


def test_power_deserialization_preserves_typed_symbols_after_plain_power():
    plain_power = symbolic.deserialize_symbolic('$N**2')

    typed_power = symbolic.deserialize_symbolic('symbol($N, dtype=dace.int16)**2')

    assert symbolic.serialize_symbolic(plain_power) == '$N**2'
    assert symbolic.serialize_symbolic(typed_power) == 'symbol($N, dtype=dace.int16)**2'


@pytest.mark.parametrize('simplify', [None, False])
def test_pystr_to_symbolic_keeps_basic_unsimplified_by_default(simplify):
    expr = sympy.Add(symbolic.symbol('N'), 1, 1, evaluate=False)

    restored = symbolic.pystr_to_symbolic(expr, simplify=simplify)

    assert restored is expr
    assert len(restored.args) == 3


def test_pystr_to_symbolic_simplifies_basic_when_requested():
    expr = sympy.Add(symbolic.symbol('N'), 1, 1, evaluate=False)

    restored = symbolic.pystr_to_symbolic(expr, simplify=True)

    assert restored == symbolic.symbol('N') + 2
    assert len(restored.args) == 2


def test_range_json_roundtrip_preserves_typed_symbol_minus_one():
    json_range = {
        'type': 'Range',
        'ranges': [{
            'start': '0',
            'end': '-1 + symbol($N, dtype=dace.int16)',
            'step': '1',
            'tile': '1',
        }],
    }

    rng = subsets.Range.from_json(json_range, {"version": dace.__version__})

    assert symbolic.serialize_symbolic(rng.ranges[0][0]) == '0'
    assert symbolic.serialize_symbolic(rng.ranges[0][1]) == '-1 + symbol($N, dtype=dace.int16)'
    assert symbolic.serialize_symbolic(rng.ranges[0][2]) == '1'
    assert symbolic.serialize_symbolic(rng.tile_sizes[0]) == '1'


def test_scalar_memlet_connector_type_after_symbolic_range_roundtrip():
    i = symbolic.symbol('i', dtype=dace.int64)
    stencil_i = symbolic.symbol('stencil_i', dtype=dace.int64)
    start = 2 * i - 2 * stencil_i + 1
    rng = subsets.Range([(start, start, 1)])
    restored = subsets.Range.from_json(rng.to_json(), {"version": dace.__version__})
    restored.replace({i: stencil_i})

    assert 'i' not in restored.free_symbols
    assert restored.ranges[0][0] == 1
    assert restored.ranges[0][1] == 1
    assert restored.num_elements() == 1

    sdfg = dace.SDFG('scalar_memlet_connector_after_symbolic_roundtrip')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_scalar('B', dace.float64)
    state = sdfg.add_state()
    read = state.add_read('A')
    tasklet = state.add_tasklet('use_scalar', {'inp'}, {'out'}, 'out = inp')
    write = state.add_write('B')
    state.add_edge(read, None, tasklet, 'inp', dace.Memlet(data='A', subset=restored))
    state.add_edge(tasklet, 'out', write, None, dace.Memlet('B[0]'))

    infer_connector_types(sdfg)

    assert tasklet.in_connectors['inp'] == dace.float64


def test_list_property_symbolic_type_json_roundtrip_supports_plain_names():
    prop = ListProperty(element_type=sympy.Basic)

    assert prop.to_json(['START']) == ['$START']

    restored = prop.from_json(prop.to_json(['START']), {"version": dace.__version__})

    assert len(restored) == 1
    assert restored[0] == symbolic.symbol('START')


def test_dict_property_symbolic_type_json_roundtrip_supports_plain_names():
    prop = DictProperty(key_type=str, value_type=sympy.Basic)

    assert prop.to_json({'N': 'N'}) == {'N': '$N'}

    restored = prop.from_json(prop.to_json({'N': 'N'}), {"version": dace.__version__})

    assert restored == {'N': symbolic.symbol('N')}


@pytest.mark.parametrize('expr', [
    sympy.Add(symbolic.TypedConstant(np.int16(10)),
              sympy.Mul(sympy.S.NegativeOne, symbolic.symbol('i'), evaluate=False),
              evaluate=False),
    sympy.Mul(symbolic.TypedConstant(np.int16(10)), symbolic.symbol('i'), evaluate=False),
    sympy.Mul(symbolic.TypedConstant(np.int16(10)), sympy.Pow(symbolic.symbol('i'), -1, evaluate=False),
              evaluate=False),
    sympy.Pow(symbolic.TypedConstant(np.int16(10)), symbolic.symbol('i'), evaluate=False),
    sympy.Mod(symbolic.symbol('i'), symbolic.TypedConstant(np.int16(3)), evaluate=False),
])
def test_typed_binary_operator_roundtrip_preserves_serialization(expr):
    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_plain_integer_roundtrip_converts_to_sympy_integer():
    restored = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(sympy.Integer(10)))

    assert restored == 10
    assert isinstance(restored, sympy.Integer)


def test_plain_sympy_integer_serializes_without_typed_suffix():
    assert symbolic.serialize_symbolic(sympy.Integer(42)) == '42'


@pytest.mark.parametrize('value', [-7, 2**80])
def test_plain_python_integer_deserialization_uses_sympy_integer(value):
    restored = symbolic.deserialize_symbolic(value)

    assert restored == value
    assert isinstance(restored, sympy.Integer)


def test_untyped_literals_keep_plain_sympy_form():
    """Untyped literals deserialize to plain SymPy numbers (no implicit
    DaCe-typing)."""
    assert symbolic.deserialize_symbolic('5') == sympy.Integer(5)
    assert isinstance(symbolic.deserialize_symbolic('5'), sympy.Integer)
    assert isinstance(symbolic.deserialize_symbolic('5.0'), sympy.Float)


@pytest.mark.parametrize(
    'text,ctype',
    [
        ('double(5)', 'double'),  # float64 cast-wrapper form
        ('int(5)', 'int'),  # int32 cast-wrapper form
        ('float(5)', 'float'),  # float32
        ('int64_t(5)', 'int64_t'),  # explicit-suffix dtype as a ctype cast
    ])
def test_cpp_ctype_cast_parses_to_typed_constant(text, ctype):
    """``double(5)``/``int(5)`` (the C++ printer's cast fallback) must round-trip
    into a TypedConstant."""
    restored = symbolic.deserialize_symbolic(text)
    assert isinstance(restored, symbolic.TypedConstant), text
    assert restored.dtype.ctype == ctype


@pytest.mark.parametrize('op', ['Min', 'Max'])
@pytest.mark.parametrize('literal', [
    '5.0',
    '5.0f64',
    '5f32',
    '7f64',
])
def test_minmax_with_float_literal_roundtrip(op, literal):
    serialized = f'{op}({literal}, $N)'
    restored = symbolic.deserialize_symbolic(serialized)
    assert symbolic.serialize_symbolic(restored) == serialized


def test_minmax_with_ctype_cast_int_literal_normalizes_then_stable():
    first = symbolic.serialize_symbolic(symbolic.deserialize_symbolic('Max(double(7), 5.0)'))
    assert first == '7f64'
    second = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(first))
    assert first == second


@pytest.mark.parametrize('serialized', [
    '5i16',
    '7i64',
    '5f32',
    '5.0f64',
    '(3.0 + 4.0j)c128',
    '(3.0 - 4.0j)c128',
    '(-3.0 + 4.0j)c128',
    '(3.0 + 4.0j)c64',
    '(4.0j)c128',
    '(-4.0j)c128',
    '(4.0j)c64',
    '(-4.0j)c64',
])
def test_typed_constant_canonical_form_roundtrips(serialized):
    restored = symbolic.deserialize_symbolic(serialized)
    assert isinstance(restored, symbolic.TypedConstant)
    assert symbolic.serialize_symbolic(restored) == serialized


@pytest.mark.parametrize('input_form,canonical', [
    ('4j', '(4.0j)c128'),
    ('-4j', '(-4.0j)c128'),
    ('(0.0 + 4.0j)c128', '(4.0j)c128'),
    ('complex(3.0, 4.0)', '(3.0 + 4.0j)c128'),
    ('dace.complex128(complex(3.0, 4.0))', '(3.0 + 4.0j)c128'),
    ('dace.complex64(complex(3.0, 4.0))', '(3.0 + 4.0j)c64'),
    ('dace.complex128(complex(3.0, -4.0))', '(3.0 - 4.0j)c128'),
])
def test_legacy_complex_form_normalizes_to_canonical_suffix(input_form, canonical):
    first = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(input_form))
    assert first == canonical
    second = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(first))
    assert first == second


def _roundtrip(expr):
    """Return (first serialization, serialization after one parse round-trip)."""
    s1 = symbolic.serialize_symbolic(expr)
    # FIX: Use deserialize_symbolic instead of pystr_to_symbolic
    s2 = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(s1))
    return s1, s2


def _ceiling_triangular():
    tN1 = symbolic.symbol('tN1')
    tk1 = symbolic.symbol('tk1', dace.int64)
    return sympy.ceiling(sympy.Rational(1, 2) * (1 + tk1) * tN1 - sympy.Rational(1, 2) * tN1 * tk1)


def _floor_triangular():
    tN2 = symbolic.symbol('tN2')
    tk2 = symbolic.symbol('tk2', dace.int64)
    return sympy.floor(sympy.Rational(1, 2) * (1 + tk2) * tN2 - sympy.Rational(1, 2) * tN2 * tk2)


def _mixed_unevaluated_sum():
    ta1, tb1, tc1 = symbolic.symbol('ta1'), symbolic.symbol('tb1'), symbolic.symbol('tc1')
    return sympy.Rational(1, 2) * (ta1 + tb1) * tc1 - sympy.Rational(1, 2) * tc1 * tb1 + ta1 * (tb1 + 1)


def _nested_ceiling():
    tN1 = symbolic.symbol('tN3')
    tk1 = symbolic.symbol('tk3', dace.int64)
    inner = sympy.ceiling((tN1 - 1) * (1 + tk1) - tN1 * tk1)
    return sympy.ceiling(sympy.Rational(1, 2) * inner + tN1)


def _stencil_bound_min():
    tN5, tb2 = symbolic.symbol('tN4'), symbolic.symbol('tb2')
    return sympy.Min(tN5 - 1, tb2 + 31) + 1


def _max_of_integers():
    ta3 = symbolic.symbol('t_out_range_0', integer=True)
    tb3 = symbolic.symbol('t_out_range_1', integer=True)
    return sympy.Max(ta3, tb3)


CEILING_AND_SIMILAR = {
    'ceiling_triangular': _ceiling_triangular,
    'floor_triangular': _floor_triangular,
    'mixed_unevaluated_sum': _mixed_unevaluated_sum,
    'nested_ceiling': _nested_ceiling,
    'stencil_bound_min': _stencil_bound_min,
    'max_of_integers': _max_of_integers,
}


@pytest.mark.parametrize('build', CEILING_AND_SIMILAR.values(), ids=list(CEILING_AND_SIMILAR))
def test_serialization_is_fixed_point(build):
    """serialize -> parse -> serialize must be byte-identical (and idempotent)."""
    expr = build()
    s1, s2 = _roundtrip(expr)
    assert s2 == s1, f'round-trip changed the serialization:\n  first: {s1!r}\n  again: {s2!r}'
    # A second pass must not drift either.
    # FIX: Use deserialize_symbolic instead of pystr_to_symbolic
    s3 = symbolic.serialize_symbolic(symbolic.deserialize_symbolic(s2))
    assert s3 == s1


@pytest.mark.parametrize('build', CEILING_AND_SIMILAR.values(), ids=list(CEILING_AND_SIMILAR))
def test_roundtrip_preserves_free_symbol_names(build):
    expr = build()
    # FIX: Use deserialize_symbolic instead of pystr_to_symbolic
    reparsed = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(expr))
    assert {s.name for s in reparsed.free_symbols} == {s.name for s in expr.free_symbols}


def test_add_order_independent_of_arg_order():
    """Pinpoints the printer: term order must not depend on Add arg order."""
    N = symbolic.symbol('N')
    k = symbolic.symbol('k', dace.int64)
    t1 = sympy.Rational(1, 2) * (1 + k) * N
    t2 = -sympy.Rational(1, 2) * N * k
    assert (symbolic.serialize_symbolic(sympy.Add(t1, t2, evaluate=False)) == symbolic.serialize_symbolic(
        sympy.Add(t2, t1, evaluate=False)))


def test_integer_symbol_assumptions_preserved():
    """No extra explicit assumptions (e.g. commutative) may leak in on reparse."""
    expr = _max_of_integers()
    # FIX: Use deserialize_symbolic instead of pystr_to_symbolic
    reparsed = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(expr))
    assert sympy.srepr(reparsed) == sympy.srepr(expr)
    assert all(s.is_integer for s in reparsed.free_symbols)


def test_int64_symbol_dtype_preserved():
    k = symbolic.symbol('k', dace.int64)
    # FIX: Use deserialize_symbolic instead of pystr_to_symbolic
    reparsed = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(k))
    (rk, ) = reparsed.free_symbols
    assert rk.name == 'k'
    assert rk.dtype == dace.int64


def test_sdfg_json_roundtrip_is_fixed_point():
    """Faithful reproduction of the original `test.sdfg` vs `test2.sdfg` mismatch.

    A symbolic array shape serializes through the same SymbolicProperty path as
    memlet volumes / num_accesses. If `add_array` rejects the ceiling shape on
    your branch, attach `volume` to a Memlet's `volume` instead and round-trip
    that Memlet's `to_json` / `from_json`.
    """
    N = symbolic.symbol('N', dace.int64)
    k = symbolic.symbol('k', dace.int64)
    volume = sympy.ceiling(sympy.Rational(1, 2) * (1 + k) * N - sympy.Rational(1, 2) * N * k)

    sdfg = dace.SDFG('roundtrip_determinism')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_array('A', [volume], dace.float64)

    j1 = sdfg.to_json()
    j2 = dace.SDFG.from_json(j1).to_json()
    assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)


if __name__ == '__main__':
    pytest.main([__file__])
