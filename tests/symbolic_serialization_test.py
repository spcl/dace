# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

import dace
from dace import subsets, symbolic
from dace.codegen.common import sym2cpp
from dace.properties import DictProperty, ListProperty
from dace.sdfg.infer_types import infer_connector_types
import sympy


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

    restored = subsets.Range.from_json(rng.to_json())
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
    serialized = 'symbol($i, dtype=dace.int64, nonnegative=True)'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_same_name_symbols_with_different_dtypes_serialize_independently():
    typed = symbolic.symbol('i', dtype=dace.int64)
    default = symbolic.symbol('i')

    assert symbolic.serialize_symbolic(typed) == 'symbol($i, dtype=dace.int64)'
    assert symbolic.serialize_symbolic(default) == '$i'


def test_typed_symbol_deserialization_does_not_strip_dtype():
    serialized = '2*(1 + symbol($M, dtype=dace.int64))*symbol($M, dtype=dace.int64)'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_rational_addition_roundtrip_preserves_serialization():
    serialized = '-10/3 + 1/3*$i'

    restored = symbolic.deserialize_symbolic(serialized)

    assert symbolic.serialize_symbolic(restored) == serialized


def test_pystr_to_symbolic_preserves_typed_symbols():
    # Prime the parser cache with an equal untyped SymPy expression.
    cache_seed_expr = symbolic.deserialize_symbolic('-1 + $N')
    cache_seed_restored = symbolic.pystr_to_symbolic(cache_seed_expr)
    typed_expr = symbolic.deserialize_symbolic('-1 + symbol($N, dtype=dace.int64)')

    typed_restored = symbolic.pystr_to_symbolic(typed_expr)

    assert cache_seed_restored is cache_seed_expr
    assert symbolic.serialize_symbolic(cache_seed_restored) == '-1 + $N'
    assert typed_restored is typed_expr
    assert symbolic.serialize_symbolic(typed_restored) == '-1 + symbol($N, dtype=dace.int64)'


def test_power_deserialization_preserves_typed_symbols_after_plain_power():
    plain_power = symbolic.deserialize_symbolic('$N**2')

    typed_power = symbolic.deserialize_symbolic('symbol($N, dtype=dace.int64)**2')

    assert symbolic.serialize_symbolic(plain_power) == '$N**2'
    assert symbolic.serialize_symbolic(typed_power) == 'symbol($N, dtype=dace.int64)**2'


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
            'end': '-1 + symbol($N, dtype=dace.int64)',
            'step': '1',
            'tile': '1',
        }],
    }

    rng = subsets.Range.from_json(json_range)

    assert symbolic.serialize_symbolic(rng.ranges[0][0]) == '0'
    assert symbolic.serialize_symbolic(rng.ranges[0][1]) == '-1 + symbol($N, dtype=dace.int64)'
    assert symbolic.serialize_symbolic(rng.ranges[0][2]) == '1'
    assert symbolic.serialize_symbolic(rng.tile_sizes[0]) == '1'


def test_scalar_memlet_connector_type_after_symbolic_range_roundtrip():
    i = symbolic.symbol('i', dtype=dace.int64)
    stencil_i = symbolic.symbol('stencil_i')
    start = 2 * i - 2 * stencil_i + 1
    rng = subsets.Range([(start, start, 1)])
    restored = subsets.Range.from_json(rng.to_json())
    restored.replace({i: stencil_i})

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

    restored = prop.from_json(prop.to_json(['START']))

    assert len(restored) == 1
    assert restored[0] == symbolic.symbol('START')


def test_dict_property_symbolic_type_json_roundtrip_supports_plain_names():
    prop = DictProperty(key_type=str, value_type=sympy.Basic)

    assert prop.to_json({'N': 'N'}) == {'N': '$N'}

    restored = prop.from_json(prop.to_json({'N': 'N'}))

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


if __name__ == '__main__':
    pytest.main([__file__])
