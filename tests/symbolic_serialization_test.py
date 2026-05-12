import numpy as np
import pytest

import dace
from dace import subsets, symbolic
from dace.codegen.common import sym2cpp


def test_symbolic_serialization_roundtrip_preserves_metadata():
    metadata_sym = symbolic.symbol('test_sym', dtype=dace.uint64, nonnegative=True)
    expr = symbolic.SymExpr(metadata_sym + symbolic.TypedConstant(np.int16(2)),
                            metadata_sym + symbolic.TypedConstant(np.int16(4)))

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


if __name__ == '__main__':
    pytest.main([__file__])
