import numpy as np

import dace
from dace import subsets, symbolic


def test_symbolic_serialization_roundtrip_preserves_metadata():
    sym = symbolic.symbol('sin', dtype=dace.uint64, nonnegative=True)
    expr = symbolic.SymExpr(sym + symbolic.TypedConstant(np.int16(2)), sym + symbolic.TypedConstant(np.int16(4)))

    serialized = symbolic.serialize_symbolic(expr)
    restored = symbolic.deserialize_symbolic(serialized)

    assert serialized.startswith('SymExpr(')
    assert '$sin' in serialized
    assert isinstance(restored, symbolic.SymExpr)

    restored_sym = next(iter(restored.expr.free_symbols))
    assert isinstance(restored_sym, symbolic.symbol)
    assert restored_sym.name == 'sin'
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
