"""Typed single-precision (fp32) constant support.

Builds on the typed-constant / typed-symbol work (#2366): a
``symbolic.TypedConstant`` carries its NumPy dtype, so an fp32 constant stays
single precision through type inference, serialization, and code generation
instead of being silently widened to fp64.  These tests pin that for the
``np.float32`` case specifically.
"""
import numpy as np
import pytest

from dace import dtypes, symbolic
from dace.sdfg import type_inference


def test_fp32_constant_infers_float32():
    """A bare fp32 constant infers as ``float32`` (not widened to ``float64``)."""
    assert symbolic.TypedConstant(np.float32(2.5)).dtype == dtypes.float32
    assert type_inference.infer_expr_type(symbolic.TypedConstant(np.float32(2.5))) == dtypes.float32


def test_fp32_constant_plus_fp32_symbol_stays_fp32():
    """fp32 constant combined with an fp32 symbol stays single precision."""
    expr = symbolic.symbol('x', dtype=dtypes.float32) + symbolic.TypedConstant(np.float32(1.5))
    assert type_inference.infer_expr_type(expr, {'x': dtypes.float32}) == dtypes.float32


def test_fp32_constant_plus_fp64_symbol_promotes():
    """fp32 constant combined with an fp64 symbol promotes to double."""
    expr = symbolic.symbol('x', dtype=dtypes.float64) + symbolic.TypedConstant(np.float32(1.5))
    assert type_inference.infer_expr_type(expr, {'x': dtypes.float64}) == dtypes.float64


def test_fp32_constant_plus_int_stays_fp32():
    """fp32 constant combined with an integer symbol stays fp32."""
    expr = symbolic.symbol('n', dtype=dtypes.int32) + symbolic.TypedConstant(np.float32(2.0))
    assert type_inference.infer_expr_type(expr, {'n': dtypes.int32}) == dtypes.float32


def test_fp32_constant_serialization_roundtrip_preserves_float32():
    """An fp32 constant serializes with the ``f32`` suffix and round-trips back
    to a ``TypedConstant`` of ``float32`` -- not float64."""
    expr = symbolic.TypedConstant(np.float32(0.1))

    serialized = symbolic.serialize_symbolic(expr)
    assert serialized.endswith('f32')

    restored = symbolic.deserialize_symbolic(serialized)
    atoms = [a for a in restored.atoms() if isinstance(a, symbolic.TypedConstant)]
    assert len(atoms) == 1
    assert atoms[0].dtype == dtypes.float32


def test_fp32_constant_codegen_emits_float_literal():
    """An fp32 constant emits a C ``float`` literal (``f`` suffix), so the
    generated arithmetic stays single precision."""
    assert symbolic.symstr(symbolic.TypedConstant(np.float32(2.5)), cpp_mode=True) == '2.5f'


if __name__ == '__main__':
    pytest.main([__file__])
