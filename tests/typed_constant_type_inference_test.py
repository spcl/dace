import numpy as np
import pytest

from dace import dtypes, symbolic
from dace.sdfg import type_inference


def test_infer_expr_type_of_typed_constant():
    expr = symbolic.TypedConstant(np.int16(2))

    inferred = type_inference.infer_expr_type(expr)

    assert inferred == dtypes.int16


def test_infer_expr_type_with_typed_constant_expression():
    expr = symbolic.symbol('N', dtype=dtypes.int32) + symbolic.TypedConstant(np.uint64(1))

    inferred = type_inference.infer_expr_type(expr, {'N': dtypes.int32})

    assert inferred == dtypes.uint64


if __name__ == '__main__':
    pytest.main([__file__])
