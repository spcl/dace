import numpy as np
import pytest

from dace import dtypes, symbolic
from dace.sdfg import type_inference


def test_infer_expr_type_of_typed_constant():
    expr = symbolic.TypedConstant(np.int16(2))

    inferred = type_inference.infer_expr_type(expr)

    assert inferred == dtypes.int16


def test_infer_expr_type_with_typed_constant_expression():
    expr = symbolic.symbol("N", dtype=dtypes.int32) + symbolic.TypedConstant(np.uint64(1))

    inferred = type_inference.infer_expr_type(expr, {"N": dtypes.int32})

    assert inferred == dtypes.uint64


def test_same_value_typed_constants_of_two_dtypes_order_in_a_sum():
    narrow = symbolic.TypedConstant(1, dtypes.int32)
    wide = symbolic.TypedConstant(1, dtypes.int64)

    total = symbolic.symbol("x") + narrow + wide

    assert narrow in total.args and wide in total.args


if __name__ == "__main__":
    pytest.main([__file__])
