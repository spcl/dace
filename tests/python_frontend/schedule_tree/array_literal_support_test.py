# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import pytest

import dace
from dace import data
from dace.frontend.python.schedule_tree.array_literal_support import infer_array_literal_descriptor


def test_infer_array_literal_descriptor_for_nested_list_constants():
    node = ast.parse('[[1, 2], [3, 4]]', mode='eval').body
    descriptor = infer_array_literal_descriptor(node, lambda _: None, lambda *_: None, lambda: {})

    assert isinstance(descriptor, data.Array)
    assert descriptor.dtype == dace.int64
    assert tuple(descriptor.shape) == (2, 2)


def test_infer_array_literal_descriptor_for_dynamic_scalar_elements():
    node = ast.parse('[A[0], B[i]]', mode='eval').body
    scalar_desc = data.Scalar(dace.float64, transient=True)

    def infer_descriptor(expr):
        text = ast.unparse(expr)
        if text in {'A[0]', 'B[i]'}:
            return scalar_desc
        return None

    descriptor = infer_array_literal_descriptor(node, infer_descriptor, lambda *_: None, lambda: {})

    assert isinstance(descriptor, data.Array)
    assert descriptor.dtype == dace.float64
    assert tuple(descriptor.shape) == (2, )


def test_infer_numpy_array_literal_descriptor_respects_dtype_and_ndmin():
    node = ast.parse('np.array([1, 2], dtype=np.float32, ndmin=2)', mode='eval').body
    context = lambda: {'np': __import__('numpy')}
    descriptor = infer_array_literal_descriptor(node, lambda _: None, lambda *_: None, context)

    assert isinstance(descriptor, data.Array)
    assert descriptor.dtype == dace.float32
    assert tuple(descriptor.shape) == (1, 2)


if __name__ == '__main__':
    pytest.main([__file__])
