# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from common import compare_numpy_output


@compare_numpy_output(check_dtype=True)
def test_dot_simple(A: dace.float32[10], B: dace.float32[10]):
    return np.dot(A, B)


if __name__ == "__main__":
    test_dot_simple()
