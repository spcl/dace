# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
from common import compare_numpy_output

import dace


@compare_numpy_output(check_dtype=True)
def test_dot_simple(A: dace.float32[10], B: dace.float32[10]):
    return np.dot(A, B)


if __name__ == "__main__":
    test_dot_simple()
