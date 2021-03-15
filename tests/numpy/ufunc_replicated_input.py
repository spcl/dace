# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math
import numpy as np
import pytest
from common import compare_numpy_output


@compare_numpy_output(non_zero=True, check_dtype=True)
def test_square_as_multiply(A: dace.complex64[10], I: dace.bool_[10]):
    np.multiply(A, A, where=I)


if __name__ == "__main__":
    test_square_as_multiply()
