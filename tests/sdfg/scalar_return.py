# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from typing import Tuple

from dace.sdfg.validation import InvalidSDFGError


@dace.program(auto_optimize=False, recreate_sdfg=True)
def testee(
    A: dace.float64[20],
) -> dace.float64:
    return dace.float64(A[3])

@pytest.mark.skip("Scalar return is not implement.")
def test_scalar_return():

    sdfg = testee.to_sdfg(validate=False)
    assert isinstance(sdfg.arrays["__return"], dace.data.Scalar)

    sdfg.validate()
    A = np.random.rand(20)
    res = sdfg(A=A)
    assert A[3] == res


def test_scalar_return_validation():
    """Test if the validation actually works.

    Todo:
        Remove this test after scalar return values are implemented and enable
        the `test_scalar_return` test.
    """
    with pytest.raises(
        InvalidSDFGError,
        match='Can not use scalar "__return" as return value.',
    ):
        testee.to_sdfg(validate=True)

if __name__ == '__main__':
    test_scalar_return_prog()
