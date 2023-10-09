# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest


def test_type_statement():

    @dace.program
    def type_statement():
        type Scalar[T] = T
        A: Scalar[dace.float32] = 0
        return A
    
    with pytest.raises(dace.frontend.python.common.DaceSyntaxError):
        type_statement()


if __name__ == '__main__':
    test_type_statement()
