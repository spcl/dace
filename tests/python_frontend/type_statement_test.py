# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
import sys

# Check if Python version is 3.12 or higher
if sys.version_info >= (3, 12):
    def test_type_statement():

        @dace.program
        def type_statement():
            type Scalar[T] = T
            A: Scalar[dace.float32] = 0
            return A

        with pytest.raises(dace.frontend.python.common.DaceSyntaxError):
            type_statement()


if __name__ == '__main__':
    if sys.version_info >= (3, 12):
        test_type_statement()
