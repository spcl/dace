# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest


# TODO: Investigate why pytest parses the DaCeProgram, even when the test is not supposed to run.
# @pytest.mark.py312
# def test_type_statement():

#     @dace.program
#     def type_statement():
#         type Scalar[T] = T
#         A: Scalar[dace.float32] = 0
#         return A
    
#     with pytest.raises(dace.frontend.python.common.DaceSyntaxError):
#         type_statement()


if __name__ == '__main__':
    # test_type_statement()
    pass
