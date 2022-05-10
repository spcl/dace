# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program(dace.uint32[2], dace.uint32[1])
def cpp_tasklet(A, B):
    @dace.tasklet('CPP')
    def index2():
        a << A[0]
        b >> B[0]
        """
        b = a;
        printf("I have been added as raw C++ code\\n");
        """


def test_simple():
    A = dace.ndarray((2, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 0

    cpp_tasklet(A, B)

    assert B[0] == 5


def test_stream():
    @dace
    def pbfcpp(a: dace.float64[20]):
        b = dace.define_stream(dace.float64)
        with dace.tasklet(dace.Language.CPP):
            out >> b
            """
            out.push(11.0);
            """
        b >> a

    a = np.random.rand(20)
    pbfcpp(a)
    assert a[0] == 11.0


if __name__ == "__main__":
    test_simple()
    test_stream()
