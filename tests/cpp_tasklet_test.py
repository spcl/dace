# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace


@dace.program
def cpp_tasklet(A: dace.uint32[2], B: dace.uint32[1]):

    @dace.tasklet('CPP')
    def index2():
        a << A[0]
        b >> B[0]
        """
        b = a;
        printf("I have been added as raw C++ code\\n");
        """


def test():
    A = dace.ndarray((2, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 0

    cpp_tasklet(A, B)

    assert B[0] == 5


if __name__ == "__main__":
    test()
