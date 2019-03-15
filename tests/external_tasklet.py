#!/usr/bin/env python
import dace


@dace.program(dace.uint32[2], dace.uint32[2])
def external_tasklet(A, B):
    @dace.map(_[1:2])
    def index(i):
        a << A[i]
        b >> B[i]
        b = a + 1  # Will fail if not replaced by the optscript

    @dace.tasklet('CPP', global_code='#include <cstdio>')
    def index2():
        a << A[0]
        b >> B[0]
        """
        b = a;
        printf("I have also been injected as raw C++ code\\n");
        """


if __name__ == '__main__':

    A = dace.ndarray((2, ))
    B = dace.ndarray((2, ))

    A[:] = 5
    B[:] = 0

    external_tasklet(A, B)

    if B[0] != 5 or B[1] != 5:
        raise RuntimeError("Expected output {}, got {}".format(5, 0))
