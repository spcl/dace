# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def internal(out: dace.float64[2]):
    with dace.tasklet:
        o1 >> out[0]
        o2 >> out[1]
        o1 = 5
        o2 = 3


@dace.program
def inline_noinput(A: dace.float64[2]):
    for i in dace.map[0:1]:
        internal(A)


if __name__ == '__main__':
    A = np.random.rand(2)

    inline_noinput(A)

    diff = np.linalg.norm(A - np.array([5., 3.]))
    print('Difference:', diff)
    exit(0 if diff < 1e-5 else 1)
