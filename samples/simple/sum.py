# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import sys

N = dace.symbol('N')


@dace.program
def sum(A: dace.float32[N], out: dace.float32[1]):
    dace.reduce(lambda a, b: a + b, A, out, identity=0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        N.set(int(sys.argv[1]))
    else:
        N.set(20)

    print('Sum of %d elements' % N.get())

    A = np.random.rand(N.get()).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)

    sum(A, out)

    diff = abs(out - np.sum(A))
    print("Difference:", diff)
    exit(1 if diff > 1e-5 else 0)
