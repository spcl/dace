# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def customreduction(A: dace.float32[20], out: dace.float32[1]):
    dace.reduce(lambda a, b: a if a < b else b, A, out, identity=9999999)


if __name__ == '__main__':
    print('Custom reduction test')
    A = np.random.rand(20).astype(np.float32)
    B = np.zeros([1], dtype=np.float32)
    customreduction(A, B)
    diff = (B - np.min(A))
    print('Difference:', diff)
    exit(0 if diff < 1e-5 else 1)
