# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def reassign(a: dace.float64[1]):
    s = 0.0
    s = 1.0
    a = s


def test():
    a = np.random.rand(1).astype(np.float64)
    reassign(a)
    assert (a == 1.0)


if __name__ == '__main__':
    test()
