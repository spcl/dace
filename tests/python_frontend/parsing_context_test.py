# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_parsing_context():
    def func(a):
        if dace.in_program():
            a[:] = 1
        else:
            a[:] = 2

    first = np.random.rand(10)
    second = np.random.rand(10)

    func(first)
    dace.program(func)(second)

    assert np.allclose(first, 2)
    assert np.allclose(second, 1)


if __name__ == '__main__':
    test_parsing_context()
