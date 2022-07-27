# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace

import dace
import numpy as np


class ArrayWrapper:
    def __init__(self, array, **kwargs):
        self.array = array

    @property
    def __array_interface__(self):
        return self.array.__array_interface__


def test_array_interface_input():
    @dace.program
    def simple_program(A: dace.float64[3, 3, 3]):
        A += 1

    simple_program.compile()

    A = np.ones((3, 3, 3))
    Awrap = ArrayWrapper(A)

    simple_program(A=Awrap)

    np.testing.assert_equal(A, 2)


if __name__ == "__main__":
    test_array_interface_input()
