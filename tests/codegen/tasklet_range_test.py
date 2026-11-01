# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loops over ``range`` in tasklets follow Python: they end once the index reaches or passes the end."""

import numpy as np

import dace


@dace.program
def strided_sums(out: dace.int64[4, 4]):
    for i in dace.map[0:4]:
        with dace.tasklet:
            o >> out[i, :]
            up = 0
            for d in range(i, 10, 4):  # The step does not divide the length for every i
                up = up + d
            down = 0
            for d in range(10, i, -3):
                down = down + d
            empty = 0
            for d in range(i + 5, 3):
                empty = empty + 1
            whole = 0
            for d in range(i):
                whole = whole + d
            o[0] = up
            o[1] = down
            o[2] = empty
            o[3] = whole


def test_range_steps():
    out = np.zeros((4, 4), dtype=np.int64)
    strided_sums(out=out)
    expected = [[sum(range(i, 10, 4)), sum(range(10, i, -3)), len(range(i + 5, 3)), sum(range(i))] for i in range(4)]
    assert np.array_equal(out, np.array(expected))


if __name__ == "__main__":
    test_range_steps()
