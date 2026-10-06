# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A comparison the loop bounds decide folds to a sympy boolean; the boolean operators must still accept it."""
import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64, positive=True)


@dace.program
def folded_and(t: dace.int32[N, N]):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0 and i + 1 < N:
                t[i, j] = 1


def test_a_comparison_the_bounds_decide_inside_and():
    t = np.zeros((5, 5), dtype=np.int32)
    folded_and(t=t, N=5)
    assert np.array_equal(t, np.triu(np.ones((5, 5), dtype=np.int32), 1))


if __name__ == '__main__':
    test_a_comparison_the_bounds_decide_inside_and()
