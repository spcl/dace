# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a callee symbol bound to a caller expression is renamed consistently. """
import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64)
M = dace.symbol('M', dtype=dace.int64)


@dace.program
def copy_callee(a: dace.float64[M]):
    b = np.ndarray((M, ), dtype=np.float64)
    for j in dace.map[0:M]:
        b[j] = a[j]
    return b


@dace.program
def prefix_caller(a: dace.float64[N], out: dace.float64[N]):
    for k in range(1, N):
        out[:k] += copy_callee(a[:k])


def test_a_callee_symbol_mapped_to_a_loop_bound_is_renamed_everywhere():
    """The callee's ``M`` is renamed through a temporary; a memlet volume holding it must follow (durbin)."""

    @dace.program
    def outer(a: dace.float64[12], out: dace.float64[12]):
        prefix_caller(a, out)

    outer.to_sdfg(simplify=True).validate()


if __name__ == '__main__':
    test_a_callee_symbol_mapped_to_a_loop_bound_is_renamed_everywhere()
