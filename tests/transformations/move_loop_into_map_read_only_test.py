# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MoveLoopIntoMap and read-only containers.

A container the loop body only reads cannot carry a dependence from one loop iteration to the next, so its subsets
need not depend on the loop variable "consistently". Reading ``b[a, i]`` and ``b[a, j]`` gives a propagated subset
such as ``b[a, Min(1, i):Max(i, N - 1) + 1]``, which the consistency check used to reject.
"""

import numpy as np
import pytest

import dace
from dace.transformation.interstate import MoveLoopIntoMap

M = dace.symbol("M")
N = dace.symbol("N")
#: Rows and columns of the test arrays.
ROWS, COLS = 12, 9


@dace.program
def read_only_union(b: dace.float64[M, N], out: dace.float64[M, N]):
    for i in range(N):
        for a in dace.map[0:M]:
            s = 0.0
            for j in range(1, N):
                s = s + b[a, j] * b[a, i]
            out[a, i] = s


def reference(b: np.ndarray) -> np.ndarray:
    return b * b[:, 1:].sum(axis=1, keepdims=True)


def test_a_loop_reading_a_read_only_container_at_two_positions_moves_into_the_map():
    sdfg = read_only_union.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MoveLoopIntoMap) == 1
    sdfg.validate()
    b = np.random.default_rng(0).random((ROWS, COLS))
    out = np.zeros((ROWS, COLS))
    sdfg(b=b, out=out, M=ROWS, N=COLS)
    assert np.allclose(out, reference(b))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
