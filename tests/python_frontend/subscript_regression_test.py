# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace


@dace.program
def dace_func(
    X: dace.float32[4, 5],
    Y: dace.float32[4, 3],
    W: dace.float32[4, 3],
    S: dace.float32[1],
):
    Xt = np.transpose(X)
    YW = W * Y
    Z = Xt @ YW

    @dace.map(_[0:5, 0:3])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << Z[i, j]
        s = log(z + 1)


def test_regression():
    dace_func.to_sdfg()


def test_an_index_negative_by_its_facts_counts_from_the_end():
    """A symbol's sign lives in the facts: ``A[-i]`` with ``i >= 1`` and ``A[-M:]`` with ``M > 0`` wrap, as in Python."""
    M = dace.symbol("M", dace.int64, positive=True)

    @dace.program
    def indexed(A: dace.float64[10], B: dace.float64[10]):
        for i in range(1, 5):
            B[i] = A[-i]

    @dace.program
    def sliced(A: dace.float64[10]):
        A[-M:] = 7.0

    source, result = np.arange(10.0), np.zeros(10)
    indexed(source, result)
    assert np.array_equal(result, [0, 9, 8, 7, 6, 0, 0, 0, 0, 0])

    sdfg = sliced.to_sdfg(simplify=False)
    assert not [e for s in sdfg.states() for e in s.edges() if str(e.data).startswith("A[-")]
    filled = np.zeros(10)
    sliced(filled, M=3)
    assert np.array_equal(filled, [0, 0, 0, 0, 0, 0, 0, 7, 7, 7])


if __name__ == "__main__":
    test_regression()
    test_an_index_negative_by_its_facts_counts_from_the_end()
