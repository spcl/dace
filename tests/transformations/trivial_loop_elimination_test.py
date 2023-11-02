# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg.nodes import MapEntry
import dace
from dace.transformation.interstate import TrivialLoopElimination
from dace.symbolic import pystr_to_symbolic
import numpy as np

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def trivial_loop(data: dace.float64[I, J]):
    for i in range(1, 2):
        for j in dace.map[0:J]:
            data[i, j] = data[i, j] + data[i - 1, j]


def test_semantic_eq():
    A1 = np.random.rand(16, 16)
    A2 = np.copy(A1)

    sdfg = trivial_loop.to_sdfg(simplify=False)
    sdfg(A1, I=A1.shape[0], J=A1.shape[1])

    count = sdfg.apply_transformations(TrivialLoopElimination)
    assert (count > 0)
    sdfg(A2, I=A1.shape[0], J=A1.shape[1])

    assert np.allclose(A1, A2)


if __name__ == '__main__':
    test_semantic_eq()
