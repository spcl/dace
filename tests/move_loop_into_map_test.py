# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.interstate import MoveLoopIntoMap
import unittest
import numpy as np

I = dace.symbol("I")
J = dace.symbol("J")

# forward loop with loop carried dependency
@dace.program
def forward_loop(data: dace.float64[I, J]):
    for i in range(4,I):
        for j in dace.map[0:J]:
            data[i, j] = data[i-1, j]
           
# backward loop with loop carried dependency
@dace.program
def backward_loop(data: dace.float64[I, J]):
    for i in range(I-2,3,-1):
        for j in dace.map[0:J]:
            data[i, j] = data[i+1, j]


@dace.program
def multiple_edges(data: dace.float64[I, J]):
    for i in range(4, I):
        for j in dace.map[1:J]:
            data[i, j] = data[i-1, j] + data[i-2, j]


class MoveLoopIntoMapTest(unittest.TestCase):
    def semantic_eq(self, program):
        A1 = np.random.rand(16, 16)
        A2 = np.copy(A1)

        sdfg = program.to_sdfg(simplify=True)
        sdfg(A1, I=A1.shape[0], J=A1.shape[1])

        count = sdfg.apply_transformations(MoveLoopIntoMap)
        self.assertGreater(count, 0)
        sdfg(A2, I=A2.shape[0], J=A2.shape[1])

        self.assertTrue(np.allclose(A1, A2))

    def test_forward_loops_semantic_eq(self):
        self.semantic_eq(forward_loop)

    def test_backward_loops_semantic_eq(self):
        self.semantic_eq(backward_loop)

    def test_multiple_edges(self):
        self.semantic_eq(multiple_edges)


if __name__ == '__main__':
    unittest.main()
