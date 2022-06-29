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
    for i in range(4, I):
        for j in dace.map[0:J]:
            data[i, j] = data[i - 1, j]


# backward loop with loop carried dependency
@dace.program
def backward_loop(data: dace.float64[I, J]):
    for i in range(I - 2, 3, -1):
        for j in dace.map[0:J]:
            data[i, j] = data[i + 1, j]


@dace.program
def multiple_edges(data: dace.float64[I, J]):
    for i in range(4, I):
        for j in dace.map[1:J]:
            data[i, j] = data[i - 1, j] + data[i - 2, j]


@dace.program
def should_not_apply_1():
    for i in range(20):
        a = np.zeros([i])


@dace.program
def should_not_apply_2():
    for i in range(2, 20):
        a = np.ndarray([i], np.float64)
        a[0:2] = 0


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

    def test_itervar_in_map_range(self):
        sdfg = should_not_apply_1.to_sdfg(simplify=True)
        count = sdfg.apply_transformations(MoveLoopIntoMap)
        self.assertEquals(count, 0)

    def test_itervar_in_data(self):
        sdfg = should_not_apply_2.to_sdfg(simplify=True)
        count = sdfg.apply_transformations(MoveLoopIntoMap)
        self.assertEquals(count, 0)


if __name__ == '__main__':
    unittest.main()
