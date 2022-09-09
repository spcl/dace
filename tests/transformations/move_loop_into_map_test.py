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


@dace.program
def should_not_apply_3():
    a = np.ndarray((2, 10), np.float64)
    for i in range(20):
        for j in dace.map[10]:
            a[i % 2, j] = a[(i + 1) % 2, j]


@dace.program
def apply_multiple_times(A: dace.float64[10, 10, 10]):
    for i in range(10):
        for j in range(10):
            for k in dace.map[0:10]:
                A[k, i, j] = i * 100 + j * 10 + k


@dace.program
def apply_multiple_times_1(A: dace.float64[10, 10, 10, 10]):
    l = 5
    for i in range(l, 10):
        for j in range(l, 10):
            for k in dace.map[0:10]:
                A[k, i, j, l] = k * 1000 + i * 100 + j * 10 + l


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

    def test_non_injective_index(self):
        sdfg = should_not_apply_3.to_sdfg(simplify=True)
        count = sdfg.apply_transformations(MoveLoopIntoMap)
        self.assertEquals(count, 0)

    def test_apply_multiple_times(self):
        sdfg = apply_multiple_times.to_sdfg(simplify=True)
        overall = 0
        count = 1
        while (count > 0):
            count = sdfg.apply_transformations_repeated(MoveLoopIntoMap, permissive=True)
            overall += count
            sdfg.simplify()

        self.assertEqual(overall, 2)

        val = np.zeros((10, 10, 10), dtype=np.float64)
        ref = val.copy()

        sdfg(A=val)
        apply_multiple_times.f(ref)

        self.assertTrue(np.allclose(val, ref))

    def test_apply_multiple_times_1(self):
        sdfg = apply_multiple_times_1.to_sdfg(simplify=True)
        overall = 0
        count = 1
        while (count > 0):
            count = sdfg.apply_transformations_repeated(MoveLoopIntoMap, permissive=True)
            overall += count
            sdfg.simplify()

        self.assertEqual(overall, 2)

        val = np.zeros((10, 10, 10, 10), dtype=np.float64)
        ref = val.copy()

        sdfg(A=val)
        apply_multiple_times_1.f(ref)

        self.assertTrue(np.allclose(val, ref))

    def test_more_than_a_map(self):
        sdfg = dace.SDFG('more_than_a_map')
        _, aarr = sdfg.add_array('A', (3, 3), dace.float64)
        _, barr = sdfg.add_array('B', (3, 3), dace.float64)
        _, oarr = sdfg.add_array('out', (3, 3), dace.float64)
        _, tarr = sdfg.add_array('tmp', (3, 3), dace.float64, transient=True)
        body = sdfg.add_state('map_state')
        aread = body.add_access('A')
        oread = body.add_access('out')
        bread = body.add_access('B')
        twrite = body.add_access('tmp')
        owrite = body.add_access('out')
        body.add_mapped_tasklet('op',
                                dict(i='0:3', j='0:3'),
                                dict(__in1=dace.Memlet.from_array('out', oarr), __in2=dace.Memlet.from_array('B',
                                                                                                             barr)),
                                '__out = __in1 - __in2',
                                dict(__out=dace.Memlet.from_array('tmp', tarr)),
                                external_edges=True,
                                input_nodes=dict(out=oread, B=bread),
                                output_nodes=dict(tmp=twrite))
        body.add_nedge(aread, oread, dace.Memlet.from_array('A', aarr))
        body.add_nedge(twrite, owrite, dace.Memlet.from_array('out', oarr))
        sdfg.add_loop(None, body, None, '_', '0', '_ < 10', '_ + 1')
        count = sdfg.apply_transformations(MoveLoopIntoMap)
        self.assertFalse(count > 0)


if __name__ == '__main__':
    unittest.main()
