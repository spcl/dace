# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import dace
import numpy as np

B, M, N, K, L, O = tuple(dace.symbol(k) for k in 'BMNKLO')


class MatrixMultiplication(unittest.TestCase):
    def test_mmm(self):
        @dace.program
        def mmmtest(a: dace.float64[M, K], b: dace.float64[K, N]):
            return a @ b

        a = np.random.rand(32, 33)
        b = np.random.rand(33, 34)
        c = mmmtest(a, b)
        self.assertEqual(list(c.shape), [32, 34])
        self.assertTrue(np.allclose(c, a @ b))

    def test_mmm_batch(self):
        @dace.program
        def mmmtest(a: dace.float64[B, M, K], b: dace.float64[B, K, N]):
            return a @ b

        a = np.random.rand(3, 34, 32)
        b = np.random.rand(3, 32, 31)
        c = mmmtest(a, b)
        self.assertEqual(list(c.shape), [3, 34, 31])
        self.assertTrue(np.allclose(c, a @ b))

    def test_mmm_batch_stationary_a(self):
        @dace.program
        def mmmtest(a: dace.float64[M, K], b: dace.float64[B, K, N]):
            return a @ b

        a = np.random.rand(34, 32)
        b = np.random.rand(3, 32, 31)
        c = mmmtest(a, b)
        self.assertEqual(list(c.shape), [3, 34, 31])
        self.assertTrue(np.allclose(c, a @ b))
    
    def test_mm_symbolic(self):
        @dace.program
        def mmtest_symbolic(a: dace.float64[M, K], b: dace.float64[O, N]):
            return a @ b

        a = np.random.rand(32, 33)
        b = np.random.rand(33, 34)
        c = mmtest_symbolic(a, b)
        self.assertEqual(list(c.shape), [32, 34])
        self.assertTrue(np.allclose(c, a @ b))
    
    def test_mmm_batch_symbolic(self):
        @dace.program
        def mmmtest_symbolic(a: dace.float64[B, M, K], b: dace.float64[L, O, N]):
            return a @ b

        a = np.random.rand(3, 34, 32)
        b = np.random.rand(3, 32, 31)
        c = mmmtest_symbolic(a, b)
        self.assertEqual(list(c.shape), [3, 34, 31])
        self.assertTrue(np.allclose(c, a @ b))


if __name__ == '__main__':
    unittest.main()
