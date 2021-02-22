# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import dace
import numpy as np

B, M, N, K = tuple(dace.symbol(k) for k in 'BMNK')


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


if __name__ == '__main__':
    unittest.main()
