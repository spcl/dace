# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import dace
import numpy as np

N = dace.symbol('N')


class MyTestCase(unittest.TestCase):

    def test_simple(self):

        @dace.program
        def eyetest():
            return np.eye(N)

        self.assertTrue(np.allclose(eyetest(N=5), np.eye(5)))

    def test_rect(self):

        @dace.program
        def eyetest():
            return np.eye(N, N + 1)

        self.assertTrue(np.allclose(eyetest(N=5), np.eye(5, 6)))

    def test_rect_subdiagonal(self):

        @dace.program
        def eyetest():
            return np.eye(N, N + 1, -1)

        self.assertTrue(np.allclose(eyetest(N=5), np.eye(5, 6, -1)))

    def test_superdiagonal(self):

        @dace.program
        def eyetest():
            return np.eye(N, k=2)

        self.assertTrue(np.allclose(eyetest(N=5), np.eye(5, k=2)))


if __name__ == '__main__':
    unittest.main()
