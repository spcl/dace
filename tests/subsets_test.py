import unittest

from sympy import ceiling

from dace.subsets import Range, Indices

import dace


class Volume(unittest.TestCase):
    def test_range(self):
        K, N, M = dace.symbol('K', positive=True), dace.symbol('N', positive=True), dace.symbol('M', positive=True)

        # A regular cube.
        r = Range([(0, K - 1, 1), (0, N - 1, 1), (0, M - 1, 1)])
        self.assertEqual(K * N * M, r.volume_exact())

        # A regular cube with offsets.
        r = Range([(1, 1 + K - 1, 1), (2, 2 + N - 1, 1), (3, 3 + M - 1, 1)])
        self.assertEqual(K * N * M, r.volume_exact())

        # A regular cube with strides.
        r = Range([(0, K - 1, 2), (0, N - 1, 3), (0, M - 1, 4)])
        self.assertEqual(ceiling(K / 2) * ceiling(N / 3) * ceiling(M / 4), r.volume_exact())

        # A regular cube with both offsets and strides.
        r = Range([(1, 1 + K - 1, 2), (2, 2 + N - 1, 3), (3, 3 + M - 1, 4)])
        self.assertEqual(ceiling(K / 2) * ceiling(N / 3) * ceiling(M / 4), r.volume_exact())

        # A 2D square on 3D coordinate system.
        r = Range([(1, 1 + K - 1, 2), (2, 2, 3), (3, 3 + M - 1, 4)])
        self.assertEqual(ceiling(K / 2) * ceiling(M / 4), r.volume_exact())

        # A 3D point.
        r = Range([(1, 1, 2), (2, 2, 3), (3, 3, 4)])
        self.assertEqual(1, r.volume_exact())

    def test_indices(self):
        # Indices always have volume 1 no matter what, since they are just points.
        ind = Indices([0, 1, 2])
        self.assertEqual(1, ind.volume_exact())

        ind = Indices([1])
        self.assertEqual(1, ind.volume_exact())

        ind = Indices([0, 2])
        self.assertEqual(1, ind.volume_exact())


if __name__ == '__main__':
    unittest.main()
