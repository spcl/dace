import unittest
from typing import Dict

import numpy as np
from sympy import ceiling

import dace
from dace.subsets import Range, Indices, SubrangeMapper
from dace.symbolic import simplify


def eval_range(r: Range, vals: Dict):
    return Range([(simplify(b).subs(vals), simplify(e).subs(vals), simplify(s).subs(vals)) for b, e, s in r.ranges])


class VolumeTest(unittest.TestCase):
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


class RangeRemapperTest(unittest.TestCase):
    def test_mapping_without_symbols(self):
        K, N, M = 6, 7, 8

        # A regular cube.
        src = Range([(0, K - 1, 1), (0, N - 1, 1), (0, M - 1, 1)])
        # A regular cube with offsets.
        dst = Range([(1, 1 + K - 1, 1), (2, 2 + N - 1, 1), (3, 3 + M - 1, 1)])
        # A Mapper
        sm = SubrangeMapper(src, dst)

        # Pick the entire range.
        self.assertEqual(dst, sm.map(src))
        # Pick a point 0, 0, 0.
        self.assertEqual(Range([(1, 1, 1), (2, 2, 1), (3, 3, 1)]),
                         sm.map(Range([(0, 0, 1), (0, 0, 1), (0, 0, 1)])))
        # Pick a point K//2, N//2, M//2.
        self.assertEqual(Range([(1 + K // 2, 1 + K // 2, 1), (2 + N // 2, 2 + N // 2, 1), (3 + M // 2, 3 + M // 2, 1)]),
                         sm.map(Range([(K // 2, K // 2, 1), (N // 2, N // 2, 1), (M // 2, M // 2, 1)])))
        # Pick a point K-1, N-1, M-1.
        self.assertEqual(Range([(1 + K - 1, 1 + K - 1, 1), (2 + N - 1, 2 + N - 1, 1), (3 + M - 1, 3 + M - 1, 1)]),
                         sm.map(Range([(K - 1, K - 1, 1), (N - 1, N - 1, 1), (M - 1, M - 1, 1)])))
        # Pick a quadrant.
        self.assertEqual(Range([(1, 1 + K // 2, 1), (2, 2 + N // 2, 1), (3, 3 + M // 2, 1)]),
                         sm.map(Range([(0, K // 2, 1), (0, N // 2, 1), (0, M // 2, 1)])))

    def test_mapping_with_symbols(self):
        K, N, M = dace.symbol('K', positive=True), dace.symbol('N', positive=True), dace.symbol('M', positive=True)

        # A regular cube.
        src = Range([(0, K - 1, 1), (0, N - 1, 1), (0, M - 1, 1)])
        # A regular cube with offsets.
        dst = Range([(1, 1 + K - 1, 1), (2, 2 + N - 1, 1), (3, 3 + M - 1, 1)])
        # A Mapper
        sm = SubrangeMapper(src, dst)

        # Pick the entire range.
        self.assertEqual(dst, sm.map(src))

        # NOTE: I couldn't make SymPy understand that `(K//2) % K == (K//2)` always holds for postive integers `K`.
        # Hence, the numerical approach.
        argslist = [{'K': k, 'N': n, 'M': m} for k, n, m in zip(np.random.randint(1, 100, size=20),
                                                                np.random.randint(1, 100, size=20),
                                                                np.random.randint(1, 100, size=20))]
        for args in argslist:
            # Pick a point K//2, N//2, M//2.
            want = eval_range(
                Range([(1 + K // 2, 1 + K // 2, 1), (2 + N // 2, 2 + N // 2, 1), (3 + M // 2, 3 + M // 2, 1)]),
                args)
            got = eval_range(
                sm.map(Range([(K // 2, K // 2, 1), (N // 2, N // 2, 1), (M // 2, M // 2, 1)])),
                args)
            self.assertEqual(want, got)
            # Pick a quadrant.
            want = eval_range(
                Range([(1, 1 + K // 2, 1), (2, 2 + N // 2, 1), (3, 3 + M // 2, 1)]),
                args)
            got = eval_range(
                sm.map(Range([(0, K // 2, 1), (0, N // 2, 1), (0, M // 2, 1)])),
                args)
            self.assertEqual(want, got)

    def test_mapping_with_reshaping(self):
        K, N, M = dace.symbol('K', positive=True), dace.symbol('N', positive=True), dace.symbol('M', positive=True)

        # A regular cube.
        src = Range([(0, K - 1, 1), (0, N - 1, 1), (0, M - 1, 1)])
        # A regular cube with different shape.
        dst = Range([(0, K - 1, 1), (0, N * M - 1, 1)])
        # A Mapper
        sm = SubrangeMapper(src, dst)

        # Pick the entire range.
        self.assertEqual(dst, sm.map(src))

        # NOTE: I couldn't make SymPy understand that `(K//2) % K == (K//2)` always holds for postive integers `K`.
        # Hence, the numerical approach.
        argslist = [{'K': k, 'N': n, 'M': m} for k, n, m in zip(np.random.randint(1, 10, size=20),
                                                                np.random.randint(1, 10, size=20),
                                                                np.random.randint(1, 10, size=20))]
        # Pick a point K//2, N//2, M//2.
        for args in argslist:
            want = eval_range(
                Range([(K // 2, K // 2, 1), ((N // 2) + (M // 2) * N, (N // 2) + (M // 2) * N, 1)]),
                args)
            got = eval_range(
                sm.map(Range([(K // 2, K // 2, 1), (N // 2, N // 2, 1), (M // 2, M // 2, 1)])),
                args)
            self.assertEqual(want, got)
        # Pick a quadrant.
        for args in argslist:
            # But its mapping cannot be expressed as a simple range with offset and stride.
            self.assertIsNone(sm.map(Range([(0, K // 2, 1), (0, N // 2, 1), (0, M // 2, 1)])))
        # Pick only points in problematic quadrants, but larger subsets elsewhere.
        for args in argslist:
            want = eval_range(
                Range([(0, K // 2, 1), ((N // 2) + (M // 2) * N, (N // 2) + (M // 2) * N, 1)]),
                args)
            got = eval_range(
                sm.map(Range([(0, K // 2, 1), (N // 2, N // 2, 1), (M // 2, M // 2, 1)])),
                args)
            self.assertEqual(want, got)


if __name__ == '__main__':
    unittest.main()
