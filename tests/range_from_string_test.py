# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Implements unit tests for dace.subsets.Range.from_string method."""

import unittest
from dace import subsets as sbs


class RangeFromStringTests(unittest.TestCase):
    """Implements unit tests for dace.subsets.Range.from_string method."""
    def test_simple_uni_dim_range(self):

        r = sbs.Range.from_string('0:M:2')
        self.assertTrue(r.pystr() == '[(0, M - 1, 2)]', msg=r.pystr())

    def test_simple_uni_dim_index(self):

        r = sbs.Range.from_string('i')
        self.assertTrue(r.pystr() == '[(i, i, 1)]', msg=r.pystr())

    def test_simple_multi_dim_range(self):

        r = sbs.Range.from_string('0:M, 4:N:3, -5:10:2')
        self.assertTrue(r.pystr() == '[(0, M - 1, 1), (4, N - 1, 3), (-5, 9, 2)]', msg=r.pystr())

    def test_simple_multi_dim_index(self):

        r = sbs.Range.from_string('i, j, k')
        self.assertTrue(r.pystr() == '[(i, i, 1), (j, j, 1), (k, k, 1)]', msg=r.pystr())

    def test_complex_uni_dim_range_1(self):

        r = sbs.Range.from_string('regtile_j * rs_j : min(K, regtile_j * rs_j + rs_j)')
        self.assertTrue(r.pystr() == '[(regtile_j*rs_j, Min(K, regtile_j*rs_j + rs_j) - 1, 1)]', msg=r.pystr())

    def test_complex_uni_dim_range_2(self):

        r = sbs.Range.from_string('tile_i * ts_i : min(int_ceil(M, rs_i), tile_i * ts_i + ts_i)')
        self.assertTrue(r.pystr() == '[(tile_i*ts_i, Min(tile_i*ts_i + ts_i, int_ceil(M, rs_i)) - 1, 1)]',
                        msg=r.pystr())

    def test_complex_multi_dim_range_1(self):

        r = sbs.Range.from_string('0:M:2, tile_i * ts_i : min(int_ceil(M, rs_i), tile_i * ts_i + ts_i)')
        self.assertTrue(
            r.pystr() == '[(0, M - 1, 2), (tile_i*ts_i, Min(tile_i*ts_i + ts_i, int_ceil(M, rs_i)) - 1, 1)]',
            msg=r.pystr())

    def test_complex_multi_dim_range_2(self):

        r = sbs.Range.from_string('tile_i * ts_i : min(int_ceil(M, rs_i), tile_i * ts_i + ts_i), 0:M:2')
        self.assertTrue(
            r.pystr() == '[(tile_i*ts_i, Min(tile_i*ts_i + ts_i, int_ceil(M, rs_i)) - 1, 1), (0, M - 1, 2)]',
            msg=r.pystr())

    def test_complex_multi_dim_range_3(self):

        r = sbs.Range.from_string(
            'tile_i * ts_i : min(int_ceil(M, rs_i), tile_i * ts_i + ts_i), regtile_j * rs_j : min(K, regtile_j * rs_j + rs_j)'
        )
        self.assertTrue(
            r.pystr() ==
            '[(tile_i*ts_i, Min(tile_i*ts_i + ts_i, int_ceil(M, rs_i)) - 1, 1), (regtile_j*rs_j, Min(K, regtile_j*rs_j + rs_j) - 1, 1)]',
            msg=r.pystr())


if __name__ == '__main__':
    unittest.main()
