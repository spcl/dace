# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import MapTilingWithOverlap
import unittest
import numpy as np

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def copy(inp: dace.float32[I, J], out: dace.float32[I, J]):
    @dace.map
    def copy(y: _[0:I], x: _[0:J]):
        i << inp[y, x]
        o >> out[y, x]
        o = i


class MapTilingWithOverlapTest(unittest.TestCase):
    def semantic_eq(self, tile_sizes):
        A = np.random.rand(16, 16).astype(np.float32)
        B1 = np.zeros((16, 16), dtype=np.float32)
        B2 = np.zeros((16, 16), dtype=np.float32)

        sdfg = copy.to_sdfg()
        sdfg(inp=A, out=B1, I=A.shape[0], J=A.shape[1])

        count = sdfg.apply_transformations(MapTilingWithOverlap,
                                           options={
                                               'tile_sizes': tile_sizes,
                                               'lower_overlap': (1, 2),
                                               'upper_overlap': (1, 2)
                                           })
        self.assertGreater(count, 0)
        sdfg(inp=A, out=B2, I=A.shape[0], J=A.shape[1])

        self.assertTrue(np.allclose(B1, B2))

    def test_semantic_eq(self):
        self.semantic_eq([3, 3])

    def test_semantic_eq_tile_size_1(self):
        self.semantic_eq([1, 1])


if __name__ == '__main__':
    unittest.main()
