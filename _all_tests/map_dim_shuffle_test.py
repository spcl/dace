# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import MapDimShuffle
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


class MapDimShuffleTest(unittest.TestCase):
    def semantic_eq(self, params):
        A = np.random.rand(16, 8).astype(np.float32)
        B1 = np.zeros((16, 8), dtype=np.float32)
        B2 = np.zeros((16, 8), dtype=np.float32)

        sdfg = copy.to_sdfg()
        sdfg(inp=A, out=B1, I=A.shape[0], J=A.shape[1])

        count = sdfg.apply_transformations(MapDimShuffle, options={'parameters': params})
        self.assertGreater(count, 0)
        sdfg(inp=A, out=B2, I=A.shape[0], J=A.shape[1])

        self.assertLess(np.linalg.norm(B1 - B2), 1e-8)

    def test_semantic_eq(self):
        self.semantic_eq(['x', 'y'])

    def test_semantic_eq_trivial_trafo(self):
        self.semantic_eq(['y', 'x'])


if __name__ == '__main__':
    unittest.main()
