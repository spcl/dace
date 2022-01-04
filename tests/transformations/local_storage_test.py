# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import dace
import numpy as np
from dace.transformation.dataflow import MapTiling, OutLocalStorage

N = dace.symbol('N')


@dace.program
def arange():
    out = np.ndarray([N], np.int32)
    for i in dace.map[0:N]:
        with dace.tasklet:
            o >> out[i]
            o = i
    return out


class LocalStorageTests(unittest.TestCase):
    def test_even(self):
        sdfg = arange.to_sdfg()
        sdfg.apply_transformations([MapTiling, OutLocalStorage], options=[{'tile_sizes': [8]}, {}])
        self.assertTrue(np.array_equal(sdfg(N=16), np.arange(16, dtype=np.int32)))

    def test_uneven(self):
        # For testing uneven decomposition, use longer buffer and ensure
        # it's not filled over
        output = np.ones(20, np.int32)
        sdfg = arange.to_sdfg()
        sdfg.apply_transformations([MapTiling, OutLocalStorage], options=[{'tile_sizes': [5]}, {}])
        dace.propagate_memlets_sdfg(sdfg)
        sdfg(N=16, __return=output)
        self.assertTrue(np.array_equal(output[:16], np.arange(16, dtype=np.int32)))
        self.assertTrue(np.array_equal(output[16:], np.ones(4, np.int32)))


if __name__ == '__main__':
    unittest.main()
