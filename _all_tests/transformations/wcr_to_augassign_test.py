# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WCRToAugAssign. """

import dace
import numpy as np
from dace.transformation.dataflow import WCRToAugAssign


def test_tasklet():

    @dace.program
    def test():
        a = np.zeros((10,))
        for i in dace.map[1:9]:
            a[i-1] += 1
        return a

    sdfg = test.to_sdfg(simplify=False)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert(np.allclose(val, ref))


def test_mapped_tasklet():

    @dace.program
    def test():
        a = np.zeros((10,))
        for i in dace.map[1:9]:
            a[i-1] += 1
        return a

    sdfg = test.to_sdfg(simplify=True)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert(np.allclose(val, ref))


if __name__ == '__main__':
    test_tasklet()
    test_mapped_tasklet()
