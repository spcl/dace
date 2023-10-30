# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WCRToAugAssign. """

import dace
import numpy as np
from dace.transformation.dataflow import WCRToAugAssign


def test_should_apply_tasklet():

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


def test_should_apply_mapped_tasklet():

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


def test_should_not_apply_tasklet():

    @dace.program
    def test():
        a = np.zeros((10,))
        for i in dace.map[1:9]:
            a[i] += a[i-1] + a[i+1]
        return a

    sdfg = test.to_sdfg(simplify=False)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert(np.allclose(val, ref))


def test_should_not_apply_mapped_tasklet():

    @dace.program
    def test():
        a = np.ones((10,))
        for i in dace.map[1:9]:
            a[i] += a[i-1] + a[i+1]
        return a

    sdfg = test.to_sdfg(simplify=True)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert(np.allclose(val, ref))


if __name__ == '__main__':
    test_should_apply_tasklet()
    test_should_apply_mapped_tasklet()
    test_should_not_apply_tasklet()
    test_should_not_apply_mapped_tasklet()
