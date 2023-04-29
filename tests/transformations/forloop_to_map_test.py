# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the ForLoopToMap transformation. """

import dace
import numpy as np

from dace.transformation.dataflow import ForLoopToMap


def test_simple_forloop_to_map():

    sdfg = dace.SDFG('simple_forloop_sdfg')
    sdfg.add_array('A', [10], dace.int32)

    state = sdfg.add_state('simple_forloop_state')
    state.add_forlooped_tasklet('simple_forloop', {'i': dace.subsets.Range([(0, 9, 1)])},
                                {},
                                '__out = i', {'__out': dace.Memlet('A[i]')},
                                external_edges=True)
    
    assert sdfg.apply_transformations(ForLoopToMap) == 1

    val = np.empty(10, dtype=np.int32)
    sdfg(A=val)
    assert np.array_equal(val, np.arange(10, dtype=np.int32))


if __name__ == '__main__':
    test_simple_forloop_to_map()
