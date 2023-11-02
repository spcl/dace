# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG

from typing import Dict, Tuple
import pytest


def create_zero_initialization(init_state: dace.SDFGState, array_name):
    sdfg = init_state.sdfg
    array_shape = sdfg.arrays[array_name].shape

    array_access_node = init_state.add_write(array_name)

    indices = ["i" + str(k) for k, _ in enumerate(array_shape)]

    init_state.add_mapped_tasklet(output_nodes={array_name: array_access_node},
                                  name=(array_name + "_init_tasklet"),
                                  map_ranges={k: "0:" + str(v)
                                              for k, v in zip(indices, array_shape)},
                                  inputs={},
                                  code='val = 0',
                                  outputs=dict(val=dace.Memlet.simple(array_access_node.data, ",".join(indices))),
                                  external_edges=True)


def create_test_sdfg():
    sdfg = dace.SDFG('test_sdfg')

    sdfg.add_array('BETA', shape=[10], dtype=dace.float32)
    sdfg.add_array('BETA_MAX', shape=[1], dtype=dace.float32)

    init_state = sdfg.add_state("init")
    state = sdfg.add_state("compute")

    sdfg.add_edge(init_state, state, dace.InterstateEdge())

    for arr in ['BETA_MAX']:
        create_zero_initialization(init_state, arr)

    BETA_MAX = state.add_access('BETA_MAX')
    BETA = state.add_access('BETA')

    beta_max_reduce = state.add_reduce(wcr="lambda a, b: max(a, b)", axes=(0, ), identity=-999999)
    beta_max_reduce.implementation = 'CUDA (device)'
    state.add_edge(BETA, None, beta_max_reduce, None, dace.memlet.Memlet.simple(BETA.data, '0:10'))
    state.add_edge(beta_max_reduce, None, BETA_MAX, None, dace.memlet.Memlet.simple(BETA_MAX.data, '0:1'))

    return sdfg


@pytest.mark.gpu
def test():
    my_max_sdfg = create_test_sdfg()
    my_max_sdfg.validate()
    my_max_sdfg.apply_transformations(GPUTransformSDFG)

    BETA = np.random.rand(10).astype(np.float32)
    BETA_MAX = np.zeros(1).astype(np.float32)

    my_max_sdfg(BETA=BETA, BETA_MAX=BETA_MAX)

    assert (np.max(BETA) == BETA_MAX[0])


if __name__ == "__main__":
    test()
