# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests how code is generated for free tasklets inside a GPU kernel nested SDFG.
"""

import dace
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import GPUPersistentKernel
import numpy as np
import pytest


def _tester(A: dace.float64[64]):
    t = 12.3
    for _ in range(5):
        A += t
        t += 1.01


def _modify_array(sdfg: dace.SDFG, storage: dace.StorageType):
    for nsdfg, aname, aval in sdfg.arrays_recursive():
        if aname == 't':
            if storage == dace.StorageType.GPU_Shared:
                aval = dace.data.Array(aval.dtype, [1], transient=aval.transient)
                nsdfg.arrays[aname] = aval
            aval.storage = storage
            break
    else:
        raise ValueError('Array not found')


def _make_program(storage: dace.StorageType, persistent=False):
    sdfg = dace.program(_tester).to_sdfg()
    sdfg.apply_gpu_transformations(simplify=False)
    _modify_array(sdfg, storage)

    if persistent:
        content_nodes = set(sdfg.nodes()) - {sdfg.start_state, sdfg.sink_nodes()[0]}
        subgraph = SubgraphView(sdfg, content_nodes)
        transform = GPUPersistentKernel()
        transform.setup_match(subgraph)
        transform.apply(sdfg)

    return sdfg


@pytest.mark.gpu
def test_global_scalar_update():
    sdfg = _make_program(dace.StorageType.GPU_Global, True)
    a = np.random.rand(64)
    aref = np.copy(a)
    _tester(aref)
    sdfg(a)
    assert np.allclose(a, aref)


@pytest.mark.gpu
def test_shared_scalar_update():
    sdfg = _make_program(dace.StorageType.GPU_Shared, persistent=True)

    a = np.random.rand(64)
    aref = np.copy(a)
    _tester(aref)

    # Ensure block size will create at least two thread-blocks
    with dace.config.set_temporary('compiler', 'cuda', 'persistent_map_SM_fraction', value=0.0001):
        with dace.config.set_temporary('compiler', 'cuda', 'persistent_map_occupancy', value=2):
            with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value='32,1,1'):
                sdfg(a)

    assert np.allclose(a, aref)


@pytest.mark.gpu
@pytest.mark.parametrize('persistent', (False, True))
def test_register_scalar_update(persistent):
    sdfg = _make_program(dace.StorageType.Register, persistent)

    a = np.random.rand(64)
    aref = np.copy(a)
    _tester(aref)
    sdfg(a)

    assert np.allclose(a, aref)


if __name__ == '__main__':
    test_global_scalar_update()
    test_shared_scalar_update()
    test_register_scalar_update(False)
    test_register_scalar_update(True)
