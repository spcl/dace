# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from re import L
import dace
import numpy as np
import pytest


def _test_transient(persistent: bool):
    @dace.program
    def transient(A: dace.float64[128, 64]):
        for i in dace.map[0:128]:
            # Create local array with the same name as an outer array
            gpu_A = dace.define_local([64], np.float64, storage=dace.StorageType.GPU_Global)
            gpu_A[:] = 0
            gpu_A[:] = 1
            A[i, :] = gpu_A

    sdfg = transient.to_sdfg()
    sdfg.apply_gpu_transformations()

    if persistent:
        sdfg.sdfg_list[-1].arrays['gpu_A'].lifetime = dace.AllocationLifetime.Persistent

    a = np.random.rand(128, 64)
    expected = np.copy(a)
    expected[:] = 1
    with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value='64,8,1'):
        sdfg(a)

    assert np.allclose(a, expected)


def _test_double_transient(persistent: bool):
    @dace.program
    def nested(A: dace.float64[64]):
        # Create local array with the same name as an outer array
        gpu_A = dace.define_local([64], np.float64, storage=dace.StorageType.GPU_Global)
        gpu_A[:] = 0
        gpu_A[:] = 1
        A[:] = gpu_A

    @dace.program
    def transient(A: dace.float64[128, 64]):
        for i in dace.map[0:128]:
            nested(A[i])

    # Simplify, but do not inline
    sdfg = transient.to_sdfg(simplify=False)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.NestedSDFG):
            node.no_inline = True
    if dace.Config.get_bool('optimizer', 'automatic_simplification'):
        sdfg.simplify()

    sdfg.apply_gpu_transformations()

    if persistent:
        sdfg.sdfg_list[-1].arrays['gpu_A'].lifetime = dace.AllocationLifetime.Persistent

    a = np.random.rand(128, 64)
    expected = np.copy(a)
    expected[:] = 1
    with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value='64,8,1'):
        sdfg(a)

    assert np.allclose(a, expected)


@pytest.mark.gpu
def test_nested_kernel_transient():
    _test_transient(False)


@pytest.mark.gpu
def test_nested_kernel_transient_persistent():
    _test_transient(True)


@pytest.mark.gpu
def test_double_nested_kernel_transient():
    _test_double_transient(False)


@pytest.mark.gpu
def test_double_nested_kernel_transient_persistent():
    _test_double_transient(True)


if __name__ == '__main__':
    test_nested_kernel_transient()
    test_nested_kernel_transient_persistent()
    test_double_nested_kernel_transient()
    test_double_nested_kernel_transient_persistent()
