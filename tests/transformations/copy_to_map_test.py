# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import CopyToMap
import copy
import pytest
import numpy as np


def _copy_to_map(storage):
    @dace
    def somecopy(a, b):
        b[:] = a

    desc_a = dace.data.Array(dace.float64, [3, 2], storage=storage, strides=(1, 32), total_size=64)
    desc_b = dace.data.Array(dace.float64, [3, 2], storage=storage, strides=(2, 1), total_size=6)
    A = dace.data.make_array_from_descriptor(desc_a, np.random.rand(3, 2))
    B = dace.data.make_array_from_descriptor(desc_b, np.random.rand(3, 2))
    sdfg = somecopy.to_sdfg(A, B)
    sdfg.apply_transformations(CopyToMap)
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(A, B)

    assert np.allclose(B, A)


def _flatten_to_map(storage):
    @dace
    def somecopy(a, b):
        b[:] = a.flatten()

    desc_a = dace.data.Array(dace.float64, [3, 2], storage=storage, strides=(1, 32), total_size=64)
    desc_b = dace.data.Array(dace.float64, [6], storage=storage, total_size=6)
    A = dace.data.make_array_from_descriptor(desc_a, np.random.rand(3, 2))
    B = dace.data.make_array_from_descriptor(desc_b, np.random.rand(6))
    sdfg = somecopy.to_sdfg(A, B)
    sdfg.apply_transformations(CopyToMap)
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(A, B)

    assert np.allclose(B, A.flatten())


def test_copy_to_map():
    _copy_to_map(dace.StorageType.CPU_Heap)


@pytest.mark.gpu
def test_copy_to_map_gpu():
    _copy_to_map(dace.StorageType.GPU_Global)


def test_flatten_to_map():
    _flatten_to_map(dace.StorageType.CPU_Heap)


@pytest.mark.gpu
def test_flatten_to_map_gpu():
    _flatten_to_map(dace.StorageType.GPU_Global)


@pytest.mark.gpu
def test_preprocess():
    """ Tests preprocessing in the GPU code generator, adding CopyToMap automatically. """
    sdfg = dace.SDFG('copytest')

    # Create two arrays with different allocation structure
    desc_inp = dace.data.Array(dace.float64, [20, 21, 22], strides=(1, 32, 32 * 21), total_size=14784)
    desc_out = dace.data.Array(dace.float64, [20, 21, 22], start_offset=5, total_size=20 * 21 * 22 + 5)

    # Construct graph
    sdfg.add_datadesc('inp', desc_inp)
    sdfg.add_datadesc('out', desc_out)
    gpudesc_inp = copy.deepcopy(desc_inp)
    gpudesc_inp.storage = dace.StorageType.GPU_Global
    gpudesc_inp.transient = True
    gpudesc_out = copy.deepcopy(desc_out)
    gpudesc_out.storage = dace.StorageType.GPU_Global
    gpudesc_out.transient = True
    sdfg.add_datadesc('gpu_inp', gpudesc_inp)
    sdfg.add_datadesc('gpu_out', gpudesc_out)

    state = sdfg.add_state()
    a = state.add_read('inp')
    b = state.add_read('gpu_inp')
    c = state.add_read('gpu_out')
    d = state.add_read('out')
    state.add_nedge(a, b, dace.Memlet('inp'))
    state.add_nedge(b, c, dace.Memlet('gpu_inp'))
    state.add_nedge(c, d, dace.Memlet('gpu_out'))

    # Create arrays with matching layout
    inp = dace.data.make_array_from_descriptor(desc_inp, np.random.rand(20, 21, 22))
    out = dace.data.make_array_from_descriptor(desc_out, np.random.rand(20, 21, 22))

    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(inp=inp, out=out)

    assert np.allclose(out, inp)


if __name__ == '__main__':
    test_copy_to_map()
    test_copy_to_map_gpu()
    test_flatten_to_map()
    test_flatten_to_map_gpu()
    test_preprocess()
