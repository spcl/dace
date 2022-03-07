# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import CopyToMap
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


if __name__ == '__main__':
    test_copy_to_map()
    test_copy_to_map_gpu()
    test_flatten_to_map()
    test_flatten_to_map_gpu()
