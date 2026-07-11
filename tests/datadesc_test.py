# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_strides():
    desc = dace.float64[1, 2, 3]
    assert desc.strides == (6, 3, 1)
    assert desc.total_size == 6
    perm_strides, perm_size = desc.strides_from_layout(0, 1, 2)
    assert perm_size == desc.total_size
    assert perm_strides == (1, 1, 2)
    desc.set_strides_from_layout(0, 2, 1)
    assert desc.strides == (1, 3, 1)
    assert desc.total_size == 6


def test_strides_alignment():
    desc = dace.float64[2, 3, 4]
    assert desc.strides == (12, 4, 1)
    assert desc.total_size == 24
    perm_strides, perm_size = desc.strides_from_layout(0, 1, 2)
    assert perm_size == desc.total_size
    assert perm_strides == (1, 2, 6)
    perm_strides, perm_size = desc.strides_from_layout(1, 0, 2, alignment=4)
    assert perm_size == 64
    assert perm_strides == (4, 1, 16)
    perm_strides, perm_size = desc.strides_from_layout(1, 0, 2, alignment=4, only_first_aligned=True)
    assert perm_size == 32
    assert perm_strides == (4, 1, 8)


def test_numpy_integral_properties():
    desc = dace.data.Array(dace.float64, (np.int32(10), ), strides=(np.int64(2), ), offset=(np.int16(1), ))
    assert desc.shape == (10, )
    assert desc.strides == (2, )
    assert desc.offset == (1, )


if __name__ == '__main__':
    test_strides()
    test_strides_alignment()
    test_numpy_integral_properties()
