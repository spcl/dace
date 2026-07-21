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


@dace.program
def numpy_integral_shape_program(A: dace.float64[np.int32(10)]):
    A += 1


def test_numpy_integral_shape_program():
    A = np.ones((10, ))
    numpy_integral_shape_program(A)
    np.testing.assert_equal(A, 2)


def test_strides_alignment_symbolic_uses_int_ceil():
    """Aligned padding of a SYMBOLIC dimension must use int_ceil, never `//`.

    `(N + a - 1) // a` builds sympy `floor(...)`; sym2cpp prints the argument WITHOUT the floor, so
    each term truncates on its own and the padded size collapses (N=1, a=8 emits 0 instead of 8).
    """
    from dace.codegen.targets.cpp import sym2cpp
    N = dace.symbol('N')
    desc = dace.data.Array(dace.float32, [N])
    _, total_size = desc.strides_from_layout(0, alignment=8)
    assert 'floor' not in str(total_size).replace('int_ceil', ''), total_size
    assert 'int_ceil' in sym2cpp(total_size), sym2cpp(total_size)


if __name__ == '__main__':
    test_strides()
    test_strides_alignment()
    test_numpy_integral_properties()
    test_numpy_integral_shape_program()
    test_strides_alignment_symbolic_uses_int_ceil()
