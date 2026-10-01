# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace import data


@dace.program
def set_by_view(A: dace.int64[10]):
    v = A
    v += 1


def test_set_by_view():
    val = np.arange(10)
    set_by_view(val)
    ref = np.arange(10)
    set_by_view.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_1(A: dace.int64[10]):
    v = A[:]
    v += 1


def test_set_by_view_1():
    val = np.arange(10)
    set_by_view_1(val)
    ref = np.arange(10)
    set_by_view_1.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_2(A: dace.int64[10]):
    v = A[1:-1]
    v += 1


def test_set_by_view_2():
    val = np.arange(10)
    set_by_view_2(val)
    ref = np.arange(10)
    set_by_view_2.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_3(A: dace.int64[10]):
    v = A[4:5]
    v += 1


def test_set_by_view_3():
    val = np.arange(10)
    set_by_view_3(val)
    ref = np.arange(10)
    set_by_view_3.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_4(A: dace.float64[10]):
    B = A[1:-1]
    B[...] = 2.0
    B += 1.0


def test_set_by_view_4():
    A = np.ones((10, ), dtype=np.float64)

    set_by_view_4(A)

    assert np.all(A[1:-1] == 3.0)
    assert A[0] == 1.0
    assert A[-1] == 1.0


@dace.program
def inner(A: dace.float64[8]):
    tmp = 2 * A[1:]
    A[:-1] = tmp


@dace.program
def set_by_view_5(A: dace.float64[10]):
    inner(A[1:-1])


def test_set_by_view_5():
    A = np.ones((10, ), dtype=np.float64)

    set_by_view_5(A)

    assert np.all(A[1:-2] == 2.0)
    assert A[0] == 1.0
    assert np.all(A[-2:] == 1.0)


@dace.program
def is_a_copy(A: dace.int64[10]):
    v = A[4]
    v += 1


def test_is_a_copy():
    val = np.arange(10)
    is_a_copy(val)
    ref = np.arange(10)
    is_a_copy.f(ref)
    assert (np.allclose(val, ref))


def test_needs_view():

    @dace.program
    def nested(q, i, j):
        q[3 + j, 4 + i, 0:3] = q[3 - i + 1, 4 + j, 3:6]

    @dace.program
    def selfcopy(q: dace.float64[128, 128, 80]):
        for i in range(1, 4):
            for j in range(1, 4):
                nested(q, i, j)

    sdfg = selfcopy.to_sdfg()
    for s in sdfg.all_sdfgs_recursive():
        assert not any(
            isinstance(d, data.Array) and not isinstance(d, data.View) and d.transient and d.shape == (3, )
            for d in s.arrays.values())


def test_needs_copy():

    @dace.program
    def nested(q, i, j):
        q[3 + j, 4 + i, 0:3] = q[3 - i + 1, 4 + j, 1:4]

    @dace.program
    def selfcopy(q: dace.float64[128, 128, 80]):
        for i in range(1, 4):
            for j in range(1, 4):
                nested(q, i, j)

    sdfg = selfcopy.to_sdfg(simplify=False)
    found_copy = False
    for s in sdfg.all_sdfgs_recursive():
        found_copy |= any(
            isinstance(d, data.Array) and not isinstance(d, data.View) and d.transient and d.shape == (3, )
            for d in s.arrays.values())
    assert found_copy


def _test_strided_copy_program(program, symbols=None):

    src = np.ones(40, dtype=np.uint32)
    dst = np.full(20, 3, dtype=np.uint32)
    ref = np.full(20, 3, dtype=np.uint32)
    ref[0:20:2] = src[0:40:4]

    symbols = symbols or {}
    base_sdfg = program.to_sdfg(simplify=False)
    base_sdfg.validate()
    base_sdfg(src=src, dst=dst, **symbols)
    assert np.array_equal(dst, ref), f"Expected {ref}, got {dst}"

    base_sdfg.simplify()
    base_sdfg.validate()
    dst = np.full(20, 3, dtype=np.uint32)  # Reset destination array
    base_sdfg(src=src, dst=dst, **symbols)
    assert np.array_equal(dst, ref), f"Expected {ref}, got {dst}"


def test_strided_copy():

    @dace.program
    def strided_copy(dst: dace.uint32[20], src: dace.uint32[40]):
        dst[0:20:2] = src[0:40:4]

    _test_strided_copy_program(strided_copy)


def test_strided_copy_symbolic_0():
    N = dace.symbol('N')

    @dace.program
    def strided_copy_symbolic_0(dst: dace.uint32[N], src: dace.uint32[2 * N]):
        dst[0:N:2] = src[0:2 * N:4]

    _test_strided_copy_program(strided_copy_symbolic_0, symbols={'N': 20})


def test_strided_copy_symbolic_1():
    N = dace.symbol('N')

    @dace.program
    def strided_copy_symbolic_1(dst: dace.uint32[N], src: dace.uint32[2 * N]):
        dst[0:N:2] = src[4 * N - 1:-1:-4]

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError):
        # This should raise an error because of the negative stride in the source.
        strided_copy_symbolic_1.to_sdfg(simplify=False)


def test_strided_copy_symbolic_2():
    N = dace.symbol('N')

    @dace.program
    def strided_copy_symbolic_2(dst: dace.uint32[20], src: dace.uint32[40]):
        dst[0:20:N] = src[0:40:2 * N]

    _test_strided_copy_program(strided_copy_symbolic_2, symbols={'N': 2})


def test_strided_copy_symbolic_3():
    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def strided_copy_symbolic_3(dst: dace.uint32[M], src: dace.uint32[2 * M]):
        dst[0:M:N] = src[0:2 * M:2 * N]

    _test_strided_copy_program(strided_copy_symbolic_3, symbols={'M': 20, 'N': 2})


def test_strided_copy_map_0():

    @dace.program
    def strided_copy_map_0(dst: dace.uint32[20], src: dace.uint32[40]):
        for i in dace.map[0:20:2]:
            dst[i] = src[i * 2]

    _test_strided_copy_program(strided_copy_map_0)


def test_strided_copy_map_1():

    @dace.program
    def strided_copy_map_1(dst: dace.uint32[20], src: dace.uint32[40]):
        for i in dace.map[0:2]:
            dst[i * 10:(i + 1) * 10:2] = src[i * 20:(i + 1) * 20:4]

    _test_strided_copy_program(strided_copy_map_1)


def test_strided_copy_map_symbolic_0():
    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def strided_copy_map_symbolic_0(dst: dace.uint32[M], src: dace.uint32[2 * M]):
        for i in dace.map[0:M:N]:
            dst[i] = src[i * 2]

    _test_strided_copy_program(strided_copy_map_symbolic_0, symbols={'M': 20, 'N': 2})


def test_strided_copy_map_symbolic_1():
    M, N = (dace.symbol(s) for s in ('M', 'N'))

    @dace.program
    def strided_copy_map_symbolic_1(dst: dace.uint32[2 * M], src: dace.uint32[4 * M]):
        for i in dace.map[0:2]:
            dst[i * M:(i + 1) * M:N] = src[i * 2 * M:(i + 1) * 2 * M:2 * N]

    _test_strided_copy_program(strided_copy_map_symbolic_1, symbols={'M': 10, 'N': 2})


@dace.program
def rebind_same_view(A: dace.int64[10]):
    v = A[1:-1]
    v += 1
    v = A[1:-1]  # the same slice of the same array: nothing about v changes
    v += 2


def test_rebind_view_to_the_same_slice():
    """Re-viewing a slice a name already views is a no-op, but it was refused.

    Each ``A[1:-1]`` builds its own anonymous view, so the two results differ by NAME while the data
    they see does not -- and the check only recognised a rebind to the whole of the same array.
    """
    sdfg = rebind_same_view.to_sdfg(simplify=False)
    # The no-op rebind must not mint a second descriptor for v; both writes go through the one view.
    assert [n for n in sdfg.arrays if n == 'v' or n.startswith('v_')] == ['v']

    val = np.arange(10)
    sdfg(A=val)
    ref = np.arange(10)
    rebind_same_view.f(ref)
    assert np.array_equal(val, ref)


@dace.program
def rebind_whole_array_view(A: dace.int64[10]):
    v = A
    v += 1
    v = A
    v += 2


def test_rebind_view_to_the_whole_array():
    """The spelling that was already accepted, so widening the check keeps accepting it."""
    val = np.arange(10)
    rebind_whole_array_view(val)
    ref = np.arange(10)
    rebind_whole_array_view.f(ref)
    assert np.array_equal(val, ref)


def test_rebind_view_to_a_different_slice_is_still_refused():
    """A DIFFERENT slice is a real rebind: the name would need a second descriptor, and a read after
    a conditional that rebinds it in one branch would need a phi. Neither is a no-op, so the error
    stands -- this pins the widened check to the case where the two views are interchangeable."""

    @dace.program
    def rebind_other_view(A: dace.int64[10]):
        v = A[1:-1]
        v += 1
        v = A[0:8]
        v += 2

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match='Cannot reassign View'):
        rebind_other_view.to_sdfg(simplify=False)


N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def rebind_chained_slices(A: dace.int64[10]):
    v = A[1:-1][1:3]
    v += 1
    v = A[1:-1][1:3]
    v += 2


@dace.program
def rebind_slice_of_a_named_view(A: dace.int64[10]):
    w = A[1:-1]
    v = w[1:3]
    v += 1
    v = w[1:3]
    v += 2


@dace.program
def rebind_symbolic_slice(A: dace.int64[N]):
    v = A[1:N - 1]
    v += 1
    v = A[1:N - 1]
    v += 2


@dace.program
def rebind_negative_and_symbolic_bound(A: dace.int64[N]):
    v = A[1:-1]
    v += 1
    v = A[1:N - 1]
    v += 2


@dace.program
def rebind_strided_slice(A: dace.int64[10]):
    v = A[1:9:2]
    v += 1
    v = A[1:9:2]
    v += 2


@dace.program
def rebind_same_size_elsewhere(A: dace.int64[10]):
    v = A[0:4]
    v += 1
    v = A[4:8]
    v += 2


@dace.program
def rebind_same_slice_of_another_array(A: dace.int64[10], B: dace.int64[10]):
    v = A[1:-1]
    v += 1
    v = B[1:-1]
    v += 2


@dace.program
def rebind_chained_to_another_inner_slice(A: dace.int64[10]):
    w = A[1:-1]
    v = w[1:3]
    v += 1
    v = w[2:4]
    v += 2


@dace.program
def rebind_chained_to_another_outer_slice(A: dace.int64[10]):
    v = A[0:8][1:3]
    v += 1
    v = A[1:9][1:3]
    v += 2


@dace.program
def rebind_symbolic_slice_of_another_extent(A: dace.int64[N]):
    v = A[0:N - 1]
    v += 1
    v = A[0:N - 2]
    v += 2


@dace.program
def rebind_slice_of_another_symbolic_array(A: dace.int64[N], B: dace.int64[M]):
    v = A[0:4]
    v += 1
    v = B[0:4]
    v += 2


@dace.program
def rebind_other_stride(A: dace.int64[10]):
    v = A[1:9:2]
    v += 1
    v = A[1:9:3]
    v += 2


@pytest.mark.parametrize('program, viewed', [
    pytest.param(rebind_chained_slices, slice(2, 4), id='chained_slices'),
    pytest.param(rebind_slice_of_a_named_view, slice(2, 4), id='slice_of_a_named_view'),
    pytest.param(rebind_symbolic_slice, slice(1, 9), id='symbolic_slice'),
    pytest.param(rebind_negative_and_symbolic_bound, slice(1, 9), id='negative_and_symbolic_bound'),
    pytest.param(rebind_strided_slice, slice(1, 9, 2), id='strided_slice'),
])
def test_rebind_view_to_a_slice_that_sees_the_same_elements(program, viewed):
    sdfg = program.to_sdfg(simplify=False)
    assert [n for n in sdfg.arrays if n == 'v' or n.startswith('v_')] == ['v']
    val = np.arange(10)
    program(A=val, **({'N': 10} if 'N' in sdfg.free_symbols else {}))
    ref = np.arange(10)
    ref[viewed] += 3
    assert np.array_equal(val, ref)


@pytest.mark.parametrize('program', [
    pytest.param(rebind_same_size_elsewhere, id='same_size_elsewhere'),
    pytest.param(rebind_same_slice_of_another_array, id='same_slice_of_another_array'),
    pytest.param(rebind_chained_to_another_inner_slice, id='chained_to_another_inner_slice'),
    pytest.param(rebind_chained_to_another_outer_slice, id='chained_to_another_outer_slice'),
    pytest.param(rebind_symbolic_slice_of_another_extent, id='symbolic_slice_of_another_extent'),
    pytest.param(rebind_slice_of_another_symbolic_array, id='slice_of_another_symbolic_array'),
    pytest.param(rebind_other_stride, id='other_stride'),
])
def test_rebind_view_to_a_slice_that_sees_other_elements_is_refused(program):
    """Views of the same size, or of the same slice of another array, are still different views."""
    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match='Cannot reassign View'):
        program.to_sdfg(simplify=False)


if __name__ == '__main__':
    test_set_by_view()
    test_set_by_view_1()
    test_set_by_view_2()
    test_set_by_view_3()
    test_set_by_view_4()
    test_set_by_view_5()
    test_is_a_copy()
    test_needs_view()
    test_needs_copy()
    test_rebind_view_to_the_same_slice()
    test_rebind_view_to_the_whole_array()
    test_rebind_view_to_a_different_slice_is_still_refused()

    test_strided_copy()
    test_strided_copy_symbolic_0()
    test_strided_copy_symbolic_1()
    test_strided_copy_symbolic_2()
    test_strided_copy_symbolic_3()
    test_strided_copy_map_0()
    test_strided_copy_map_1()
    test_strided_copy_map_symbolic_0()
    test_strided_copy_map_symbolic_1()
