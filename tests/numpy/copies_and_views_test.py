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


def test_overlapping_self_copy_reads_before_writing():
    """A copy whose source and target overlap must read the source first."""

    @dace.program
    def selfcopy(q: dace.float64[8]):
        q[0:5] = q[1:6]

    values = np.arange(8, dtype=np.float64)
    reference = values.copy()
    reference[0:5] = reference[1:6]

    result = values.copy()
    selfcopy(result)
    assert np.allclose(result, reference)


def test_strided_copy_map_to_outer_transient():
    """A copy inside a map, into a transient declared outside it, leaves the scope."""

    @dace.program
    def via_transient(dst: dace.uint32[20], src: dace.uint32[40]):
        tmp = np.full([20], 3, dtype=np.uint32)
        for i in dace.map[0:2]:
            tmp[i * 10:(i + 1) * 10:2] = src[i * 20:(i + 1) * 20:4]
        dst[:] = tmp

    src = np.arange(1, 41, dtype=np.uint32)
    reference = np.full(20, 3, dtype=np.uint32)
    for i in range(2):
        reference[i * 10:(i + 1) * 10:2] = src[i * 20:(i + 1) * 20:4]

    dst = np.zeros(20, dtype=np.uint32)
    via_transient(dst=dst, src=src.copy())
    assert np.array_equal(dst, reference)


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


def _negative_start(src: dace.int64[8]):
    return src[-1:1:-3]


def _negative_start_open_stop(src: dace.int64[8]):
    return src[-2::-2]


def _negative_stop(src: dace.int64[8]):
    return src[5:-8:-2]


def _reversed_whole(src: dace.int64[8]):
    return src[::-1]


def _reversed_range(src: dace.int64[8]):
    return src[6:2:-1]


@pytest.mark.parametrize('function', [
    _negative_start,
    _negative_start_open_stop,
    _negative_stop,
    _reversed_whole,
    _reversed_range,
])
def test_negative_step_slice_bounds(function):
    """
    Negative-step slices whose bounds are themselves negative.

    Python counts a negative bound from the end of the axis, so ``a[-1:1:-3]``
    starts at the last element. Reading such a bound literally rejected these
    outright; the related defect is that an explicit ``-1`` stop was treated
    like an OMITTED one -- see :func:`test_negative_step_empty_slice`.
    """
    src = np.arange(8, dtype=np.int64)
    assert np.array_equal(np.asarray(dace.program(function)(src.copy())), function(src.copy()))


def test_negative_step_empty_slice():
    """
    An explicit ``-1`` stop names the LAST element, unlike an omitted stop
    which stands for "past the front of the axis". With a negative step that
    makes ``a[7:-1:-2]`` empty, where ``a[7::-2]`` has four elements.

    An empty result has no representation here, so this is refused rather than
    returned -- but it must not silently come back as the four elements the
    omitted-stop reading would give.
    """

    @dace.program
    def empty_slice(src: dace.int64[8]):
        return src[7:-1:-2]

    assert len(empty_slice.f(np.arange(8, dtype=np.int64))) == 0
    with pytest.raises(Exception):
        empty_slice(np.arange(8, dtype=np.int64))


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
    test_overlapping_self_copy_reads_before_writing()

    for _function in (_negative_start, _negative_start_open_stop, _negative_stop, _reversed_whole, _reversed_range):
        test_negative_step_slice_bounds(_function)
    test_negative_step_empty_slice()

    test_strided_copy()
    test_strided_copy_symbolic_0()
    test_strided_copy_symbolic_1()
    test_strided_copy_symbolic_2()
    test_strided_copy_symbolic_3()
    test_strided_copy_map_0()
    test_strided_copy_map_1()
    test_strided_copy_map_to_outer_transient()
    test_strided_copy_map_symbolic_0()
    test_strided_copy_map_symbolic_1()
