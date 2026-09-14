# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``dace.libraries.mpi.utils.is_access_contiguous``.

Contiguity is a stride property: an access is a contiguous run iff its elements occupy consecutive
memory slots. The check must therefore read the array strides -- a leading-singleton, full-trailing
access such as ``A[k, :, :]`` is contiguous in C but not in Fortran, and vice versa for ``A[:, :, k]``.
Each case is cross-checked against a direct memory-offset oracle over concrete strides.
"""
import itertools

import dace
from dace.libraries.mpi.utils import is_access_contiguous


def _c_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _f_strides(shape):
    strides = [1] * len(shape)
    for i in range(1, len(shape)):
        strides[i] = strides[i - 1] * shape[i - 1]
    return strides


def _memlet(shape, ranges, strides):
    """A memlet accessing ``ranges`` of an array with the given ``strides``."""
    desc = dace.data.Array(dace.float64, tuple(shape), strides=tuple(strides))
    m = dace.Memlet(data="x", subset=dace.subsets.Range(ranges))
    return m, desc


def _oracle(ranges, strides) -> bool:
    """Ground truth: the accessed elements form one contiguous memory run."""
    axes = [range(b, e + 1, s) for (b, e, s) in ranges]
    offs = sorted(sum(i * st for i, st in zip(idx, strides)) for idx in itertools.product(*axes))
    return offs == list(range(offs[0], offs[0] + len(offs)))


def _check(shape, ranges, strides):
    m, desc = _memlet(shape, ranges, strides)
    return is_access_contiguous(m, desc), _oracle(ranges, strides)


###############################################################################
# Named cases (the review motivation)
###############################################################################


def test_c_plane_leading_point_is_contiguous():
    """C-packed A[k, :, :] -- leading point + full trailing -- is contiguous (the old check missed this)."""
    shape = [2, 3, 4]
    ranges = [(1, 1, 1), (0, 2, 1), (0, 3, 1)]
    got, ref = _check(shape, ranges, _c_strides(shape))
    assert ref is True
    assert got is True


def test_fortran_plane_leading_point_is_not_contiguous():
    """Fortran-packed A[k, :, :] is NOT contiguous (stride between the trailing elements)."""
    shape = [2, 3, 4]
    ranges = [(1, 1, 1), (0, 2, 1), (0, 3, 1)]
    got, ref = _check(shape, ranges, _f_strides(shape))
    assert ref is False
    assert got is False


def test_fortran_plane_trailing_point_is_contiguous():
    """Fortran-packed A[:, :, k] IS contiguous (the mirror of the C case)."""
    shape = [2, 3, 4]
    ranges = [(0, 1, 1), (0, 2, 1), (2, 2, 1)]
    got, ref = _check(shape, ranges, _f_strides(shape))
    assert ref is True
    assert got is True


def test_c_trailing_point_is_not_contiguous():
    """C-packed A[:, :, k] is NOT contiguous."""
    shape = [2, 3, 4]
    ranges = [(0, 1, 1), (0, 2, 1), (2, 2, 1)]
    got, ref = _check(shape, ranges, _c_strides(shape))
    assert ref is False
    assert got is False


def test_full_array_is_contiguous():
    shape = [3, 5, 7]
    ranges = [(0, s - 1, 1) for s in shape]
    for strides in (_c_strides(shape), _f_strides(shape)):
        got, ref = _check(shape, ranges, strides)
        assert ref is True and got is True


def test_partial_last_dim_c_is_contiguous():
    """C-packed A[k1, k2, 0:p] -- a run along the fastest dim."""
    shape = [2, 3, 8]
    ranges = [(1, 1, 1), (2, 2, 1), (0, 4, 1)]
    got, ref = _check(shape, ranges, _c_strides(shape))
    assert ref is True and got is True


def test_partial_last_dim_with_spanned_middle_not_contiguous():
    shape = [2, 6, 4]
    ranges = [(1, 1, 1), (0, 2, 1), (0, 1, 1)]  # spanned middle, PARTIAL last -> gaps between rows
    got, ref = _check(shape, ranges, _c_strides(shape))
    assert ref is False and got is False


def test_nonunit_step_is_not_contiguous():
    shape = [16]
    ranges = [(0, 15, 2)]
    got, ref = _check(shape, ranges, _c_strides(shape))
    assert ref is False and got is False


def test_1d_subrange_is_contiguous():
    shape = [16]
    ranges = [(3, 9, 1)]
    got, ref = _check(shape, ranges, _c_strides(shape))
    assert ref is True and got is True


def test_symbolic_full_and_point():
    n = dace.symbol("n")
    desc = dace.data.Array(dace.float64, (n, n), strides=(n, 1))
    full = dace.Memlet(data="x", subset=dace.subsets.Range([(0, n - 1, 1), (0, n - 1, 1)]))
    row = dace.Memlet(data="x", subset=dace.subsets.Range([(2, 2, 1), (0, n - 1, 1)]))
    col = dace.Memlet(data="x", subset=dace.subsets.Range([(0, n - 1, 1), (2, 2, 1)]))
    assert is_access_contiguous(full, desc) is True
    assert is_access_contiguous(row, desc) is True  # C-packed row: full fastest dim
    assert is_access_contiguous(col, desc) is False  # strided column


def test_other_subset_raises():
    import pytest
    desc = dace.data.Array(dace.float64, (4, 4), strides=(4, 1))
    m = dace.Memlet(data="x", subset=dace.subsets.Range([(0, 3, 1), (0, 3, 1)]))
    m.other_subset = dace.subsets.Range([(0, 3, 1), (0, 3, 1)])
    with pytest.raises(ValueError):
        is_access_contiguous(m, desc)


###############################################################################
# Property sweep vs the offset oracle, over C and Fortran layouts
###############################################################################


def test_sweep_matches_oracle():
    shapes = [[4], [3, 5], [2, 3, 4], [2, 2, 2, 3]]
    for shape in shapes:
        n = len(shape)
        # per dim: full, a point at 0, a point at end, or a partial prefix
        per_dim_choices = []
        for s in shape:
            choices = [(0, s - 1, 1)]
            choices.append((0, 0, 1))
            if s > 1:
                choices.append((s - 1, s - 1, 1))
                choices.append((0, s - 2, 1))
            per_dim_choices.append(choices)
        for ranges in itertools.product(*per_dim_choices):
            ranges = list(ranges)
            for strides in (_c_strides(shape), _f_strides(shape)):
                got, ref = _check(shape, ranges, strides)
                assert got == ref, f"shape={shape} ranges={ranges} strides={strides}: got {got}, oracle {ref}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("is_access_contiguous tests PASS")
