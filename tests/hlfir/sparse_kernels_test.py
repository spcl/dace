"""Sparse-matrix and indirect-indexing kernels written in plain Fortran
(no intrinsics).  Pin the bridge's coverage of the three primitives that
matter for sparse linear algebra:

  * **Vector gather**  --  ``output(i) = input(idx(i))``.
  * **Vector scatter**  --  ``output(idx(i)) = input(i)``.
  * **CSR SpMV**  --  ``y(i) = sum_j values(j) * x(col_idx(j))`` over the
    row range ``row_ptr(i):row_ptr(i+1)-1``.

Gather and scatter exercise loop-variable indirection through a single
``hlfir.designate``.  CSR SpMV additionally needs variable-bound inner
loops whose bounds are themselves array-element loads  --  the harder
case the bridge does not yet lower (currently xfailed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_vector_gather(tmp_path: Path):
    """``output(i) = input(idx(i))``  --  read from a permuted view."""
    src = """
SUBROUTINE gather_kernel(input, idx, output, n)
  integer, intent(in) :: n
  double precision, dimension(n) :: input, output
  integer, dimension(n) :: idx
  integer :: i
  do i = 1, n
    output(i) = input(idx(i))
  end do
END SUBROUTINE gather_kernel
"""
    sdfg = build_sdfg(src, tmp_path, name='gather_kernel').build()

    n = 5
    inp = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64, order='F')
    idx = np.array([3, 1, 4, 1, 5], dtype=np.int32, order='F')
    out = np.zeros(n, dtype=np.float64, order='F')

    sdfg(input=inp, idx=idx, output=out, n=n)

    np.testing.assert_array_equal(out, inp[idx - 1])


def test_vector_scatter(tmp_path: Path):
    """``output(idx(i)) = input(i)``  --  write through a permutation.

    Caller is responsible for ensuring ``idx`` is a permutation
    (Fortran scatter semantics on non-injective indices are
    implementation-defined; this test uses a permutation)."""
    src = """
SUBROUTINE scatter_kernel(input, idx, output, n)
  integer, intent(in) :: n
  double precision, dimension(n) :: input, output
  integer, dimension(n) :: idx
  integer :: i
  do i = 1, n
    output(idx(i)) = input(i)
  end do
END SUBROUTINE scatter_kernel
"""
    sdfg = build_sdfg(src, tmp_path, name='scatter_kernel').build()

    n = 5
    inp = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64, order='F')
    idx = np.array([3, 1, 4, 5, 2], dtype=np.int32, order='F')
    out = np.zeros(n, dtype=np.float64, order='F')

    sdfg(input=inp, idx=idx, output=out, n=n)

    expected = np.zeros(n, dtype=np.float64)
    for i in range(n):
        expected[idx[i] - 1] = inp[i]
    np.testing.assert_array_equal(out, expected)


def test_csr_spmv(tmp_path: Path):
    """Compressed-sparse-row matrix-vector product:

        y(i) = sum_{j = row_ptr(i)}^{row_ptr(i+1)-1} values(j) * x(col_idx(j))

    Each row's nonzero range comes from two consecutive entries of
    ``row_ptr``; both endpoints feed an inner DO loop's bounds.  The
    bridge renders these as proper DaCe-form subscripts in the
    LoopRegion init / cond, with the outer iter rename applied and
    the per-axis offset symbol substracted so the access lands at the
    correct C-side element after specialise."""
    src = """
SUBROUTINE csr_spmv_kernel(values, col_idx, row_ptr, x, y, n, nnz)
  integer, intent(in) :: n, nnz
  double precision, dimension(nnz) :: values
  integer, dimension(nnz) :: col_idx
  integer, dimension(n + 1) :: row_ptr
  double precision, dimension(n) :: x, y
  integer :: i, j
  double precision :: acc
  do i = 1, n
    acc = 0.0d0
    do j = row_ptr(i), row_ptr(i + 1) - 1
      acc = acc + values(j) * x(col_idx(j))
    end do
    y(i) = acc
  end do
END SUBROUTINE csr_spmv_kernel
"""
    sdfg = build_sdfg(src, tmp_path, name='csr_spmv_kernel').build()

    # Sparse matrix (3x4):
    #   [[1, 0, 2, 0],
    #    [0, 3, 0, 0],
    #    [0, 0, 4, 5]]
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order='F')
    col_idx = np.array([1, 3, 2, 3, 4], dtype=np.int32, order='F')
    row_ptr = np.array([1, 3, 4, 6], dtype=np.int32, order='F')
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64, order='F')
    y = np.zeros(3, dtype=np.float64, order='F')

    sdfg(values=values, col_idx=col_idx, row_ptr=row_ptr, x=x, y=y, n=3, nnz=5)

    np.testing.assert_array_equal(y, [1.0 * 1.0 + 2.0 * 3.0, 3.0 * 2.0, 4.0 * 3.0 + 5.0 * 4.0])
