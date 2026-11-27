# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe through its Fortran frontend, the separate dace-fortran package: one small program for each part of DaCe that
Fortran semantics reach. dace-fortran carries the frontend's own suite; run these with the ``fortran`` workflow."""

import numpy as np
import pytest

from dace.sdfg.nodes import LibraryNode

pytestmark = pytest.mark.fortran


def compile_fortran(source: str, tmp_path, name: str):
    from dace_fortran.build import build_sdfg

    return build_sdfg(source, name=name, out_dir=str(tmp_path))


def test_loop_over_column_major_array(tmp_path):
    sdfg = compile_fortran(
        """
subroutine fill(a)
  double precision, intent(out) :: a(3, 2)
  integer :: i, j
  do j = 1, 2
    do i = 1, 3
      a(i, j) = 10 * i + j
    end do
  end do
end subroutine fill
""",
        tmp_path,
        "fill",
    )
    a = np.zeros((3, 2), order="F")
    sdfg(a=a)
    assert np.array_equal(a, [[11, 12], [21, 22], [31, 32]])


def test_lower_bounds_and_sections(tmp_path):
    sdfg = compile_fortran(
        """
subroutine section_sum(d, res)
  double precision, intent(in) :: d(2:6)
  double precision, intent(out) :: res(2)
  res(1) = SUM(d)
  res(2) = SUM(d(3:5))
end subroutine section_sum
""",
        tmp_path,
        "section_sum",
    )
    res = np.zeros(2, order="F")
    sdfg(d=np.arange(1.0, 6.0), res=res)
    assert np.array_equal(res, [15, 9])


def test_remainder_and_bit_runtime(tmp_path):
    sdfg = compile_fortran(
        """
subroutine runtime(a, b, res)
  integer, intent(in) :: a, b
  integer, intent(out) :: res(3)
  res(1) = MOD(a, b)
  res(2) = MODULO(a, b)
  res(3) = ISHFT(a, -28)
end subroutine runtime
""",
        tmp_path,
        "runtime",
    )
    res = np.zeros(3, order="F", dtype=np.int32)
    sdfg(a=np.int32(-7), b=np.int32(3), res=res)
    assert np.array_equal(res, [-1, 2, 15])


def test_matmul_library_node(tmp_path):
    sdfg = compile_fortran(
        """
subroutine product(a, x, y)
  double precision, intent(in) :: a(2, 3), x(3)
  double precision, intent(out) :: y(2)
  y = MATMUL(a, x)
end subroutine product
""",
        tmp_path,
        "product",
    )
    assert any(isinstance(node, LibraryNode) for node, _ in sdfg.all_nodes_recursive())
    a = np.asfortranarray(np.arange(1.0, 7.0).reshape(2, 3))
    x = np.array([1.0, 2.0, 3.0])
    y = np.zeros(2, order="F")
    sdfg(a=a, x=x, y=y)
    assert np.array_equal(y, a @ x)


def test_conditional_on_loop_scalar(tmp_path):
    sdfg = compile_fortran(
        """
subroutine positive_sum(d, n, res)
  integer, intent(in) :: n
  double precision, intent(in) :: d(n)
  double precision, intent(out) :: res
  integer :: i
  res = 0
  do i = 1, n
    if (d(i) > 0) res = res + d(i)
  end do
end subroutine positive_sum
""",
        tmp_path,
        "positive_sum",
    )
    res = np.zeros(1)
    sdfg(d=np.array([1.0, -2.0, 3.0]), n=3, res=res)
    assert res[0] == 4


if __name__ == "__main__":
    import pathlib
    import tempfile

    for test in (
        test_loop_over_column_major_array,
        test_lower_bounds_and_sections,
        test_remainder_and_bit_runtime,
        test_matmul_library_node,
        test_conditional_on_loop_scalar,
    ):
        with tempfile.TemporaryDirectory() as folder:
            test(pathlib.Path(folder))
