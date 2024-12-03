# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from tests.fortran.fortran_test_helper import create_singular_sdfg_from_string, SourceCodeBuilder


def test_fortran_frontend_dot():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: arg2
  double precision, dimension(2) :: res1
  res1(1) = dot_product(arg1, arg2)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=False)
    # TODO: We should re-enable `simplify()` once we merge it.
    # sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)
    res1 = np.full([2], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1
        arg2[i] = i + 5

    sdfg(arg1=arg1, arg2=arg2, res1=res1)

    assert res1[0] == np.dot(arg1, arg2)


def test_fortran_frontend_dot_range():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: arg2
  double precision, dimension(2) :: res1
  res1(1) = dot_product(arg1(1:3), arg2(1:3))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=False)
    # TODO: We should re-enable `simplify()` once we merge it.
    # sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 42, order="F", dtype=np.float64)
    res1 = np.full([2], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1
        arg2[i] = i + 5

    sdfg(arg1=arg1, arg2=arg2, res1=res1)
    assert res1[0] == np.dot(arg1, arg2)


if __name__ == "__main__":
    test_fortran_frontend_dot()
    test_fortran_frontend_dot_range()
