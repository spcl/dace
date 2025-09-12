# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from tests.fortran.fortran_test_helper import SourceCodeBuilder
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
"""
    Handling of elemental intrinsics:
    - arr = func(arr)
    - arr = func(arr(:))
    - arr = func(arr(low:high))
    - struct%arr = func(struct%arr)
    - arr = arr + exp(arr)
"""


def test_fortran_frontend_elemental_exp():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1 = exp(arg1)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    py_res = np.exp(arg1)
    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_pardecl():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1 = exp(arg1(:))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    py_res = np.exp(arg1)
    for f_res, p_res in zip(res, py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_subset():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1(2:4) = exp(arg1(2:4))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    print(res)
    assert res[0] == 0
    assert res[4] == 0
    py_res = np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_struct():
    sources, main = SourceCodeBuilder().add_file(
        """

MODULE test_types
    IMPLICIT NONE
    TYPE array_container
        double precision, DIMENSION(5) :: data
    END TYPE array_container
END MODULE

MODULE test_elemental
    USE test_types
    IMPLICIT NONE

    CONTAINS

    subroutine test_func(arg1, res1)
    TYPE(array_container) :: container
    double precision, dimension(5) :: arg1
    double precision, dimension(5) :: res1

    container%data = arg1

    res1(2:4) = exp(container%data(2:4))

    end subroutine test_func

END MODULE

""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'test_elemental.test_func', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    print(res)
    assert res[0] == 0
    assert res[4] == 0
    py_res = np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_subset_hoist():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1(2:4) = 1.0 - exp(arg1(2:4))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    print(res)
    assert res[0] == 0
    assert res[4] == 0
    py_res = 1.0 - np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_exp_complex():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(arg1, arg2, res1)
  double precision, dimension(5) :: arg1
  double precision, dimension(5) :: res1
  res1(2:4) = arg1(2:4) - exp(arg1(2:4))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    res = np.full([size], 0, order="F", dtype=np.float64)

    for i in range(size):
        arg1[i] = i + 1

    sdfg(arg1=arg1, res1=res)

    print(res)
    assert res[0] == 0
    assert res[4] == 0
    py_res = arg1[1:4] - np.exp(arg1[1:4])
    for f_res, p_res in zip(res[1:4], py_res):
        assert abs(f_res - p_res) < 10**-9


def test_fortran_frontend_elemental_ecrad_bug():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(ng_var_114, od_var_115, trans_dir_dir_var_119)
  INTEGER, INTENT(IN) :: ng_var_114
  REAL(KIND = 8), INTENT(IN), DIMENSION(ng_var_114) :: od_var_115
  REAL(KIND = 8), INTENT(OUT), DIMENSION(ng_var_114) :: trans_dir_dir_var_119
  REAL(KIND = 8) :: mu0

  mu0 = 3.14

  trans_dir_dir_var_119 = MAX(- MAX(od_var_115 * (1.0D0 / mu0), 0.0D0), - 1000.0D0)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5

    arg_in = np.full([size], 42, order="F", dtype=np.float64)
    arg_out = np.full([size], 0, order="F", dtype=np.float64)

    mu0 = 3.14

    for i in range(size):
        arg_in[i] = i + 1

    sdfg(sym_ng_var_114=size, ng_var_114=size, od_var_115=arg_in, trans_dir_dir_var_119=arg_out)

    assert np.allclose(arg_out, np.maximum(-np.maximum(arg_in * 1.0 / mu0, 0.0), -1000.0))


if __name__ == "__main__":
    #test_fortran_frontend_elemental_exp()
    #test_fortran_frontend_elemental_exp_pardecl()
    #test_fortran_frontend_elemental_exp_subset()
    #test_fortran_frontend_elemental_exp_struct()
    #test_fortran_frontend_elemental_exp_subset_hoist()
    #test_fortran_frontend_elemental_exp_complex()
    test_fortran_frontend_elemental_ecrad_bug()
