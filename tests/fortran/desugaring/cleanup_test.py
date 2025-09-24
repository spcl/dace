# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace.frontend.fortran.ast_desugaring import cleanup
from tests.fortran.fortran_test_helper import SourceCodeBuilder, parse_and_improve


def test_globally_unique_names():
    sources, main = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  type :: Square
    real :: sides(2, 2)
  end type Square
  integer, parameter :: k = 4
  real :: circle = 2.0_k
contains
  real function perim(this, m)
    implicit none
    class(Square), intent(IN) :: this
    real, intent(IN) :: m
    perim = m*sum(this%sides)
  end function perim
  function area(this, m)
    implicit none
    class(Square), intent(IN) :: this
    real, intent(IN) :: m
    real, dimension(2, 2) :: area
    area = m*sum(this%sides)
  end function area
end module lib

subroutine main
  use lib
  use lib, only: perim
  use lib, only: p2 => perim
  use lib, only: circle
  implicit none
  type(Square) :: s
  real :: a
  integer :: i, j
  s%sides = 0.5
  s%sides(1, 1) = 1.0
  s%sides(2, 1) = 1.0
  do i = 1, 2
    do j = 1, 2
      s%sides(i, j) = 7.0
    end do
  end do
  a = perim(s, 1.0)
  a = p2(s, 1.0)
  s%sides = area(s, 4.1)
  circle = 5.0
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = cleanup.assign_globally_unique_subprogram_names(ast, {("main", )})
    ast = cleanup.assign_globally_unique_variable_names(ast, set())

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: sides(2, 2)
  END TYPE Square
  INTEGER, PARAMETER :: k = 4
  REAL :: circle = 2.0_k
  CONTAINS
  REAL FUNCTION perim(this_var_0, m_var_1)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this_var_0
    REAL, INTENT(IN) :: m_var_1
    perim = m_var_1 * SUM(this_var_0 % sides)
  END FUNCTION perim
  FUNCTION area_fn_2(this_var_3, m_var_4)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this_var_3
    REAL, INTENT(IN) :: m_var_4
    REAL, DIMENSION(2, 2) :: area_fn_2
    area_fn_2 = m_var_4 * SUM(this_var_3 % sides)
  END FUNCTION area_fn_2
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: circle
  USE lib, ONLY: area_fn_2
  USE lib, ONLY: perim
  USE lib, ONLY: perim
  USE lib
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  INTEGER :: i, j
  s % sides = 0.5
  s % sides(1, 1) = 1.0
  s % sides(2, 1) = 1.0
  DO i = 1, 2
    DO j = 1, 2
      s % sides(i, j) = 7.0
    END DO
  END DO
  a = perim(s, 1.0)
  a = perim(s, 1.0)
  s % sides = area_fn_2(s, 4.1)
  circle = 5.0
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_remove_binds():
    sources, main = (SourceCodeBuilder().add_file(
        """
module lib
  type, bind(C) :: cmplx
    real :: r = 1., i = 2.
  end type cmplx
  integer, bind(C) :: ii = 7
  interface operator(+)
    module procedure :: add_cmplx
  end interface
contains
  function add_cmplx(a, b) result(c) bind(C, name='add_cmplx')
    type(cmplx), intent(in) :: a, b
    type(cmplx) :: c
    c%r = a%r + b%r
    c%i = a%i + b%i
  end function add_cmplx
  subroutine fun() bind(C)
  end subroutine fun
end module lib

subroutine main
  use lib, only : cmplx, operator(+), fun
  type(cmplx) :: a, b
  b = a + a
  call fun
end subroutine main
""",
        "main",
    ).check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = cleanup.remove_access_and_bind_statements(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  TYPE :: cmplx
    REAL :: r = 1., i = 2.
  END TYPE cmplx
  INTEGER :: ii = 7
  INTERFACE OPERATOR(+)
    MODULE PROCEDURE :: add_cmplx
  END INTERFACE
  CONTAINS
  FUNCTION add_cmplx(a, b) RESULT(c)
    TYPE(cmplx), INTENT(IN) :: a, b
    TYPE(cmplx) :: c
    c % r = a % r + b % r
    c % i = a % i + b % i
  END FUNCTION add_cmplx
  SUBROUTINE fun
  END SUBROUTINE fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: cmplx, OPERATOR(+), fun
  TYPE(cmplx) :: a, b
  b = a + a
  CALL fun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_remove_contiguous_statements():
    # TODO: We're testing here that FParser can even parse these (it couldn't in v0.1.3). Do we want to remove these?
    sources, main = (SourceCodeBuilder().add_file(
        """
subroutine main(a)
  implicit none
  type T
    integer, contiguous, pointer :: x(:)
  end type T
  integer, contiguous, target :: a(:)
  type(T) :: z
  z % x => a
  a = sum(z % x)
end subroutine main
""",
        "main",
    ).check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = cleanup.remove_access_and_bind_statements(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(a)
  IMPLICIT NONE
  TYPE :: T
    INTEGER, CONTIGUOUS, POINTER :: x(:)
  END TYPE T
  INTEGER, CONTIGUOUS, TARGET :: a(:)
  TYPE(T) :: z
  z % x => a
  a = SUM(z % x)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_consolidate_global_data():
    sources, main = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  logical :: inited_var = .false.
  logical :: uninited_var
  integer, dimension(3) :: iarr1 = [1, 2, 3]
  integer :: iarr2(3) = [2, 3, 4]
  type cfg
    real :: foo = 1.9
    integer :: bar
  end type cfg
contains
  subroutine update(what)
    implicit none
    logical, intent(out) :: what
    what = .true.
  end subroutine update
end module

subroutine main
  use lib
  implicit none
  real :: a = 1.0
  call update(inited_var)
  call update(uninited_var)
  if (inited_var .and. uninited_var) a = 7.1
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = cleanup.consolidate_global_data_into_arg(ast)

    got = ast.tofortran()
    want = """
MODULE global_mod
  TYPE :: global_data_type
    LOGICAL :: inited_var = .FALSE.
    LOGICAL :: uninited_var
    INTEGER, DIMENSION(3) :: iarr1 = [1, 2, 3]
    INTEGER :: iarr2(3) = [2, 3, 4]
  END TYPE global_data_type
END MODULE global_mod
MODULE lib
  IMPLICIT NONE
  LOGICAL :: inited_var = .FALSE.
  LOGICAL :: uninited_var
  INTEGER, DIMENSION(3) :: iarr1 = [1, 2, 3]
  INTEGER :: iarr2(3) = [2, 3, 4]
  TYPE :: cfg
    REAL :: foo = 1.9
    INTEGER :: bar
  END TYPE cfg
  CONTAINS
  SUBROUTINE update(global_data, what)
    USE global_mod, ONLY: global_data_type
    IMPLICIT NONE
    TYPE(global_data_type) :: global_data
    LOGICAL, INTENT(OUT) :: what
    what = .TRUE.
  END SUBROUTINE update
END MODULE
SUBROUTINE main(global_data)
  USE global_mod, ONLY: global_data_type
  USE lib, ONLY: update
  IMPLICIT NONE
  TYPE(global_data_type) :: global_data
  REAL :: a = 1.0
  CALL update(global_data, global_data % inited_var)
  CALL update(global_data, global_data % uninited_var)
  IF (global_data % inited_var .AND. global_data % uninited_var) a = 7.1
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_create_global_initializers():
    sources, main = (SourceCodeBuilder().add_file("""
module lib
  implicit none
  logical :: inited_var = .false.
  logical :: uninited_var
  integer, parameter :: const = 1
  integer, dimension(3) :: iarr1 = [1, 2, 3]
  integer :: iarr2(3) = [2, 3, 4]
  type cfg
    real :: foo = 1.9
    integer :: bar
  end type cfg
  type(cfg) :: globalo
contains
  subroutine update(what)
    implicit none
    logical, intent(out) :: what
    what = .true.
  end subroutine update
end module

subroutine main
  use lib
  implicit none
  real :: a = 1.0
  call update(inited_var)
  call update(uninited_var)
  if (inited_var .and. uninited_var) a = 7.1
end subroutine main
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)
    ast = cleanup.create_global_initializers(ast, [("main", )])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  LOGICAL :: inited_var = .FALSE.
  LOGICAL :: uninited_var
  INTEGER, PARAMETER :: const = 1
  INTEGER, DIMENSION(3) :: iarr1 = [1, 2, 3]
  INTEGER :: iarr2(3) = [2, 3, 4]
  TYPE :: cfg
    REAL :: foo = 1.9
    INTEGER :: bar
  END TYPE cfg
  TYPE(cfg) :: globalo
  CONTAINS
  SUBROUTINE update(what)
    IMPLICIT NONE
    LOGICAL, INTENT(OUT) :: what
    what = .TRUE.
  END SUBROUTINE update
  SUBROUTINE type_init_cfg_0(this)
    IMPLICIT NONE
    TYPE(cfg) :: this
    this % foo = 1.9
  END SUBROUTINE type_init_cfg_0
END MODULE
SUBROUTINE main
  USE lib
  IMPLICIT NONE
  REAL :: a = 1.0
  CALL global_init_fn
  CALL update(inited_var)
  CALL update(uninited_var)
  IF (inited_var .AND. uninited_var) a = 7.1
END SUBROUTINE main
SUBROUTINE global_init_fn
  USE lib, ONLY: inited_var
  USE lib, ONLY: iarr1
  USE lib, ONLY: iarr2
  USE lib, ONLY: globalo
  USE lib, ONLY: type_init_cfg_0
  IMPLICIT NONE
  inited_var = .FALSE.
  iarr1 = [1, 2, 3]
  iarr2 = [2, 3, 4]
  CALL type_init_cfg_0(globalo)
END SUBROUTINE global_init_fn
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


if __name__ == "__main__":
    test_globally_unique_names()
    test_remove_binds()
    test_remove_contiguous_statements()
    test_consolidate_global_data()
    test_create_global_initializers()
