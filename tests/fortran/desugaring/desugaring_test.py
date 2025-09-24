# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace.frontend.fortran.ast_desugaring import desugaring, cleanup
from tests.fortran.desugaring.common import parse_and_improve
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_procedure_replacer():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  contains
    procedure :: area
    procedure :: area_alt => area
    procedure :: get_area
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m * this%side * this%side
  end function area
  subroutine get_area(this, a)
    implicit none
    class(Square), intent(in) :: this
    real, intent(out) :: a
    a = area(this, 1.0)
  end subroutine get_area
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: a

  s%side = 1.0
  a = s%area(1.0)
  a = s%area_alt(1.0)
  call s%get_area(a)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
  SUBROUTINE get_area(this, a)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(OUT) :: a
    a = area(this, 1.0)
  END SUBROUTINE get_area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: get_area_deconproc_2 => get_area
  USE lib, ONLY: area_deconproc_1 => area
  USE lib, ONLY: area_deconproc_0 => area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side = 1.0
  a = area_deconproc_0(s, 1.0)
  a = area_deconproc_1(s, 1.0)
  CALL get_area_deconproc_2(s, a)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_nested():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Value
    real :: val
  contains
    procedure :: get_value
  end type Value
  type Square
    type(Value) :: side
  contains
    procedure :: get_area
  end type Square
contains
  real function get_value(this)
    implicit none
    class(Value), intent(in) :: this
    get_value = this%val
  end function get_value
  real function get_area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    real :: side
    side = this%side%get_value()
    get_area = m*side*side
  end function get_area
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: a

  s%side%val = 1.0
  a = s%get_area(1.0)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Value
    REAL :: val
  END TYPE Value
  TYPE :: Square
    TYPE(Value) :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION get_value(this)
    IMPLICIT NONE
    CLASS(Value), INTENT(IN) :: this
    get_value = this % val
  END FUNCTION get_value
  REAL FUNCTION get_area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    REAL :: side
    side = get_value(this % side)
    get_area = m * side * side
  END FUNCTION get_area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: get_area_deconproc_0 => get_area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side % val = 1.0
  a = get_area_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_name_collision_with_exisiting_var():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  contains
    procedure :: area
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m*this%side*this%side
  end function area
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: area

  s%side = 1.0
  area = s%area(1.0)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: area_deconproc_0 => area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: area
  s % side = 1.0
  area = area_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_procedure_replacer_name_collision_with_another_import():
    sources, main = SourceCodeBuilder().add_file("""
module lib_1
  implicit none
  type Square
    real :: side
  contains
    procedure :: area
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m*this%side*this%side
  end function area
end module lib_1

module lib_2
  implicit none
  type Circle
    real :: rad
  contains
    procedure :: area
  end type Circle
contains
  real function area(this, m)
    implicit none
    class(Circle), intent(in) :: this
    real, intent(in) :: m
    area = m*this%rad*this%rad
  end function area
end module lib_2

subroutine main
  use lib_1, only: Square
  use lib_2, only: Circle
  implicit none
  type(Square) :: s
  type(Circle) :: c
  real :: area

  s%side = 1.0
  area = s%area(1.0)
  c%rad = 1.0
  area = c%area(1.0)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib_1
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
END MODULE lib_1
MODULE lib_2
  IMPLICIT NONE
  TYPE :: Circle
    REAL :: rad
  END TYPE Circle
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Circle), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % rad * this % rad
  END FUNCTION area
END MODULE lib_2
SUBROUTINE main
  USE lib_2, ONLY: area_deconproc_1 => area
  USE lib_1, ONLY: area_deconproc_0 => area
  USE lib_1, ONLY: Square
  USE lib_2, ONLY: Circle
  IMPLICIT NONE
  TYPE(Square) :: s
  TYPE(Circle) :: c
  REAL :: area
  s % side = 1.0
  area = area_deconproc_0(s, 1.0)
  c % rad = 1.0
  area = area_deconproc_1(c, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_generic_replacer():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  contains
    procedure :: area_real
    procedure :: area_integer
    generic :: g_area => area_real, area_integer
  end type Square
contains
  real function area_real(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area_real = m*this%side*this%side
  end function area_real
  real function area_integer(this, m)
    implicit none
    class(Square), intent(in) :: this
    integer, intent(in) :: m
    area_integer = m*this%side*this%side
  end function area_integer
end module lib

subroutine main
  use lib, only: Square
  implicit none
  type(Square) :: s
  real :: a
  real :: mr = 1.0
  integer :: mi = 1

  s%side = 1.0
  a = s%g_area(mr)
  a = s%g_area(mi)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area_real(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area_real = m * this % side * this % side
  END FUNCTION area_real
  REAL FUNCTION area_integer(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    INTEGER, INTENT(IN) :: m
    area_integer = m * this % side * this % side
  END FUNCTION area_integer
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: area_integer_deconproc_1 => area_integer
  USE lib, ONLY: area_real_deconproc_0 => area_real
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  REAL :: mr = 1.0
  INTEGER :: mi = 1
  s % side = 1.0
  a = area_real_deconproc_0(s, mr)
  a = area_integer_deconproc_1(s, mi)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_association_replacer():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: side
  end type Square
contains
  real function area(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    area = m*this%side*this%side
  end function area
end module lib

subroutine main
  use lib, only: Square, area
  implicit none
  type(Square) :: s
  real :: a

  associate(side => s%side)
    s%side = 0.5
    side = 1.0
    a = area(s, 1.0)
  end associate
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_associations(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
  END TYPE Square
  CONTAINS
  REAL FUNCTION area(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    area = m * this % side * this % side
  END FUNCTION area
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: Square, area
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side = 0.5
  s % side = 1.0
  a = area(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_association_replacer_array_access():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: sides(2, 2)
  contains
    procedure :: area => perim
  end type Square
contains
  real function perim(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    perim = m * sum(this%sides)
  end function perim
end module lib

subroutine main
  use lib, only: Square, perim
  implicit none
  type(Square) :: s
  real :: a

  associate(sides => s%sides)
    s%sides = 0.5
    s%sides(1, 1) = 1.0
    sides(2, 2) = 1.0
    a = perim(s, 1.0)
    a = s%area(1.0)
  end associate
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_enums(ast)
    ast = desugaring.deconstruct_associations(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: sides(2, 2)
  END TYPE Square
  CONTAINS
  REAL FUNCTION perim(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    perim = m * SUM(this % sides)
  END FUNCTION perim
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: perim_deconproc_0 => perim
  USE lib, ONLY: Square, perim
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % sides = 0.5
  s % sides(1, 1) = 1.0
  s % sides(2, 2) = 1.0
  a = perim(s, 1.0)
  a = perim_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_association_replacer_array_access_within_array_access():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type Square
    real :: sides(2, 2)
  contains
    procedure :: area => perim
  end type Square
contains
  real function perim(this, m)
    implicit none
    class(Square), intent(in) :: this
    real, intent(in) :: m
    perim = m * sum(this%sides)
  end function perim
end module lib

subroutine main
  use lib, only: Square, perim
  implicit none
  type(Square) :: s
  real :: a

  associate(sides => s%sides(:, 1))
    s%sides = 0.5
    s%sides(1, 1) = 1.0
    sides(2) = 1.0
    a = perim(s, 1.0)
    a = s%area(1.0)
  end associate
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_associations(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: sides(2, 2)
  END TYPE Square
  CONTAINS
  REAL FUNCTION perim(this, m)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this
    REAL, INTENT(IN) :: m
    perim = m * SUM(this % sides)
  END FUNCTION perim
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: perim_deconproc_0 => perim
  USE lib, ONLY: Square, perim
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % sides = 0.5
  s % sides(1, 1) = 1.0
  s % sides(2, 1) = 1.0
  a = perim(s, 1.0)
  a = perim_deconproc_0(s, 1.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_enum_bindings_become_constants():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 42
  enum, bind(c)
    enumerator :: a, b, c
  end enum
  enum, bind(c)
    enumerator :: d = a, e, f
  end enum
  enum, bind(c)
    enumerator :: g = k, h = k, i = k + 1
  end enum
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_enums(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 42
  INTEGER, PARAMETER :: a = 0 + 0
  INTEGER, PARAMETER :: b = 0 + 1
  INTEGER, PARAMETER :: c = 0 + 2
  INTEGER, PARAMETER :: d = a + 0
  INTEGER, PARAMETER :: e = a + 1
  INTEGER, PARAMETER :: f = a + 2
  INTEGER, PARAMETER :: g = k + 0
  INTEGER, PARAMETER :: h = k + 0
  INTEGER, PARAMETER :: i = k + 1 + 0
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_aliasing_through_module_procedure():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun
  end interface fun
contains
  real function real_fun()
    implicit none
    real_fun = 1.0
  end function real_fun
end module lib

subroutine main
  use lib, only: fun
  implicit none
  real d(4)
  d(2) = fun()
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_associations(ast)
    ast = cleanup.correct_for_function_calls(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun
  END INTERFACE fun
  CONTAINS
  REAL FUNCTION real_fun()
    IMPLICIT NONE
    real_fun = 1.0
  END FUNCTION real_fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: fun
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = fun()
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_module_procedures():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun
  end interface fun
  interface not_fun
    module procedure not_real_fun
  end interface not_fun
  interface same_name
    module procedure same_name, real_fun
  end interface same_name
contains
  real function real_fun()
    implicit none
    real_fun = 1.0
  end function real_fun
  subroutine not_real_fun(a)
    implicit none
    real, intent(out) :: a
    a = 1.0
  end subroutine not_real_fun
  real function same_name(x)
    implicit none
    real, intent(in) :: x
    same_name = x
  end function same_name
end module lib

subroutine main
  use lib, only: fun, not_fun, same_name
  implicit none
  real d(4)
  d(2) = fun()
  call not_fun(d(3))
  d(4) = same_name(2.0)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun
  END INTERFACE fun
  INTERFACE not_fun
    MODULE PROCEDURE not_real_fun
  END INTERFACE not_fun
  INTERFACE same_name
    MODULE PROCEDURE same_name, real_fun
  END INTERFACE same_name
  CONTAINS
  REAL FUNCTION real_fun()
    IMPLICIT NONE
    real_fun = 1.0
  END FUNCTION real_fun
  SUBROUTINE not_real_fun(a)
    IMPLICIT NONE
    REAL, INTENT(OUT) :: a
    a = 1.0
  END SUBROUTINE not_real_fun
  REAL FUNCTION same_name(x)
    IMPLICIT NONE
    REAL, INTENT(IN) :: x
    same_name = x
  END FUNCTION same_name
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: not_real_fun_deconiface_1 => not_real_fun, real_fun_deconiface_0 => real_fun, same_name_deconiface_2 => same_name
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = real_fun_deconiface_0()
  CALL not_real_fun_deconiface_1(d(3))
  d(4) = same_name_deconiface_2(2.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_subroutine_decls():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface
    subroutine fun(z)
      implicit none
      real, intent(out) :: z
    end subroutine fun
  end interface
end module lib

subroutine main
  use lib, only: no_fun => fun
  implicit none
  real d(4)
  call no_fun(d(3))
end subroutine main

subroutine fun(z)
  implicit none
  real, intent(out) :: z
  z = 1.0
end subroutine fun
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE fun(z)
      IMPLICIT NONE
      REAL, INTENT(OUT) :: z
    END SUBROUTINE fun
  END INTERFACE
END MODULE lib
SUBROUTINE main
  IMPLICIT NONE
  REAL :: d(4)
  CALL fun(d(3))
END SUBROUTINE main
SUBROUTINE fun(z)
  IMPLICIT NONE
  REAL, INTENT(OUT) :: z
  z = 1.0
END SUBROUTINE fun
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_optional_args():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun, integer_fun
  end interface fun
contains
  real function real_fun(x)
    implicit none
    real, intent(in), optional :: x
    if (.not.(present(x))) then
      real_fun = 1.0
    else
      real_fun = x
    end if
  end function real_fun
  integer function integer_fun(x)
    implicit none
    integer, intent(in) :: x
    integer_fun = x * 2
  end function integer_fun
end module lib

subroutine main
  use lib, only: fun
  implicit none
  real d(4)
  d(2) = fun()
  d(3) = fun(x=4)
  d(4) = fun(x=5.0)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun, integer_fun
  END INTERFACE fun
  CONTAINS
  REAL FUNCTION real_fun(x)
    IMPLICIT NONE
    REAL, INTENT(IN), OPTIONAL :: x
    IF (.NOT. (PRESENT(x))) THEN
      real_fun = 1.0
    ELSE
      real_fun = x
    END IF
  END FUNCTION real_fun
  INTEGER FUNCTION integer_fun(x)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: x
    integer_fun = x * 2
  END FUNCTION integer_fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: integer_fun_deconiface_1 => integer_fun, real_fun_deconiface_0 => real_fun, real_fun_deconiface_2 => real_fun
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = real_fun_deconiface_0()
  d(3) = integer_fun_deconiface_1(x = 4)
  d(4) = real_fun_deconiface_2(x = 5.0)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_interface_replacer_with_keyworded_args():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  interface fun
    module procedure real_fun
  end interface fun
contains
  real function real_fun(w, x, y, z)
    implicit none
    real, intent(in) :: w
    real, intent(in), optional :: x
    real, intent(in) :: y
    real, intent(in), optional :: z
    if (.not.(present(x))) then
      real_fun = 1.0
    else
      real_fun = w + y
    end if
  end function real_fun
end module lib

subroutine main
  use lib, only: fun
  implicit none
  real d(3)
  d(1) = fun(1.0, 2.0, 3.0, 4.0)  ! all present, no keyword
  d(2) = fun(y=1.1, w=3.1)  ! only required ones, keyworded
  d(3) = fun(1.2, 2.2, y=3.2)  ! partially keyworded, last optional omitted.
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTERFACE fun
    MODULE PROCEDURE real_fun
  END INTERFACE fun
  CONTAINS
  REAL FUNCTION real_fun(w, x, y, z)
    IMPLICIT NONE
    REAL, INTENT(IN) :: w
    REAL, INTENT(IN), OPTIONAL :: x
    REAL, INTENT(IN) :: y
    REAL, INTENT(IN), OPTIONAL :: z
    IF (.NOT. (PRESENT(x))) THEN
      real_fun = 1.0
    ELSE
      real_fun = w + y
    END IF
  END FUNCTION real_fun
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: real_fun_deconiface_0 => real_fun, real_fun_deconiface_1 => real_fun, real_fun_deconiface_2 => real_fun
  IMPLICIT NONE
  REAL :: d(3)
  d(1) = real_fun_deconiface_0(1.0, 2.0, 3.0, 4.0)
  d(2) = real_fun_deconiface_1(y = 1.1, w = 3.1)
  d(3) = real_fun_deconiface_2(1.2, 2.2, y = 3.2)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_generic_replacer_deducing_array_types():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type T
    real :: val(2, 2)
  contains
    procedure :: copy_matrix
    procedure :: copy_vector
    procedure :: copy_scalar
    generic :: copy => copy_matrix, copy_vector, copy_scalar
  end type T
contains
  subroutine copy_scalar(this, m)
    implicit none
    class(T), intent(in) :: this
    real, intent(out) :: m
    m = this%val(1, 1)
  end subroutine copy_scalar
  subroutine copy_vector(this, m)
    implicit none
    class(T), intent(in) :: this
    real, dimension(:), intent(out) :: m
    m = this%val(1, 1)
  end subroutine copy_vector
  subroutine copy_matrix(this, m)
    implicit none
    class(T), intent(in) :: this
    real, dimension(:, :), intent(out) :: m
    m = this%val(1, 1)
  end subroutine copy_matrix
end module lib

subroutine main
  use lib, only: T
  implicit none
  type(T) :: s, s1
  real, dimension(4, 4) :: a
  real :: b(4, 4)

  s%val = 1.0
  call s%copy(a)
  call s%copy(a(2, 2))
  call s%copy(b(:, 2))
  call s%copy(b(:, :))
  call s%copy(s1%val(:, 1))
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_procedure_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: T
    REAL :: val(2, 2)
  END TYPE T
  CONTAINS
  SUBROUTINE copy_scalar(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_scalar
  SUBROUTINE copy_vector(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, DIMENSION(:), INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_vector
  SUBROUTINE copy_matrix(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, DIMENSION(:, :), INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_matrix
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: copy_vector_deconproc_4 => copy_vector
  USE lib, ONLY: copy_matrix_deconproc_3 => copy_matrix
  USE lib, ONLY: copy_vector_deconproc_2 => copy_vector
  USE lib, ONLY: copy_scalar_deconproc_1 => copy_scalar
  USE lib, ONLY: copy_matrix_deconproc_0 => copy_matrix
  USE lib, ONLY: T
  IMPLICIT NONE
  TYPE(T) :: s, s1
  REAL, DIMENSION(4, 4) :: a
  REAL :: b(4, 4)
  s % val = 1.0
  CALL copy_matrix_deconproc_0(s, a)
  CALL copy_scalar_deconproc_1(s, a(2, 2))
  CALL copy_vector_deconproc_2(s, b(:, 2))
  CALL copy_matrix_deconproc_3(s, b(:, :))
  CALL copy_vector_deconproc_4(s, s1 % val(:, 1))
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_convert_data_statements_into_assignments():
    sources, main = SourceCodeBuilder().add_file("""
subroutine fun(res)
  implicit none
  real :: val = 0.0
  real, dimension(2) :: d
  real, dimension(2), intent(out) :: res
  data val/1.0/, d/2*4.2/
  data d(1:2)/2*4.2/
  data d/5.1, 5.2/
  res(:) = val*d(:)
end subroutine fun

subroutine main(res)
  implicit none
  real, dimension(2) :: res
  call fun(res)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.convert_data_statements_into_assignments(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE fun(res)
  IMPLICIT NONE
  REAL :: val = 0.0
  REAL, DIMENSION(2) :: d
  REAL, DIMENSION(2), INTENT(OUT) :: res
  val = 1.0
  d(:) = 4.2
  d(1 : 2) = 4.2
  d(1) = 5.1
  d(2) = 5.2
  res(:) = val * d(:)
END SUBROUTINE fun
SUBROUTINE main(res)
  IMPLICIT NONE
  REAL, DIMENSION(2) :: res
  CALL fun(res)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_deconstruct_statement_functions():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(3, 4, 5)
  double precision :: ptare, rtt(2), foedelta, foeldcp
  double precision :: ralvdcp(2), ralsdcp(2), res
  foedelta(ptare) = max(0.0, sign(1.d0, ptare - rtt(1)))
  foeldcp(ptare) = foedelta(ptare)*ralvdcp(1) + (1.0 - foedelta(ptare))*ralsdcp(1)
  rtt(1) = 4.5
  ralvdcp(1) = 4.9
  ralsdcp(1) = 5.1
  d(1, 1, 1) = foeldcp(3.d0)
  res = foeldcp(3.d0)
  d(1, 1, 2) = res
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstruct_statement_functions(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  DOUBLE PRECISION :: d(3, 4, 5)
  DOUBLE PRECISION :: ptare, rtt(2)
  DOUBLE PRECISION :: ralvdcp(2), ralsdcp(2), res
  rtt(1) = 4.5
  ralvdcp(1) = 4.9
  ralsdcp(1) = 5.1
  d(1, 1, 1) = foeldcp(3.D0, rtt, ralvdcp, ralsdcp)
  res = foeldcp(3.D0, rtt, ralvdcp, ralsdcp)
  d(1, 1, 2) = res
  CONTAINS
  DOUBLE PRECISION FUNCTION foedelta(ptare, rtt)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: ptare
    DOUBLE PRECISION, INTENT(IN) :: rtt(2)
    foedelta = MAX(0.0, SIGN(1.D0, ptare - rtt(1)))
  END FUNCTION foedelta
  DOUBLE PRECISION FUNCTION foeldcp(ptare, rtt, ralvdcp, ralsdcp)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: ptare
    DOUBLE PRECISION, INTENT(IN) :: rtt(2)
    DOUBLE PRECISION, INTENT(IN) :: ralvdcp(2)
    DOUBLE PRECISION, INTENT(IN) :: ralsdcp(2)
    foeldcp = foedelta(ptare, rtt) * ralvdcp(1) + (1.0 - foedelta(ptare, rtt)) * ralsdcp(1)
  END FUNCTION foeldcp
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_goto_statements():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  real, intent(inout) :: d
  integer :: i

  ! forward-only gotos
  i = 0
  if (i > 5) go to 10000  ! not taken
  i = 7
  if (i > 5) goto 10001  ! taken
  i = 1
  if (i > 5) then
    goto 10002
    i = 9
  else if (i > 6) then
    i = 10
  else
    i = 11
  end if
10001 i = 6
10000 continue
  i = 2
10002 continue
  d = 7.1*i
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = desugaring.deconstuct_goto_statements(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  REAL, INTENT(INOUT) :: d
  INTEGER :: i
  LOGICAL :: goto_0
  LOGICAL :: goto_1
  LOGICAL :: goto_2
  i = 0
  goto_0 = .FALSE.
  IF (i > 5) goto_0 = .TRUE.
  IF (.NOT. (goto_0)) i = 7
  goto_1 = .FALSE.
  IF (.NOT. (goto_0) .AND. i > 5) goto_1 = .TRUE.
  IF (.NOT. (goto_1) .AND. .NOT. (goto_0)) i = 1
  goto_2 = .FALSE.
  IF (.NOT. (goto_1) .AND. .NOT. (goto_0) .AND. i > 5) THEN
    goto_2 = .TRUE.
    i = 9
  ELSE IF (.NOT. (goto_1) .AND. .NOT. (goto_0) .AND. i > 6) THEN
    i = 10
  ELSE IF (.NOT. (goto_1) .AND. .NOT. (goto_0)) THEN
    i = 11
  END IF
10001 CONTINUE
  IF (.NOT. (goto_2) .AND. .NOT. (goto_0)) i = 6
10000 CONTINUE
  IF (.NOT. (goto_2)) i = 2
10002 CONTINUE
  d = 7.1 * i
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_operator_overloading():
    sources, main = SourceCodeBuilder().add_file(
        """
module lib
  type cmplx
    real :: r = 1., i = 2.
  end type cmplx
  interface operator(+)
    module procedure :: add_cmplx
  end interface
contains
  function add_cmplx(a, b) result(c)
    type(cmplx), intent(in) :: a, b
    type(cmplx) :: c
    c%r = a%r + b%r
    c%i = a%i + b%i
  end function add_cmplx
end module lib

subroutine main
  use lib, only : cmplx, operator(+)
  type(cmplx) :: a, b
  b = a + a
end subroutine main
""", 'main').check_with_gfortran().get()
    ast = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
MODULE lib
  TYPE :: cmplx
    REAL :: r = 1., i = 2.
  END TYPE cmplx
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
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: cmplx, OPERATOR(+)
  TYPE(cmplx) :: a, b
  b = a + a
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()