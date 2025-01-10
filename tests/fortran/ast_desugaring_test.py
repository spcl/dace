from typing import Dict

from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.ast_desugaring import correct_for_function_calls, deconstruct_enums, \
    deconstruct_interface_calls, deconstruct_procedure_calls, deconstruct_associations, \
    assign_globally_unique_subprogram_names, assign_globally_unique_variable_names, prune_branches, \
    const_eval_nodes, prune_unused_objects, inject_const_evals, ConstTypeInjection, ConstInstanceInjection, \
    make_practically_constant_arguments_constants, make_practically_constant_global_vars_constants, \
    exploit_locally_constant_variables
from dace.frontend.fortran.fortran_parser import recursive_ast_improver
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    assert 'main.f90' in sources
    reader = FortranStringReader(sources['main.f90'])
    ast = parser(reader)
    ast = recursive_ast_improver(ast, sources, [], parser)
    ast = correct_for_function_calls(ast)
    assert isinstance(ast, Program)
    return ast


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
""").add_file("""
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
    ast = deconstruct_procedure_calls(ast)

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
""").add_file("""
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
    ast = deconstruct_procedure_calls(ast)

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
""").add_file("""
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
    ast = deconstruct_procedure_calls(ast)

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
""").add_file("""
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
""").add_file("""
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
    ast = deconstruct_procedure_calls(ast)

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
""").add_file("""
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
    ast = deconstruct_procedure_calls(ast)

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
""").add_file("""
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
    ast = deconstruct_associations(ast)

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
""").add_file("""
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
    ast = deconstruct_enums(ast)
    ast = deconstruct_associations(ast)
    ast = deconstruct_procedure_calls(ast)

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
""").add_file("""
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
    ast = deconstruct_associations(ast)
    ast = deconstruct_procedure_calls(ast)

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


def test_uses_allows_indirect_aliasing():
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
""").add_file("""
module lib2
  use lib
  implicit none
end module lib2
""").add_file("""
subroutine main
  use lib2, only: Square, perim
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
    ast = deconstruct_associations(ast)
    ast = deconstruct_procedure_calls(ast)

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
MODULE lib2
  USE lib
  IMPLICIT NONE
END MODULE lib2
SUBROUTINE main
  USE lib, ONLY: perim_deconproc_0 => perim
  USE lib2, ONLY: Square, perim
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
    ast = deconstruct_enums(ast)

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
""").add_file("""
subroutine main
  use lib, only: fun
  implicit none
  real d(4)
  d(2) = fun()
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = deconstruct_associations(ast)
    ast = correct_for_function_calls(ast)
    ast = deconstruct_procedure_calls(ast)

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
end module lib
""").add_file("""
subroutine main
  use lib, only: fun, not_fun
  implicit none
  real d(4)
  d(2) = fun()
  call not_fun(d(3))
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
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
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: not_real_fun_deconiface_1 => not_real_fun
  USE lib, ONLY: real_fun_deconiface_0 => real_fun
  IMPLICIT NONE
  REAL :: d(4)
  d(2) = real_fun_deconiface_0()
  CALL not_real_fun_deconiface_1(d(3))
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
""").add_file("""
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
    ast = deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
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
""").add_file("""
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
    ast = deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
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
  USE lib, ONLY: real_fun_deconiface_2 => real_fun
  USE lib, ONLY: integer_fun_deconiface_1 => integer_fun
  USE lib, ONLY: real_fun_deconiface_0 => real_fun
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
""").add_file("""
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
    ast = deconstruct_interface_calls(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
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
  USE lib, ONLY: real_fun_deconiface_2 => real_fun
  USE lib, ONLY: real_fun_deconiface_1 => real_fun
  USE lib, ONLY: real_fun_deconiface_0 => real_fun
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
""").add_file("""
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
    ast = deconstruct_procedure_calls(ast)

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


def test_globally_unique_names():
    sources, main = SourceCodeBuilder().add_file("""
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
""").add_file("""
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
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = assign_globally_unique_subprogram_names(ast, {('main',)})
    ast = assign_globally_unique_variable_names(ast, set())

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: sides(2, 2)
  END TYPE Square
  INTEGER, PARAMETER :: k_deconglobalvar_3 = 4
  REAL :: circle_deconglobalvar_4 = 2.0_k_deconglobalvar_3
  CONTAINS
  REAL FUNCTION perim_deconglobalfn_5(this_deconglobalvar_6, m_deconglobalvar_7)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this_deconglobalvar_6
    REAL, INTENT(IN) :: m_deconglobalvar_7
    perim_deconglobalfn_5 = m_deconglobalvar_7 * SUM(this_deconglobalvar_6 % sides)
  END FUNCTION perim_deconglobalfn_5
  FUNCTION area_deconglobalfn_8(this_deconglobalvar_9, m_deconglobalvar_10)
    IMPLICIT NONE
    CLASS(Square), INTENT(IN) :: this_deconglobalvar_9
    REAL, INTENT(IN) :: m_deconglobalvar_10
    REAL, DIMENSION(2, 2) :: area_deconglobalfn_8
    area_deconglobalfn_8 = m_deconglobalvar_10 * SUM(this_deconglobalvar_9 % sides)
  END FUNCTION area_deconglobalfn_8
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: circle_deconglobalvar_4
  USE lib, ONLY: area_deconglobalfn_8
  USE lib, ONLY: perim_deconglobalfn_5
  USE lib, ONLY: perim_deconglobalfn_5
  USE lib
  IMPLICIT NONE
  TYPE(Square) :: s_deconglobalvar_13
  REAL :: a_deconglobalvar_14
  INTEGER :: i_deconglobalvar_15, j_deconglobalvar_16
  s_deconglobalvar_13 % sides = 0.5
  s_deconglobalvar_13 % sides(1, 1) = 1.0
  s_deconglobalvar_13 % sides(2, 1) = 1.0
  DO i_deconglobalvar_15 = 1, 2
    DO j_deconglobalvar_16 = 1, 2
      s_deconglobalvar_13 % sides(i_deconglobalvar_15, j_deconglobalvar_16) = 7.0
    END DO
  END DO
  a_deconglobalvar_14 = perim_deconglobalfn_5(s_deconglobalvar_13, 1.0)
  a_deconglobalvar_14 = perim_deconglobalfn_5(s_deconglobalvar_13, 1.0)
  s_deconglobalvar_13 % sides = area_deconglobalfn_8(s_deconglobalvar_13, 4.1)
  circle_deconglobalvar_4 = 5.0
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_branch_pruning():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 4
  integer :: a = -1, b = -1

  if (k < 2) then
    a = k
  else if (k < 5) then
    b = k
  else
    a = k
    b = k
  end if
  if (k < 5) a = 70 + k
  if (k > 5) a = 70 - k
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_branches(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 4
  INTEGER :: a = - 1, b = - 1
  b = k
  a = 70 + k
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_object_pruning():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer :: a = 8
    real :: b = 2.0
    logical :: c = .false.
  end type config
  type used_config
    integer :: a = -1
    real :: b = -2.0
  end type used_config
  type big_config
    type(config) :: big
  end type big_config
  type(config) :: globalo
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib
""").add_file("""
subroutine main
  use lib
  implicit none
  type(used_config) :: ucfg
  integer :: i = 7
  real :: a = 1
  ucfg%b = a*i
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_unused_objects(ast, [('main',)])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: used_config
    INTEGER :: a = - 1
    REAL :: b = - 2.0
  END TYPE used_config
  CONTAINS
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: used_config
  IMPLICIT NONE
  TYPE(used_config) :: ucfg
  INTEGER :: i = 7
  REAL :: a = 1
  ucfg % b = a * i
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_constant_resolving_expressions():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 8
  integer :: a = -1, b = -1
  real, parameter :: pk = 4.1_k
  real(kind=selected_real_kind(5, 5)) :: p = 1.0_k

  if (k < 2) then
    a = k
    p = k*pk
  else if (k < 5) then
    b = k
    p = p + k*pk
  else
    a = k
    b = k
    p = a*p + k*pk
  end if
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 8
  INTEGER :: a = - 1, b = - 1
  REAL, PARAMETER :: pk = 4.1D0
  REAL(KIND = 4) :: p = 1.0D0
  IF (.FALSE.) THEN
    a = 8
    p = 32.79999923706055D0
  ELSE IF (.FALSE.) THEN
    b = 8
    p = p + 32.79999923706055D0
  ELSE
    a = 8
    b = 8
    p = a * p + 32.79999923706055D0
  END IF
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_constant_resolving_non_expressions():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 8
  integer :: i
  real :: a = 1
  do i = 2, k
    a = a + i * k
  end do
  a = fun(k)
  call not_fun(k, a)
  contains
  real function fun(x)
    integer, intent(in) :: x
    fun = x * k
  end function fun
  subroutine not_fun(x, y)
    integer, intent(in) :: x
    real, intent(out) :: y
    y = x * k
  end subroutine not_fun
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 8
  INTEGER :: i
  REAL :: a = 1
  DO i = 2, 8
    a = a + i * 8
  END DO
  a = fun(8)
  CALL not_fun(8, a)
  CONTAINS
  REAL FUNCTION fun(x)
    INTEGER, INTENT(IN) :: x
    fun = x * 8
  END FUNCTION fun
  SUBROUTINE not_fun(x, y)
    INTEGER, INTENT(IN) :: x
    REAL, INTENT(OUT) :: y
    y = x * 8
  END SUBROUTINE not_fun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_type():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer :: a = 8
    real :: b = 2.0
    logical :: c = .false.
  end type config
  type big_config
    type(config) :: big
  end type big_config
  type(config) :: globalo
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib
""").add_file("""
subroutine main(cfg)
  use lib
  implicit none
  type(big_config), intent(in) :: cfg
  real :: a = 1
  a = cfg%big%b + a * globalo%a
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = inject_const_evals(ast, [
        ConstTypeInjection(None, ('lib', 'config'), ('a',), '42'),
        ConstTypeInjection(None, ('lib', 'config'), ('b',), '10000.0')
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER :: a = 8
    REAL :: b = 2.0
    LOGICAL :: c = .FALSE.
  END TYPE config
  TYPE :: big_config
    TYPE(config) :: big
  END TYPE big_config
  TYPE(config) :: globalo
  CONTAINS
  SUBROUTINE fun(this)
    IMPLICIT NONE
    TYPE(config), INTENT(INOUT) :: this
    this % b = 5.1
  END SUBROUTINE fun
END MODULE lib
SUBROUTINE main(cfg)
  USE lib
  IMPLICIT NONE
  TYPE(big_config), INTENT(IN) :: cfg
  REAL :: a = 1
  a = 10000.0 + a * 42
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_instance():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer :: a = 8
    real :: b = 2.0
    logical :: c = .false.
  end type config
  type big_config
    type(config) :: big
  end type big_config
  type(config) :: globalo
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib
""").add_file("""
subroutine main(cfg)
  use lib
  implicit none
  type(big_config), intent(in) :: cfg
  real :: a = 1
  a = cfg%big%b + a * globalo%a
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = inject_const_evals(ast, [
        ConstInstanceInjection(None, ('lib', 'globalo'), ('a',), '42'),
        ConstInstanceInjection(None, ('main', 'cfg'), ('big', 'b'), '10000.0')
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER :: a = 8
    REAL :: b = 2.0
    LOGICAL :: c = .FALSE.
  END TYPE config
  TYPE :: big_config
    TYPE(config) :: big
  END TYPE big_config
  TYPE(config) :: globalo
  CONTAINS
  SUBROUTINE fun(this)
    IMPLICIT NONE
    TYPE(config), INTENT(INOUT) :: this
    this % b = 5.1
  END SUBROUTINE fun
END MODULE lib
SUBROUTINE main(cfg)
  USE lib
  IMPLICIT NONE
  TYPE(big_config), INTENT(IN) :: cfg
  REAL :: a = 1
  a = 10000.0 + a * 42
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_array():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer, allocatable :: a(:, :)
  end type config
contains
  real function fun(this)
    implicit none
    type(config), intent(inout) :: this
    if (allocated(this%a)) then  ! This will be replaced even though it is an out (i.e., beware of invalid injections).
      fun = 5.1
    else
      fun = -1
    endif
  end function fun
end module lib
""").add_file("""
subroutine main(cfg)
  use lib
  implicit none
  type(config), intent(in) :: cfg
  real :: a = 1
  if (allocated(cfg%a)) a = 7.2
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = inject_const_evals(ast, [
        ConstTypeInjection(None, ('lib', 'config'), ('a_a',), 'true'),
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER, ALLOCATABLE :: a(:, :)
  END TYPE config
  CONTAINS
  REAL FUNCTION fun(this)
    IMPLICIT NONE
    TYPE(config), INTENT(INOUT) :: this
    IF (.TRUE.) THEN
      fun = 5.1
    ELSE
      fun = - 1
    END IF
  END FUNCTION fun
END MODULE lib
SUBROUTINE main(cfg)
  USE lib
  IMPLICIT NONE
  TYPE(config), INTENT(IN) :: cfg
  REAL :: a = 1
  IF (.TRUE.) a = 7.2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_practically_constant_arguments():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
contains
  real function fun(cond, kwcond, opt)
    implicit none
    logical, intent(in) :: cond, kwcond
    logical, optional, intent(in) :: opt
    logical :: real_opt = .false.
    if (present(opt)) then
      real_opt = opt
    end if
    if (cond .and. kwcond .and. real_opt) then
      fun = -2.7
    else
      fun = 4.2
    end if
  end function fun

  real function not_fun(cond, kwcond, opt)
    implicit none
    logical, intent(in) :: cond, kwcond
    logical, optional, intent(in) :: opt
    logical :: real_opt = .false.
    if (present(opt)) then
      real_opt = opt
    end if
    if (cond .and. kwcond .and. real_opt) then
      not_fun = -500.1
    else
      not_fun = 9600.8
    end if
  end function not_fun

  subroutine user_1()
    implicit none
    real :: c
    c = fun(.false., kwcond=.false., opt=.true.)*not_fun(.false., kwcond=.false., opt=.false.)
  end subroutine user_1

  subroutine user_2()
    implicit none
    real :: c
    c = 3*fun(.false., kwcond=.false., opt=.true.)*not_fun(.true., kwcond=.true., opt=.true.)
  end subroutine user_2
end module lib
""").add_file("""
subroutine main()
  use lib
  implicit none
  call user_1()
  call user_2()
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = make_practically_constant_arguments_constants(ast, [('main',)])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  CONTAINS
  REAL FUNCTION fun(cond, kwcond, opt)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond, kwcond
    LOGICAL, OPTIONAL, INTENT(IN) :: opt
    LOGICAL :: real_opt = .FALSE.
    IF (.TRUE.) THEN
      real_opt = .TRUE.
    END IF
    IF (.FALSE. .AND. .FALSE. .AND. real_opt) THEN
      fun = - 2.7
    ELSE
      fun = 4.2
    END IF
  END FUNCTION fun
  REAL FUNCTION not_fun(cond, kwcond, opt)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond, kwcond
    LOGICAL, OPTIONAL, INTENT(IN) :: opt
    LOGICAL :: real_opt = .FALSE.
    IF (.TRUE.) THEN
      real_opt = opt
    END IF
    IF (cond .AND. kwcond .AND. real_opt) THEN
      not_fun = - 500.1
    ELSE
      not_fun = 9600.8
    END IF
  END FUNCTION not_fun
  SUBROUTINE user_1
    IMPLICIT NONE
    REAL :: c
    c = fun(.FALSE., kwcond = .FALSE., opt = .TRUE.) * not_fun(.FALSE., kwcond = .FALSE., opt = .FALSE.)
  END SUBROUTINE user_1
  SUBROUTINE user_2
    IMPLICIT NONE
    REAL :: c
    c = 3 * fun(.FALSE., kwcond = .FALSE., opt = .TRUE.) * not_fun(.TRUE., kwcond = .TRUE., opt = .TRUE.)
  END SUBROUTINE user_2
END MODULE lib
SUBROUTINE main
  USE lib
  IMPLICIT NONE
  CALL user_1
  CALL user_2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_practically_constant_global_vars_constants():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  logical :: fixed_cond = .false.
  logical :: movable_cond = .false.
contains
  subroutine update(what)
    implicit none
    logical, intent(out) :: what
    what = .true.
  end subroutine update
end module lib
""").add_file("""
subroutine main
  use lib
  implicit none
  real :: a = 1.0
  call update(movable_cond)
  movable_cond = .not. movable_cond
  if (fixed_cond .and. movable_cond) a = 7.1
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = make_practically_constant_global_vars_constants(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  LOGICAL, PARAMETER :: fixed_cond = .FALSE.
  LOGICAL :: movable_cond = .FALSE.
  CONTAINS
  SUBROUTINE update(what)
    IMPLICIT NONE
    LOGICAL, INTENT(OUT) :: what
    what = .TRUE.
  END SUBROUTINE update
END MODULE lib
SUBROUTINE main
  USE lib
  IMPLICIT NONE
  REAL :: a = 1.0
  CALL update(movable_cond)
  movable_cond = .NOT. movable_cond
  IF (fixed_cond .AND. movable_cond) a = 7.1
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_exploit_locally_constant_variables():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main()
  implicit none
  logical :: cond = .true.
  real :: out = 0.

  if (cond) out = out + 1.
  out = out*2
  if (cond) then
    out = out + 1.
  else
    out = out - 1.
  end if

  if (out .gt. 20) cond = .false.
  if (cond) out = out + 100.

  cond = .true.
  out = 7.2
  out = out*2.0
  out = fun(.not. cond, out)

  if (cond) out = out + 1.

contains
  real function fun(cond, out)
    implicit none
    logical, intent(in) :: cond
    real, intent(inout) :: out
    if (cond) out = out + 42
    fun = out + 1.0
  end function fun
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  LOGICAL :: cond = .TRUE.
  REAL :: out = 0.
  IF (.TRUE.) out = 0. + 1.
  out = out * 2
  IF (.TRUE.) THEN
    out = out + 1.
  ELSE
    out = out - 1.
  END IF
  IF (out .GT. 20) cond = .FALSE.
  IF (cond) out = out + 100.
  cond = .TRUE.
  out = 7.2
  out = 7.2 * 2.0
  out = fun(.NOT. .TRUE., out)
  IF (.TRUE.) out = out + 1.
  CONTAINS
  REAL FUNCTION fun(cond, out)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond
    REAL, INTENT(INOUT) :: out
    IF (cond) out = out + 42
    fun = out + 1.0
  END FUNCTION fun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_exploit_locally_constant_struct_members():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main()
  implicit none
  type config
    logical :: cond = .true.
  end type config
  type(config) :: cond
  real :: out = 0.

  cond % cond = .true.
  if (cond % cond) out = out + 1.
  out = out*2
  if (cond % cond) then
    out = out + 1.
  else
    out = out - 1.
  end if

  if (out .gt. 20) cond % cond = .false.
  if (cond % cond) out = out + 100.

  cond % cond = .true.
  out = 7.2
  out = out*2.0
  out = fun(.not. cond % cond, out)

  if (cond % cond) out = out + 1.

contains
  real function fun(cond, out)
    implicit none
    logical, intent(in) :: cond
    real, intent(inout) :: out
    if (cond) out = out + 42
    fun = out + 1.0
  end function fun
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  TYPE :: config
    LOGICAL :: cond = .TRUE.
  END TYPE config
  TYPE(config) :: cond
  REAL :: out = 0.
  cond % cond = .TRUE.
  IF (.TRUE.) out = 0. + 1.
  out = out * 2
  IF (.TRUE.) THEN
    out = out + 1.
  ELSE
    out = out - 1.
  END IF
  IF (out .GT. 20) cond % cond = .FALSE.
  IF (cond % cond) out = out + 100.
  cond % cond = .TRUE.
  out = 7.2
  out = 7.2 * 2.0
  out = fun(.NOT. .TRUE., out)
  IF (.TRUE.) out = out + 1.
  CONTAINS
  REAL FUNCTION fun(cond, out)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond
    REAL, INTENT(INOUT) :: out
    IF (cond) out = out + 42
    fun = out + 1.0
  END FUNCTION fun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()
