from typing import Dict, Optional, Iterable

from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.ast_desugaring import correct_for_function_calls, deconstruct_enums, \
    deconstruct_interface_calls, deconstruct_procedure_calls, deconstruct_associations, \
    assign_globally_unique_subprogram_names, assign_globally_unique_variable_names, prune_branches, \
    const_eval_nodes, prune_unused_objects, inject_const_evals, ConstTypeInjection, ConstInstanceInjection, \
    make_practically_constant_arguments_constants, make_practically_constant_global_vars_constants, \
    exploit_locally_constant_variables, create_global_initializers, convert_data_statements_into_assignments, \
    deconstruct_statement_functions, deconstuct_goto_statements, SPEC, remove_access_and_bind_statements, \
    identifier_specs, alias_specs, consolidate_uses
from dace.frontend.fortran.fortran_parser import construct_full_ast
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str], entry_points: Optional[Iterable[SPEC]] = None):
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(sources, parser, entry_points=entry_points)
    ast = correct_for_function_calls(ast)
    assert isinstance(ast, Program)
    return ast


def test_spec_mapping():
    sources, main = SourceCodeBuilder().add_file("""
module lib  ! should be present
  abstract interface  ! should NOT be present
    subroutine fun  ! should be present
    end subroutine fun
  end interface
end module lib
""").check_with_gfortran().get()
    ast = construct_full_ast(sources, ParserFactory().create(std="f2008"))

    ident_map = identifier_specs(ast)
    assert ident_map.keys() == {('lib',), ('lib', 'fun')}

    alias_map = alias_specs(ast)
    assert alias_map.keys() == {('lib',), ('lib', 'fun')}


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

module lib2
  use lib
  implicit none
end module lib2

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


def test_uses_with_renames():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: pi4 = 9
  integer, parameter :: i4 = selected_int_kind(pi4)  ! `i4` will be const-evaluated to 4
end module lib

module main
contains
  subroutine fun(d)
    use lib, only: ik4 => i4  ! After const-evaluation, will be redundant.
    integer(ik4) :: i  ! `ik4` will also be const-evaluated to 4
    real, intent(out) :: d(2)
    i = 4
    d(2) = 5.5 + i
  end subroutine fun
end module main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    # A constant-evaluation will pin the constant values.
    ast = const_eval_nodes(ast)
    # A use-consolidation will remove the now-redundant use.
    ast = consolidate_uses(ast)
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: pi4 = 9
  INTEGER, PARAMETER :: i4 = 4
END MODULE lib
MODULE main
  CONTAINS
  SUBROUTINE fun(d)
    INTEGER(KIND = 4) :: i
    REAL, INTENT(OUT) :: d(2)
    i = 4
    d(2) = 5.5 + i
  END SUBROUTINE fun
END MODULE main
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
  type(used_config) :: garray(4)
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib

subroutine main
  use lib
  implicit none
  type(used_config) :: ucfg
  integer :: i = 7
  real :: a = 1
  ucfg%b = a*i
  garray(3)%b = a*i*2
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_unused_objects(ast, [('main',)])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: used_config
    REAL :: b = - 2.0
  END TYPE used_config
  TYPE(used_config) :: garray(4)
  CONTAINS
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: garray, used_config
  IMPLICIT NONE
  TYPE(used_config) :: ucfg
  INTEGER :: i = 7
  REAL :: a = 1
  ucfg % b = a * i
  garray(3) % b = a * i * 2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_completely_unsed_modules_are_pruned_early():
    sources, main = SourceCodeBuilder().add_file("""
module used
  implicit none
contains
  real function fun()
    fun = 1.
  end function fun
end module used

module unused
  implicit none
contains
  real function fun()
    fun = 2.
  end function fun
end module unused

subroutine main(d)
  use used
  implicit none
  real, intent(inout) :: d
  d = fun()
end subroutine main
""", 'main').check_with_gfortran().get()
    ast = parse_and_improve(sources, [('main',)])

    got = ast.tofortran()
    print(got)
    want = """
MODULE used
  IMPLICIT NONE
  CONTAINS
  REAL FUNCTION fun()
    fun = 1.
  END FUNCTION fun
END MODULE used
SUBROUTINE main(d)
  USE used
  IMPLICIT NONE
  REAL, INTENT(INOUT) :: d
  d = fun()
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


def test_config_injection_allocatable_fixing():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer, allocatable :: a(:, :)
  end type config
end module lib

subroutine main(cfg, b, c, d)
  use lib
  implicit none
  type(config), intent(in) :: cfg
  real, allocatable, intent(inout) :: b(:)
  real, allocatable, intent(inout) :: c(:, :)
  real, allocatable, intent(inout) :: d(:)
  real :: a = 1
  if (allocated(cfg%a)) a = 7.2
  if (allocated(b)) b = 7.2
  if (allocated(c)) c = 7.2
  if (allocated(d)) d = 7.2
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = inject_const_evals(ast, [
        ConstTypeInjection(None, ('lib', 'config'), ('a_a',), 'true'),
        ConstTypeInjection(None, ('lib', 'config'), ('a_d0_s',), '3'),
        ConstTypeInjection(None, ('lib', 'config'), ('a_o0_s',), '1'),
        ConstTypeInjection(None, ('lib', 'config'), ('a_d1_s',), '3'),
        ConstTypeInjection(None, ('lib', 'config'), ('a_o1_s',), '2'),
        ConstInstanceInjection(None, ('main', 'b_a'), tuple(), 'true'),
        ConstInstanceInjection(None, ('main', 'b_d0_s'), tuple(), '4'),
        ConstInstanceInjection(None, ('main', 'b_o0_s'), tuple(), '1'),
        ConstInstanceInjection(None, ('main', 'c_a'), tuple(), 'true'),
        ConstInstanceInjection(None, ('main', 'c_d0_s'), tuple(), '4'),
        ConstInstanceInjection(None, ('main', 'c_o0_s'), tuple(), '1'),
        ConstInstanceInjection(None, ('main', 'd_a'), tuple(), 'false'),
        ConstInstanceInjection(None, ('main', 'd_d0_s'), tuple(), '4'),
        ConstInstanceInjection(None, ('main', 'd_o0_s'), tuple(), '1'),
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER(KIND = 4) :: a(1 : 3, 2 : 4)
  END TYPE config
END MODULE lib
SUBROUTINE main(cfg, b, c, d)
  USE lib
  IMPLICIT NONE
  TYPE(config), INTENT(IN) :: cfg
  REAL(KIND = 4), INTENT(INOUT) :: b(1 : 4)
  REAL, ALLOCATABLE, INTENT(INOUT) :: c(:, :)
  REAL(KIND = 4), INTENT(INOUT) :: d(1 : 4)
  REAL :: a = 1
  IF (.TRUE.) a = 7.2
  IF (.TRUE.) b = 7.2
  IF (.TRUE.) c = 7.2
  IF (.FALSE.) d = 7.2
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
  integer :: i
  logical :: arr(5) = .true.

  ! `cond` is known in this block and doesn't change. `out` is unknown`, since it changes conditionally.
  if (cond) out = out + 1.
  out = out*2
  if (cond) then
    out = out + 1.
  else
    out = out - 1.
  end if

  ! `cond` is unknown after this, since it changes conditionally.
  if (out .gt. 20) cond = .false.
  if (cond) out = out + 100.

  ! `cond` is known again, and even `out` this time.
  cond = .true.
  out = 7.2
  out = out*2.0
  out = fun(.not. cond, out)

  ! A simple do loop with `i` as loop variable.
  do i=1, 20
    out = out + 1.
  end do
  ! TODO: `i` should be known at this point, since do loop is deterministic.
  i = i + 1

  ! A simple do-while loop with `i` as loop variable, `i` becomes unknown.
  i = 0
  do while (i < 10)
    out = out + 1
    i = i + 1
  end do

  ! Just making sure that `cond` is still known after all the loops.
  if (cond) out = out + 1.

  ! `cond` evaluation inside a branch should also happen.
  if (cond) then
    cond = .true.
    if (cond) then
      out = out + 1.
    else
      out = out + 7.
    end if
  end if

  ! The content of an array we don't track.
  arr = .false.
  do i=1, 5
    if (arrfun(arr) .or. arr(2)) then
      out = out + 3.14
    end if
  end do

contains
  real function fun(cond, out)
    implicit none
    logical, intent(in) :: cond
    real, intent(inout) :: out
    if (cond) out = out + 42
    fun = out + 1.0
  end function fun
  logical function arrfun(arr)
    implicit none
    logical, intent(in) :: arr(:)
    arrfun = arr(1)
  end function arrfun
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
  INTEGER :: i
  LOGICAL :: arr(5) = .TRUE.
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
  DO i = 1, 20
    out = out + 1.
  END DO
  i = i + 1
  i = 0
  DO WHILE (i < 10)
    out = out + 1
    i = i + 1
  END DO
  IF (.TRUE.) out = out + 1.
  IF (.TRUE.) THEN
    cond = .TRUE.
    IF (.TRUE.) THEN
      out = out + 1.
    ELSE
      out = out + 7.
    END IF
  END IF
  arr = .FALSE.
  DO i = 1, 5
    IF (arrfun(arr) .OR. arr(2)) THEN
      out = out + 3.14
    END IF
  END DO
  CONTAINS
  REAL FUNCTION fun(cond, out)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond
    REAL, INTENT(INOUT) :: out
    IF (cond) out = out + 42
    fun = out + 1.0
  END FUNCTION fun
  LOGICAL FUNCTION arrfun(arr)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: arr(:)
    arrfun = arr(1)
  END FUNCTION arrfun
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


def test_create_global_initializers():
    sources, main = SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = create_global_initializers(ast, [('main',)])

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


def test_convert_data_statements_into_assignments():
    sources, main = SourceCodeBuilder().add_file("""
subroutine fun(res)
  implicit none
  real :: val = 0.0
  real, dimension(2) :: d
  real, dimension(2), intent(out) :: res
  data val/1.0/, d/2*4.2/
  data d(1:2)/2*4.2/
  res(:) = val*d(:)
end subroutine fun

subroutine main(res)
  implicit none
  real, dimension(2) :: res
  call fun(res)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = convert_data_statements_into_assignments(ast)

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
    ast = deconstruct_statement_functions(ast)

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
    ast = deconstuct_goto_statements(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  REAL, INTENT(INOUT) :: d
  INTEGER :: i
  LOGICAL :: goto_0 = .FALSE.
  LOGICAL :: goto_1 = .FALSE.
  LOGICAL :: goto_2 = .FALSE.
  i = 0
  IF (i > 5) goto_0 = .TRUE.
  IF (.NOT. (goto_0)) i = 7
  IF (.NOT. (goto_0) .AND. i > 5) goto_1 = .TRUE.
  IF (.NOT. (goto_1) .AND. .NOT. (goto_0)) i = 1
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
    sources, main = SourceCodeBuilder().add_file("""
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
    print(got)
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


def test_remove_binds():
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = remove_access_and_bind_statements(ast)

    got = ast.tofortran()
    print(got)
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
    sources, main = SourceCodeBuilder().add_file("""
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
""", 'main').check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = remove_access_and_bind_statements(ast)

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
