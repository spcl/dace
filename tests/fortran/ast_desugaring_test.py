from typing import Dict

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.ast_utils import UseAllPruneList
from dace.frontend.fortran.fortran_parser import deconstruct_procedure_calls, recursive_ast_improver, \
    prune_unused_children, simplified_dependency_graph, deconstruct_associations, correct_for_function_calls, \
    deconstruct_enums, deconstruct_interface_calls
from tests.fortran.fotran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    assert 'main.f90' in sources
    reader = FortranStringReader(sources['main.f90'])
    ast = parser(reader)
    assert isinstance(ast, Program)

    ast, dep_graph, interface_blocks = recursive_ast_improver(ast, sources, [], parser)
    assert isinstance(ast, Program)
    assert not any(nx.simple_cycles(dep_graph))
    return ast, dep_graph, interface_blocks


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area', 'get_area'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area', 'get_area'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'get_area', 'area', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'area', 'get_area']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'get_area'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'get_area'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'get_area', 'Square', 'get_value', 'Value'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'get_area']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'area', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'area']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib_1', 'lib_2', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib_1'), ('main', 'lib_2')}
    assert set(u.string for u in dep_graph.edges['main', 'lib_1']['obj_list']) == {'Square', 'area'}
    assert set(u.string for u in dep_graph.edges['main', 'lib_2']['obj_list']) == {'Circle', 'area'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib_1', 'lib_2', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib_1'), ('main', 'lib_2')}
    assert set(u.string for u in dep_graph.edges['main', 'lib_1']['obj_list']) == {'Square', 'area'}
    assert set(u.string for u in dep_graph.edges['main', 'lib_2']['obj_list']) == {'Circle', 'area'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib_1': {'area', 'Square'}, 'lib_2': {'area', 'Circle'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib_1': ['Square', 'area'], 'lib_2': ['Circle', 'area']}
    assert rename_dict == {'lib_1': {}, 'lib_2': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = correct_for_function_calls(ast)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area_integer', 'area_real'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area_integer',
                                                                                    'area_real'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'Square', 'area_integer', 'area_real'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'area_real', 'area_integer']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'area', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'area']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = deconstruct_enums(ast)
    ast = deconstruct_associations(ast)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'perim'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'perim'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'perim', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'perim']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = deconstruct_associations(ast)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'perim'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'perim'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'perim', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'perim']}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = deconstruct_associations(ast)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'lib2', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib2'), ('lib2', 'lib')}
    assert (set(
        '*' if isinstance(u, UseAllPruneList) else u.string for u in dep_graph.edges['main', 'lib2']['obj_list'])
            == {'Square', 'perim'})
    assert (set('*' if isinstance(u, UseAllPruneList) else u.string for u in dep_graph.edges['lib2', 'lib']['obj_list'])
            == {'*'})
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(dep_graph.nodes) == {'lib', 'lib2', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib2'), ('lib2', 'lib')}
    assert (set(
        '*' if isinstance(u, UseAllPruneList) else u.string for u in dep_graph.edges['main', 'lib2']['obj_list'])
            == {'Square', 'perim'})
    assert (set('*' if isinstance(u, UseAllPruneList) else u.string for u in dep_graph.edges['lib2', 'lib']['obj_list'])
            == {'*'})
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'perim', 'Square'}, 'lib2': {'perim', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert {k: set(v) for k, v in name_dict.items()} == {'lib': {'Square', 'perim'}, 'lib2': {'Square', 'perim'}}
    assert rename_dict == {'lib': {}, 'lib2': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
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

    assert not set(dep_graph.nodes)
    assert not set(dep_graph.edges)
    assert not interface_blocks


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = deconstruct_associations(ast)
    ast = correct_for_function_calls(ast)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert ({k1: {k2: {v.string for v in vs} for k2, vs in k2v.items()} for k1, k2v in interface_blocks.items()}
            == {'lib': {'fun': {'real_fun'}}})

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert ({k1: {k2: {v.string for v in vs} for k2, vs in k2v.items()} for k1, k2v in interface_blocks.items()}
            == {'lib': {'fun': {'real_fun'}}})
    assert ({k: set(v) for k, v in actually_used_in_module.items()} == {'lib': {'real_fun'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert {k: set(v) for k, v in name_dict.items()} == {'lib': {'real_fun'}}
    assert rename_dict == {'lib': {}}


def test_interface_replacer_with_module_procedures():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  public :: fun
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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = correct_for_function_calls(ast)
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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert ({k1: {k2: {v.string for v in vs} for k2, vs in k2v.items()} for k1, k2v in interface_blocks.items()}
            == {'lib': {'fun': {'real_fun'}, 'not_fun': {'not_real_fun'}}})

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert ({k1: {k2: {v.string for v in vs} for k2, vs in k2v.items()} for k1, k2v in interface_blocks.items()}
            == {'lib': {'fun': {'real_fun'}, 'not_fun': {'not_real_fun'}}})
    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'real_fun', 'not_real_fun'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert {k: set(v) for k, v in name_dict.items()} == {'lib': {'real_fun', 'not_real_fun'}}
    assert rename_dict == {'lib': {}}


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
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast = correct_for_function_calls(ast)
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

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert not interface_blocks
    assert {k: set(v) for k, v in actually_used_in_module.items()} == {'main': set(), 'lib': {'no_fun', 'fun'}}

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert {k: set(v) for k, v in name_dict.items()} == {'lib': {'fun'}}
    assert rename_dict == {'lib': {'fun': 'no_fun'}}


def test_generic_replacer_deducing_array_types():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type T
    real :: val(2, 2)
  contains
    procedure :: copy_array
    procedure :: copy_scalar
    generic :: copy => copy_array, copy_scalar
  end type T
contains
  subroutine copy_scalar(this, m)
    implicit none
    class(T), intent(in) :: this
    real, intent(out) :: m
    m = this%val(1,1)
  end subroutine copy_scalar
  subroutine copy_array(this, m)
    implicit none
    class(T), intent(in) :: this
    real, dimension(:, :), intent(out) :: m
    m = this%val(1,1)
  end subroutine copy_array
end module lib
""").add_file("""
subroutine main
  use lib, only: T
  implicit none
  type(T) :: s
  real, dimension(4, 4) :: a
  real :: b(4, 4)

  s%val = 1.0
  call s%copy(a)
  call s%copy(a(2,2))
  call s%copy(b(:,:))
end subroutine main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks = parse_and_improve(sources)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

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
  SUBROUTINE copy_array(this, m)
    IMPLICIT NONE
    CLASS(T), INTENT(IN) :: this
    REAL, DIMENSION(:, :), INTENT(OUT) :: m
    m = this % val(1, 1)
  END SUBROUTINE copy_array
END MODULE lib
SUBROUTINE main
  USE lib, ONLY: copy_array_deconproc_2 => copy_array
  USE lib, ONLY: copy_scalar_deconproc_1 => copy_scalar
  USE lib, ONLY: copy_array_deconproc_0 => copy_array
  USE lib, ONLY: T
  IMPLICIT NONE
  TYPE(T) :: s
  REAL, DIMENSION(4, 4) :: a
  REAL :: b(4, 4)
  s % val = 1.0
  CALL copy_array_deconproc_0(s, a)
  CALL copy_scalar_deconproc_1(s, a(2, 2))
  CALL copy_array_deconproc_2(s, b(:, :))
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'T', 'copy_array', 'copy_scalar'}
    assert not interface_blocks

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'T', 'copy_array', 'copy_scalar'}
    assert not interface_blocks

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'copy_array', 'copy_scalar', 'T'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['T', 'copy_array', 'copy_scalar']}
    assert rename_dict == {'lib': {}}
