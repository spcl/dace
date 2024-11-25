from typing import Dict

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Program, Name, Rename
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.fortran_parser import deconstruct_procedure_calls, recursive_ast_improver, prune_unused_children, simplified_dependency_graph
from tests.fortran.fotran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    assert 'main.f90' in sources
    reader = FortranStringReader(sources['main.f90'])
    ast = parser(reader)
    assert isinstance(ast, Program)

    ast, dep_graph, interface_blocks, asts = recursive_ast_improver(ast, sources, [], parser)
    assert isinstance(ast, Program)
    assert not any(nx.simple_cycles(dep_graph))
    return ast, dep_graph, interface_blocks, asts


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
    area = m * this%side*this%side
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
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)
    ast, dep_graph = deconstruct_procedure_calls(ast, dep_graph)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: Square
    REAL :: side
    CONTAINS
    PROCEDURE :: area
    PROCEDURE :: area_alt => area
    PROCEDURE :: get_area
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
  USE lib, ONLY: get_area
  USE lib, ONLY: area
  USE lib, ONLY: area
  USE lib, ONLY: Square
  IMPLICIT NONE
  TYPE(Square) :: s
  REAL :: a
  s % side = 1.0
  a = area(s, 1.0)
  a = area(s, 1.0)
  CALL get_area(s, a)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in dep_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area', 'get_area'}
    assert not interface_blocks
    assert set(asts.keys()) == {'lib'}

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)

    # Nothing changed here.
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert set(u.string for u in simple_graph.edges['main', 'lib']['obj_list']) == {'Square', 'area', 'get_area'}
    assert not interface_blocks
    assert set(asts.keys()) == {'lib'}

    assert ({k: set(v) for k, v in actually_used_in_module.items()}
            == {'lib': {'get_area', 'area', 'Square'}, 'main': set()})

    name_dict, rename_dict = prune_unused_children(ast, simple_graph, actually_used_in_module)
    got = ast.tofortran()
    # Still want the same program, because nothing should have been pruned.
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert name_dict == {'lib': ['Square', 'area', 'get_area']}
    assert not rename_dict == {'lib': []}
