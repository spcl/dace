# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Program, Name
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.fortran_parser import recursive_ast_improver, simplified_dependency_graph, \
    create_sdfg_from_string
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


def test_minimal():
    """
    A minimal program with not much to "recursively improve".
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  d(2) = 5.5
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
PROGRAM main
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = 5.5
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # Verify that there is not much else to the program.
    assert not set(dep_graph.nodes)
    assert not interface_blocks
    assert not asts

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module


def test_toplevel_subroutine():
    """
    A simple program with not much to "recursively improve", but this time the subroutine is defined outside and called
    from the main program.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  call fun(d)
end program main

subroutine fun(d)
  implicit none
  double precision d(4)
  d(2) = 5.5
end subroutine fun
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
PROGRAM main
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  CALL fun(d)
END PROGRAM main
SUBROUTINE fun(d)
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = 5.5
END SUBROUTINE fun
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # Verify that there is not much else to the program.
    assert not set(dep_graph.nodes)
    assert not interface_blocks
    assert not asts

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module


def test_standalone_subroutine():
    """
    A standalone subroutine, with no program or module in sight.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine fun(d)
  implicit none
  double precision d(4)
  d(2) = 5.5
end subroutine fun
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
SUBROUTINE fun(d)
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = 5.5
END SUBROUTINE fun
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # Verify that there is not much else to the program.
    assert not set(dep_graph.nodes)
    assert not interface_blocks
    assert not asts

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module


def test_program_contains_subroutine():
    """
    The same simple program, but this time the subroutine is defined inside the main program that calls it.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  call fun(d)
contains
  subroutine fun(d)
    implicit none
    double precision d(4)
    d(2) = fun2()
  end subroutine fun
  real function fun2()
    implicit none
    fun2 = 5.5
  end function fun2
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
PROGRAM main
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  CALL fun(d)
  CONTAINS
  SUBROUTINE fun(d)
    IMPLICIT NONE
    DOUBLE PRECISION :: d(4)
    d(2) = fun2()
  END SUBROUTINE fun
  REAL FUNCTION fun2()
    IMPLICIT NONE
    fun2 = 5.5
  END FUNCTION fun2
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # Verify that there is not much else to the program.
    assert not set(dep_graph.nodes)
    assert not interface_blocks
    assert not asts

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module


def test_program_contains_interface_block():
    """
    The same simple program, but this time the subroutine is defined inside the main program that calls it.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none

  ! We can have an interface with no name
  interface
    real function fun()
      implicit none
    end function fun
  end interface

  ! We can even have multiple interfaces with no name
  interface
    real function fun2()
      implicit none
    end function fun2
  end interface

  double precision d(4)
  d(2) = fun()
end program main

real function fun()
  implicit none
  fun = 5.5
end function fun
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
PROGRAM main
  IMPLICIT NONE
  INTERFACE
    REAL FUNCTION fun()
      IMPLICIT NONE
    END FUNCTION fun
  END INTERFACE
  INTERFACE
    REAL FUNCTION fun2()
      IMPLICIT NONE
    END FUNCTION fun2
  END INTERFACE
  DOUBLE PRECISION :: d(4)
  d(2) = fun()
END PROGRAM main
REAL FUNCTION fun()
  IMPLICIT NONE
  fun = 5.5
END FUNCTION fun
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert interface_blocks == {
        'main': {'': [Name('fun'), Name('fun2')]},
    }
    # Verify that there is not much else to the program.
    assert not set(dep_graph.nodes)
    assert not asts

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module


def test_uses_module():
    """
    A simple program, but this time the subroutine is defined in a module. The main program uses the module and calls
    the subroutine. So, we should have "recursively improved" the AST by parsing that module and constructing the
    dependency graph.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
contains
  subroutine fun(d)
    implicit none
    double precision d(4)
    d(2) = 5.5
  end subroutine fun
end module lib
""").add_file("""
program main
  use lib
  implicit none
  double precision d(4)
  call fun(d)
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
MODULE lib
  CONTAINS
  SUBROUTINE fun(d)
    IMPLICIT NONE
    DOUBLE PRECISION :: d(4)
    d(2) = 5.5
  END SUBROUTINE fun
END MODULE lib
PROGRAM main
  USE lib
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  CALL fun(d)
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # This time we have a module dependency.
    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(asts.keys()) == {'lib'}

    # Verify that there is not much else to the program.
    assert not interface_blocks

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert set(simple_graph.nodes) == {'main', 'lib'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'main': []}


def test_uses_module_which_uses_module():
    """
    A simple program, but this time the subroutine is defined in a module. The main program uses the module and calls
    the subroutine. So, we should have "recursively improved" the AST by parsing that module and constructing the
    dependency graph.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
contains
  subroutine fun(d)
    implicit none
    double precision d(4)
    d(2) = 5.5
  end subroutine fun
end module lib
""").add_file("""
module lib_indirect
  use lib
contains
  subroutine fun_indirect(d)
    implicit none
    double precision d(4)
    call fun(d)
  end subroutine fun_indirect
end module lib_indirect
""").add_file("""
program main
  use lib_indirect, only: fun_indirect
  implicit none
  double precision d(4)
  call fun_indirect(d)
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
MODULE lib
  CONTAINS
  SUBROUTINE fun(d)
    IMPLICIT NONE
    DOUBLE PRECISION :: d(4)
    d(2) = 5.5
  END SUBROUTINE fun
END MODULE lib
MODULE lib_indirect
  USE lib
  CONTAINS
  SUBROUTINE fun_indirect(d)
    IMPLICIT NONE
    DOUBLE PRECISION :: d(4)
    CALL fun(d)
  END SUBROUTINE fun_indirect
END MODULE lib_indirect
PROGRAM main
  USE lib_indirect, ONLY: fun_indirect
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  CALL fun_indirect(d)
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # This time we have a module dependency.
    assert set(dep_graph.nodes) == {'lib', 'lib_indirect', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib_indirect'), ('lib_indirect', 'lib')}
    assert set(asts.keys()) == {'lib', 'lib_indirect'}

    # Verify that there is not much else to the program.
    assert not interface_blocks

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert set(simple_graph.nodes) == {'main', 'lib', 'lib_indirect'}
    assert set(simple_graph.edges) == {('main', 'lib_indirect'), ('lib_indirect', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'lib_indirect': ['fun_indirect', 'fun'], 'main': []}


def test_interface_block_contains_module_procedure():
    """
    The same simple program, but this time the subroutine is defined inside the main program that calls it.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
implicit none
contains
  real function fun()
    implicit none
    fun = 5.5
  end function fun
end module lib
""").add_file("""
program main
  use lib
  implicit none

  interface xi
    module procedure fun
  end interface xi

  double precision d(4)
  d(2) = fun()
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  CONTAINS
  REAL FUNCTION fun()
    IMPLICIT NONE
    fun = 5.5
  END FUNCTION fun
END MODULE lib
PROGRAM main
  USE lib
  IMPLICIT NONE
  INTERFACE xi
    MODULE PROCEDURE fun
  END INTERFACE xi
  DOUBLE PRECISION :: d(4)
  d(2) = fun()
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert set(dep_graph.nodes) == {'lib', 'main'}
    assert set(dep_graph.edges) == {('main', 'lib')}
    assert set(asts.keys()) == {'lib'}
    assert interface_blocks == {
        'main': {'xi': [Name('fun')]},
    }

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert set(simple_graph.nodes) == {'lib', 'main'}
    assert actually_used_in_module == {'lib': ['fun'], 'main': []}


def test_module_contains_interface_block():
    """
    The same simple program, but this time the subroutine is defined inside the main program that calls it.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
contains
  real function fun()
    implicit none
    fun = 5.5
  end function fun
end module lib
""").add_file("""
module lib_indirect
  use lib, only: fun
  implicit none
  interface xi
    module procedure fun
  end interface xi

contains
  real function fun2()
    implicit none
    fun2 = 4.2
  end function fun2
end module lib_indirect
""").add_file("""
program main
  use lib_indirect, only : fun, fun2
  implicit none

  double precision d(4)
  d(2) = fun()
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  CONTAINS
  REAL FUNCTION fun()
    IMPLICIT NONE
    fun = 5.5
  END FUNCTION fun
END MODULE lib
MODULE lib_indirect
  USE lib, ONLY: fun
  IMPLICIT NONE
  INTERFACE xi
    MODULE PROCEDURE fun
  END INTERFACE xi
  CONTAINS
  REAL FUNCTION fun2()
    IMPLICIT NONE
    fun2 = 4.2
  END FUNCTION fun2
END MODULE lib_indirect
PROGRAM main
  USE lib_indirect, ONLY: fun, fun2
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = fun()
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert set(dep_graph.nodes) == {'main', 'lib', 'lib_indirect'}
    assert set(dep_graph.edges) == {('main', 'lib_indirect'), ('lib_indirect', 'lib')}
    assert set(asts.keys()) == {'lib', 'lib_indirect'}
    assert interface_blocks == {
        'lib_indirect': {'xi': [Name('fun')]},
    }

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert set(simple_graph.nodes) == {'main', 'lib', 'lib_indirect'}
    assert set(simple_graph.edges) == {('main', 'lib_indirect'), ('lib_indirect', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'lib_indirect': ['fun', 'fun2'], 'main': []}


def test_program_contains_type():
    """
    A simple program that has a type defintion, and a function that uses it.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type

  real :: d(5, 5)
  call type_test_function(d)

contains

  subroutine type_test_function(d)
    real d(5, 5)
    type(simple_type) :: s
    s%w(1, 1, 1) = 5.5
    d(2, 1) = 5.5 + s%w(1, 1, 1)
  end subroutine type_test_function
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    got = ast.tofortran()
    want = """
PROGRAM main
  IMPLICIT NONE
  TYPE :: simple_type
    REAL :: w(5, 5, 5), z(5)
    INTEGER :: a
    REAL :: name
  END TYPE simple_type
  REAL :: d(5, 5)
  CALL type_test_function(d)
  CONTAINS
  SUBROUTINE type_test_function(d)
    REAL :: d(5, 5)
    TYPE(simple_type) :: s
    s % w(1, 1, 1) = 5.5
    d(2, 1) = 5.5 + s % w(1, 1, 1)
  END SUBROUTINE type_test_function
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    assert not set(dep_graph.nodes)
    assert not asts
    assert not interface_blocks

    # Verify simplification of the dependency graph.
    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph.copy(), interface_blocks)
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module
