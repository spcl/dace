# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.fortran_parser import recursive_ast_improver, simplified_dependency_graph, \
    prune_unused_children
from tests.fortran.fotran_test_helper import SourceCodeBuilder


def parse_improve_and_simplify(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    assert 'main.f90' in sources
    reader = FortranStringReader(sources['main.f90'])
    ast = parser(reader)
    assert isinstance(ast, Program)

    dep_graph = nx.DiGraph()
    asts = {}
    interface_blocks = {}
    ast = recursive_ast_improver(ast,
                                 sources,
                                 [],
                                 parser,
                                 interface_blocks,
                                 exclude_list=[],
                                 missing_modules=[],
                                 dep_graph=dep_graph,
                                 asts=asts)
    assert isinstance(ast, Program)
    assert not any(nx.simple_cycles(dep_graph))

    simple_graph, actually_used_in_module = simplified_dependency_graph(dep_graph, interface_blocks)
    parse_order = list(reversed(list(nx.topological_sort(simple_graph))))

    return ast, parse_order, simple_graph, actually_used_in_module, asts


def test_minimal_no_pruning():
    """
    NOTE: We have a very similar test in `recursive_ast_improver_test.py`.
    A minimal program that does not have any modules. So, `recompute_children()` should be a noop here.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  d(2) = 5.5
end program main
""").check_with_gfortran().get()
    ast, parse_order, simple_graph, actually_used_in_module, asts = parse_improve_and_simplify(sources)

    # Verify simplification of the dependency graph. This should already be the case from the corresponding test in
    # `recursive_ast_improver_test.py`.
    assert not asts
    assert not set(simple_graph.nodes)
    assert not actually_used_in_module

    # Now the actual operation that we are testing.
    name_dict, rename_dict = prune_unused_children(ast, parse_order, simple_graph, actually_used_in_module)

    #  Since there was no module, it should be the exact same AST as the corresponding test in
    #  `recursive_ast_improver_test.py`.
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

    # Verify
    assert not name_dict
    assert not rename_dict


def test_uses_module_which_uses_module_no_pruning():
    """
    NOTE: We have a very similar test in `recursive_ast_improver_test.py`.
    A simple program that uses modules, which in turn uses another module. The main program uses the module and calls
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
    ast, parse_order, simple_graph, actually_used_in_module, asts = parse_improve_and_simplify(sources)

    # Verify simplification of the dependency graph. This should already be the case from the corresponding test in
    # `recursive_ast_improver_test.py`.
    assert set(asts.keys()) == {'lib', 'lib_indirect'}
    assert set(simple_graph.nodes) == {'main', 'lib', 'lib_indirect'}
    assert set(simple_graph.edges) == {('main', 'lib_indirect'), ('lib_indirect', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'lib_indirect': ['fun_indirect', 'fun'], 'main': []}

    # Now the actual operation that we are testing.
    name_dict, rename_dict = prune_unused_children(ast, parse_order, simple_graph, actually_used_in_module)

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

    # Verify
    assert name_dict == {'lib': ['fun'], 'lib_indirect': ['fun_indirect']}
    assert rename_dict == {'lib': {}, 'lib_indirect': {}}


def test_module_contains_interface_block_no_pruning():
    """
    NOTE: We have a very similar test in `recursive_ast_improver_test.py`.
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
    ast, parse_order, simple_graph, actually_used_in_module, asts = parse_improve_and_simplify(sources)

    # Verify simplification of the dependency graph. This should already be the case from the corresponding test in
    # `recursive_ast_improver_test.py`.
    assert set(asts.keys()) == {'lib', 'lib_indirect'}
    assert set(simple_graph.nodes) == {'main', 'lib', 'lib_indirect'}
    assert set(simple_graph.edges) == {('main', 'lib_indirect'), ('lib_indirect', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'lib_indirect': ['fun', 'fun2'], 'main': []}

    # Now the actual operation that we are testing.
    name_dict, rename_dict = prune_unused_children(ast, parse_order, simple_graph, actually_used_in_module)

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

    # Verify
    assert name_dict == {'lib': ['fun'], 'lib_indirect': ['fun', 'fun2']}
    assert rename_dict == {'lib': {}, 'lib_indirect': {}}


def test_uses_module_but_prunes_unused_defs():
    """
    A simple program, but this time the subroutine is defined in a module, that also has some unused subroutine.
    The main program uses the module and calls the subroutine. So, we should have "recursively improved" the AST by
    parsing that module and constructing the dependency graph. Then after simplification, that unused subroutine should
    be gone from the dependency graph.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
contains
  subroutine fun(d)
    implicit none
    double precision d(4)
    d(2) = 5.5
  end subroutine fun
  subroutine not_fun(d)  ! `main` only uses `fun`, so this should be dropped after simplification
    implicit none
    double precision d(4)
    d(2) = 4.2
  end subroutine not_fun
  integer function real_fun()  ! `main` only uses `fun`, so this should be dropped after simplification
    implicit none
    real_fun = 4.7
  end function real_fun
end module lib
""").add_file("""
program main
  use lib, only: fun
  implicit none
  double precision d(4)
  call fun(d)
end program main
""").check_with_gfortran().get()
    ast, parse_order, simple_graph, actually_used_in_module, asts = parse_improve_and_simplify(sources)

    # Verify simplification of the dependency graph.
    assert set(asts.keys()) == {'lib'}
    assert set(simple_graph.nodes) == {'main', 'lib'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'main': []}

    # Now the actual operation that we are testing.
    name_dict, rename_dict = prune_unused_children(ast, parse_order, simple_graph, actually_used_in_module)

    # `not_fun` and `real_fun` should be gone!
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
  USE lib, ONLY: fun
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  CALL fun(d)
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()

    # Verify
    assert name_dict == {'lib': ['fun']}
    assert rename_dict == {'lib': {}}
