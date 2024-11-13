# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Main_Program, Specification_Part, Execution_Part, \
    Subroutine_Subprogram, Call_Stmt, Subroutine_Stmt, \
    End_Subroutine_Stmt, Module_Stmt, Module, End_Module_Stmt, Module_Subprogram_Part, Contains_Stmt, Use_Stmt, \
    Assignment_Stmt, Interface_Block, Interface_Stmt, Program_Stmt, Implicit_Part, End_Program_Stmt
from fparser.two.Fortran2003 import Program
from fparser.two.Fortran2008 import Procedure_Stmt, Type_Declaration_Stmt
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.fortran_parser import recursive_ast_improver, simplified_dependency_graph, \
    recompute_children
from tests.fortran.fotran_test_helper import SourceCodeBuilder, FortranASTMatcher as M


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


def test_minimal():
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
    name_dict, rename_dict = recompute_children(ast, parse_order, simple_graph, actually_used_in_module)

    #  Since there was no module, it should be the exact same AST as the corresponding test in
    #  `recursive_ast_improver_test.py`.
    m = M(Program, [
        M(Main_Program, [
            M(Program_Stmt),  # program main
            M(Specification_Part, [
                M(Implicit_Part),  # implicit none
                M(Type_Declaration_Stmt),  # double precision d(4)
            ]),
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
            M(End_Program_Stmt),  # end program main
        ]),
    ])
    m.check(ast)

    # Verify
    assert not name_dict
    assert not rename_dict


def test_uses_module():
    """
    NOTE: We have a very similar test in `recursive_ast_improver_test.py`.
    A simple program that uses modules. A subroutine is defined in a module. The main program uses the module and calls
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
    ast, parse_order, simple_graph, actually_used_in_module, asts = parse_improve_and_simplify(sources)

    # Verify simplification of the dependency graph. This should already be the case from the corresponding test in
    # `recursive_ast_improver_test.py`.
    assert set(asts.keys()) == {'lib'}
    assert set(simple_graph.nodes) == {'main', 'lib'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'main': []}

    # Now the actual operation that we are testing.
    name_dict, rename_dict = recompute_children(ast, parse_order, simple_graph, actually_used_in_module)

    #  However, the AST should now be a little different from the original dependency graph computed in
    #  `recursive_ast_improver_test.py`. So, our matcher needs to reflect that too.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                M(Use_Stmt),  # use lib, only : fun
                *M.IGNORE(2),  # implicit none; double precision d(4)
            ]),
            M(Execution_Part, [M(Call_Stmt)]),  # call fun(d)
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M(Module_Stmt, [M.IGNORE(), M.NAMED('lib')]),  # module lib
            M(Module_Subprogram_Part, [
                M(Contains_Stmt),  # contains
                M(Subroutine_Subprogram, [
                    M(Subroutine_Stmt),  # subroutine fun(d)
                    M(Specification_Part),  # implicit none; double precision d(4)
                    M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
                    M(End_Subroutine_Stmt),  # end subroutine fun
                ]),
            ]),
            M(End_Module_Stmt),  # end module lib
        ]),
    ])
    m.check(ast)

    # Verify
    assert name_dict == {'lib': ['fun']}
    assert rename_dict == {'lib': {}}


def test_uses_module_which_uses_module():
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
    name_dict, rename_dict = recompute_children(ast, parse_order, simple_graph, actually_used_in_module)

    #  However, the AST should now be a little different from the original dependency graph computed in
    #  `recursive_ast_improver_test.py`. So, our matcher needs to reflect that too.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                M(Use_Stmt),  # use lib_indirect, only: fun_indirect
                *M.IGNORE(2),  # implicit none; double precision d(4)
            ]),
            M(Execution_Part, [M(Call_Stmt)]),  # call fun_indirect(d)
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M(Module_Stmt, [M.IGNORE(), M.NAMED('lib_indirect')]),
            # module lib_indirect
            M(Specification_Part, [M(Use_Stmt)]),  # use lib
            M(Module_Subprogram_Part, [
                # TODO: Why the `contains` node is gone?
                # M(Contains_Stmt),  # contains
                M(Subroutine_Subprogram),  # subroutine fun_indirect(d) ... end subroutine fun_indirect
            ]),
            M(End_Module_Stmt),  # end module lib_indirect
        ]),
        M(Module, [
            M(Module_Stmt, [M.IGNORE(), M.NAMED('lib')]),  # module lib
            M(Module_Subprogram_Part, [
                M(Contains_Stmt),  # contains
                M(Subroutine_Subprogram),  # subroutine fun(d) ... end subroutine fun
            ]),
            M(End_Module_Stmt),  # end module lib
        ]),
    ])
    m.check(ast)

    # Verify
    assert name_dict == {'lib': ['fun'], 'lib_indirect': ['fun_indirect']}
    assert rename_dict == {'lib': {}, 'lib_indirect': {}}


def test_module_contains_interface_block():
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
    name_dict, rename_dict = recompute_children(ast, parse_order, simple_graph, actually_used_in_module)

    #  However, the AST should now be a little different from the original dependency graph computed in
    #  `recursive_ast_improver_test.py`. So, our matcher needs to reflect that too.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [M.IGNORE()] * 3),
            # use lib_indirect, only : fun, fun2; implicit none; double precision d(4)
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = fun()
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M.IGNORE(),  # module lib_indirect
            M(Specification_Part, [
                *M.IGNORE(2),  # use lib, only: fun; implicit none
                M(Interface_Block, [
                    M(Interface_Stmt, [M.NAMED('xi')]),  # interface xi
                    M(Procedure_Stmt, [  # module procedure fun
                        M('Procedure_Name_List', [M.NAMED('fun')]),
                        *M.IGNORE(2),
                    ]),
                    M.IGNORE(),  # end interface xi
                ]),
            ]),
            M(Module_Subprogram_Part),  # contains; real function fun2(); implicit none; fun2 = 4.2; end function fun2
            M.IGNORE(),  # end module lib
        ]),
        M(Module, [
            *M.IGNORE(2),  # module lib; implicit none
            M(Module_Subprogram_Part),  # contains; real function fun(); implicit none; fun = 5.5; end function fun
            M.IGNORE(),  # end module lib
        ]),
    ])
    m.check(ast)

    # Verify
    assert name_dict == {'lib': ['fun'], 'lib_indirect': ['fun', 'fun2']}
    assert rename_dict == {'lib': {}, 'lib_indirect': {}}


def test_uses_module_but_prunes_unused_defs():
    """
    NOTE: We have a very similar test in `recursive_ast_improver_test.py`.
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

    # Verify simplification of the dependency graph. This should already be the case from the corresponding test in
    # `recursive_ast_improver_test.py`.
    assert set(asts.keys()) == {'lib'}
    assert set(simple_graph.nodes) == {'main', 'lib'}
    assert set(simple_graph.edges) == {('main', 'lib')}
    assert actually_used_in_module == {'lib': ['fun'], 'main': []}

    # Now the actual operation that we are testing.
    name_dict, rename_dict = recompute_children(ast, parse_order, simple_graph, actually_used_in_module)

    #  However, the AST should now be a little different from the original dependency graph computed in
    #  `recursive_ast_improver_test.py`. So, our matcher needs to reflect that too.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                M(Use_Stmt),  # use lib
                *M.IGNORE(2),  # implicit none; double precision d(4)
            ]),
            M(Execution_Part, [M(Call_Stmt)]),  # call fun(d)
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M(Module_Stmt, [M.IGNORE(), M.NAMED('lib')]),  # module lib
            M(Module_Subprogram_Part, [
                M(Contains_Stmt),  # contains
                M(Subroutine_Subprogram, [
                    M(Subroutine_Stmt, [M.IGNORE(), M.NAMED('fun'), *M.IGNORE(2)]),  # subroutine fun(d)
                    M(Specification_Part),  # implicit none; double precision d(4)
                    M(Execution_Part),  # d(2) = 5.5
                    M(End_Subroutine_Stmt),  # end subroutine fun
                ]),
                # NOTE: the `not_fun(d)` and `real_fun()` are gone!
            ]),
            M(End_Module_Stmt),  # end module lib
        ]),
    ])
    m.check(ast)

    # Verify
    assert name_dict == {'lib': ['fun']}
    assert rename_dict == {'lib': {}}
