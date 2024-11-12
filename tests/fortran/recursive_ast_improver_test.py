# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict

import networkx as nx
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Main_Program, Program, Program_Stmt, Specification_Part, Execution_Part, \
    End_Program_Stmt, Implicit_Part, Assignment_Stmt, Subroutine_Subprogram, Call_Stmt, Subroutine_Stmt, \
    End_Subroutine_Stmt, Module_Stmt, Module, End_Module_Stmt, Module_Subprogram_Part, Contains_Stmt, Use_Stmt, Name, \
    Internal_Subprogram_Part, Function_Subprogram, Function_Stmt, End_Function_Stmt, Interface_Block, Interface_Stmt
from fparser.two.Fortran2008 import Type_Declaration_Stmt, Procedure_Stmt
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.fortran_parser import recursive_ast_improver, simplified_dependency_graph
from tests.fortran.fotran_test_helper import SourceCodeBuilder, FortranASTMatcher as M


def parse_and_improve(sources: Dict[str, str]):
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

    # A pretty thorough matcher to make sure that we got all the parts of this simple program correct.
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

    # A matcher focused on correctly parsing the subroutine definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part),  # implicit none; double precision d(4)
            M(Execution_Part, [M(Call_Stmt)]),  # call fun(d)
            M.IGNORE(),  # end program main
        ]),
        M(Subroutine_Subprogram, [
            M(Subroutine_Stmt),  # subroutine fun(d)
            M(Specification_Part, [
                M(Implicit_Part),  # implicit none
                M(Type_Declaration_Stmt),  # double precision d(4)
            ]),
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
            M(End_Subroutine_Stmt),  # end subroutine fun
        ]),
    ])
    m.check(ast)

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

    # A matcher focused on correctly parsing the subroutine definitions and uses.
    m = M(Program, [
        M(Subroutine_Subprogram, [
            M(Subroutine_Stmt),  # subroutine fun(d)
            M(Specification_Part, [
                M(Implicit_Part),  # implicit none
                M(Type_Declaration_Stmt),  # double precision d(4)
            ]),
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
            M(End_Subroutine_Stmt),  # end subroutine fun
        ]),
    ])
    m.check(ast)

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
    d(2) = 5.5
  end subroutine fun
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    # A matcher focused on correctly parsing the subroutine definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part),  # implicit none; double precision d(4)
            M(Execution_Part, [M(Call_Stmt)]),  # call fun(d)
            M(Internal_Subprogram_Part, [
                M(Contains_Stmt),  # contains
                M(Subroutine_Subprogram, [
                    M(Subroutine_Stmt),  # subroutine fun(d)
                    M(Specification_Part, [
                        M.IGNORE(),  # implicit none
                        M(Type_Declaration_Stmt),  # double precision d(4)
                    ]),
                    M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
                    M(End_Subroutine_Stmt),  # end subroutine fun
                ]),
            ]),
            M.IGNORE(),  # end program main
        ]),
    ])
    m.check(ast)

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

    # A matcher focused on correctly parsing the subroutine definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part),  # implicit none; double precision d(4)
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = fun()
            M.IGNORE(),  # end program main
        ]),
        M(Function_Subprogram, [
            M(Function_Stmt),  # subroutine fun()
            M(Specification_Part, [
                M(Implicit_Part),  # implicit none
            ]),
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
            M(End_Function_Stmt),  # end subroutine fun
        ]),
    ])
    m.check(ast)

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

    # A matcher focused on correctly parsing the module definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                M(Use_Stmt),  # use lib, only : fun
                *[M.IGNORE()] * 2,  # implicit none; double precision d(4)
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
                    M(Execution_Part),  # d(2) = 5.5
                    M(End_Subroutine_Stmt),  # end subroutine fun
                ]),
            ]),
            M(End_Module_Stmt),  # end module lib
        ]),
    ])
    m.check(ast)

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

    # A matcher focused on correctly parsing the module definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                M(Use_Stmt),  # use lib_indirect, only: fun_indirect
                *[M.IGNORE()] * 2,  # implicit none; double precision d(4)
            ]),
            M(Execution_Part, [M(Call_Stmt)]),  # call fun_indirect(d)
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M(Module_Stmt, [M.IGNORE(), M.NAMED('lib_indirect')]),
            # module lib_indirect
            M(Specification_Part, [M(Use_Stmt)]),  # use lib
            M(Module_Subprogram_Part, [
                M(Contains_Stmt),  # contains
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

    # A matcher focused on correctly parsing the subroutine definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                *[M.IGNORE()] * 2,  # use lib; implicit none
                M(Interface_Block, [
                    M(Interface_Stmt, [M.NAMED('xi')]),  # interface xi
                    M(Procedure_Stmt, [  # module procedure fun
                        M('Procedure_Name_List', [M.NAMED('fun')]),
                        *[M.IGNORE()] * 2,
                    ]),
                    M.IGNORE(),  # end interface xi
                ]),
                M.IGNORE(),  # double precision d(4)
            ]),
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = fun()
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            *[M.IGNORE()] * 2,  # module lib; implicit none
            M(Module_Subprogram_Part, [
                M.IGNORE(),  # contains
                M(Function_Subprogram),  # real function fun(); implicit none; fun = 5.5; end function fun
            ]),
            M.IGNORE(),  # end module lib
        ]),
    ])
    m.check(ast)

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

    # A matcher focused on correctly parsing the subroutine definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [M.IGNORE()]*3),
            # use lib_indirect, only : fun, fun2; implicit none; double precision d(4)
            M(Execution_Part),  # d(2) = fun()
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M.IGNORE(),  # module lib_indirect
            M(Specification_Part, [
                *[M.IGNORE()] * 2,  # use lib, only: fun; implicit none
                M(Interface_Block, [
                    M(Interface_Stmt, [M.NAMED('xi')]),  # interface xi
                    M(Procedure_Stmt, [  # module procedure fun
                        M('Procedure_Name_List', [M.NAMED('fun')]),
                        *[M.IGNORE()] * 2,
                    ]),
                    M.IGNORE(),  # end interface xi
                ]),
            ]),
            M(Module_Subprogram_Part),  # contains; real function fun2(); implicit none; fun2 = 4.2; end function fun2
            M.IGNORE(),  # end module lib
        ]),
        M(Module, [
            *[M.IGNORE()] * 2,  # module lib; implicit none
            M(Module_Subprogram_Part),  # contains; real function fun(); implicit none; fun = 5.5; end function fun
            M.IGNORE(),  # end module lib
        ]),
    ])
    m.check(ast)

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


def test_uses_module_but_prunes_unused_defs():
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
  subroutine not_fun(d)  ! `main` only uses `fun`, so this should be dropped after simplification
    implicit none
    double precision d(4)
    d(2) = 4.2
  end subroutine not_fun
end module lib
""").add_file("""
program main
  use lib, only: fun
  implicit none
  double precision d(4)
  call fun(d)
end program main
""").check_with_gfortran().get()
    ast, dep_graph, interface_blocks, asts = parse_and_improve(sources)

    # A matcher focused on correctly parsing the module definitions and uses.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part, [
                M(Use_Stmt),  # use lib
                *[M.IGNORE()] * 2,  # implicit none; double precision d(4)
            ]),
            M(Execution_Part, [M(Call_Stmt)]),  # call fun(d)
            M.IGNORE(),  # end program main
        ]),
        M(Module, [
            M(Module_Stmt, [M.IGNORE(), M(Name, has_attr={'string': M(has_value='lib')})]),  # module lib
            M(Module_Subprogram_Part, [
                M(Contains_Stmt),  # contains
                M(Subroutine_Subprogram, [
                    M(Subroutine_Stmt,  # subroutine fun(d)
                      [M.IGNORE(), M(Name, has_attr={'string': M(has_value='fun')}), *[M.IGNORE()] * 2]),
                    M(Specification_Part),  # implicit none; double precision d(4)
                    M(Execution_Part),  # d(2) = 5.5
                    M(End_Subroutine_Stmt),  # end subroutine fun
                ]),
                M(Subroutine_Subprogram, [
                    M(Subroutine_Stmt,  # subroutine not_fun(d)
                      [M.IGNORE(), M(Name, has_attr={'string': M(has_value='not_fun')}), *[M.IGNORE()] * 2]),
                    M(Specification_Part),  # implicit none; double precision d(4)
                    M(Execution_Part),  # d(2) = 4.2
                    M(End_Subroutine_Stmt),  # end subroutine not_fun
                ]),
            ]),
            M(End_Module_Stmt),  # end module lib
        ]),
    ])
    m.check(ast)

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
