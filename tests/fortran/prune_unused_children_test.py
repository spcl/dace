# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, List

from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from dace.frontend.fortran.ast_desugaring.analysis import ident_spec
from dace.frontend.fortran.ast_desugaring.pruning import prune_unused_objects
from dace.frontend.fortran.ast_desugaring.types import SPEC
from dace.frontend.fortran.ast_desugaring.utils import ENTRY_POINT_OBJECT_CLASSES, NAMED_STMTS_OF_INTEREST_CLASSES, \
    find_name_of_node
from dace.frontend.fortran.ast_utils import children_of_type, singular
from dace.frontend.fortran.fortran_parser import construct_full_ast
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(sources, parser)
    assert isinstance(ast, Program)
    return ast


def find_entrypoint_objects_named(ast: Program, name: str) -> List[SPEC]:
    objs: List[SPEC] = []
    for n in walk(ast, ENTRY_POINT_OBJECT_CLASSES):
        assert isinstance(n, ENTRY_POINT_OBJECT_CLASSES)
        if not isinstance(n.parent, Program):
            continue
        if find_name_of_node(n) == name:
            stmt = singular(children_of_type(n, NAMED_STMTS_OF_INTEREST_CLASSES))
            objs.append(ident_spec(stmt))
    return objs


def prune_from_main(ast: Program) -> Program:
    return prune_unused_objects(ast, find_entrypoint_objects_named(ast, 'main'))


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
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

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


def test_toplevel_subroutine_no_pruning():
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
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

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


def test_program_contains_subroutine_no_pruning():
    """
    A simple program with not much to "recursively improve", but this time the subroutine is defined outside and called
    from the main program.
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
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

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
    d(2) = 5.5
  END SUBROUTINE fun
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_standalone_subroutine_no_pruning():
    """
    A standalone subroutine, with no program or module in sight.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  implicit none
  double precision d(4)
  d(2) = 5.5
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main(d)
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = 5.5
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_toplevel_subroutine_uses_another_module_no_pruning():
    """
    A simple program with not much to "recursively improve", but this time the subroutine is defined outside and called
    from the main program.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  double precision :: val = 5.5
end module lib

program main
  implicit none
  double precision d(4)
  call fun(d)
end program main

subroutine fun(d)
  use lib
  implicit none
  double precision d(4)
  d(2) = val
end subroutine fun
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  DOUBLE PRECISION :: val = 5.5
END MODULE lib
PROGRAM main
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  CALL fun(d)
END PROGRAM main
SUBROUTINE fun(d)
  USE lib, ONLY: val
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = val
END SUBROUTINE fun
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


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

module lib_indirect
  use lib
contains
  subroutine fun_indirect(d)
    implicit none
    double precision d(4)
    call fun(d)
  end subroutine fun_indirect
end module lib_indirect

program main
  use lib_indirect, only: fun_indirect
  implicit none
  double precision d(4)
  call fun_indirect(d)
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

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
  CONTAINS
  SUBROUTINE fun_indirect(d)
    USE lib, ONLY: fun
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

program main
  use lib_indirect, only : fun, fun2
  implicit none

  double precision d(4)
  d(2) = fun()
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

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
  USE lib, ONLY: fun
  IMPLICIT NONE
  DOUBLE PRECISION :: d(4)
  d(2) = fun()
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


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

program main
  use lib, only: fun
  implicit none
  double precision d(4)
  call fun(d)
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

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


def test_module_contains_used_and_unused_types_prunes_unused_defs():
    """
    Module has type definition that the program does not use, so it gets pruned.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none

  type used_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type used_type

  type dead_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type dead_type
end module lib

program main
  use lib, only : used_type
  implicit none

  real :: d(5, 5)
  call type_test_function(d)

contains

  subroutine type_test_function(d)
    real d(5, 5)
    type(used_type) :: s
    s%w(1, 1, 1) = 5.5
    d(2, 1) = 5.5 + s%w(1, 1, 1)
  end subroutine type_test_function
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    # `not_fun` and `real_fun` should be gone!
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: used_type
    REAL :: w(5, 5, 5)
  END TYPE used_type
END MODULE lib
PROGRAM main
  IMPLICIT NONE
  REAL :: d(5, 5)
  CALL type_test_function(d)
  CONTAINS
  SUBROUTINE type_test_function(d)
    USE lib, ONLY: used_type
    REAL :: d(5, 5)
    TYPE(used_type) :: s
    s % w(1, 1, 1) = 5.5
    d(2, 1) = 5.5 + s % w(1, 1, 1)
  END SUBROUTINE type_test_function
END PROGRAM main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_module_contains_used_and_unused_variables():
    """
    Module has unused variables. But we don't prune variables.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: used = 1
  real, parameter :: unused = 4.2
end module lib

program main
  use lib, only: used
  implicit none

  real :: d(5, 5)
  call type_test_function(d)

contains

  subroutine type_test_function(d)
    real d(5, 5)
    d(2, 1) = used
  end subroutine type_test_function
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    # `not_fun` and `real_fun` should be gone!
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: used = 1
END MODULE lib
PROGRAM main
  IMPLICIT NONE
  REAL :: d(5, 5)
  CALL type_test_function(d)
  CONTAINS
  SUBROUTINE type_test_function(d)
    USE lib, ONLY: used
    REAL :: d(5, 5)
    d(2, 1) = used
  END SUBROUTINE type_test_function
END PROGRAM main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_module_contains_used_and_unused_variables_with_use_all_prunes_unused():
    """
    Module has unused variables that are pulled in with "use-all".
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: used = 1
  real, parameter :: unused = 4.2
end module lib

program main
  use lib
  implicit none

  real :: d(5, 5)
  call type_test_function(d)

contains

  subroutine type_test_function(d)
    real d(5, 5)
    d(2, 1) = used
  end subroutine type_test_function
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    # `not_fun` and `real_fun` should be gone!
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: used = 1
END MODULE lib
PROGRAM main
  IMPLICIT NONE
  REAL :: d(5, 5)
  CALL type_test_function(d)
  CONTAINS
  SUBROUTINE type_test_function(d)
    USE lib, ONLY: used
    REAL :: d(5, 5)
    d(2, 1) = used
  END SUBROUTINE type_test_function
END PROGRAM main
    """.strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_use_statement_multiple():
    """
    We have multiple uses of the same module.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: a = 1
  real, parameter :: b = 4.2
  real, parameter :: c = -7.1
end module lib

program main
  use lib, only: a
  use lib, only: b
  implicit none

  real :: d(5, 5)
  call type_test_function(d)

contains

  subroutine type_test_function(d)
    real d(5, 5)
    d(1, 1) = a
    d(1, 1) = b
  end subroutine type_test_function
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    # `not_fun` and `real_fun` should be gone!
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: a = 1
  REAL, PARAMETER :: b = 4.2
END MODULE lib
PROGRAM main
  IMPLICIT NONE
  REAL :: d(5, 5)
  CALL type_test_function(d)
  CONTAINS
  SUBROUTINE type_test_function(d)
    USE lib, ONLY: a, b
    REAL :: d(5, 5)
    d(1, 1) = a
    d(1, 1) = b
  END SUBROUTINE type_test_function
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_use_statement_multiple_with_useall_prunes_unused():
    """
    We have multiple uses of the same module. One of them is a "use-all".
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: a = 1
  real, parameter :: b = 4.2
  real, parameter :: c = -7.1
end module lib

program main
  use lib
  use lib, only: a
  implicit none

  real :: d(5, 5)
  call type_test_function(d)

contains

  subroutine type_test_function(d)
    real d(5, 5)
    d(1, 1) = a
    d(1, 1) = b
  end subroutine type_test_function
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    # `not_fun` and `real_fun` should be gone!
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: a = 1
  REAL, PARAMETER :: b = 4.2
END MODULE lib
PROGRAM main
  IMPLICIT NONE
  REAL :: d(5, 5)
  CALL type_test_function(d)
  CONTAINS
  SUBROUTINE type_test_function(d)
    USE lib, ONLY: a, b
    REAL :: d(5, 5)
    d(1, 1) = a
    d(1, 1) = b
  END SUBROUTINE type_test_function
END PROGRAM main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_subroutine_contains_function_no_pruning():
    """
    A function is defined inside a subroutine that calls it. A main program uses the top-level subroutine.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
contains
  subroutine fun(d)
    implicit none
    double precision d(4)
    d(2) = fun2()

  contains
    real function fun2()
      implicit none
      fun2 = 5.5
    end function fun2
  end subroutine fun
end module lib

program main
  use lib, only: fun
  implicit none

  double precision d(4)
  call fun(d)
end program main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = prune_from_main(ast)

    # `not_fun` and `real_fun` should be gone!
    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE fun(d)
    IMPLICIT NONE
    DOUBLE PRECISION :: d(4)
    d(2) = fun2()
    CONTAINS
    REAL FUNCTION fun2()
      IMPLICIT NONE
      fun2 = 5.5
    END FUNCTION fun2
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


if __name__ == "__main__":
    test_minimal_no_pruning()
    test_toplevel_subroutine_no_pruning()
    test_program_contains_subroutine_no_pruning()
    test_standalone_subroutine_no_pruning()
    test_toplevel_subroutine_uses_another_module_no_pruning()
    test_uses_module_which_uses_module_no_pruning()
    test_module_contains_interface_block_no_pruning()
    test_uses_module_but_prunes_unused_defs()
    test_module_contains_used_and_unused_types_prunes_unused_defs()
    test_module_contains_used_and_unused_variables()
    test_use_statement_multiple()
    test_use_statement_multiple_with_useall_prunes_unused()
    test_subroutine_contains_function_no_pruning()
