# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
import dace.frontend.fortran.ast_transforms as ast_transforms
from dace.frontend.fortran.ast_internal_classes import Program_Node
from dace.frontend.fortran.fortran_parser import ParseConfig, create_internal_ast
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_parent():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  d(1) = 0
  call fun(d)
end program main

subroutine fun(d)
  double precision d(4)
  d(2) = 5.5
end subroutine fun
""", 'main').check_with_gfortran().get()
    cfg = ParseConfig(main=sources['main.f90'], sources=sources)
    _, ast = create_internal_ast(cfg)
    ast_transforms.ParentScopeAssigner().visit(ast)

    assert not ast.parent
    assert isinstance(ast, Program_Node)
    assert ast.main_program is not None

    main_program = ast.main_program
    # Both executed lines
    for execution in main_program.execution_part.execution:
        assert execution.parent == main_program
    # call to the function
    call_node = main_program.execution_part.execution[1]
    assert isinstance(call_node, ast_internal_classes.Call_Expr_Node)
    for arg in call_node.args:
        assert arg.parent == main_program

    for subroutine in ast.subroutine_definitions:
        assert not subroutine.parent
        assert subroutine.execution_part.parent == subroutine
        for execution in subroutine.execution_part.execution:
            assert execution.parent == subroutine


def test_fortran_frontend_module():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  ! good enough approximation
  integer, parameter :: pi = 4
end module lib
""").add_file("""
program main
  implicit none
  double precision d(4)
  d(1) = 0
  call fun(d)
end program main

subroutine fun(d)
  use lib, only: pi
  implicit none
  double precision d(4)
  d(2) = pi
end subroutine fun
""", 'main').check_with_gfortran().get()
    cfg = ParseConfig(main=sources['main.f90'], sources=sources)
    _, ast = create_internal_ast(cfg)
    ast_transforms.ParentScopeAssigner().visit(ast)

    assert not ast.parent
    assert isinstance(ast, Program_Node)
    assert not ast.main_program.parent

    assert len(ast.modules) == 1
    module = ast.modules[0]
    assert not module.parent

    assert module.specification_part is not None
    assert len(module.specification_part.symbols) == 1
    specification = module.specification_part.symbols[0]
    assert specification.parent == module


if __name__ == "__main__":
    test_fortran_frontend_parent()
    test_fortran_frontend_module()
