# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
import dace.frontend.fortran.ast_transforms as ast_transforms
from dace.frontend.fortran.fortran_parser import create_internal_ast, ParseConfig
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_parent():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  double precision, dimension(5) :: arr  ! This will be pruned by preprocessor.
  double precision, dimension(50:54) :: arr3
  call fun(d, arr3)
end program main

! Intentionally gave unique names to variables, to not have them renamed automatically.
subroutine fun(d1, arr4)
  implicit none
  double precision d1(4)
  double precision, dimension(50:54) :: arr4
  d1(2) = 5.5
end subroutine fun
""").check_with_gfortran().get()
    cfg = ParseConfig(sources=sources, entry_points=[('main', )])
    _, program = create_internal_ast(cfg)
    ast_transforms.ParentScopeAssigner().visit(program)
    visitor = ast_transforms.ScopeVarsDeclarations(program)
    visitor.visit(program)

    for var in ['d', 'arr3']:
        assert ('main', var) in visitor.scope_vars
        decl = visitor.scope_vars[('main', var)]
        assert isinstance(decl, ast_internal_classes.Var_Decl_Node)
        assert decl.name == var

    for var in ['d1', 'arr4']:
        assert ('fun', var) in visitor.scope_vars
        decl = visitor.scope_vars[('fun', var)]
        assert isinstance(decl, ast_internal_classes.Var_Decl_Node)
        assert decl.name == var


if __name__ == "__main__":
    test_fortran_frontend_parent()
