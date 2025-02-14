# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from dace.frontend.fortran import fortran_parser

import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes


def test_fortran_frontend_parent():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM scope_test
                    implicit none
                    double precision d(4)
                    double precision, dimension(5) :: arr
                    double precision, dimension(50:54) :: arr3
                    CALL scope_test_function(d)
                    end

                    SUBROUTINE scope_test_function(d)
                    double precision d(4)
                    double precision, dimension(50:54) :: arr4

                    d(2)=5.5

                    END SUBROUTINE scope_test_function
                    """

    ast, functions = fortran_parser.create_ast_from_string(test_string, "array_access_test")
    ast_transforms.ParentScopeAssigner().visit(ast)
    visitor = ast_transforms.ScopeVarsDeclarations()
    visitor.visit(ast)

    for var in ['d', 'arr', 'arr3']:
        assert ('scope_test', var) in visitor.scope_vars
        assert isinstance(visitor.scope_vars[('scope_test', var)], ast_internal_classes.Var_Decl_Node)
        assert visitor.scope_vars[('scope_test', var)].name == var

    for var in ['d', 'arr4']:
        assert ('scope_test_function', var) in visitor.scope_vars
        assert visitor.scope_vars[('scope_test_function', var)].name == var

if __name__ == "__main__":

    test_fortran_frontend_parent()
