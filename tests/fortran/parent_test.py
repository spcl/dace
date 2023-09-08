# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from dace.frontend.fortran import fortran_parser

import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes


def test_fortran_frontend_parent():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM access_test
                    implicit none
                    double precision d(4)
                    d(1)=0
                    CALL array_access_test_function(d)
                    end

                    SUBROUTINE array_access_test_function(d)
                    double precision d(4)

                    d(2)=5.5

                    END SUBROUTINE array_access_test_function
                    """
    ast, functions = fortran_parser.create_ast_from_string(test_string, "array_access_test")
    ast_transforms.ParentScopeAssigner().visit(ast)

    assert ast.parent is None
    assert ast.main_program.parent == ast

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

        assert subroutine.parent == ast
        assert subroutine.execution_part.parent == subroutine
        for execution in subroutine.execution_part.execution:
            assert execution.parent == subroutine


if __name__ == "__main__":

    test_fortran_frontend_parent()
