# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Utility classes for Fortran intrinsic transformations.
"""

from typing import Union

from dace.frontend.fortran import ast_internal_classes


class VariableProcessor:
    """
    Helper class for variable resolution and declaration lookup.

    Provides methods to find variable declarations across different scopes (local, module-level).
    """

    def __init__(self, scope_vars, ast):
        self.scope_vars = scope_vars
        self.ast = ast

    def get_var(
        self,
        parent: ast_internal_classes.FNode,
        variable: Union[
            ast_internal_classes.Data_Ref_Node,
            ast_internal_classes.Name_Node,
            ast_internal_classes.Array_Subscript_Node,
        ],
    ):
        if isinstance(variable, ast_internal_classes.Data_Ref_Node):

            _, var_decl, cur_val = self.ast.structures.find_definition(
                self.scope_vars, variable
            )
            return var_decl, cur_val

        assert isinstance(
            variable,
            (ast_internal_classes.Name_Node, ast_internal_classes.Array_Subscript_Node),
        )
        if isinstance(variable, ast_internal_classes.Name_Node):
            name = variable.name
        elif isinstance(variable, ast_internal_classes.Array_Subscript_Node):
            name = variable.name.name

        if self.scope_vars.contains_var(parent, name):
            return self.scope_vars.get_var(parent, name), variable
        elif name in self.ast.module_declarations:
            return self.ast.module_declarations[name], variable
        else:
            raise RuntimeError(
                f"Couldn't find the declaration of variable {name} in function {parent.name.name}!"
            )

    def get_var_declaration(
        self,
        parent: ast_internal_classes.FNode,
        variable: Union[
            ast_internal_classes.Data_Ref_Node,
            ast_internal_classes.Name_Node,
            ast_internal_classes.Array_Subscript_Node,
        ],
    ):
        return self.get_var(parent, variable)[0]
