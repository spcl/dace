# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""

This module contains the abstract base classes that all intrinsic transformations
inherit from:
- IntrinsicTransformation: Interface for intrinsic replacement strategy
- IntrinsicNodeTransformer: Base class for AST node transformers
"""

from abc import abstractmethod
from typing import Union

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_transforms import (
    NodeTransformer,
    ParentScopeAssigner,
    ScopeVarsDeclarations,
)


class IntrinsicTransformation:
    """
    Abstract base class for all intrinsic transformation strategies.

    Subclasses must implement methods for replacing intrinsic function names
    and determining whether additional AST transformations are needed.
    """

    @staticmethod
    @abstractmethod
    def replaced_name(func_name: str) -> str:
        """Return the replacement name for an intrinsic function."""
        pass

    @staticmethod
    @abstractmethod
    def replace(
        func_name: ast_internal_classes.Name_Node,
        args: ast_internal_classes.Arg_List_Node,
        line,
        symbols: list,
    ) -> ast_internal_classes.FNode:
        """
        Replace an intrinsic function call with its transformed AST representation.
        """
        pass

    @staticmethod
    def has_transformation() -> bool:
        """Return True if this intrinsic requires additional AST transformation passes."""
        return False


class IntrinsicNodeTransformer(NodeTransformer):
    """
    Abstract base class for intrinsic-specific AST transformers.

    Provides scope analysis and variable resolution capabilities for transformations
    that need to inspect or modify variables in the AST.
    """

    def initialize(self, ast):
        """
        Initialize the transformer with AST context.

        Sets up scope analysis and variable processing for the transformation.
        This must be called before the transformation is applied.

        :param ast: The AST to transform
        """
        # We need to rerun the assignment because transformations could have created
        # new AST nodes
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)
        self.ast = ast

        from dace.frontend.fortran.intrinsics.utils import VariableProcessor

        self.var_processor = VariableProcessor(self.scope_vars, self.ast)

    def get_var_declaration(
        self,
        parent: ast_internal_classes.FNode,
        variable: Union[
            ast_internal_classes.Data_Ref_Node,
            ast_internal_classes.Name_Node,
            ast_internal_classes.Array_Subscript_Node,
        ],
    ):
        """
        Get the declaration for a variable reference.

        :param parent: The parent scope node
        :param variable: The variable reference to resolve
        :return: The variable declaration node
        """
        return self.var_processor.get_var_declaration(parent, variable)

    @staticmethod
    @abstractmethod
    def func_name() -> str:
        """Return the name of the function this transformer handles."""
        pass
