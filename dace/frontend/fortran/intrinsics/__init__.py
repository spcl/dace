# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module provides the main orchestrator for Fortran intrinsic functions and
their transformations. It coordinates between different intrinsic categories:
- Direct replacements (SIZE, LBOUND, UBOUND, etc.)
- Loop-based transformations (SUM, PRODUCT, ANY, ALL, COUNT, etc.)
- Math functions (trigonometric, type conversion, BLAS, bitwise)
- SDFG-level optimizations (library node replacements)

The FortranIntrinsics class is the main entry point used by the Fortran parser
to handle intrinsic function replacements and transformations.
"""

from typing import Any, List, Union

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_transforms import (
    NodeTransformer,
)
from dace.frontend.fortran.intrinsics.direct_replacements import DirectReplacement
from dace.frontend.fortran.intrinsics.loop_based import (
    LoopBasedReplacement,
    Sum,
    Product,
    Any,
    All,
    Count,
    MinVal,
    MaxVal,
    Merge,
)
from dace.frontend.fortran.intrinsics.math_functions import MathFunctions

FASTNode = Any


class FortranIntrinsics:
    IMPLEMENTATIONS_AST = {
        "SUM": Sum,
        "PRODUCT": Product,
        "ANY": Any,
        "COUNT": Count,
        "ALL": All,
        "MINVAL": MinVal,
        "MAXVAL": MaxVal,
        "MERGE": Merge,
    }

    def __init__(self):
        self._transformations_to_run = {}

    def transformations(self) -> List[NodeTransformer]:
        return list(self._transformations_to_run.values())

    @staticmethod
    def function_names() -> List[str]:
        # list of all functions that are created by initial transformation, before doing full replacement
        # this prevents other parser components from replacing our function calls with array subscription nodes
        return [
            *list(LoopBasedReplacement.INTRINSIC_TO_DACE.values()),
            *MathFunctions.temporary_functions(),
            *DirectReplacement.temporary_functions(),
        ]

    @staticmethod
    def retained_function_names() -> List[str]:
        # list of all DaCe functions that we use after full parsing
        return MathFunctions.dace_functions()

    def replace_function_name(
        self, node: Union[FASTNode, ast_internal_classes.Name_Node]
    ) -> ast_internal_classes.Name_Node:

        if isinstance(node, ast_internal_classes.Name_Node):
            func_name = node.name
        else:
            func_name = node.string

        # TODO: implement and categorize the intrinsic functions below - if necessary
        # These functions are currently not implemented
        #
        # SIGN is handled by an explicit AST transformation outside of intrinsics (legacy solution)
        replacements = {
            "SIGN": "__dace_sign",
            "SPREAD": "__dace_spread",
            "TRIM": "__dace_trim",
            "LEN_TRIM": "__dace_len_trim",
            "ASSOCIATED": "__dace_associated",
            "MAXLOC": "__dace_maxloc",
            "FRACTION": "__dace_fraction",
            "NEW_LINE": "__dace_new_line",
            "PRECISION": "__dace_precision",
            "MINLOC": "__dace_minloc",
            "LEN": "__dace_len",
            "SCAN": "__dace_scan",
            "RANDOM_SEED": "__dace_random_seed",
            "RANDOM_NUMBER": "__dace_random_number",
            "DATE_AND_TIME": "__dace_date_and_time",
            "RESHAPE": "__dace_reshape",
        }

        if func_name in replacements:
            return ast_internal_classes.Name_Node(name=replacements[func_name])
        elif DirectReplacement.replacable_name(func_name):

            if DirectReplacement.has_transformation(func_name):
                # self._transformations_to_run.add(DirectReplacement.get_transformation())
                transformation = DirectReplacement.get_transformation()
                if transformation.func_name() not in self._transformations_to_run:
                    self._transformations_to_run[transformation.func_name()] = (
                        transformation
                    )

            return DirectReplacement.replace_name(func_name)
        elif MathFunctions.replacable(func_name):

            transformation = MathFunctions.get_transformation()
            if transformation.func_name() not in self._transformations_to_run:
                self._transformations_to_run[transformation.func_name()] = (
                    transformation
                )

            return MathFunctions.replace(func_name)

        if self.IMPLEMENTATIONS_AST[func_name].has_transformation():

            if hasattr(self.IMPLEMENTATIONS_AST[func_name], "Transformation"):
                transformation = self.IMPLEMENTATIONS_AST[func_name].Transformation()
            else:
                transformation = self.IMPLEMENTATIONS_AST[func_name].get_transformation(
                    func_name
                )

            if transformation.func_name() not in self._transformations_to_run:
                self._transformations_to_run[transformation.func_name()] = (
                    transformation
                )

        return ast_internal_classes.Name_Node(
            name=self.IMPLEMENTATIONS_AST[func_name].replaced_name(func_name)
        )

    def replace_function_reference(
        self,
        name: ast_internal_classes.Name_Node,
        args: ast_internal_classes.Arg_List_Node,
        line,
        symbols: dict,
    ):

        func_types = {
            "__dace_sign": "DOUBLE",
        }
        if name.name in func_types:
            # FIXME: this will be progressively removed
            call_type = func_types[name.name]
            return ast_internal_classes.Call_Expr_Node(
                name=name,
                type=call_type,
                args=args.args,
                line_number=line,
                subroutine=False,
            )
        elif DirectReplacement.replacable(name.name):
            return DirectReplacement.replace(name.name, args, line, symbols)
        else:
            # We will do the actual type replacement later
            # To that end, we need to know the input types - but these we do not know at the moment.
            return ast_internal_classes.Call_Expr_Node(
                name=name,
                type="VOID",
                subroutine=False,
                args=args.args,
                line_number=line,
            )

    @staticmethod
    def output_size(node: ast_internal_classes.Call_Expr_Node):

        name = node.name.name.split("__dace_")
        if len(name) != 2:
            return None, None, "VOID"

        sizes = []
        for arg in node.args:

            if isinstance(
                arg,
                (
                    ast_internal_classes.Int_Literal_Node,
                    ast_internal_classes.Real_Literal_Node,
                ),
            ):
                sizes.append(1)
            else:
                sizes.append(arg.sizes)

        input_type = node.args[0].type
        return_type = "VOID"

        func_name = name[1].upper()

        if func_name in FortranIntrinsics.IMPLEMENTATIONS_AST:

            replacement_rule = FortranIntrinsics.IMPLEMENTATIONS_AST[func_name]
            res = replacement_rule.output_size(node.args)
            if res is None:
                return None, None, "VOID"
            else:
                sizes = res[0]
                return_type = res[1]

        elif func_name in MathFunctions.INTRINSIC_TO_DACE:

            replacement_rule = MathFunctions.INTRINSIC_TO_DACE[func_name]
            if isinstance(replacement_rule, dict):
                replacement_rule = replacement_rule[input_type]

            if (
                isinstance(replacement_rule, MathFunctions.MathTransformation)
                and replacement_rule.size_function is not None
            ):

                sizes = replacement_rule.size_function(node, sizes)
            else:

                if input_type != "VOID":

                    if replacement_rule.return_type == "FIRST_ARG":
                        return_type = input_type
                    elif replacement_rule.return_type == "CALL_EXPR":
                        return_type = node.type
                    else:
                        return_type = replacement_rule.return_type

                sizes = sizes[0]

        else:
            return None, None, "VOID"

        if isinstance(sizes, ast_internal_classes.Int_Literal_Node):
            return sizes, [1], return_type
        elif isinstance(sizes, list):
            return sizes, [1] * len(sizes), return_type
        else:
            return [], [1], return_type
