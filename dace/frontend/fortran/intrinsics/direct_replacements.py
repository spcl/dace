# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""

This module contains Fortran intrinsics that are directly replaced with simpler
AST constructs or literal values during parsing. These intrinsics do not require
loop generation, AST transformations or SDFG transformations.

Implemented intrinsics:
- SIZE: Array size computation
- LBOUND/UBOUND: Array bounds
- BIT_SIZE: Type bit width
- PRESENT: Optional argument presence check
- ALLOCATED: Allocation status check
- SELECTED_INT_KIND/SELECTED_REAL_KIND: Kind selectors
- EPSILON: Machine epsilon

"""

import math
import sys
from collections import namedtuple

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_utils import fortrantypes2dacetypes
from dace.frontend.fortran.intrinsics.base import (
    IntrinsicTransformation,
    IntrinsicNodeTransformer,
)


class DirectReplacement(IntrinsicTransformation):
    """
    Direct replacement intrinsics that are replaced with simpler AST nodes.

    This class handles intrinsics that can be replaced at parse time with
    either literal values (Replacement) or transformed AST nodes (Transformation).
    """

    Replacement = namedtuple("Replacement", "function")
    Transformation = namedtuple("Transformation", "function")

    class ASTTransformation(IntrinsicNodeTransformer):
        """AST transformer for direct replacement intrinsics that need type propagation."""

        @staticmethod
        def func_name() -> str:
            return "direct_replacement"

        def visit_BinOp_Node(self, binop_node: ast_internal_classes.BinOp_Node):

            if not isinstance(binop_node.rval, ast_internal_classes.Call_Expr_Node):
                return binop_node

            node = binop_node.rval

            name = node.name.name.split("__dace_")
            if len(name) != 2 or name[1] not in DirectReplacement.FUNCTIONS:
                return binop_node
            func_name = name[1]

            replacement_rule = DirectReplacement.FUNCTIONS[func_name]
            if isinstance(replacement_rule, DirectReplacement.Transformation):
                # FIXME: we do not have line number in binop?
                binop_node.rval, input_type = replacement_rule.function(
                    self, node, 0
                )  # binop_node.line)

                var = binop_node.lval

                # replace types of return variable - LHS of the binary operator
                # we only propagate that for the assignment
                # we handle extracted call variables this way
                # but we can also have different shapes, e.g., `maxval(something) > something_else`
                # hence the check
                if isinstance(
                    var,
                    (
                        ast_internal_classes.Name_Node,
                        ast_internal_classes.Array_Subscript_Node,
                        ast_internal_classes.Data_Ref_Node,
                    ),
                ):

                    var_decl = self.get_var_declaration(var.parent, var)
                    var_decl.type = input_type

                var.type = input_type

            return binop_node

    def replace_size(
        transformer: IntrinsicNodeTransformer,
        var: ast_internal_classes.Call_Expr_Node,
        line,
    ):

        if len(var.args) not in (1, 2):
            raise ValueError("Incorrect number of arguments to SIZE intrinsic")

        # get variable declaration for the first argument
        var_decl = transformer.get_var_declaration(var.parent, var.args[0])

        # one arg to SIZE? compute the total number of elements
        if len(var.args) == 1:

            if len(var_decl.sizes) == 1:
                return (var_decl.sizes[0], "INTEGER")

            ret = ast_internal_classes.BinOp_Node(
                lval=var_decl.sizes[0],
                rval=ast_internal_classes.Name_Node(name="INTRINSIC_TEMPORARY"),
                op="*",
            )
            cur_node = ret
            for i in range(1, len(var_decl.sizes) - 1):
                cur_node.rval = ast_internal_classes.BinOp_Node(
                    lval=var_decl.sizes[i],
                    rval=ast_internal_classes.Name_Node(name="INTRINSIC_TEMPORARY"),
                    op="*",
                )
                cur_node = cur_node.rval

            cur_node.rval = var_decl.sizes[-1]
            return (ret, "INTEGER")

        # two arguments? We return number of elements in a given rank
        rank = var.args[1]
        # we do not support symbolic argument to DIM - it must be a literal
        if not isinstance(rank, ast_internal_classes.Int_Literal_Node):
            raise NotImplementedError()
        value = int(rank.value)
        return (var_decl.sizes[value - 1], "INTEGER")

    def _replace_lbound_ubound(
        func: str,
        transformer: IntrinsicNodeTransformer,
        var: ast_internal_classes.Call_Expr_Node,
        line,
    ):

        if len(var.args) not in (1, 2):
            raise ValueError(
                f"Incorrect number of arguments to {func.upper()} intrinsic"
            )

        # get variable declaration for the first argument
        var_decl = transformer.get_var_declaration(var.parent, var.args[0])

        # one arg to LBOUND/UBOUND? not needed currently
        if len(var.args) == 1:
            raise NotImplementedError()

        # two arguments? We return number of elements in a given rank
        rank = var.args[1]
        # we do not support symbolic argument to DIM - it must be a literal
        if not isinstance(rank, ast_internal_classes.Int_Literal_Node):
            raise NotImplementedError()

        rank_value = int(rank.value)

        is_assumed = isinstance(
            var_decl.offsets[rank_value - 1], ast_internal_classes.Name_Node
        ) and var_decl.offsets[rank_value - 1].name.startswith("__f2dace_")

        if func == "lbound":

            if is_assumed and not var_decl.alloc:
                value = ast_internal_classes.Int_Literal_Node(value="1")
            elif isinstance(var_decl.offsets[rank_value - 1], int):
                value = ast_internal_classes.Int_Literal_Node(
                    value=str(var_decl.offsets[rank_value - 1])
                )
            else:
                value = var_decl.offsets[rank_value - 1]

        else:
            if isinstance(var_decl.sizes[rank_value - 1], ast_internal_classes.FNode):
                size = var_decl.sizes[rank_value - 1]
            else:
                size = ast_internal_classes.Int_Literal_Node(
                    value=var_decl.sizes[rank_value - 1]
                )

            if is_assumed and not var_decl.alloc:
                value = size
            else:
                if isinstance(
                    var_decl.offsets[rank_value - 1], ast_internal_classes.FNode
                ):
                    offset = var_decl.offsets[rank_value - 1]
                elif isinstance(var_decl.offsets[rank_value - 1], int):
                    offset = ast_internal_classes.Int_Literal_Node(
                        value=str(var_decl.offsets[rank_value - 1])
                    )
                else:
                    offset = ast_internal_classes.Int_Literal_Node(
                        value=var_decl.offsets[rank_value - 1]
                    )

                value = ast_internal_classes.BinOp_Node(
                    op="+",
                    lval=size,
                    rval=ast_internal_classes.BinOp_Node(
                        op="-",
                        lval=offset,
                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                        line_number=line,
                    ),
                    line_number=line,
                )

        return (value, "INTEGER")

    def replace_lbound(
        transformer: IntrinsicNodeTransformer,
        var: ast_internal_classes.Call_Expr_Node,
        line,
    ):
        return DirectReplacement._replace_lbound_ubound(
            "lbound", transformer, var, line
        )

    def replace_ubound(
        transformer: IntrinsicNodeTransformer,
        var: ast_internal_classes.Call_Expr_Node,
        line,
    ):
        return DirectReplacement._replace_lbound_ubound(
            "ubound", transformer, var, line
        )

    def replace_bit_size(
        transformer: IntrinsicNodeTransformer,
        var: ast_internal_classes.Call_Expr_Node,
        line,
    ):

        if len(var.args) != 1:
            raise ValueError("Incorrect number of arguments to BIT_SIZE intrinsic")

        # get variable declaration for the first argument
        var_decl = transformer.get_var_declaration(var.parent, var.args[0])

        dace_type = fortrantypes2dacetypes[var_decl.type]
        type_size = dace_type().itemsize * 8

        return (ast_internal_classes.Int_Literal_Node(value=str(type_size)), "INTEGER")

    def replace_int_kind(args: ast_internal_classes.Arg_List_Node, line, symbols: list):
        if isinstance(args.args[0], ast_internal_classes.Int_Literal_Node):
            arg0 = args.args[0].value
        elif isinstance(args.args[0], ast_internal_classes.Name_Node):
            if args.args[0].name in symbols:
                arg0 = symbols[args.args[0].name].value
            else:
                raise ValueError("Only symbols can be names in selector")
        else:
            raise ValueError("Only literals or symbols can be arguments in selector")
        return ast_internal_classes.Int_Literal_Node(
            value=str(math.ceil((math.log2(math.pow(10, int(arg0))) + 1) / 8)),
            line_number=line,
        )

    def replace_real_kind(
        args: ast_internal_classes.Arg_List_Node, line, symbols: list
    ):
        if isinstance(args.args[0], ast_internal_classes.Int_Literal_Node):
            arg0 = args.args[0].value
        elif isinstance(args.args[0], ast_internal_classes.Name_Node):
            if args.args[0].name in symbols:
                arg0 = symbols[args.args[0].name].value
            else:
                raise ValueError("Only symbols can be names in selector")
        else:
            raise ValueError("Only literals or symbols can be arguments in selector")
        if len(args.args) == 2:
            if isinstance(args.args[1], ast_internal_classes.Int_Literal_Node):
                arg1 = args.args[1].value
            elif isinstance(args.args[1], ast_internal_classes.Name_Node):
                if args.args[1].name in symbols:
                    arg1 = symbols[args.args[1].name].value
                else:
                    raise ValueError("Only symbols can be names in selector")
            else:
                raise ValueError(
                    "Only literals or symbols can be arguments in selector"
                )
        else:
            arg1 = 0
        if int(arg0) >= 9 or int(arg1) > 126:
            return ast_internal_classes.Int_Literal_Node(value="8", line_number=line)
        elif int(arg0) >= 3 or int(arg1) > 14:
            return ast_internal_classes.Int_Literal_Node(value="4", line_number=line)
        else:
            return ast_internal_classes.Int_Literal_Node(value="2", line_number=line)

    def replace_present(
        transformer: IntrinsicNodeTransformer,
        call: ast_internal_classes.Call_Expr_Node,
        line,
    ):

        if len(call.args) != 1:
            raise ValueError("PRESENT intrinsic expects exactly one argument")
        if not isinstance(call.args[0], ast_internal_classes.Name_Node):
            raise TypeError("Argument to PRESENT must be a variable name")

        var_name = call.args[0].name
        test_var_name = f"__f2dace_OPTIONAL_{var_name}"

        return (ast_internal_classes.Name_Node(name=test_var_name), "LOGICAL")

    def replace_allocated(
        transformer: IntrinsicNodeTransformer,
        call: ast_internal_classes.Call_Expr_Node,
        line,
    ):

        if len(call.args) != 1:
            raise ValueError("ALLOCATED intrinsic expects exactly one argument")
        if not isinstance(call.args[0], ast_internal_classes.Name_Node):
            raise TypeError("Argument to ALLOCATED must be a variable name")

        var_name = call.args[0].name
        test_var_name = f"__f2dace_ALLOCATED_{var_name}"

        return (ast_internal_classes.Name_Node(name=test_var_name), "LOGICAL")

    def replacement_epsilon(
        args: ast_internal_classes.Arg_List_Node, line, symbols: list
    ):

        ret_val = sys.float_info.epsilon
        return ast_internal_classes.Real_Literal_Node(value=str(ret_val))

    FUNCTIONS = {
        "SELECTED_INT_KIND": Replacement(replace_int_kind),
        "SELECTED_REAL_KIND": Replacement(replace_real_kind),
        "EPSILON": Replacement(replacement_epsilon),
        "BIT_SIZE": Transformation(replace_bit_size),
        "SIZE": Transformation(replace_size),
        "LBOUND": Transformation(replace_lbound),
        "UBOUND": Transformation(replace_ubound),
        "PRESENT": Transformation(replace_present),
        "ALLOCATED": Transformation(replace_allocated),
    }

    @staticmethod
    def temporary_functions():

        # temporary functions created by us -> f becomes __dace_f
        # We provide this to tell Fortran parser that these are function calls,
        # not array accesses
        funcs = list(DirectReplacement.FUNCTIONS.keys())
        return [f"__dace_{f}" for f in funcs]

    @staticmethod
    def replacable_name(func_name: str) -> bool:
        return func_name in DirectReplacement.FUNCTIONS

    @staticmethod
    def replace_name(func_name: str) -> str:
        return ast_internal_classes.Name_Node(name=f"__dace_{func_name}")

    @staticmethod
    def replacable(func_name: str) -> bool:
        orig_name = func_name.split("__dace_")
        if len(orig_name) > 1 and orig_name[1] in DirectReplacement.FUNCTIONS:
            return isinstance(
                DirectReplacement.FUNCTIONS[orig_name[1]], DirectReplacement.Replacement
            )
        return False

    @staticmethod
    def replace(
        func_name: str, args: ast_internal_classes.Arg_List_Node, line, symbols: list
    ) -> ast_internal_classes.FNode:
        # Here we already have __dace_func
        fname = func_name.split("__dace_")[1]
        return DirectReplacement.FUNCTIONS[fname].function(args, line, symbols)

    def has_transformation(fname: str) -> bool:
        return isinstance(
            DirectReplacement.FUNCTIONS[fname], DirectReplacement.Transformation
        )

    @staticmethod
    def get_transformation() -> IntrinsicNodeTransformer:
        return DirectReplacement.ASTTransformation()
