# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module contains mathematical intrinsic functions that are mapped to DaCe
math operations or library calls.
"""

import numpy as np
import warnings
from collections import namedtuple

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_transforms import NeedsTypeInferenceException
from dace.frontend.fortran.intrinsics.base import (
    IntrinsicTransformation,
    IntrinsicNodeTransformer,
)
from dace.frontend.fortran.intrinsics.sdfg_transformations import (
    IntrinsicSDFGTransformation,
)


class MathFunctions(IntrinsicTransformation):
    """
    Implemented intrinsics:
    - Trigonometric: SIN, COS, SINH, COSH, TANH, ASIN, ACOS, ATAN, ATAN2
    - Math: MIN, MAX, SQRT, ABS, POW, EXP, LOG, MOD, MODULO, FLOOR, SCALE, EXPONENT, EPSILON
    - Type conversion: INT, AINT, NINT, ANINT, REAL, DBLE
    - BLAS: DOT_PRODUCT, MATMUL, TRANSPOSE
    - Bitwise: IBSET, IBCLR, IAND, IEOR, ISHFT, BTEST, IBITS
    """

    MathTransformation = namedtuple(
        "MathTransformation",
        "function return_type size_function",
        defaults=[None, None, None],
    )
    MathReplacement = namedtuple(
        "MathReplacement", "function replacement_function return_type"
    )

    def generate_scale(arg: ast_internal_classes.Call_Expr_Node):

        # SCALE(X, I) becomes: X * pow(RADIX(X), I)
        # In our case, RADIX(X) is always 2
        line = arg.line_number
        x = arg.args[0]
        i = arg.args[1]
        const_two = ast_internal_classes.Int_Literal_Node(value="2")

        # I and RADIX(X) are both integers
        rval = ast_internal_classes.Call_Expr_Node(
            name=ast_internal_classes.Name_Node(name="pow"),
            type="INTEGER",
            args=[const_two, i],
            line_number=line,
            subroutine=False,
        )

        mult = ast_internal_classes.BinOp_Node(
            op="*", lval=x, rval=rval, line_number=line
        )

        # pack it into parentheses, just to be sure
        return ast_internal_classes.Parenthesis_Expr_Node(expr=mult)

    def generate_epsilon(args: ast_internal_classes.Call_Expr_Node):
        if len(args.args) != 1:
            raise ValueError("EPSILON intrinsic expects exactly one argument")

        arg_type = args.args[0].type
        if arg_type == "VOID":
            raise NeedsTypeInferenceException("epsilon", args.line_number)

        """
        Determine the appropriate epsilon value based on type
        REAL (32-bit) -> numpy.float32 epsilon
        DOUBLE (64-bit) -> numpy.float64 epsilon
        """
        if arg_type == "REAL":
            return ast_internal_classes.Real_Literal_Node(
                value=str(np.finfo(np.float32).eps)
            )
        elif arg_type == "DOUBLE":
            return ast_internal_classes.Double_Literal_Node(
                value=str(np.finfo(np.float64).eps)
            )
        else:
            raise NotImplementedError()

    def generate_aint(arg: ast_internal_classes.Call_Expr_Node):

        # The call to AINT can contain a second KIND parameter
        # We ignore it a the moment.
        # However, to map into C's trunc, we need to drop it.
        if len(arg.args) > 1:
            warnings.warn(
                "AINT with KIND parameter is not supported! Ignoring that parameter."
            )
            del arg.args[1]

        fname = arg.name.name.split("__dace_")[1]
        if fname in "AINT":
            arg.name = ast_internal_classes.Name_Node(name="trunc")
        elif fname == "NINT":
            arg.name = ast_internal_classes.Name_Node(name="iround")
        elif fname == "ANINT":
            arg.name = ast_internal_classes.Name_Node(name="round")
        else:
            raise NotImplementedError()

        return arg

    def generate_real(arg: ast_internal_classes.Call_Expr_Node):

        # The call to REAL can contain a second KIND parameter.
        # If it is 8, we need to return a double.
        if len(arg.args) == 2:
            if not isinstance(arg.args[1], ast_internal_classes.Int_Literal_Node):
                raise TypeError("KIND argument to REAL must be an integer literal")
            if arg.args[1].value not in ("4", "8"):
                raise NotImplementedError()

            arg.type = "DOUBLE" if arg.args[1].value == "8" else "REAL"
            func_name = "double" if arg.args[1].value == "8" else "float"

            del arg.args[1]
        else:
            arg.type = "REAL"
            func_name = "float"

        arg.name = ast_internal_classes.Name_Node(name=func_name)

        return arg

    @staticmethod
    def _initialize_transformations():
        # dictionary comprehension cannot access class members
        ret = {}
        for (
            name,
            value,
        ) in IntrinsicSDFGTransformation.INTRINSIC_TRANSFORMATIONS.items():
            ret[name] = MathFunctions.MathTransformation(value, "FIRST_ARG")
        return ret

    INTRINSIC_TO_DACE = {
        "MIN": MathTransformation("min", "FIRST_ARG"),
        "MAX": MathTransformation("max", "FIRST_ARG"),
        "SQRT": MathTransformation("sqrt", "FIRST_ARG"),
        "ABS": MathTransformation("abs", "FIRST_ARG"),
        "POW": MathTransformation("pow", "FIRST_ARG"),
        "EXP": MathTransformation("exp", "FIRST_ARG"),
        "EPSILON": MathReplacement(None, generate_epsilon, "FIRST_ARG"),
        # Documentation states that the return type of LOG is always REAL,
        # but the kind is the same as of the first argument.
        # However, we already replaced kind with types used in DaCe.
        # Thus, a REAL that is really DOUBLE will be double in the first argument.
        "LOG": MathTransformation("log", "FIRST_ARG"),
        "MOD": {
            "INTEGER": MathTransformation("Mod", "INTEGER"),
            "REAL": MathTransformation("Mod_float", "REAL"),
            "DOUBLE": MathTransformation("Mod_float", "DOUBLE"),
        },
        "MODULO": {
            "INTEGER": MathTransformation("Modulo", "INTEGER"),
            "REAL": MathTransformation("Modulo_float", "REAL"),
            "DOUBLE": MathTransformation("Modulo_float", "DOUBLE"),
        },
        "FLOOR": {
            "REAL": MathTransformation("floor", "INTEGER"),
            "DOUBLE": MathTransformation("floor", "INTEGER"),
        },
        "SCALE": MathReplacement(None, generate_scale, "FIRST_ARG"),
        "EXPONENT": MathTransformation("frexp", "INTEGER"),
        "INT": MathTransformation("int", "INTEGER"),
        "AINT": MathReplacement("trunc", generate_aint, "FIRST_ARG"),
        "NINT": MathReplacement("iround", generate_aint, "INTEGER"),
        "ANINT": MathReplacement("round", generate_aint, "FIRST_ARG"),
        "REAL": MathReplacement("float", generate_real, "CALL_EXPR"),
        "DBLE": MathTransformation("double", "DOUBLE"),
        "SIN": MathTransformation("sin", "FIRST_ARG"),
        "COS": MathTransformation("cos", "FIRST_ARG"),
        "SINH": MathTransformation("sinh", "FIRST_ARG"),
        "COSH": MathTransformation("cosh", "FIRST_ARG"),
        "TANH": MathTransformation("tanh", "FIRST_ARG"),
        "ASIN": MathTransformation("asin", "FIRST_ARG"),
        "ACOS": MathTransformation("acos", "FIRST_ARG"),
        "ATAN": MathTransformation("atan", "FIRST_ARG"),
        "ATAN2": MathTransformation("atan2", "FIRST_ARG"),
        "DOT_PRODUCT": MathTransformation("__dace_blas_dot", "FIRST_ARG"),
        "TRANSPOSE": MathTransformation(
            "__dace_transpose", "FIRST_ARG", IntrinsicSDFGTransformation.transpose_size
        ),
        "MATMUL": MathTransformation(
            "__dace_matmul", "FIRST_ARG", IntrinsicSDFGTransformation.matmul_size
        ),
        "IBSET": MathTransformation("bitwise_set", "INTEGER"),
        "IEOR": MathTransformation("bitwise_xor", "INTEGER"),
        "ISHFT": MathTransformation("bitwise_shift", "INTEGER"),
        "IBCLR": MathTransformation("bitwise_clear", "INTEGER"),
        "BTEST": MathTransformation("bitwise_test", "INTEGER"),
        "IBITS": MathTransformation("bitwise_extract", "INTEGER"),
        "IAND": MathTransformation("bitwise_and", "INTEGER"),
    }

    class TypeTransformer(IntrinsicNodeTransformer):

        def func_type(self, node: ast_internal_classes.Call_Expr_Node):
            # take the first arg
            arg = node.args[0]
            if isinstance(
                arg,
                (
                    ast_internal_classes.Real_Literal_Node,
                    ast_internal_classes.Double_Literal_Node,
                    ast_internal_classes.Int_Literal_Node,
                    ast_internal_classes.Call_Expr_Node,
                    ast_internal_classes.BinOp_Node,
                    ast_internal_classes.UnOp_Node,
                ),
            ):
                return arg.type
            elif isinstance(
                arg,
                (
                    ast_internal_classes.Name_Node,
                    ast_internal_classes.Array_Subscript_Node,
                    ast_internal_classes.Data_Ref_Node,
                ),
            ):
                return self.get_var_declaration(node.parent, arg).type
            else:
                raise NotImplementedError(type(arg))

        def replace_call(
            self,
            old_call: ast_internal_classes.Call_Expr_Node,
            new_call: ast_internal_classes.FNode,
        ):

            parent = old_call.parent

            # We won't need it if the CallExtractor will properly support nested function calls.
            # Then, all function calls should be a binary op: val = func()
            if isinstance(parent, ast_internal_classes.BinOp_Node):
                if parent.lval == old_call:
                    parent.lval = new_call
                else:
                    parent.rval = new_call
            elif isinstance(parent, ast_internal_classes.UnOp_Node):
                parent.lval = new_call
            elif isinstance(parent, ast_internal_classes.Parenthesis_Expr_Node):
                parent.expr = new_call
            elif isinstance(parent, ast_internal_classes.Call_Expr_Node):
                for idx, arg in enumerate(parent.args):
                    if arg == old_call:
                        parent.args[idx] = new_call
                        break
            else:
                raise NotImplementedError()

        def visit_BinOp_Node(self, binop_node: ast_internal_classes.BinOp_Node):

            if not isinstance(binop_node.rval, ast_internal_classes.Call_Expr_Node):
                return binop_node

            node = binop_node.rval

            name = node.name.name.split("__dace_")

            if len(name) != 2 or name[1] not in MathFunctions.INTRINSIC_TO_DACE:
                return binop_node
            func_name = name[1]

            # Visit all children before we expand this call.
            # We need that to properly get the type.
            new_args = []
            for arg in node.args:
                new_args.append(self.visit(arg))
            node.args = new_args

            input_type = self.func_type(node)
            if input_type == "VOID":
                raise NeedsTypeInferenceException(func_name, node.line_number)

            replacement_rule = MathFunctions.INTRINSIC_TO_DACE[func_name]
            if isinstance(replacement_rule, dict):
                replacement_rule = replacement_rule[input_type]
            if replacement_rule.return_type == "FIRST_ARG":
                return_type = input_type
            elif replacement_rule.return_type == "CALL_EXPR":
                return_type = binop_node.rval.type
            else:
                return_type = replacement_rule.return_type

            if isinstance(replacement_rule, MathFunctions.MathTransformation):
                node.name = ast_internal_classes.Name_Node(
                    name=replacement_rule.function
                )
                node.type = return_type
            else:
                binop_node.rval = replacement_rule.replacement_function(node)

            return binop_node

    @staticmethod
    def dace_functions():

        # list of final dace functions which we create
        funcs = list(MathFunctions.INTRINSIC_TO_DACE.values())
        res = []
        # flatten nested lists
        for f in funcs:
            if isinstance(f, dict):
                res.extend([v.function for k, v in f.items() if v.function is not None])
            else:
                if f.function is not None:
                    res.append(f.function)
        return res

    @staticmethod
    def temporary_functions():

        # temporary functions created by us -> f becomes __dace_f
        # We provide this to tell Fortran parser that these are function calls,
        # not array accesses
        funcs = list(MathFunctions.INTRINSIC_TO_DACE.keys())
        return [f"__dace_{f}" for f in funcs]

    @staticmethod
    def replacable(func_name: str) -> bool:
        return func_name in MathFunctions.INTRINSIC_TO_DACE

    @staticmethod
    def replace(func_name: str) -> ast_internal_classes.FNode:
        return ast_internal_classes.Name_Node(name=f"__dace_{func_name}")

    def has_transformation() -> bool:
        return True

    @staticmethod
    def get_transformation() -> TypeTransformer:
        return MathFunctions.TypeTransformer()
