
import math
from typing import Any

from dace.frontend.fortran import ast_internal_classes

FASTNode = Any

class FortranIntrinsics:

    def replace_function_name(self, node: FASTNode) -> ast_internal_classes.Name_Node:

        func_name = node.string
        replacements = {
            "INT": "__dace_int",
            "DBLE": "__dace_dble",
            "SQRT": "sqrt",
            "COSH": "cosh",
            "ABS": "abs",
            "MIN": "min",
            "MAX": "max",
            "EXP": "exp",
            "EPSILON": "__dace_epsilon",
            "TANH": "tanh",
            "SUM": "__dace_sum",
            "SIGN": "__dace_sign",
            "EXP": "exp",
            "SELECTED_INT_KIND": "__dace_selected_int_kind",
            "SELECTED_REAL_KIND": "__dace_selected_real_kind",
        }
        return ast_internal_classes.Name_Node(name=replacements[func_name])

    def replace_function_reference(self, name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line):

        if name.name == "__dace_selected_int_kind":
            return ast_internal_classes.Int_Literal_Node(value=str(
                math.ceil((math.log2(math.pow(10, int(args.args[0].value))) + 1) / 8)),
                                                         line_number=line)
        # This selects the smallest kind that can hold the given number of digits (fp64,fp32 or fp16)
        elif name.name == "__dace_selected_real_kind":
            if int(args.args[0].value) >= 9 or int(args.args[1].value) > 126:
                return ast_internal_classes.Int_Literal_Node(value="8", line_number=line)
            elif int(args.args[0].value) >= 3 or int(args.args[1].value) > 14:
                return ast_internal_classes.Int_Literal_Node(value="4", line_number=line)
            else:
                return ast_internal_classes.Int_Literal_Node(value="2", line_number=line)

        func_types = {
            "__dace_int": "INT",
            "__dace_dble": "DOUBLE",
            "sqrt": "DOUBLE",
            "cosh": "DOUBLE",
            "abs": "DOUBLE",
            "min": "DOUBLE",
            "max": "DOUBLE",
            "exp": "DOUBLE",
            "__dace_epsilon": "DOUBLE",
            "tanh": "DOUBLE",
            "__dace_sum": "DOUBLE",
            "__dace_sign": "DOUBLE",
            "exp": "DOUBLE",
            "__dace_selected_int_kind": "INT",
            "__dace_selected_real_kind": "INT",
        }
        call_type = func_types[name.name]
        return ast_internal_classes.Call_Expr_Node(name=name, type=call_type, args=args.args, line_number=line)

