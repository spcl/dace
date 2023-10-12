
from abc import abstractmethod
import math
from typing import Any

from dace.frontend.fortran import ast_internal_classes

FASTNode = Any

class IntrinsicTransformation:

    def __init__(self, func_name: str, args: ast_internal_classes.Arg_List_Node, line):
        self.func_name = func_name
        self.args = args
        self.line = line

    @staticmethod
    @abstractmethod
    def replaced_name(func_name: str) -> str:
        pass

class SelectedKind(IntrinsicTransformation):

    FUNCTIONS = {
        "SELECTED_INT_KIND": "__dace_selected_int_kind",
        "SELECTED_REAL_KIND": "__dace_selected_real_kind",
    }

    def __init__(self, func_name: str, args: ast_internal_classes.Arg_List_Node, line):
        super().__init__(func_name, args, line)

    @staticmethod
    def replaced_name(func_name: str) -> str:
        return SelectedKind.FUNCTIONS[func_name]

    def replace(self) -> ast_internal_classes.FNode:

        if self.func_name == "__dace_selected_int_kind":
            return ast_internal_classes.Int_Literal_Node(value=str(
                math.ceil((math.log2(math.pow(10, int(self.args.args[0].value))) + 1) / 8)),
                                                         line_number=self.line)
        # This selects the smallest kind that can hold the given number of digits (fp64,fp32 or fp16)
        elif self.func_name == "__dace_selected_real_kind":
            if int(self.args.args[0].value) >= 9 or int(self.args.args[1].value) > 126:
                return ast_internal_classes.Int_Literal_Node(value="8", line_number=self.line)
            elif int(self.args.args[0].value) >= 3 or int(self.args.args[1].value) > 14:
                return ast_internal_classes.Int_Literal_Node(value="4", line_number=self.line)
            else:
                return ast_internal_classes.Int_Literal_Node(value="2", line_number=self.line)

        raise NotImplemented()

class FortranIntrinsics:

    IMPLEMENTATIONS_AST = {
        "SELECTED_INT_KIND": SelectedKind,
        "SELECTED_REAL_KIND": SelectedKind
    }

    IMPLEMENTATIONS_DACE = {
        "__dace_selected_int_kind": SelectedKind,
        "__dace_selected_real_kind": SelectedKind
    }

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
            "EXP": "exp"
        }
        if func_name in replacements:
            return ast_internal_classes.Name_Node(name=replacements[func_name])
        else:
            return ast_internal_classes.Name_Node(name=self.IMPLEMENTATIONS_AST[func_name].replaced_name(func_name))

    def replace_function_reference(self, name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line):

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
            "exp": "DOUBLE"
        }
        if name.name in func_types:
            # FIXME: this will be progressively removed
            call_type = func_types[name.name]
            return ast_internal_classes.Call_Expr_Node(name=name, type=call_type, args=args.args, line_number=line)
        else:
            return self.IMPLEMENTATIONS_DACE[name.name](name.name, args, line).replace()
