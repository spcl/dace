
from abc import abstractmethod
import math
from typing import Any, List, Set, Type

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_transforms import NodeVisitor, NodeTransformer, ParentScopeAssigner, ScopeVarsDeclarations, par_Decl_Range_Finder, mywalk

FASTNode = Any

class IntrinsicTransformation:

    @staticmethod
    @abstractmethod
    def replaced_name(func_name: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line) -> ast_internal_classes.FNode:
        pass

    @staticmethod
    def has_transformation() -> bool:
        return False

class SelectedKind(IntrinsicTransformation):

    FUNCTIONS = {
        "SELECTED_INT_KIND": "__dace_selected_int_kind",
        "SELECTED_REAL_KIND": "__dace_selected_real_kind",
    }

    @staticmethod
    def replaced_name(func_name: str) -> str:
        return SelectedKind.FUNCTIONS[func_name]

    @staticmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line) -> ast_internal_classes.FNode:

        if func_name.name == "__dace_selected_int_kind":
            return ast_internal_classes.Int_Literal_Node(value=str(
                math.ceil((math.log2(math.pow(10, int(args.args[0].value))) + 1) / 8)),
                                                         line_number=line)
        # This selects the smallest kind that can hold the given number of digits (fp64,fp32 or fp16)
        elif func_name.name == "__dace_selected_real_kind":
            if int(args.args[0].value) >= 9 or int(args.args[1].value) > 126:
                return ast_internal_classes.Int_Literal_Node(value="8", line_number=line)
            elif int(args.args[0].value) >= 3 or int(args.args[1].value) > 14:
                return ast_internal_classes.Int_Literal_Node(value="4", line_number=line)
            else:
                return ast_internal_classes.Int_Literal_Node(value="2", line_number=line)

        raise NotImplemented()

class LoopBasedReplacement:

    @staticmethod
    def replaced_name(func_name: str) -> str:
        replacements = {
            "SUM": "__dace_sum"
        }
        return replacements[func_name]

    @staticmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line) -> ast_internal_classes.FNode:
        func_types = {
            "__dace_sum": "DOUBLE"
        }
        call_type = func_types[func_name.name]
        return ast_internal_classes.Call_Expr_Node(name=func_name, type=call_type, args=args.args, line_number=line)

    @staticmethod
    def has_transformation() -> bool:
        return True

class Sum(LoopBasedReplacement):

    class SumLoopNodeLister(NodeVisitor):
        """
        Finds all sum operations that have to be transformed to loops in the AST
        """
        def __init__(self):
            self.nodes: List[ast_internal_classes.FNode] = []

        def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):

            if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
                if node.rval.name.name == "__dace_sum":
                    self.nodes.append(node)

        def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
            return

    class Transformation(NodeTransformer):

        """
        Transforms the AST by removing array sums and replacing them with loops
        """
        def __init__(self, ast):
            self.count = 0
            ParentScopeAssigner().visit(ast)
            self.scope_vars = ScopeVarsDeclarations()
            self.scope_vars.visit(ast)

        def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
            newbody = []
            for child in node.execution:
                lister = Sum.SumLoopNodeLister()
                lister.visit(child)
                res = lister.nodes
                if res is not None and len(res) > 0:

                    current = child.lval
                    val = child.rval

                    rvals = []
                    for i in mywalk(val):
                        if isinstance(i, ast_internal_classes.Call_Expr_Node) and i.name.name == '__dace_sum':

                            for arg in i.args:

                                # supports syntax SUM(arr)
                                if isinstance(arg, ast_internal_classes.Name_Node):
                                    array_node = ast_internal_classes.Array_Subscript_Node(parent=arg.parent)
                                    array_node.name = arg

                                    # If we access SUM(arr) where arr has many dimensions,
                                    # We need to create a ParDecl_Node for each dimension
                                    dims = len(self.scope_vars.get_var(node.parent, arg.name).sizes)
                                    array_node.indices = [ast_internal_classes.ParDecl_Node(type='ALL')] * dims

                                    rvals.append(array_node)

                                # supports syntax SUM(arr(:))
                                if isinstance(arg, ast_internal_classes.Array_Subscript_Node):
                                    rvals.append(arg)

                    if len(rvals) != 1:
                        raise NotImplementedError("Only one array can be summed")
                    val = rvals[0]
                    rangeposrval = []
                    rangesrval = []

                    par_Decl_Range_Finder(val, rangesrval, rangeposrval, self.count, newbody, self.scope_vars, True)

                    # Initialize the result variable
                    newbody.append(
                        ast_internal_classes.BinOp_Node(
                            lval=current,
                            op="=",
                            rval=ast_internal_classes.Int_Literal_Node(value="0"),
                            line_number=child.line_number
                        )
                    )
                    range_index = 0
                    body = ast_internal_classes.BinOp_Node(lval=current,
                                                        op="=",
                                                        rval=ast_internal_classes.BinOp_Node(
                                                            lval=current,
                                                            op="+",
                                                            rval=val,
                                                            line_number=child.line_number),
                                                        line_number=child.line_number)
                    for i in rangesrval:
                        initrange = i[0]
                        finalrange = i[1]
                        init = ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                            op="=",
                            rval=initrange,
                            line_number=child.line_number)
                        cond = ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                            op="<=",
                            rval=finalrange,
                            line_number=child.line_number)
                        iter = ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                            op="=",
                            rval=ast_internal_classes.BinOp_Node(
                                lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                                op="+",
                                rval=ast_internal_classes.Int_Literal_Node(value="1")),
                            line_number=child.line_number)
                        current_for = ast_internal_classes.Map_Stmt_Node(
                            init=init,
                            cond=cond,
                            iter=iter,
                            body=ast_internal_classes.Execution_Part_Node(execution=[body]),
                            line_number=child.line_number)
                        body = current_for
                        range_index += 1

                    newbody.append(body)

                    self.count = self.count + range_index
                else:
                    newbody.append(self.visit(child))
            return ast_internal_classes.Execution_Part_Node(execution=newbody)

class FortranIntrinsics:

    IMPLEMENTATIONS_AST = {
        "SELECTED_INT_KIND": SelectedKind,
        "SELECTED_REAL_KIND": SelectedKind,
        "SUM": Sum
    }

    IMPLEMENTATIONS_DACE = {
        "__dace_selected_int_kind": SelectedKind,
        "__dace_selected_real_kind": SelectedKind,
        "__dace_sum": Sum
    }

    def __init__(self):
        self._transformations_to_run = set()

    def transformations(self) -> Set[Type[NodeTransformer]]:
        return self._transformations_to_run

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
            "SIGN": "__dace_sign",
            "EXP": "exp"
        }
        if func_name in replacements:
            return ast_internal_classes.Name_Node(name=replacements[func_name])
        else:

            if self.IMPLEMENTATIONS_AST[func_name].has_transformation():
                self._transformations_to_run.add(self.IMPLEMENTATIONS_AST[func_name].Transformation)

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
            "__dace_sign": "DOUBLE",
        }
        if name.name in func_types:
            # FIXME: this will be progressively removed
            call_type = func_types[name.name]
            return ast_internal_classes.Call_Expr_Node(name=name, type=call_type, args=args.args, line_number=line)
        else:
            return self.IMPLEMENTATIONS_DACE[name.name].replace(name, args, line)
