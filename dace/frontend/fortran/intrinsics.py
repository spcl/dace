
from abc import abstractmethod
import copy
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
            "SUM": "__dace_sum",
            "ANY": "__dace_any"
        }
        return replacements[func_name]

    @staticmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line) -> ast_internal_classes.FNode:
        func_types = {
            "__dace_sum": "DOUBLE",
            "__dace_any": "DOUBLE"
        }
        # FIXME: Any requires sometimes returning an array of booleans
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

                    par_Decl_Range_Finder(val, rangesrval, rangeposrval, [], self.count, newbody, self.scope_vars, True)

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

class Any(LoopBasedReplacement):

    class AnyLoopNodeLister(NodeVisitor):
        """
        Finds all sum operations that have to be transformed to loops in the AST
        """
        def __init__(self):
            self.nodes: List[ast_internal_classes.FNode] = []

        def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):

            if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
                if node.rval.name.name == "__dace_any":
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

        def _parse_array(self, node: ast_internal_classes.Execution_Part_Node, arg: ast_internal_classes.FNode) -> ast_internal_classes.Array_Subscript_Node:

            # supports syntax ANY(arr)
            if isinstance(arg, ast_internal_classes.Name_Node):
                array_node = ast_internal_classes.Array_Subscript_Node(parent=arg.parent)
                array_node.name = arg

                # If we access SUM(arr) where arr has many dimensions,
                # We need to create a ParDecl_Node for each dimension
                dims = len(self.scope_vars.get_var(node.parent, arg.name).sizes)
                array_node.indices = [ast_internal_classes.ParDecl_Node(type='ALL')] * dims

                return array_node

            # supports syntax ANY(arr(:))
            if isinstance(arg, ast_internal_classes.Array_Subscript_Node):
                return arg

        def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
            newbody = []
            for child in node.execution:
                lister = Any.AnyLoopNodeLister()
                lister.visit(child)
                res = lister.nodes
                if res is not None and len(res) > 0:

                    current = child.lval
                    val = child.rval

                    rvals = []
                    rangesrval = []
                    for i in mywalk(val):
                        if isinstance(i, ast_internal_classes.Call_Expr_Node) and i.name.name == '__dace_any':

                            if len(i.args) > 1:
                                raise NotImplementedError("Fortran ANY with the DIM parameter is not supported!")
                            arg = i.args[0]

                            array_node = self._parse_array(node, arg)
                            if array_node is not None:
                                rvals.append(array_node)

                                if len(rvals) != 1:
                                    raise NotImplementedError("Only one array can be summed")
                                val = rvals[0]
                                rangeposrval = []

                                par_Decl_Range_Finder(val, rangesrval, rangeposrval, [], self.count, newbody, self.scope_vars, True)
                                cond = ast_internal_classes.BinOp_Node(op="==",
                                                                    rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                                    lval=copy.deepcopy(val),
                                                                    line_number=child.line_number)
                            else:

                                # supports syntax ANY(logical op)
                                # the logical op can be:
                                #
                                # (1) arr1 op arr2
                                # where arr1 and arr2 are name node or array subscript node
                                # there, we need to extract shape and verify they are the same
                                #
                                # (2) arr1 op scalar
                                # there, we ignore the scalar because it's not an array
                                if isinstance(arg, ast_internal_classes.BinOp_Node):

                                    left_side_arr  = self._parse_array(node, arg.lval)
                                    right_side_arr  = self._parse_array(node, arg.rval)
                                    has_two_arrays = left_side_arr is not None and right_side_arr is not None

                                    if not has_two_arrays:

                                        # if one side of the operator is scalar, then parsing array
                                        # will return none
                                        dominant_array = left_side_arr
                                        if left_side_arr is None:
                                            dominant_array = right_side_arr

                                        rangeposrval = []
                                        rangeslen_left = []
                                        rangeposrval = []
                                        par_Decl_Range_Finder(dominant_array, rangesrval, rangeposrval, rangeslen_left, self.count, newbody, self.scope_vars, True)
                                        val = arg

                                        cond = copy.deepcopy(val)
                                        if left_side_arr is not None:
                                            cond.lval = dominant_array
                                        if right_side_arr is not None:
                                            cond.rval = dominant_array

                                        continue


                                    if len(left_side_arr.indices) != len(right_side_arr.indices):
                                        raise TypeError("Can't parse Fortran ANY with different array ranks!")

                                    for left_idx, right_idx in zip(left_side_arr.indices, right_side_arr.indices):
                                        if left_idx.type != right_idx.type:
                                            raise TypeError("Can't parse Fortran ANY with different array ranks!")

                                    rangeposrval = []
                                    rangeslen_left = []
                                    rangeposrval = []
                                    par_Decl_Range_Finder(left_side_arr, rangesrval, rangeposrval, rangeslen_left, self.count, newbody, self.scope_vars, True)
                                    val = arg

                                    rangesrval_right = []
                                    rangeslen_right = []
                                    par_Decl_Range_Finder(right_side_arr, rangesrval_right, [], rangeslen_right, self.count, newbody, self.scope_vars, True)

                                    for left_len, right_len in zip(rangeslen_left, rangeslen_right):
                                        if left_len != right_len:
                                            raise TypeError("Can't support Fortran ANY with different array ranks!")

                                    # Now, the loop will be dictated by the left array
                                    # If the access pattern on the right array is different, we need to shfit it - for every dimension.
                                    # For example, we can have arr(1:3) == arr2(3:5)
                                    # Then, loop_idx is from 1 to 3
                                    # arr becomes arr[loop_idx]
                                    # but arr2 must be arr2[loop_idx + 2]
                                    for i in range(len(right_side_arr.indices)):

                                        idx_var = right_side_arr.indices[i]
                                        start_loop = rangesrval[i][0]
                                        end_loop = rangesrval_right[i][0]

                                        difference = int(end_loop.value) - int(start_loop.value) + 1
                                        if difference != 0:
                                            new_index = ast_internal_classes.BinOp_Node(
                                                lval=idx_var,
                                                op="+",
                                                rval=ast_internal_classes.Int_Literal_Node(value=str(difference)),
                                                line_number=child.line_number
                                            )
                                            right_side_arr.indices[i] = new_index

                                    # Now, we need to convert the array to a proper subscript node
                                    cond = copy.deepcopy(val)
                                    cond.lval = left_side_arr
                                    cond.rval = right_side_arr

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

                    # Here begins the specialized implementation
                    body_if = ast_internal_classes.Execution_Part_Node(execution=[
                        ast_internal_classes.BinOp_Node(
                            lval=copy.deepcopy(current),
                            op="=",
                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                            line_number=child.line_number
                        ),
                        # TODO: we should make the `break` generation conditional based on the architecture
                        # For parallel maps, we should have no breaks
                        # For sequential loop, we want a break to be faster
                        #ast_internal_classes.Break_Node(
                        #    line_number=child.line_number
                        #)
                    ])
                    body = ast_internal_classes.If_Stmt_Node(
                        cond=cond,
                        body=body_if,
                        body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
                        line_number=child.line_number
                    )
                    # Here ends the specialized implementation

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
        "SUM": Sum,
        "ANY": Any
    }

    IMPLEMENTATIONS_DACE = {
        "__dace_selected_int_kind": SelectedKind,
        "__dace_selected_real_kind": SelectedKind,
        "__dace_sum": Sum,
        "__dace_any": Any
    }

    def __init__(self):
        self._transformations_to_run = set()

    def transformations(self) -> Set[Type[NodeTransformer]]:
        return self._transformations_to_run

    @staticmethod
    def function_names() -> List[str]:
        return list(FortranIntrinsics.IMPLEMENTATIONS_DACE.keys())

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
