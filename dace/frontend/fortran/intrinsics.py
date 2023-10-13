
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
            "ANY": "__dace_any",
            "ALL": "__dace_all"
        }
        return replacements[func_name]

    @staticmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line) -> ast_internal_classes.FNode:
        func_types = {
            "__dace_sum": "DOUBLE",
            "__dace_any": "INTEGER",
            "__dace_all": "INTEGER"
        }
        # FIXME: Any requires sometimes returning an array of booleans
        call_type = func_types[func_name.name]
        return ast_internal_classes.Call_Expr_Node(name=func_name, type=call_type, args=args.args, line_number=line)

    @staticmethod
    def has_transformation() -> bool:
        return True

class LoopBasedReplacementVisitor(NodeVisitor):

    """
    Finds all intrinsic operations that have to be transformed to loops in the AST
    """
    def __init__(self, func_name: str):
        self._func_name = func_name
        self.nodes: List[ast_internal_classes.FNode] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):

        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
            if node.rval.name.name == self._func_name:
                self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return

class LoopBasedReplacementTransformation(NodeTransformer):

    """
    Transforms the AST by removing intrinsic call and replacing it with loops
    """
    def __init__(self, ast):
        self.count = 0
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations()
        self.scope_vars.visit(ast)

        self.rvals = []


    @abstractmethod
    def func_name(self) -> str:
        pass

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):
        pass

    @abstractmethod
    def _summarize_args(self, node: ast_internal_classes.FNode, new_func_body: List[ast_internal_classes.FNode]):
        pass

    @abstractmethod
    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:
        pass

    @abstractmethod
    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:
        pass

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):

        newbody = []
        for child in node.execution:
            lister = LoopBasedReplacementVisitor(self.func_name())
            lister.visit(child)
            res = lister.nodes

            if res is None or len(res) == 0:
                newbody.append(self.visit(child))
                continue

            self.loop_ranges = []
            # We need to reinitialize variables as the class is reused for transformation between different
            # calls to the same intrinsic.
            self._initialize()

            # Visit all intrinsic arguments and extract arrays
            for i in mywalk(child.rval):
                if isinstance(i, ast_internal_classes.Call_Expr_Node) and i.name.name == self.func_name():
                    self._parse_call_expr_node(i)

            # Verify that all of intrinsic args are correct and prepare them for loop generation
            self._summarize_args(child, newbody)

            # Initialize the result variable
            newbody.append(self._initialize_result(child))

            # Generate the intrinsic-specific logic inside loop body
            body = self._generate_loop_body(child)

            # Now generate the multi-dimensiona loop header and updates
            range_index = 0
            for i in self.loop_ranges:
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
        return ast_internal_classes.Execution_Part_Node(execution=newbody)

class Sum(LoopBasedReplacement):

    """
        In this class, we implement the transformation for Fortran intrinsic SUM(:)
        We support two ways of invoking the function - by providing array name and array subscript.
        We do NOT support the *DIM* argument.

        During the loop construction, we add a single variable storing the partial result.
        Then, we generate a binary node accumulating the result.
    """

    class Transformation(LoopBasedReplacementTransformation):

        def __init__(self, ast):
            super().__init__(ast)

        def func_name(self) -> str:
            return "__dace_sum"

        def _initialize(self):
            self.rvals = []
            self.argument_variable = None

        def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

            for arg in node.args:

                # supports syntax SUM(arr)
                if isinstance(arg, ast_internal_classes.Name_Node):
                    array_node = ast_internal_classes.Array_Subscript_Node(parent=arg.parent)
                    array_node.name = arg

                    # If we access SUM(arr) where arr has many dimensions,
                    # We need to create a ParDecl_Node for each dimension
                    dims = len(self.scope_vars.get_var(node.parent, arg.name).sizes)
                    array_node.indices = [ast_internal_classes.ParDecl_Node(type='ALL')] * dims

                    self.rvals.append(array_node)

                # supports syntax SUM(arr(:))
                if isinstance(arg, ast_internal_classes.Array_Subscript_Node):
                    self.rvals.append(arg)


        def _summarize_args(self, node: ast_internal_classes.FNode, new_func_body: List[ast_internal_classes.FNode]):

            if len(self.rvals) != 1:
                raise NotImplementedError("Only one array can be summed")

            self.argument_variable = self.rvals[0]

            par_Decl_Range_Finder(self.argument_variable, self.loop_ranges, [], [], self.count, new_func_body, self.scope_vars, True)

        def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

            return ast_internal_classes.BinOp_Node(
                lval=node.lval,
                op="=",
                rval=ast_internal_classes.Int_Literal_Node(value="0"),
                line_number=node.line_number
            )

        def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

            return ast_internal_classes.BinOp_Node(
                lval=node.lval,
                op="=",
                rval=ast_internal_classes.BinOp_Node(
                    lval=node.lval,
                    op="+",
                    rval=self.argument_variable,
                    line_number=node.line_number
                ),
                line_number=node.line_number
            )

class AnyAllTransformation(LoopBasedReplacementTransformation):

    def __init__(self, ast):
        super().__init__(ast)

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

    def _initialize(self):
        self.rvals = []

        self.first_array = None
        self.second_array = None
        self.dominant_array = None
        self.cond = None

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        if len(node.args) > 1:
            raise NotImplementedError("Fortran ANY with the DIM parameter is not supported!")
        arg = node.args[0]

        array_node = self._parse_array(node, arg)
        if array_node is not None:

            self.first_array = array_node

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
            if not isinstance(arg, ast_internal_classes.BinOp_Node):
                return

            self.first_array  = self._parse_array(node, arg.lval)
            self.second_array  = self._parse_array(node, arg.rval)
            has_two_arrays = self.first_array is not None and self.second_array is not None

            # array and scalar - simplified case
            if not has_two_arrays:

                # if one side of the operator is scalar, then parsing array
                # will return none
                self.dominant_array = self.first_array
                if self.dominant_array is None:
                    self.dominant_array = self.second_array

                # replace the array subscript node in the binary operation
                # ignore this when the operand is a scalar
                self.cond = copy.deepcopy(arg)
                if self.first_array is not None:
                    self.cond.lval = self.dominant_array
                if self.second_array is not None:
                    self.cond.rval = self.dominant_array

                return


            if len(self.first_array.indices) != len(self.second_array.indices):
                raise TypeError("Can't parse Fortran ANY with different array ranks!")

            for left_idx, right_idx in zip(self.first_array.indices, self.second_array.indices):
                if left_idx.type != right_idx.type:
                    raise TypeError("Can't parse Fortran ANY with different array ranks!")

            # Now, we need to convert the array to a proper subscript node
            self.cond = copy.deepcopy(arg)
            self.cond.lval = self.first_array
            self.cond.rval = self.second_array

    def _summarize_args(self, node: ast_internal_classes.FNode, new_func_body: List[ast_internal_classes.FNode]):

        # The main argument is an array, not a binary operation
        if self.cond is None:

            par_Decl_Range_Finder(self.first_array, self.loop_ranges, [], [], self.count, new_func_body, self.scope_vars, True)
            self.cond = ast_internal_classes.BinOp_Node(
                op="==",
                rval=ast_internal_classes.Int_Literal_Node(value="1"),
                lval=copy.deepcopy(self.first_array),
                line_number=node.line_number
            )
            return

        # we have a binary operation with an array and a scalar
        if self.dominant_array is not None:

            par_Decl_Range_Finder(self.dominant_array, self.loop_ranges, [], [], self.count, new_func_body, self.scope_vars, True)
            return

        # we have a binary operation with two arrays

        rangeslen_left = []
        par_Decl_Range_Finder(self.first_array, self.loop_ranges, [], rangeslen_left, self.count, new_func_body, self.scope_vars, True)

        loop_ranges_right = []
        rangeslen_right = []
        par_Decl_Range_Finder(self.second_array, loop_ranges_right, [], rangeslen_right, self.count, new_func_body, self.scope_vars, True)

        for left_len, right_len in zip(rangeslen_left, rangeslen_right):
            if left_len != right_len:
                raise TypeError("Can't support Fortran ANY with different array ranks!")

        # Now, the loop will be dictated by the left array
        # If the access pattern on the right array is different, we need to shfit it - for every dimension.
        # For example, we can have arr(1:3) == arr2(3:5)
        # Then, loop_idx is from 1 to 3
        # arr becomes arr[loop_idx]
        # but arr2 must be arr2[loop_idx + 2]
        for i in range(len(self.second_array.indices)):

            idx_var = self.second_array.indices[i]
            start_loop = self.loop_ranges[i][0]
            end_loop = loop_ranges_right[i][0]

            difference = int(end_loop.value) - int(start_loop.value)
            if difference != 0:
                new_index = ast_internal_classes.BinOp_Node(
                    lval=idx_var,
                    op="+",
                    rval=ast_internal_classes.Int_Literal_Node(value=str(difference)),
                    line_number=node.line_number
                )
                self.second_array.indices[i] = new_index

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        init_value = None
        if 'any' in self.func_name():
            init_value = "0"
        else:
            init_value = "1"

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.Int_Literal_Node(value=init_value),
            line_number=node.line_number
        )

    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:
        
        """
        For any, we check if the condition is true and then set the value to true
        For all, we check if the condition is NOT true and then set the value to false
        """

        assign_value = None
        if 'any' in self.func_name():
            assign_value = "1"
        else:
            assign_value = "0"

        body_if = ast_internal_classes.Execution_Part_Node(execution=[
            ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="=",
                rval=ast_internal_classes.Int_Literal_Node(value=assign_value),
                line_number=node.line_number
            ),
            # TODO: we should make the `break` generation conditional based on the architecture
            # For parallel maps, we should have no breaks
            # For sequential loop, we want a break to be faster
            #ast_internal_classes.Break_Node(
            #    line_number=node.line_number
            #)
        ])

        condition = None
        if 'any' in self.func_name():
            condition = self.cond
        else:
            condition = ast_internal_classes.UnOp_Node(
                op="not",
                lval=self.cond
            )

        return ast_internal_classes.If_Stmt_Node(
            cond=condition,
            body=body_if,
            body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
            line_number=node.line_number
        )

class Any(LoopBasedReplacement):

    """
        In this class, we implement the transformation for Fortran intrinsic ANY
        We support three ways of invoking the function - by providing array name, array subscript,
        and a binary operation.
        We do NOT support the *DIM* argument.

        First, we split the implementation between three scenarios:
        (1) ANY(arr)
        (2) ANY(arr1 op arr2)
        (3) ANY(arr1 op scalar)
        Depending on the scenario, we verify if all participating arrays have the same rank.
        We determine the loop range based on the arrays, and convert all array accesses to depend on
        the loop. We take special care for situations where arrays have different subscripts, e.g.,
        arr1(1:3) op arr2(5:7) - the second array needs a shift when indexing based on loop iterator.

        During the loop construction, we add a single variable storing the partial result.
        Then, we generate an if condition inside the loop to check if the value is true or not.
        For (1), we check if the array entry is equal to 1.
        For (2), we reuse the provided binary operation.
        When the condition is true, we set the value to true and exit.
    """
    class Transformation(AnyAllTransformation):

        def __init__(self, ast):
            super().__init__(ast)

        def func_name(self) -> str:
            return "__dace_any"

class All(LoopBasedReplacement):

    """
        In this class, we implement the transformation for Fortran intrinsic ALL.
        The implementation is very similar to ANY.
        The main difference is that we initialize the partial result to 1,
        and set it to 0 if any of the evaluated conditions is false.
    """
    class Transformation(AnyAllTransformation):

        def __init__(self, ast):
            super().__init__(ast)

        def func_name(self) -> str:
            return "__dace_all"


class FortranIntrinsics:

    IMPLEMENTATIONS_AST = {
        "SELECTED_INT_KIND": SelectedKind,
        "SELECTED_REAL_KIND": SelectedKind,
        "SUM": Sum,
        "ANY": Any,
        "ALL": All
    }

    IMPLEMENTATIONS_DACE = {
        "__dace_selected_int_kind": SelectedKind,
        "__dace_selected_real_kind": SelectedKind,
        "__dace_sum": Sum,
        "__dace_any": Any,
        "__dace_all": All
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
