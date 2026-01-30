# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""

This module contains Fortran intrinsics that are transformed into explicit loop
structures for array operations. These are typically reduction operations and
array manipulation functions.

Implemented intrinsics:
- SUM: Array summation
- PRODUCT: Array product
- ANY: Logical OR reduction
- ALL: Logical AND reduction
- COUNT: Count true elements
- MINVAL: Find minimum value
- MAXVAL: Find maximum value
- MERGE: Conditional selection

"""

import copy
from abc import abstractmethod
from typing import List, Optional, Tuple

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_transforms import NodeVisitor, par_Decl_Range_Finder
from dace.frontend.fortran.ast_utils import fortrantypes2dacetypes, mywalk, is_literal
from dace.frontend.fortran.intrinsics.base import IntrinsicNodeTransformer


class LoopBasedReplacement:
    """
    Parent class for all implementations of loop-based intrinsics.

    Each variant defines its own Transformation class that
    implements the actual AST transformation.
    """

    INTRINSIC_TO_DACE = {
        "SUM": "__dace_sum",
        "PRODUCT": "__dace_product",
        "ANY": "__dace_any",
        "ALL": "__dace_all",
        "COUNT": "__dace_count",
        "MINVAL": "__dace_minval",
        "MAXVAL": "__dace_maxval",
        "MERGE": "__dace_merge",
    }

    @staticmethod
    def replaced_name(func_name: str) -> str:
        return LoopBasedReplacement.INTRINSIC_TO_DACE[func_name]

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
        self.calls: List[ast_internal_classes.FNode] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
            if node.rval.name.name == self._func_name:
                self.nodes.append(node)
                self.calls.append(node.rval)
        self.visit(node.lval)
        self.visit(node.rval)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        if node.name.name == self._func_name:
            if node not in self.calls:
                self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class LoopBasedReplacementTransformation(IntrinsicNodeTransformer):
    """
    Transforms the AST by removing intrinsic call and replacing it with loops
    """

    def __init__(self):
        self.count = 0
        self.rvals = []

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):
        pass

    @abstractmethod
    def _summarize_args(
        self,
        exec_node: ast_internal_classes.Execution_Part_Node,
        node: ast_internal_classes.FNode,
        new_func_body: List[ast_internal_classes.FNode],
    ):
        pass

    @abstractmethod
    def _initialize_result(
        self, node: ast_internal_classes.FNode
    ) -> Optional[ast_internal_classes.BinOp_Node]:
        pass

    @abstractmethod
    def _generate_loop_body(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.BinOp_Node:
        pass

    def _skip_result_assignment(self):
        return False

    """
        When replacing Fortran's AST reference to an intrinsic function, we set a dummy variable with VOID type.
        The reason is that at the point, we do not know the types of arguments. For many intrinsics, the return
        type will depend on the input types.

        When transforming the AST, we gather all scopes and variable declarations in that scope.
        Then, we can query the types of input arguments and properly determine the return type.

        Both the type of the variable and its corresponding Var_Decl_node need to be updated!
    """

    @abstractmethod
    def _update_result_type(self, var: ast_internal_classes.Name_Node):
        pass

    def _parse_array(
        self,
        node: ast_internal_classes.Execution_Part_Node,
        arg: ast_internal_classes.FNode,
        dims_count: Optional[int] = -1,
    ) -> ast_internal_classes.Array_Subscript_Node:

        # supports syntax func(arr)
        if isinstance(arg, ast_internal_classes.Name_Node):
            # If we access SUM(arr) where arr has many dimensions,
            # We need to create a ParDecl_Node for each dimension
            array_sizes = self.get_var_declaration(node.parent, arg).sizes
            if array_sizes is None:

                raise NeedsTypeInferenceException(self.func_name(), node.line_number)

            dims = len(array_sizes)

            # it's a scalar!
            if dims == 0:
                return None

            if isinstance(arg, ast_internal_classes.Name_Node):
                return ast_internal_classes.Array_Subscript_Node(
                    name=arg,
                    parent=arg.parent,
                    type="VOID",
                    indices=[ast_internal_classes.ParDecl_Node(type="ALL")] * dims,
                )

        # supports syntax func(struct%arr) and func(struct%arr(:))
        if isinstance(arg, ast_internal_classes.Data_Ref_Node):

            array_sizes = self.get_var_declaration(node.parent, arg).sizes
            if array_sizes is None:

                raise NeedsTypeInferenceException(self.func_name(), node.line_number)

            dims = len(array_sizes)

            # it's a scalar!
            if dims == 0:
                return None

            _, _, cur_val = self.ast.structures.find_definition(self.scope_vars, arg)

            if isinstance(cur_val.part_ref, ast_internal_classes.Data_Ref_Node):
                raise TypeError(
                    "Unexpected nested Data_Ref_Node in structure definition"
                )

            if isinstance(cur_val.part_ref, ast_internal_classes.Name_Node):
                cur_val.part_ref = ast_internal_classes.Array_Subscript_Node(
                    name=cur_val.part_ref,
                    parent=arg.parent,
                    type="VOID",
                    indices=[ast_internal_classes.ParDecl_Node(type="ALL")] * dims,
                )
            return arg

        # supports syntax func(arr(:))
        if isinstance(arg, ast_internal_classes.Array_Subscript_Node):
            return arg

        return None

    def _parse_binary_op(
        self,
        node: ast_internal_classes.Call_Expr_Node,
        arg: ast_internal_classes.BinOp_Node,
    ) -> Tuple[
        ast_internal_classes.Array_Subscript_Node,
        Optional[ast_internal_classes.Array_Subscript_Node],
        ast_internal_classes.BinOp_Node,
    ]:
        """
        Supports passing binary operations as an input to function.
        In both cases, we extract the arrays used, and return a brand
        new binary operation that has array references replaced.
        We return both arrays (second optionaly None) and the binary op.

        The binary op can be:

        (1) arr1 op arr2
        where arr1 and arr2 are name node or array subscript node
        #there, we need to extract shape and verify they are the same

        (2) arr1 op scalar
        there, we ignore the scalar because it's not an array

        """
        if not isinstance(arg, ast_internal_classes.BinOp_Node):
            return (None, None, None)

        first_array = self._parse_array(node, arg.lval)
        second_array = self._parse_array(node, arg.rval)
        has_two_arrays = first_array is not None and second_array is not None

        # array and scalar - simplified case
        if not has_two_arrays:

            # if one side of the operator is scalar, then parsing array
            # will return none
            dominant_array = first_array
            if dominant_array is None:
                dominant_array = second_array

            # replace the array subscript node in the binary operation
            # ignore this when the operand is a scalar
            cond = copy.deepcopy(arg)
            if first_array is not None:
                cond.lval = dominant_array
            if second_array is not None:
                cond.rval = dominant_array

            return (dominant_array, None, cond)

        if len(first_array.indices) != len(second_array.indices):
            raise TypeError("Can't parse Fortran binary op with different array ranks!")

        for left_idx, right_idx in zip(first_array.indices, second_array.indices):
            if left_idx.type != right_idx.type:
                raise TypeError(
                    "Can't parse Fortran binary op with different array ranks!"
                )

        # Now, we need to convert the array to a proper subscript node
        cond = copy.deepcopy(arg)
        cond.lval = first_array
        cond.rval = second_array

        return (first_array, second_array, cond)

    def _adjust_array_ranges(
        self,
        node: ast_internal_classes.FNode,
        array: ast_internal_classes.Array_Subscript_Node,
        loop_ranges_main: list,
        loop_ranges_array: list,
    ):
        """
        When given a binary operator with arrays as an argument to the intrinsic,
        one array will dictate loop range.
        However, the other array can potentially have a different access range.
        Thus, we need to add an offset to the loop iterator when accessing array elements.

        If the access pattern on the right array is different, we need to shfit it - for every dimension.
        For example, we can have arr(1:3) == arr2(3:5)
        Then, loop_idx is from 1 to 3
        arr becomes arr[loop_idx]
        but arr2 must be arr2[loop_idx + 2]
        """
        for i in range(len(array.indices)):

            idx_var = array.indices[i]
            start_loop = loop_ranges_main[i][0]
            end_loop = loop_ranges_array[i][0]

            difference = ast_internal_classes.BinOp_Node(
                lval=end_loop, op="-", rval=start_loop, line_number=node.line_number
            )
            new_index = ast_internal_classes.BinOp_Node(
                lval=idx_var, op="+", rval=difference, line_number=node.line_number
            )
            array.indices[i] = new_index

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
                if (
                    isinstance(i, ast_internal_classes.Call_Expr_Node)
                    and i.name.name == self.func_name()
                ):
                    self._parse_call_expr_node(i)

            # Verify that all of intrinsic args are correct and prepare them for loop generation
            self._summarize_args(node, child, newbody)

            # Change the type of result variable
            self._update_result_type(child.lval)

            # Initialize the result variable
            init_stm = self._initialize_result(child)
            if init_stm is not None:
                newbody.append(init_stm)

            # Generate the intrinsic-specific logic inside loop body
            body = self._generate_loop_body(child)

            # Now generate the multi-dimensiona loop header and updates
            range_index = 0
            for i in self.loop_ranges:
                initrange = i[0]
                finalrange = i[1]
                init = ast_internal_classes.BinOp_Node(
                    lval=ast_internal_classes.Name_Node(
                        name="tmp_parfor_" + str(self.count + range_index)
                    ),
                    op="=",
                    rval=initrange,
                    line_number=child.line_number,
                )
                cond = ast_internal_classes.BinOp_Node(
                    lval=ast_internal_classes.Name_Node(
                        name="tmp_parfor_" + str(self.count + range_index)
                    ),
                    op="<=",
                    rval=finalrange,
                    line_number=child.line_number,
                )
                iter = ast_internal_classes.BinOp_Node(
                    lval=ast_internal_classes.Name_Node(
                        name="tmp_parfor_" + str(self.count + range_index)
                    ),
                    op="=",
                    rval=ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(
                            name="tmp_parfor_" + str(self.count + range_index)
                        ),
                        op="+",
                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                    ),
                    line_number=child.line_number,
                )
                current_for = ast_internal_classes.Map_Stmt_Node(
                    init=init,
                    cond=cond,
                    iter=iter,
                    body=ast_internal_classes.Execution_Part_Node(execution=[body]),
                    line_number=child.line_number,
                )
                body = current_for
                range_index += 1

            newbody.append(body)

            self.count = self.count + range_index
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class SumProduct(LoopBasedReplacementTransformation):
    """
    Parent transformation class for implementations of sum and product.

    Both have the same logic: we aggregate inputs, but with different
    arithmetical operations. They differ in two ways: initial value
    of aggregation (0 for sum, 1 for product) and the arithmetical operation.
    """

    def _initialize(self):
        self.rvals = []
        self.argument_variable = None

        self.function_name = "Sum/Product"

    def _update_result_type(self, var: ast_internal_classes.Name_Node):
        """
        For both SUM and PRODUCT, the result type depends on the input variable.
        """
        input_type = self.get_var_declaration(var.parent, self.argument_variable)

        var_decl = self.get_var_declaration(var.parent, var)
        var.type = input_type.type
        var_decl.type = input_type.type

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        for arg in node.args:

            array_node = self._parse_array(node, arg)

            if array_node is not None:
                self.rvals.append(array_node)
            else:
                raise NotImplementedError(
                    "We do not support non-array arguments for SUM/PRODUCT"
                )

    def _summarize_args(
        self,
        exec_node: ast_internal_classes.Execution_Part_Node,
        node: ast_internal_classes.FNode,
        new_func_body: List[ast_internal_classes.FNode],
    ):

        if len(self.rvals) != 1:
            raise NotImplementedError("Only one array can be summed")

        self.argument_variable = self.rvals[0]

        par_Decl_Range_Finder(
            self.argument_variable,
            self.loop_ranges,
            [],
            self.count,
            new_func_body,
            self.scope_vars,
            self.ast.structures,
            True,
        )

    def _initialize_result(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.Int_Literal_Node(value=self._result_init_value()),
            line_number=node.line_number,
        )

    def _generate_loop_body(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.BinOp_Node(
                lval=node.lval,
                op=self._result_update_op(),
                rval=self.argument_variable,
                line_number=node.line_number,
            ),
            line_number=node.line_number,
        )


class Sum(LoopBasedReplacement):
    """
    In this class, we implement the transformation for Fortran intrinsic SUM(:)
    We support two ways of invoking the function - by providing array name and array subscript.
    We do NOT support the *DIM* argument.

    During the loop construction, we add a single variable storing the partial result.
    Then, we generate a binary node accumulating the result.
    """

    class Transformation(SumProduct):

        @staticmethod
        def func_name() -> str:
            return "__dace_sum"

        def _result_init_value(self):
            return "0"

        def _result_update_op(self):
            return "+"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        return [], first_arg.type


class Product(LoopBasedReplacement):
    """
    In this class, we implement the transformation for Fortran intrinsic PRODUCT(:)
    We support two ways of invoking the function - by providing array name and array subscript.
    We do NOT support the *DIM* and *MASK* arguments.

    During the loop construction, we add a single variable storing the partial result.
    Then, we generate a binary node accumulating the result.
    """

    class Transformation(SumProduct):

        @staticmethod
        def func_name() -> str:
            return "__dace_product"

        def _result_init_value(self):
            return "1"

        def _result_update_op(self):
            return "*"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        return [], first_arg.type


class AnyAllCountTransformation(LoopBasedReplacementTransformation):
    """
    Parent transformation class for implementations of three very similar
    intrinsics: any, all, and count. All count or check the number of values
    that satisfy a given condition. Thus, they implement a very similar loop pattern.

    For any, we initialize the aggregated value to 0 and set to 1 as soon as we have a match.
    For all, we initialize to 1 and set the value to 0 as soon as we have a value not matching the condition.
    For count, we initalize to 0 and increment every time we have a match.
    """

    def _initialize(self):
        self.rvals = []

        self.first_array = None
        self.second_array = None
        self.dominant_array = None
        self.cond = None

    def _update_result_type(self, var: ast_internal_classes.Name_Node):
        """
        For all functions, the result type is INTEGER.
        Theoretically, we should return LOGICAL for ANY and ALL,
        but we no longer use booleans on DaCe side.
        """
        var_decl = self.get_var_declaration(var.parent, var)
        var.type = "INTEGER"
        var_decl.type = "INTEGER"

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        if len(node.args) > 1:
            raise NotImplementedError(
                "Fortran ANY with the DIM parameter is not supported!"
            )
        arg = node.args[0]

        array_node = self._parse_array(node, arg)
        if array_node is None:
            # it's just a scalar - create a fake array for processing
            range_const = ast_internal_classes.Int_Literal_Node(value="0")
            array_node = ast_internal_classes.Array_Subscript_Node(
                name=arg,
                parent=arg.parent,
                type="VOID",
                indices=[
                    ast_internal_classes.ParDecl_Node(
                        type="RANGE", range=[range_const, range_const]
                    )
                ],
                sizes=[],
            )

        self.first_array = array_node
        self.cond = ast_internal_classes.BinOp_Node(
            op="==",
            rval=ast_internal_classes.Int_Literal_Node(value="1"),
            lval=self.first_array,
            line_number=node.line_number,
        )

    def _summarize_args(
        self,
        exec_node: ast_internal_classes.Execution_Part_Node,
        node: ast_internal_classes.FNode,
        new_func_body: List[ast_internal_classes.FNode],
    ):

        rangeslen_left = []
        par_Decl_Range_Finder(
            self.first_array,
            self.loop_ranges,
            rangeslen_left,
            self.count,
            new_func_body,
            self.scope_vars,
            self.ast.structures,
            True,
        )

        if self.second_array is None:
            return

        loop_ranges_right = []
        rangeslen_right = []
        par_Decl_Range_Finder(
            self.second_array,
            loop_ranges_right,
            rangeslen_right,
            self.count,
            new_func_body,
            self.scope_vars,
            self.ast.structures,
            True,
        )

        for left_len, right_len in zip(rangeslen_left, rangeslen_right):
            if left_len != right_len:
                raise TypeError("Can't support Fortran ANY with different array ranks!")

        # In this intrinsic, the left array dictates loop range.
        # Thus, we only need to adjust the second array
        self._adjust_array_ranges(
            node, self.second_array, self.loop_ranges, loop_ranges_right
        )

    def _initialize_result(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.BinOp_Node:

        init_value = self._result_init_value()

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.Int_Literal_Node(value=init_value),
            line_number=node.line_number,
        )

    def _generate_loop_body(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.If_Stmt_Node:
        """
        For any, we check if the condition is true and then set the value to true
        For all, we check if the condition is NOT true and then set the value to false
        """

        body_if = ast_internal_classes.Execution_Part_Node(
            execution=[
                self._result_loop_update(node),
                # TODO: we should make the `break` generation conditional based on the architecture
                # For parallel maps, we should have no breaks
                # For sequential loop, we want a break to be faster
                # ast_internal_classes.Break_Node(
                #    line_number=node.line_number
                # )
            ]
        )

        return ast_internal_classes.If_Stmt_Node(
            cond=self._loop_condition(),
            body=body_if,
            body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
            line_number=node.line_number,
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

    class Transformation(AnyAllCountTransformation):

        def _result_init_value(self):
            return "0"

        def _result_loop_update(self, node: ast_internal_classes.FNode):
            return ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="=",
                rval=ast_internal_classes.Int_Literal_Node(value="1"),
                line_number=node.line_number,
            )

        def _loop_condition(self):
            return self.cond

        @staticmethod
        def func_name() -> str:
            return "__dace_any"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        # Without DIM parameter, it only returns scalars
        return [], "LOGICAL"


class All(LoopBasedReplacement):
    """
    In this class, we implement the transformation for Fortran intrinsic ALL.
    The implementation is very similar to ANY.
    The main difference is that we initialize the partial result to 1,
    and set it to 0 if any of the evaluated conditions is false.
    """

    class Transformation(AnyAllCountTransformation):

        def _result_init_value(self):
            return "1"

        def _result_loop_update(self, node: ast_internal_classes.FNode):
            return ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="=",
                rval=ast_internal_classes.Int_Literal_Node(value="0"),
                line_number=node.line_number,
            )

        def _loop_condition(self):
            return ast_internal_classes.UnOp_Node(op="not", lval=self.cond)

        @staticmethod
        def func_name() -> str:
            return "__dace_all"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        # Without DIM parameter, it only returns scalars
        return [], "LOGICAL"


class Count(LoopBasedReplacement):
    """
    In this class, we implement the transformation for Fortran intrinsic COUNT.
    The implementation is very similar to ANY and ALL.
    The main difference is that we initialize the partial result to 0
    and increment it if any of the evaluated conditions is true.

    We do not support the KIND argument.
    """

    class Transformation(AnyAllCountTransformation):

        def _result_init_value(self):
            return "0"

        def _result_loop_update(self, node: ast_internal_classes.FNode):
            update = ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="+",
                rval=ast_internal_classes.Int_Literal_Node(value="1"),
                line_number=node.line_number,
            )
            return ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="=",
                rval=update,
                line_number=node.line_number,
            )

        def _loop_condition(self):
            return self.cond

        @staticmethod
        def func_name() -> str:
            return "__dace_count"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        # Without DIM parameter, it only returns scalars
        return [], "INTEGER"


class MinMaxValTransformation(LoopBasedReplacementTransformation):
    """
    Parent transformation class for implementations of two similar intrinsics: minval and maxval.

    For minval, we initialize the value to the largest possible for the type and check for a lower value in the loop.
    For maxval, we initialize the value to the lowest possible and check for a greater value in the loop.
    """

    def _initialize(self):
        self.rvals = []
        self.argument_variable = None

    def _update_result_type(self, var: ast_internal_classes.Name_Node):
        """
        For both MINVAL and MAXVAL, the result type depends on the input variable.
        """

        input_type = self.get_var_declaration(var.parent, self.argument_variable)

        var_decl = self.get_var_declaration(var.parent, var)
        var.type = input_type.type
        var_decl.type = input_type.type

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        for arg in node.args:

            array_node = self._parse_array(node, arg)

            if array_node is not None:
                self.rvals.append(array_node)
            else:
                raise NotImplementedError(
                    "We do not support non-array arguments for MINVAL/MAXVAL"
                )

    def _summarize_args(
        self,
        exec_node: ast_internal_classes.Execution_Part_Node,
        node: ast_internal_classes.FNode,
        new_func_body: List[ast_internal_classes.FNode],
    ):

        if len(self.rvals) != 1:
            raise NotImplementedError("Only one array can be summed")

        self.argument_variable = self.rvals[0]

        par_Decl_Range_Finder(
            self.argument_variable,
            self.loop_ranges,
            [],
            self.count,
            new_func_body,
            self.scope_vars,
            self.ast.structures,
            declaration=True,
        )

    def _initialize_result(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=self._result_init_value(self.argument_variable),
            line_number=node.line_number,
        )

    def _generate_loop_body(
        self, node: ast_internal_classes.FNode
    ) -> ast_internal_classes.If_Stmt_Node:

        cond = ast_internal_classes.BinOp_Node(
            lval=self.argument_variable,
            op=self._condition_op(),
            rval=node.lval,
            line_number=node.line_number,
        )
        body_if = ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=copy.deepcopy(self.argument_variable),
            line_number=node.line_number,
        )
        return ast_internal_classes.If_Stmt_Node(
            cond=cond,
            body=body_if,
            body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
            line_number=node.line_number,
        )


class MinVal(LoopBasedReplacement):
    """
    In this class, we implement the transformation for Fortran intrinsic MINVAL.

    We do not support the MASK and DIM argument.
    """

    class Transformation(MinMaxValTransformation):

        def _result_init_value(self, array: ast_internal_classes.Array_Subscript_Node):

            var_decl = self.get_var_declaration(array.parent, array)

            # TODO: this should be used as a call to HUGE
            fortran_type = var_decl.type
            dace_type = fortrantypes2dacetypes[fortran_type]
            from dace.dtypes import max_value

            max_val = max_value(dace_type)

            if fortran_type == "INTEGER":
                return ast_internal_classes.Int_Literal_Node(value=str(max_val))
            elif fortran_type == "DOUBLE":
                return ast_internal_classes.Real_Literal_Node(value=str(max_val))

        def _condition_op(self):
            return "<"

        @staticmethod
        def func_name() -> str:
            return "__dace_minval"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        # Without DIM parameter, it only returns scalars
        return [], first_arg.type


class MaxVal(LoopBasedReplacement):
    """
    In this class, we implement the transformation for Fortran intrinsic MAXVAL.

    We do not support the MASK and DIM argument.
    """

    class Transformation(MinMaxValTransformation):

        def _result_init_value(self, array: ast_internal_classes.Array_Subscript_Node):

            var_decl = self.get_var_declaration(array.parent, array)

            # TODO: this should be used as a call to HUGE
            fortran_type = var_decl.type
            dace_type = fortrantypes2dacetypes[fortran_type]
            from dace.dtypes import min_value

            min_val = min_value(dace_type)

            if fortran_type == "INTEGER":
                return ast_internal_classes.Int_Literal_Node(value=str(min_val))
            elif fortran_type == "DOUBLE":
                return ast_internal_classes.Real_Literal_Node(value=str(min_val))

        def _condition_op(self):
            return ">"

        @staticmethod
        def func_name() -> str:
            return "__dace_maxval"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        # Without DIM parameter, it only returns scalars
        return [], first_arg.type


class Merge(LoopBasedReplacement):
    """
    Implementation of the MERGE intrinsic.

    The implementation is complex because we need to support multiple cases:
    - Arguments are arrays
    - Arguments are scalars (glorified if condition)
    - Mask can be an array or a binary expression
    - All arrays - inputs and mask - can have arbitrary offsets and pardecls
    """

    class Transformation(LoopBasedReplacementTransformation):

        def _initialize(self):
            self.rvals = []

            self.first_array = None
            self.second_array = None
            self.mask_first_array = None
            self.mask_second_array = None
            self.mask_cond = None
            self.destination_array = None

        @staticmethod
        def func_name() -> str:
            return "__dace_merge"

        def _update_result_type(self, var: ast_internal_classes.Name_Node):

            if is_literal(self.first_array):
                input_type = self.first_array.type
            else:
                input_type = self.get_var_declaration(var.parent, self.first_array).type

            var_decl = self.get_var_declaration(var.parent, var)
            var.type = input_type
            var_decl.type = input_type

        def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

            if len(node.args) != 3:
                raise NotImplementedError("Expected three arguments to MERGE!")

            # First argument is always an array
            self.first_array = self._parse_array(node, node.args[0])

            # Second argument is always an array
            self.second_array = self._parse_array(node, node.args[1])

            # weird overload of MERGE - passing two scalars
            if self.first_array is None or self.second_array is None:
                self.uses_scalars = True
                self.first_array = node.args[0]
                self.second_array = node.args[1]
                self.mask_cond = node.args[2]

                return

            else:
                len_pardecls_first_array = 0
                len_pardecls_second_array = 0

                indices = None
                if isinstance(self.first_array, ast_internal_classes.Data_Ref_Node):
                    # it would be nice to return it directly from `parse_array`
                    # but this requires refactoring across the entire module
                    _, _, cur_val = self.ast.structures.find_definition(
                        self.scope_vars, self.first_array
                    )
                    indices = cur_val.part_ref.indices
                else:
                    indices = self.first_array.indices

                for ind in indices:
                    pardecls = [
                        i
                        for i in mywalk(ind)
                        if isinstance(i, ast_internal_classes.ParDecl_Node)
                    ]
                    len_pardecls_first_array += len(pardecls)

                first_array_indices_count = len(indices)

                indices = None
                if isinstance(self.second_array, ast_internal_classes.Data_Ref_Node):
                    # it would be nice to return it directly from `parse_array`
                    # but this requires refactoring across the entire module
                    _, _, cur_val = self.ast.structures.find_definition(
                        self.scope_vars, self.second_array
                    )
                    indices = cur_val.part_ref.indices
                else:
                    indices = self.second_array.indices

                for ind in indices:
                    pardecls = [
                        i
                        for i in mywalk(ind)
                        if isinstance(i, ast_internal_classes.ParDecl_Node)
                    ]
                    len_pardecls_second_array += len(pardecls)
                if len_pardecls_first_array != len_pardecls_second_array:
                    raise ValueError("MERGE arguments must have the same rank")
                if len_pardecls_first_array == 0:
                    self.uses_scalars = True
                else:
                    self.uses_scalars = False

            # Last argument is either an array or a binary op

            arg = node.args[2]
            if self.uses_scalars:
                self.mask_cond = arg
            else:

                array_node = self._parse_array(
                    node, node.args[2], dims_count=first_array_indices_count
                )
                if array_node is not None:

                    self.mask_first_array = array_node

                    self.mask_cond = ast_internal_classes.BinOp_Node(
                        op="==",
                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                        lval=self.mask_first_array,
                        line_number=node.line_number,
                    )
                else:
                    self.mask_cond = arg

        def _summarize_args(
            self,
            exec_node: ast_internal_classes.Execution_Part_Node,
            node: ast_internal_classes.FNode,
            new_func_body: List[ast_internal_classes.FNode],
        ):

            if self.uses_scalars:
                self.destination_array = node.lval
                return

            # The first main argument is an array -> this dictates loop boundaries
            # Other arrays, regardless if they appear as the second array or mask, need to have the same loop boundary.
            par_Decl_Range_Finder(
                self.first_array,
                self.loop_ranges,
                [],
                self.count,
                new_func_body,
                self.scope_vars,
                self.ast.structures,
                True,
                allow_scalars=True,
            )

            loop_ranges = []
            par_Decl_Range_Finder(
                self.second_array,
                loop_ranges,
                [],
                self.count,
                new_func_body,
                self.scope_vars,
                self.ast.structures,
                True,
                allow_scalars=True,
            )

            if isinstance(self.second_array, ast_internal_classes.Data_Ref_Node):
                _, _, cur_val = self.ast.structures.find_definition(
                    self.scope_vars, self.second_array
                )
                self._adjust_array_ranges(
                    node, cur_val.part_ref, self.loop_ranges, loop_ranges
                )
            else:
                self._adjust_array_ranges(
                    node, self.second_array, self.loop_ranges, loop_ranges
                )

            # parse destination

            if not isinstance(node.lval, ast_internal_classes.Name_Node):
                raise TypeError(
                    f"Expected a variable name as the destination for MERGE, but got {type(node.lval)}"
                )

            array_decl = self.get_var_declaration(exec_node.parent, node.lval)
            if array_decl.sizes is None or len(array_decl.sizes) == 0:

                # for destination array, sizes might be unknown when we use arg extractor
                # in that situation, we look at the size of the first argument
                dims = len(self.first_array.indices)
            else:
                dims = len(array_decl.sizes)

            # type inference! this is necessary when the destination array is
            # not known exactly, e.g., in recursive calls.
            if array_decl.sizes is None or len(array_decl.sizes) == 0:

                first_input = self.get_var_declaration(node.parent, node.rval.args[0])
                array_decl.sizes = copy.deepcopy(first_input.sizes)
                array_decl.offsets = [1] * len(array_decl.sizes)
                array_decl.type = first_input.type

                node.lval.sizes = array_decl.sizes

            if len(node.lval.sizes) > 0:
                self.destination_array = ast_internal_classes.Array_Subscript_Node(
                    name=node.lval,
                    parent=node.lval.parent,
                    type="VOID",
                    indices=[ast_internal_classes.ParDecl_Node(type="ALL")] * dims,
                )
                par_Decl_Range_Finder(
                    self.destination_array,
                    [],
                    [],
                    self.count,
                    new_func_body,
                    self.scope_vars,
                    self.ast.structures,
                    True,
                )
            else:
                self.destination_array = node.lval

            if self.mask_first_array is not None:
                loop_ranges = []
                par_Decl_Range_Finder(
                    self.mask_first_array,
                    loop_ranges,
                    [],
                    self.count,
                    new_func_body,
                    self.scope_vars,
                    self.ast.structures,
                    True,
                    allow_scalars=True,
                )
                self._adjust_array_ranges(
                    node, self.mask_first_array, self.loop_ranges, loop_ranges
                )

            if self.mask_second_array is not None:
                loop_ranges = []
                par_Decl_Range_Finder(
                    self.mask_second_array,
                    loop_ranges,
                    [],
                    self.count,
                    new_func_body,
                    self.scope_vars,
                    self.ast.structures,
                    True,
                    allow_scalars=True,
                )
                self._adjust_array_ranges(
                    node, self.mask_second_array, self.loop_ranges, loop_ranges
                )

        def _initialize_result(
            self, node: ast_internal_classes.FNode
        ) -> Optional[ast_internal_classes.BinOp_Node]:
            """
            We don't use result variable in MERGE.
            """
            return None

        def _generate_loop_body(
            self, node: ast_internal_classes.FNode
        ) -> ast_internal_classes.If_Stmt_Node:
            """
            We check if the condition is true. If yes, then we write from the first array.
            Otherwise, we copy data from the second array.
            """

            copy_first = ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(self.destination_array),
                op="=",
                rval=self.first_array,
                line_number=node.line_number,
            )

            copy_second = ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(self.destination_array),
                op="=",
                rval=self.second_array,
                line_number=node.line_number,
            )

            body_if = ast_internal_classes.Execution_Part_Node(execution=[copy_first])

            body_else = ast_internal_classes.Execution_Part_Node(
                execution=[copy_second]
            )

            # for scalar operations, we need to extract first element if it's an array
            if self.uses_scalars and isinstance(
                self.mask_cond, ast_internal_classes.Name_Node
            ):
                definition = self.scope_vars.get_var(node.parent, self.mask_cond.name)

                if definition.sizes is not None and len(definition.sizes) > 0:
                    self.mask_cond = ast_internal_classes.Array_Subscript_Node(
                        name=self.mask_cond,
                        type=self.mask_cond.type,
                        indices=[ast_internal_classes.Int_Literal_Node(value="1")]
                        * len(definition.sizes),
                    )

            return ast_internal_classes.If_Stmt_Node(
                cond=self.mask_cond,
                body=body_if,
                body_else=body_else,
                line_number=node.line_number,
            )

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == "VOID":
            return None

        return first_arg.sizes, first_arg.type
