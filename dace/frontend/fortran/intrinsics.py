from abc import abstractmethod
import copy
import math
from collections import namedtuple
from typing import Any, List, Optional, Set, Tuple, Type

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_utils import fortrantypes2dacetypes
from dace.frontend.fortran.ast_transforms import NodeVisitor, NodeTransformer, ParentScopeAssigner, ScopeVarsDeclarations, par_Decl_Range_Finder, mywalk

FASTNode = Any


class IntrinsicTransformation:

    @staticmethod
    @abstractmethod
    def replaced_name(func_name: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node,
                line) -> ast_internal_classes.FNode:
        pass

    @staticmethod
    def has_transformation() -> bool:
        return False


class IntrinsicNodeTransformer(NodeTransformer):

    def initialize(self, ast):
        # We need to rerun the assignment because transformations could have created
        # new AST nodes
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations()
        self.scope_vars.visit(ast)

    @staticmethod
    @abstractmethod
    def func_name(self) -> str:
        pass


class DirectReplacement(IntrinsicTransformation):

    Replacement = namedtuple("Replacement", "function")
    Transformation = namedtuple("Transformation", "function")

    class ASTTransformation(IntrinsicNodeTransformer):

        def visit_BinOp_Node(self, binop_node: ast_internal_classes.BinOp_Node):

            if not isinstance(binop_node.rval, ast_internal_classes.Call_Expr_Node):
                return binop_node

            node = binop_node.rval

            name = node.name.name.split('__dace_')
            if len(name) != 2 or name[1] not in DirectReplacement.FUNCTIONS:
                return binop_node
            func_name = name[1]

            replacement_rule = DirectReplacement.FUNCTIONS[func_name]
            if isinstance(replacement_rule, DirectReplacement.Transformation):

                # FIXME: we do not have line number in binop?
                binop_node.rval, input_type = replacement_rule.function(node, self.scope_vars, 0)  #binop_node.line)
                print(binop_node, binop_node.lval, binop_node.rval)

                # replace types of return variable - LHS of the binary operator
                var = binop_node.lval
                if isinstance(var.name, ast_internal_classes.Name_Node):
                    name = var.name.name
                else:
                    name = var.name
                var_decl = self.scope_vars.get_var(var.parent, name)
                var.type = input_type
                var_decl.type = input_type

            return binop_node

            #self.scope_vars.get_var(node.parent, arg.name).

    def replace_size(var: ast_internal_classes.Call_Expr_Node, scope_vars: ScopeVarsDeclarations, line):

        if len(var.args) not in [1, 2]:
            raise RuntimeError()

        # get variable declaration for the first argument
        var_decl = scope_vars.get_var(var.parent, var.args[0].name)

        # one arg to SIZE? compute the total number of elements
        if len(var.args) == 1:

            if len(var_decl.sizes) == 1:
                return (var_decl.sizes[0], "INTEGER")

            ret = ast_internal_classes.BinOp_Node(lval=var_decl.sizes[0], rval=None, op="*")
            cur_node = ret
            for i in range(1, len(var_decl.sizes) - 1):

                cur_node.rval = ast_internal_classes.BinOp_Node(lval=var_decl.sizes[i], rval=None, op="*")
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

    def replace_bit_size(var: ast_internal_classes.Call_Expr_Node, scope_vars: ScopeVarsDeclarations, line):

        if len(var.args) != 1:
            raise RuntimeError()

        # get variable declaration for the first argument
        var_decl = scope_vars.get_var(var.parent, var.args[0].name)

        dace_type = fortrantypes2dacetypes[var_decl.type]
        type_size = dace_type().itemsize * 8

        return (ast_internal_classes.Int_Literal_Node(value=str(type_size)), "INTEGER")

    def replace_int_kind(args: ast_internal_classes.Arg_List_Node, line):
        return ast_internal_classes.Int_Literal_Node(value=str(
            math.ceil((math.log2(math.pow(10, int(args.args[0].value))) + 1) / 8)),
                                                     line_number=line)

    def replace_real_kind(args: ast_internal_classes.Arg_List_Node, line):
        if int(args.args[0].value) >= 9 or int(args.args[1].value) > 126:
            return ast_internal_classes.Int_Literal_Node(value="8", line_number=line)
        elif int(args.args[0].value) >= 3 or int(args.args[1].value) > 14:
            return ast_internal_classes.Int_Literal_Node(value="4", line_number=line)
        else:
            return ast_internal_classes.Int_Literal_Node(value="2", line_number=line)

    FUNCTIONS = {
        "SELECTED_INT_KIND": Replacement(replace_int_kind),
        "SELECTED_REAL_KIND": Replacement(replace_real_kind),
        "BIT_SIZE": Transformation(replace_bit_size),
        "SIZE": Transformation(replace_size)
    }

    @staticmethod
    def temporary_functions():

        # temporary functions created by us -> f becomes __dace_f
        # We provide this to tell Fortran parser that these are function calls,
        # not array accesses
        funcs = list(DirectReplacement.FUNCTIONS.keys())
        return [f'__dace_{f}' for f in funcs]

    @staticmethod
    def replacable_name(func_name: str) -> bool:
        return func_name in DirectReplacement.FUNCTIONS

    @staticmethod
    def replace_name(func_name: str) -> str:
        #return ast_internal_classes.Name_Node(name=DirectReplacement.FUNCTIONS[func_name][0])
        return ast_internal_classes.Name_Node(name=f'__dace_{func_name}')

    @staticmethod
    def replacable(func_name: str) -> bool:
        orig_name = func_name.split('__dace_')
        if len(orig_name) > 1 and orig_name[1] in DirectReplacement.FUNCTIONS:
            return isinstance(DirectReplacement.FUNCTIONS[orig_name[1]], DirectReplacement.Replacement)
        return False

    @staticmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node,
                line) -> ast_internal_classes.FNode:

        # Here we already have __dace_func
        fname = func_name.split('__dace_')[1]
        return DirectReplacement.FUNCTIONS[fname].function(args, line)

    def has_transformation(fname: str) -> bool:
        return isinstance(DirectReplacement.FUNCTIONS[fname], DirectReplacement.Transformation)

    @staticmethod
    def get_transformation() -> IntrinsicNodeTransformer:
        return DirectReplacement.ASTTransformation()


class LoopBasedReplacement:

    INTRINSIC_TO_DACE = {
        "SUM": "__dace_sum",
        "PRODUCT": "__dace_product",
        "ANY": "__dace_any",
        "ALL": "__dace_all",
        "COUNT": "__dace_count",
        "MINVAL": "__dace_minval",
        "MAXVAL": "__dace_maxval",
        "MERGE": "__dace_merge"
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
    def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                        new_func_body: List[ast_internal_classes.FNode]):
        pass

    @abstractmethod
    def _initialize_result(self, node: ast_internal_classes.FNode) -> Optional[ast_internal_classes.BinOp_Node]:
        pass

    @abstractmethod
    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:
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

    def _parse_array(self, node: ast_internal_classes.Execution_Part_Node,
                     arg: ast_internal_classes.FNode) -> ast_internal_classes.Array_Subscript_Node:

        # supports syntax func(arr)
        if isinstance(arg, ast_internal_classes.Name_Node):
            array_node = ast_internal_classes.Array_Subscript_Node(parent=arg.parent)
            array_node.name = arg

            # If we access SUM(arr) where arr has many dimensions,
            # We need to create a ParDecl_Node for each dimension
            dims = len(self.scope_vars.get_var(node.parent, arg.name).sizes)
            array_node.indices = [ast_internal_classes.ParDecl_Node(type='ALL')] * dims

            return array_node

        # supports syntax func(arr(:))
        if isinstance(arg, ast_internal_classes.Array_Subscript_Node):
            return arg

    def _parse_binary_op(
        self, node: ast_internal_classes.Call_Expr_Node, arg: ast_internal_classes.BinOp_Node
    ) -> Tuple[ast_internal_classes.Array_Subscript_Node, Optional[ast_internal_classes.Array_Subscript_Node],
               ast_internal_classes.BinOp_Node]:
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
            return False

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
                raise TypeError("Can't parse Fortran binary op with different array ranks!")

        # Now, we need to convert the array to a proper subscript node
        cond = copy.deepcopy(arg)
        cond.lval = first_array
        cond.rval = second_array

        return (first_array, second_array, cond)

    def _adjust_array_ranges(self, node: ast_internal_classes.FNode, array: ast_internal_classes.Array_Subscript_Node,
                             loop_ranges_main: list, loop_ranges_array: list):
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

            difference = int(end_loop.value) - int(start_loop.value)
            if difference != 0:
                new_index = ast_internal_classes.BinOp_Node(
                    lval=idx_var,
                    op="+",
                    rval=ast_internal_classes.Int_Literal_Node(value=str(difference)),
                    line_number=node.line_number)
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
                if isinstance(i, ast_internal_classes.Call_Expr_Node) and i.name.name == self.func_name():
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


class SumProduct(LoopBasedReplacementTransformation):

    def _initialize(self):
        self.rvals = []
        self.argument_variable = None

    def _update_result_type(self, var: ast_internal_classes.Name_Node):
        """
            For both SUM and PRODUCT, the result type depends on the input variable.
        """
        input_type = self.scope_vars.get_var(var.parent, self.argument_variable.name.name)

        var_decl = self.scope_vars.get_var(var.parent, var.name)
        var.type = input_type.type
        var_decl.type = input_type.type

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        for arg in node.args:

            array_node = self._parse_array(node, arg)

            if array_node is not None:
                self.rvals.append(array_node)
            else:
                raise NotImplementedError("We do not support non-array arguments for SUM/PRODUCT")

    def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                        new_func_body: List[ast_internal_classes.FNode]):

        if len(self.rvals) != 1:
            raise NotImplementedError("Only one array can be summed")

        self.argument_variable = self.rvals[0]

        par_Decl_Range_Finder(self.argument_variable, self.loop_ranges, [], [], self.count, new_func_body,
                              self.scope_vars, True)

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.Int_Literal_Node(value=self._result_init_value()),
            line_number=node.line_number)

    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(lval=node.lval,
                                               op="=",
                                               rval=ast_internal_classes.BinOp_Node(lval=node.lval,
                                                                                    op=self._result_update_op(),
                                                                                    rval=self.argument_variable,
                                                                                    line_number=node.line_number),
                                               line_number=node.line_number)


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


class AnyAllCountTransformation(LoopBasedReplacementTransformation):

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
        var_decl = self.scope_vars.get_var(var.parent, var.name)
        var.type = "INTEGER"
        var_decl.type = "INTEGER"

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        if len(node.args) > 1:
            raise NotImplementedError("Fortran ANY with the DIM parameter is not supported!")
        arg = node.args[0]

        array_node = self._parse_array(node, arg)
        if array_node is not None:
            self.first_array = array_node
            self.cond = ast_internal_classes.BinOp_Node(op="==",
                                                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                        lval=self.first_array,
                                                        line_number=node.line_number)
        else:
            self.first_array, self.second_array, self.cond = self._parse_binary_op(node, arg)

    def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                        new_func_body: List[ast_internal_classes.FNode]):

        rangeslen_left = []
        par_Decl_Range_Finder(self.first_array, self.loop_ranges, [], rangeslen_left, self.count, new_func_body,
                              self.scope_vars, True)
        if self.second_array is None:
            return

        loop_ranges_right = []
        rangeslen_right = []
        par_Decl_Range_Finder(self.second_array, loop_ranges_right, [], rangeslen_right, self.count, new_func_body,
                              self.scope_vars, True)

        for left_len, right_len in zip(rangeslen_left, rangeslen_right):
            if left_len != right_len:
                raise TypeError("Can't support Fortran ANY with different array ranks!")

        # In this intrinsic, the left array dictates loop range.
        # Thus, we only need to adjust the second array
        self._adjust_array_ranges(node, self.second_array, self.loop_ranges, loop_ranges_right)

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        init_value = self._result_init_value()

        return ast_internal_classes.BinOp_Node(lval=node.lval,
                                               op="=",
                                               rval=ast_internal_classes.Int_Literal_Node(value=init_value),
                                               line_number=node.line_number)

    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:
        """
        For any, we check if the condition is true and then set the value to true
        For all, we check if the condition is NOT true and then set the value to false
        """

        body_if = ast_internal_classes.Execution_Part_Node(execution=[
            self._result_loop_update(node),
            # TODO: we should make the `break` generation conditional based on the architecture
            # For parallel maps, we should have no breaks
            # For sequential loop, we want a break to be faster
            #ast_internal_classes.Break_Node(
            #    line_number=node.line_number
            #)
        ])

        return ast_internal_classes.If_Stmt_Node(cond=self._loop_condition(),
                                                 body=body_if,
                                                 body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
                                                 line_number=node.line_number)


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

            return ast_internal_classes.BinOp_Node(lval=copy.deepcopy(node.lval),
                                                   op="=",
                                                   rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                   line_number=node.line_number)

        def _loop_condition(self):
            return self.cond

        @staticmethod
        def func_name() -> str:
            return "__dace_any"


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

            return ast_internal_classes.BinOp_Node(lval=copy.deepcopy(node.lval),
                                                   op="=",
                                                   rval=ast_internal_classes.Int_Literal_Node(value="0"),
                                                   line_number=node.line_number)

        def _loop_condition(self):
            return ast_internal_classes.UnOp_Node(op="not", lval=self.cond)

        @staticmethod
        def func_name() -> str:
            return "__dace_all"


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

            update = ast_internal_classes.BinOp_Node(lval=copy.deepcopy(node.lval),
                                                     op="+",
                                                     rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                     line_number=node.line_number)
            return ast_internal_classes.BinOp_Node(lval=copy.deepcopy(node.lval),
                                                   op="=",
                                                   rval=update,
                                                   line_number=node.line_number)

        def _loop_condition(self):
            return self.cond

        @staticmethod
        def func_name() -> str:
            return "__dace_count"


class MinMaxValTransformation(LoopBasedReplacementTransformation):

    def _initialize(self):
        self.rvals = []
        self.argument_variable = None

    def _update_result_type(self, var: ast_internal_classes.Name_Node):
        """
            For both MINVAL and MAXVAL, the result type depends on the input variable.
        """

        input_type = self.scope_vars.get_var(var.parent, self.argument_variable.name.name)

        var_decl = self.scope_vars.get_var(var.parent, var.name)
        var.type = input_type.type
        var_decl.type = input_type.type

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        for arg in node.args:

            array_node = self._parse_array(node, arg)

            if array_node is not None:
                self.rvals.append(array_node)
            else:
                raise NotImplementedError("We do not support non-array arguments for MINVAL/MAXVAL")

    def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                        new_func_body: List[ast_internal_classes.FNode]):

        if len(self.rvals) != 1:
            raise NotImplementedError("Only one array can be summed")

        self.argument_variable = self.rvals[0]

        par_Decl_Range_Finder(self.argument_variable, self.loop_ranges, [], [], self.count, new_func_body,
                              self.scope_vars, True)

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(lval=node.lval,
                                               op="=",
                                               rval=self._result_init_value(self.argument_variable),
                                               line_number=node.line_number)

    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        cond = ast_internal_classes.BinOp_Node(lval=self.argument_variable,
                                               op=self._condition_op(),
                                               rval=node.lval,
                                               line_number=node.line_number)
        body_if = ast_internal_classes.BinOp_Node(lval=node.lval,
                                                  op="=",
                                                  rval=self.argument_variable,
                                                  line_number=node.line_number)
        return ast_internal_classes.If_Stmt_Node(cond=cond,
                                                 body=body_if,
                                                 body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
                                                 line_number=node.line_number)


class MinVal(LoopBasedReplacement):
    """
        In this class, we implement the transformation for Fortran intrinsic MINVAL.

        We do not support the MASK and DIM argument.
    """

    class Transformation(MinMaxValTransformation):

        def _result_init_value(self, array: ast_internal_classes.Array_Subscript_Node):

            var_decl = self.scope_vars.get_var(array.parent, array.name.name)

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


class MaxVal(LoopBasedReplacement):
    """
        In this class, we implement the transformation for Fortran intrinsic MAXVAL.

        We do not support the MASK and DIM argument.
    """

    class Transformation(MinMaxValTransformation):

        def _result_init_value(self, array: ast_internal_classes.Array_Subscript_Node):

            var_decl = self.scope_vars.get_var(array.parent, array.name.name)

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


class Merge(LoopBasedReplacement):

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
            """
                We can ignore the result type, because we exempted this
                transformation from generating a result.
                In MERGE, we write directly to the destination array.
                Thus, we store this result array for future use.
            """
            pass

        def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

            if len(node.args) != 3:
                raise NotImplementedError("Expected three arguments to MERGE!")

            # First argument is always an array
            self.first_array = self._parse_array(node, node.args[0])
            assert self.first_array is not None

            # Second argument is always an array
            self.second_array = self._parse_array(node, node.args[1])
            assert self.second_array is not None

            # Last argument is either an array or a binary op
            arg = node.args[2]
            array_node = self._parse_array(node, node.args[2])
            if array_node is not None:

                self.mask_first_array = array_node
                self.mask_cond = ast_internal_classes.BinOp_Node(op="==",
                                                                 rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                                 lval=self.mask_first_array,
                                                                 line_number=node.line_number)

            else:

                self.mask_first_array, self.mask_second_array, self.mask_cond = self._parse_binary_op(node, arg)

        def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                            new_func_body: List[ast_internal_classes.FNode]):

            self.destination_array = self._parse_array(exec_node, node.lval)

            # The first main argument is an array -> this dictates loop boundaries
            # Other arrays, regardless if they appear as the second array or mask, need to have the same loop boundary.
            par_Decl_Range_Finder(self.first_array, self.loop_ranges, [], [], self.count, new_func_body,
                                  self.scope_vars, True)

            loop_ranges = []
            par_Decl_Range_Finder(self.second_array, loop_ranges, [], [], self.count, new_func_body, self.scope_vars,
                                  True)
            self._adjust_array_ranges(node, self.second_array, self.loop_ranges, loop_ranges)

            par_Decl_Range_Finder(self.destination_array, [], [], [], self.count, new_func_body, self.scope_vars, True)

            if self.mask_first_array is not None:
                loop_ranges = []
                par_Decl_Range_Finder(self.mask_first_array, loop_ranges, [], [], self.count, new_func_body,
                                      self.scope_vars, True)
                self._adjust_array_ranges(node, self.mask_first_array, self.loop_ranges, loop_ranges)

            if self.mask_second_array is not None:
                loop_ranges = []
                par_Decl_Range_Finder(self.mask_second_array, loop_ranges, [], [], self.count, new_func_body,
                                      self.scope_vars, True)
                self._adjust_array_ranges(node, self.mask_second_array, self.loop_ranges, loop_ranges)

        def _initialize_result(self, node: ast_internal_classes.FNode) -> Optional[ast_internal_classes.BinOp_Node]:
            """
                We don't use result variable in MERGE.
            """
            return None

        def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:
            """
                We check if the condition is true. If yes, then we write from the first array.
                Otherwise, we copy data from the second array.
            """

            copy_first = ast_internal_classes.BinOp_Node(lval=copy.deepcopy(self.destination_array),
                                                         op="=",
                                                         rval=self.first_array,
                                                         line_number=node.line_number)

            copy_second = ast_internal_classes.BinOp_Node(lval=copy.deepcopy(self.destination_array),
                                                          op="=",
                                                          rval=self.second_array,
                                                          line_number=node.line_number)

            body_if = ast_internal_classes.Execution_Part_Node(execution=[copy_first])

            body_else = ast_internal_classes.Execution_Part_Node(execution=[copy_second])

            return ast_internal_classes.If_Stmt_Node(cond=self.mask_cond,
                                                     body=body_if,
                                                     body_else=body_else,
                                                     line_number=node.line_number)


class MathFunctions(IntrinsicTransformation):

    MathTransformation = namedtuple("MathTransformation", "function return_type")
    MathReplacement = namedtuple("MathReplacement", "function replacement_function return_type")

    def generate_scale(arg: ast_internal_classes.Call_Expr_Node):

        # SCALE(X, I) becomes: X * pow(RADIX(X), I)
        # In our case, RADIX(X) is always 2
        line = arg.line_number
        x = arg.args[0]
        i = arg.args[1]
        const_two = ast_internal_classes.Int_Literal_Node(value="2")

        # I and RADIX(X) are both integers
        rval = ast_internal_classes.Call_Expr_Node(name=ast_internal_classes.Name_Node(name="pow"),
                                                   type="INTEGER",
                                                   args=[const_two, i],
                                                   line_number=line)

        mult = ast_internal_classes.BinOp_Node(op="*", lval=x, rval=rval, line_number=line)

        # pack it into parentheses, just to be sure
        return ast_internal_classes.Parenthesis_Expr_Node(expr=mult)

    def generate_aint(arg: ast_internal_classes.Call_Expr_Node):

        # The call to AINT can contain a second KIND parameter
        # We ignore it a the moment.
        # However, to map into C's trunc, we need to drop it.
        if len(arg.args) > 1:
            del arg.args[1]

        fname = arg.name.name.split('__dace_')[1]
        if fname in "AINT":
            arg.name = ast_internal_classes.Name_Node(name="trunc")
        elif fname == "NINT":
            arg.name = ast_internal_classes.Name_Node(name="iround")
        elif fname == "ANINT":
            arg.name = ast_internal_classes.Name_Node(name="round")
        else:
            raise NotImplementedError()

        return arg

    INTRINSIC_TO_DACE = {
        "MIN": MathTransformation("min", "FIRST_ARG"),
        "MAX": MathTransformation("max", "FIRST_ARG"),
        "SQRT": MathTransformation("sqrt", "FIRST_ARG"),
        "ABS": MathTransformation("abs", "FIRST_ARG"),
        "EXP": MathTransformation("exp", "FIRST_ARG"),
        # Documentation states that the return type of LOG is always REAL,
        # but the kind is the same as of the first argument.
        # However, we already replaced kind with types used in DaCe.
        # Thus, a REAL that is really DOUBLE will be double in the first argument.
        "LOG": MathTransformation("log", "FIRST_ARG"),
        "MOD": {
            "INTEGER": MathTransformation("Mod", "INTEGER"),
            "REAL": MathTransformation("Mod_float", "REAL"),
            "DOUBLE": MathTransformation("Mod_float", "DOUBLE")
        },
        "MODULO": {
            "INTEGER": MathTransformation("Modulo", "INTEGER"),
            "REAL": MathTransformation("Modulo_float", "REAL"),
            "DOUBLE": MathTransformation("Modulo_float", "DOUBLE")
        },
        "FLOOR": {
            "REAL": MathTransformation("floor", "INTEGER"),
            "DOUBLE": MathTransformation("floor", "INTEGER")
        },
        "SCALE": MathReplacement(None, generate_scale, "FIRST_ARG"),
        "EXPONENT": MathTransformation("frexp", "INTEGER"),
        "INT": MathTransformation("int", "INTEGER"),
        "AINT": MathReplacement("trunc", generate_aint, "FIRST_ARG"),
        "NINT": MathReplacement("iround", generate_aint, "INTEGER"),
        "ANINT": MathReplacement("round", generate_aint, "FIRST_ARG"),
        "REAL": MathTransformation("float", "REAL"),
        "DBLE": MathTransformation("double", "DOUBLE"),
        "SIN": MathTransformation("sin", "FIRST_ARG"),
        "COS": MathTransformation("cos", "FIRST_ARG"),
        "SINH": MathTransformation("sinh", "FIRST_ARG"),
        "COSH": MathTransformation("cosh", "FIRST_ARG"),
        "TANH": MathTransformation("tanh", "FIRST_ARG"),
        "ASIN": MathTransformation("asin", "FIRST_ARG"),
        "ACOS": MathTransformation("acos", "FIRST_ARG"),
        "ATAN": MathTransformation("atan", "FIRST_ARG"),
        "ATAN2": MathTransformation("atan2", "FIRST_ARG")
    }

    class TypeTransformer(IntrinsicNodeTransformer):

        def func_type(self, node: ast_internal_classes.Call_Expr_Node):

            # take the first arg
            arg = node.args[0]
            if isinstance(arg, ast_internal_classes.Real_Literal_Node):
                return 'REAL'
            elif isinstance(arg, ast_internal_classes.Int_Literal_Node):
                return 'INTEGER'
            elif isinstance(arg, ast_internal_classes.Call_Expr_Node):
                return arg.type
            elif isinstance(arg, ast_internal_classes.Name_Node):
                input_type = self.scope_vars.get_var(node.parent, arg.name)
                return input_type.type
            else:
                input_type = self.scope_vars.get_var(node.parent, arg.name.name)
                return input_type.type

        def replace_call(self, old_call: ast_internal_classes.Call_Expr_Node, new_call: ast_internal_classes.FNode):

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

            name = node.name.name.split('__dace_')
            if len(name) != 2 or name[1] not in MathFunctions.INTRINSIC_TO_DACE:
                return binop_node
            func_name = name[1]

            # Visit all children before we expand this call.
            # We need that to properly get the type.
            for arg in node.args:
                self.visit(arg)

            return_type = None
            input_type = None
            input_type = self.func_type(node)

            replacement_rule = MathFunctions.INTRINSIC_TO_DACE[func_name]
            if isinstance(replacement_rule, dict):
                replacement_rule = replacement_rule[input_type]
            if replacement_rule.return_type == "FIRST_ARG":
                return_type = input_type
            else:
                return_type = replacement_rule.return_type

            if isinstance(replacement_rule, MathFunctions.MathTransformation):
                node.name = ast_internal_classes.Name_Node(name=replacement_rule.function)
                node.type = return_type

            else:
                binop_node.rval = replacement_rule.replacement_function(node)

            # replace types of return variable - LHS of the binary operator
            var = binop_node.lval
            name = None
            if isinstance(var.name, ast_internal_classes.Name_Node):
                name = var.name.name
            else:
                name = var.name
            var_decl = self.scope_vars.get_var(var.parent, name)
            var.type = input_type
            var_decl.type = input_type

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
        return [f'__dace_{f}' for f in funcs]

    @staticmethod
    def replacable(func_name: str) -> bool:
        return func_name in MathFunctions.INTRINSIC_TO_DACE

    @staticmethod
    def replace(func_name: str) -> ast_internal_classes.FNode:
        return ast_internal_classes.Name_Node(name=f'__dace_{func_name}')

    def has_transformation() -> bool:
        return True

    @staticmethod
    def get_transformation() -> TypeTransformer:
        return MathFunctions.TypeTransformer()


class FortranIntrinsics:

    IMPLEMENTATIONS_AST = {
        "SUM": Sum,
        "PRODUCT": Product,
        "ANY": Any,
        "COUNT": Count,
        "ALL": All,
        "MINVAL": MinVal,
        "MAXVAL": MaxVal,
        "MERGE": Merge
    }

    EXEMPTED_FROM_CALL_EXTRACTION = [Merge]

    def __init__(self):
        self._transformations_to_run = set()

    def transformations(self) -> Set[Type[NodeTransformer]]:
        return self._transformations_to_run

    @staticmethod
    def function_names() -> List[str]:
        # list of all functions that are created by initial transformation, before doing full replacement
        # this prevents other parser components from replacing our function calls with array subscription nodes
        return [
            *list(LoopBasedReplacement.INTRINSIC_TO_DACE.values()), *MathFunctions.temporary_functions(),
            *DirectReplacement.temporary_functions()
        ]

    @staticmethod
    def retained_function_names() -> List[str]:
        # list of all DaCe functions that we use after full parsing
        return MathFunctions.dace_functions()

    @staticmethod
    def call_extraction_exemptions() -> List[str]:
        return [
            *[func.Transformation.func_name() for func in FortranIntrinsics.EXEMPTED_FROM_CALL_EXTRACTION]
            #*MathFunctions.temporary_functions()
        ]

    def replace_function_name(self, node: FASTNode) -> ast_internal_classes.Name_Node:

        func_name = node.string
        replacements = {
            "SIGN": "__dace_sign",
        }
        if func_name in replacements:
            return ast_internal_classes.Name_Node(name=replacements[func_name])
        elif DirectReplacement.replacable_name(func_name):
            if DirectReplacement.has_transformation(func_name):
                self._transformations_to_run.add(DirectReplacement.get_transformation())
            return DirectReplacement.replace_name(func_name)
        elif MathFunctions.replacable(func_name):
            self._transformations_to_run.add(MathFunctions.get_transformation())
            return MathFunctions.replace(func_name)

        if self.IMPLEMENTATIONS_AST[func_name].has_transformation():

            if hasattr(self.IMPLEMENTATIONS_AST[func_name], "Transformation"):
                self._transformations_to_run.add(self.IMPLEMENTATIONS_AST[func_name].Transformation())
            else:
                self._transformations_to_run.add(self.IMPLEMENTATIONS_AST[func_name].get_transformation(func_name))

        return ast_internal_classes.Name_Node(name=self.IMPLEMENTATIONS_AST[func_name].replaced_name(func_name))

    def replace_function_reference(self, name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node,
                                   line):

        func_types = {
            "__dace_sign": "DOUBLE",
        }
        if name.name in func_types:
            # FIXME: this will be progressively removed
            call_type = func_types[name.name]
            return ast_internal_classes.Call_Expr_Node(name=name, type=call_type, args=args.args, line_number=line)
        elif DirectReplacement.replacable(name.name):
            return DirectReplacement.replace(name.name, args, line)
        else:
            # We will do the actual type replacement later
            # To that end, we need to know the input types - but these we do not know at the moment.
            return ast_internal_classes.Call_Expr_Node(name=name, type="VOID", args=args.args, line_number=line)
