import copy
import math
import sys
from abc import abstractmethod
from collections import namedtuple
from typing import Any, List, Optional, Tuple, Union

from numpy import array_repr

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_transforms import NodeVisitor, NodeTransformer, ParentScopeAssigner, \
    ScopeVarsDeclarations, TypeInference, par_Decl_Range_Finder, NeedsTypeInferenceException
from dace.frontend.fortran.ast_utils import fortrantypes2dacetypes, mywalk
from dace.libraries.blas.nodes.dot import dot_libnode
from dace.libraries.blas.nodes.gemm import gemm_libnode
from dace.libraries.standard.nodes import Transpose
from dace.sdfg import SDFGState, SDFG, nodes
from dace.sdfg.graph import OrderedDiGraph
from dace.transformation import transformation as xf

FASTNode = Any

def is_literal(node: ast_internal_classes.FNode) -> bool:
    return isinstance(node, (ast_internal_classes.Int_Literal_Node, ast_internal_classes.Double_Literal_Node, ast_internal_classes.Real_Literal_Node, ast_internal_classes.Bool_Literal_Node))

class IntrinsicTransformation:

    @staticmethod
    @abstractmethod
    def replaced_name(func_name: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def replace(func_name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node, line,
                symbols: list) -> ast_internal_classes.FNode:
        pass

    @staticmethod
    def has_transformation() -> bool:
        return False

class VariableProcessor:

    def __init__(self, scope_vars, ast):
        self.scope_vars = scope_vars
        self.ast = ast

    def get_var(
        self,
        parent: ast_internal_classes.FNode,
        variable: Union[
            ast_internal_classes.Data_Ref_Node, ast_internal_classes.Name_Node,
            ast_internal_classes.Array_Subscript_Node
        ]
    ):

        if isinstance(variable, ast_internal_classes.Data_Ref_Node):

            _, var_decl, cur_val = self.ast.structures.find_definition(self.scope_vars, variable)
            return var_decl, cur_val

        assert isinstance(variable, (ast_internal_classes.Name_Node, ast_internal_classes.Array_Subscript_Node))
        if isinstance(variable, ast_internal_classes.Name_Node):
            name = variable.name
        elif isinstance(variable, ast_internal_classes.Array_Subscript_Node):
            name = variable.name.name

        if self.scope_vars.contains_var(parent, name):
            return self.scope_vars.get_var(parent, name), variable
        elif name in self.ast.module_declarations:
            return self.ast.module_declarations[name], variable
        else:
            raise RuntimeError(f"Couldn't find the declaration of variable {name} in function {parent.name.name}!")

    def get_var_declaration(
        self,
        parent: ast_internal_classes.FNode,
        variable: Union[
            ast_internal_classes.Data_Ref_Node, ast_internal_classes.Name_Node,
            ast_internal_classes.Array_Subscript_Node
        ]
    ):
        return self.get_var(parent, variable)[0]

class IntrinsicNodeTransformer(NodeTransformer):

    def initialize(self, ast):
        # We need to rerun the assignment because transformations could have created
        # new AST nodes
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)
        self.ast = ast

        self.var_processor = VariableProcessor(self.scope_vars, self.ast)

    def get_var_declaration(
        self,
        parent: ast_internal_classes.FNode,
        variable: Union[
            ast_internal_classes.Data_Ref_Node, ast_internal_classes.Name_Node,
            ast_internal_classes.Array_Subscript_Node
        ]
    ):
        return self.var_processor.get_var_declaration(parent, variable)

    @staticmethod
    @abstractmethod
    def func_name() -> str:
        pass

    # @staticmethod
    # @abstractmethod
    # def transformation_name(self) -> str:
    #    pass


class DirectReplacement(IntrinsicTransformation):
    Replacement = namedtuple("Replacement", "function")
    Transformation = namedtuple("Transformation", "function")

    class ASTTransformation(IntrinsicNodeTransformer):

        @staticmethod
        def func_name() -> str:
            return "direct_replacement"

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
                binop_node.rval, input_type = replacement_rule.function(self, node, 0)  # binop_node.line)

                var = binop_node.lval

                # replace types of return variable - LHS of the binary operator
                # we only propagate that for the assignment
                # we handle extracted call variables this way
                # but we can also have different shapes, e.g., `maxval(something) > something_else`
                # hence the check
                if isinstance(var, (ast_internal_classes.Name_Node, ast_internal_classes.Array_Subscript_Node, ast_internal_classes.Data_Ref_Node)):

                    var_decl = self.get_var_declaration(var.parent, var)
                    var_decl.type = input_type

                var.type = input_type

            return binop_node

    def replace_size(transformer: IntrinsicNodeTransformer, var: ast_internal_classes.Call_Expr_Node, line):

        if len(var.args) not in [1, 2]:
            assert False, "Incorrect arguments to size!"

        # get variable declaration for the first argument
        var_decl = transformer.get_var_declaration(var.parent, var.args[0])

        # one arg to SIZE? compute the total number of elements
        if len(var.args) == 1:

            if len(var_decl.sizes) == 1:
                return (var_decl.sizes[0], "INTEGER")

            ret = ast_internal_classes.BinOp_Node(
                lval=var_decl.sizes[0],
                rval=ast_internal_classes.Name_Node(name="INTRINSIC_TEMPORARY"),
                op="*"
            )
            cur_node = ret
            for i in range(1, len(var_decl.sizes) - 1):
                cur_node.rval = ast_internal_classes.BinOp_Node(
                    lval=var_decl.sizes[i],
                    rval=ast_internal_classes.Name_Node(name="INTRINSIC_TEMPORARY"),
                    op="*"
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

    def _replace_lbound_ubound(func: str, transformer: IntrinsicNodeTransformer,
                               var: ast_internal_classes.Call_Expr_Node, line):

        if len(var.args) not in [1, 2]:
            assert False, "Incorrect arguments to lbound/ubound"

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

        is_assumed = isinstance(var_decl.offsets[rank_value - 1], ast_internal_classes.Name_Node) and var_decl.offsets[
            rank_value - 1].name.startswith("__f2dace_")

        if func == 'lbound':

            if is_assumed and not var_decl.alloc:
                value = ast_internal_classes.Int_Literal_Node(value="1")
            elif isinstance(var_decl.offsets[rank_value - 1], int):
                value = ast_internal_classes.Int_Literal_Node(value=str(var_decl.offsets[rank_value - 1]))
            else:
                value = var_decl.offsets[rank_value - 1]

        else:
            if isinstance(var_decl.sizes[rank_value - 1], ast_internal_classes.FNode):
                size = var_decl.sizes[rank_value - 1]
            else:
                size = ast_internal_classes.Int_Literal_Node(value=var_decl.sizes[rank_value - 1])

            if is_assumed and not var_decl.alloc:
                value = size
            else:
                if isinstance(var_decl.offsets[rank_value - 1], ast_internal_classes.FNode):
                    offset = var_decl.offsets[rank_value - 1]
                elif isinstance(var_decl.offsets[rank_value - 1], int):
                    offset = ast_internal_classes.Int_Literal_Node(value=str(var_decl.offsets[rank_value - 1]))
                else:
                    offset = ast_internal_classes.Int_Literal_Node(value=var_decl.offsets[rank_value - 1])

                value = ast_internal_classes.BinOp_Node(
                    op="+",
                    lval=size,
                    rval=ast_internal_classes.BinOp_Node(
                        op="-",
                        lval=offset,
                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                        line_number=line
                    ),
                    line_number=line
                )

        return (value, "INTEGER")

    def replace_lbound(transformer: IntrinsicNodeTransformer, var: ast_internal_classes.Call_Expr_Node, line):
        return DirectReplacement._replace_lbound_ubound("lbound", transformer, var, line)

    def replace_ubound(transformer: IntrinsicNodeTransformer, var: ast_internal_classes.Call_Expr_Node, line):
        return DirectReplacement._replace_lbound_ubound("ubound", transformer, var, line)

    def replace_bit_size(transformer: IntrinsicNodeTransformer, var: ast_internal_classes.Call_Expr_Node, line):

        if len(var.args) != 1:
            assert False, "Incorrect arguments to bit_size"

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
        return ast_internal_classes.Int_Literal_Node(value=str(
            math.ceil((math.log2(math.pow(10, int(arg0))) + 1) / 8)),
            line_number=line)

    def replace_real_kind(args: ast_internal_classes.Arg_List_Node, line, symbols: list):
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
                raise ValueError("Only literals or symbols can be arguments in selector")
        else:
            arg1 = 0
        if int(arg0) >= 9 or int(arg1) > 126:
            return ast_internal_classes.Int_Literal_Node(value="8", line_number=line)
        elif int(arg0) >= 3 or int(arg1) > 14:
            return ast_internal_classes.Int_Literal_Node(value="4", line_number=line)
        else:
            return ast_internal_classes.Int_Literal_Node(value="2", line_number=line)

    def replace_present(transformer: IntrinsicNodeTransformer, call: ast_internal_classes.Call_Expr_Node, line):

        assert len(call.args) == 1
        assert isinstance(call.args[0], ast_internal_classes.Name_Node)

        var_name = call.args[0].name
        test_var_name = f'__f2dace_OPTIONAL_{var_name}'

        return (ast_internal_classes.Name_Node(name=test_var_name), "LOGICAL")

    def replace_allocated(transformer: IntrinsicNodeTransformer, call: ast_internal_classes.Call_Expr_Node, line):

        assert len(call.args) == 1
        assert isinstance(call.args[0], ast_internal_classes.Name_Node)

        var_name = call.args[0].name
        test_var_name = f'__f2dace_ALLOCATED_{var_name}'

        return (ast_internal_classes.Name_Node(name=test_var_name), "LOGICAL")

    def replacement_epsilon(args: ast_internal_classes.Arg_List_Node, line, symbols: list):

        # assert len(args) == 1
        # assert isinstance(args[0], ast_internal_classes.Name_Node)

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
        "ALLOCATED": Transformation(replace_allocated)
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
        # return ast_internal_classes.Name_Node(name=DirectReplacement.FUNCTIONS[func_name][0])
        return ast_internal_classes.Name_Node(name=f'__dace_{func_name}')

    @staticmethod
    def replacable(func_name: str) -> bool:
        orig_name = func_name.split('__dace_')
        if len(orig_name) > 1 and orig_name[1] in DirectReplacement.FUNCTIONS:
            return isinstance(DirectReplacement.FUNCTIONS[orig_name[1]], DirectReplacement.Replacement)
        return False

    @staticmethod
    def replace(func_name: str, args: ast_internal_classes.Arg_List_Node, line, symbols: list) \
            -> ast_internal_classes.FNode:
        # Here we already have __dace_func
        fname = func_name.split('__dace_')[1]
        return DirectReplacement.FUNCTIONS[fname].function(args, line, symbols)

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
                     arg: ast_internal_classes.FNode, dims_count: Optional[int] = -1
                     ) -> ast_internal_classes.Array_Subscript_Node:

        # supports syntax func(arr)
        if isinstance(arg, ast_internal_classes.Name_Node):
            # If we access SUM(arr) where arr has many dimensions,
            # We need to create a ParDecl_Node for each dimension
            # array_sizes = self.scope_vars.get_var(node.parent, arg.name).sizes
            array_sizes = self.get_var_declaration(node.parent, arg).sizes
            if array_sizes is None:

                raise NeedsTypeInferenceException(self.func_name(), node.line_number)

            dims = len(array_sizes)

            # it's a scalar!
            if dims == 0:
                return None

            if isinstance(arg, ast_internal_classes.Name_Node):
                return ast_internal_classes.Array_Subscript_Node(
                    name=arg, parent=arg.parent, type='VOID',
                    indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * dims)

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

            if isinstance(cur_val.part_ref, ast_internal_classes.Name_Node):
                cur_val.part_ref = ast_internal_classes.Array_Subscript_Node(
                    name=cur_val.part_ref, parent=arg.parent, type='VOID',
                    indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * dims
                )
                ##i
                #else:
                #    cur_val.part_ref = ast_internal_classes.Array_Subscript_Node(
                #        name=cur_val.part_ref.name, parent=arg.parent, type='VOID',
                #        indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * dims
                #    )
            return arg

        # supports syntax func(arr(:))
        if isinstance(arg, ast_internal_classes.Array_Subscript_Node):
            return arg

        return None

    def _parse_binary_op(self, node: ast_internal_classes.Call_Expr_Node, arg: ast_internal_classes.BinOp_Node) -> \
            Tuple[
                ast_internal_classes.Array_Subscript_Node,
                Optional[ast_internal_classes.Array_Subscript_Node],
                ast_internal_classes.BinOp_Node
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

            difference = ast_internal_classes.BinOp_Node(
                lval=end_loop,
                op="-",
                rval=start_loop,
                line_number=node.line_number
            )
            new_index = ast_internal_classes.BinOp_Node(
                lval=idx_var,
                op="+",
                rval=difference,
                line_number=node.line_number
            )
            array.indices[i] = new_index
            #difference = int(end_loop.value) - int(start_loop.value)
            #if difference != 0:
            #    new_index = ast_internal_classes.BinOp_Node(
            #        lval=idx_var,
            #        op="+",
            #        rval=ast_internal_classes.Int_Literal_Node(value=str(difference)),
            #        line_number=node.line_number
            #    )
            #    array.indices[i] = new_index

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
                raise NotImplementedError("We do not support non-array arguments for SUM/PRODUCT")

    def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                        new_func_body: List[ast_internal_classes.FNode]):

        if len(self.rvals) != 1:
            raise NotImplementedError("Only one array can be summed")

        self.argument_variable = self.rvals[0]

        par_Decl_Range_Finder(self.argument_variable, self.loop_ranges, [], self.count, new_func_body,
                              self.scope_vars, self.ast.structures, True)

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.Int_Literal_Node(value=self._result_init_value()),
            line_number=node.line_number
        )

    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=ast_internal_classes.BinOp_Node(
                lval=node.lval,
                op=self._result_update_op(),
                rval=self.argument_variable,
                line_number=node.line_number
            ),
            line_number=node.line_number
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
        if first_arg.type == 'VOID':
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
        if first_arg.type == 'VOID':
            return None

        return [], first_arg.type

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
        var_decl = self.get_var_declaration(var.parent, var)
        var.type = "INTEGER"
        var_decl.type = "INTEGER"

    def _parse_call_expr_node(self, node: ast_internal_classes.Call_Expr_Node):

        if len(node.args) > 1:
            raise NotImplementedError("Fortran ANY with the DIM parameter is not supported!")
        arg = node.args[0]

        array_node = self._parse_array(node, arg)
        if array_node is None:
            # it's just a scalar - create a fake array for processing
            range_const = ast_internal_classes.Int_Literal_Node(value="0")
            array_node = ast_internal_classes.Array_Subscript_Node(
                name=arg, parent=arg.parent, type='VOID',
                indices=[
                    ast_internal_classes.ParDecl_Node(
                        type='RANGE',
                        range=[range_const, range_const]
                    )
                ],
                sizes = []
            )

        self.first_array = array_node
        self.cond = ast_internal_classes.BinOp_Node(
            op="==",
            rval=ast_internal_classes.Int_Literal_Node(value="1"),
            lval=self.first_array,
            line_number=node.line_number
        )

    def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                        new_func_body: List[ast_internal_classes.FNode]):

        rangeslen_left = []
        par_Decl_Range_Finder(self.first_array, self.loop_ranges, rangeslen_left, self.count, new_func_body,
                            self.scope_vars, self.ast.structures, True)

        if self.second_array is None:
            return

        loop_ranges_right = []
        rangeslen_right = []
        par_Decl_Range_Finder(self.second_array, loop_ranges_right, rangeslen_right, self.count, new_func_body,
                              self.scope_vars, self.ast.structures, True)

        for left_len, right_len in zip(rangeslen_left, rangeslen_right):
            if left_len != right_len:
                raise TypeError("Can't support Fortran ANY with different array ranks!")

        # In this intrinsic, the left array dictates loop range.
        # Thus, we only need to adjust the second array
        self._adjust_array_ranges(node, self.second_array, self.loop_ranges, loop_ranges_right)

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        init_value = self._result_init_value()

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

        body_if = ast_internal_classes.Execution_Part_Node(execution=[
            self._result_loop_update(node),
            # TODO: we should make the `break` generation conditional based on the architecture
            # For parallel maps, we should have no breaks
            # For sequential loop, we want a break to be faster
            # ast_internal_classes.Break_Node(
            #    line_number=node.line_number
            # )
        ])

        return ast_internal_classes.If_Stmt_Node(
            cond=self._loop_condition(),
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

    class Transformation(AnyAllCountTransformation):

        def _result_init_value(self):
            return "0"

        def _result_loop_update(self, node: ast_internal_classes.FNode):
            return ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="=",
                rval=ast_internal_classes.Int_Literal_Node(value="1"),
                line_number=node.line_number
            )

        def _loop_condition(self):
            return self.cond

        @staticmethod
        def func_name() -> str:
            return "__dace_any"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == 'VOID':
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
                line_number=node.line_number
            )

        def _loop_condition(self):
            return ast_internal_classes.UnOp_Node(
                op="not",
                lval=self.cond
            )

        @staticmethod
        def func_name() -> str:
            return "__dace_all"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == 'VOID':
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
                line_number=node.line_number
            )
            return ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(node.lval),
                op="=",
                rval=update,
                line_number=node.line_number
            )

        def _loop_condition(self):
            return self.cond

        @staticmethod
        def func_name() -> str:
            return "__dace_count"

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == 'VOID':
            return None

        # Without DIM parameter, it only returns scalars
        return [], "INTEGER"


class MinMaxValTransformation(LoopBasedReplacementTransformation):

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

            #if isinstance(arg, ast_internal_classes.Data_Ref_Node):
            #    self.rvals.append(arg)
            #    continue

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

        par_Decl_Range_Finder(self.argument_variable, self.loop_ranges, [], self.count, new_func_body,
                              self.scope_vars, self.ast.structures, declaration=True)

    def _initialize_result(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        return ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=self._result_init_value(self.argument_variable),
            line_number=node.line_number
        )

    def _generate_loop_body(self, node: ast_internal_classes.FNode) -> ast_internal_classes.BinOp_Node:

        cond = ast_internal_classes.BinOp_Node(
            lval=self.argument_variable,
            op=self._condition_op(),
            rval=node.lval,
            line_number=node.line_number
        )
        body_if = ast_internal_classes.BinOp_Node(
            lval=node.lval,
            op="=",
            rval=copy.deepcopy(self.argument_variable),
            line_number=node.line_number
        )
        return ast_internal_classes.If_Stmt_Node(
            cond=cond,
            body=body_if,
            body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
            line_number=node.line_number
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
        if first_arg.type == 'VOID':
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
        if first_arg.type == 'VOID':
            return None

        # Without DIM parameter, it only returns scalars
        return [], first_arg.type


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

                for ind in self.first_array.indices:
                    pardecls = [i for i in mywalk(ind) if isinstance(i, ast_internal_classes.ParDecl_Node)]
                    len_pardecls_first_array += len(pardecls)
                for ind in self.second_array.indices:
                    pardecls = [i for i in mywalk(ind) if isinstance(i, ast_internal_classes.ParDecl_Node)]
                    len_pardecls_second_array += len(pardecls)    
                assert len_pardecls_first_array == len_pardecls_second_array
                if len_pardecls_first_array == 0:
                    self.uses_scalars = True
                else:
                    self.uses_scalars = False

            # Last argument is either an array or a binary op

            arg = node.args[2]
            if self.uses_scalars:
                self.mask_cond = arg
            else:

                array_node = self._parse_array(node, node.args[2], dims_count=len(self.first_array.indices))
                if array_node is not None:

                    self.mask_first_array = array_node

                    self.mask_cond = ast_internal_classes.BinOp_Node(
                        op="==",
                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                        lval=self.mask_first_array,
                        line_number=node.line_number
                    )
                else:
                    self.mask_cond = arg

                #else:

                #    self.mask_first_array, self.mask_second_array, self.mask_cond = self._parse_binary_op(node, arg)

        def _summarize_args(self, exec_node: ast_internal_classes.Execution_Part_Node, node: ast_internal_classes.FNode,
                            new_func_body: List[ast_internal_classes.FNode]):

            if self.uses_scalars:
                self.destination_array = node.lval
                return


            # The first main argument is an array -> this dictates loop boundaries
            # Other arrays, regardless if they appear as the second array or mask, need to have the same loop boundary.
            par_Decl_Range_Finder(self.first_array, self.loop_ranges, [], self.count, new_func_body,
                                  self.scope_vars, self.ast.structures, True, allow_scalars=True)

            loop_ranges = []
            par_Decl_Range_Finder(self.second_array, loop_ranges, [], self.count, new_func_body,
                                  self.scope_vars, self.ast.structures, True, allow_scalars=True)
            self._adjust_array_ranges(node, self.second_array, self.loop_ranges, loop_ranges)

            # parse destination

            assert isinstance(node.lval, ast_internal_classes.Name_Node)

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
                    name=node.lval, parent=node.lval.parent, type='VOID',
                    indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * dims
                )
                par_Decl_Range_Finder(self.destination_array, [], [], self.count,
                                    new_func_body, self.scope_vars, self.ast.structures, True)
            else:
                self.destination_array = node.lval

            if self.mask_first_array is not None:
                loop_ranges = []
                par_Decl_Range_Finder(self.mask_first_array, loop_ranges, [], self.count, new_func_body,
                                      self.scope_vars, self.ast.structures, True, allow_scalars=True)
                self._adjust_array_ranges(node, self.mask_first_array, self.loop_ranges, loop_ranges)

            if self.mask_second_array is not None:
                loop_ranges = []
                par_Decl_Range_Finder(self.mask_second_array, loop_ranges, [], self.count, new_func_body,
                                      self.scope_vars, self.ast.structures, True, allow_scalars=True)
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

            copy_first = ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(self.destination_array),
                op="=",
                rval=self.first_array,
                line_number=node.line_number
            )

            copy_second = ast_internal_classes.BinOp_Node(
                lval=copy.deepcopy(self.destination_array),
                op="=",
                rval=self.second_array,
                line_number=node.line_number
            )

            body_if = ast_internal_classes.Execution_Part_Node(execution=[
                copy_first
            ])

            body_else = ast_internal_classes.Execution_Part_Node(execution=[
                copy_second
            ])

            # for scalar operations, we need to extract first element if it's an array
            if self.uses_scalars and isinstance(self.mask_cond, ast_internal_classes.Name_Node):
                definition = self.scope_vars.get_var(node.parent, self.mask_cond.name)

                if definition.sizes is not None and len(definition.sizes) > 0:
                    self.mask_cond = ast_internal_classes.Array_Subscript_Node(
                        name = self.mask_cond,
                        type = self.mask_cond.type,
                        indices= [ast_internal_classes.Int_Literal_Node(value="1")] * len(definition.sizes)
                    )

            return ast_internal_classes.If_Stmt_Node(
                cond=self.mask_cond,
                body=body_if,
                body_else=body_else,
                line_number=node.line_number
            )

    @staticmethod
    def output_size(args: ast_internal_classes.FNode) -> Optional[Tuple[list, str]]:

        first_arg = args[0]
        if first_arg.type == 'VOID':
            return None

        return first_arg.sizes, first_arg.type


class IntrinsicSDFGTransformation(xf.SingleStateTransformation):
    array1 = xf.PatternNode(nodes.AccessNode)
    array2 = xf.PatternNode(nodes.AccessNode)
    tasklet = xf.PatternNode(nodes.Tasklet)
    out = xf.PatternNode(nodes.AccessNode)

    def blas_dot(self, state: SDFGState, sdfg: SDFG):
        dot_libnode(None, sdfg, state, self.array1.data, self.array2.data, self.out.data)

    def blas_matmul(self, state: SDFGState, sdfg: SDFG):
        gemm_libnode(
            None,
            sdfg,
            state,
            self.array1.data,
            self.array2.data,
            self.out.data,
            1.0,
            0.0,
            False,
            False
        )


    def transpose(self, state: SDFGState, sdfg: SDFG):

        libnode = Transpose("transpose", dtype=sdfg.arrays[self.array1.data].dtype)
        state.add_node(libnode)

        state.add_edge(self.array1, None, libnode, "_inp", sdfg.make_array_memlet(self.array1.data))
        state.add_edge(libnode, "_out", self.out, None, sdfg.make_array_memlet(self.out.data))

    @staticmethod
    def transpose_size(node: ast_internal_classes.Call_Expr_Node, arg_sizes: List[ List[ast_internal_classes.FNode] ]):

        assert len(arg_sizes) == 1
        return list(reversed(arg_sizes[0]))

    @staticmethod
    def matmul_size(node: ast_internal_classes.Call_Expr_Node, arg_sizes: List[ List[ast_internal_classes.FNode] ]):

        assert len(arg_sizes) == 2
        return [
            arg_sizes[0][0],
            arg_sizes[1][1]
        ]

    LIBRARY_NODE_TRANSFORMATIONS = {
        "__dace_blas_dot": blas_dot,
        "__dace_transpose": transpose,
        "__dace_matmul": blas_matmul
    }

    @classmethod
    def expressions(cls):

        graphs = []

        # Match tasklets with two inputs, like dot
        g = OrderedDiGraph()
        g.add_node(cls.array1)
        g.add_node(cls.array2)
        g.add_node(cls.tasklet)
        g.add_node(cls.out)
        g.add_edge(cls.array1, cls.tasklet, None)
        g.add_edge(cls.array2, cls.tasklet, None)
        g.add_edge(cls.tasklet, cls.out, None)
        graphs.append(g)

        # Match tasklets with one input, like transpose
        g = OrderedDiGraph()
        g.add_node(cls.array1)
        g.add_node(cls.tasklet)
        g.add_node(cls.out)
        g.add_edge(cls.array1, cls.tasklet, None)
        g.add_edge(cls.tasklet, cls.out, None)
        graphs.append(g)

        return graphs

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:

        import ast
        for node in ast.walk(self.tasklet.code.code[0]):
            if isinstance(node, ast.Call):
                if node.func.id in self.LIBRARY_NODE_TRANSFORMATIONS:
                    self.func = self.LIBRARY_NODE_TRANSFORMATIONS[node.func.id]
                    return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):

        self.func(self, state, sdfg)

        for in_edge in state.in_edges(self.tasklet):
            state.remove_memlet_path(in_edge)

        for in_edge in state.out_edges(self.tasklet):
            state.remove_memlet_path(in_edge)

        state.remove_node(self.tasklet)


class MathFunctions(IntrinsicTransformation):
    MathTransformation = namedtuple("MathTransformation", "function return_type size_function", defaults=[None, None, None])
    MathReplacement = namedtuple("MathReplacement", "function replacement_function return_type")

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
            op="*",
            lval=x,
            rval=rval,
            line_number=line
        )

        # pack it into parentheses, just to be sure
        return ast_internal_classes.Parenthesis_Expr_Node(expr=mult)

    def generate_epsilon(arg: ast_internal_classes.Call_Expr_Node):
        ret_val = sys.float_info.epsilon
        return ast_internal_classes.Real_Literal_Node(value=str(ret_val))

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

    @staticmethod
    def _initialize_transformations():
        # dictionary comprehension cannot access class members
        ret = {}
        for name, value in IntrinsicSDFGTransformation.INTRINSIC_TRANSFORMATIONS.items():
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
        "ATAN2": MathTransformation("atan2", "FIRST_ARG"),
        "DOT_PRODUCT": MathTransformation("__dace_blas_dot", "FIRST_ARG"),
        "TRANSPOSE": MathTransformation("__dace_transpose", "FIRST_ARG", IntrinsicSDFGTransformation.transpose_size),
        "MATMUL": MathTransformation("__dace_matmul", "FIRST_ARG", IntrinsicSDFGTransformation.matmul_size),
        "IBSET": MathTransformation("bitwise_set", "INTEGER"),
        "IEOR": MathTransformation("bitwise_xor", "INTEGER"),
        "ISHFT": MathTransformation("bitwise_shift", "INTEGER"),
        "IBCLR": MathTransformation("bitwise_clear", "INTEGER"),
        "BTEST": MathTransformation("bitwise_test", "INTEGER"),
        "IBITS": MathTransformation("bitwise_extract", "INTEGER"),
        "IAND": MathTransformation("bitwise_and", "INTEGER")
    }

    class TypeTransformer(IntrinsicNodeTransformer):

        def func_type(self, node: ast_internal_classes.Call_Expr_Node):
            # take the first arg
            arg = node.args[0]
            if isinstance(arg, (ast_internal_classes.Real_Literal_Node, ast_internal_classes.Double_Literal_Node,
                                ast_internal_classes.Int_Literal_Node, ast_internal_classes.Call_Expr_Node,
                                ast_internal_classes.BinOp_Node, ast_internal_classes.UnOp_Node)):
                return arg.type
            elif isinstance(arg, (ast_internal_classes.Name_Node, ast_internal_classes.Array_Subscript_Node, ast_internal_classes.Data_Ref_Node)):
                return self.get_var_declaration(node.parent, arg).type
            else:
                raise NotImplementedError(type(arg))

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
            new_args = []
            for arg in node.args:
                new_args.append(self.visit(arg))
            node.args = new_args

            input_type = self.func_type(node)
            if input_type == 'VOID':
                #assert input_type != 'VOID', f"Unexpected void input at line number: {node.line_number}"
                raise NeedsTypeInferenceException(func_name, node.line_number)

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

    # All functions return an array
    # Our call extraction transformation only supports scalars
    #
    # No longer needed!
    EXEMPTED_FROM_CALL_EXTRACTION = [
    ]

    def __init__(self):
        self._transformations_to_run = {}

    def transformations(self) -> List[NodeTransformer]:
        return list(self._transformations_to_run.values())

    @staticmethod
    def function_names() -> List[str]:
        # list of all functions that are created by initial transformation, before doing full replacement
        # this prevents other parser components from replacing our function calls with array subscription nodes
        return [*list(LoopBasedReplacement.INTRINSIC_TO_DACE.values()), *MathFunctions.temporary_functions(),
                *DirectReplacement.temporary_functions()]

    @staticmethod
    def retained_function_names() -> List[str]:
        # list of all DaCe functions that we use after full parsing
        return MathFunctions.dace_functions()

    @staticmethod
    def call_extraction_exemptions() -> List[str]:
        return FortranIntrinsics.EXEMPTED_FROM_CALL_EXTRACTION

    def replace_function_name(self, node: Union[FASTNode, ast_internal_classes.Name_Node]) -> ast_internal_classes.Name_Node:

        if isinstance(node, ast_internal_classes.Name_Node):
            func_name = node.name
        else:
            func_name = node.string

        replacements = {
            "SIGN": "__dace_sign",
            # TODO implement and categorize the intrinsic functions below
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
                    self._transformations_to_run[transformation.func_name()] = transformation

            return DirectReplacement.replace_name(func_name)
        elif MathFunctions.replacable(func_name):

            transformation = MathFunctions.get_transformation()
            if transformation.func_name() not in self._transformations_to_run:
                self._transformations_to_run[transformation.func_name()] = transformation

            return MathFunctions.replace(func_name)

        if self.IMPLEMENTATIONS_AST[func_name].has_transformation():

            if hasattr(self.IMPLEMENTATIONS_AST[func_name], "Transformation"):
                transformation = self.IMPLEMENTATIONS_AST[func_name].Transformation()
            else:
                transformation = self.IMPLEMENTATIONS_AST[func_name].get_transformation(func_name)

            if transformation.func_name() not in self._transformations_to_run:
                self._transformations_to_run[transformation.func_name()] = transformation

        return ast_internal_classes.Name_Node(name=self.IMPLEMENTATIONS_AST[func_name].replaced_name(func_name))

    def replace_function_reference(self, name: ast_internal_classes.Name_Node, args: ast_internal_classes.Arg_List_Node,
                                   line, symbols: dict):

        func_types = {
            "__dace_sign": "DOUBLE",
        }
        if name.name in func_types:
            # FIXME: this will be progressively removed
            call_type = func_types[name.name]
            return ast_internal_classes.Call_Expr_Node(name=name, type=call_type, args=args.args, line_number=line,subroutine=False)
        elif DirectReplacement.replacable(name.name):
            return DirectReplacement.replace(name.name, args, line, symbols)
        else:
            # We will do the actual type replacement later
            # To that end, we need to know the input types - but these we do not know at the moment.
            return ast_internal_classes.Call_Expr_Node(
                name=name, type="VOID", subroutine=False,
                args=args.args, line_number=line
            )

    @staticmethod
    def output_size(node: ast_internal_classes.Call_Expr_Node):

        name = node.name.name.split('__dace_')
        #if len(name) != 2 or name[1].upper() not in MathFunctions.INTRINSIC_SIZE_FUNCTIONS:
        if len(name) != 2:
            return None, None, 'VOID'

        sizes = []
        for arg in node.args:

            if isinstance(arg, (ast_internal_classes.Int_Literal_Node, ast_internal_classes.Real_Literal_Node)):
                sizes.append(1)
            else:
                sizes.append(arg.sizes)

        input_type = node.args[0].type
        return_type = 'VOID'

        func_name = name[1].upper()

        if func_name in FortranIntrinsics.IMPLEMENTATIONS_AST:

            replacement_rule = FortranIntrinsics.IMPLEMENTATIONS_AST[func_name]
            res = replacement_rule.output_size(node.args)
            if res is None:
                return None, None, 'VOID'
            else:
                sizes = res[0]
                return_type = res[1]

        elif func_name in MathFunctions.INTRINSIC_TO_DACE:

            replacement_rule = MathFunctions.INTRINSIC_TO_DACE[func_name]
            if isinstance(replacement_rule, dict):
                replacement_rule = replacement_rule[input_type]

            if isinstance(replacement_rule, MathFunctions.MathTransformation) and replacement_rule.size_function is not None:

                sizes = replacement_rule.size_function(node, sizes)
            else:

                if input_type != 'VOID':

                    if replacement_rule.return_type == "FIRST_ARG":
                        return_type = input_type
                    else:
                        return_type = replacement_rule.return_type

                sizes = sizes[0]

        else:
            return None, None, 'VOID'

        if isinstance(sizes, ast_internal_classes.Int_Literal_Node):
            return sizes, [1], return_type
        elif isinstance(sizes, list):
            return sizes, [1] * len(sizes), return_type
        else:
            return [], [1], return_type
