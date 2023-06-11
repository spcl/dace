# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from dace.frontend.fortran import ast_components, ast_internal_classes
from typing import List, Tuple, Set
import copy


def iter_fields(node: ast_internal_classes.FNode):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    if not hasattr(node, "_fields"):
        a = 1
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node: ast_internal_classes.FNode):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    #print("CLASS: ",node.__class__)
    #if isinstance(node,DeclRefExpr):
    #print("NAME: ", node.name)

    for name, field in iter_fields(node):
        #print("NASME:",name)
        if isinstance(field, ast_internal_classes.FNode):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast_internal_classes.FNode):
                    yield item


class NodeVisitor(object):
    """
    A base node visitor class for Fortran ASTs.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """
    def visit(self, node: ast_internal_classes.FNode):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast_internal_classes.FNode):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value)


class NodeTransformer(NodeVisitor):
    """
    A base node visitor that walks the abstract syntax tree and allows
    modification of nodes.
    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace old nodes. 
    """
    def as_list(self, x):
        if isinstance(x, list):
            return x
        if x is None:
            return []
        return [x]

    def generic_visit(self, node: ast_internal_classes.FNode):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_internal_classes.FNode):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_internal_classes.FNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_internal_classes.FNode):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class FindFunctionAndSubroutines(NodeVisitor):
    """
    Finds all function and subroutine names in the AST
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        self.nodes.append(node.name)

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):
        self.nodes.append(node.name)


class FindInputs(NodeVisitor):
    """
    Finds all inputs (reads) in the AST node and its children
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        self.nodes.append(node)

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.nodes.append(node.name)
        for i in node.indices:
            self.visit(i)

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                pass
            elif isinstance(node.lval, ast_internal_classes.Array_Subscript_Node):
                for i in node.lval.indices:
                    self.visit(i)

        else:
            self.visit(node.lval)
        self.visit(node.rval)


class FindOutputs(NodeVisitor):
    """
    Finds all outputs (writes) in the AST node and its children
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                self.nodes.append(node.lval)
            elif isinstance(node.lval, ast_internal_classes.Array_Subscript_Node):
                self.nodes.append(node.lval.name)
            self.visit(node.rval)


class FindFunctionCalls(NodeVisitor):
    """
    Finds all function calls in the AST node and its children
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        self.nodes.append(node)
        for i in node.args:
            self.visit(i)


class CallToArray(NodeTransformer):
    """
    Fortran does not differentiate between arrays and functions.
    We need to go over and convert all function calls to arrays.
    So, we create a closure of all math and defined functions and 
    create array expressions for the others.
    """
    def __init__(self, funcs=None):
        if funcs is None:
            funcs = []
        self.funcs = funcs
        self.excepted_funcs = [
            "malloc", "exp", "pow", "sqrt", "cbrt", "max", "abs", "min", "__dace_sum", "__dace_sign", "tanh",
            "__dace_epsilon"
        ]

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if isinstance(node.name, str):
            return node
        if node.name.name in self.excepted_funcs or node.name in self.funcs:
            processed_args = []
            for i in node.args:
                arg = CallToArray(self.funcs).visit(i)
                processed_args.append(arg)
            node.args = processed_args
            return node
        indices = [CallToArray(self.funcs).visit(i) for i in node.args]
        return ast_internal_classes.Array_Subscript_Node(name=node.name, indices=indices)


class CallExtractorNodeLister(NodeVisitor):
    """
    Finds all function calls in the AST node and its children that have to be extracted into independent expressions
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Call_Expr_Node] = []

    def visit_For_Stmt_Node(self, node: ast_internal_classes.For_Stmt_Node):
        return

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        stop = False
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                stop = True
        if not stop and node.name.name not in [
                "malloc", "exp", "pow", "sqrt", "cbrt", "max", "min", "abs", "tanh", "__dace_epsilon"
        ]:
            self.nodes.append(node)
        return self.generic_visit(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class CallExtractor(NodeTransformer):
    """
    Uses the CallExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """
    def __init__(self, count=0):
        self.count = count

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        if node.name.name in ["malloc", "exp", "pow", "sqrt", "cbrt", "max", "min", "abs", "tanh", "__dace_epsilon"]:
            return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                return self.generic_visit(node)
        if not hasattr(self, "count"):
            self.count = 0
        else:
            self.count = self.count + 1
        tmp = self.count
        return ast_internal_classes.Name_Node(name="tmp_call_" + str(tmp - 1))

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = CallExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            for i in res:
                if i == child:
                    res.pop(res.index(i))
            temp = self.count
            if res is not None:
                for i in range(0, len(res)):

                    if (res[i].name.name == "__dace_sum"):
                        newbody.append(
                            ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                ast_internal_classes.Var_Decl_Node(
                                    name="tmp_call_" + str(temp),
                                    type=res[i].type,
                                    sizes=None,
                                )
                            ]))
                        newbody.append(
                            ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="tmp_call_" +
                                                                                                str(temp)),
                                                            op="=",
                                                            rval=ast_internal_classes.Int_Literal_Node(value="0"),
                                                            line_number=child.line_number))
                    else:

                        newbody.append(
                            ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                ast_internal_classes.Var_Decl_Node(
                                    name="tmp_call_" + str(temp),
                                    type=res[i].type,
                                    sizes=None,
                                )
                            ]))
                    newbody.append(
                        ast_internal_classes.BinOp_Node(op="=",
                                                        lval=ast_internal_classes.Name_Node(name="tmp_call_" +
                                                                                            str(temp),
                                                                                            type=res[i].type),
                                                        rval=res[i],
                                                        line_number=child.line_number))
                    temp = temp + 1
            if isinstance(child, ast_internal_classes.Call_Expr_Node):
                new_args = []
                if hasattr(child, "args"):
                    for i in child.args:
                        new_args.append(self.visit(i))
                new_child = ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                name=child.name,
                                                                args=new_args,
                                                                line_number=child.line_number)
                newbody.append(new_child)
            else:
                newbody.append(self.visit(child))

        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class IndexExtractorNodeLister(NodeVisitor):
    """
    Finds all array subscript expressions in the AST node and its children that have to be extracted into independent expressions
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Array_Subscript_Node] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if node.name.name in ["sqrt", "exp", "pow", "max", "min", "abs", "tanh"]:
            return self.generic_visit(node)
        else:
            return

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class IndexExtractor(NodeTransformer):
    """
    Uses the IndexExtractorNodeLister to find all array subscript expressions
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the index expression with the variable.
    """
    def __init__(self, count=0):
        self.count = count

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if node.name.name in ["sqrt", "exp", "pow", "max", "min", "abs", "tanh"]:
            return self.generic_visit(node)
        else:
            return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        tmp = self.count
        new_indices = []
        for i in node.indices:
            if isinstance(i, ast_internal_classes.ParDecl_Node):
                new_indices.append(i)
            else:
                new_indices.append(ast_internal_classes.Name_Node(name="tmp_index_" + str(tmp)))
                tmp = tmp + 1
        self.count = tmp
        return ast_internal_classes.Array_Subscript_Node(
            name=node.name,
            indices=new_indices,
        )

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = IndexExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            temp = self.count
            if res is not None:
                for j in res:
                    for i in j.indices:
                        if isinstance(i, ast_internal_classes.ParDecl_Node):
                            continue
                        else:
                            tmp_name = "tmp_index_" + str(temp)
                            temp = temp + 1
                            newbody.append(
                                ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                    ast_internal_classes.Var_Decl_Node(name=tmp_name,
                                                                       type="INTEGER",
                                                                       sizes=None,
                                                                       line_number=child.line_number)
                                ],
                                                                    line_number=child.line_number))
                            newbody.append(
                                ast_internal_classes.BinOp_Node(
                                    op="=",
                                    lval=ast_internal_classes.Name_Node(name=tmp_name),
                                    rval=ast_internal_classes.BinOp_Node(
                                        op="-",
                                        lval=i,
                                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                        line_number=child.line_number),
                                    line_number=child.line_number))
            newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class SignToIf(NodeTransformer):
    """
    Transforms all sign expressions into if statements
    """
    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node) and node.rval.name.name == "__dace_sign":
            args = node.rval.args
            lval = node.lval
            cond = ast_internal_classes.BinOp_Node(op=">=",
                                                   rval=ast_internal_classes.Real_Literal_Node(value="0.0"),
                                                   lval=args[1],
                                                   line_number=node.line_number)
            body_if = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.Call_Expr_Node(
                                                    name=ast_internal_classes.Name_Node(name="abs"),
                                                    type="DOUBLE",
                                                    args=[copy.deepcopy(args[0])],
                                                    line_number=node.line_number),
                                                line_number=node.line_number)
            ])
            body_else = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.UnOp_Node(
                                                    op="-",
                                                    lval=ast_internal_classes.Call_Expr_Node(
                                                        name=ast_internal_classes.Name_Node(name="abs"),
                                                        type="DOUBLE",
                                                        args=[copy.deepcopy(args[0])],
                                                        line_number=node.line_number),
                                                    line_number=node.line_number),
                                                line_number=node.line_number)
            ])
            return (ast_internal_classes.If_Stmt_Node(cond=cond,
                                                      body=body_if,
                                                      body_else=body_else,
                                                      line_number=node.line_number))

        else:
            return self.generic_visit(node)


class RenameArguments(NodeTransformer):
    """
    Renames all arguments of a function to the names of the arguments of the function call
    Used when eliminating function statements
    """
    def __init__(self, node_args: list, call_args: list):
        self.node_args = node_args
        self.call_args = call_args

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        for i, j in zip(self.node_args, self.call_args):
            if node.name == j.name:
                return copy.deepcopy(i)
        return node


class ReplaceFunctionStatement(NodeTransformer):
    """
    Replaces a function statement with its content, similar to propagating a macro
    """
    def __init__(self, statement, replacement):
        self.name = statement.name
        self.content = replacement

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if node.name == self.name:
            return ast_internal_classes.Parenthesis_Expr_Node(expr=copy.deepcopy(self.content))
        else:
            return self.generic_visit(node)


class ReplaceFunctionStatementPass(NodeTransformer):
    """
    Replaces a function statement with its content, similar to propagating a macro
    """
    def __init__(self, statefunc: list):
        self.funcs = statefunc

    def visit_Structure_Constructor_Node(self, node: ast_internal_classes.Structure_Constructor_Node):
        for i in self.funcs:
            if node.name.name == i[0].name.name:
                ret_node = copy.deepcopy(i[1])
                ret_node = RenameArguments(node.args, i[0].args).visit(ret_node)
                return ast_internal_classes.Parenthesis_Expr_Node(expr=ret_node)
        return self.generic_visit(node)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        for i in self.funcs:
            if node.name.name == i[0].name.name:
                ret_node = copy.deepcopy(i[1])
                ret_node = RenameArguments(node.args, i[0].args).visit(ret_node)
                return ast_internal_classes.Parenthesis_Expr_Node(expr=ret_node)
        return self.generic_visit(node)


def functionStatementEliminator(node=ast_internal_classes.Program_Node):
    """
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    :note Should only be used on the program node
    """
    main_program = localFunctionStatementEliminator(node.main_program)
    function_definitions = [localFunctionStatementEliminator(i) for i in node.function_definitions]
    subroutine_definitions = [localFunctionStatementEliminator(i) for i in node.subroutine_definitions]
    modules = []
    for i in node.modules:
        module_function_definitions = [localFunctionStatementEliminator(j) for j in i.function_definitions]
        module_subroutine_definitions = [localFunctionStatementEliminator(j) for j in i.subroutine_definitions]
        modules.append(
            ast_internal_classes.Module_Node(
                name=i.name,
                specification_part=i.specification_part,
                subroutine_definitions=module_subroutine_definitions,
                function_definitions=module_function_definitions,
            ))
    return ast_internal_classes.Program_Node(main_program=main_program,
                                             function_definitions=function_definitions,
                                             subroutine_definitions=subroutine_definitions,
                                             modules=modules)


def localFunctionStatementEliminator(node: ast_internal_classes.FNode):
    """
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    """
    spec = node.specification_part.specifications
    exec = node.execution_part.execution
    new_exec = exec.copy()
    to_change = []
    for i in exec:
        if isinstance(i, ast_internal_classes.BinOp_Node):
            if i.op == "=":
                if isinstance(i.lval, ast_internal_classes.Call_Expr_Node) or isinstance(
                        i.lval, ast_internal_classes.Structure_Constructor_Node):
                    function_statement_name = i.lval.name
                    is_actually_function_statement = False
                    #In Fortran, function statement are defined as scalar values, but called as arrays, so by identifiying that it is called as a call_expr or structure_constructor, we also need to match the specification part and see that it is scalar rather than an array.
                    found = False
                    for j in spec:
                        if found:
                            break
                        for k in j.vardecl:
                            if k.name == function_statement_name.name:
                                if k.sizes is None:
                                    is_actually_function_statement = True
                                    function_statement_type = k.type
                                    j.vardecl.remove(k)
                                    found = True
                                    break
                    if is_actually_function_statement:
                        to_change.append([i.lval, i.rval])
                        new_exec.remove(i)
                        print("Function statement found and removed: ", function_statement_name)
                    else:
                        #There are no function statements after the first one that isn't
                        break
    still_changing = True
    while still_changing:
        still_changing = False
        for i in to_change:
            rval = i[1]
            calls = FindFunctionCalls()
            calls.visit(rval)
            for j in to_change:
                for k in calls.nodes:
                    if k.name == j[0].name:
                        calls_to_replace = FindFunctionCalls()
                        calls_to_replace.visit(j[1])
                        #must check if it is recursive and contains other function statements
                        it_is_simple = True
                        for l in calls_to_replace.nodes:
                            for m in to_change:
                                if l.name == m[0].name:
                                    it_is_simple = False
                        if it_is_simple:
                            still_changing = True
                            i[1] = ReplaceFunctionStatement(j[0], j[1]).visit(rval)
    final_exec = []
    for i in new_exec:
        final_exec.append(ReplaceFunctionStatementPass(to_change).visit(i))
    node.execution_part.execution = final_exec
    node.specification_part.specifications = spec
    return node


class ArrayLoopNodeLister(NodeVisitor):
    """
    Finds all array operations that have to be transformed to loops in the AST
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.FNode] = []
        self.range_nodes: List[ast_internal_classes.FNode] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        rval_pardecls = [i for i in mywalk(node.rval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        lval_pardecls = [i for i in mywalk(node.lval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        if len(lval_pardecls) > 0:
            if len(rval_pardecls) == 1:
                self.range_nodes.append(node)
                self.nodes.append(node)
                return
            elif len(rval_pardecls) > 1:
                for i in rval_pardecls:
                    if i != rval_pardecls[0]:
                        raise NotImplementedError("Only supporting one range in right expression")

                self.range_nodes.append(node)
                self.nodes.append(node)
                return
            else:
                self.nodes.append(node)
                return

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


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


def par_Decl_Range_Finder(node: ast_internal_classes.Array_Subscript_Node,
                          ranges: list,
                          rangepos: list,
                          count: int,
                          newbody: list,
                          declaration=True,
                          is_sum_to_loop=False):
    """
    Helper function for the transformation of array operations and sums to loops
    :param node: The AST to be transformed
    :param ranges: The ranges of the loop
    :param rangepos: The positions of the ranges
    :param count: The current count of the loop
    :param newbody: The new basic block that will contain the loop
    :param declaration: Whether the declaration of the loop variable is needed
    :param is_sum_to_loop: Whether the transformation is for a sum to loop
    :return: Ranges, rangepos, newbody
    """

    currentindex = 0
    indices = []
    for i in node.indices:
        if isinstance(i, ast_internal_classes.ParDecl_Node):
            if i.type == "ALL":
                ranges.append([
                    ast_internal_classes.Int_Literal_Node(value="1"),
                    ast_internal_classes.Name_Range_Node(name="f2dace_MAX",
                                                         type="INTEGER",
                                                         arrname=node.name,
                                                         pos=currentindex)
                ])
            else:
                ranges.append([i.range[0], i.range[1]])
            rangepos.append(currentindex)
            if declaration:
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(
                            name="tmp_parfor_" + str(count + len(rangepos) - 1), type="INTEGER", sizes=None, init=None)
                    ]))
            indices.append(ast_internal_classes.Name_Node(name="tmp_parfor_" + str(count + len(rangepos) - 1)))
        else:
            indices.append(i)
        currentindex += 1

    node.indices = indices


class ArrayToLoop(NodeTransformer):
    """
    Transforms the AST by removing array expressions and replacing them with loops
    """
    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            lister = ArrayLoopNodeLister()
            lister.visit(child)
            res = lister.nodes
            res_range = lister.range_nodes
            if res is not None and len(res) > 0:

                current = child.lval
                val = child.rval
                ranges = []
                rangepos = []
                par_Decl_Range_Finder(current, ranges, rangepos, self.count, newbody, True)

                if res_range is not None and len(res_range) > 0:
                    rvals = [i for i in mywalk(val) if isinstance(i, ast_internal_classes.Array_Subscript_Node)]
                    for i in rvals:
                        rangeposrval = []
                        rangesrval = []

                        par_Decl_Range_Finder(i, rangesrval, rangeposrval, self.count, newbody, False)

                        for i, j in zip(ranges, rangesrval):
                            if i != j:
                                if isinstance(i, list) and isinstance(j, list) and len(i) == len(j):
                                    for k, l in zip(i, j):
                                        if k != l:
                                            if isinstance(k, ast_internal_classes.Name_Range_Node) and isinstance(
                                                    l, ast_internal_classes.Name_Range_Node):
                                                if k.name != l.name:
                                                    raise NotImplementedError("Ranges must be the same")
                                            else:
                                                raise NotImplementedError("Ranges must be the same")
                                else:
                                    raise NotImplementedError("Ranges must be identical")

                range_index = 0
                body = ast_internal_classes.BinOp_Node(lval=current, op="=", rval=val, line_number=child.line_number)
                for i in ranges:
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


def mywalk(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        yield node


class SumToLoop(NodeTransformer):
    """
    Transforms the AST by removing array sums and replacing them with loops
    """
    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            lister = SumLoopNodeLister()
            lister.visit(child)
            res = lister.nodes
            if res is not None and len(res) > 0:

                current = child.lval
                val = child.rval
                rvals = [i for i in mywalk(val) if isinstance(i, ast_internal_classes.Array_Subscript_Node)]
                if len(rvals) != 1:
                    raise NotImplementedError("Only one array can be summed")
                val = rvals[0]
                rangeposrval = []
                rangesrval = []

                par_Decl_Range_Finder(val, rangesrval, rangeposrval, self.count, newbody, False, True)

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


class RenameVar(NodeTransformer):
    def __init__(self, oldname: str, newname: str):
        self.oldname = oldname
        self.newname = newname

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        return ast_internal_classes.Name_Node(name=self.newname) if node.name == self.oldname else node


class ForDeclarer(NodeTransformer):
    """
    Ensures that each loop iterator is unique by extracting the actual iterator and assigning it to a uniquely named local variable
    """
    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Map_Stmt_Node):
                newbody.append(self.visit(child))
                continue
            if isinstance(child, ast_internal_classes.For_Stmt_Node):
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(
                            name="_for_it_" + str(self.count), type="INTEGER", sizes=None, init=None)
                    ]))
                final_assign = ast_internal_classes.BinOp_Node(lval=child.init.lval,
                                                               op="=",
                                                               rval=child.cond.rval,
                                                               line_number=child.line_number)
                newfor = RenameVar(child.init.lval.name, "_for_it_" + str(self.count)).visit(child)
                self.count += 1
                newfor = self.visit(newfor)
                newbody.append(newfor)

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)
