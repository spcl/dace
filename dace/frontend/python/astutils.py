# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Various AST parsing utilities for DaCe. """
import ast
import astunparse
from collections import OrderedDict
import inspect
import sympy
from typing import Any, Dict, List, Tuple

from dace import dtypes, symbolic


def _remove_outer_indentation(src: str):
    """ Removes extra indentation from a source Python function.
        :param src: Source code (possibly indented).
        :return: Code after de-indentation.
    """
    lines = src.split('\n')
    indentation = len(lines[0]) - len(lines[0].lstrip())
    return '\n'.join([line[indentation:] for line in lines])


def function_to_ast(f):
    """ Obtain the source code of a Python function and create an AST.
        :param f: Python function.
        :return: A 4-tuple of (AST, function filename, function line-number,
                               source code as string).
    """
    try:
        src = inspect.getsource(f)
    # TypeError: X is not a module, class, method, function, traceback, frame,
    # or code object; OR OSError: could not get source code
    except (TypeError, OSError):
        raise TypeError('Cannot obtain source code for dace program. This may '
                        'happen if you are using the "python" default '
                        'interpreter. Please either use the "ipython" '
                        'interpreter, a Jupyter or Colab notebook, or place '
                        'the source code in a file and import it.')

    src_file = inspect.getfile(f)
    _, src_line = inspect.findsource(f)
    src_ast = ast.parse(_remove_outer_indentation(src))
    ast.increment_lineno(src_ast, src_line)

    return src_ast, src_file, src_line, src


def rname(node):
    """ Obtains names from different types of AST nodes. """

    if isinstance(node, str):
        return node
    if isinstance(node, ast.Num):
        return str(node.n)
    if isinstance(node, ast.Name):  # form x
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Subscript):  # form A[a:b,...,c:d]
        return rname(node.value)
    if isinstance(node, ast.Attribute):  # form @dace.attr_noparams
        return rname(node.value) + '.' + rname(node.attr)
    if isinstance(node, ast.Call):  # form @dace.attr(...)
        if isinstance(node.func, ast.Name):
            return node.func.id
        return node.func.value.id + '.' + node.func.attr
    if isinstance(node, ast.FunctionDef):  # form def func(...)
        return node.name
    if isinstance(node, ast.keyword):
        return node.arg
    try:
        if isinstance(node, ast.arg):  # form func(..., arg, ...)
            return node.arg
    except AttributeError:
        pass

    raise TypeError('Invalid AST node type: ' + str(type(node)))


def subscript_to_ast_slice(node, without_array=False):
    """ Converts an AST subscript to slice on the form
        (<name>, [<3-tuples of AST nodes>]). If an ast.Name is passed, returns
        (name, None), implying the full range. 
        :param node: The AST node to convert.
        :param without_array: If True, returns only the slice. Otherwise,
                              returns a 2-tuple of (array, range).
    """

    if isinstance(node, ast.Name):
        # None implies the full array. We can't create artificial
        # (None, None, None) tuples, because we don't know the dimensionality of
        #  the array at this point
        result_arr, result_slice = node.id, None
        return result_slice if without_array else (result_arr, result_slice)

    if not isinstance(node, ast.Subscript):
        raise TypeError('AST node is not a subscript')

    # ND Index
    if isinstance(node.slice, ast.Index):
        if isinstance(node.slice.value, ast.Tuple):
            result_slice = [dim for dim in node.slice.value.elts]
        else:
            result_slice = [node.slice.value]
    # 1D slice
    elif isinstance(node.slice, ast.Slice):
        result_slice = [(node.slice.lower, node.slice.upper, node.slice.step)]
    else:  # ND slice
        result_slice = []

        for d in node.slice.dims:
            if isinstance(d, ast.Index):
                result_slice.append(d.value)
            else:
                result_slice.append((d.lower, d.upper, d.step))

    if without_array:
        return result_slice
    else:
        return (rname(node.value), result_slice)


def subscript_to_ast_slice_recursive(node):
    """ Converts an AST subscript to a slice in a recursive manner into nested
        subscripts.
        @see: subscript_to_ast_slice
    """
    result = []
    while isinstance(node, ast.Subscript):
        result.insert(0, subscript_to_ast_slice(node, True))
        node = node.value

    return result


def unparse(node):
    """ Unparses an AST node to a Python string, chomping trailing newline. """
    if node is None:
        return None
    return astunparse.unparse(node).strip()


# Helper function to convert an ND subscript AST node to a list of 3-tuple
# slice strings
def subscript_to_slice(node, arrays, without_array=False):
    """ Converts an AST subscript to slice on the form
        (<name>, [<3-tuples of indices>]). If an ast.Name is passed, return
        (name, None), implying the full range. """

    name, ast_slice = subscript_to_ast_slice(node)
    if name in arrays:
        arrname = name
    else:
        arrname = None

    rng = astrange_to_symrange(ast_slice, arrays, arrname)
    if without_array:
        return rng
    else:
        return name, rng


def astrange_to_symrange(astrange, arrays, arrname=None):
    """ Converts an AST range (array, [(start, end, skip)]) to a symbolic math 
        range, using the obtained array sizes and resolved symbols. """
    if arrname is not None:
        arrdesc = arrays[arrname]

        # If the array is a scalar, return None
        if arrdesc.shape is None:
            return None

        # If range is the entire array, use the array descriptor to obtain the
        # entire range
        if astrange is None:
            return [
                (symbolic.pystr_to_symbolic(0),
                 symbolic.pystr_to_symbolic(symbolic.symbol_name_or_value(s)) -
                 1, symbolic.pystr_to_symbolic(1)) for s in arrdesc.shape
            ]

        missing_slices = len(arrdesc.shape) - len(astrange)
        if missing_slices < 0:
            raise ValueError(
                'Mismatching shape {} - range {} dimensions'.format(
                    arrdesc.shape, astrange))
        for i in range(missing_slices):
            astrange.append((None, None, None))

    result = [None] * len(astrange)
    for i, r in enumerate(astrange):
        if isinstance(r, tuple):
            begin, end, skip = r
            # Default values
            if begin is None:
                begin = symbolic.pystr_to_symbolic(0)
            else:
                begin = symbolic.pystr_to_symbolic(unparse(begin))
            if end is None and arrname is None:
                raise SyntaxError('Cannot define range without end')
            elif end is not None:
                end = symbolic.pystr_to_symbolic(unparse(end)) - 1
            else:
                end = symbolic.pystr_to_symbolic(
                    symbolic.symbol_name_or_value(arrdesc.shape[i])) - 1
            if skip is None:
                skip = symbolic.pystr_to_symbolic(1)
            else:
                skip = symbolic.pystr_to_symbolic(unparse(skip))
        else:
            # In the case where a single element is given
            begin = symbolic.pystr_to_symbolic(unparse(r))
            end = begin
            skip = symbolic.pystr_to_symbolic(1)

        result[i] = (begin, end, skip)

    return result


def negate_expr(node):
    """ Negates an AST expression by adding a `Not` AST node in front of it. 
    """
    from dace.properties import CodeBlock  # Avoid import loop
    if isinstance(node, CodeBlock):
        node = node.code
    if hasattr(node, "__len__"):
        if len(node) > 1:
            raise ValueError("negate_expr only expects "
                             "single expressions, got: {}".format(node))
        expr = node[0]
    else:
        expr = node
    if isinstance(expr, ast.Expr):
        expr = expr.value

    newexpr = ast.Expr(value=ast.UnaryOp(op=ast.Not(), operand=expr))
    newexpr = ast.copy_location(newexpr, expr)
    return ast.fix_missing_locations(newexpr)


class ExtNodeTransformer(ast.NodeTransformer):
    """ A `NodeTransformer` subclass that walks the abstract syntax tree and
        allows modification of nodes. As opposed to `NodeTransformer`,
        this class is capable of traversing over top-level expressions in 
        bodies in order to discern DaCe statements from others.
    """
    def visit_TopLevel(self, node):
        clsname = type(node).__name__
        if getattr(self, "visit_TopLevel" + clsname, False):
            return getattr(self, "visit_TopLevel" + clsname)(node)
        else:
            return self.visit(node)

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        if (field == 'body'
                                or field == 'orelse') and isinstance(
                                    value, ast.Expr):
                            clsname = type(value).__name__
                            if getattr(self, "visit_TopLevel" + clsname, False):
                                value = getattr(self, "visit_TopLevel" +
                                                clsname)(value)
                            else:
                                value = self.visit(value)
                        else:
                            value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class ExtNodeVisitor(ast.NodeVisitor):
    """ A `NodeVisitor` subclass that walks the abstract syntax tree. 
        As opposed to `NodeVisitor`, this class is capable of traversing over 
        top-level expressions in bodies in order to discern DaCe statements 
        from others. """
    def visit_TopLevel(self, node):
        clsname = type(node).__name__
        if getattr(self, "visit_TopLevel" + clsname, False):
            getattr(self, "visit_TopLevel" + clsname)(node)
        else:
            self.visit(node)

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                for value in old_value:
                    if isinstance(value, ast.AST):
                        if (field == 'body' or field == 'orelse'):
                            clsname = type(value).__name__
                            if getattr(self, "visit_TopLevel" + clsname, False):
                                getattr(self, "visit_TopLevel" + clsname)(value)
                            else:
                                self.visit(value)
                        else:
                            self.visit(value)
            elif isinstance(old_value, ast.AST):
                self.visit(old_value)
        return node


class ASTFindReplace(ast.NodeTransformer):
    def __init__(self, repldict: Dict[str, str]):
        self.repldict = repldict

    def visit_Name(self, node: ast.Name):
        if node.id in self.repldict:
            node.id = self.repldict[node.id]
        return node

    def visit_keyword(self, node: ast.keyword):
        if node.arg in self.repldict:
            node.arg = self.repldict[node.arg]
        return self.generic_visit(node)


class TaskletFreeSymbolVisitor(ast.NodeVisitor):
    """ 
    Simple Python AST visitor to find free symbols in a code, not including
    attributes and function calls.
    """
    def __init__(self, defined_syms):
        super().__init__()
        self.free_symbols = set()
        self.defined = set(defined_syms)

    def visit_Call(self, node: ast.Call):
        for arg in node.args:
            self.visit(arg)
        for kwarg in node.keywords:
            self.visit(kwarg)

    def visit_Attribute(self, node):
        pass

    def visit_AnnAssign(self, node):
        # Skip visiting annotation
        self.visit(node.target)
        self.visit(node.value)

    def visit_Name(self, node):
        if (isinstance(node.ctx, ast.Load) and node.id not in self.defined
                and isinstance(node.id, str)):
            self.free_symbols.add(node.id)
        else:
            self.defined.add(node.id)
        self.generic_visit(node)
