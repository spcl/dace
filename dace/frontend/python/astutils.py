# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Various AST parsing utilities for DaCe. """
import ast
import astunparse
import copy
from collections import OrderedDict
from io import StringIO
import inspect
import numbers
import numpy
import sympy
import sys
from typing import Any, Dict, List, Optional, Set, Union

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
        src_file = inspect.getfile(f)
        _, src_line = inspect.findsource(f)
    # TypeError: X is not a module, class, method, function, traceback, frame,
    # or code object; OR OSError: could not get source code
    except (TypeError, OSError):
        # Try to import dill to obtain code from compiled functions
        try:
            import dill
            src = dill.source.getsource(f)
            src_file = '<interpreter>'
            src_line = 0
        except (ImportError, ModuleNotFoundError, TypeError, OSError):
            raise TypeError('Cannot obtain source code for dace program. This may '
                            'happen if you are using the "python" default '
                            'interpreter. Please either use the "ipython" '
                            'interpreter, a Jupyter or Colab notebook, or place '
                            'the source code in a file and import it.')

    src_ast = ast.parse(_remove_outer_indentation(src))
    ast.increment_lineno(src_ast, src_line)

    return src_ast, src_file, src_line, src


def is_constant(node: ast.AST) -> bool:
    """
    Returns True iff the AST node is a constant value
    """
    if sys.version_info >= (3, 8):
        if isinstance(node, ast.Constant):
            return True
    if isinstance(node, (ast.Num, ast.Str, ast.NameConstant)):  # For compatibility
        return True
    return False


def evalnode(node: ast.AST, gvars: Dict[str, Any]) -> Any:
    """
    Tries to evaluate an AST node given only global variables.

    :param node: The AST node/subtree to evaluate.
    :param gvars: A dictionary mapping names to variables.
    :return: The result of evaluation, or raises ``SyntaxError`` on any
             failure to evaluate.
    """
    if not isinstance(node, ast.AST):
        return node
    if isinstance(node, ast.Index):  # For compatibility
        node = node.value
    if isinstance(node, ast.Num):  # For compatibility
        return node.n
    if sys.version_info >= (3, 8):
        if isinstance(node, ast.Constant):
            return node.value

    # Replace internal constants with their values
    node = copy_tree(node)
    cext = ConstantExtractor(gvars)
    cext.visit(node)
    gvars = copy.copy(gvars)
    gvars.update(cext.gvars)
    node = ast.fix_missing_locations(node)

    try:
        # Ensure context is load so eval works (e.g., when using value as lhs)
        if not isinstance(getattr(node, 'ctx', False), ast.Load):
            node.ctx = ast.Load()
        return eval(compile(ast.Expression(node), '<string>', mode='eval'), gvars)
    except:  # Anything can happen here
        raise SyntaxError


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
        if not isinstance(node.func, ast.Attribute):
            raise TypeError(f'Unexpected call expression type {type(node.func).__name__}: {unparse(node.func)}')
        name = node.func.attr
        # Handle calls with submodules and methods, e.g. numpy.add.reduce
        value = node.func.value
        while isinstance(value, ast.Attribute):
            name = value.attr + '.' + name
            value = value.value
        if isinstance(value, ast.Name):
            name = value.id + '.' + name
        else:
            raise NotImplementedError("Unsupported AST {n} node nested inside "
                                      "AST call node: {s}".format(n=type(value), s=unparse(value)))
        return name
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

    # Python <3.9 compatibility
    result_slice = None
    if isinstance(node.slice, ast.Index):
        slc = node.slice.value
        if not isinstance(slc, ast.Tuple):
            result_slice = [slc]
    elif isinstance(node.slice, ast.ExtSlice):
        slc = tuple(node.slice.dims)
    else:
        slc = node.slice

    # Decode slice tuple
    if result_slice is None:
        if isinstance(slc, ast.Tuple):
            slices = slc.elts
        elif isinstance(slc, tuple):
            slices = slc
        else:
            slices = [slc]
        result_slice = []
        for s in slices:
            # Slice
            if isinstance(s, ast.Slice):
                result_slice.append((s.lower, s.upper, s.step))
            elif isinstance(s, ast.Index):  # Index (Python <3.9)
                result_slice.append(s.value)
            else:  # Index
                result_slice.append(s)

    if without_array:
        return result_slice
    else:
        return (rname(node.value), result_slice)


def subscript_to_ast_slice_recursive(node):
    """ Converts an AST subscript to a slice in a recursive manner into nested
        subscripts.
        
        :see: subscript_to_ast_slice
    """
    result = []
    while isinstance(node, ast.Subscript):
        result.insert(0, subscript_to_ast_slice(node, True))
        node = node.value

    return result


class ExtUnparser(astunparse.Unparser):
    def _Subscript(self, t):
        self.dispatch(t.value)
        self.write('[')
        # Compatibility
        if isinstance(t.slice, ast.Index):
            slc = t.slice.value
        else:
            slc = t.slice
        # Get rid of the tuple parentheses in expressions like "A[(i, j)]""
        if isinstance(slc, ast.Tuple):
            for elt in slc.elts[:-1]:
                self.dispatch(elt)
                self.write(', ')
            self.dispatch(slc.elts[-1])
        else:
            self.dispatch(slc)
        self.write(']')


def unparse(node):
    """ Unparses an AST node to a Python string, chomping trailing newline. """
    if node is None:
        return None
    if isinstance(node, (ast.AST, list)):
        v = StringIO()
        ExtUnparser(node, file=v)
        return v.getvalue().strip()
    # Support for SymPy expressions
    if isinstance(node, sympy.Basic):
        return sympy.printing.pycode(node)
    # Support for numerical constants
    if isinstance(node, (numbers.Number, numpy.bool_)):
        return str(node)
    # Suport for string
    if isinstance(node, str):
        return node

    raise TypeError(f'Unsupported type "{type(node).__name__}" for unparse')


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


def slice_to_subscript(arrname, range):
    """ Converts a name and subset to a Python AST Subscript object. """
    return ast.parse(f'{arrname}[{range}]').body[0].value


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
            return [(symbolic.pystr_to_symbolic(0), symbolic.pystr_to_symbolic(symbolic.symbol_name_or_value(s)) - 1,
                     symbolic.pystr_to_symbolic(1)) for s in arrdesc.shape]

        missing_slices = len(arrdesc.shape) - len(astrange)
        if missing_slices < 0:
            raise ValueError('Mismatching shape {} - range {} dimensions'.format(arrdesc.shape, astrange))
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
                if (begin < 0) == True:
                    begin += arrdesc.shape[i]
            if end is None and arrname is None:
                raise SyntaxError('Cannot define range without end')
            elif end is not None:
                end = symbolic.pystr_to_symbolic(unparse(end)) - 1
                if (end < 0) == True:
                    end += arrdesc.shape[i]
            else:
                end = symbolic.pystr_to_symbolic(symbolic.symbol_name_or_value(arrdesc.shape[i])) - 1
            if skip is None:
                skip = symbolic.pystr_to_symbolic(1)
            else:
                skip = symbolic.pystr_to_symbolic(unparse(skip))
        else:
            # In the case where a single element is given
            begin = symbolic.pystr_to_symbolic(unparse(r))
            if (begin < 0) == True:
                begin += arrdesc.shape[i]
            end = begin
            skip = symbolic.pystr_to_symbolic(1)

        result[i] = (begin, end, skip)

    return result


def negate_expr(node):
    """ Negates an AST expression by adding a `Not` AST node in front of it. 
    """

    # Negation support for SymPy expressions
    if isinstance(node, sympy.Basic):
        return sympy.Not(node)
    # Support for numerical constants
    if isinstance(node, (numbers.Number, numpy.bool_)):
        return str(not node)
    # Negation support for strings (most likely dace.Data.Scalar names)
    if isinstance(node, str):
        return "not ({})".format(node)

    from dace.properties import CodeBlock  # Avoid import loop
    if isinstance(node, CodeBlock):
        node = node.code
    if hasattr(node, "__len__"):
        if len(node) > 1:
            raise ValueError("negate_expr only expects " "single expressions, got: {}".format(node))
        expr = node[0]
    else:
        expr = node
    if isinstance(expr, ast.Expr):
        expr = expr.value

    newexpr = ast.Expr(value=ast.UnaryOp(op=ast.Not(), operand=copy_tree(expr)))
    newexpr = ast.copy_location(newexpr, expr)
    return ast.fix_missing_locations(newexpr)


def copy_tree(node: ast.AST) -> ast.AST:
    """
    Copies an entire AST without copying the non-AST parts (e.g., constant values).
    A form of reduced deepcopy.

    :param node: The tree to copy.
    :return: The copied tree.
    """
    class Copier(ast.NodeTransformer):
        def visit_Num(self, node):
            # Ignore n
            return ast.copy_location(ast.Num(n=node.n), node)

        def visit_Constant(self, node):
            # Ignore value
            return ast.copy_location(ast.Constant(value=node.value, kind=node.kind), node)

        def visit(self, node):
            if node.__class__.__name__ in {'Num', 'Constant'}:
                method = 'visit_' + node.__class__.__name__
                visitor = getattr(self, method, self.generic_visit)
                return visitor(node)
            newnode = copy.copy(node)
            return self.generic_visit(newnode)

        def generic_visit(self, node):
            for field, old_value in ast.iter_fields(node):
                if isinstance(old_value, list):
                    new_values = []
                    for value in old_value:
                        if isinstance(value, ast.AST):
                            value = self.visit(value)
                            if value is None:
                                continue
                            elif not isinstance(value, ast.AST):
                                new_values.extend(value)
                                continue
                        new_values.append(value)
                    # Copy list instead of modifying it in-place
                    setattr(node, field, new_values)
                elif isinstance(old_value, ast.AST):
                    new_node = self.visit(old_value)
                    if new_node is None:
                        delattr(node, field)
                    else:
                        setattr(node, field, new_node)
            return node

    return Copier().visit(node)


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
                        if (field == 'body' or field == 'orelse') and isinstance(value, ast.Expr):
                            value = self.visit_TopLevel(value)
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


class NameFound(Exception):
    pass


class ASTFindReplace(ast.NodeTransformer):
    def __init__(self, repldict: Dict[str, str], trigger_names: Set[str] = None):
        self.replace_count = 0
        self.repldict = repldict
        self.trigger_names = trigger_names or set()
        # If ast.Names were given, use them as keys as well
        self.repldict.update({k.id: v for k, v in self.repldict.items() if isinstance(k, ast.Name)})

    def visit_Name(self, node: ast.Name):
        if node.id in self.trigger_names:
            raise NameFound(node.id)
        if node.id in self.repldict:
            val = self.repldict[node.id]
            if isinstance(val, ast.AST):
                new_node = ast.copy_location(val, node)
            else:
                new_node = ast.copy_location(ast.parse(str(self.repldict[node.id])).body[0].value, node)
            self.replace_count += 1
            return new_node

        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg in self.trigger_names:
            raise NameFound(node.arg)
        if node.arg in self.repldict:
            val = self.repldict[node.arg]
            if isinstance(val, ast.AST):
                val = unparse(val)
            node.arg = val
            self.replace_count += 1
        return self.generic_visit(node)


class RemoveSubscripts(ast.NodeTransformer):
    def __init__(self, keywords: Set[str]):
        self.keywords = keywords

    def visit_Subscript(self, node: ast.Subscript):
        if rname(node) in self.keywords:
            return ast.copy_location(node.value, node)

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
        if node.value is not None:
            self.visit(node.value)

    def visit_Name(self, node):
        if (isinstance(node.ctx, ast.Load) and node.id not in self.defined and isinstance(node.id, str)
                and node.id not in ('inf', 'nan')):
            self.free_symbols.add(node.id)
        else:
            self.defined.add(node.id)
        self.generic_visit(node)


class AnnotateTopLevel(ExtNodeTransformer):
    def visit_TopLevel(self, node):
        node.toplevel = True
        return super().visit_TopLevel(node)


class ConstantExtractor(ast.NodeTransformer):
    def __init__(self, globals: Dict[str, Any]):
        super().__init__()
        self.id = 0
        self.globals = globals
        self.gvars: Dict[str, Any] = {}

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.globals and self.globals[node.id] is SyntaxError:
            raise SyntaxError
        return self.generic_visit(node)

    def visit_Constant(self, node):
        return self.visit_Num(node)

    def visit_Num(self, node: ast.Num):
        newname = f'__uu{self.id}'
        self.gvars[newname] = node.n
        self.id += 1
        return ast.copy_location(ast.Name(id=newname, ctx=ast.Load()), node)


class ASTHelperMixin:
    """ A mixin that adds useful helper functions for AST node transformers and visitors """
    def generic_visit_filtered(self, node: ast.AST, filter: Optional[Set[str]] = None):
        """
        Modification of ast.NodeTransformer.generic_visit that visits all fields without the
        set of filtered fields.

        :param node: AST to visit.
        :param filter: Set of strings of fields to skip.
        """
        filter = filter or set()
        for field, old_value in ast.iter_fields(node):
            if field in filter:
                continue
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
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

    def generic_visit_field(self, node: ast.AST, field: str) -> ast.AST:
        """
        Modification of ast.NodeTransformer.generic_visit that only visits one
        field.

        :param node: AST to visit.
        :param field: Field to visit.
        """
        old_value = getattr(node, field)
        if isinstance(old_value, list):
            new_values = []
            for value in old_value:
                if isinstance(value, ast.AST):
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


def create_constant(value: Any, node: Optional[ast.AST] = None) -> ast.AST:
    """
    Cross-Python-AST-version helper function that creates an AST constant node from a given value.

    :param value: The value to create a constant from.
    :param node: An optional node to copy the source location information from.
    :return: An AST node (``ast.Constant`` after Python 3.8) that represents this value.
    """
    if sys.version_info >= (3, 8):
        newnode = ast.Constant(value=value, kind='')
    else:
        if value is None:
            newnode = ast.NameConstant(value=None)
        elif isinstance(value, str):
            newnode = ast.Str(s=value)
        else:
            newnode = ast.Num(n=value)

    if node is not None:
        newnode = ast.copy_location(newnode, node)

    return newnode


def escape_string(value: Union[bytes, str]):
    """
    Converts special Python characters in strings back to their parsable version (e.g., newline to ``\\n``)
    """
    if isinstance(value, bytes):
        return f"{chr(0xFFFF)}{value.decode('utf-8')}"
    if sys.version_info >= (3, 0):
        return value.encode("unicode_escape").decode("utf-8")
    # Python 2.x
    return value.encode('string_escape')


def parse_function_arguments(node: ast.Call, argnames: List[str]) -> Dict[str, ast.AST]:
    """
    Parses function arguments (both positional and keyword) from a Call node,
    based on the function's argument names. If an argument was not given, it will
    not be in the result.
    """
    result = {}
    for arg, aname in zip(node.args, argnames):
        result[aname] = arg
    for kw in node.keywords:
        result[kw.arg] = kw.value
    return result
