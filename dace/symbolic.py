# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from functools import lru_cache
import sympy
import pickle
import re
from typing import Any, Dict, Optional, Set, Tuple, Union
import warnings
import numpy

import sympy.abc
import sympy.printing.str

from dace import dtypes

DEFAULT_SYMBOL_TYPE = dtypes.int32


class symbol(sympy.Symbol):
    """ Defines a symbolic expression. Extends SymPy symbols with DaCe-related
        information. """

    s_currentsymbol = 0

    def __new__(cls, name=None, dtype=DEFAULT_SYMBOL_TYPE, **assumptions):
        if name is None:
            # Set name dynamically
            name = "sym_" + str(symbol.s_currentsymbol)
            symbol.s_currentsymbol += 1
        elif name.startswith('__DACE'):
            raise NameError('Symbols cannot start with __DACE')
        elif not dtypes.validate_name(name):
            raise NameError('Invalid symbol name "%s"' % name)

        if not isinstance(dtype, dtypes.typeclass):
            raise TypeError('dtype must be a DaCe type, got %s' % str(dtype))

        dkeys = [k for k, v in dtypes.DTYPE_TO_TYPECLASS.items() if v == dtype]
        is_integer = [
            issubclass(k, int) or issubclass(k, numpy.integer) for k in dkeys
        ]
        if 'integer' in assumptions or not numpy.any(is_integer):
            # Using __xnew__ as the regular __new__ is cached, which leads
            # to modifying different references of symbols with the same name.
            self = sympy.Symbol.__xnew__(cls, name, **assumptions)
        else:
            self = sympy.Symbol.__xnew__(cls, name, integer=True, **assumptions)

        self.dtype = dtype
        self._constraints = []
        self.value = None
        return self

    def set(self, value):
        warnings.warn('symbol.set is deprecated, use keyword arguments',
                      DeprecationWarning)
        if value is not None:
            # First, check constraints
            self.check_constraints(value)

        self.value = self.dtype(value)

    def __getstate__(self):
        return dict(
            super().__getstate__(), **{
                'value': self.value,
                'dtype': self.dtype,
                '_constraints': self._constraints
            })

    def is_initialized(self):
        return self.value is not None

    def get(self):
        warnings.warn('symbol.get is deprecated, use keyword arguments',
                      DeprecationWarning)
        if self.value is None:
            raise UnboundLocalError('Uninitialized symbol value for \'' +
                                    self.name + '\'')
        return self.value

    def set_constraints(self, constraint_list):
        try:
            iter(constraint_list)
            self._constraints = constraint_list
        except TypeError:  # constraint_list is not iterable
            self._constraints = [constraint_list]

        # Check for the new constraints and reset symbol value if necessary
        if symbol.s_values[self.name] is not None:
            try:
                self.check_constraints(symbol.s_values[self.name])
            except RuntimeError:
                self.reset()  # Reset current value
                raise

    def add_constraints(self, constraint_list):
        try:
            iter(constraint_list)
            symbol.s_constraints[self.name].extend(constraint_list)
        except TypeError:  # constraint_list is not iterable
            symbol.s_constraints[self.name].append(constraint_list)

        # Check for the new constraints and reset symbol value if necessary
        if symbol.s_values[self.name] is not None:
            try:
                self.check_constraints(symbol.s_values[self.name])
            except RuntimeError:
                self.reset()  # Reset current value
                raise

    @property
    def constraints(self):
        return self._constraints

    def check_constraints(self, value):
        fail = None
        for constraint in self.constraints:
            try:
                eval_cons = constraint.subs({self: value})
                if not eval_cons:
                    fail = constraint
                    break
            except (AttributeError, TypeError, ValueError):
                raise RuntimeError(
                    'Cannot validate constraint %s for symbol %s' %
                    (str(constraint), self.name))
        if fail is not None:
            raise RuntimeError(
                'Value %s invalidates constraint %s for symbol %s' %
                (str(value), str(fail), self.name))

    def get_or_return(self, uninitialized_ret):
        return self.value or uninitialized_ret


class SymExpr(object):
    """ Symbolic expressions with support for an overapproximation expression.
    """
    def __init__(self,
                 main_expr: Union[str, 'SymExpr'],
                 approx_expr: Optional[Union[str, 'SymExpr']] = None):
        self._main_expr = pystr_to_symbolic(main_expr)
        if approx_expr is None:
            self._approx_expr = self._main_expr
        else:
            self._approx_expr = pystr_to_symbolic(approx_expr)

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        if len(args) == 2:
            main_expr, approx_expr = args
            # If values are equivalent, create a normal symbolic expression
            if approx_expr is None or main_expr == approx_expr:
                return main_expr
        return super(SymExpr, cls).__new__(cls)

    @property
    def expr(self):
        return self._main_expr

    @property
    def approx(self):
        return self._approx_expr

    def subs(self, repldict):
        return SymExpr(self._main_expr.subs(repldict),
                       self._approx_expr.subs(repldict))

    def match(self, *args, **kwargs):
        return self._main_expr.match(*args, **kwargs)

    def __hash__(self):
        return hash((self.expr, self.approx))

    def __str__(self):
        if self.expr != self.approx:
            return str(self.expr) + " (~" + str(self.approx) + ")"
        else:
            return str(self.expr)

    def __add__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr + other.expr, self.approx + other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr + other, self.approx + other)
        return self + pystr_to_symbolic(other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr - other.expr, self.approx - other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr - other, self.approx - other)
        return self - pystr_to_symbolic(other)

    def __rsub__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(other.expr - self.expr, other.approx - self.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(other - self.expr, other - self.approx)
        return pystr_to_symbolic(other) - self

    def __mul__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr * other.expr, self.approx * other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr * other, self.approx * other)
        return self * pystr_to_symbolic(other)

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr / other.expr, self.approx / other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr / other, self.approx / other)
        return self / pystr_to_symbolic(other)

    __truediv__ = __div__

    def __floordiv__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr // other.expr, self.approx // other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr // other, self.approx // other)
        return self // pystr_to_symbolic(other)

    def __mod__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr % other.expr, self.approx % other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr % other, self.approx % other)
        return self % pystr_to_symbolic(other)

    def __pow__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr**other.expr, self.approx**other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr**other, self.approx**other)
        return self**pystr_to_symbolic(other)

    def __eq__(self, other):
        if isinstance(other, sympy.Expr):
            return self.expr == other
        if isinstance(other, SymExpr):
            return self.expr == other.expr and self.approx == other.approx
        return self == pystr_to_symbolic(other)


# Type hint for symbolic expressions
SymbolicType = Union[sympy.Basic, SymExpr]


def symvalue(val):
    """ Returns the symbol value if it is a symbol. """
    if isinstance(val, symbol):
        return val.get()
    return val


# http://stackoverflow.com/q/3844948/
def _checkEqualIvo(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def symtype(expr):
    """ Returns the inferred symbol type from a symbolic expression. """
    stypes = [s.dtype for s in symlist(expr).values()]
    if len(stypes) == 0:
        return DEFAULT_SYMBOL_TYPE
    elif _checkEqualIvo(stypes):
        return stypes[0]
    else:
        raise TypeError(
            'Cannot infer symbolic type from expression "%s"'
            ' with symbols [%s]' % (str(expr), ', '.join(
                [str(s) + ": " + str(s.dtype) for s in symlist(expr)])))


def symlist(values):
    """ Finds symbol dependencies of expressions. """
    result = {}
    try:
        values = iter(values)
    except TypeError:
        values = [values]

    for expr in values:
        if isinstance(expr, SymExpr):
            true_expr = expr.expr
        elif isinstance(expr, sympy.Basic):
            true_expr = expr
        else:
            continue
        for atom in true_expr.atoms():
            if isinstance(atom, symbol):
                result[atom.name] = atom
    return result


def evaluate(expr: Union[sympy.Basic, int, float],
             symbols: Dict[Union[symbol, str], Union[int, float]]) -> \
        Union[int, float, numpy.number]:
    """
    Evaluates an expression to a constant based on a mapping from symbols
    to values.
    :param expr: The expression to evaluate.
    :param symbols: A mapping of symbols to their values.
    :return: A constant value based on ``expr`` and ``symbols``.
    """
    if isinstance(expr, SymExpr):
        return evaluate(expr.expr, symbols)
    if issymbolic(expr, set(map(str, symbols.keys()))):
        raise TypeError('Expression cannot be evaluated to a constant')
    if isinstance(expr, (int, float, numpy.number)):
        return expr

    # Evaluate all symbols
    syms = {(sname if isinstance(sname, sympy.Symbol) else symbol(sname)):
            sval.get() if isinstance(sval, symbol) else sval
            for sname, sval in symbols.items()}

    return expr.subs(syms)


def issymbolic(value, constants=None):
    """ Returns True if an expression is symbolic with respect to its contents
        and a given dictionary of constant values. """
    constants = constants or {}
    if isinstance(value, SymExpr):
        return issymbolic(value.expr)
    if isinstance(value, symbol) and value.name not in constants:
        return True
    if isinstance(value, sympy.Basic):
        for atom in value.atoms():
            if isinstance(atom, symbol) and atom.name not in constants:
                return True
    return False


def overapproximate(expr):
    """ Takes a sympy expression and returns its maximal possible value
        in specific cases. """
    if isinstance(expr, SymExpr):
        if expr.expr != expr.approx:
            return expr.approx
        else:
            return overapproximate(expr.expr)
    if not isinstance(expr, sympy.Basic):
        return expr
    a = sympy.Wild('a')
    b = sympy.Wild('b')
    c = sympy.Wild('c')

    # If Min(x, N-y), return the non-symbolic of the two components
    match = expr.match(sympy.Min(a, b) + c)
    if match is not None and len(match) == 3:
        # First, construct the min expression with "c" inline
        newexpr = sympy.Min(match[a] + match[c], match[b] + match[c])
        # Match again
        match = newexpr.match(sympy.Min(a, b))
        if match is not None and len(match) == 2:
            if issymbolic(match[a]) and not issymbolic(match[b]):
                return match[b]
            if issymbolic(match[b]) and not issymbolic(match[a]):
                return match[a]

    # If ceiling((k * ((N - 1) / k))) + k), return N
    a = sympy.Wild('a', properties=[lambda k: k.is_Symbol or k.is_Integer])
    b = sympy.Wild('b', properties=[lambda k: k.is_Symbol or k.is_Integer])
    int_floor = sympy.Function('int_floor')
    match = expr.match(sympy.ceiling(b * int_floor(a - 1, b)) + b)
    if match is not None and len(match) == 2:
        return match[a]

    return expr


def resolve_symbol_to_constant(symb, start_sdfg):
    """
    Tries to resolve a symbol to constant, by looking up into SDFG's constants,
    following nested SDFGs hierarchy if necessary.
    :param symb: symbol to resolve to constant
    :param start_sdfg: starting SDFG
    :return: the constant value if the symbol is resolved, None otherwise
    """
    if not issymbolic(symb):
        return symb
    else:
        sdfg = start_sdfg
        while sdfg is not None:
            if not issymbolic(symb, sdfg.constants):
                return evaluate(symb, sdfg.constants)
            else:
                sdfg = sdfg.parent_sdfg
        # can not be resolved
        return None


def symbols_in_ast(tree):
    """ Walks an AST and finds all names, excluding function names. """
    to_visit = list(tree.__dict__.items())
    symbols = []
    while len(to_visit) > 0:
        (key, val) = to_visit.pop()
        if key == "func":
            continue
        if isinstance(val, ast.Name):
            symbols.append(val.id)
            continue
        if isinstance(val, ast.expr):
            to_visit += list(val.__dict__.items())
        if isinstance(val, list):
            to_visit += [(key, v) for v in val]
    return dtypes.deduplicate(symbols)


def symbol_name_or_value(val):
    """ Returns the symbol name if symbol, otherwise the value as a string. """
    if isinstance(val, symbol):
        return val.name
    return str(val)


def sympy_to_dace(exprs, symbol_map=None):
    """ Convert all `sympy.Symbol`s to DaCe symbols, according to
        `symbol_map`. """
    repl = {}
    symbol_map = symbol_map or {}

    oneelem = False
    try:
        iter(exprs)
    except TypeError:
        oneelem = True
        exprs = [exprs]

    exprs = list(exprs)

    for i, expr in enumerate(exprs):
        if isinstance(expr, sympy.Basic):
            for atom in expr.atoms():
                if isinstance(atom, sympy.Symbol):
                    try:
                        repl[atom] = symbol_map[atom.name]
                    except KeyError:
                        # Symbol is not in map, create a DaCe symbol with same assumptions
                        repl[atom] = symbol(atom.name, **atom.assumptions0)
            exprs[i] = expr.subs(repl)
    if oneelem:
        return exprs[0]
    return exprs


def is_sympy_userfunction(expr):
    """ Returns True if the expression is a SymPy function. """
    try:
        return issubclass(type(type(expr)),
                          sympy.core.function.UndefinedFunction)
    except AttributeError:
        return issubclass(type(type(expr)), sympy.function.UndefinedFunction)


def swalk(expr, enter_functions=False):
    """ Walk over a symbolic expression tree (similar to `ast.walk`).
        Returns an iterator that yields the values and recurses into functions,
        if specified.
    """
    yield expr
    for arg in expr.args:
        if not enter_functions and is_sympy_userfunction(arg):
            yield arg
            continue
        yield from swalk(arg)


_builtin_userfunctions = {
    'int_floor', 'int_ceil', 'min', 'Min', 'max', 'Max', 'not', 'Not', 'Eq',
    'NotEq', 'Ne'
}


def contains_sympy_functions(expr):
    """ Returns True if expression contains Sympy functions. """
    if is_sympy_userfunction(expr):
        if str(expr.func) in _builtin_userfunctions:
            return False
        return True
    for arg in expr.args:
        if contains_sympy_functions(arg):
            return True
    return False


def free_symbols_and_functions(expr: Union[SymbolicType, str]) -> Set[str]:
    if not isinstance(expr, (sympy.Basic, str)):
        return set()
    if isinstance(expr, str):
        if dtypes.validate_name(expr):
            return {expr}
        expr = pystr_to_symbolic(expr)

    result = {str(k) for k in expr.free_symbols}
    for atom in swalk(expr):
        if (is_sympy_userfunction(atom)
                and str(atom.func) not in _builtin_userfunctions):
            result.add(str(atom.func))
    return result


def sympy_numeric_fix(expr):
    """ Fix for printing out integers as floats with ".00000000".
        Converts the float constants in a given expression to integers. """
    if not isinstance(expr, sympy.Basic):
        try:
            # NOTE: If expr is ~ 1.8e308, i.e. infinity, `numpy.int64(expr)`
            # will throw OverflowError (which we want).
            # `int(1.8e308) == expr` evaluates unfortunately to True
            # because Python has variable-bit integers.
            if numpy.int64(expr) == expr:
                return int(expr)
        except OverflowError:
            if expr > 0:
                return sympy.oo
            else:
                return -sympy.oo
        return expr

    if isinstance(expr, sympy.Number) and expr == int(expr):
        return int(expr)
    return expr


def sympy_intdiv_fix(expr):
    """ Fix for SymPy printing out reciprocal values when they should be
        integral in "ceiling/floor" sympy functions.
    """
    nexpr = expr
    if not isinstance(expr, sympy.Basic):
        return expr

    # The properties avoid matching the silly case "ceiling(N/32)" as
    # ceiling of 1/N and 1/32
    a = sympy.Wild('a', properties=[lambda k: k.is_Symbol or k.is_Integer])
    b = sympy.Wild('b', properties=[lambda k: k.is_Symbol or k.is_Integer])
    c = sympy.Wild('c')
    d = sympy.Wild('d')
    e = sympy.Wild('e',
                   properties=[
                       lambda k: isinstance(k, sympy.Basic) and not isinstance(
                           k, sympy.Atom)
                   ])
    int_ceil = sympy.Function('int_ceil')
    int_floor = sympy.Function('int_floor')

    processed = 1
    while processed > 0:
        processed = 0
        for ceil in nexpr.find(sympy.ceiling):
            # Simple ceiling
            m = ceil.match(sympy.ceiling(a / b))
            if m is not None:
                nexpr = nexpr.subs(ceil, int_ceil(m[a], m[b]))
                processed += 1
                continue
            # Ceiling of ceiling: "ceil(ceil(c/d) / b)"
            m = ceil.match(sympy.ceiling(int_ceil(c, d) / b))
            if m is not None:
                nexpr = nexpr.subs(ceil, int_ceil(int_ceil(m[c], m[d]), m[b]))
                processed += 1
                continue
            # Ceiling of ceiling: "ceil(a / ceil(c/d))"
            m = ceil.match(sympy.ceiling(a / int_ceil(c, d)))
            if m is not None:
                nexpr = nexpr.subs(ceil, int_ceil(m[a], int_ceil(m[c], m[d])))
                processed += 1
                continue
            # Match ceiling of multiplication with our custom integer functions
            m = ceil.match(sympy.ceiling(a * int_floor(c, d)))
            if m is not None:
                nexpr = nexpr.subs(ceil, m[a] * int_floor(m[c], m[d]))
                processed += 1
                continue
            m = ceil.match(sympy.ceiling(a * int_ceil(c, d)))
            if m is not None:
                nexpr = nexpr.subs(ceil, m[a] * int_ceil(m[c], m[d]))
                processed += 1
                continue
            # Ceiling with composite expression at the numerator
            m = ceil.match(sympy.ceiling(e / b))
            if m is not None:
                nexpr = nexpr.subs(ceil, int_ceil(m[e], m[b]))
                processed += 1
                continue
        for floor in nexpr.find(sympy.floor):
            # Simple floor
            m = floor.match(sympy.floor(a / b))
            if m is not None:
                nexpr = nexpr.subs(floor, int_floor(m[a], m[b]))
                processed += 1
                continue
            # Floor of floor: "floor(floor(c/d) / b)"
            m = floor.match(sympy.floor(int_floor(c, d) / b))
            if m is not None:
                nexpr = nexpr.subs(floor, int_floor(int_floor(m[c], m[d]),
                                                    m[b]))
                processed += 1
                continue
            # Floor of floor: "floor(a / floor(c/d))"
            m = floor.match(sympy.floor(a / int_floor(c, d)))
            if m is not None:
                nexpr = nexpr.subs(floor, int_floor(m[a], int_floor(m[c],
                                                                    m[d])))
                processed += 1
                continue
            # floor with composite expression
            m = floor.match(sympy.floor(e / b))
            if m is not None:
                nexpr = nexpr.subs(floor, int_floor(m[e], m[b]))
                processed += 1
                continue
    return nexpr


def sympy_divide_fix(expr):
    """ Fix SymPy printouts where integer division such as "tid/2" turns
        into ".5*tid".
    """
    nexpr = expr
    if not isinstance(expr, sympy.Basic):
        return expr

    int_floor = sympy.Function('int_floor')

    processed = 1
    while processed > 0:
        processed = 0
        for candidate in nexpr.find(sympy.Mul):
            for i, arg in enumerate(candidate.args):
                if isinstance(arg, sympy.Number) and abs(arg) >= 1:
                    continue
                if isinstance(arg, sympy.Number) and (1 / arg) == int(1 / arg):
                    ri = i
                    break
            else:
                continue
            nexpr = nexpr.subs(
                candidate,
                int_floor(
                    sympy.Mul(*(candidate.args[:ri] + candidate.args[ri + 1:])),
                    int(1 / candidate.args[ri])))
            processed += 1

    return nexpr


def simplify_ext(expr):
    """
    An extended version of simplification with expression fixes for sympy.
    :param expr: A sympy expression.
    :return: Simplified version of the expression.
    """
    if not isinstance(expr, sympy.Basic):
        return expr
    a = sympy.Wild('a')
    b = sympy.Wild('b')
    c = sympy.Wild('c')

    # Push expressions into both sides of min/max.
    # Example: Min(N, 4) + 1 => Min(N + 1, 5)
    dic = expr.match(sympy.Min(a, b) + c)
    if dic:
        return sympy.Min(dic[a] + dic[c], dic[b] + dic[c])
    dic = expr.match(sympy.Max(a, b) + c)
    if dic:
        return sympy.Max(dic[a] + dic[c], dic[b] + dic[c])
    return expr


class SympyBooleanConverter(ast.NodeTransformer):
    """ 
    Replaces boolean operations with the appropriate SymPy functions to avoid
    non-symbolic evaluation.
    """
    _ast_to_sympy_comparators = {
        ast.Eq: 'Eq',
        ast.Gt: 'Gt',
        ast.GtE: 'Ge',
        ast.Lt: 'Lt',
        ast.LtE: 'Le',
        ast.NotEq: 'Ne',
        # Python-specific
        ast.In: 'In',
        ast.Is: 'Is',
        ast.IsNot: 'IsNot',
        ast.NotIn: 'NotIn',
    }

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            func_node = ast.copy_location(
                ast.Name(id=type(node.op).__name__, ctx=ast.Load()), node)
            new_node = ast.Call(func=func_node,
                                args=[self.visit(node.operand)],
                                keywords=[])
            return ast.copy_location(new_node, node)
        return node

    def visit_BoolOp(self, node):
        func_node = ast.copy_location(
            ast.Name(id=type(node.op).__name__, ctx=ast.Load()), node)
        new_node = ast.Call(func=func_node,
                            args=[self.visit(value) for value in node.values],
                            keywords=[])
        return ast.copy_location(new_node, node)

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise NotImplementedError
        op = node.ops[0]
        arguments = [node.left, node.comparators[0]]
        func_node = ast.copy_location(
            ast.Name(
                id=SympyBooleanConverter._ast_to_sympy_comparators[type(op)],
                ctx=ast.Load()), node)
        new_node = ast.Call(func=func_node,
                            args=[self.visit(arg) for arg in arguments],
                            keywords=[])
        return ast.copy_location(new_node, node)


@lru_cache(2048)
def pystr_to_symbolic(expr, symbol_map=None, simplify=None):
    """ Takes a Python string and converts it into a symbolic expression. """
    from dace.frontend.python.astutils import unparse  # Avoid import loops

    if isinstance(expr, (SymExpr, sympy.Basic)):
        return expr
    if isinstance(expr, str) and dtypes.validate_name(expr):
        return symbol(expr)

    symbol_map = symbol_map or {}
    locals = {
        'min': sympy.Min,
        'max': sympy.Max,
        'True': sympy.true,
        'False': sympy.false,
        'GtE': sympy.Ge,
        'LtE': sympy.Le,
        'NotEq': sympy.Ne,
    }
    # _clash1 enables all one-letter variables like N as symbols
    # _clash also allows pi, beta, zeta and other common greek letters
    locals.update(sympy.abc._clash)

    # Sympy processes "not/and/or" as direct evaluation. Replace with
    # And/Or(x, y), Not(x)
    if isinstance(expr, str) and re.search(r'\bnot\b|\band\b|\bor\b|==|!=',
                                           expr):
        expr = unparse(SympyBooleanConverter().visit(ast.parse(expr).body[0]))

    # TODO: support SymExpr over-approximated expressions
    try:
        return sympy_to_dace(sympy.sympify(expr, locals, evaluate=simplify),
                             symbol_map)
    except TypeError:  # Symbol object is not subscriptable
        # Replace subscript expressions with function calls
        expr = expr.replace('[', '(')
        expr = expr.replace(']', ')')
        return sympy_to_dace(sympy.sympify(expr, locals, evaluate=simplify),
                             symbol_map)


@lru_cache(maxsize=2048)
def simplify(expr: SymbolicType) -> SymbolicType:
    return sympy.simplify(expr)


class DaceSympyPrinter(sympy.printing.str.StrPrinter):
    """ Several notational corrections for integer math and C++ translation
        that sympy.printing.cxxcode does not provide. """
    def __init__(self, arrays, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = arrays or set()

    def _print_Float(self, expr):
        if int(expr) == expr:
            return str(int(expr))
        return super()._print_Float(expr)

    def _print_Function(self, expr):
        if str(expr.func) in self.arrays:
            return f'{expr.func}[{expr.args[0]}]'
        if str(expr.func) == 'int_floor':
            return '((%s) / (%s))' % (self._print(
                expr.args[0]), self._print(expr.args[1]))
        return super()._print_Function(expr)

    def _print_Mod(self, expr):
        return '((%s) %% (%s))' % (self._print(
            expr.args[0]), self._print(expr.args[1]))

    def _print_Equality(self, expr):
        return '((%s) == (%s))' % (self._print(
            expr.args[0]), self._print(expr.args[1]))

    def _print_Unequality(self, expr):
        return '((%s) != (%s))' % (self._print(
            expr.args[0]), self._print(expr.args[1]))

    def _print_Not(self, expr):
        return '(not (%s))' % self._print(expr.args[0])

    def _print_Infinity(self, expr):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    def _print_Pow(self, expr):
        base = self._print(expr.args[0])
        exponent = self._print(expr.args[1])
        try:
            int_exp = int(exponent)
            assert(int_exp > 0)
            res = "({})".format(base)
            for _ in range(1, int_exp):
                res += "*{}".format(base)
            return res
        except ValueError:
            return "dace::math::pow({f}, {s})".format(
                f=self._print(expr.args[0]), s=self._print(expr.args[1]))


def symstr(sym, arrayexprs: Optional[Set[str]] = None) -> str:
    """ 
    Convert a symbolic expression to a C++ compilable expression. 
    :param sym: Symbolic expression to convert.
    :param arrayexprs: Set of names of arrays, used to convert SymPy 
                       user-functions back to array expressions.
    :return: C++-compilable expression.
    """
    def repstr(s):
        return s.replace('Min', 'min').replace('Max', 'max')

    if isinstance(sym, SymExpr):
        return symstr(sym.expr, arrayexprs)

    try:
        sym = sympy_numeric_fix(sym)
        sym = sympy_intdiv_fix(sym)
        sym = sympy_divide_fix(sym)

        sstr = DaceSympyPrinter(arrayexprs).doprint(sym)

        if isinstance(sym,
                      symbol) or isinstance(sym, sympy.Symbol) or isinstance(
                          sym, sympy.Number) or dtypes.isconstant(sym):
            return repstr(sstr)
        else:
            return '(' + repstr(sstr) + ')'
    except (AttributeError, TypeError, ValueError):
        sstr = DaceSympyPrinter(arrayexprs).doprint(sym)
        return '(' + repstr(sstr) + ')'


def _spickle(obj):
    return str(obj), {
        s.name: (s.dtype, s._assumptions)
        for s in symlist(obj).values()
    }


def _sunpickle(obj):
    s, slist = obj
    # Create symbols
    for sname, (stype, assumptions) in slist.items():
        symbol(sname, stype, **assumptions)
    return pystr_to_symbolic(s)


class SympyAwarePickler(pickle.Pickler):
    """
    Custom Pickler class that safely saves SymPy expressions
    with function definitions in expressions (e.g., int_ceil).
    """
    def persistent_id(self, obj):
        if isinstance(obj, sympy.Basic):
            # Save sympy expression as srepr
            return ("DaCeSympyExpression", _spickle(obj))
        else:
            # Pickle everything else normally
            return None


class SympyAwareUnpickler(pickle.Unpickler):
    """
    Custom Unpickler class that safely restores SymPy expressions
    with function definitions in expressions (e.g., int_ceil).
    """
    def persistent_load(self, pid):
        type_tag, value = pid
        if type_tag == "DaCeSympyExpression":
            return _sunpickle(value)
        else:
            raise pickle.UnpicklingError("unsupported persistent object")


def equalize_symbol(sym: sympy.Expr) -> sympy.Expr:
    """
    If a symbol or symbolic expressions has multiple symbols with the same
    name, it substitutes them with the last symbol (as they appear in
    s.free_symbols).
    """
    symdict = {s.name: s for s in sym.free_symbols}
    repldict = {s: symdict[s.name] for s in sym.free_symbols}
    return sym.subs(repldict)


def equalize_symbols(a: sympy.Expr,
                     b: sympy.Expr) -> Tuple[sympy.Expr, sympy.Expr]:
    """
    If the 2 input expressions use different symbols but with the same name,
    it substitutes the symbols of the second expressions with those of the
    first expression.
    """
    a = equalize_symbol(a)
    b = equalize_symbol(b)
    a_syms = {s.name: s for s in a.free_symbols}
    b_syms = {s.name: s for s in b.free_symbols}
    common_names = set(a_syms.keys()).intersection(set(b_syms.keys()))
    if common_names:
        repldict = dict()
        for name in common_names:
            repldict[b_syms[name]] = a_syms[name]
        b = b.subs(repldict)
    return a, b


def inequal_symbols(a: Union[sympy.Expr, Any], b: Union[sympy.Expr,
                                                        Any]) -> bool:
    """
    Compares 2 symbolic expressions and returns True if they are not equal.
    """
    if not isinstance(a, sympy.Expr) or not isinstance(b, sympy.Expr):
        return a != b
    else:
        a, b = equalize_symbols(a, b)
        # NOTE: We simplify in an attempt to remove inconvenient methods, such
        # as `ceiling` and `floor`, if the symbol assumptions allow it.
        # We subtract and compare to zero according to the SymPy documentation
        # (https://docs.sympy.org/latest/tutorial/gotchas.html).
        return (a - b).simplify() != 0
