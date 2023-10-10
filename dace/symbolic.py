# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from functools import lru_cache
import sympy
import pickle
import re
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
import warnings
import numpy

import sympy.abc
import sympy.printing.str

from dace import dtypes

DEFAULT_SYMBOL_TYPE = dtypes.int32
_NAME_TOKENS = re.compile(r'[a-zA-Z_][a-zA-Z_0-9]*')

# NOTE: Up to (including) version 1.8, sympy.abc._clash is a dictionary of the
# form {'N': sympy.abc.N, 'I': sympy.abc.I, 'pi': sympy.abc.pi}
# Since version 1.9, the values of this dictionary are None. In the dictionary
# below, we recreate it to be as in versions < 1.9.
_sympy_clash = {k: v if v else getattr(sympy.abc, k) for k, v in sympy.abc._clash.items()}


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
        is_integer = [issubclass(k, int) or issubclass(k, numpy.integer) for k in dkeys]
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
        warnings.warn('symbol.set is deprecated, use keyword arguments', DeprecationWarning)
        if value is not None:
            # First, check constraints
            self.check_constraints(value)

        self.value = self.dtype(value)

    def __getstate__(self):
        return dict(self.assumptions0, **{'value': self.value, 'dtype': self.dtype, '_constraints': self._constraints})

    def _eval_subs(self, old, new):
        """
        From sympy: Override this stub if you want to do anything more than
        attempt a replacement of old with new in the arguments of self.

        See also
        ========

        _subs
        """
        try:
            # Compare DaCe symbols by name rather than type/assumptions
            if self.name == old.name:
                return new
            return None
        except AttributeError:
            return None

    def is_initialized(self):
        return self.value is not None

    def get(self):
        warnings.warn('symbol.get is deprecated, use keyword arguments', DeprecationWarning)
        if self.value is None:
            raise UnboundLocalError('Uninitialized symbol value for \'' + self.name + '\'')
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
                raise RuntimeError('Cannot validate constraint %s for symbol %s' % (str(constraint), self.name))
        if fail is not None:
            raise RuntimeError('Value %s invalidates constraint %s for symbol %s' % (str(value), str(fail), self.name))

    def get_or_return(self, uninitialized_ret):
        return self.value or uninitialized_ret


class SymExpr(object):
    """ Symbolic expressions with support for an overapproximation expression.
    """

    def __init__(self, main_expr: Union[str, 'SymExpr'], approx_expr: Optional[Union[str, 'SymExpr']] = None):
        self._main_expr = pystr_to_symbolic(main_expr)
        if approx_expr is None:
            self._approx_expr = self._main_expr
        else:
            self._approx_expr = pystr_to_symbolic(approx_expr)

    def __new__(cls, *args, **kwargs):
        main_expr, approx_expr = None, None
        if len(args) == 0:
            if 'main_expr' in kwargs:
                main_expr = kwargs['main_expr']
            if 'approx_expr' in kwargs:
                approx_expr = kwargs['approx_expr']
        if len(args) == 1:
            main_expr = args[0]
            if 'approx_expr' in kwargs:
                approx_expr = kwargs['approx_expr']
        if len(args) == 2:
            main_expr, approx_expr = args
        # If values are equivalent, create a normal symbolic expression
        if main_expr and (approx_expr is None or main_expr == approx_expr):
            if isinstance(main_expr, str):
                return pystr_to_symbolic(main_expr)
            return main_expr
        return super(SymExpr, cls).__new__(cls)

    @property
    def expr(self):
        return self._main_expr

    @property
    def approx(self):
        return self._approx_expr

    def subs(self, repldict):
        return SymExpr(self._main_expr.subs(repldict), self._approx_expr.subs(repldict))

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

    def __lt__(self, other):
        if isinstance(other, sympy.Expr):
            return self.expr < other
        if isinstance(other, SymExpr):
            return self.expr < other.expr
        return self < pystr_to_symbolic(other)

    def __gt__(self, other):
        if isinstance(other, sympy.Expr):
            return self.expr > other
        if isinstance(other, SymExpr):
            return self.expr > other.expr
        return self > pystr_to_symbolic(other)


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
        raise TypeError('Cannot infer symbolic type from expression "%s"'
                        ' with symbols [%s]' %
                        (str(expr), ', '.join([str(s) + ": " + str(s.dtype) for s in symlist(expr)])))


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
        for atom in sympy.preorder_traversal(true_expr):
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
    if isinstance(expr, list):
        return [evaluate(e, symbols) for e in expr]
    if isinstance(expr, tuple):
        return tuple(evaluate(e, symbols) for e in expr)
    if isinstance(expr, SymExpr):
        return evaluate(expr.expr, symbols)
    if issymbolic(expr, set(map(str, symbols.keys()))):
        raise TypeError(f'Symbolic expression "{expr}" cannot be evaluated to a constant')
    if isinstance(expr, (int, float, numpy.number)):
        return expr

    # Evaluate all symbols
    syms = {(sname if isinstance(sname, sympy.Symbol) else symbol(sname)):
            sval.get() if isinstance(sval, symbol) else sval
            for sname, sval in symbols.items()}

    # Filter out `None` values, callables, and iterables but not strings (for SymPy 1.12)
    syms = {
        k: v
        for k, v in syms.items() if not (v is None or isinstance(v, (Callable, Iterable))) or isinstance(v, str)
    }
    # Convert strings to SymPy symbols (for SymPy 1.12)
    syms = {k: sympy.Symbol(v) if isinstance(v, str) else v for k, v in syms.items()}

    return expr.subs(syms)


def issymbolic(value, constants=None):
    """ Returns True if an expression is symbolic with respect to its contents
        and a given dictionary of constant values. """
    constants = constants or {}
    if isinstance(value, SymExpr):
        return issymbolic(value.expr)
    if isinstance(value, (sympy.Symbol, symbol)) and value.name not in constants:
        return True
    if isinstance(value, sympy.Basic):
        for atom in value.atoms():
            if isinstance(atom, (sympy.Symbol, symbol)) and atom.name not in constants:
                return True
    return False


def overapproximate(expr):
    """
    Takes a sympy expression and returns its maximal possible value
    in specific cases.
    """
    if isinstance(expr, list):
        return [overapproximate(elem) for elem in expr]
    return _overapproximate(expr)


@lru_cache(maxsize=2048)
def _overapproximate(expr):
    if isinstance(expr, SymExpr):
        if expr.expr != expr.approx:
            return expr.approx
        else:
            return overapproximate(expr.expr)
    if not isinstance(expr, sympy.Basic):
        return expr
    if isinstance(expr, sympy.Number):
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


def symbols_in_ast(tree: ast.AST):
    """ Walks an AST and finds all names, excluding function names. """
    symbols = []
    skip = set()
    for node in ast.walk(tree):
        if node in skip:
            continue
        if isinstance(node, ast.Call):
            skip.add(node.func)
        if isinstance(node, ast.Name):
            symbols.append(node.id)
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
        return issubclass(type(type(expr)), sympy.core.function.UndefinedFunction)
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
    'int_floor', 'int_ceil', 'abs', 'Abs', 'min', 'Min', 'max', 'Max', 'not', 'Not', 'Eq', 'NotEq', 'Ne', 'AND', 'OR',
    'pow', 'round'
}


def contains_sympy_functions(expr):
    """ Returns True if expression contains Sympy functions. """
    if is_sympy_userfunction(expr):
        if str(expr.func) in _builtin_userfunctions:
            return False
        return True
    if not isinstance(expr, sympy.Basic):
        return False
    for arg in expr.args:
        if contains_sympy_functions(arg):
            return True
    return False


def free_symbols_and_functions(expr: Union[SymbolicType, str]) -> Set[str]:
    if isinstance(expr, str):
        if dtypes.validate_name(expr):
            return {expr}
        expr = pystr_to_symbolic(expr)
    if not isinstance(expr, sympy.Basic):
        return set()

    result = {str(k) for k in expr.free_symbols}
    for atom in swalk(expr):
        if (is_sympy_userfunction(atom) and str(atom.func) not in _builtin_userfunctions):
            result.add(str(atom.func))
    return result


def sympy_numeric_fix(expr):
    """ Fix for printing out integers as floats with ".00000000".
        Converts the float constants in a given expression to integers. """
    if not isinstance(expr, sympy.Basic) or isinstance(expr, sympy.Number):
        try:
            # NOTE: If expr is ~ 1.8e308, i.e. infinity, `numpy.int64(expr)`
            # will throw OverflowError (which we want).
            # `int(1.8e308) == expr` evaluates unfortunately to True
            # because Python has variable-bit integers.
            if numpy.int64(expr) == expr:
                return int(expr)
        except OverflowError:
            try:
                if numpy.float64(expr) == expr:
                    return expr
            except OverflowError:
                if expr > 0:
                    return sympy.oo
                else:
                    return -sympy.oo
    return expr


class int_floor(sympy.Function):

    @classmethod
    def eval(cls, x, y):
        """
        Evaluates integer floor division (``floor(x / y)``).

        :param x: Numerator.
        :param y: Denominator.
        :return: Return value (literal or symbolic).
        """
        if x.is_Number and y.is_Number:
            return x // y

    def _eval_is_integer(self):
        return True


class int_ceil(sympy.Function):

    @classmethod
    def eval(cls, x, y):
        """
        Evaluates integer ceiling division (``ceil(x / y)``).

        :param x: Numerator.
        :param y: Denominator.
        :return: Return value (literal or symbolic).
        """
        if x.is_Number and y.is_Number:
            return sympy.ceiling(x / y)

    def _eval_is_integer(self):
        return True


class OR(sympy.Function):

    @classmethod
    def eval(cls, x, y):
        """
        Evaluates logical or.

        :param x: First operand.
        :param y: Second operand.
        :return: Return value (literal or symbolic).
        """
        if x.is_Boolean and y.is_Boolean:
            return x or y

    def _eval_is_boolean(self):
        return True


class AND(sympy.Function):

    @classmethod
    def eval(cls, x, y):
        """
        Evaluates logical and.

        :param x: First operand.
        :param y: Second operand.
        :return: Return value (literal or symbolic).
        """
        if x.is_Boolean and y.is_Boolean:
            return x and y

    def _eval_is_boolean(self):
        return True


class IfExpr(sympy.Function):

    @classmethod
    def eval(cls, x, y, z):
        """
        Evaluates a ternary operator.

        :param x: Predicate.
        :param y: If true return this.
        :param z: If false return this.
        :return: Return value (literal or symbolic).
        """
        if x.is_Boolean:
            return (y if x else z)


class BitwiseAnd(sympy.Function):
    pass


class BitwiseOr(sympy.Function):
    pass


class BitwiseXor(sympy.Function):
    pass


class BitwiseNot(sympy.Function):
    pass


class LeftShift(sympy.Function):
    pass


class RightShift(sympy.Function):
    pass


class ROUND(sympy.Function):

    @classmethod
    def eval(cls, x):
        """
        Evaluates rounding to integer.

        :param x: Value to round.
        :return: Return value (literal or symbolic).
        """
        if x.is_Number:
            return round(x)

    def _eval_is_integer(self):
        return True


class Is(sympy.Function):
    pass


class IsNot(sympy.Function):
    pass


class Attr(sympy.Function):
    """
    Represents a get-attribute call on a function, equivalent to ``a.b`` in Python.
    """

    @property
    def free_symbols(self):
        return {sympy.Symbol(str(self))}

    def __str__(self):
        return f'{self.args[0]}.{self.args[1]}'


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
    e = sympy.Wild('e', properties=[lambda k: isinstance(k, sympy.Basic) and not isinstance(k, sympy.Atom)])
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
                nexpr = nexpr.subs(floor, int_floor(int_floor(m[c], m[d]), m[b]))
                processed += 1
                continue
            # Floor of floor: "floor(a / floor(c/d))"
            m = floor.match(sympy.floor(a / int_floor(c, d)))
            if m is not None:
                nexpr = nexpr.subs(floor, int_floor(m[a], int_floor(m[c], m[d])))
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
                int_floor(sympy.Mul(*(candidate.args[:ri] + candidate.args[ri + 1:])), int(1 / candidate.args[ri])))
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


def evaluate_optional_arrays(expr, sdfg):
    """
    Evaluate Is(...) and IsNot(...) expressions for arrays.

    :param expr: The symbolic expression to evaluate.
    :param sdfg: SDFG that contains arrays.
    :return: A simplified version of the expression.
    """
    if not isinstance(expr, sympy.Basic):
        return expr

    none = symbol('NoneSymbol')

    def _process_is(elem: Union[Is, IsNot]):
        if elem.args[0] == none:
            if elem.args[1] == none:  # Both arguments are None
                return True
            cand = str(elem.args[1])
            if cand in sdfg.arrays and sdfg.arrays[cand].optional is False:
                # Equivalent to `None is x` for a non-optional x
                return False

        else:  # elem.args[0] is not None
            if elem.args[1] == none:
                cand = str(elem.args[0])
                if cand in sdfg.arrays and sdfg.arrays[cand].optional is False:
                    # Equivalent to `x is None` for a non-optional x
                    return False

        # Neither argument is None
        return None

    # Check internal expressions
    reevaluate = False
    for elem in sympy.postorder_traversal(expr):
        if any(a.func is Is or a.func is IsNot for a in elem.args):
            args = list(elem.args)
            for i, a in enumerate(args):
                changed = False
                if a.func is Is:
                    res = _process_is(a)
                    if res is not None:
                        args[i] = sympy.sympify(res)
                        changed = True
                elif a.func is IsNot:
                    res = _process_is(a)
                    if res is not None:
                        args[i] = sympy.sympify(not res)
                        changed = True
                if changed:
                    elem._args = tuple(args)
                    reevaluate = True
    if reevaluate:  # If an internal expression changed, re-simplify expression as a whole
        expr = expr.simplify()

    # Check top-level expression
    if expr.func is Is or expr.func is IsNot:
        res = _process_is(expr)
        if res is not None:
            return sympy.sympify(res if expr.func is Is else not res)

    return expr


class PythonOpToSympyConverter(ast.NodeTransformer):
    """ 
    Replaces various operations with the appropriate SymPy functions to avoid non-symbolic evaluation.
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

    _ast_to_sympy_functions = {
        ast.BitAnd: 'BitwiseAnd',
        ast.BitOr: 'BitwiseOr',
        ast.BitXor: 'BitwiseXor',
        ast.Invert: 'BitwiseNot',
        ast.LShift: 'LeftShift',
        ast.RShift: 'RightShift',
        ast.FloorDiv: 'int_floor',
    }

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            func_node = ast.copy_location(ast.Name(id=type(node.op).__name__, ctx=ast.Load()), node)
            new_node = ast.Call(func=func_node, args=[self.visit(node.operand)], keywords=[])
            return ast.copy_location(new_node, node)
        elif isinstance(node.op, ast.Invert):
            func_node = ast.copy_location(ast.Name(id=self._ast_to_sympy_functions[type(node.op)], ctx=ast.Load()),
                                          node)
            new_node = ast.Call(func=func_node, args=[self.visit(node.operand)], keywords=[])
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in self._ast_to_sympy_functions:
            func_node = ast.copy_location(ast.Name(id=self._ast_to_sympy_functions[type(node.op)], ctx=ast.Load()),
                                          node)
            new_node = ast.Call(func=func_node,
                                args=[self.visit(value) for value in (node.left, node.right)],
                                keywords=[])
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        func_node = ast.copy_location(ast.Name(id=type(node.op).__name__, ctx=ast.Load()), node)

        # First two arguments are given as one call
        new_node = ast.Call(func=func_node, args=[self.visit(value) for value in node.values[:2]], keywords=[])
        new_node = ast.copy_location(new_node, node)
        # If more than two arguments, chain bool op calls (``and(and(x,y), z)``)
        for i in range(2, len(node.values)):
            new_node = ast.Call(func=func_node, args=[new_node, self.visit(node.values[i])], keywords=[])
            new_node = ast.copy_location(new_node, node)

        return ast.copy_location(new_node, node)

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise NotImplementedError
        op = node.ops[0]
        arguments = [node.left, node.comparators[0]]
        func_node = ast.copy_location(ast.Name(id=self._ast_to_sympy_comparators[type(op)], ctx=ast.Load()), node)
        new_node = ast.Call(func=func_node, args=[self.visit(arg) for arg in arguments], keywords=[])
        return ast.copy_location(new_node, node)

    def visit_Constant(self, node):
        if node.value is None:
            return ast.copy_location(ast.Name(id='NoneSymbol', ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_NameConstant(self, node):
        return self.visit_Constant(node)

    def visit_IfExp(self, node):
        new_node = ast.Call(func=ast.Name(id='IfExpr', ctx=ast.Load),
                            args=[self.visit(node.test),
                                  self.visit(node.body),
                                  self.visit(node.orelse)],
                            keywords=[])
        return ast.copy_location(new_node, node)
    
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Attribute):
            attr = ast.Subscript(value=ast.Name(id=node.value.attr, ctx=ast.Load()), slice=node.slice, ctx=ast.Load())
            new_node = ast.Call(func=ast.Name(id='Attr', ctx=ast.Load),
                                args=[self.visit(node.value.value), self.visit(attr)],
                                keywords=[])
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        new_node = ast.Call(func=ast.Name(id='Attr', ctx=ast.Load),
                            args=[self.visit(node.value), ast.Name(id=node.attr, ctx=ast.Load)],
                            keywords=[])
        return ast.copy_location(new_node, node)


@lru_cache(maxsize=16384)
def pystr_to_symbolic(expr, symbol_map=None, simplify=None) -> sympy.Basic:
    """ Takes a Python string and converts it into a symbolic expression. """
    from dace.frontend.python.astutils import unparse  # Avoid import loops

    if isinstance(expr, (SymExpr, sympy.Basic)):
        return expr
    if isinstance(expr, str):
        try:
            return sympy.Integer(int(expr))
        except ValueError:
            pass
        try:
            return sympy.Float(float(expr))
        except ValueError:
            pass
        if dtypes.validate_name(expr):
            return symbol(expr)

    symbol_map = symbol_map or {}
    locals = {
        'abs': sympy.Abs,
        'min': sympy.Min,
        'max': sympy.Max,
        'True': sympy.true,
        'False': sympy.false,
        'GtE': sympy.Ge,
        'LtE': sympy.Le,
        'NotEq': sympy.Ne,
        'floor': sympy.floor,
        'ceil': sympy.ceiling,
        'round': ROUND,
        # Convert and/or to special sympy functions to avoid boolean evaluation
        'And': AND,
        'Or': OR,
        'var': sympy.Symbol('var'),
        'root': sympy.Symbol('root'),
        'arg': sympy.Symbol('arg'),
        'Is': Is,
        'IsNot': IsNot,
        'BitwiseAnd': BitwiseAnd,
        'BitwiseOr': BitwiseOr,
        'BitwiseXor': BitwiseXor,
        'BitwiseNot': BitwiseNot,
        'LeftShift': LeftShift,
        'RightShift': RightShift,
        'int_floor': int_floor,
        'int_ceil': int_ceil,
        'IfExpr': IfExpr,
        'Mod': sympy.Mod,
        'Attr': Attr,
    }
    # _clash1 enables all one-letter variables like N as symbols
    # _clash also allows pi, beta, zeta and other common greek letters
    locals.update(_sympy_clash)

    if isinstance(expr, str):
        # Sympy processes "not/and/or" as direct evaluation. Replace with And/Or(x, y), Not(x)
        # Also replaces bitwise operations with user-functions since SymPy does not support bitwise operations.
        if re.search(r'\bnot\b|\band\b|\bor\b|\bNone\b|==|!=|\bis\b|\bif\b|[&]|[|]|[\^]|[~]|[<<]|[>>]|[//]|[\.]', expr):
            expr = unparse(PythonOpToSympyConverter().visit(ast.parse(expr).body[0]))

    # TODO: support SymExpr over-approximated expressions
    try:
        return sympy_to_dace(sympy.sympify(expr, locals, evaluate=simplify), symbol_map)
    except (TypeError, sympy.SympifyError):  # Symbol object is not subscriptable
        # Replace subscript expressions with function calls
        expr = expr.replace('[', '(')
        expr = expr.replace(']', ')')
        return sympy_to_dace(sympy.sympify(expr, locals, evaluate=simplify), symbol_map)


@lru_cache(maxsize=2048)
def simplify(expr: SymbolicType) -> SymbolicType:
    return sympy.simplify(expr)


class DaceSympyPrinter(sympy.printing.str.StrPrinter):
    """ Several notational corrections for integer math and C++ translation
        that sympy.printing.cxxcode does not provide. """

    def __init__(self, arrays, cpp_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = arrays or set()
        self.cpp_mode = cpp_mode

    def _print_Float(self, expr):
        nf = sympy_numeric_fix(expr)
        if isinstance(nf, int) or nf != expr:
            return self._print(nf)
        return super()._print_Float(expr)

    def _print_Function(self, expr):
        if str(expr.func) in self.arrays:
            return f'{expr.func}[{expr.args[0]}]'
        if self.cpp_mode and str(expr.func) == 'int_floor':
            return '((%s) / (%s))' % (self._print(expr.args[0]), self._print(expr.args[1]))
        if str(expr.func) == 'AND':
            return f'(({self._print(expr.args[0])}) and ({self._print(expr.args[1])}))'
        if str(expr.func) == 'OR':
            return f'(({self._print(expr.args[0])}) or ({self._print(expr.args[1])}))'
        if str(expr.func) == 'Attr':
            return f'{self._print(expr.args[0])}.{self._print(expr.args[1])}'
        return super()._print_Function(expr)

    def _print_Mod(self, expr):
        return '((%s) %% (%s))' % (self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Equality(self, expr):
        return '((%s) == (%s))' % (self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Unequality(self, expr):
        return '((%s) != (%s))' % (self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Not(self, expr):
        return '(not (%s))' % self._print(expr.args[0])

    def _print_Infinity(self, expr):
        if self.cpp_mode:
            return 'INFINITY'
        return super()._print_Infinity(expr)

    def _print_NegativeInfinity(self, expr):
        if self.cpp_mode:
            return '-INFINITY'
        return super()._print_NegativeInfinity(expr)

    def _print_Symbol(self, expr):
        if expr.name == 'NoneSymbol':
            return 'nullptr' if self.cpp_mode else 'None'
        return super()._print_Symbol(expr)

    def _print_Pow(self, expr):
        base = self._print(expr.args[0])
        exponent = self._print(expr.args[1])

        # Special case for square root
        if self.cpp_mode:
            try:
                if float(exponent) == 0.5:
                    return f'dace::math::sqrt({base})'
            except ValueError:
                pass

        # Special case for integer powers
        try:
            int_exp = int(exponent)
            if int_exp == 0:
                return '1'
            negative = int_exp < 0
            if negative:
                int_exp = -int_exp
            res = "({})".format(base)
            for _ in range(1, int_exp):
                res += " * ({})".format(base)

            if negative:
                res = f'reciprocal({res})'
            return res
        except ValueError:
            if self.cpp_mode:
                return "dace::math::pow({f}, {s})".format(f=self._print(expr.args[0]), s=self._print(expr.args[1]))
            else:
                return f'({self._print(expr.args[0])}) ** ({self._print(expr.args[1])})'


@lru_cache(maxsize=16384)
def symstr(sym, arrayexprs: Optional[Set[str]] = None, cpp_mode=False) -> str:
    """ 
    Convert a symbolic expression to a compilable expression. 

    :param sym: Symbolic expression to convert.
    :param arrayexprs: Set of names of arrays, used to convert SymPy 
                       user-functions back to array expressions.
    :param cpp_mode: If True, returns a C++-compilable expression. Otherwise,
                     returns a Python expression.
    :return: Expression in string format depending on the value of ``cpp_mode``.
    """

    if isinstance(sym, SymExpr):
        return symstr(sym.expr, arrayexprs, cpp_mode=cpp_mode)

    try:
        sym = sympy_numeric_fix(sym)
        sym = sympy_intdiv_fix(sym)
        sym = sympy_divide_fix(sym)

        sstr = DaceSympyPrinter(arrayexprs, cpp_mode).doprint(sym)

        if isinstance(sym, symbol) or isinstance(sym, sympy.Symbol) or isinstance(
                sym, sympy.Number) or dtypes.isconstant(sym):
            return sstr
        else:
            return '(' + sstr + ')'
    except (AttributeError, TypeError, ValueError):
        sstr = DaceSympyPrinter(arrayexprs, cpp_mode).doprint(sym)
        return '(' + sstr + ')'


def safe_replace(mapping: Dict[Union[SymbolicType, str], Union[SymbolicType, str]],
                 replace_callback: Callable[[Dict[str, str]], None],
                 value_as_string: bool = False) -> None:
    """
    Safely replaces symbolic expressions that may clash with each other via a
    two-step replacement. For example, the mapping ``{M: N, N: M}`` would be
    translated to replacing ``{N, M} -> __dacesym_{N, M}`` followed by
    ``__dacesym{N, M} -> {M, N}``.

    :param mapping: The replacement dictionary.
    :param replace_callback: A callable function that receives a replacement
                             dictionary and performs the replacement (can be 
                             unsafe).
    :param value_as_string: Replacement values are replaced as strings rather 
                            than symbols.
    """
    # First, filter out direct (to constants) and degenerate (N -> N) replacements
    repl = {}
    invrepl = {}
    for k, v in mapping.items():
        # Degenerate
        if str(k) == str(v):
            continue

        # Not symbolic
        try:
            if not value_as_string:
                v = pystr_to_symbolic(v)
        except (TypeError, ValueError, AttributeError, sympy.SympifyError):
            repl[k] = v
            continue

        # Constant
        try:
            float(v)
            repl[k] = v
            continue
        except (TypeError, ValueError, AttributeError):
            pass

        # Otherwise, symbolic replacement
        repl[k] = f'__dacesym_{k}'
        invrepl[f'__dacesym_{k}'] = v

    if len(repl) == 0:
        return

    # Make the two-step replacement
    replace_callback(repl)
    if len(invrepl) == 0:
        return
    replace_callback(invrepl)


@lru_cache(16384)
def _spickle(obj):
    return str(obj)


def _sunpickle(obj):
    return pystr_to_symbolic(obj)


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


def equalize_symbols(a: sympy.Expr, b: sympy.Expr) -> Tuple[sympy.Expr, sympy.Expr]:
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


def inequal_symbols(a: Union[sympy.Expr, Any], b: Union[sympy.Expr, Any]) -> bool:
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


def equal(a: SymbolicType, b: SymbolicType, is_length: bool = True) -> Union[bool, None]:
    """
    Compares 2 symbolic expressions and returns True if they are equal, False if they are inequal,
    and None if the comparison is inconclusive.

    :param a: First symbolic expression.
    :param b: Second symbolic expression.
    :param is_length: If True, the assumptions that a, b are integers and positive are made.
    """

    args = [arg.expr if isinstance(arg, SymExpr) else arg for arg in (a, b)]

    if any([args is None for args in args]):
        return False

    facts = []
    if is_length:
        for arg in args:
            facts += [sympy.Q.integer(arg), sympy.Q.positive(arg)]

    with sympy.assuming(*facts):
        return sympy.ask(sympy.Q.is_true(sympy.Eq(*args)))


def symbols_in_code(code: str, potential_symbols: Set[str] = None,
                    symbols_to_ignore: Set[str] = None) -> Set[str]:
    """
    Tokenizes a code string for symbols and returns a set thereof.

    :param code: The code to tokenize.
    :param potential_symbols: If not None, filters symbols to this given set.
    :param symbols_to_ignore: If not None, filters out symbols from this set.
    """
    if not code:
        return set()
    if potential_symbols is not None and len(potential_symbols) == 0:
        # Don't bother tokenizing for an empty set of potential symbols
        return set()
    
    tokens = set(re.findall(_NAME_TOKENS, code))
    if potential_symbols is not None:
        tokens &= potential_symbols
    if symbols_to_ignore is None:
        return tokens
    return tokens - symbols_to_ignore
