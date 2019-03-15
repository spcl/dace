import ast
import sympy

from sympy import Sum, Product, log, floor, ceiling
import sympy as functions
from sympy.abc import _clash
from sympy.printing.str import StrPrinter

from dace import types

DEFAULT_SYMBOL_TYPE = types.int32


class symbol(sympy.Symbol):
    """ Defines a symbolic expression. Extends SymPy symbols with DaCe-related
        information. """

    s_currentsymbol = 0
    s_values = {}
    s_types = {}
    s_constraints = {}

    @staticmethod
    def erase_symbols(symlist):
        for sym in symlist:
            del symbol.s_values[sym]
            del symbol.s_types[sym]
            del symbol.s_constraints[sym]

    def __new__(cls, name=None, dtype=DEFAULT_SYMBOL_TYPE, **assumptions):
        if name is None:
            # Set name dynamically
            name = "sym_" + str(symbol.s_currentsymbol)
            symbol.s_currentsymbol += 1
        elif name.startswith('__DACE'):
            raise NameError('Symbols cannot start with __DACE')

        if not isinstance(dtype, types.typeclass):
            raise TypeError('dtype must be a DaCe type, got %s' % str(dtype))

        if 'integer' in assumptions or 'int' not in str(dtype):
            self = sympy.Symbol.__new__(cls, name, **assumptions)
        else:
            self = sympy.Symbol.__new__(cls, name, integer=True, **assumptions)

        if name not in symbol.s_types:
            symbol.s_values[name] = None
            symbol.s_constraints[name] = []
            symbol.s_types[name] = dtype
        else:
            if dtype != DEFAULT_SYMBOL_TYPE and dtype != symbol.s_types[name]:
                raise TypeError('Type mismatch for existing symbol "%s" (%s) '
                                'and new type %s' %
                                (name, str(symbol.s_types[name]), str(dtype)))

        # Arrays to update when value is set
        self._arrays_to_update = []

        return self

    @staticmethod
    def from_name(name, **kwargs):
        if name in symbol.s_types:
            return symbol(name, symbol.s_types[name], **kwargs)
        return symbol(name, **kwargs)

    def set(self, value):
        if value is not None:
            # First, check constraints
            self.check_constraints(value)

        symbol.s_values[self.name] = symbol.s_types[self.name](value)

        for arr in self._arrays_to_update:
            arr.update_resolved_symbol(self)

    def reset(self):
        self.set(None)

    def is_initialized(self):
        return symbol.s_values[self.name] is not None

    def get(self):
        if symbol.s_values[self.name] is None:
            raise UnboundLocalError('Uninitialized symbol value for \'' +
                                    self.name + '\'')
        return symbol.s_values[self.name]

    def set_constraints(self, constraint_list):
        try:
            iter(constraint_list)
            symbol.s_constraints[self.name] = constraint_list
        except TypeError:  # constraint_list is not iterable
            symbol.s_constraints[self.name] = [constraint_list]

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
        return symbol.s_constraints[self.name]

    @property
    def dtype(self):
        return symbol.s_types[self.name]

    def check_constraints(self, value):
        fail = None
        for constraint in symbol.s_constraints[self.name]:
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
        if symbol.s_values[self.name] is None:
            return uninitialized_ret
        return symbol.s_values[self.name]


class SymExpr(object):
    """ Symbolic expressions with support for an overapproximation expression.
    """

    def __init__(self, main_expr: str, approx_expr: str = None):
        self._main_expr = pystr_to_symbolic(main_expr)
        if approx_expr is None:
            self._approx_expr = self._main_expr
        else:
            self._approx_expr = pystr_to_symbolic(approx_expr)

    @property
    def expr(self):
        return self._main_expr

    @property
    def approx(self):
        return self._approx_expr

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

    def __sub__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr - other.expr, self.approx - other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr - other, self.approx - other)
        return self - pystr_to_symbolic(other)

    def __mul__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr * other.expr, self.approx * other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr * other, self.approx * other)
        return self * pystr_to_symbolic(other)

    def __div__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr / other.expr, self.approx / other.approx)
        if isinstance(other, sympy.Expr):
            return SymExpr(self.expr / other, self.approx / other)
        return self / pystr_to_symbolic(other)

    __truediv__ = __div__

    def __floordiv__(self, other):
        if isinstance(other, SymExpr):
            return SymExpr(self.expr // other.expr,
                           self.approx // other.approx)
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


def eval(expr,
         uninitialized_value=None,
         keep_uninitialized=False,
         constants={}):
    """ Evaluates a complex expression with symbols, replacing all
        symbols with their values. """
    if isinstance(expr, SymExpr):
        return eval(expr.expr, uninitialized_value, keep_uninitialized)
    if not isinstance(expr, sympy.Expr):
        return expr

    result = expr
    if uninitialized_value is None:
        for atom in expr.atoms():
            if isinstance(atom, symbol):
                if atom.name in constants:
                    result = result.replace(atom, constants[atom.name])
                else:
                    try:
                        result = result.replace(atom, atom.get())
                    except (AttributeError, TypeError, ValueError):
                        if keep_uninitialized: pass
                        else: raise
    else:
        for atom in expr.atoms():
            if isinstance(atom, symbol):
                if atom.name in constants:
                    result = result.replace(atom, constants[atom.name])
                else:
                    result = result.replace(
                        atom, atom.get_or_return(uninitialized_value))

    if isinstance(result, sympy.Integer):
        return int(sympy.N(result))
    elif isinstance(result, sympy.Float):
        return float(sympy.N(result))

    return sympy.N(result)


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


# TODO: Merge with symlist
def symbols_in_sympy_expr(expr):
    """ Returns a list of free symbols in a SymPy Expression. """
    if not isinstance(expr, sympy.Expr):
        raise TypeError("Expected sympy.Expr, got: {}".format(
            type(expr).__name__))
    symbols = expr.free_symbols
    return map(str, symbols)


def issymbolic(value, constants={}):
    """ Returns True if an expression is symbolic with respect to its contents
        and a given dictionary of constant values. """
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
    return types.deduplicate(symbols)


def getsymbols(compilation_args):
    """ Helper function to get symbols from a list of decorator arguments
        ('@dace.program(...)/sdfg.compile'). """
    from dace import data

    result = {}
    for arg in compilation_args:
        if issymbolic(arg):
            # If argument is a symbol, we will resolve it on call.
            # No need for it to be a dependency
            pass
        elif isinstance(arg, data.Array):
            for d in arg.shape:
                if issymbolic(d):
                    result.update(symlist(d))
        else:
            try:
                result.update(getattr(
                    arg,
                    '_symlist'))  # Add all (not yet added) symbols to result
            except AttributeError:
                pass

    return result


def symbol_name_or_value(val):
    """ Returns the symbol name if symbol, otherwise the value as a string. """
    if isinstance(val, symbol):
        return val.name
    return str(val)


def sympy_to_dace(exprs, symbol_map={}):
    """ Convert all `sympy.Symbol`s to DaCe symbols, according to 
        `symbol_map`. """
    repl = {}

    oneelem = False
    try:
        exprs_iter = iter(exprs)
    except TypeError:
        oneelem = True
        exprs = [exprs]

    for i, expr in enumerate(exprs):
        if isinstance(expr, sympy.Basic):
            for atom in expr.atoms():
                if isinstance(atom, sympy.Symbol):
                    try:
                        repl[atom] = symbol_map[atom.name]
                    except KeyError:
                        # Symbol is not in map, create a DaCe symbol with same assumptions
                        repl[atom] = symbol.from_name(atom.name,
                                                      **atom.assumptions0)
            exprs[i] = expr.subs(repl)
    if oneelem:
        return exprs[0]
    return exprs


def is_sympy_userfunction(expr):
    """ Returns True if the expression is a SymPy function. """
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


def contains_sympy_functions(expr):
    """ Returns True if expression contains Sympy functions. """
    if is_sympy_userfunction(expr):
        return True
    for arg in expr.args:
        if contains_sympy_functions(arg):
            return True
    return False


def sympy_numeric_fix(expr):
    """ Fix for printing out integers as floats with ".00000000". 
        Converts the float constants in a given expression to integers. """
    if not isinstance(expr, sympy.Basic):
        if int(expr) == expr:
            return int(expr)
        return expr

    if isinstance(expr, sympy.Number) and expr == int(expr):
        return int(expr)
    return expr


def sympy_ceiling_fix(expr):
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
        for candidate in nexpr.find(sympy.mul.Mul):
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
                    sympy.mul.Mul(*(
                        candidate.args[:ri] + candidate.args[ri + 1:])),
                    int(1 / candidate.args[ri])))
            processed += 1

    return nexpr


def pystr_to_symbolic(expr, symbol_map={}):
    """ Takes a Python string and converts it into a symbolic expression. """
    if isinstance(expr, SymExpr):
        return expr

    locals = {'min': sympy.Min, 'max': sympy.Max}
    # _clash1 enables all one-letter variables like N as symbols
    # _clash also allows pi, beta, zeta and other common greek letters
    locals.update(_clash)

    # Sympy processes "not" as direct evaluation rather than negation
    if isinstance(expr, str) and 'not' in expr:
        expr = expr.replace('not', 'Not')

    return sympy_to_dace(sympy.sympify(expr, locals), symbol_map)


class DaceSympyPrinter(StrPrinter):
    """ Several notational corrections for integer math and C++ translation
        that sympy.printing.cxxcode does not provide. """

    def _print_Float(self, expr):
        if int(expr) == expr:
            return str(int(expr))
        return super()._print_Float(expr)

    def _print_Function(self, expr):
        if str(expr.func) == 'int_floor':
            return '((%s) / (%s))' % (self._print(expr.args[0]),
                                      self._print(expr.args[1]))
        return super()._print_Function(expr)

    def _print_Mod(self, expr):
        return '((%s) %% (%s))' % (self._print(expr.args[0]),
                                   self._print(expr.args[1]))


def symstr(sym):
    """ Convert a symbolic expression to a C++ compilable expression. """

    def repstr(s):
        return s.replace('Min', 'min').replace('Max', 'max')

    if isinstance(sym, SymExpr):
        return symstr(sym.expr)

    try:
        sym = sympy_numeric_fix(sym)
        sym = sympy_ceiling_fix(sym)
        sym = sympy_divide_fix(sym)

        sstr = DaceSympyPrinter().doprint(sym)

        if isinstance(sym,
                      symbol) or isinstance(sym, sympy.Symbol) or isinstance(
                          sym, sympy.Number) or types.isconstant(sym):
            return repstr(sstr)
        else:
            return '(' + repstr(sstr) + ')'
    except (AttributeError, TypeError, ValueError):
        sstr = DaceSympyPrinter().doprint(sym)
        return '(' + repstr(sstr) + ')'
