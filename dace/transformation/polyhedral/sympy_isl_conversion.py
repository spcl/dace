# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" functionality to convert sympy to isl and vice versa."""

import islpy as isl
import sympy as sp
from dace import symbolic
from typing import Optional
from dace.symbolic import SymExpr, symbol
from dace import dtypes
from sympy import nsimplify
import re


class PythonPrinter(sp.printing.str.StrPrinter):
    """
    prints a sympy expression (with DaCe syntax)
    to executable python code (with DaCe functions)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Float(self, expr):
        if int(expr) == expr:
            return str(int(expr))
        return super()._print_Float(expr)

    def _print_Function(self, expr):
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


def sympy_to_pystr(sym) -> str:
    """
    converts a sympy expression to a string
    """
    def repstr(s):
        return s.replace('Min', 'min').replace('Max', 'max')

    if isinstance(sym, SymExpr):
        return sympy_to_pystr(sym.expr)

    try:
        sstr = PythonPrinter().doprint(sym)
        if isinstance(sym, symbol) or isinstance(sym, sp.Symbol) or isinstance(
                sym, sp.Number) or dtypes.isconstant(sym):
            return repstr(sstr)
        else:
            return '(' + repstr(sstr) + ')'
    except (AttributeError, TypeError, ValueError):
        sstr = PythonPrinter().doprint(sym)
        return '(' + repstr(sstr) + ')'


def to_sympy(node: isl.AstExpr):
    """
    converts an ISL AST expression to sympy (with DaCe syntax)
    more documentation in the ISL Manual 2020 p.222-223
    """
    def sp_bin_op(left, right):
        child_expr = [left, right]
        if node.op_get_type() == isl.ast_expr_op_type.sub:
            return sp.Add(child_expr[0], -child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.add:
            return sp.Add(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.mul:
            return sp.Mul(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.and_:
            return sp.And(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.and_then:
            return sp.And(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.eq:
            return sp.Eq(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.ge:
            return sp.Ge(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.gt:
            return sp.Gt(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.le:
            return sp.Le(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.lt:
            return sp.Lt(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.div:
            # Exact division. That is, the result is known to be an integer.
            return child_expr[0] / child_expr[1]
        elif node.op_get_type() == isl.ast_expr_op_type.pdiv_q:
            # Result of integer division, where dividend is known to be
            # non-negative. The divisor is known to be positive.
            int_floor = sp.Function('int_floor')
            return int_floor(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.fdiv_q:
            # Result of integer division, rounded towards negative infinity.
            # The divisor is known to be positive.
            floord = sp.Function('floord')
            return floord(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.pdiv_r:
            # Remainder of integer division, where dividend is known to be
            # non-negative. The divisor is known to be positive.
            return sp.Mod(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.zdiv_r:
            # Equal to zero iff the remainder on integer division is zero.
            # The divisor is known to be positive.
            return sp.Mod(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.max:
            return sp.Max(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.min:
            return sp.Min(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.or_:
            return sp.Or(child_expr[0], child_expr[1])
        elif node.op_get_type() == isl.ast_expr_op_type.or_else:
            return sp.Or(child_expr[0], child_expr[1])
        else:
            raise NotImplementedError

    if node.get_type() == isl.ast_expr_type.id:
        return symbol(node.get_id().name, dtype=dtypes.int64)
    elif node.get_type() == isl.ast_expr_type.name:
        raise NotImplementedError
    elif node.get_type() == isl.ast_expr_type.int:
        return sp.Integer(node.get_val().get_num_si())
    elif node.get_type() == isl.ast_expr_type.op:
        if node.op_get_type() == isl.ast_expr_op_type.minus:
            return -to_sympy(node.get_op_arg(0))
        elif node.op_get_type() == isl.ast_expr_op_type.access:
            # An array access. The number of arguments of the isl_ast_expr is
            # one more than the number of index expressions in the array access,
            # the first argument representing the array being accessed.
            raise NotImplementedError
        elif node.op_get_type() == isl.ast_expr_op_type.call:
            # A function call. The number of arguments of the isl_ast_expr is
            # one more than the number of arguments in the function call, the
            # first argument representing the function being called.
            raise NotImplementedError
        elif node.op_get_type() == isl.ast_expr_op_type.cond:
            # Conditional operator defined on three arguments. If the first
            # argument evaluates to true, then the result is equal to the
            # second argument. Otherwise, the result is equal to the third
            # argument. The second and third argument may only be evaluated
            # if the first argument evaluates to true and false, respectively.
            # Corresponds to a ? b: c in C.
            raise NotImplementedError
        elif node.op_get_type() == isl.ast_expr_op_type.member:
            raise NotImplementedError
        elif node.op_get_type() == isl.ast_expr_op_type.select:
            a = to_sympy(node.get_op_arg(0))
            b = to_sympy(node.get_op_arg(1))
            c = to_sympy(node.get_op_arg(2))
            # Conditional operator defined on three arguments. If the first
            # argument evaluates to true, then the result is equal to the
            # second argument. Otherwise, the result is equal to the third
            # argument. The second and third argument may be evaluated
            # independently of the value of the first argument. Corresponds
            # to a * b + (1 - a) * c in C.
            if a == True:
                return b
            else:
                return c
        else:
            child_expr = []
            for child in [
                    node.get_op_arg(i) for i in range(node.get_op_n_arg())
            ]:
                child_expr.append(to_sympy(child))
            result = sp_bin_op(child_expr[0], child_expr[1])
            for exp in child_expr[2:]:
                result = sp_bin_op(result, exp)
            return result

    else:
        raise NotImplementedError


def extract_end_cond(condition, itersym):
    # Find condition by matching expressions
    end: Optional[symbolic.SymbolicType] = None
    a = sp.Wild('a')
    match = condition.match(itersym < a)
    if match:
        end = match[a] - 1
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            end = match[a] + 1
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            end = match[a]
    return end


def extract_step_cond(condition, itersym):
    # Find condition by matching expressions
    step: Optional[symbolic.SymbolicType] = None
    a = sp.Wild('a')
    match = condition.match(itersym + a)
    if match:
        step = match[a]
    return step


class SympyToPwAff:
    """
    converts a sympy expression to an isl PwAff
    """
    def __init__(self, space, constants=None):
        """
        initializing the space of the returning PwAff
        """
        self.space = space
        if constants is None:
            self.constants = {}
        else:
            self.constants = constants

        result = {}
        zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space))
        result[0] = isl.PwAff.from_aff(zero)

        var_dict = zero.get_var_dict()
        for name, (dt, idx) in var_dict.items():
            result[name] = isl.PwAff.from_aff(
                zero.set_coefficient_val(dt, idx, 1))

        self.result = result

    def _get_zero(self):
        return self.result[0]

    def _get_var(self, v):
        if v.name in self.constants:
            return self._get_val(self.constants[v.name])
        else:
            if v.name not in self.result:
                print(1)
            return self.result[v.name]

    def _get_val(self, v):
        if isinstance(v, sp.Integer):
            val = isl.Val(int(v))
            aff_v = isl.Aff.val_on_domain(
                isl.LocalSpace.from_space(self.space), val)
            return isl.PwAff.from_aff(aff_v)
        elif isinstance(v, sp.Rational):
            aff_p = isl.Aff.val_on_domain(
                isl.LocalSpace.from_space(self.space), isl.Val(int(v.p)))
            aff_q = isl.Aff.val_on_domain(
                isl.LocalSpace.from_space(self.space), isl.Val(int(v.q)))
            return isl.PwAff.from_aff(aff_p.div(aff_q))
        else:
            raise NotImplementedError("Not Supported Sympy Expression")

    def _apply_multi_args_op(self, op, args):
        first = self.visit(args[0])
        second = self.visit(args[1])
        result = op(first, second)
        for arg in args[2:]:
            result = op(result, self.visit(arg))
        return result

    def visit(self, sym_expr):
        """
        convert sympy expression to an isl PwAff expression
        """
        if isinstance(sym_expr, sp.Symbol):
            return self._get_var(sym_expr)
        elif isinstance(sym_expr, sp.Number):
            if sym_expr.is_Float:
                return self._get_val(nsimplify(sym_expr))
            else:
                return self._get_val(sym_expr)
        elif isinstance(sym_expr, sp.Mul):
            return self._apply_multi_args_op(isl.PwAff.mul, sym_expr.args)
        elif isinstance(sym_expr, sp.Add):
            return self._apply_multi_args_op(isl.PwAff.add, sym_expr.args)
        elif isinstance(sym_expr, sp.Mod) and isinstance(
                sym_expr.args[1], sp.Integer):
            first = self.visit(sym_expr.args[0])
            second = isl.Val(int(sym_expr.args[1]))
            return isl.PwAff.mod_val(first, second)
        elif isinstance(sym_expr, sp.Equality):
            return self._apply_multi_args_op(isl.PwAff.eq_set, sym_expr.args)
        elif isinstance(sym_expr, sp.Unequality):
            return self._apply_multi_args_op(isl.PwAff.ne_set, sym_expr.args)
        elif isinstance(sym_expr, sp.GreaterThan):
            return self._apply_multi_args_op(isl.PwAff.ge_set, sym_expr.args)
        elif isinstance(sym_expr, sp.LessThan):
            return self._apply_multi_args_op(isl.PwAff.le_set, sym_expr.args)
        elif isinstance(sym_expr, sp.StrictGreaterThan):
            return self._apply_multi_args_op(isl.PwAff.gt_set, sym_expr.args)
        elif isinstance(sym_expr, sp.StrictLessThan):
            return self._apply_multi_args_op(isl.PwAff.lt_set, sym_expr.args)
        elif isinstance(sym_expr, sp.Max):
            return self._apply_multi_args_op(isl.PwAff.max, sym_expr.args)
        elif isinstance(sym_expr, sp.Min):
            return self._apply_multi_args_op(isl.PwAff.min, sym_expr.args)
        elif isinstance(sym_expr, sp.And):
            # Not, Xor, Implies, Equivalent
            result = self.visit(sym_expr.args[0])
            for arg in sym_expr.args[1:]:
                result = result.intersect(self.visit(arg))
            return result
        elif isinstance(sym_expr, sp.Or):
            result = self.visit(sym_expr.args[0])
            for arg in sym_expr.args[1:]:
                result = result.union(self.visit(arg))
            return result
        elif isinstance(sym_expr, sp.Function('floord')):
            res = self._apply_multi_args_op(isl.PwAff.div, sym_expr.args)
            return res.floor()
        elif isinstance(sym_expr, sp.Function('int_floor')):
            return self._apply_multi_args_op(isl.PwAff.div, sym_expr.args)
        elif isinstance(sym_expr, sp.Not):
            result = self.visit(sym_expr.args[0])
            return result.neg()
        else:
            if sym_expr:
                raise NotImplementedError("Not Supported Sympy Expression")
            return self._get_zero()


################################################################################
# Parse an ISL set to sympy ####################################################
################################################################################


def _isl_str_to_sympy(expr_str, simplify=True):
    expr_str = expr_str.replace('mod', '%')
    expr_str = expr_str.replace('<=', 'LtE')
    expr_str = expr_str.replace('>=', 'GtE')
    expr_str = expr_str.replace('!=', 'Ne')
    expr_str = expr_str.replace('=', '==')
    expr_str = expr_str.replace('LtE', '<=')
    expr_str = expr_str.replace('GtE', '>=')
    expr_str = expr_str.replace('Ne', '!=')

    def repl(m):
        return "{0} {1} and {1} {2}".format(m.group(1), m.group(2), m.group(3))

    expr_str = re.sub(r"(<=|<|>|>=){1} (\w+) (<=|<|>|>=){1}", repl, expr_str)

    return symbolic.pystr_to_symbolic(expr_str.strip(), simplify=simplify)


def _split_multiplication(expr_str, params):
    """
    list of symbols e.g. ['N', 'M']
    split expr_str "3N + 4M" into "3 * N + 4 * M"
    """
    expr_str = expr_str.strip()

    def repl(m):
        return m.group(1) + " * " + m.group(2)

    for p in params:
        expr_str = re.sub(r"(\d)+({0})".format(p), repl, expr_str)
    return expr_str


def _parse_isl_constraints(constraint):
    params = constraint.get_var_names(isl.dim_type.param)
    variables = constraint.get_var_names(isl.dim_type.set)
    body_str = str(constraint).split('->')[-1].strip()[1:-1].strip()
    cond_str = ''.join(body_str.split(":")[1:]).strip()
    symbols = params + variables
    cond_str = _split_multiplication(cond_str, symbols)
    constraint = _isl_str_to_sympy(cond_str)
    return params, variables, constraint


def parse_isl_pwaff(isl_pwa):
    params = isl_pwa.get_var_names(isl.dim_type.param)
    variables = isl_pwa.get_var_names(isl.dim_type.set)
    body_str = str(isl_pwa).split('->')[-1].strip()[1:-1].strip()
    symbols = params + variables
    try:
        [idx_str, cond_str] = body_str.split(":")
        idx_str = idx_str.strip()[1:-1]
        idx_str = _split_multiplication(idx_str, symbols)
        index = _isl_str_to_sympy(idx_str)
        cond_str = _split_multiplication(cond_str, symbols)
        constraint = _isl_str_to_sympy(cond_str)
        return params, variables, constraint, index
    except:
        idx_str = body_str
        idx_str = idx_str.strip()[1:-1]
        idx_str = _split_multiplication(idx_str, symbols)
        index = _isl_str_to_sympy(idx_str)
        return params, variables, None, index


def get_overapprox_range_list_from_set(isl_set):
    range_list = []
    for i in range(isl_set.n_dim()):
        stride = isl_set.get_stride(i).get_num_si()
        stride_sym = _isl_str_to_sympy(str(stride))
        pwaff_min = isl_set.dim_min(i)
        _, _, _, min_sym = parse_isl_pwaff(pwaff_min)
        pwaff_max = isl_set.dim_max(i)
        _, _, _, max_sym = parse_isl_pwaff(pwaff_max)
        dim_rng = (min_sym, max_sym, stride_sym)
        range_list.append(dim_rng)
    return range_list


def parse_isl_set(isl_set: isl.Set):
    """
    Parse an ISL set to sympy tuples
    :param isl_set: The ISL Set
    :return name: the name of the set
    :return params: a list of set parameters
    :return variables: a list of set variables
    :return constraints: a list of sympy constraints
    """
    name = isl_set.get_tuple_name()
    params = isl_set.get_var_names(isl.dim_type.param)
    variables = isl_set.get_var_names(isl.dim_type.set)
    basic_sets = isl_set.get_basic_sets()
    constraints = []
    for bs in basic_sets:
        _, _, constraint = _parse_isl_constraints(bs)
        constraints.append(constraint)
    return name, params, variables, constraints
