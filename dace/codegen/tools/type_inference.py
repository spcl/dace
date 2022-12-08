# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Type inference: traverses code and returns types for all undefined symbols according to C semantics
    infer() has a lenient implementation: if something it not inferred (for example an unsupported construct) it will not
        return anything and it will not produce errors

    This module is inspired by astunparse: https://github.com/simonpercivall/astunparse
"""

import numpy as np
import ast
from dace import dtypes
from dace import symbolic
from dace.codegen import cppunparse
from dace.symbolic import symbol, SymExpr, symstr
import sympy
import sys
import dace.frontend.python.astutils
import inspect
from typing import Union


def infer_types(code, symbols=None):
    """
    Perform type inference on the given code.

    :param code: a string, AST, or symbolic expression
    :param symbols: optional,  already known symbols with their types. This is a dictionary "symbol name" -> dytpes.typeclass:
    :return: a dictionary "symbol name" -> dtypes.typeclass of inferred symbols
    """
    symbols = symbols or {}
    inferred_symbols = {}
    if isinstance(code, str):
        _dispatch(ast.parse(code), symbols, inferred_symbols)
    elif isinstance(code, ast.AST):
        _dispatch(code, symbols, inferred_symbols)
    elif isinstance(code, sympy.Basic) or isinstance(code, SymExpr):
        _dispatch(ast.parse(symstr(code)), symbols, inferred_symbols)
    elif isinstance(code, list):
        # call infer for any code elements, maintaining a list of inferred_symbols so far
        # defined symbols get updated with newly inferred symbols
        defined_symbols = symbols.copy()
        for c in code:
            defined_symbols.update(inferred_symbols)
            inf_symbols = infer_types(c, defined_symbols)
            inferred_symbols.update(inf_symbols)
    return inferred_symbols


def infer_expr_type(code, symbols=None):
    """
    Return inferred type of a given expression.
    
    :param code: code string (an expression) or symbolic expression
    :param symbols: already defined symbols (if any) in a dictionary "symbol name" -> dytpes.typeclass:
    :return: inferred type
    """
    symbols = symbols or {}
    inferred_symbols = {}
    if isinstance(code, (str, float, int, complex)):
        parsed_ast = ast.parse(str(code))
    elif isinstance(code, sympy.Basic):
        parsed_ast = ast.parse(sympy.printing.pycode(code))
    elif isinstance(code, SymExpr):
        parsed_ast = ast.parse(sympy.printing.pycode(code.expr))
    else:
        raise TypeError(f"Cannot convert type {type(code)} to a Python AST.")

    # The parsed AST must only contain one expression
    if hasattr(parsed_ast, "body") and isinstance(parsed_ast.body[0], ast.Expr):
        return _dispatch(parsed_ast.body[0], symbols, inferred_symbols)
    else:
        raise TypeError("Expected expression, got: {}".format(type(code)))


def _dispatch(tree, symbols, inferred_symbols):
    """Dispatcher function, dispatching tree type T to method _T."""
    try:
        tree = iter(tree)
        for t in tree:
            _dispatch(t, symbols, inferred_symbols)
    except TypeError:
        current_module = sys.modules[__name__]
        meth = getattr(current_module, "_" + tree.__class__.__name__)
        return meth(tree, symbols, inferred_symbols)


def _Module(tree, symbols, inferred_symbols):
    for stmt in tree.body:
        _dispatch(stmt, symbols, inferred_symbols)


def _Interactive(tree, symbols, inferred_symbols):
    for stmt in tree.body:
        _dispatch(stmt, symbols, inferred_symbols)


def _Expression(tree, symbols, inferred_symbols):
    return _dispatch(tree.body, symbols, inferred_symbols)


def _Expr(tree, symbols, inferred_symbols):
    return _dispatch(tree.value, symbols, inferred_symbols)


def _dispatch_lhs_tuple(targets, symbols, inferred_symbols):
    for target in targets:
        _dispatch(target, symbols, inferred_symbols)


def _Assign(t, symbols, inferred_symbols):
    # Handle the case of a tuple output
    if len(t.targets) > 1:
        _dispatch_lhs_tuple(t.targets, symbols, inferred_symbols)
    else:
        target = t.targets[0]
        if isinstance(target, ast.Tuple):
            if len(target.elts) > 1:
                _dispatch_lhs_tuple(target.elts, symbols, inferred_symbols)
            target = target.elts[0]

        if not isinstance(
                target,
            (ast.Subscript, ast.Attribute)) and not target.id in symbols and not target.id in inferred_symbols:
            # the target is not already defined: we should try to infer the type looking at the value
            inferred_type = _dispatch(t.value, symbols, inferred_symbols)
            inferred_symbols[target.id] = inferred_type

        inferred_type = _dispatch(target, symbols, inferred_symbols)
    _dispatch(t.value, symbols, inferred_symbols)


def _AugAssign(t, symbols, inferred_symbols):
    _dispatch(t.target, symbols, inferred_symbols)
    # Operations that require a function call
    if t.op.__class__.__name__ in cppunparse.CPPUnparser.funcops:
        separator, func = cppunparse.CPPUnparser.funcops[t.op.__class__.__name__]
        if not t.target.id in symbols and not t.target.id in inferred_symbols:
            _dispatch(t.target, symbols, inferred_symbols)
            inferred_type = _dispatch(t.value, symbols, inferred_symbols)
            inferred_symbols[t.target.id] = inferred_type
    else:
        if not t.target.id in symbols and not t.target.id in inferred_symbols:
            inferred_type = _dispatch(t.value, symbols, inferred_symbols)
            inferred_symbols[t.target.id] = inferred_type


def _AnnAssign(t, symbols, inferred_symbols):
    if isinstance(t.target, ast.Tuple):
        if len(t.target.elts) > 1:
            _dispatch_lhs_tuple(t.target.elts, symbols, inferred_symbols)
        else:
            target = t.target.elts[0]
    else:
        target = t.target

    # Assignment of the form x: int = 0 is converted to int x = (int)0;
    if not target.id in symbols and not target.id in inferred_symbols:
        # get the type indicated into the annotation
        inferred_type = _infer_dtype(t.annotation)
        if not inferred_type:
            inferred_type = _dispatch(t.annotation, symbols, inferred_symbols)
        inferred_symbols[target.id] = inferred_type

        _dispatch(t.annotation, symbols, inferred_symbols)

    _dispatch(t.target, symbols, inferred_symbols)

    if t.value:
        _dispatch(t.annotation, symbols, inferred_symbols)
        _dispatch(t.value, symbols, inferred_symbols)


def _Return(t, symbols, inferred_symbols):
    if t.value:
        _dispatch(t.value, symbols, inferred_symbols)


def _generic_FunctionDef(t, symbols, inferred_symbols):
    for deco in t.decorator_list:
        _dispatch(deco, symbols, inferred_symbols)

    if getattr(t, "returns", False):
        if isinstance(t.returns, ast.NameConstant):
            if t.returns.value is not None:
                _dispatch(t.returns, symbols, inferred_symbols)
        else:
            _dispatch(t.returns, symbols, inferred_symbols)

    _dispatch(t.args, symbols, inferred_symbols)
    _dispatch(t.body, symbols, inferred_symbols)


def _FunctionDef(t, symbols, inferred_symbols):
    _generic_FunctionDef(t, symbols, inferred_symbols)


def _AsyncFunctionDef(t, symbols, inferred_symbols):
    _generic_FunctionDef(t, symbols, inferred_symbols)


def _generic_For(t, symbols, inferred_symbols):
    if isinstance(t.target, ast.Tuple):
        if len(t.target.elts) == 1:
            (elt, ) = t.target.elts
            if elt.id not in symbols and elt not in inferred_symbols:
                inferred_type = _dispatch(elt, symbols, inferred_symbols)
                inferred_symbols[elt] = inferred_type
        else:
            for elt in t.target.elts:
                if elt.id not in symbols and elt not in inferred_symbols:
                    inferred_type = _dispatch(elt, symbols, inferred_symbols)
                    inferred_symbols[elt] = inferred_type
    else:
        inferred_type = _dispatch(t.target, symbols, inferred_symbols)
        if t.target.id not in symbols and t.target.id not in inferred_symbols:
            inferred_symbols[t.target.id] = inferred_type

    _dispatch(t.iter, symbols, inferred_symbols)
    _dispatch(t.body, symbols, inferred_symbols)


def _For(t, symbols, inferred_symbols):
    _generic_For(t, symbols, inferred_symbols)


def _AsyncFor(t, symbols, inferred_symbols):
    _generic_For(t, symbols, inferred_symbols)


def _If(t, symbols, inferred_symbols):
    _dispatch(t.test, symbols, inferred_symbols)
    _dispatch(t.body, symbols, inferred_symbols)

    while (t.orelse and len(t.orelse) == 1 and isinstance(t.orelse[0], ast.If)):
        t = t.orelse[0]
        _dispatch(t.test, symbols, inferred_symbols)
        _dispatch(t.body, symbols, inferred_symbols)

    # final else
    if t.orelse:
        _dispatch(t.orelse, symbols, inferred_symbols)


def _While(t, symbols, inferred_symbols):
    _dispatch(t.test, symbols, inferred_symbols)
    _dispatch(t.body, symbols, inferred_symbols)


def _Str(t, symbols, inferred_symbols):
    return dtypes.pointer(dtypes.int8)


def _FormattedValue(t, symbols, inferred_symbols):
    # FormattedValue(expr value, int? conversion, expr? format_spec)
    _dispatch(t.value, symbols, inferred_symbols)

    if t.format_spec is not None:
        if not isinstance(t.format_spec, ast.Str):
            _dispatch(t.format_spec, symbols, inferred_symbols)


def _JoinedStr(t, symbols, inferred_symbols):
    for value in t.values:
        if not isinstance(value, ast.Str):
            _dispatch(value, symbols, inferred_symbols)
    return dtypes.pointer(dtypes.int8)


def _Name(t, symbols, inferred_symbols):
    if t.id in cppunparse._py2c_reserved:
        return dtypes.typeclass(np.result_type(t.id))
    else:
        # check if this name is a python type, it is in defined_symbols or in local symbols.
        # If yes, take the type
        inferred_type = None

        # if this is a statement generated from a tasklet with a dynamic memlet, it could have a leading * (pointer)
        t_id = t.id[1:] if t.id.startswith('*') else t.id
        if t_id.strip("()") in cppunparse._py2c_typeconversion:
            inferred_type = cppunparse._py2c_typeconversion[t_id.strip("()")]
        elif t_id in symbols:
            # defined symbols could have dtypes, in case convert it to typeclass
            inferred_type = symbols[t_id]
            if isinstance(inferred_type, np.dtype):
                inferred_type = dtypes.typeclass(inferred_type.type)
            elif isinstance(inferred_type, symbolic.symbol):
                inferred_type = inferred_type.dtype
        elif t_id in inferred_symbols:
            inferred_type = inferred_symbols[t_id]
        return inferred_type


def _NameConstant(t, symbols, inferred_symbols):
    return dtypes.result_type_of(dtypes.typeclass(type(t.value)), dtypes.typeclass(np.min_scalar_type(t.value).name))


def _Constant(t, symbols, inferred_symbols):
    # String value
    if isinstance(t.value, (str, bytes)):
        return dtypes.pointer(dtypes.int8)

    # Numeric value
    return dtypes.result_type_of(dtypes.typeclass(type(t.value)), dtypes.typeclass(np.min_scalar_type(t.value).name))


def _Num(t, symbols, inferred_symbols):
    # get the minimum between the minimum type needed to represent this number and the corresponding default data types
    # e.g., if num=1, then it will be represented by using the default integer type (int32 if C data types are used)
    return dtypes.result_type_of(dtypes.typeclass(type(t.n)), dtypes.typeclass(np.min_scalar_type(t.n).name))


def _IfExp(t, symbols, inferred_symbols):
    _dispatch(t.test, symbols, inferred_symbols)
    type_body = _dispatch(t.body, symbols, inferred_symbols)
    type_orelse = _dispatch(t.orelse, symbols, inferred_symbols)
    return dtypes.result_type_of(type_body, type_orelse)


def _Tuple(t, symbols, inferred_symbols):
    for elt in t.elts:
        _dispatch(elt, symbols, inferred_symbols)


def _UnaryOp(t, symbols, inferred_symbols):
    return _dispatch(t.operand, symbols, inferred_symbols)


def _BinOp(t, symbols, inferred_symbols):
    # Operations that require a function call
    if t.op.__class__.__name__ in cppunparse.CPPUnparser.funcops:
        separator, func = cppunparse.CPPUnparser.funcops[t.op.__class__.__name__]

        # get the type of left and right operands for type inference
        type_left = _dispatch(t.left, symbols, inferred_symbols)
        type_right = _dispatch(t.right, symbols, inferred_symbols)
        # infer type and returns
        return dtypes.result_type_of(type_left, type_right)
    # Special case for integer power
    elif t.op.__class__.__name__ == 'Pow':
        if (isinstance(t.right, (ast.Num, ast.Constant)) and int(t.right.n) == t.right.n and t.right.n >= 0):
            if t.right.n != 0:
                type_left = _dispatch(t.left, symbols, inferred_symbols)
                for i in range(int(t.right.n) - 1):
                    _dispatch(t.left, symbols, inferred_symbols)
            return dtypes.result_type_of(type_left, dtypes.typeclass(np.uint32))
        else:
            type_left = _dispatch(t.left, symbols, inferred_symbols)
            type_right = _dispatch(t.right, symbols, inferred_symbols)
            return dtypes.result_type_of(type_left, type_right)
    else:

        # get left and right types for type inference
        type_left = _dispatch(t.left, symbols, inferred_symbols)
        type_right = _dispatch(t.right, symbols, inferred_symbols)
        return dtypes.result_type_of(type_left, type_right)


def _Compare(t, symbols, inferred_symbols):
    # If any vector occurs in the comparision, the inferred type is a bool vector
    inf_type = _dispatch(t.left, symbols, inferred_symbols)
    vec_len = None
    if isinstance(inf_type, dtypes.vector):
        vec_len = inf_type.veclen
    for o, e in zip(t.ops, t.comparators):
        if o.__class__.__name__ not in cppunparse.CPPUnparser.cmpops:
            continue
        inf_type = _dispatch(e, symbols, inferred_symbols)
        if isinstance(inf_type, dtypes.vector):
            # Make sure all occuring vectors are of same size
            if vec_len is not None and vec_len != inf_type.veclen:
                raise SyntaxError('Inconsistent vector lengths in Compare')
            vec_len = inf_type.veclen
    return dtypes.vector(dace.bool, vec_len) if vec_len is not None else dtypes.bool


def _BoolOp(t, symbols, inferred_symbols):
    # If any vector occurs in the bool op, the inferred type is also a bool vector
    vec_len = None
    for v in t.values:
        inf_type = _dispatch(v, symbols, inferred_symbols)
        if isinstance(inf_type, dtypes.vector):
            # Make sure all occuring vectors are of same size
            if vec_len is not None and vec_len != inf_type.veclen:
                raise SyntaxError('Inconsistent vector lengths in BoolOp')
            vec_len = inf_type.veclen
    return dtypes.vector(dace.bool, vec_len) if vec_len is not None else dtypes.bool


def _infer_dtype(t: Union[ast.Name, ast.Attribute]):
    name = dace.frontend.python.astutils.rname(t)
    if '.' in name:
        dtype_str = name[name.rfind('.') + 1:]
    else:
        dtype_str = name

    dtype = getattr(dtypes, dtype_str, False)
    if isinstance(dtype, dtypes.typeclass):
        return dtype
    if isinstance(dtype, np.dtype):
        return dtypes.typeclass(dtype.type)

    return None


def _Attribute(t, symbols, inferred_symbols):
    inferred_type = _dispatch(t.value, symbols, inferred_symbols)
    return inferred_type


def _Call(t, symbols, inferred_symbols):
    inf_type = _dispatch(t.func, symbols, inferred_symbols)

    # Dispatch the arguments and determine their types
    arg_types = [_dispatch(e, symbols, inferred_symbols) for e in t.args]

    for e in t.keywords:
        _dispatch(e, symbols, inferred_symbols)

    # If the function symbol is known, always return the defined type
    if inf_type:
        return inf_type

    # In case of a typeless math function, determine the return type based on the arguments
    name = dace.frontend.python.astutils.rname(t)
    idx = name.rfind('.')
    if idx > -1:
        module = name[:name.rfind('.')]
    else:
        module = ''
    if module == 'math':
        return dtypes.result_type_of(arg_types[0], *arg_types)

    # Reading from an Intel channel returns the channel type
    if name == 'read_channel_intel':
        return arg_types[0]

    if name in ('abs', 'log'):
        return arg_types[0]
    if name in ('min', 'max'): # binary math operations that do not exist in the math module
        return dtypes.result_type_of(arg_types[0], *arg_types)
    if name in ('round', ):
        return dtypes.typeclass(int)   

    # dtypes (dace.int32, np.float64) can be used as functions
    inf_type = _infer_dtype(t)
    if inf_type:
        return inf_type

    # In any other case simply return None
    return None


def _Subscript(t, symbols, inferred_symbols):
    value_type = _dispatch(t.value, symbols, inferred_symbols)
    slice_type = _dispatch(t.slice, symbols, inferred_symbols)

    if isinstance(slice_type, dtypes.pointer):
        raise SyntaxError('Invalid syntax (pointer given as slice)')

    # A slice as subscript (e.g. [0:N]) returns a pointer
    if isinstance(t.slice, ast.Slice):
        return value_type

    # A vector as subscript of a pointer returns a vector of the base type
    if isinstance(value_type, dtypes.pointer) and isinstance(slice_type, dtypes.vector):
        if not np.issubdtype(slice_type.type, np.integer):
            raise SyntaxError('Subscript must be some integer type')
        return dtypes.vector(value_type.base_type, slice_type.veclen)

    # Otherwise (some index as subscript) we return the base type
    if isinstance(value_type, dtypes.typeclass):
        return value_type.base_type

    return value_type


def _Index(t, symbols, inferred_symbols):
    return _dispatch(t.value, symbols, inferred_symbols)


def _Slice(t, symbols, inferred_symbols):
    if t.lower:
        _dispatch(t.lower, symbols, inferred_symbols)
    if t.upper:
        _dispatch(t.upper, symbols, inferred_symbols)
    if t.step:
        _dispatch(t.step, symbols, inferred_symbols)


def _ExtSlice(t, symbols, inferred_symbols):
    for d in t.dims:
        _dispatch(d, symbols, inferred_symbols)


# argument
def _arg(t, symbols, inferred_symbols):
    if t.annotation:
        #argument with annotation, we can derive the type
        inferred_type = _dispatch(t.annotation, symbols, inferred_symbols)
        inferred_symbols[t.arg] = inferred_type


# others
def _arguments(t, symbols, inferred_symbols):
    first = True
    # normal arguments
    defaults = [None] * (len(t.args) - len(t.defaults)) + t.defaults
    for a, d in zip(t.args, defaults):
        _dispatch(a, symbols, inferred_symbols)
        if d:
            _dispatch(d, symbols, inferred_symbols)

        # varargs, or bare '*' if no varargs but keyword-only arguments present
        if t.vararg or getattr(t, "kwonlyargs", False):
            raise SyntaxError('Invalid C++')

        # keyword-only arguments
        if getattr(t, "kwonlyargs", False):
            raise SyntaxError('Invalid C++')

        # kwargs
        if t.kwarg:
            raise SyntaxError('Invalid C++')


def _Lambda(t, symbols, inferred_symbols):
    _dispatch(t.args, symbols, inferred_symbols)
    _dispatch(t.body, symbols, inferred_symbols)


#####################################################
# Constructs that are not involved in type inference
#####################################################


def _Pass(t, symbols, inferred_symbols):
    pass


def _Break(t, symbols, inferred_symbols):
    pass


def _Continue(t, symbols, inferred_symbols):
    pass


def _Assert(t, symbols, inferred_symbols):
    #Nothing to infer
    pass


def _Print(t, symbols, inferred_symbols):
    #Nothing to infer
    pass


def _Raise(t, symbols, inferred_symbols):
    pass


def _Try(t, symbols, inferred_symbols):
    pass


def _TryExcept(t, symbols, inferred_symbols):
    pass


def _TryFinally(t, symbols, inferred_symbols):
    pass


def _ExceptHandler(t, symbols, inferred_symbols):
    pass


def _Bytes(t, symbols, inferred_symbols):
    pass


def _Ellipsis(t, symbols, inferred_symbols):
    pass


def _alias(t, symbols, inferred_symbols):
    pass


###########################################
# Invalid C/C++ will do not infer anything
##########################################


def _Import(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _ImportFrom(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _Delete(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _Exec(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _Global(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _Nonlocal(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _Yield(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _YieldFrom(t, symbols, inferred_symbols):
    # Nothing to infer
    pass


def _ClassDef(t, symbols, inferred_symbols):
    pass


def _generic_With(t, symbols, inferred_symbols):
    pass


def _With(t, symbols, inferred_symbols):
    pass


def _AsyncWith(t, symbols, inferred_symbols):
    pass


def _Repr(t, symbols, inferred_symbols):
    pass


def _List(t, symbols, inferred_symbols):
    pass


def _ListComp(t, symbols, inferred_symbols):
    pass


def _GeneratorExp(t, symbols, inferred_symbols):
    pass


def _SetComp(t, symbols, inferred_symbols):
    pass


def _DictComp(t, symbols, inferred_symbols):
    pass


def _comprehension(t, symbols, inferred_symbols):
    pass


def _Set(t, symbols, inferred_symbols):
    pass


def _Dict(t, symbols, inferred_symbols):
    pass


def _Starred(t, symbols, inferred_symbols):
    pass


def _keyword(t, symbols, inferred_symbols):
    pass


def _withitem(t, symbols, inferred_symbols):
    pass


def _Await(t, symbols, inferred_symbols):
    pass
