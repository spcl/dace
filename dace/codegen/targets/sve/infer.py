# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Inference: This module patches certain dispatchers in the `type_inference.py`, to better suit SVE.
"""
import dace.codegen.targets.sve.util
import numpy as np
import ast
from dace import dtypes
from dace.codegen import cppunparse
from dace.symbolic import SymExpr
from dace.symbolic import symstr
import sympy
import sys

def infer_expr_type(ast, symbols=None):
    symbols = symbols or {}
    inferred_symbols = {}
    return _dispatch(ast, symbols, inferred_symbols)


def _dispatch(tree, symbols, inferred_symbols):
    """Dispatcher function, dispatching tree type T to method _T."""
    try:
        tree = iter(tree)
        for t in tree:
            _dispatch(t, symbols, inferred_symbols)
    except TypeError:
        # The infer module overwrites some methods of `type_inference` to suit SVE
        # If a dispatcher is defined in `infer`, it will be called, otherwise the original one.
        patch = sys.modules[__name__]
        name = "_" + tree.__class__.__name__
        meth = None
        if hasattr(patch, name):
            meth = getattr(patch, name)
        else:
            meth = getattr(dace.codegen.tools.type_inference, name)

        return meth(tree, symbols, inferred_symbols)

def _Call(t, symbols, inferred_symbols):
    # The only permitted calls in SVE are for math functions.
    # Since the math functions are independent of the data type (can accept a lot of different types),
    # as the result type of the call, we return the resulting type of all arguments.
    # FIXME: Ensure that it is passed some float?
    inf_type = _dispatch(t.func, symbols, inferred_symbols)
    arg_types = []
    for e in t.args:
        arg_types.append(_dispatch(e, symbols, inferred_symbols))
    for e in t.keywords:
        _dispatch(e, symbols, inferred_symbols)
    return inf_type or dtypes.result_type_of(arg_types[0], *arg_types)
