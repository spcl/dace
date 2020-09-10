# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import re
from collections import namedtuple
from typing import Any, Dict, Union

from dace import data, subsets
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.memlet import Memlet
from dace.symbolic import pystr_to_symbolic
from dace.frontend.python.common import DaceSyntaxError

MemletType = Union[ast.Call, ast.Attribute, ast.Subscript, ast.Name]
MemletExpr = namedtuple('MemletExpr', ['name', 'accesses', 'wcr', 'subset'])


def inner_eval_ast(defined, node, additional_syms=None):
    if isinstance(node, ast.AST):
        code = astutils.unparse(node)
    else:
        return node

    syms = {}
    syms.update(defined)
    if additional_syms is not None:
        syms.update(additional_syms)

    # First try to evaluate normally
    try:
        return eval(code, syms)
    except:  # Literally anything can happen here
        # If doesn't work, try to evaluate as a sympy expression
        # Replace subscript expressions with function calls (sympy support)
        code = code.replace('[', '(')
        code = code.replace(']', ')')
        return pystr_to_symbolic(code)


def pyexpr_to_symbolic(defined_arrays_and_symbols: Dict[str, Any],
                       expr_ast: ast.AST):
    """ Converts a Python AST expression to a DaCe symbolic expression
        with error checks (raises `SyntaxError` on failure).
        :param defined_arrays_and_symbols: Defined arrays and symbols
               in the context of this expression.
        :param expr_ast: The Python AST expression to convert.
        :return: Symbolic expression.
    """
    # TODO!
    return inner_eval_ast(defined_arrays_and_symbols, expr_ast)


def _ndslice_to_subset(ndslice):
    is_tuple = [isinstance(x, tuple) for x in ndslice]
    if not any(is_tuple):
        return subsets.Indices(ndslice)
    else:
        if not all(is_tuple):
            # If a mix of ranges and indices is found, convert to range
            for i in range(len(ndslice)):
                if not is_tuple[i]:
                    ndslice[i] = (ndslice[i], ndslice[i], 1)
        return subsets.Range(ndslice)


def _fill_missing_slices(das, ast_ndslice, array, indices):
    # Filling ndslice with default values from array dimensions
    # if ranges not specified (e.g., of the form "A[:]")
    ndslice = [None] * len(array.shape)
    ndslice_size = 1
    offsets = []
    idx = 0
    for i, dim in enumerate(ast_ndslice):
        if isinstance(dim, tuple):
            rb = pyexpr_to_symbolic(das, dim[0] or 0)
            re = pyexpr_to_symbolic(das, dim[1] or array.shape[indices[i]]) - 1
            rs = pyexpr_to_symbolic(das, dim[2] or 1)
            ndslice[i] = (rb, re, rs)
            offsets.append(i)
            idx += 1
        else:
            ndslice[i] = pyexpr_to_symbolic(das, dim)

    # Extend slices to unspecified dimensions
    for i in range(len(ast_ndslice), len(array.shape)):
        # ndslice[i] = (0, array.shape[idx] - 1, 1)
        # idx += 1
        ndslice[i] = (0, array.shape[i] - 1, 1)
        offsets.append(i)

    return ndslice, offsets


def parse_memlet_subset(array: data.Data, node: Union[ast.Name, ast.Subscript],
                        das: Dict[str, Any]):
    array_dependencies = {}

    # Get memlet range
    ndslice = [(0, s - 1, 1) for s in array.shape]
    if isinstance(node, ast.Subscript):
        # Parse and evaluate ND slice(s) (possibly nested)
        ast_ndslices = astutils.subscript_to_ast_slice_recursive(node)
        offsets = list(range(len(array.shape)))

        # Loop over nd-slices (A[i][j][k]...)
        subset_array = []
        for ast_ndslice in ast_ndslices:
            # Cut out dimensions that were indexed in the previous slice
            narray = copy.deepcopy(array)
            narray.shape = [
                s for i, s in enumerate(array.shape) if i in offsets
            ]

            # Loop over the N dimensions
            ndslice, offsets = _fill_missing_slices(das, ast_ndslice, narray,
                                                    offsets)
            subset_array.append(_ndslice_to_subset(ndslice))

        subset = subset_array[0]

        # Compose nested indices, e.g., of the form "A[i,:,j,:][k,l]"
        for i in range(1, len(subset_array)):
            subset = subset.compose(subset_array[i])

        # Compute additional array dependencies (as a result of
        # indirection)
        # for dim in subset:
        #     if not isinstance(dim, tuple): dim = [dim]
        #     for r in dim:
        #         for expr in symbolic.swalk(r):
        #             if symbolic.is_sympy_userfunction(expr):
        #                 arr = expr.func.__name__
        #                 array_dependencies[arr] = self.curnode.globals[arr]

    else:  # Use entire range
        subset = _ndslice_to_subset(ndslice)

    return subset


# Parses a memlet statement
def ParseMemlet(visitor, defined_arrays_and_symbols: Dict[str, Any],
                node: MemletType):
    das = defined_arrays_and_symbols
    arrname = rname(node)
    if arrname not in das:
        raise DaceSyntaxError(visitor, node,
                              'Use of undefined data "%s" in memlet' % arrname)
    array = das[arrname]

    # Determine number of accesses to the memlet (default is the slice size)
    num_accesses = None
    write_conflict_resolution = None
    # Detects expressions of the form "A(2)[...]", "A(300)", "A(1, sum)[:]"
    if isinstance(node, ast.Call):
        if len(node.args) < 1 or len(node.args) > 3:
            raise DaceSyntaxError(
                visitor, node,
                'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = pyexpr_to_symbolic(das, node.args[0])
        if len(node.args) >= 2:
            write_conflict_resolution = node.args[1]
    elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
        if len(node.value.args) < 1 or len(node.value.args) > 3:
            raise DaceSyntaxError(
                visitor, node,
                'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = pyexpr_to_symbolic(das, node.value.args[0])
        if len(node.value.args) >= 2:
            write_conflict_resolution = node.value.args[1]

    subset = parse_memlet_subset(array, node, das)

    # If undefined, default number of accesses is the slice size
    if num_accesses is None:
        num_accesses = subset.num_elements()

    return MemletExpr(arrname, num_accesses, write_conflict_resolution, subset)


def parse_memlet(visitor, src: MemletType, dst: MemletType,
                 defined_arrays_and_symbols: Dict[str, data.Data]):
    srcexpr, dstexpr, localvar = None, None, None
    if isinstance(src,
                  ast.Name) and rname(src) not in defined_arrays_and_symbols:
        localvar = rname(src)
    else:
        srcexpr = ParseMemlet(visitor, defined_arrays_and_symbols, src)
    if isinstance(dst,
                  ast.Name) and rname(dst) not in defined_arrays_and_symbols:
        if localvar is not None:
            raise DaceSyntaxError(
                visitor, src,
                'Memlet source and destination cannot both be local variables')
        localvar = rname(dst)
    else:
        dstexpr = ParseMemlet(visitor, defined_arrays_and_symbols, dst)

    if srcexpr is not None and dstexpr is not None:
        # Create two memlets
        raise NotImplementedError
    elif srcexpr is not None:
        expr = srcexpr
    else:
        expr = dstexpr

    return localvar, Memlet.simple(expr.name,
                                   expr.subset,
                                   num_accesses=expr.accesses,
                                   wcr_str=expr.wcr)
