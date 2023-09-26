# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from dace import data, dtypes, subsets
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.memlet import Memlet
from dace.symbolic import pystr_to_symbolic, SymbolicType
from dace.frontend.python.common import DaceSyntaxError

MemletType = Union[ast.Call, ast.Attribute, ast.Subscript, ast.Name]


@dataclass
class MemletExpr:
    name: str
    accesses: SymbolicType
    wcr: Optional[ast.AST]
    subset: subsets.Range
    new_axes: List[int]
    arrdims: Dict[int, str]


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


def pyexpr_to_symbolic(defined_arrays_and_symbols: Dict[str, Any], expr_ast: ast.AST):
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


def _parse_dim_atom(das, atom):
    result = pyexpr_to_symbolic(das, atom)
    if isinstance(result, data.Data):
        return pystr_to_symbolic(astutils.unparse(atom))
    return result


def _fill_missing_slices(das, ast_ndslice, array, indices):
    # Filling ndslice with default values from array dimensions
    # if ranges not specified (e.g., of the form "A[:]")
    ndslice = [None] * len(array.shape)
    offsets = []
    new_axes = []
    arrdims: Dict[int, str] = {}
    idx = 0
    new_idx = 0
    has_ellipsis = False
    for dim in ast_ndslice:
        if isinstance(dim, (str, list, slice)):
            dim = ast.Name(id=dim)

        if isinstance(dim, tuple):
            rb = _parse_dim_atom(das, dim[0] or 0)
            re = _parse_dim_atom(das, dim[1] or array.shape[indices[idx]]) - 1
            rs = _parse_dim_atom(das, dim[2] or 1)
            # NOTE: try/except for cases where rb/re are not symbols/numbers
            try:
                if (rb < 0) == True:
                    rb += array.shape[indices[idx]]
            except (TypeError, ValueError):
                pass
            try:
                if (re < 0) == True:
                    re += array.shape[indices[idx]]
            except (TypeError, ValueError):
                pass
            ndslice[idx] = (rb, re, rs)
            offsets.append(idx)
            idx += 1
            new_idx += 1
        elif (isinstance(dim, ast.Ellipsis) or dim is Ellipsis
              or (isinstance(dim, ast.Constant) and dim.value is Ellipsis)
              or (isinstance(dim, ast.Name) and dim.id is Ellipsis)):
            if has_ellipsis:
                raise IndexError('an index can only have a single ellipsis ("...")')
            has_ellipsis = True
            remaining_dims = len(ast_ndslice) - idx - 1
            for j in range(idx, len(ndslice) - remaining_dims):
                ndslice[j] = (0, array.shape[j] - 1, 1)
                idx += 1
                new_idx += 1
        elif (dim is None or (isinstance(dim, (ast.Constant, ast.NameConstant)) and dim.value is None)):
            new_axes.append(new_idx)
            new_idx += 1
            # NOTE: Do not increment idx here
        elif isinstance(dim, ast.Name) and isinstance(dim.id, (list, tuple)):
            # List/tuple literal
            ndslice[idx] = (0, array.shape[idx] - 1, 1)
            arrdims[indices[idx]] = dim.id
            idx += 1
            new_idx += 1
        elif isinstance(dim, ast.Name) and isinstance(dim.id, slice):
            # slice literal
            rb, re, rs = dim.id.start, dim.id.stop, dim.id.step
            if rb is None:
                rb = 0
            if re is None:
                re = array.shape[indices[idx]]
            if rs is None:
                rs = 1

            ndslice[idx] = (rb, re - 1, rs)
            idx += 1
            new_idx += 1
        elif (isinstance(dim, ast.Name) and dim.id in das and isinstance(das[dim.id], data.Array)):
            # Accessing an array with another
            desc = das[dim.id]
            if desc.dtype == dtypes.bool:
                # Boolean array indexing
                if len(ast_ndslice) > 1:
                    raise IndexError(f'Invalid indexing into array "{dim.id}". ' 'Only one boolean array is allowed.')
                if tuple(desc.shape) != tuple(array.shape):
                    raise IndexError(f'Invalid indexing into array "{dim.id}". '
                                     'Shape of boolean index must match original array.')
            elif desc.dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16,
                                dtypes.uint32, dtypes.uint64):
                # Integer array indexing
                pass
            else:
                raise ValueError(f'Unsupported indexing into array "{dim.id}". '
                                 'Only integer and boolean arrays are supported.')

            if data._prod(desc.shape) == 1:
                # Special case: one-element array treated as scalar
                ndslice[idx] = (dim.id, dim.id, 1)
            else:
                ndslice[idx] = (0, array.shape[idx] - 1, 1)
                arrdims[indices[idx]] = dim.id

            idx += 1
            new_idx += 1
        elif (isinstance(dim, ast.Name) and dim.id in das and isinstance(das[dim.id], data.Scalar)):
            ndslice[idx] = (dim.id, dim.id, 1)
            idx += 1
            new_idx += 1
        else:
            r = pyexpr_to_symbolic(das, dim)
            if (r < 0) == True:
                r += array.shape[indices[idx]]
            ndslice[idx] = r
            idx += 1
            new_idx += 1

    # Extend slices to unspecified dimensions
    for i in range(idx, len(array.shape)):
        # ndslice[i] = (0, array.shape[idx] - 1, 1)
        # idx += 1
        ndslice[i] = (0, array.shape[i] - 1, 1)
        offsets.append(i)

    return ndslice, offsets, new_axes, arrdims


def parse_memlet_subset(array: data.Data,
                        node: Union[ast.Name, ast.Subscript],
                        das: Dict[str, Any],
                        parsed_slice: Any = None) -> Tuple[subsets.Range, List[int], List[int]]:
    """ 
    Parses an AST subset and returns access range, as well as new dimensions to
    add.
    
    :param array: Accessed data descriptor (used for filling in missing data, 
                  e.g., negative indices or empty shapes).
    :param node: AST node representing whole array or subset thereof.
    :param das: Dictionary of defined arrays and symbols mapped to their values.
    :return: A 3-tuple of (subset, list of new axis indices, list of index-to-array-dimension correspondence).
    """
    # Get memlet range
    ndslice = [(0, s - 1, 1) for s in array.shape]
    extra_dims = []
    arrdims: Dict[int, str] = {}
    if isinstance(node, ast.Subscript):
        # Parse and evaluate ND slice(s) (possibly nested)
        if parsed_slice:
            cnode = copy.copy(node)
            cnode.slice = parsed_slice
        else:
            cnode = node
        ast_ndslices = astutils.subscript_to_ast_slice_recursive(cnode)
        offsets = list(range(len(array.shape)))

        # Loop over nd-slices (A[i][j][k]...)
        subset_array = []
        for idx, ast_ndslice in enumerate(ast_ndslices):
            # Cut out dimensions that were indexed in the previous slice
            narray = copy.deepcopy(array)
            narray.shape = [s for i, s in enumerate(array.shape) if i in offsets]

            # Loop over the N dimensions
            ndslice, offsets, new_extra_dims, arrdims = _fill_missing_slices(das, ast_ndslice, narray, offsets)
            if new_extra_dims and idx != (len(ast_ndslices) - 1):
                raise NotImplementedError('New axes only implemented for last ' 'slice')
            if arrdims and len(ast_ndslices) != 1:
                raise NotImplementedError('Array dimensions not implemented ' 'for consecutive subscripts')
            extra_dims = new_extra_dims
            subset_array.append(_ndslice_to_subset(ndslice))

        subset = subset_array[0]

        # Compose nested indices, e.g., of the form "A[i,:,j,:][k,l]"
        for i in range(1, len(subset_array)):
            subset = subset.compose(subset_array[i])

    else:  # Use entire range
        subset = _ndslice_to_subset(ndslice)

    return subset, extra_dims, arrdims


# Parses a memlet statement
def ParseMemlet(visitor,
                defined_arrays_and_symbols: Dict[str, Any],
                node: MemletType,
                parsed_slice: Any = None) -> MemletExpr:
    das = defined_arrays_and_symbols
    arrname = rname(node)
    if arrname not in das:
        raise DaceSyntaxError(visitor, node, 'Use of undefined data "%s" in memlet' % arrname)
    array = das[arrname]

    # Determine number of accesses to the memlet (default is the slice size)
    num_accesses = None
    write_conflict_resolution = None
    # Detects expressions of the form "A(2)[...]", "A(300)", "A(1, sum)[:]"
    if isinstance(node, ast.Call):
        if len(node.args) < 1 or len(node.args) > 3:
            raise DaceSyntaxError(
                visitor, node, 'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = pyexpr_to_symbolic(das, node.args[0])
        if len(node.args) >= 2:
            write_conflict_resolution = node.args[1]
    elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
        if len(node.value.args) < 1 or len(node.value.args) > 3:
            raise DaceSyntaxError(
                visitor, node, 'Number of accesses in memlet must be a number, symbolic '
                'expression, or -1 (dynamic)')
        num_accesses = pyexpr_to_symbolic(das, node.value.args[0])
        if len(node.value.args) >= 2:
            write_conflict_resolution = node.value.args[1]

    try:
        subset, new_axes, arrdims = parse_memlet_subset(array, node, das, parsed_slice)
    except IndexError:
        raise DaceSyntaxError(visitor, node, 'Failed to parse memlet expression due to dimensionality. '
                              f'Array dimensions: {array.shape}, expression in code: {astutils.unparse(node)}')

    # If undefined, default number of accesses is the slice size
    if num_accesses is None:
        num_accesses = subset.num_elements()

    return MemletExpr(arrname, num_accesses, write_conflict_resolution, subset, new_axes, arrdims)


def parse_memlet(visitor, src: MemletType, dst: MemletType, defined_arrays_and_symbols: Dict[str, data.Data]):
    srcexpr, dstexpr, localvar = None, None, None
    if isinstance(src, ast.Name) and rname(src) not in defined_arrays_and_symbols:
        localvar = rname(src)
    else:
        srcexpr = ParseMemlet(visitor, defined_arrays_and_symbols, src)
    if isinstance(dst, ast.Name) and rname(dst) not in defined_arrays_and_symbols:
        if localvar is not None:
            raise DaceSyntaxError(visitor, src, 'Memlet source and destination cannot both be local variables')
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

    return localvar, Memlet.simple(expr.name, expr.subset, num_accesses=expr.accesses, wcr_str=expr.wcr)
