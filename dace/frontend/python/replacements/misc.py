# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for miscellaneous functions and convenience utility functions, such as a function that
calls element-wise operations on data containers.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import Memlet, SDFG, SDFGState, dtypes

import ast
import functools
from typing import Union


@oprepo.replaces('slice')
def _slice(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *args, **kwargs):
    return (slice(*args, **kwargs), )


@oprepo.replaces_operator('Array', 'MatMult', otherclass='StorageType')
def _cast_storage(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, stype: dtypes.StorageType) -> str:
    desc = sdfg.arrays[arr]
    desc.storage = stype
    return arr


@oprepo.replaces('dace.elementwise')
def elementwise(pv: 'ProgramVisitor',
                sdfg: SDFG,
                state: SDFGState,
                func: Union[StringLiteral, str],
                in_array: str,
                out_array=None):
    """
    Apply a lambda function to each element in the input.
    """

    inparr = sdfg.arrays[in_array]
    restype = sdfg.arrays[in_array].dtype

    if out_array is None:
        out_array, outarr = sdfg.add_temp_transient(inparr.shape, restype, inparr.storage)
    else:
        outarr = sdfg.arrays[out_array]

    func_ast = ast.parse(func.value if isinstance(func, StringLiteral) else func)
    try:
        lambda_ast = func_ast.body[0].value
        if len(lambda_ast.args.args) != 1:
            raise SyntaxError("Expected lambda with one arg, but {} has {}".format(func, len(lambda_ast.args.arrgs)))
        arg = lambda_ast.args.args[0].arg
        replaced_ast = astutils.ASTFindReplace({arg: '__inp'}).visit(lambda_ast.body)
        body = astutils.unparse(replaced_ast)
    except AttributeError:
        raise SyntaxError("Could not parse func {}".format(func))

    code = "__out = {}".format(body)

    num_elements = functools.reduce(lambda x, y: x * y, inparr.shape)
    if num_elements == 1:
        inp = state.add_read(in_array)
        out = state.add_write(out_array)
        tasklet = state.add_tasklet("_elementwise_", {'__inp'}, {'__out'}, code)
        state.add_edge(inp, None, tasklet, '__inp', Memlet.from_array(in_array, inparr))
        state.add_edge(tasklet, '__out', out, None, Memlet.from_array(out_array, outarr))
    else:
        state.add_mapped_tasklet(
            name="_elementwise_",
            map_ranges={
                f'__i{dim}': f'0:{N}'
                for dim, N in enumerate(inparr.shape)
            },
            inputs={'__inp': Memlet.simple(in_array, ','.join([f'__i{dim}' for dim in range(len(inparr.shape))]))},
            code=code,
            outputs={'__out': Memlet.simple(out_array, ','.join([f'__i{dim}' for dim in range(len(inparr.shape))]))},
            external_edges=True)

    return out_array
