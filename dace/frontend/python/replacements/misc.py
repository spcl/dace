# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for miscellaneous functions and convenience utility functions, such as a function that
calls element-wise operations on data containers.
"""

import dace  # noqa
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import Memlet, SDFG, SDFGState, dtypes, symbolic
from dace.frontend.python.common import DaceSyntaxError

import ast
import functools
from numbers import Number


@oprepo.replaces("slice")
def _slice(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *args, **kwargs):
    return (slice(*args, **kwargs),)


@oprepo.replaces_method("Stream", "append")
def _stream_append(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, stream: str, value) -> None:
    """``S.append(value)`` pushes one element (a scalar, a constant or a symbolic expression) onto stream ``S``."""
    if isinstance(value, str) and value in sdfg.arrays:
        if sdfg.arrays[value].total_size != 1:
            raise DaceSyntaxError(pv, None, f"{stream}.append takes one element, got {value}")
        inputs = {"__inp": Memlet(data=value, subset="0")}
        code = "__out = __inp"
    elif isinstance(value, (Number, symbolic.SymExpr)) or symbolic.issymbolic(value):
        inputs = {}
        code = f"__out = {symbolic.symstr(value)}"
    else:
        raise DaceSyntaxError(pv, None, f"Unsupported argument {value} for {stream}.append")
    tasklet = state.add_tasklet(
        f"{stream}_append", dict.fromkeys(inputs), dict.fromkeys(["__out"]), code, debuginfo=pv.current_lineinfo
    )
    for conn, memlet in inputs.items():
        state.add_edge(state.add_read(memlet.data), None, tasklet, conn, memlet)
    state.add_edge(tasklet, "__out", state.add_write(stream), None, Memlet(data=stream, subset="0"))


@oprepo.replaces("dace.atomic.add")
@oprepo.replaces("dace.frontend.python.interface.atomic.add")
def _atomic_add(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, array: str, index, value) -> str:
    """``old = dace.atomic.add(array, index, value)``: an atomic update whose old value is read (see ``dace.atomic``).
    The index and the value are scalars, constants or symbolic expressions."""
    if not isinstance(array, str) or array not in sdfg.arrays:
        raise DaceSyntaxError(pv, None, f"dace.atomic.add updates an array, got {array}")
    inputs = {}

    def operand(name: str, operand) -> str:
        if isinstance(operand, str) and operand in sdfg.arrays:
            if sdfg.arrays[operand].total_size != 1:
                raise DaceSyntaxError(pv, None, f"dace.atomic.add takes scalar indices and values, got {operand}")
            inputs[name] = Memlet(data=operand, subset="0")
            return name
        if isinstance(operand, (Number, symbolic.SymExpr)) or symbolic.issymbolic(operand):
            return symbolic.symstr(operand)
        raise DaceSyntaxError(pv, None, f"Unsupported operand {operand} for dace.atomic.add")

    indices = index if isinstance(index, (tuple, list)) else (index,)
    subscript = [operand(f"__index{i}", idx) for i, idx in enumerate(indices)]
    addend = operand("__value", value)
    desc = sdfg.arrays[array]
    if any(conn in inputs for conn in subscript):  # An index read from data: the tasklet addresses the array
        target = Memlet.from_array(array, desc, wcr="lambda a, b: a + b")
        element = f"__target[{', '.join(subscript)}]"
    else:  # The memlet addresses the element
        target = Memlet(data=array, subset=", ".join(subscript), wcr="lambda a, b: a + b")
        element = "__target"
    target.dynamic = True
    old, _ = sdfg.add_scalar("atomic_old", desc.dtype, transient=True, find_new_name=True)
    # Assigning the connector of a dynamic write-conflict resolution updates atomically; reading it gives the old value
    tasklet = state.add_tasklet(
        "atomic_add",
        dict.fromkeys(inputs),
        dict.fromkeys(["__target", "__old"]),
        f"__old = {element} = {addend}",
        debuginfo=pv.current_lineinfo,
    )
    for conn, memlet in inputs.items():
        state.add_edge(state.add_read(memlet.data), None, tasklet, conn, memlet)
    state.add_edge(tasklet, "__target", state.add_write(array), None, target)
    state.add_edge(tasklet, "__old", state.add_write(old), None, Memlet(data=old, subset="0"))
    return old


@oprepo.replaces_operator("Array", "MatMult", otherclass="StorageType")
def _cast_storage(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, stype: dtypes.StorageType) -> str:
    desc = sdfg.arrays[arr]
    desc.storage = stype
    return arr


@oprepo.replaces("dace.elementwise")
def elementwise(
    pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, func: StringLiteral | str, in_array: str, out_array=None
):
    """
    Apply a lambda function to each element in the input.
    """

    inparr = sdfg.arrays[in_array]
    restype = sdfg.arrays[in_array].dtype

    if out_array is None:
        out_array, outarr = pv.add_temp_transient(inparr.shape, restype, inparr.storage)
    else:
        outarr = sdfg.arrays[out_array]

    func_ast = ast.parse(func.value if isinstance(func, StringLiteral) else func)
    try:
        lambda_ast = func_ast.body[0].value
        if len(lambda_ast.args.args) != 1:
            raise SyntaxError(f"Expected lambda with one arg, but {func} has {len(lambda_ast.args.arrgs)}")
        arg = lambda_ast.args.args[0].arg
        replaced_ast = astutils.ASTFindReplace({arg: "__inp"}).visit(lambda_ast.body)
        body = astutils.unparse(replaced_ast)
    except AttributeError:
        raise SyntaxError(f"Could not parse func {func}")

    code = f"__out = {body}"

    num_elements = functools.reduce(lambda x, y: x * y, inparr.shape)
    if num_elements == 1:
        inp = state.add_read(in_array)
        out = state.add_write(out_array)
        tasklet = state.add_tasklet("_elementwise_", {"__inp"}, {"__out"}, code)
        state.add_edge(inp, None, tasklet, "__inp", Memlet.from_array(in_array, inparr))
        state.add_edge(tasklet, "__out", out, None, Memlet.from_array(out_array, outarr))
    else:
        state.add_mapped_tasklet(
            name="_elementwise_",
            map_ranges={f"__i{dim}": f"0:{N}" for dim, N in enumerate(inparr.shape)},
            inputs={"__inp": Memlet.simple(in_array, ",".join([f"__i{dim}" for dim in range(len(inparr.shape))]))},
            code=code,
            outputs={"__out": Memlet.simple(out_array, ",".join([f"__i{dim}" for dim in range(len(inparr.shape))]))},
            external_edges=True,
        )

    return out_array
