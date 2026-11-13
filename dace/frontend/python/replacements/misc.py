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
import copy
import functools
from collections.abc import Callable
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
    # A push publishes what was written before it (the consumer may read it): it is not moved or fused with the writes
    tasklet = state.add_tasklet(
        f"{stream}_append",
        dict.fromkeys(inputs),
        dict.fromkeys(["__out"]),
        code,
        debuginfo=pv.current_lineinfo,
        side_effects=True,
    )
    for conn, memlet in inputs.items():
        state.add_edge(state.add_read(memlet.data), None, tasklet, conn, memlet)
    state.add_edge(tasklet, "__out", state.add_write(stream), None, Memlet(data=stream, subset="0"))


# The updates of ``dace.atomic`` (see there), as write-conflict resolutions. A subtraction adds the negated value, which
# targets have an atomic instruction for.
_ATOMIC_UPDATES = {
    "add": "lambda a, b: a + b",
    "sub": "lambda a, b: a + b",
    "prod": "lambda a, b: a * b",
    "min": "lambda a, b: min(a, b)",
    "max": "lambda a, b: max(a, b)",
    "and_": "lambda a, b: a & b",
    "or_": "lambda a, b: a | b",
    "xor": "lambda a, b: a ^ b",
    "exch": "lambda a, b: b",
}


def _atomic_function(name: str) -> Callable[..., str]:
    """The replacement of ``old = dace.atomic.<name>(array, index, value)``: an atomic update whose old value is
    read."""

    def update(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, array: str, index, value) -> str:
        return _atomic_update(
            pv, sdfg, state, f"dace.atomic.{name}", _ATOMIC_UPDATES[name], array, index, value, negate=name == "sub"
        )

    return update


for _name in _ATOMIC_UPDATES:
    oprepo.replaces(f"dace.atomic.{_name}")(_atomic_function(_name))
    oprepo.replaces(f"dace.frontend.python.interface.atomic.{_name}")(_atomic_function(_name))


@oprepo.replaces("dace.atomic.custom")
@oprepo.replaces("dace.frontend.python.interface.atomic.custom")
def _atomic_custom(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op, array: str, index, value) -> str:
    """``old = dace.atomic.custom(lambda a, b: ..., array, index, value)``: an atomic update with a function of the old
    and the new value, whose old value is read (see ``dace.atomic``)."""
    if isinstance(op, StringLiteral):
        op = op.value
    try:
        function = ast.parse(op).body[0].value if isinstance(op, str) else None
    except SyntaxError:
        function = None
    if not isinstance(function, ast.Lambda) or len(function.args.args) != 2:
        raise DaceSyntaxError(pv, None, f"dace.atomic.custom updates with a lambda of two arguments, got {op}")
    return _atomic_update(pv, sdfg, state, "dace.atomic.custom", op, array, index, value)


def _atomic_update(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    function: str,
    wcr: str,
    array: str,
    index,
    value,
    negate: bool = False,
) -> str:
    """An atomic update of ``array[index]`` with ``value`` through the write-conflict resolution ``wcr``, whose old
    value is read. The index and the value are scalars, constants or symbolic expressions.

    :param function: The name of the program function, for errors.
    :param negate: Updates with the negated value.
    :return: The name of the scalar that holds the old value.
    """
    if not isinstance(array, str) or array not in sdfg.arrays:
        raise DaceSyntaxError(pv, None, f"{function} updates an array, got {array}")
    inputs = {}

    def operand(name: str, operand) -> str:
        if isinstance(operand, str) and operand in sdfg.arrays:
            if sdfg.arrays[operand].total_size != 1:
                raise DaceSyntaxError(pv, None, f"{function} takes scalar indices and values, got {operand}")
            inputs[name] = Memlet(data=operand, subset="0")
            return name
        if isinstance(operand, (Number, symbolic.SymExpr)) or symbolic.issymbolic(operand):
            return symbolic.symstr(operand)
        raise DaceSyntaxError(pv, None, f"Unsupported operand {operand} for {function}")

    indices = index if isinstance(index, (tuple, list)) else (index,)
    # The elements of a tuple of expressions are the values of the expressions (lists of one result)
    indices = [idx[0] if isinstance(idx, list) and len(idx) == 1 else idx for idx in indices]
    subscript = [operand(f"__index{i}", idx) for i, idx in enumerate(indices)]
    update = operand("__value", value)
    if negate:
        update = f"-({update})"
    desc = sdfg.arrays[array]
    if any(conn in inputs for conn in subscript):  # An index read from data: the tasklet addresses the array
        target = Memlet.from_array(array, desc, wcr=wcr)
        element = f"__target[{', '.join(subscript)}]"
    else:  # The memlet addresses the element
        target = Memlet(data=array, subset=", ".join(subscript), wcr=wcr)
        element = "__target"
    target.dynamic = True
    # The old value is a read of the element, which the tasklet takes from its output connector: an input edge of the
    # element (unused in the code) shows the read to analyses, so that the writes before the update stay live
    inputs["__element"] = Memlet(data=array, subset=copy.deepcopy(target.subset), dynamic=True)
    old, _ = sdfg.add_scalar("atomic_old", desc.dtype, transient=True, find_new_name=True)
    # Assigning the connector of a dynamic write-conflict resolution updates atomically; reading it gives the old value
    tasklet = state.add_tasklet(
        function.replace("dace.", "").replace(".", "_").rstrip("_"),
        dict.fromkeys(inputs),
        dict.fromkeys(["__target", "__old"]),
        f"__old = {element} = {update}",
        debuginfo=pv.current_lineinfo,
        side_effects=True,  # Ordered after the writes before it (acquire-release): not moved or fused with them
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
