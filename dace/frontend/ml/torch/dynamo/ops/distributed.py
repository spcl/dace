# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowerings of ``torch.distributed`` functional collectives and point-to-point operations to
:mod:`dace.libraries.comm` nodes.

Dynamo rewrites ``dist.all_reduce(t)`` and friends into functional operators (``_c10d_functional.*``) that return a new
tensor, followed by ``wait_tensor``. Collectives lower to blocking library calls into fresh output containers, so
``wait_tensor`` of a collective's result is the result itself. Point-to-point operations (``isend``, ``irecv``,
``batch_p2p_ops``, traced with ``TORCHDYNAMO_ENABLE_P2P_COMPILATION=1``) are nonblocking: their ``wait_tensor`` emits a
``Wait`` that orders the readers of a received tensor after it.

Every node communicates over a communicator argument of the program, one per process group and implementation (see
:mod:`..distributed`): ``MPI`` for CPU tensors and ``NCCL`` (RCCL on ROCm) for GPU tensors. Communication buffers are
contiguous; strided inputs are copied first.
"""

import dataclasses
from typing import Any

import sympy

from dace import data, dtypes
from dace.frontend.ml.torch.dynamo import distributed
from dace.frontend.ml.torch.dynamo.context import (
    LoweringContext,
    TensorValue,
    TupleValue,
    UnsupportedOpError,
    as_const,
    as_sym,
    contiguous_strides,
    sanitize_name,
)
from dace.frontend.ml.torch.dynamo.ops import register_lowering, resolve
from dace.libraries import comm
from dace.memlet import Memlet
from dace.sdfg import nodes

#: ``torch.distributed`` reduction names -> canonical names of :func:`dace.libraries.comm.utils.reduction_op`
_REDUCE_OPS = {
    "sum": "sum",
    "avg": "avg",
    "product": "prod",
    "min": "min",
    "max": "max",
    "band": "band",
    "bor": "bor",
    "bxor": "bxor",
}


@dataclasses.dataclass
class PendingP2P:
    """A nonblocking point-to-point operation whose ``wait_tensor`` has not been lowered yet."""

    implementation: str
    request: str  #: Request container (an ``MPI_Request`` array, or an integer token for NCCL)
    buffer: TensorValue  #: The sent or received buffer
    receive: bool


# ---------------------------------------------------------------------------------------------- helpers
def _val(node):
    return node.meta["val"]


def _reduce_op(op: Any) -> str:
    name = str(as_const(op)).lower()
    if name not in _REDUCE_OPS:
        raise UnsupportedOpError("_c10d_functional", f"reduction operator {name} is not supported")
    return _REDUCE_OPS[name]


def _group(group_name: Any) -> distributed.GroupInfo:
    return distributed.resolve_group(as_const(group_name))


def _communicator(ctx: LoweringContext, group: distributed.GroupInfo, implementation: str) -> str:
    """The program argument that holds this process's communicator of ``group`` (see :mod:`..distributed`)."""
    for name, key in ctx.communicators.items():
        if key == (group.name, implementation):
            return name
    name = ctx.new_name(f"dace_comm_{group.name}_{implementation.lower()}")
    ctx.containers[name] = data.Scalar(dtypes.int64, transient=False)
    ctx.communicators[name] = (group.name, implementation)
    return name


def _is_contiguous(t: TensorValue) -> bool:
    if t.is_view:
        return False
    for size, stride, expected in zip(t.tshape, t.tstrides, contiguous_strides(t.tshape)):
        if size != 1 and sympy.simplify(sympy.sympify(stride) - sympy.sympify(expected)) != 0:
            return False
    return True


def _contiguous(ctx: LoweringContext, t: TensorValue) -> TensorValue:
    """``t``, or a contiguous copy of it."""
    if _is_contiguous(t):
        return t
    copy = ctx.add_array(t.name + "_contiguous", t.tshape, t.torch_dtype, device=t.device)
    ctx.emit_copy(t, copy)
    return copy


def _output(ctx: LoweringContext, node, val=None) -> TensorValue:
    out = ctx.add_tensor_like("t_" + node.name, _val(node) if val is None else val)
    if not _is_contiguous(out):
        raise UnsupportedOpError(node.target, "the result of a collective must be contiguous")
    return out


def _int_scalar(ctx: LoweringContext, prefix: str, value: Any) -> str:
    """A transient ``int32`` scalar set to ``value`` (a number or symbolic expression)."""
    name = ctx.new_name(prefix)
    ctx.containers[name] = data.Scalar(dtypes.int32, transient=True)
    ctx.emit_tasklet(prefix, {}, f"__out = {as_sym(value)}", {"__out": Memlet(f"{name}[0]")})
    return name


def _emit(
    ctx: LoweringContext,
    libnode: nodes.LibraryNode,
    group: distributed.GroupInfo,
    implementation: str,
    inputs: dict[str, Memlet],
    outputs: dict[str, Memlet],
) -> None:
    """Emits a communication node over the communicator of ``group``."""
    libnode.implementation = implementation
    comm_name = _communicator(ctx, group, implementation)
    libnode.add_in_connector("_comm")
    inputs = dict(inputs, _comm=Memlet(f"{comm_name}[0]"))
    for conn in list(libnode.in_connectors):
        if conn not in inputs:
            libnode.remove_in_connector(conn)
    for conn in list(libnode.out_connectors):
        if conn not in outputs:
            libnode.remove_out_connector(conn)
    for conn in inputs:
        libnode.add_in_connector(conn)
    for conn in outputs:
        libnode.add_out_connector(conn)
    ctx.emit_library_call(libnode, inputs, outputs)


def _items(value: Any) -> list[Any]:
    return list(value.items) if isinstance(value, TupleValue) else list(value)


def _check_group_size(node, group: distributed.GroupInfo, group_size: Any) -> None:
    if int(as_const(group_size)) != group.size:
        raise UnsupportedOpError(node.target, f"group size {group_size} differs from that of group {group.name}")


# ---------------------------------------------------------------------------------------------- collectives
def _all_reduce(ctx: LoweringContext, node, tensor: TensorValue, val, op: str, group) -> TensorValue:
    tensor = _contiguous(ctx, tensor)
    out = _output(ctx, node, val)
    _emit(
        ctx,
        comm.Allreduce(sanitize_name(node.name), op=op),
        group,
        distributed.implementation_for(tensor.device),
        {"_inbuffer": tensor.memlet()},
        {"_outbuffer": out.memlet()},
    )
    return out


@register_lowering(*resolve("_c10d_functional.all_reduce.default"))
def lower_all_reduce(ctx: LoweringContext, node, tensor: TensorValue, reduce_op, group_name):
    return _all_reduce(ctx, node, tensor, _val(node), _reduce_op(reduce_op), _group(group_name))


@register_lowering(*resolve("_c10d_functional.all_reduce_coalesced.default"))
def lower_all_reduce_coalesced(ctx: LoweringContext, node, tensors, reduce_op, group_name):
    op, group = _reduce_op(reduce_op), _group(group_name)
    return TupleValue([_all_reduce(ctx, node, t, v, op, group) for t, v in zip(_items(tensors), _val(node))])


def _all_gather(ctx: LoweringContext, node, tensor: TensorValue, val, group) -> TensorValue:
    tensor = _contiguous(ctx, tensor)
    out = _output(ctx, node, val)
    _emit(
        ctx,
        comm.Allgather(sanitize_name(node.name)),
        group,
        distributed.implementation_for(tensor.device),
        {"_inbuffer": tensor.memlet()},
        {"_outbuffer": out.memlet()},
    )
    return out


@register_lowering(
    *resolve(
        "_c10d_functional.all_gather_into_tensor.default",
        "_c10d_functional_autograd.all_gather_into_tensor.default",
    )
)
def lower_all_gather_into_tensor(ctx: LoweringContext, node, tensor: TensorValue, group_size, group_name):
    group = _group(group_name)
    _check_group_size(node, group, group_size)
    return _all_gather(ctx, node, tensor, _val(node), group)


@register_lowering(*resolve("_c10d_functional.all_gather_into_tensor_coalesced.default"))
def lower_all_gather_into_tensor_coalesced(ctx: LoweringContext, node, tensors, group_size, group_name):
    group = _group(group_name)
    _check_group_size(node, group, group_size)
    return TupleValue([_all_gather(ctx, node, t, v, group) for t, v in zip(_items(tensors), _val(node))])


def _reduce_scatter(ctx: LoweringContext, node, tensor: TensorValue, val, op: str, group) -> TensorValue:
    tensor = _contiguous(ctx, tensor)
    out = _output(ctx, node, val)
    _emit(
        ctx,
        comm.ReduceScatter(sanitize_name(node.name), op=op),
        group,
        distributed.implementation_for(tensor.device),
        {"_inbuffer": tensor.memlet()},
        {"_outbuffer": out.memlet()},
    )
    return out


@register_lowering(
    *resolve(
        "_c10d_functional.reduce_scatter_tensor.default",
        "_c10d_functional_autograd.reduce_scatter_tensor.default",
    )
)
def lower_reduce_scatter_tensor(ctx: LoweringContext, node, tensor: TensorValue, reduce_op, group_size, group_name):
    group = _group(group_name)
    _check_group_size(node, group, group_size)
    return _reduce_scatter(ctx, node, tensor, _val(node), _reduce_op(reduce_op), group)


@register_lowering(*resolve("_c10d_functional.reduce_scatter_tensor_coalesced.default"))
def lower_reduce_scatter_tensor_coalesced(ctx: LoweringContext, node, tensors, reduce_op, group_size, group_name):
    op, group = _reduce_op(reduce_op), _group(group_name)
    _check_group_size(node, group, group_size)
    return TupleValue([_reduce_scatter(ctx, node, t, v, op, group) for t, v in zip(_items(tensors), _val(node))])


@register_lowering(*resolve("_c10d_functional.broadcast.default"))
def lower_broadcast(ctx: LoweringContext, node, tensor: TensorValue, src, group_name):
    """``src`` is a group rank. The broadcast runs in place on a copy of the input."""
    group = _group(group_name)
    out = _output(ctx, node)
    ctx.emit_copy(tensor, out)
    root = _int_scalar(ctx, "root", src)
    _emit(
        ctx,
        comm.Bcast(sanitize_name(node.name)),
        group,
        distributed.implementation_for(tensor.device),
        {"_inbuffer": out.memlet(), "_root": Memlet(f"{root}[0]")},
        {"_outbuffer": out.memlet()},
    )
    return out


@register_lowering(
    *resolve(
        "_c10d_functional.all_to_all_single.default",
        "_c10d_functional_autograd.all_to_all_single.default",
    )
)
def lower_all_to_all_single(
    ctx: LoweringContext, node, tensor: TensorValue, output_split_sizes, input_split_sizes, group_name
):
    """Splits are numbers of rows (along dimension 0) per group rank; none means equal splits."""
    group = _group(group_name)
    tensor = _contiguous(ctx, tensor)
    out = _output(ctx, node)
    implementation = distributed.implementation_for(tensor.device)
    out_splits = [as_sym(s) for s in _items(output_split_sizes or [])]
    in_splits = [as_sym(s) for s in _items(input_split_sizes or [])]
    if not out_splits and not in_splits:
        libnode = comm.Alltoall(sanitize_name(node.name))
    else:
        if len(out_splits) != group.size or len(in_splits) != group.size:
            raise UnsupportedOpError(node.target, f"expected one split size per rank of group {group.name}")
        row = sympy.Integer(1)
        for size in tensor.tshape[1:]:
            row = row * size
        libnode = comm.Alltoallv(
            sanitize_name(node.name),
            send_counts=[s * row for s in in_splits],
            recv_counts=[s * row for s in out_splits],
        )
    _emit(ctx, libnode, group, implementation, {"_inbuffer": tensor.memlet()}, {"_outbuffer": out.memlet()})
    return out


# ---------------------------------------------------------------------------------------------- point to point
def _request(ctx: LoweringContext, implementation: str) -> str:
    """A request container: an ``MPI_Request``, or an integer token that orders an NCCL operation before its wait."""
    name = ctx.new_name("request")
    if implementation == "MPI":
        ctx.containers[name] = data.Array(dtypes.opaque("MPI_Request"), [1], transient=True)
    else:
        ctx.containers[name] = data.Scalar(dtypes.int32, transient=True)
    return name


def _send(ctx: LoweringContext, name: str, tensor: TensorValue, peer, tag, group) -> TensorValue:
    tensor = _contiguous(ctx, tensor)
    implementation = distributed.implementation_for(tensor.device)
    request = _request(ctx, implementation)
    _emit(
        ctx,
        comm.Isend(sanitize_name(name)),
        group,
        implementation,
        {
            "_buffer": tensor.memlet(),
            "_dest": Memlet(f"{_int_scalar(ctx, 'dest', peer)}[0]"),
            "_tag": Memlet(f"{_int_scalar(ctx, 'tag', tag)}[0]"),
        },
        {"_request": Memlet(f"{request}[0]")},
    )
    result = dataclasses.replace(tensor)  # A distinct value for wait_tensor to find
    ctx.pending_p2p[id(result)] = PendingP2P(implementation, request, tensor, receive=False)
    return result


def _receive(ctx: LoweringContext, name: str, val, peer, tag, group) -> TensorValue:
    out = ctx.add_tensor_like("t_" + name, val)
    if not _is_contiguous(out):
        raise UnsupportedOpError("_c10d_functional.irecv", "the received tensor must be contiguous")
    implementation = distributed.implementation_for(out.device)
    request = _request(ctx, implementation)
    _emit(
        ctx,
        comm.Irecv(sanitize_name(name)),
        group,
        implementation,
        {
            "_src": Memlet(f"{_int_scalar(ctx, 'src', peer)}[0]"),
            "_tag": Memlet(f"{_int_scalar(ctx, 'tag', tag)}[0]"),
        },
        {"_buffer": out.memlet(), "_request": Memlet(f"{request}[0]")},
    )
    ctx.pending_p2p[id(out)] = PendingP2P(implementation, request, out, receive=True)
    return out


@register_lowering(*resolve("_c10d_functional.isend.default"))
def lower_isend(ctx: LoweringContext, node, tensor: TensorValue, dst, tag, group_name):
    """``dst`` is a group rank."""
    return _send(ctx, node.name, tensor, dst, tag, _group(group_name))


@register_lowering(*resolve("_c10d_functional.irecv.default"))
def lower_irecv(ctx: LoweringContext, node, tensor: TensorValue, src, tag, group_name):
    """``src`` is a group rank. The result is a new tensor; the argument only gives its shape."""
    return _receive(ctx, node.name, _val(node), src, tag, _group(group_name))


@register_lowering(*resolve("_c10d_functional.batch_p2p_ops.default"))
def lower_batch_p2p_ops(ctx: LoweringContext, node, op_list, peer_list, tag_list, tensors, group_name):
    group = _group(group_name)
    results = []
    for i, (op, peer, tag, tensor, val) in enumerate(
        zip(as_const(op_list), _items(peer_list), _items(tag_list), _items(tensors), _val(node))
    ):
        name = f"{node.name}_{i}"
        if op == "isend":
            results.append(_send(ctx, name, tensor, peer, tag, group))
        elif op == "irecv":
            results.append(_receive(ctx, name, val, peer, tag, group))
        else:
            raise UnsupportedOpError(node.target, f"point-to-point operation {op} is not supported")
    return TupleValue(results)


def _wait(ctx: LoweringContext, pending: PendingP2P) -> None:
    """Emits the ``Wait`` of a pending operation. It reads the buffer, and writes a received one, so that the buffer
    lives until the wait and the readers of a received tensor follow it."""
    wait = comm.Wait(ctx.unique("wait"))
    buffer = pending.buffer
    inputs = {"_request": Memlet(f"{pending.request}[0]"), "_in_buffer": buffer.memlet()}
    outputs = {"_out_buffer": buffer.memlet()} if pending.receive else {}
    wait.implementation = pending.implementation
    for conn in list(wait.out_connectors):
        wait.remove_out_connector(conn)
    for conn in inputs:
        wait.add_in_connector(conn)
    for conn in outputs:
        wait.add_out_connector(conn)
    ctx.emit_library_call(wait, inputs, outputs)


@register_lowering(*resolve("_c10d_functional.wait_tensor.default"))
def lower_wait_tensor(ctx: LoweringContext, node, tensor):
    """Collectives are blocking: their results are ready. A point-to-point result waits for its operation."""
    pending = ctx.pending_p2p.pop(id(tensor), None)
    if pending is not None:
        _wait(ctx, pending)
        return pending.buffer
    return tensor


def wait_pending(ctx: LoweringContext) -> None:
    """Emits the waits of point-to-point operations whose ``wait_tensor`` the graph does not contain."""
    for pending in list(ctx.pending_p2p.values()):
        _wait(ctx, pending)
    ctx.pending_p2p.clear()
