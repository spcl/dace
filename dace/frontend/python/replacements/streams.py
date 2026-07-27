# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Replacements for stream (``dace.define_stream``/``dace.define_streamarray``)
methods.

These implement the *call* surface for streams -- ``stream.push(value)`` and
``stream.pop()`` -- which mirrors the bulk push/pop the code generator already
emits for an access-node-to-access-node copy involving a
:class:`~dace.data.Stream` (see ``codegen.targets.cpu.CPUCodeGen._emit_copy``).

The equivalent statement-level memlet-shift syntax (``A[i] >> ostream(-1)``,
``ostream >> oarray``) is supported by the legacy Python frontend only; the
next-generation frontend deliberately does not carry it forward, so these
methods are the portable spelling of both operations.
"""
from numbers import Number
from typing import Any, Optional, Union

import sympy as sp

from dace import Memlet, SDFG, SDFGState, data, subsets, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.replacements.utils import ProgramVisitor


def _stream_descriptor(sdfg: SDFG, name: str, method: str) -> data.Stream:
    desc = sdfg.arrays.get(name)
    if not isinstance(desc, data.Stream):
        raise DaceSyntaxError(None, None, f"'{name}.{method}' requires a stream, but got {type(desc).__name__}.")
    return desc


def _resolve_count(count: Any, method: str) -> Optional[symbolic.SymbolicType]:
    """Normalize an element-count argument to a constant/symbolic value."""
    if count is None:
        return None
    if isinstance(count, (Number, sp.Basic)) or symbolic.issymbolic(count):
        return count
    raise DaceSyntaxError(None, None, f"'{method}' expects a constant or symbolic element count, but got '{count}'.")


@oprepo.replaces_method('Stream', 'push')
def _stream_push(pv: ProgramVisitor,
                 sdfg: SDFG,
                 state: SDFGState,
                 stream: str,
                 value: Union[str, Number, sp.Expr],
                 count: Any = None) -> str:
    """
    Push data onto a stream.

    :param stream: Name of the stream to push onto.
    :param value: Data container holding the value(s) to push, or a
                  constant/symbolic scalar.
    :param count: Number of pushed elements. Omitted (the default) marks the
                  write dynamic, matching the ``stream(-1)`` memlet-shift form,
                  which is the only correct choice under a conditional or a
                  filtering map.
    :return: The stream's own name. Pushing produces no new container, but
             both frontends key "this call mutates its receiver" off the
             returned name (the legacy frontend promotes it from an input to
             an output of the enclosing scope; the schedule-tree frontend
             recognizes ``result is target`` as an in-place mutation), so
             returning ``[]`` here would drop the write.
    """
    desc = _stream_descriptor(sdfg, stream, 'push')
    num_accesses = _resolve_count(count, f'{stream}.push')
    # The code generator keys stream copies off the memlet's container, so the
    # memlet must describe the *stream* side of the edge in both directions
    # (matching ``newast.ProgramVisitor.visit_TopLevelExpr``).
    memlet = Memlet.simple(stream,
                           subsets.Range.from_array(desc),
                           num_accesses=-1 if num_accesses is None else num_accesses)

    wnode = state.add_write(stream)
    if isinstance(value, str) and value in sdfg.arrays:
        if isinstance(sdfg.arrays[value], data.Stream):
            raise DaceSyntaxError(None, None, f"'{stream}.push' does not support pushing a stream ('{value}').")
        state.add_nedge(state.add_read(value), wnode, memlet)
    elif isinstance(value, (Number, sp.Expr)) or symbolic.issymbolic(value):
        tasklet = state.add_tasklet(f'_push_{stream}_', {}, {'__out'}, f'__out = {symbolic.symstr(value)}')
        state.add_edge(tasklet, '__out', wnode, None, memlet)
    else:
        raise DaceSyntaxError(None, None, f"Unsupported value '{value}' for '{stream}.push'.")

    return stream


@oprepo.replaces_method('Stream', 'pop')
def _stream_pop(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, stream: str, count: Any = None) -> str:
    """
    Drain a stream into a new transient array.

    A stream holds a runtime-determined number of elements, so the result is
    zero-padded: elements past the drained prefix read as zero. Without that
    the returned value would be partly uninitialized, and unlike the legacy
    ``ostream >> oarray`` shift -- which drained into a container the program
    could pre-initialize itself -- a caller of ``pop`` has no opportunity to
    initialize the buffer it is handed.

    :param stream: Name of the stream to pop from.
    :param count: Maximum number of popped elements. Defaults to the stream's
                  buffer size, i.e. the whole stream.
    :return: Name of the transient holding the popped elements.
    """
    desc = _stream_descriptor(sdfg, stream, 'pop')
    num_elements = _resolve_count(count, f'{stream}.pop')
    if num_elements is None:
        num_elements = desc.buffer_size
    if num_elements == 0:
        raise DaceSyntaxError(
            None, None, f"'{stream}.pop' cannot infer how many elements to pop from a stream with no "
            'buffer size; pass an explicit count.')

    name, result = pv.add_temp_transient([num_elements], desc.dtype)
    state.add_mapped_tasklet(f'_pop_init_{stream}_', {'__i0': f'0:{num_elements}'}, {},
                             '__out = 0', {'__out': Memlet.simple(name, '__i0')},
                             external_edges=True)

    # The drain goes into its own state so it orders after the zero-fill.
    state = pv._add_state(f'_pop_{stream}_')
    state.add_nedge(
        state.add_read(stream), state.add_write(name),
        Memlet(data=stream, subset=subsets.Range.from_array(desc), other_subset=subsets.Range.from_array(result)))
    return name


# -------------------------------------------------------------------- #
#  Descriptor inference for streams (schedule-tree frontend)            #
# -------------------------------------------------------------------- #

from dace.frontend.common.op_repository import infers_method_descriptor  # noqa: E402


@infers_method_descriptor('Stream', 'push')
def _infer_stream_push(self_desc: data.Data, value: Any = None, count: Any = None, **_kw):
    del value, count
    if not isinstance(self_desc, data.Stream):
        return None
    return ()


@infers_method_descriptor('Stream', 'pop')
def _infer_stream_pop(self_desc: data.Data, count: Any = None, **_kw):
    if not isinstance(self_desc, data.Stream):
        return None
    num_elements = count if count is not None else self_desc.buffer_size
    if not isinstance(num_elements, (Number, sp.Basic)) and not symbolic.issymbolic(num_elements):
        return None
    if num_elements == 0:
        return None
    return data.Array(self_desc.dtype, [num_elements], transient=True)
