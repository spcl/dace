# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions.
"""
from typing import Dict, List, Optional, Tuple

import dace
import copy
from dace.sdfg import nodes

STREAM_CONN = "__stream"


def add_stream_descriptor(sdfg: dace.SDFG, stream_input: Optional[dace.data.Data]):
    """Add a single-stream Scalar(gpuStream_t) ``stream`` descriptor to the
    expansion SDFG. The libnode's ``stream`` connector is bound as one
    ``gpuStream_t`` value regardless of how the parent sources it (e.g. as a
    slice of ``gpu_streams[id]``)."""
    if stream_input is None:
        return
    sdfg.add_scalar(STREAM_CONN, dace.dtypes.gpuStream_t, storage=dace.dtypes.StorageType.Register, transient=False)


def wire_stream_to(sdfg: dace.SDFG, state: dace.SDFGState, target: nodes.Node, target_conn: str,
                   stream_input: Optional[dace.data.Data]):
    """Connect the SDFG-level ``stream`` access node to ``target`` on
    ``target_conn``. No-op when no stream is wired. For map entries the
    connector is added on the target; for tasklets it must already exist."""
    if stream_input is None:
        return
    stream_access = state.add_access(STREAM_CONN)
    if isinstance(target, nodes.MapEntry):
        target.add_in_connector(target_conn)
    state.add_edge(stream_access, None, target, target_conn,
                   dace.memlet.Memlet.from_array(STREAM_CONN, sdfg.arrays[STREAM_CONN]))


def wire_stream_through_map(sdfg: dace.SDFG, state: dace.SDFGState, map_entry: nodes.MapEntry, target: nodes.Node,
                            target_conn: str):
    """Wire the SDFG-level ``stream`` access node *through* ``map_entry``
    into ``target.target_conn``: adds ``IN_stream`` / ``OUT_stream``
    pass-through connectors on the map and the two memlet edges
    ``stream → MapEntry.IN_stream`` and ``MapEntry.OUT_stream → target``.

    Used when a Sequential map encloses a stream-using consumer and the
    DaCe convention requires the connector to thread through the scope.
    Caller must have added the ``target_conn`` in-connector on ``target``."""
    stream_access = state.add_access(STREAM_CONN)
    in_conn = f"IN_{STREAM_CONN}"
    out_conn = f"OUT_{STREAM_CONN}"
    map_entry.add_in_connector(in_conn)
    map_entry.add_out_connector(out_conn)
    state.add_edge(stream_access, None, map_entry, in_conn,
                   dace.memlet.Memlet.from_array(STREAM_CONN, sdfg.arrays[STREAM_CONN]))
    state.add_edge(map_entry, out_conn, target, target_conn,
                   dace.memlet.Memlet.from_array(STREAM_CONN, sdfg.arrays[STREAM_CONN]))


def extract_stream_and_dynamic_inputs(
        node: nodes.Node, sdfg: dace.SDFG, state: dace.SDFGState,
        reserved_conns: Tuple[str, ...]) -> Tuple[Optional[dace.data.Data], Dict[str, dace.data.Data]]:
    """Extract the optional ``stream`` descriptor and dynamic scalar inputs of a library node.

    Edges whose ``dst_conn`` is in ``reserved_conns`` or equal to ``STREAM_CONN`` are skipped,
    as are empty-memlet dependency edges. ``stream_input`` is ``None`` when no stream edge is wired.

    :return: ``(stream_input, dynamic_inputs)``.
    """
    stream_ies = [ie for ie in state.in_edges(node) if ie.dst_conn == STREAM_CONN]
    if len(stream_ies) > 1:
        raise ValueError(f"{type(node).__name__} expects at most one '{STREAM_CONN}' input edge.")
    stream_input = sdfg.arrays[stream_ies[0].data.data] if stream_ies else None

    reserved = set(reserved_conns) | {STREAM_CONN}
    dynamic_inputs = {}
    for ie in state.in_edges(node):
        if ie.dst_conn in reserved or ie.data.is_empty():
            continue
        datadesc = sdfg.arrays[ie.data.data]
        if not isinstance(datadesc, dace.data.Scalar):
            raise ValueError("Dynamic inputs (not connected to `_in`) must be scalars.")
        dynamic_inputs[ie.dst_conn] = datadesc

    return stream_input, dynamic_inputs


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dimensions from a (subset, strides) pair.

    Surviving strides are scaled by the subset step (``stride * s``) so they describe the access
    pattern as a view into the parent array -- a no-op for unit-step subsets, and the effective
    per-element distance for strided ones.
    """
    collapsed_shape = []
    collapsed_strides = []
    for (b, e, s), stride in zip(subset, strides):
        length = (e + 1 - b) // s
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
    return collapsed_shape, collapsed_strides


def add_dynamic_inputs(dynamic_inputs: Dict[str, dace.data.Data], sdfg: dace.SDFG, subset: dace.subsets.Range,
                       state: dace.SDFGState) -> List[dace.symbolic.SymExpr]:
    """Promote dynamic map-range inputs to SDFG-level data descriptors.

    For each dynamic input not already in the SDFG (e.g. a runtime-determined array dimension),
    the descriptor is added, existing symbolic references are renamed with a ``sym_`` prefix, and
    a pre-assignment state reads the concrete value into the symbol. No-op when nothing needs
    promoting.

    :return: the collapsed (non-singleton) map lengths after substitution.
    """
    pre_assignments = dict()
    map_lengths = [dace.symbolic.SymExpr((e + 1 - b) // s) for (b, e, s) in subset]

    for dynamic_input_name, datadesc in dynamic_inputs.items():
        if dynamic_input_name in sdfg.arrays:
            continue

        if dynamic_input_name in sdfg.symbols:
            continue

        sdfg.replace(str(dynamic_input_name), "sym_" + str(dynamic_input_name))
        ndesc = copy.deepcopy(datadesc)
        ndesc.transient = False
        sdfg.add_datadesc(dynamic_input_name, ndesc)
        # Should be scalar
        if isinstance(ndesc, dace.data.Scalar):
            pre_assignments["sym_" + dynamic_input_name] = f"{dynamic_input_name}"
        else:
            assert tuple(ndesc.shape) == (1, )
            pre_assignments["sym_" + dynamic_input_name] = f"{dynamic_input_name}[0]"

        new_map_lengths = []
        for ml in map_lengths:
            nml = ml.subs({str(dynamic_input_name): "sym_" + str(dynamic_input_name)})
            new_map_lengths.append(nml)
        map_lengths = new_map_lengths

    if pre_assignments != dict():
        sdfg.add_state_before(state=state, label="pre_assign", is_start_block=True, assignments=pre_assignments)

    collapsed_map_lengths = [ml for ml in map_lengths if ml != 1]
    return collapsed_map_lengths
