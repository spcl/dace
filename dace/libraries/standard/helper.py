# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions.
"""
from typing import List, Tuple

import dace
import copy

STREAM_CONN = "stream"


def extract_stream_and_dynamic_inputs(node, sdfg, state, reserved_conns: Tuple[str, ...]) -> Tuple[object, dict]:
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


def collapse_shape_and_strides(subset: List[dace.subsets.Range], strides: List[dace.symbolic.SymExpr]):
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


def add_dynamic_inputs(dynamic_inputs, sdfg: dace.SDFG, subset: dace.subsets.Range, state: dace.SDFGState):
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
            assert ndesc.shape == (1, ) or ndesc.shape == [
                1,
            ]
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
