# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions.
"""
from typing import Callable, Dict, List, Tuple

import dace
import copy
from dace.sdfg import nodes

# The ambient GPU stream symbol libnode CUDA expansions reference. There is
# no pre-existing global constant for it -- the legacy codegen hardcodes the
# bare literal -- so this is the canonical declaration. The name keeps the
# same expanded IR valid under both the legacy codegen (which declares
# ``__dace_current_stream``) and the experimental codegen (whose type-based
# prelude binds it once the stream scheduler wires the connector).
CURRENT_STREAM_NAME = "__dace_current_stream"


def extract_dynamic_inputs(node: nodes.Node, sdfg: dace.SDFG, state: dace.SDFGState,
                           reserved_conns: Tuple[str, ...]) -> Dict[str, dace.data.Data]:
    """Extract the dynamic scalar inputs of a library node.

    Edges whose ``dst_conn`` is in ``reserved_conns`` or equal to
    ``CURRENT_STREAM_NAME`` are skipped, as are empty-memlet dependency
    edges. The stream is no longer wired into the libnode: expansions emit
    ``CURRENT_STREAM_NAME`` (the legacy ambient-stream name) directly and
    the GPU stream scheduler binds it post-expansion.

    :param node: The library node whose input edges are inspected.
    :param sdfg: The SDFG owning the data descriptors.
    :param state: The state containing ``node``.
    :param reserved_conns: Connector names that are not dynamic inputs.
    :returns: ``{connector: scalar descriptor}`` for each dynamic input.
    :raises ValueError: If a dynamic input is not a scalar.
    """
    reserved = set(reserved_conns) | {CURRENT_STREAM_NAME}
    dynamic_inputs = {}
    for ie in state.in_edges(node):
        if ie.dst_conn in reserved or ie.data.is_empty():
            continue
        datadesc = sdfg.arrays[ie.data.data]
        if not isinstance(datadesc, dace.data.Scalar):
            raise ValueError("Dynamic inputs (not connected to ``_in``) must be scalars.")
        dynamic_inputs[ie.dst_conn] = datadesc

    return dynamic_inputs


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dimensions from a (subset, strides) pair.

    Surviving strides are scaled by the subset step (``stride * s``) so they describe the access
    pattern as a view into the parent array -- a no-op for unit-step subsets, and the effective
    per-element distance for strided ones.

    :param subset: The access range, one ``(begin, end, step)`` per dimension.
    :param strides: The parent array strides, aligned with ``subset``.
    :returns: ``(collapsed_shape, collapsed_strides)`` with singletons removed.
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

    :param dynamic_inputs: Map from connector name to its data descriptor.
    :param sdfg: The expansion SDFG to add descriptors to.
    :param subset: The output access range whose map lengths are derived.
    :param state: The state before which the pre-assignment state is inserted.
    :returns: The collapsed (non-singleton) map lengths after substitution.
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


def auto_dispatch(node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG,
                  select_fn: Callable[[nodes.LibraryNode, dace.SDFGState, dace.SDFG], str], library_cls: type):
    """Dispatch a library node's ``'Auto'`` implementation to the one picked by ``select_fn``.

    Sets ``node.implementation`` to the resolved name so introspection
    (debug output, downstream passes) reflects what was actually picked.

    :param node: the library node being expanded.
    :param parent_state: state containing ``node``.
    :param parent_sdfg: SDFG containing ``parent_state``.
    :param select_fn: callable returning a concrete implementation name (not ``'Auto'``).
    :param library_cls: the library node class with the ``implementations`` dict.
    :returns: whatever the resolved expansion returns.
    """
    impl_name = select_fn(node, parent_state, parent_sdfg)
    assert impl_name != 'Auto', f"{select_fn.__name__} must not return 'Auto'."
    node.implementation = impl_name
    return library_cls.implementations[impl_name].expansion(node, parent_state, parent_sdfg)
