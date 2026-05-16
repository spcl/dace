# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
MemsetLibraryNode representing 0-memsets over contiguous subsets. Each expansion accepts
an optional ``stream`` in-connector so the generated memset is bound to a
caller-provided ``gpuStream_t`` instead of default ``__dace_current_stream``.
"""
from functools import reduce
import operator
from typing import Any, List, Optional, Tuple

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.transformation.transformation import ExpandTransformation
from .. import environments

from dace.libraries.standard.helper import (STREAM_CONN as _STREAM_CONN, add_dynamic_inputs, add_stream_descriptor as
                                            _add_stream_descriptor, extract_stream_and_dynamic_inputs, wire_stream_to as
                                            _wire_stream_to)

# Outer connector name this libnode publishes. Republished as
# ``MemsetLibraryNode.OUTPUT_CONNECTOR_NAME`` so external consumers
# reference a class constant instead of a string.
_OUTPUT_CONNECTOR_NAME = "_mset_out"


def _make_memset_skeleton(
    node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
) -> Tuple[dace.SDFG, dace.SDFGState, str, dace.data.Data, dace.subsets.Range, List[Any], List[Any],
           Optional[dace.data.Data]]:
    """Build the shared SDFG skeleton for the mapped (``ExpandPure``) memset expansion.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: ``(sdfg, state, out_name, out, out_subset, map_lengths,
        out_shape_collapsed, stream_input)``.
    """
    out_name, out, out_subset, dynamic_inputs, stream_input = node.validate(parent_sdfg, parent_state)
    keep = [(e + 1 - b) // s != 1 for (b, e, s) in out_subset]
    out_shape_collapsed = [(e + 1 - b) // s for (b, e, s), k in zip(out_subset, keep) if k]
    out_strides_collapsed = [stride for stride, k in zip(out.strides, keep) if k]

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    sdfg.schedule = dace.dtypes.ScheduleType.Sequential

    state = sdfg.add_state(f"{node.label}_state")
    map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, out_subset, state)
    _add_stream_descriptor(sdfg, stream_input)

    return sdfg, state, out_name, out, out_subset, map_lengths, out_shape_collapsed, stream_input


def _validate_no_dynamic_inputs(node: "MemsetLibraryNode", dynamic_inputs):
    """Reject dynamic scalar inputs on direct-tasklet memset paths.

    Direct-tasklet expansions have no surrounding map to bind dynamic scalar
    inputs; the ``'pure'`` implementation handles that case instead.

    :param node: The memset library node being expanded.
    :param dynamic_inputs: Dynamic scalar inputs found on ``node``.
    :raises NotImplementedError: If any dynamic input is present.
    """
    if dynamic_inputs:
        raise NotImplementedError(
            f"{type(node).__name__} direct-tasklet expansion does not support dynamic input scalars; "
            f"use the 'pure' implementation for this case.")


def _make_cuda_memset_tasklet(node: "MemsetLibraryNode", parent_state: dace.SDFGState,
                              parent_sdfg: dace.SDFG) -> nodes.Tasklet:
    """Build a tasklet emitting ``cudaMemsetAsync(_out, 0, n)``.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: The CUDA memset tasklet.
    """
    out_name, out, out_subset, dynamic_inputs, stream_input = node.validate(parent_sdfg, parent_state)
    _validate_no_dynamic_inputs(node, dynamic_inputs)

    cp_size = reduce(operator.mul, [(e + 1 - b) // s for (b, e, s) in out_subset], 1)
    has_stream = stream_input is not None
    code = (f"cudaMemsetAsync({_OUTPUT_CONNECTOR_NAME}, 0, "
            f"{sym2cpp(cp_size)} * sizeof({out.dtype.ctype}), {_STREAM_CONN});")

    in_conns = {_STREAM_CONN: dace.dtypes.gpuStream_t} if has_stream else {}
    return nodes.Tasklet(node.name,
                         inputs=in_conns,
                         outputs={_OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                         code=code,
                         language=dace.Language.CPP)


def _make_cpu_memset_tasklet(node: "MemsetLibraryNode", parent_state: dace.SDFGState,
                             parent_sdfg: dace.SDFG) -> nodes.Tasklet:
    """Build a tasklet emitting ``memset(_out, 0, n)``.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: The CPU memset tasklet.
    """
    out_name, out, out_subset, dynamic_inputs, _stream = node.validate(parent_sdfg, parent_state)
    _validate_no_dynamic_inputs(node, dynamic_inputs)

    cp_size = reduce(operator.mul, [(e + 1 - b) // s for (b, e, s) in out_subset], 1)
    code = f"memset({_OUTPUT_CONNECTOR_NAME}, 0, {sym2cpp(cp_size)} * sizeof({out.dtype.ctype}));"

    return nodes.Tasklet(node.name,
                         inputs={},
                         outputs={_OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                         code=code,
                         language=dace.Language.CPP)


def select_memset_implementation(node, parent_state, parent_sdfg) -> str:
    """Resolve an ``'Auto'`` ``MemsetLibraryNode`` implementation to a concrete one.

    Returns ``'pure'`` (Sequential element-zero map) in device scope or when
    dynamic scalar inputs are present, since ``cudaMemsetAsync`` cannot be
    issued from a kernel and only the mapped expansion supports dynamic inputs;
    ``'CUDA'`` (``cudaMemsetAsync``) for host-issued GPU-destination memsets;
    otherwise ``'CPU'`` (``std::memset``).

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: One of ``'pure'``, ``'CUDA'``, or ``'CPU'``.
    """
    from dace.sdfg.scope import is_devicelevel_gpu

    out_name, out, out_subset, dynamic_inputs, _stream = node.validate(parent_sdfg, parent_state)

    if is_devicelevel_gpu(parent_sdfg, parent_state, node) or dynamic_inputs:
        return 'pure'

    if out.storage == dace.dtypes.StorageType.GPU_Global:
        return 'CUDA'
    return 'CPU'


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Default expansion: dispatches to the implementation chosen by
    :func:`select_memset_implementation` based on the destination storage,
    dynamic inputs, and the surrounding scope."""
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        impl_name = select_memset_implementation(node, parent_state, parent_sdfg)
        assert impl_name != 'Auto', "select_memset_implementation must not return 'Auto'."
        node.implementation = impl_name
        return MemsetLibraryNode.implementations[impl_name].expansion(node, parent_state, parent_sdfg)


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, _out_subset, map_lengths, _, stream_input = _make_memset_skeleton(
            node, parent_state, parent_sdfg)

        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        outputs = {"_memset_out": dace.memlet.Memlet(f"{out_name}[{','.join(map_params)}]")}
        schedule = (dace.dtypes.ScheduleType.GPU_Device
                    if out.storage == dace.dtypes.StorageType.GPU_Global else dace.dtypes.ScheduleType.Default)
        _, map_entry, _ = state.add_mapped_tasklet(f"{node.label}_tasklet",
                                                   map_rng,
                                                   dict(),
                                                   "_memset_out = 0",
                                                   outputs,
                                                   schedule=schedule,
                                                   external_edges=True)

        if schedule == dace.dtypes.ScheduleType.GPU_Device:
            _wire_stream_to(sdfg, state, map_entry, _STREAM_CONN, stream_input)

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_cuda_memset_tasklet(node, parent_state, parent_sdfg)


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_cpu_memset_tasklet(node, parent_state, parent_sdfg)


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    implementations = {"Auto": ExpandAuto, "pure": ExpandPure, "CUDA": ExpandCUDA, "CPU": ExpandCPU}
    default_implementation = 'Auto'

    # Connector name this libnode publishes. External consumers (tests,
    # other passes) must reference this constant instead of the string
    # literal so a future rename is a single-line change.
    OUTPUT_CONNECTOR_NAME = _OUTPUT_CONNECTOR_NAME

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(
            self, sdfg: dace.SDFG,
            state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range, dict, Optional[dace.data.Data]]:
        """Validate the node's wiring and resolve its output and inputs.

        :param sdfg: The SDFG owning the data descriptors.
        :param state: The state containing this node.
        :returns: ``(out_name, out, out_subset, dynamic_inputs, stream_input)``.
        :raises ValueError: If the node does not have exactly one output edge.
        """
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == _OUTPUT_CONNECTOR_NAME]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{_OUTPUT_CONNECTOR_NAME}`` output edge.")

        oe = data_oes[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        stream_input, dynamic_inputs = extract_stream_and_dynamic_inputs(self, sdfg, state, reserved_conns=())

        return out_name, out, out_subset, dynamic_inputs, stream_input
