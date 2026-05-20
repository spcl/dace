# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MemsetLibraryNode`` representing 0-memsets over contiguous subsets.

The CUDA expansion emits the ambient ``__dace_current_stream`` symbol; the
GPU stream scheduler binds it post-expansion (legacy codegen declares it
directly), so the libnode carries no stream input connector itself.
"""
from typing import Any, List, Tuple

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation
from .. import environments

from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, add_dynamic_inputs, auto_dispatch,
                                            extract_dynamic_inputs)


def _make_memset_skeleton(
    node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
) -> Tuple[dace.SDFG, dace.SDFGState, str, dace.data.Data, dace.subsets.Range, List[Any], List[Any]]:
    """Build the shared SDFG skeleton for the mapped (``ExpandPure``) memset expansion.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: ``(sdfg, state, out_name, out, out_subset, map_lengths,
        out_shape_collapsed)``.
    """
    out_name, out, out_subset, dynamic_inputs = node.validate(parent_sdfg, parent_state)
    keep = [(e + 1 - b) // s != 1 for (b, e, s) in out_subset]
    out_shape_collapsed = [(e + 1 - b) // s for (b, e, s), k in zip(out_subset, keep) if k]
    out_strides_collapsed = [stride for stride, k in zip(out.strides, keep) if k]

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    sdfg.schedule = dace.dtypes.ScheduleType.Sequential

    state = sdfg.add_state(f"{node.label}_state")
    map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, out_subset, state)

    return sdfg, state, out_name, out, out_subset, map_lengths, out_shape_collapsed


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


def _make_memset_tasklet(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG, *,
                         cuda: bool) -> nodes.Tasklet:
    """Build a direct memset tasklet.

    Emits the stream-bound ``cudaMemsetAsync`` form when ``cuda`` is set,
    otherwise plain ``memset``.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :param cuda: Emit ``cudaMemsetAsync`` (else ``memset``).
    :returns: The memset tasklet.
    """
    _out_name, out, out_subset, dynamic_inputs = node.validate(parent_sdfg, parent_state)
    _validate_no_dynamic_inputs(node, dynamic_inputs)

    nbytes = f"{sym2cpp(out_subset.num_elements())} * sizeof({out.dtype.ctype})"
    if cuda:
        code = f"cudaMemsetAsync({MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, 0, {nbytes}, {CURRENT_STREAM_NAME});"
    else:
        code = f"memset({MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, 0, {nbytes});"

    return nodes.Tasklet(node.name,
                         inputs={},
                         outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
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
    out_name, out, out_subset, dynamic_inputs = node.validate(parent_sdfg, parent_state)

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
        return auto_dispatch(node, parent_state, parent_sdfg, select_memset_implementation, MemsetLibraryNode)


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, _out_subset, map_lengths, _ = _make_memset_skeleton(node, parent_state, parent_sdfg)

        # Inner-tasklet connector name; local to this wrapper SDFG (not the
        # libnode's outer connector, which is ``OUTPUT_CONNECTOR_NAME``).
        inner_out = "_memset_out"
        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        outputs = {inner_out: dace.memlet.Memlet(f"{out_name}[{','.join(map_params)}]")}
        schedule = (dace.dtypes.ScheduleType.GPU_Device
                    if out.storage == dace.dtypes.StorageType.GPU_Global else dace.dtypes.ScheduleType.Default)
        state.add_mapped_tasklet(f"{node.label}_tasklet",
                                 map_rng,
                                 dict(),
                                 f"{inner_out} = 0",
                                 outputs,
                                 schedule=schedule,
                                 external_edges=True)

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_memset_tasklet(node, parent_state, parent_sdfg, cuda=True)


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_memset_tasklet(node, parent_state, parent_sdfg, cuda=False)


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    implementations = {"Auto": ExpandAuto, "pure": ExpandPure, "CUDA": ExpandCUDA, "CPU": ExpandCPU}
    default_implementation = 'Auto'

    # Connector name exposed for library node builders.
    OUTPUT_CONNECTOR_NAME = "_mset_out"

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range, dict]:
        """Validate the node's wiring and resolve its output and inputs.

        :param sdfg: The SDFG owning the data descriptors.
        :param state: The state containing this node.
        :returns: ``(out_name, out, out_subset, dynamic_inputs)``.
        :raises ValueError: If the node does not have exactly one output edge.
        """
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == MemsetLibraryNode.OUTPUT_CONNECTOR_NAME]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}`` output edge.")

        oe = data_oes[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        dynamic_inputs = extract_dynamic_inputs(self, sdfg, state, reserved_conns=())

        return out_name, out, out_subset, dynamic_inputs
