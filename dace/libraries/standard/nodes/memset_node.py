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

from dace.libraries.standard.helper import (STREAM_CONN as _STREAM_CONN, add_dynamic_inputs,
                                            extract_stream_and_dynamic_inputs)
from dace.libraries.standard.nodes.copy_node import _add_stream_descriptor, _wire_stream_to


def _make_memset_skeleton(
    node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
) -> Tuple[dace.SDFG, dace.SDFGState, str, dace.data.Data, dace.subsets.Range, List[Any], List[Any],
           Optional[dace.data.Data]]:
    """Shared SDFG skeleton for every memset expansion."""
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


def _make_memset_memcpy_tasklet(sdfg: dace.SDFG, state: dace.SDFGState, out_name: str, out: dace.data.Data,
                                out_shape_collapsed: List[Any], cp_size: Any, stream_input: Optional[dace.data.Data],
                                cuda: bool):
    """Emit a ``memset`` / ``cudaMemsetAsync`` tasklet writing zeros to ``out``."""
    tasklet_inputs = set()
    if cuda:
        stream_expr = _STREAM_CONN if stream_input is not None else "__dace_current_stream"
        if stream_input is not None:
            tasklet_inputs.add(_STREAM_CONN)
        code = (f"cudaMemsetAsync(_memset_out, 0, {sym2cpp(cp_size)} * sizeof({out.dtype.ctype}), "
                f"{stream_expr});")
    else:
        code = f"memset(_memset_out, 0, {sym2cpp(cp_size)} * sizeof({out.dtype.ctype}));"

    out_access = state.add_access(out_name)
    tasklet = state.add_tasklet(name="memset_tasklet",
                                inputs=tasklet_inputs,
                                outputs={"_memset_out"},
                                code=code,
                                language=dace.Language.CPP)
    state.add_edge(
        tasklet, "_memset_out", out_access, None,
        dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in out_shape_collapsed])))
    if cuda:
        _wire_stream_to(sdfg, state, tasklet, _STREAM_CONN, stream_input)


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
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, _out_subset, map_lengths, out_shape_collapsed, stream_input = (
            _make_memset_skeleton(node, parent_state, parent_sdfg))
        cp_size = reduce(operator.mul, map_lengths, 1)
        _make_memset_memcpy_tasklet(sdfg, state, out_name, out, out_shape_collapsed, cp_size, stream_input, cuda=True)
        return sdfg


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, _out_subset, map_lengths, out_shape_collapsed, stream_input = (
            _make_memset_skeleton(node, parent_state, parent_sdfg))
        cp_size = reduce(operator.mul, map_lengths, 1)
        _make_memset_memcpy_tasklet(sdfg, state, out_name, out, out_shape_collapsed, cp_size, stream_input, cuda=False)
        return sdfg


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    implementations = {"pure": ExpandPure, "CUDA": ExpandCUDA, "CPU": ExpandCPU}
    default_implementation = 'pure'

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def validate(
            self, sdfg: dace.SDFG,
            state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range, dict, Optional[dace.data.Data]]:
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == "_out"]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one `_out` output edge.")

        oe = data_oes[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        stream_input, dynamic_inputs = extract_stream_and_dynamic_inputs(self, sdfg, state, reserved_conns=())

        return out_name, out, out_subset, dynamic_inputs, stream_input
