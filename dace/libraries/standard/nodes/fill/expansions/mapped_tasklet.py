# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Element-wise mapped fill: the device-neutral parallel form, and the fallback for everything a
single call cannot express (non-contiguous subsets, non-byte-splat GPU values, device scope)."""

from typing import TYPE_CHECKING

import dace
from dace import library
from dace.libraries.standard.nodes.fill.common import VALUE_CONNECTOR_NAME, make_fill_skeleton, python_literal
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    pass


@library.register_expansion(FillLibraryNode, "pure")
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, map_lengths = make_fill_skeleton(node, parent_state)

        value_info = node.value_descriptor(parent_state)
        if value_info is not None:
            _val_src_name, val_desc, _val_subset = value_info
            # Name the wrapper array after the libnode connector so the nested SDFG connector matches.
            val_name = VALUE_CONNECTOR_NAME
            sdfg.add_array(val_name, [1], val_desc.dtype, val_desc.storage)
            inner_val = "_val"
            inputs = {inner_val: dace.memlet.Memlet(f"{val_name}[0]")}
            value_expr = inner_val
        else:
            inputs = dict()
            value_expr = python_literal(node.value, out.dtype)

        # Must not collide with the wrapper SDFG's parameter array (named after outer connector).
        inner_out = "_out"
        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        # Symbolic bounds, never a rendered string: the range parser splits on ':', so any extent
        # whose spelling carries one comes back as bogus tokens.
        map_rng = {i: (0, s - 1, 1) for i, s in zip(map_params, map_lengths)}
        outputs = {inner_out: dace.memlet.Memlet(f"{out_name}[{','.join(map_params)}]")}
        # Inside a kernel a device-scheduled map would be a nested kernel, so the storage cannot pick the schedule.
        if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
            schedule = dace.dtypes.ScheduleType.Sequential
        elif out.storage == dace.dtypes.StorageType.GPU_Global:
            schedule = dace.dtypes.ScheduleType.GPU_Device
        else:
            schedule = dace.dtypes.ScheduleType.Default
        input_nodes = None
        if value_info is not None and schedule == dace.dtypes.ScheduleType.GPU_Device:
            # A kernel cannot read host memory: the value reaches it as a scalar argument.
            local, _ = sdfg.add_scalar(f"{VALUE_CONNECTOR_NAME}_local", val_desc.dtype, transient=True)
            input_nodes = {local: state.add_access(local)}
            state.add_nedge(state.add_read(val_name), input_nodes[local], dace.memlet.Memlet(f"{val_name}[0]"))
            inputs = {inner_val: dace.memlet.Memlet(f"{local}[0]")}
        entry = state.add_mapped_tasklet(
            f"{node.label}_tasklet",
            map_rng,
            inputs,
            f"{inner_out} = {value_expr}",
            outputs,
            schedule=schedule,
            input_nodes=input_nodes,
            external_edges=True,
        )[1]
        # Expanded after the offload chose its kernels' blocks, so the kernel would keep the codegen default
        if schedule == dace.dtypes.ScheduleType.GPU_Device:
            from dace.transformation.passes.gpu_block_size_selection import BLOCK_THREADS_BY_RANK  # Avoid import loop

            entry.map.gpu_block_size = [BLOCK_THREADS_BY_RANK[1], 1, 1]

        return sdfg
