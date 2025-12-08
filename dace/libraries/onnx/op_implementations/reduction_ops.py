# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction operations for ONNX.

This module contains implementations of reduction operations including:
- ReduceSum, ReduceMean: Standard reductions over specified axes
- ReduceMax, ReduceMin: Min/max reductions
- CumSum: Cumulative sum along an axis
- Sum: Element-wise sum of multiple inputs

"""

import copy
import typing

import dace
import numpy as np
from dace import SDFG, SDFGState
from dace.sdfg.nodes import Node
from dace.sdfg.utils import in_desc_with_name, in_edge_with_name, out_desc_with_name
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.common import iterables_equal
from dace.libraries.onnx.op_implementations.utils import (empty_sdfg_for_node, in_desc_with_name, op_implementation,
                                                          out_desc_with_name, program_for_node)

# ============================================================================
# Cumulative Sum
# ============================================================================


@op_implementation(op="CumSum", name="pure")
class PureCumSum(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        if node.exclusive or node.reverse:
            return False
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axis").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axis = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axis").src.data].numpy().item()

        def prog(x, y):
            y[:] = np.cumsum(x, axis=axis)

        return program_for_node(prog, sdfg, state, node)


# ============================================================================
# ReduceMean Operations
# ============================================================================


@op_implementation(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = None
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                axes = tuple(axes)
        else:
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.mean(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


# ============================================================================
# ReduceSum Operations
# ============================================================================


@op_implementation(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = None
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                axes = tuple(axes)
        else:
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.sum(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


# ============================================================================
# ReduceMax and ReduceMin Operations
# ============================================================================


@op_implementation(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = None
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                axes = tuple(axes)
        else:
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.max(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = None
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                axes = tuple(axes)
        else:
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.min(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


# ============================================================================
# Sum (Multi-input sum)
# ============================================================================


@op_implementation(op="Sum", name="pure")
class PureSum(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # check that all shapes are arrays, and that the shapes are all equal
        shape = None
        for edge in node.iter_inputs_in_onnx_order(state):
            desc = in_desc_with_name(node, state, sdfg, edge.dst_conn)
            if shape is None:
                shape = desc.shape

            if not iterables_equal(shape, desc.shape):
                return False

        if not iterables_equal(shape, out_desc_with_name(node, state, sdfg, "sum").shape):
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.name)
        input_names = []
        for e in node.iter_inputs_in_onnx_order(state):
            new_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, e.dst_conn))
            new_desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, new_desc)
            input_names.append(e.dst_conn)

        new_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "sum"))
        new_desc.transient = False
        nsdfg.add_datadesc("sum", new_desc)

        nstate = nsdfg.add_state()
        # we know all shapes are equal to the output shape
        shape = out_desc_with_name(node, state, sdfg, "sum").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        tasklet, _, _ = nstate.add_mapped_tasklet(
            node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs={f"__{inp}": dace.Memlet(f"{inp}[{index_str}]")
                    for inp in input_names},
            code=f"__sum = {' + '.join(f'__{inp}' for inp in input_names)}",
            outputs={"__sum": dace.Memlet(f"sum[{index_str}]")},
            external_edges=True)

        tasklet.in_connectors = {f"__{inp}": in_desc_with_name(node, state, sdfg, inp).dtype for inp in input_names}
        tasklet.out_connectors = {"__sum": out_desc_with_name(node, state, sdfg, "sum").dtype}
        return nsdfg
