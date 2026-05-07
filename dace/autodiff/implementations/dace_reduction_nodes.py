# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
DaCe Library Node Backward Pass Implementations for Automatic Differentiation.

This module provides backward pass implementations for DaCe standard library nodes
in the automatic differentiation system. Each class implements the BackwardImplementation
interface to compute gradients for specific library operations during reverse-mode
automatic differentiation.

"""

import copy
import typing

# DaCe core imports
import dace
import dace.dtypes as dtypes
import dace.libraries.standard.nodes
from dace import SDFGState, SDFG, Memlet
from dace.sdfg.nodes import Node

# DaCe frontend imports
from dace.frontend.operations import detect_reduction_type
from dace.registry import autoregister_params

# Autodiff imports
from dace.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult, AutoDiffException

# Utility imports
from dace.sdfg.utils import in_desc_with_name, out_desc_with_name


@autoregister_params(node_type=dace.libraries.standard.nodes.Reduce, name="pure")
class ReverseReduce(BackwardImplementation):
    """Backward implementation for DaCe Reduce library nodes.

    Supports Sum, Max, and Min reduction operations. The backward pass distributes
    gradients appropriately based on the reduction type:
    - Sum: Broadcasts gradients uniformly across reduced dimensions
    - Max/Min: Routes gradients only to positions that achieved the extremal value
    """

    @staticmethod
    def backward_can_be_applied(node: Node, state: SDFGState, sdfg: SDFG) -> bool:
        """Check if backward pass can be applied to this reduction node.

        :param node: The reduction node to check.
        :param state: The SDFG state containing the node (unused but required by interface).
        :param sdfg: The SDFG containing the state (unused but required by interface).
        :return: True if backward pass can be applied, False otherwise.
        """
        reduction_type = detect_reduction_type(node.wcr)
        if reduction_type not in (dtypes.ReductionType.Sum, dtypes.ReductionType.Max, dtypes.ReductionType.Min):
            return False

        return True

    @staticmethod
    def backward(forward_node: Node, context: BackwardContext, given_gradients: typing.List[typing.Optional[str]],
                 required_gradients: typing.List[typing.Optional[str]]) -> typing.Tuple[Node, BackwardResult]:
        """Generate the backward pass for a reduction node.

        :param forward_node: The forward reduction node.
        :param context: The backward pass context.
        :param given_gradients: List of gradient names provided to this node.
        :param required_gradients: List of gradient names required by this node.
        :return: Tuple of the backward node and the backward result.
        :raises AutoDiffException: If the node has invalid number of edges.
        """
        reduction_type = detect_reduction_type(forward_node.wcr)

        if len(given_gradients) != 1:
            raise AutoDiffException(f"Invalid SDFG: reduce node {forward_node} should have exactly one output edge, "
                                    f"got {len(given_gradients)} output gradients")

        if len(required_gradients) != 1:
            raise AutoDiffException(f"Invalid SDFG: reduce node {forward_node} should have exactly one input edge, "
                                    f"got {len(required_gradients)} input gradients")

        input_name = next(iter(required_gradients))
        in_desc = in_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, input_name)

        output_name = next(iter(given_gradients))
        out_desc = out_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, output_name)

        all_axes: typing.List[int] = list(range(len(in_desc.shape)))
        reduce_axes: typing.List[int] = all_axes if forward_node.axes is None else forward_node.axes
        non_reduce_axes: typing.List[int] = [i for i in all_axes if i not in reduce_axes]

        result = BackwardResult.empty()

        return ReverseReduce._backward_reduction(forward_node, context, result, reduction_type, input_name, output_name,
                                                 in_desc, out_desc, all_axes, non_reduce_axes)

    @staticmethod
    def _backward_reduction(forward_node: Node, context: BackwardContext, result: BackwardResult,
                            reduction_type: dtypes.ReductionType, input_name: str, output_name: str, in_desc, out_desc,
                            all_axes: typing.List[int],
                            non_reduce_axes: typing.List[int]) -> typing.Tuple[Node, BackwardResult]:
        """Backward pass for Sum/Max/Min reductions.

        - Sum: Broadcasts gradients uniformly across reduced dimensions
        - Max/Min: Routes gradients to positions that achieved the extremal value,
                   split equally among tied values

        :param forward_node: The forward reduction node.
        :param context: The backward pass context.
        :param result: The backward result to populate.
        :param reduction_type: The type of reduction (Sum, Max, or Min).
        :param input_name: Name of the input connector.
        :param output_name: Name of the output connector.
        :param in_desc: Input data descriptor.
        :param out_desc: Output data descriptor.
        :param all_axes: List of all axes indices.
        :param non_reduce_axes: List of axes not being reduced.
        :return: Tuple of the nested SDFG node and the backward result.
        """
        is_extremal = reduction_type in (dtypes.ReductionType.Max, dtypes.ReductionType.Min)
        type_name = {
            dtypes.ReductionType.Sum: "sum",
            dtypes.ReductionType.Max: "max",
            dtypes.ReductionType.Min: "min"
        }[reduction_type]

        sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") + "_")

        rev_input_conn_name = "input_gradient"
        rev_output_conn_name = "output_gradient"

        result.required_grad_names[output_name] = rev_output_conn_name
        result.given_grad_names[input_name] = rev_input_conn_name

        sdfg.add_array(rev_input_conn_name, shape=out_desc.shape, dtype=out_desc.dtype, strides=out_desc.strides)
        sdfg.add_array(rev_output_conn_name, shape=in_desc.shape, dtype=in_desc.dtype, strides=in_desc.strides)

        nsdfg_inputs = {rev_input_conn_name}

        if is_extremal:
            extremal_conn_name = f"input_{type_name}"
            extremal_idx_conn_name = f"input_{type_name}_idx"
            sdfg.add_array(extremal_conn_name, shape=out_desc.shape, dtype=out_desc.dtype, strides=out_desc.strides)
            sdfg.add_array(extremal_idx_conn_name, shape=in_desc.shape, dtype=in_desc.dtype, strides=in_desc.strides)
            nsdfg_inputs.update({extremal_conn_name, extremal_idx_conn_name})

            # Add transient array to count matching elements per output position
            count_arr_name = f"_{type_name}_count"
            sdfg.add_array(count_arr_name, shape=out_desc.shape, dtype=out_desc.dtype, transient=True)

        reduce_all_axes = forward_node.axes is None or set(range(len(in_desc.shape))) == set(forward_node.axes)

        if is_extremal:
            # Two-state approach for max/min:
            # State 1: Count elements matching extremal value
            # State 2: Compute normalized gradient

            count_state = sdfg.add_state(f"count_{type_name}_{id(forward_node)}")
            grad_state = sdfg.add_state(f"grad_{type_name}_{id(forward_node)}")
            sdfg.add_edge(count_state, grad_state, dace.InterstateEdge())

            # State 1: Count matching elements
            count_memlet = Memlet.simple(count_arr_name,
                                         "0" if reduce_all_axes else ",".join("i" + str(i) for i in non_reduce_axes),
                                         wcr_str="lambda x, y: x + y")
            extremal_val_memlet_count = Memlet.simple(
                extremal_conn_name, "0" if reduce_all_axes else ",".join("i" + str(i) for i in non_reduce_axes))
            extremal_idx_memlet_count = Memlet.simple(extremal_idx_conn_name, ",".join("i" + str(i) for i in all_axes))

            _, _, count_exit = count_state.add_mapped_tasklet(
                f"_count_{type_name}_matches_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__extremal_val": extremal_val_memlet_count,
                    "__extremal_val_idx": extremal_idx_memlet_count
                },
                "__count = 1.0 if __extremal_val == __extremal_val_idx else 0.0", {"__count": count_memlet},
                external_edges=True)

            # Set count array to zero before accumulation
            count_out_edges = count_state.out_edges(count_exit)
            if len(count_out_edges) == 1:
                count_out_node = count_out_edges[0].dst
                if isinstance(count_out_node, dace.nodes.AccessNode):
                    count_out_node.setzero = True

            # State 2: Compute normalized gradient (grad / count)
            reduction_memlet = Memlet.simple(
                rev_input_conn_name, "0" if reduce_all_axes else ",".join("i" + str(i) for i in non_reduce_axes))
            reverse_reduction_memlet = Memlet.simple(rev_output_conn_name,
                                                     ",".join("i" + str(i) for i in all_axes),
                                                     wcr_str="lambda x, y: x + y")
            extremal_val_memlet = Memlet.simple(
                extremal_conn_name, "0" if reduce_all_axes else ",".join("i" + str(i) for i in non_reduce_axes))
            extremal_idx_memlet = Memlet.simple(extremal_idx_conn_name, ",".join("i" + str(i) for i in all_axes))
            count_read_memlet = Memlet.simple(
                count_arr_name, "0" if reduce_all_axes else ",".join("i" + str(i) for i in non_reduce_axes))

            tasklet_inputs = {
                "__in": reduction_memlet,
                "__extremal_val": extremal_val_memlet,
                "__extremal_val_idx": extremal_idx_memlet,
                "__count": count_read_memlet
            }
            tasklet_code = "__out = __in / __count if __extremal_val == __extremal_val_idx else 0"

            _, _, exit_map = grad_state.add_mapped_tasklet(f"_{type_name}_grad_" +
                                                           str(reduction_type).replace(".", "_") + "_", {
                                                               "i" + str(i): "0:{}".format(shape)
                                                               for i, shape in enumerate(in_desc.shape)
                                                           },
                                                           tasklet_inputs,
                                                           tasklet_code, {"__out": reverse_reduction_memlet},
                                                           external_edges=True)

            state = grad_state
        else:
            # Sum reduction: simple broadcast
            state = sdfg.add_state(f"block_{id(forward_node)}")
            reduction_memlet = Memlet.simple(
                rev_input_conn_name, "0" if reduce_all_axes else ",".join("i" + str(i) for i in non_reduce_axes))
            reverse_reduction_memlet = Memlet.simple(rev_output_conn_name,
                                                     ",".join("i" + str(i) for i in all_axes),
                                                     wcr_str="lambda x, y: x + y")
            tasklet_inputs = {"__in": reduction_memlet}
            tasklet_code = "__out = __in"

            _, _, exit_map = state.add_mapped_tasklet(f"_{type_name}_grad_" + str(reduction_type).replace(".", "_") +
                                                      "_", {
                                                          "i" + str(i): "0:{}".format(shape)
                                                          for i, shape in enumerate(in_desc.shape)
                                                      },
                                                      tasklet_inputs,
                                                      tasklet_code, {"__out": reverse_reduction_memlet},
                                                      external_edges=True)

        nsdfg = context.backward_state.add_nested_sdfg(sdfg, nsdfg_inputs, {rev_output_conn_name})

        out_edges = state.out_edges(exit_map)
        if len(out_edges) != 1:
            raise AutoDiffException(f"Expected exactly one output edge from map exit, got {len(out_edges)}")
        out_edge = out_edges[0]
        out_node = out_edge.dst
        if not isinstance(out_node, dace.nodes.AccessNode):
            raise AutoDiffException(f"Expected AccessNode as output, got {type(out_node)}")
        out_node.setzero = True

        if not is_extremal:
            return nsdfg, result

        backward_state = context.backward_state
        fwd_in_edges = context.forward_state.in_edges(forward_node)
        if len(fwd_in_edges) != 1:
            raise AutoDiffException(f"Expected exactly one input edge to forward node, got {len(fwd_in_edges)}")
        fwd_in_edge = fwd_in_edges[0]
        fwd_in_node = fwd_in_edge.src
        if not isinstance(fwd_in_node, dace.nodes.AccessNode):
            raise AutoDiffException(f"Expected AccessNode as input source, got {type(fwd_in_node)}")

        # Register forward input array for data forwarding (in case it's overwritten)
        if fwd_in_node.data not in context.backward_generator.backward_input_arrays:
            data_desc = copy.deepcopy(context.forward_sdfg.arrays[fwd_in_node.data])
            context.backward_generator.backward_input_arrays[fwd_in_node.data] = data_desc

        bwd_read = backward_state.add_read(fwd_in_node.data)
        backward_state.add_edge(bwd_read, None, nsdfg, extremal_idx_conn_name, copy.deepcopy(fwd_in_edge.data))

        if isinstance(context.forward_sdfg.arrays[fwd_in_node.data], (dace.data.View, dace.data.ArrayView)):
            in_edge = context.forward_state.in_edges(fwd_in_node)
            if len(in_edge) != 1:
                raise AutoDiffException(f"Expected exactly one input edge to view node, got {len(in_edge)}")
            in_edge = in_edge[0]
            in_node = in_edge.src
            if isinstance(in_node, dace.nodes.AccessNode):
                if isinstance(context.forward_sdfg.arrays[in_node.data], (dace.data.View, dace.data.ArrayView)):
                    raise AutoDiffException(f"Nested views are not supported: {in_node.data}")
                bwd_in_read = backward_state.add_read(in_node.data)
                backward_state.add_edge(bwd_in_read, None, bwd_read, "views", copy.deepcopy(in_edge.data))

        fwd_out_edges = context.forward_state.out_edges(forward_node)
        if len(fwd_out_edges) != 1:
            raise AutoDiffException(f"Expected exactly one output edge from forward node, got {len(fwd_out_edges)}")
        fwd_out_edge = fwd_out_edges[0]
        fwd_out_node = fwd_out_edge.dst
        if not isinstance(fwd_out_node, dace.nodes.AccessNode):
            raise AutoDiffException(f"Expected AccessNode as output destination, got {type(fwd_out_node)}")

        # Register forward output array for data forwarding (in case it's overwritten)
        if fwd_out_node.data not in context.backward_generator.backward_input_arrays:
            data_desc = copy.deepcopy(context.forward_sdfg.arrays[fwd_out_node.data])
            context.backward_generator.backward_input_arrays[fwd_out_node.data] = data_desc

        bwd_out_read = backward_state.add_read(fwd_out_node.data)
        backward_state.add_edge(bwd_out_read, None, nsdfg, extremal_conn_name, copy.deepcopy(fwd_out_edge.data))

        if isinstance(context.forward_sdfg.arrays[fwd_out_node.data], (dace.data.View, dace.data.ArrayView)):
            out_edge = context.forward_state.out_edges(fwd_out_node)
            if len(out_edge) != 1:
                raise AutoDiffException(f"Expected exactly one output edge from view node, got {len(out_edge)}")
            out_edge = out_edge[0]
            out_node = out_edge.dst
            if isinstance(out_node, dace.nodes.AccessNode):
                if isinstance(context.forward_sdfg.arrays[out_node.data], (dace.data.View, dace.data.ArrayView)):
                    raise AutoDiffException(f"Nested views are not supported: {out_node.data}")
                bwd_in_read = backward_state.add_read(out_node.data)
                backward_state.add_edge(bwd_in_read, None, bwd_out_read, "views", copy.deepcopy(out_edge.data))

        return nsdfg, result
