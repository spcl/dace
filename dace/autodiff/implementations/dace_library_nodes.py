import typing

import dace.dtypes as dtypes
import dace.libraries.standard.nodes
from dace import SDFGState, SDFG, Memlet
from dace.frontend.operations import detect_reduction_type
from dace.registry import autoregister_params
from dace.sdfg.nodes import Node
import copy
from dace.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult, AutoDiffException
from dace.util import in_desc_with_name, out_desc_with_name, out_edge_with_name

@autoregister_params(node_type=dace.libraries.standard.nodes.Reduce,
                     name="pure")
class ReverseReduce(BackwardImplementation):

    @staticmethod
    def backward_can_be_applied(node: Node, state: SDFGState,
                                sdfg: SDFG) -> bool:
        reduction_type = detect_reduction_type(node.wcr)
        if reduction_type not in (dtypes.ReductionType.Sum,
                                  dtypes.ReductionType.Max,
                                  dtypes.ReductionType.Min):
            return False

        return True

    @staticmethod
    def backward(
        forward_node: Node, context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[Node, BackwardResult]:
        reduction_type = detect_reduction_type(forward_node.wcr)

        if len(given_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one output edge"
                .format(forward_node))

        if len(required_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one input edge"
                .format(forward_node))

        input_name = next(iter(required_gradients))
        in_desc = in_desc_with_name(forward_node, context.forward_state,
                                    context.forward_sdfg, input_name)

        output_name = next(iter(given_gradients))
        out_desc = out_desc_with_name(forward_node, context.forward_state,
                                      context.forward_sdfg, output_name)

        all_axes: typing.List[int] = list(range(len(in_desc.shape)))
        reduce_axes: typing.List[
            int] = all_axes if forward_node.axes is None else forward_node.axes
        non_reduce_axes: typing.List[int] = [
            i for i in all_axes if i not in reduce_axes
        ]

        result = BackwardResult.empty()

        if reduction_type is dtypes.ReductionType.Sum:
            # in this case, we need to simply scatter the grad across the axes that were reduced

            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") +
                        "_")
            # Create a name with a random number to avoid name clashes
            # TODO: This is not an issue in the backward pass
            # but sometimes simplify will inline these SDFGs and create states with the same label
            state_label = f"block_{id(forward_node)}"
            state = sdfg.add_state(state_label)

            rev_input_conn_name = "input_gradient"
            rev_output_conn_name = "output_gradient"
            result.required_grad_names[output_name] = rev_output_conn_name
            result.given_grad_names[input_name] = rev_input_conn_name

            # It is important to add the strides in the case of accesses to a view where the shape is not enough
            _, rev_input_arr = sdfg.add_array(rev_input_conn_name,
                                              shape=out_desc.shape,
                                              dtype=out_desc.dtype,
                                              strides=out_desc.strides)
            _, rev_output_arr = sdfg.add_array(rev_output_conn_name,
                                               shape=in_desc.shape,
                                               dtype=in_desc.dtype,
                                               strides=in_desc.strides)

            # Make sure the output is set to zero
            reduce_all_axes = forward_node.axes is None or set(
                range(len(in_desc.shape))) == set(forward_node.axes)

            _, _, exit_map = state.add_mapped_tasklet(
                "_distribute_grad_" + str(reduction_type).replace(".", "_") +
                "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__in":
                    Memlet.simple(
                        rev_input_conn_name,
                        "0" if reduce_all_axes else ",".join(
                            "i" + str(i) for i in non_reduce_axes))
                },
                "__out = __in", {
                    "__out":
                    Memlet.simple(rev_output_conn_name,
                                  ",".join("i" + str(i) for i in all_axes),
                                  wcr_str="lambda x, y: x + y")
                },
                external_edges=True)

            # Get the output AccessNode and setzero
            out_eges = state.out_edges(exit_map)
            assert len(out_eges) == 1
            out_edge = out_eges[0]
            out_node = out_edge.dst
            assert isinstance(out_node, dace.nodes.AccessNode)
            out_node.setzero = True
            return context.backward_state.add_nested_sdfg(
                sdfg, {rev_input_conn_name},
                {rev_output_conn_name}), result

        # TODO: Remove code duplication between Max and Min reductions
        elif reduction_type is dtypes.ReductionType.Max:

            # In this case, we need to get the index of the minimum value
            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") +
                        "_")
            # Create a name with a random number to avoid name clashes
            # TODO: This is not an issue in the backward pass
            # but sometimes simplify will inline these SDFGs and create states with the same label
            state_label = f"block_{id(forward_node)}"
            state = sdfg.add_state(state_label)

            rev_input_conn_name = "input_gradient"
            rev_output_conn_name = "output_gradient"
            max_conn_name = "input_max"
            max_idx_conn_name = "input_max_idx"
            rev_output_conn_name = "output_gradient"
            result.required_grad_names[output_name] = rev_output_conn_name
            result.given_grad_names[input_name] = rev_input_conn_name

            # It is important to add the strides in the case of accesses to a view where the shape is not enough
            _, rev_input_arr = sdfg.add_array(rev_input_conn_name,
                                              shape=out_desc.shape,
                                              dtype=out_desc.dtype,
                                              strides=out_desc.strides)

            _, rev_output_arr = sdfg.add_array(rev_output_conn_name,
                                               shape=in_desc.shape,
                                               dtype=in_desc.dtype,
                                               strides=in_desc.strides)

            # Get the forwarded data descriptors for the max operation and add them to the new SDFG
            _, max_desc = sdfg.add_array(max_conn_name,
                                         shape=out_desc.shape,
                                         dtype=out_desc.dtype,
                                         strides=out_desc.strides)

            _, max_idx_desc = sdfg.add_array(max_idx_conn_name,
                                             shape=in_desc.shape,
                                             dtype=in_desc.dtype,
                                             strides=in_desc.strides)

            reduce_all_axes = forward_node.axes is None or set(
                range(len(in_desc.shape))) == set(forward_node.axes)

            # prepare memlets for the tasklet
            reduction_memlet = Memlet.simple(
                rev_input_conn_name,
                "0" if reduce_all_axes else ",".join("i" + str(i)
                                                     for i in non_reduce_axes))
            reverse_reduction_memelt = Memlet.simple(
                rev_output_conn_name,
                ",".join("i" + str(i) for i in all_axes),
                wcr_str="lambda x, y: x + y")

            max_val_memlet = Memlet.simple(
                max_conn_name,
                "0" if reduce_all_axes else ",".join("i" + str(i)
                                                     for i in non_reduce_axes))

            max_val_index_memlet = Memlet.simple(
                max_idx_conn_name, ",".join("i" + str(i) for i in all_axes))
            tasklet_code = "__out = __in if __max_val == __max_val_idx else 0"
            _, _, exit_map = state.add_mapped_tasklet(
                "_max_grad_" + str(reduction_type).replace(".", "_") + "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__in": reduction_memlet,
                    "__max_val": max_val_memlet,
                    "__max_val_idx": max_val_index_memlet
                },
                tasklet_code, {"__out": reverse_reduction_memelt},
                external_edges=True)

            # Add the nested SDFG to the backward state
            nsdfg = context.backward_state.add_nested_sdfg(
                sdfg,
                {rev_input_conn_name, max_conn_name, max_idx_conn_name},
                {rev_output_conn_name})

            # Get the output AccessNode and setzero
            out_eges = state.out_edges(exit_map)
            assert len(out_eges) == 1
            out_edge = out_eges[0]
            out_node = out_edge.dst
            assert isinstance(out_node, dace.nodes.AccessNode)
            out_node.setzero = True

            # We need to manually add the required inputs here
            backward_state = context.backward_state
            fwd_in_edges = context.forward_state.in_edges(forward_node)
            assert len(fwd_in_edges) == 1
            fwd_in_edge = fwd_in_edges[0]
            fwd_in_node = fwd_in_edge.src
            assert isinstance(fwd_in_node, dace.nodes.AccessNode)
            bwd_read = backward_state.add_read(fwd_in_node.data)
            backward_state.add_edge(bwd_read, None, nsdfg, max_idx_conn_name,
                                    copy.deepcopy(fwd_in_edge.data))
            # if this is a view
            if isinstance(context.forward_sdfg.arrays[fwd_in_node.data],
                          (dace.data.View, dace.data.ArrayView)):
                # Get incoming edge
                in_edge = context.forward_state.in_edges(fwd_in_node)
                assert len(in_edge) == 1
                in_edge = in_edge[0]
                in_node = in_edge.src
                if isinstance(in_node, dace.nodes.AccessNode):
                    # Make sure this is not a view iteself
                    assert not isinstance(
                        context.forward_sdfg.arrays[in_node.data],
                        (dace.data.View, dace.data.ArrayView))
                    # Add the read node
                    bwd_in_read = backward_state.add_read(in_node.data)
                    # Add the edge
                    backward_state.add_edge(bwd_in_read, None,
                                            bwd_read, "views",
                                            copy.deepcopy(in_edge.data))

            fwd_out_edges = context.forward_state.out_edges(forward_node)
            assert len(fwd_out_edges) == 1
            fwd_out_edge = fwd_out_edges[0]
            fwd_out_node = fwd_out_edge.dst
            assert isinstance(fwd_out_node, dace.nodes.AccessNode)
            bwd_out_read = backward_state.add_read(fwd_out_node.data)
            backward_state.add_edge(bwd_out_read, None, nsdfg, max_conn_name,
                                    copy.deepcopy(fwd_out_edge.data))
            # avoid ambigous views
            # if this is a view
            if isinstance(context.forward_sdfg.arrays[fwd_out_node.data],
                          (dace.data.View, dace.data.ArrayView)):
                # Get incoming edge
                out_edge = context.forward_state.out_edges(fwd_out_node)
                assert len(out_edge) == 1
                out_edge = out_edge[0]
                out_node = out_edge.dst
                if isinstance(out_node, dace.nodes.AccessNode):
                    # Make sure this is not a view iteself
                    assert not isinstance(
                        context.forward_sdfg.arrays[out_node.data],
                        (dace.data.View, dace.data.ArrayView))
                    # Add the read node
                    bwd_in_read = backward_state.add_read(out_node.data)
                    # Add the edge
                    backward_state.add_edge(bwd_in_read, None, bwd_out_read,
                                            "views",
                                            copy.deepcopy(out_edge.data))
            return nsdfg, result
        elif reduction_type is dtypes.ReductionType.Min:

            # In this case, we need to get the index of the minimum value
            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") +
                        "_")
            # Create a name with a random number to avoid name clashes
            # TODO: This is not an issue in the backward pass
            # but sometimes simplify will inline these SDFGs and create states with the same label
            state_label = f"block_{id(forward_node)}"
            state = sdfg.add_state(state_label)

            rev_input_conn_name = "input_gradient"
            rev_output_conn_name = "output_gradient"
            min_conn_name = "input_min"
            min_idx_conn_name = "input_min_idx"
            rev_output_conn_name = "output_gradient"
            result.required_grad_names[output_name] = rev_output_conn_name
            result.given_grad_names[input_name] = rev_input_conn_name

            # It is important to add the strides in the case of accesses to a view where the shape is not enough
            _, rev_input_arr = sdfg.add_array(rev_input_conn_name,
                                              shape=out_desc.shape,
                                              dtype=out_desc.dtype,
                                              strides=out_desc.strides)

            _, rev_output_arr = sdfg.add_array(rev_output_conn_name,
                                               shape=in_desc.shape,
                                               dtype=in_desc.dtype,
                                               strides=in_desc.strides)

            # Get the forwarded data descriptors for the min operation and add them to the new SDFG
            _, min_desc = sdfg.add_array(min_conn_name,
                                         shape=out_desc.shape,
                                         dtype=out_desc.dtype,
                                         strides=out_desc.strides)

            _, min_idx_desc = sdfg.add_array(min_idx_conn_name,
                                             shape=in_desc.shape,
                                             dtype=in_desc.dtype,
                                             strides=in_desc.strides)

            reduce_all_axes = forward_node.axes is None or set(
                range(len(in_desc.shape))) == set(forward_node.axes)

            # prepare memlets for the tasklet
            reduction_memlet = Memlet.simple(
                rev_input_conn_name,
                "0" if reduce_all_axes else ",".join("i" + str(i)
                                                     for i in non_reduce_axes))
            reverse_reduction_memelt = Memlet.simple(
                rev_output_conn_name,
                ",".join("i" + str(i) for i in all_axes),
                wcr_str="lambda x, y: x + y")

            min_val_memlet = Memlet.simple(
                min_conn_name,
                "0" if reduce_all_axes else ",".join("i" + str(i)
                                                     for i in non_reduce_axes))

            min_val_index_memlet = Memlet.simple(
                min_idx_conn_name, ",".join("i" + str(i) for i in all_axes))
            tasklet_code = "__out = __in if __min_val == __min_val_idx else 0"
            _, _, exit_map = state.add_mapped_tasklet(
                "_min_grad_" + str(reduction_type).replace(".", "_") + "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__in": reduction_memlet,
                    "__min_val": min_val_memlet,
                    "__min_val_idx": min_val_index_memlet
                },
                tasklet_code, {"__out": reverse_reduction_memelt},
                external_edges=True)

            # Add the nested SDFG to the backward state
            nsdfg = context.backward_state.add_nested_sdfg(
                sdfg,
                {rev_input_conn_name, min_conn_name, min_idx_conn_name},
                {rev_output_conn_name})

            # Get the output AccessNode and setzero
            out_eges = state.out_edges(exit_map)
            assert len(out_eges) == 1
            out_edge = out_eges[0]
            out_node = out_edge.dst
            assert isinstance(out_node, dace.nodes.AccessNode)
            out_node.setzero = True

            # We need to manually add the required inputs here
            backward_state = context.backward_state
            fwd_in_edges = context.forward_state.in_edges(forward_node)
            assert len(fwd_in_edges) == 1
            fwd_in_edge = fwd_in_edges[0]
            fwd_in_node = fwd_in_edge.src
            assert isinstance(fwd_in_node, dace.nodes.AccessNode)
            bwd_read = backward_state.add_read(fwd_in_node.data)
            backward_state.add_edge(bwd_read, None, nsdfg, min_idx_conn_name,
                                    copy.deepcopy(fwd_in_edge.data))
            # if this is a view
            if isinstance(context.forward_sdfg.arrays[fwd_in_node.data],
                          (dace.data.View, dace.data.ArrayView)):
                # Get incoming edge
                in_edge = context.forward_state.in_edges(fwd_in_node)
                assert len(in_edge) == 1
                in_edge = in_edge[0]
                in_node = in_edge.src
                if isinstance(in_node, dace.nodes.AccessNode):
                    # Make sure this is not a view iteself
                    assert not isinstance(
                        context.forward_sdfg.arrays[in_node.data],
                        (dace.data.View, dace.data.ArrayView))
                    # Add the read node
                    bwd_in_read = backward_state.add_read(in_node.data)
                    # Add the edge
                    backward_state.add_edge(bwd_in_read, None,
                                            bwd_read, "views",
                                            copy.deepcopy(in_edge.data))

            fwd_out_edges = context.forward_state.out_edges(forward_node)
            assert len(fwd_out_edges) == 1
            fwd_out_edge = fwd_out_edges[0]
            fwd_out_node = fwd_out_edge.dst
            assert isinstance(fwd_out_node, dace.nodes.AccessNode)
            bwd_out_read = backward_state.add_read(fwd_out_node.data)
            backward_state.add_edge(bwd_out_read, None, nsdfg, min_conn_name,
                                    copy.deepcopy(fwd_out_edge.data))
            # avoid ambigous views
            # if this is a view
            if isinstance(context.forward_sdfg.arrays[fwd_out_node.data],
                          (dace.data.View, dace.data.ArrayView)):
                # Get incoming edge
                out_edge = context.forward_state.out_edges(fwd_out_node)
                assert len(out_edge) == 1
                out_edge = out_edge[0]
                out_node = out_edge.dst
                if isinstance(out_node, dace.nodes.AccessNode):
                    # Make sure this is not a view iteself
                    assert not isinstance(
                        context.forward_sdfg.arrays[out_node.data],
                        (dace.data.View, dace.data.ArrayView))
                    # Add the read node
                    bwd_in_read = backward_state.add_read(out_node.data)
                    # Add the edge
                    backward_state.add_edge(bwd_in_read, None, bwd_out_read,
                                            "views",
                                            copy.deepcopy(out_edge.data))

            return nsdfg, result
        else:
            raise AutoDiffException(
                "Unsupported reduction type '{}'".format(reduction_type))


@autoregister_params(node_type=dace.libraries.standard.nodes.Reduce,
                     name="pure")
class ReverseReduceMax(BackwardImplementation):

    @staticmethod
    def backward_can_be_applied(node: Node, state: SDFGState,
                                sdfg: SDFG) -> bool:
        reduction_type = detect_reduction_type(node.wcr)
        if reduction_type is not dtypes.ReductionType.Max:
            return False
        return True

    @staticmethod
    def backward(
        forward_node: Node, context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[Node, BackwardResult]:
        reduction_type = detect_reduction_type(forward_node.wcr)

        if len(given_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one output edge"
                .format(forward_node))

        if len(required_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one input edge"
                .format(forward_node))

        input_name = next(iter(required_gradients))
        in_desc = in_desc_with_name(forward_node, context.forward_state,
                                    context.forward_sdfg, input_name)

        output_name = next(iter(given_gradients))
        out_desc = out_desc_with_name(forward_node, context.forward_state,
                                      context.forward_sdfg, output_name)

        all_axes: typing.List[int] = list(range(len(in_desc.shape)))
        reduce_axes: typing.List[
            int] = all_axes if forward_node.axes is None else forward_node.axes
        non_reduce_axes: typing.List[int] = [
            i for i in all_axes if i not in reduce_axes
        ]

        result = BackwardResult.empty()

        if reduction_type is dtypes.ReductionType.Sum:
            # in this case, we need to simply scatter the grad across the axes that were reduced

            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") +
                        "_")
            # Create a name with a random number to avoid name clashes
            # TODO: This is not an issue in the backward pass
            # but sometimes simplify will inline these SDFGs and create states with the same label
            state_label = f"block_{id(forward_node)}"
            state = sdfg.add_state(state_label)

            rev_input_conn_name = "input_gradient"
            rev_output_conn_name = "output_gradient"
            result.required_grad_names[output_name] = rev_output_conn_name
            result.given_grad_names[input_name] = rev_input_conn_name

            # It is important to add the strides in the case of accesses to a view where the shape is not enough
            _, rev_input_arr = sdfg.add_array(rev_input_conn_name,
                                              shape=out_desc.shape,
                                              dtype=out_desc.dtype,
                                              strides=out_desc.strides)
            _, rev_output_arr = sdfg.add_array(rev_output_conn_name,
                                               shape=in_desc.shape,
                                               dtype=in_desc.dtype,
                                               strides=in_desc.strides)
            # Make sure the output is set to zero
            reduce_all_axes = forward_node.axes is None or set(
                range(len(in_desc.shape))) == set(forward_node.axes)

            _, _, exit_map = state.add_mapped_tasklet(
                "_distribute_grad_" + str(reduction_type).replace(".", "_") +
                "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__in":
                    Memlet.simple(
                        rev_input_conn_name,
                        "0" if reduce_all_axes else ",".join(
                            "i" + str(i) for i in non_reduce_axes))
                },
                "__out = __in", {
                    "__out":
                    Memlet.simple(rev_output_conn_name,
                                  ",".join("i" + str(i) for i in all_axes),
                                  wcr_str="lambda x, y: x + y")
                },
                external_edges=True)

            # Get the output AccessNode and setzero
            out_eges = state.out_edges(exit_map)
            assert len(out_eges) == 1
            out_edge = out_eges[0]
            out_node = out_edge.dst
            assert isinstance(out_node, dace.nodes.AccessNode)
            out_node.setzero = True
            return context.backward_state.add_nested_sdfg(
                sdfg, {rev_input_conn_name},
                {rev_output_conn_name}), result
        else:
            raise AutoDiffException(
                "Unsupported reduction type '{}'".format(reduction_type))
