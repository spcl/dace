import typing

import dace.dtypes as dtypes
import dace.libraries.standard.nodes
from dace import SDFGState, SDFG, Memlet
from dace.frontend.operations import detect_reduction_type
from dace.registry import autoregister_params
from dace.sdfg.nodes import Node

from dace.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult, AutoDiffException
from dace.util import in_desc_with_name, out_desc_with_name, out_edge_with_name


@autoregister_params(node_type=dace.libraries.standard.nodes.Reduce, name="pure")
class ReverseReduce(BackwardImplementation):

    @staticmethod
    def backward_can_be_applied(node: Node, state: SDFGState, sdfg: SDFG) -> bool:
        reduction_type = detect_reduction_type(node.wcr)
        if reduction_type is not dtypes.ReductionType.Sum:
            return False

        return True

    @staticmethod
    def backward(forward_node: Node, context: BackwardContext, given_gradients: typing.List[typing.Optional[str]],
                 required_gradients: typing.List[typing.Optional[str]]) -> typing.Tuple[Node, BackwardResult]:
        reduction_type = detect_reduction_type(forward_node.wcr)

        if len(given_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one output edge".format(forward_node))

        if len(required_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one input edge".format(forward_node))

        input_name = next(iter(required_gradients))
        in_desc = in_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, input_name)

        output_name = next(iter(given_gradients))
        out_desc = out_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, output_name)

        all_axes: typing.List[int] = list(range(len(in_desc.shape)))
        reduce_axes: typing.List[int] = all_axes if forward_node.axes is None else forward_node.axes
        non_reduce_axes: typing.List[int] = [i for i in all_axes if i not in reduce_axes]

        result = BackwardResult.empty()

        if reduction_type is dtypes.ReductionType.Sum:
            # in this case, we need to simply scatter the grad across the axes that were reduced

            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") + "_")
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
            reduce_all_axes = forward_node.axes is None or set(range(len(in_desc.shape))) == set(forward_node.axes)

            _, _, exit_map = state.add_mapped_tasklet(
                "_distribute_grad_" + str(reduction_type).replace(".", "_") + "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__in":
                    Memlet.simple(rev_input_conn_name, "0" if reduce_all_axes else ",".join("i" + str(i)
                                                                                            for i in non_reduce_axes))
                },
                "__out = __in", {
                    "__out":
                    Memlet.simple(
                        rev_output_conn_name, ",".join("i" + str(i) for i in all_axes), wcr_str="lambda x, y: x + y")
                },
                external_edges=True)

            # Get the output AccessNode and setzero
            out_eges = state.out_edges(exit_map)
            assert len(out_eges) == 1
            out_edge = out_eges[0]
            out_node = out_edge.dst
            assert isinstance(out_node, dace.nodes.AccessNode)
            out_node.setzero = True
            return context.backward_state.add_nested_sdfg(sdfg, None, {rev_input_conn_name},
                                                          {rev_output_conn_name}), result
        else:
            raise AutoDiffException("Unsupported reduction type '{}'".format(reduction_type))
