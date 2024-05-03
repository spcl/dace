from typing import Dict

from dace import registry, properties, SDFG, SDFGState, nodes as nd, dtypes
from dace.sdfg import graph
from dace.transformation import transformation
from dace.sdfg import utils as sdutil

from dace.libraries.onnx import ONNXModel
from dace.libraries.onnx.converters import clean_onnx_name


@properties.make_properties
class ConstantDeviceCopyElimination(transformation.SingleStateTransformation):
    """ Move Host to Device copies to SDFG initialization by adding a post_compile_hook
    """

    # pattern matching only checks that the type of the node matches,
    host_node = transformation.PatternNode(nd.AccessNode)
    device_node = transformation.PatternNode(nd.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(ConstantDeviceCopyElimination.host_node,
                                   ConstantDeviceCopyElimination.device_node)
        ]

    def can_be_applied(self,
                       graph: graph.OrderedMultiDiConnectorGraph,
                       expr_index: int,
                       sdfg,
                       permissive: bool = False):

        host_node: nd.AccessNode = self.host_node
        device_node: nd.AccessNode = self.device_node

        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        # the only edge out of the host node must be to the device node
        if graph.out_degree(host_node) > 1:
            return False

        # the only edge into the device node must be from the host node
        if graph.in_degree(device_node) > 1:
            return False

        # host node must be non-transient, device node must be transient
        if host_node.desc(
                sdfg).transient or not device_node.desc(sdfg).transient:
            return False

        # only support GPU for now
        if device_node.desc(sdfg).storage is not dtypes.StorageType.GPU_Global:
            return False

        return host_node.data in sdfg._parent_onnx_model.clean_weights

    def match_to_str(self, graph):
        host_node: nd.AccessNode = self.host_node
        return "Move host-to-device copy of {} to SDFG initialization".format(
            host_node.data)

    def apply(self, state: SDFGState, sdfg: SDFG):
        parent: ONNXModel = sdfg._parent_onnx_model
        host_node: nd.AccessNode = self.host_node
        device_node: nd.AccessNode = self.device_node

        onnx_host_name = find_unclean_onnx_name(parent, host_node.data)
        device_node.desc(sdfg).transient = False

        state.remove_node(host_node)

        parent.weights[device_node.data] = parent.weights[onnx_host_name]


def find_unclean_onnx_name(model: ONNXModel, name: str) -> str:
    unclean_name = [n for n in model.weights if clean_onnx_name(n) == name]
    if len(unclean_name) != 1:
        raise ValueError(f"Could not find unclean name for name {name}")
    return unclean_name[0]
