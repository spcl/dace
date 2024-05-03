import logging
from typing import List, Tuple

from dace.libraries.ort_api import KernelSession, ExecutableKernelContext
from dace.libraries.ort_api import ORTCAPIInterface

log = logging.getLogger(__name__)


def check_op(sdfg, state, node, cuda=False) -> Tuple[List[bool], List[bool]]:
    """ Check whether a ONNXOp node has an implementation in ORT, and return whether it's inputs
        and outputs are expected to be in host memory.

        :param sdfg: the sdfg of the node.
        :param state: the state of the node.
        :param node: the node to evaluate:
        :return: two lists of booleans. The i-th boolean of first list indicates whether the i-th boolean is expected
                 to be allocated in host memory. The second list is the same, but for the outputs.
        :raises: :class:`~dace.libraries.ort_api.ORTAPIError` if ORT doesn't support the node.
    """
    log.debug(f"Checking node {node}")

    with ORTCAPIInterface() as api,\
            KernelSession(api, cuda=cuda) as session,\
            ExecutableKernelContext(api, session, node.name, node.schema.name) as context:

        for attribute, onnx_attribute in node.schema.attributes.items():
            if hasattr(node, attribute):
                context.add_attribute(attribute, getattr(node, attribute),
                                      onnx_attribute.attribute_type)

        for edge, is_input in node.iter_edges(state):
            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            if is_input:
                context.add_input(edge_dtype)
            else:
                context.add_output(edge_dtype)
        with context.try_create_kernel(1 if cuda else 0) as kernel:
            return kernel.check_io_locations()
