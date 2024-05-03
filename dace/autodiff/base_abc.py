"""
Abstract Base Classes for Autodiff
"""
import abc
import dataclasses
import logging
import typing

import dace.registry
from dace.sdfg import SDFG, SDFGState, nodes as nd

from dace.libraries.onnx.nodes.onnx_op import ONNXOp

log = logging.getLogger(__name__)


class AutoDiffException(Exception):
    """ Base class for all exceptions related to automatic differentiation failures. """
    pass


@dataclasses.dataclass
class BackwardContext:
    """ A tuple holding the graph context required to construct reverse nodes """
    forward_sdfg: SDFG  #: the forward SDFG
    forward_state: SDFGState  #: the forward SDFG state
    backward_sdfg: SDFG  #: the backward SDFG
    backward_state: SDFGState  #: the backward SDFG state
    backward_generator: 'dace.autodiff.BackwardPassGenerator'  #: the backward pass generator


@dataclasses.dataclass
class BackwardResult:
    """ The return type of a differentiated node. It contains the names of the gradients the node calculates and
     requires.
    """

    #: mapping from names of output connectors to the connector name of the gradient for that connector.
    required_grad_names: typing.Dict[typing.Optional[str],
                                     typing.Optional[str]]

    #: mapping from names of input connectors to the connector name of the gradient for that connector.
    given_grad_names: typing.Dict[typing.Optional[str], typing.Optional[str]]

    #: mapping from names of gradients to whether they should be zeroed out on initialization.
    zero_init: typing.Dict[typing.Optional[str], typing.Optional[bool]]

    def __init__(self, required_grad_names, given_grad_names, zero_init=None):
        self.required_grad_names = required_grad_names
        self.given_grad_names = given_grad_names
        self.zero_init = zero_init or {}

    @staticmethod
    def empty():
        return BackwardResult(given_grad_names={},
                              required_grad_names={},
                              zero_init={})


@dace.registry.make_registry
class BackwardImplementation(abc.ABC):
    """ ABC for ONNX op forward implementations.

        This registry accepts two types of registrations.
        The register function expects an argument ``node_type=TYPE`` where ``TYPE`` is the type of node that this
        backward implementation supports.
        It can also take an argument ``op=node_name`` where ``node_name`` is the string of the ONNX op it supports,
        e.g. ``"Conv"``.

        It also expects a ``name`` argument that names the implementation.
    """
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: SDFGState,
                                sdfg: SDFG) -> bool:
        """ Return whether this expansion can be applied.

            :param node: the candidate node.
            :param state: the candidate state.
            :param sdfg: the candidate sdfg.
        """
        return True

    @staticmethod
    @abc.abstractmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[nd.Node, BackwardResult]:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            For each input connector with name ``n`` of the forward in required_grads, the returned backward node must
            add an output connector with name ``required_grads[n]`` that will output the gradient for that input.

            If any input from the forward pass is required, simply add a connector with the same name as the connector
            on the forward node. The input will later be connected as required.

            :param forward_node: the node for which the backward pass should be generated for.
            :param context: the context for this node (see
                            :class:`~dace.autodiff.backward_implementation.BackwardContext`).
            :param given_gradients: The names of outputs of the node that gradients will be connected for.
            :param required_gradients: The names of connectors that gradients should be generated for.
            :return: the reverse node and gradient names
                     (see :class:`~dace.autodiff.backward_implementation.BackwardResult`).
        """
        ...


# register the implementations
import dace.autodiff.implementations


def find_backward_implementation(
        forward_sdfg: SDFG, forward_state: SDFGState,
        node: nd.Node) -> typing.Optional[BackwardImplementation]:
    """ Try to find the backward implementation for ``node``.

        :forward_sdfg: the parent sdfg of the node.
        :forward_state: the parent sdfg state of the node.
        :node: the node to find the implementation for.
        :return: the BackwardImplementation for node if one is registered and can be applied, else node.
    """
    valid_impls = []
    for impl, args in BackwardImplementation.extensions().items():
        if "name" not in args:
            raise ValueError(
                f"Expected name in arguments of implementation {impl}.")

        if "node_type" in args and isinstance(node, args["node_type"]) or (
                isinstance(node, ONNXOp) and "op" in args
                and node.schema.name == args["op"]):

            if impl.backward_can_be_applied(node, forward_state, forward_sdfg):
                valid_impls.append((args["name"], impl))

    if isinstance(node, ONNXOp) and node.backward_implementation:

        implementation = node.backward_implementation
    elif isinstance(node, ONNXOp) and node.default_backward_implementation:
        implementation = node.default_backward_implementation
    else:
        implementation = None

    if implementation:
        filtered_impls = [
            i for name, i in valid_impls if name == implementation
        ]
        if filtered_impls:
            return filtered_impls[0]

        log.warning(
            f"Set backward_implementation {node.backward_implementation} on {node}, but it could not be"
            f" applied. Falling back to default selection.")
    if valid_impls:
        return valid_impls[0][1]
    else:
        return None
