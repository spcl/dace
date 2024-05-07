import abc
import typing

from dace import SDFG, SDFGState
from dace.sdfg.nodes import Node
from dace.registry import make_registry

from dace.libraries.onnx.nodes.onnx_op import ONNXOp


@make_registry
class ONNXForward(abc.ABC):
    """ ABC for ONNX op forward implementations.

        The register function expects an argument `op` containing the ONNX op name (string).
    """

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        """ Return whether this expansion can be applied.

            :param node: the candidate node.
            :param state: the candidate state.
            :param sdfg: the candidate sdfg.
            :return: whether this expansion can be applied.
        """
        return True

    @staticmethod
    @abc.abstractmethod
    def forward(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        """ Expand `node` and return its expansion.

            :param node: the candidate node.
            :param state: the candidate state.
            :param sdfg: the candidate sdfg.
            :return: the expanded node.
        """
        ...

    @classmethod
    def registered_implementations(cls, op_name: str) -> typing.List[typing.Tuple[str, "ONNXForward"]]:
        impls = []
        for impl, args in cls.extensions().items():
            if "op" in args and args["op"] == op_name:
                impls.append((args["name"], impl))

        return impls


# register expansions
import dace.libraries.onnx.op_implementations
