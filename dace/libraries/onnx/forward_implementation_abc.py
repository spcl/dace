# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Abstract Base Class for ONNX Operation Forward Implementations.

This module defines the interface that all ONNX operation implementations must
follow in DaCe. It uses a registry pattern to allow multiple implementations
for each ONNX operation, enabling:

- Pure Python implementations for correctness
- Optimized implementations for performance
- Hardware-specific implementations (CPU, GPU, FPGA)
- Custom user-provided implementations

The ONNXForward ABC provides:
- Registration mechanism via @make_registry decorator
- Implementation selection based on applicability
- Expansion of ONNX ops to DaCe SDFG nodes

Implementation Registration:
    Implementations register themselves by inheriting from ONNXForward and
    using the @op_implementation decorator with:
    - `op`: ONNX operation name (e.g., "Conv", "MatMul")
    - `name`: Implementation name (e.g., "pure", "optimized")

Example:
    @op_implementation(op="MatMul", name="pure")
    class PureMatMul(ONNXForward):
        @staticmethod
        def forward(node, state, sdfg):
            # Implementation here
            pass
"""

import abc
import typing

from dace import SDFG, SDFGState
from dace.registry import make_registry
from dace.sdfg.nodes import Node

from dace.libraries.onnx.nodes.onnx_op import ONNXOp


@make_registry
class ONNXForward(abc.ABC):
    """
    Abstract base class for ONNX operation forward implementations.

    This class defines the interface for implementing ONNX operations in DaCe.
    Subclasses must implement the `forward` method to expand an ONNX operation
    node into DaCe SDFG constructs.

    The registry system allows multiple implementations per operation, with
    selection based on applicability criteria.
    """

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        """
        Check whether this implementation can be applied to the given node.

        This method is called during SDFG expansion to determine if this
        implementation is suitable for the given context. The default
        implementation returns True (always applicable).

        Args:
            node: The ONNX operation node to expand.
            state: The SDFG state containing the node.
            sdfg: The parent SDFG.

        Returns:
            True if this implementation can be applied, False otherwise.
        """
        return True

    @staticmethod
    @abc.abstractmethod
    def forward(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        """
        Expand an ONNX operation node into DaCe SDFG constructs.

        This is the main method that must be implemented by subclasses. It takes
        an ONNX operation node and replaces it with equivalent DaCe constructs
        (tasklets, nested SDFGs, library nodes, etc.).

        Args:
            node: The ONNX operation node to expand.
            state: The SDFG state containing the node.
            sdfg: The parent SDFG.

        Returns:
            The expanded node or a nested SDFG representing the operation.
        """
        ...

    @classmethod
    def registered_implementations(cls, op_name: str) -> typing.List[typing.Tuple[str, "ONNXForward"]]:
        """
        Get all registered implementations for a specific ONNX operation.

        Args:
            op_name: The ONNX operation name (e.g., "Conv", "MatMul").

        Returns:
            List of tuples (implementation_name, implementation_class) for
            all registered implementations of the given operation.
        """
        impls = []
        for impl, args in cls.extensions().items():
            if "op" in args and args["op"] == op_name:
                impls.append((args["name"], impl))

        return impls


# Import op_implementations to trigger registration of all implementations
import dace.libraries.onnx.op_implementations
