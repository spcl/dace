# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Union

import numpy as np

import dace
from dace import SDFG, SDFGState, nodes as nd

from dace.libraries.onnx.op_implementations.utils import op_implementation, program_for_node
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.forward_implementation_abc import ONNXForward

from dace.util import in_desc_with_name


@op_implementation(op="SoftmaxCrossEntropyLoss", name="pure")
class PureSoftmaxCrossEntropyLoss(ONNXForward):
    """Pure implementation of SoftmaxCrossEntropyLoss operation."""

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        """Check if this implementation can be applied to the given node.

        Args:
            node: The SoftmaxCrossEntropyLoss ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            True if the implementation can be applied, False otherwise
        """
        # Softmax is weird in opset 11, so let's stick to 2D for now
        if len(in_desc_with_name(node, state, sdfg, "scores").shape) != 2:
            return False

        if node.ignore_index is not None and node.ignore_index >= 0:
            return False

        # FIXME: support this
        if 'weights' in node.in_connectors:
            return False
        if 'log_prob' in node.out_connectors:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> Union[nd.Node, SDFG]:
        """Generate the forward pass implementation for SoftmaxCrossEntropyLoss.

        Args:
            node: The SoftmaxCrossEntropyLoss ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            A nested SDFG implementing the SoftmaxCrossEntropyLoss operation
        """

        if node.reduction == 'mean':

            def reduction(x):
                return np.mean(x)
        elif node.reduction == 'none':

            def reduction(x):
                return x
        elif node.reduction == 'sum':

            def reduction(x):
                return np.sum(x)
        else:
            raise ValueError("Unsupported reduction: {}".format(node.reduction))
        reduction = dace.program(reduction)

        # This implementation doesn't use ONNX LogSoftmax, and thus saves the
        # final sum reduction by just grabbing the label scores directly, and
        # skipping the computation of log softmax for all non-label scores
        def prog(scores, labels, output):
            # Extract the scores for the labels

            # Compute the log softmax normalization
            maximum = np.maximum.reduce(scores, axis=1, keepdims=True)
            max_sub = scores - maximum
            exponent = np.exp(max_sub)
            sum = np.add.reduce(exponent, axis=1)
            log_sum = np.log(sum)

            # Compute the loss values
            label_exponents = max_sub[:, labels]
            losses = log_sum - label_exponents
            output[:] = reduction(losses)

        return program_for_node(prog, sdfg, state, node)
