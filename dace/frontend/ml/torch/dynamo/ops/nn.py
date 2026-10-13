# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The dropout backward, lowered to a DaCe map.

Forward convolution and dropout, pooling (max and average), the max-pool backward and the convolution backward are all
library nodes (:mod:`dace.libraries.torch.nodes.aten`, emitted by the generic lowering in :mod:`.generic`) that expand
to :mod:`dace.libraries.ml` without any padded copy of the input. Only the elementwise dropout backward remains a map
here.
"""

import torch

from dace.frontend.ml.torch.dynamo.context import LoweringContext, TensorValue

from . import register_lowering, resolve
from .pointwise import lower_pointwise_values

aten = torch.ops.aten


@register_lowering(*resolve("aten.native_dropout_backward.default"))
def lower_native_dropout_backward(ctx: LoweringContext, node, grad_output: TensorValue, mask: TensorValue, scale):
    """``grad_output * scale`` where the (boolean) mask is set, zero elsewhere."""
    return lower_pointwise_values(ctx, node, [grad_output, mask, scale], "(({0}) * ({2}) if ({1}) else 0)", nocast=(1,))
