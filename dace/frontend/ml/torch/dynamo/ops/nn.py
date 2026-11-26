# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Nearest-neighbor upsampling and dropout backward, lowered to DaCe maps.

Forward convolution and dropout, pooling (max and average), the max-pool backward and the convolution backward are all
library nodes (:mod:`dace.libraries.torch.nodes.aten`, emitted by the generic lowering in :mod:`.generic`) that expand
to :mod:`dace.libraries.ml` without any padded copy of the input.
"""

import torch

from dace.frontend.ml.torch.dynamo.context import LoweringContext, TensorValue, as_const, index_memlet

from . import register_lowering, resolve
from .pointwise import lower_pointwise_values

aten = torch.ops.aten


def _lower_nearest_upsample(ctx: LoweringContext, node, x: TensorValue, scales):
    """One output-element map, with local float32 coordinate calculations as in ATen."""
    out = ctx.add_tensor_like("t_" + node.name, node.meta["val"])
    params = [f"__i{k}" for k in range(out.rank)]
    source_indices = params[:2]
    code = []
    for d, scale in enumerate(scales):
        isize, osize = x.tshape[d + 2], out.tshape[d + 2]
        oi = params[d + 2]
        inverse_scale = (
            f"dace.float32(1.0 / ({scale!r}))"
            if scale is not None and scale > 0
            else f"dace.float32({isize}) / dace.float32({osize})"
        )
        code.append(f"__scale{d} = {inverse_scale}")
        index = f"min(dace.int64(dace.float32({oi}) * __scale{d}), ({isize}) - 1)"
        # ATen's CPU 2D legacy nearest kernel has identity/doubling fast paths,
        # even when explicit scales would otherwise select different pixels.
        if x.rank == 4 and x.device.type == "cpu":
            index = f"({oi} if ({osize}) == ({isize}) else ({oi} // 2 if ({osize}) == 2 * ({isize}) else {index}))"
        code.append(f"__p{d} = {index}")
        source_indices.append(f"__p{d}")
    # Memlet subscripts omit unit dimensions and apply the input's actual strides.
    source_indices = [i for i, size in zip(source_indices, x.tshape) if size != 1]
    source = f"__x[{', '.join(source_indices)}]" if source_indices else "__x"
    code.append(f"__out = {source}")
    ctx.emit_mapped_tasklet(
        node.name,
        params,
        [(0, s - 1, 1) for s in out.tshape],
        {"__x": x.memlet()},
        "\n".join(code),
        {"__out": index_memlet(out.name, params)},
    )
    return out


@register_lowering(*resolve(*(f"aten.upsample_nearest{nd}d.default" for nd in (1, 2, 3))))
def lower_upsample_nearest(ctx: LoweringContext, node, x: TensorValue, output_size, *scales, **kwargs):
    nd = x.rank - 2
    names = ("scales",) if nd == 1 else (("scales_h", "scales_w") if nd == 2 else ("scales_d", "scales_h", "scales_w"))
    values = [as_const(kwargs.get(name, scales[i] if i < len(scales) else None)) for i, name in enumerate(names)]
    return _lower_nearest_upsample(ctx, node, x, values)


@register_lowering(*resolve(*(f"aten.upsample_nearest{nd}d.vec" for nd in (1, 2, 3))))
def lower_upsample_nearest_vec(ctx: LoweringContext, node, x: TensorValue, output_size, scale_factors):
    scales = as_const(scale_factors)
    return _lower_nearest_upsample(ctx, node, x, scales if scales is not None else [None] * (x.rank - 2))


@register_lowering(*resolve("aten.native_dropout_backward.default"))
def lower_native_dropout_backward(ctx: LoweringContext, node, grad_output: TensorValue, mask: TensorValue, scale):
    """``grad_output * scale`` where the (boolean) mask is set, zero elsewhere."""
    return lower_pointwise_values(ctx, node, [grad_output, mask, scale], "(({0}) * ({2}) if ({1}) else 0)", nocast=(1,))
