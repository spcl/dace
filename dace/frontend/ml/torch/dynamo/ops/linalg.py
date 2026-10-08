# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Linear-algebra lowerings onto DaCe BLAS library nodes."""

import torch


from ..context import LoweringContext, TensorValue, as_sym
from . import register_lowering

aten = torch.ops.aten


def _matmul(ctx: LoweringContext, node, a: TensorValue, b: TensorValue) -> TensorValue:
    from dace.libraries.blas import MatMul

    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    ctx.emit_library_call(MatMul("mm_" + node.name), {"_a": a.memlet(), "_b": b.memlet()}, {"_c": out.memlet()})
    return out


@register_lowering(aten.mm.default, aten.bmm.default, aten.mv.default, aten.dot.default, aten.matmul.default)
def lower_mm(ctx: LoweringContext, node, a: TensorValue, b: TensorValue):
    return _matmul(ctx, node, a, b)


@register_lowering(aten.addmm.default)
def lower_addmm(ctx: LoweringContext, node, bias: TensorValue, a: TensorValue, b: TensorValue, *, beta=1, alpha=1):
    from .pointwise import lower_pointwise_values

    mm = _matmul(ctx, node, a, b)
    beta = as_sym(beta) if not isinstance(beta, float) else beta
    alpha = as_sym(alpha) if not isinstance(alpha, float) else alpha
    if beta == 1 and alpha == 1:
        return lower_pointwise_values(ctx, node, [mm, bias], "{0} + {1}", prefix="t_" + node.name + "_bias")
    return lower_pointwise_values(
        ctx, node, [mm, bias], f"({alpha}) * {{0}} + ({beta}) * {{1}}", prefix="t_" + node.name + "_bias"
    )
