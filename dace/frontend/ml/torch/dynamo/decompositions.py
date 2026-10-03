# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Decomposition table handed to AOTAutograd.

Follows the Inductor strategy: decompose composite ATen operators into a small primitive set (pointwise, reductions,
views, matmul, convolution) that the frontend lowers natively. Operators that DaCe lowers directly (library nodes or
structured maps) are removed from the table so they are not decomposed into less efficient forms.
"""
from typing import Dict, Iterable, Optional

import torch

aten = torch.ops.aten

#: Composite operators that ``core_aten_decompositions`` leaves intact but we want decomposed into primitives.
EXTRA_DECOMPOSITIONS = [
    '_softmax', '_log_softmax', 'softmax', 'log_softmax', 'native_layer_norm', 'layer_norm', 'native_group_norm',
    'group_norm', '_native_batch_norm_legit_no_training', '_native_batch_norm_legit', 'native_batch_norm',
    'batch_norm', 'gelu', 'silu', 'mish', 'hardtanh', 'hardswish', 'hardsigmoid', 'leaky_relu', 'elu', 'celu', 'selu',
    'softplus', 'log_sigmoid_forward', 'log_sigmoid', 'native_dropout', 'dropout', 'tril', 'triu', 'masked_fill',
    'baddbmm', 'logsumexp', 'var_mean', 'var', 'std', 'std_mean', 'norm', 'linalg_vector_norm', 'repeat', 'roll',
    'stack', 'unbind', 'split', 'split_with_sizes', 'chunk', 'narrow', 'expand_as', 'reshape', 'flatten', 'squeeze',
    'unsqueeze', 'index_select', 'embedding', '_unsafe_index', 'nan_to_num', 'cumsum', 'upsample_nearest2d',
    '_adaptive_avg_pool2d', 'avg_pool2d', 'binary_cross_entropy_with_logits', 'mse_loss', 'l1_loss', 'smooth_l1_loss',
    'huber_loss', 'nll_loss_forward', 'nll_loss', 'cross_entropy_loss'
]

#: Operators lowered natively by the frontend; removed from the decomposition table.
NATIVE_OPS = [
    'mm', 'bmm', 'addmm', 'mv', 'dot', 'matmul', 'linear', 'convolution', '_to_copy', 'clone', 'cat', 'clamp',
    'clamp_min', 'clamp_max', 'expand', 'view', '_unsafe_view', 'permute', 't', 'transpose', 'slice', 'select', 'alias',
    'detach', 'sum', 'mean', 'amax', 'amin', 'prod', 'max', 'min', 'argmax', 'argmin', 'where', 'full', 'full_like',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like', 'new_zeros', 'new_ones', 'new_full', 'new_empty',
    'arange', 'scalar_tensor', 'addcmul', 'addcdiv', 'lerp', 'rsub', 'sigmoid', 'tanh', 'relu', 'flip', 'exp', 'log',
    'sqrt', 'rsqrt', 'square', 'reciprocal', 'pow', 'copy'
]


def _resolve_packets(names: Iterable):
    result = []
    for name in names:
        if isinstance(name, str):
            try:
                result.append(getattr(aten, name))
            except (AttributeError, RuntimeError):
                continue
        else:
            result.append(name)
    return result


def _materialize(table) -> Dict:
    if hasattr(table, 'materialize'):
        return dict(table.materialize())
    return dict(table)


def _packets(ops: Iterable):
    result = set()
    for op in ops:
        if isinstance(op, torch._ops.OpOverloadPacket):
            result.add(op)
        elif isinstance(op, torch._ops.OpOverload):
            result.add(op.overloadpacket)
    return result


def build_decomposition_table(extra: Optional[Iterable] = None, native_ops: Optional[Iterable] = None) -> Dict:
    """
    Builds the decomposition table for AOTAutograd.

    :param extra: Additional operators (overloads or packets) to decompose.
    :param native_ops: Additional operators that must not be decomposed (lowered natively by the user/frontend).
    """
    from torch._decomp import core_aten_decompositions, get_decompositions

    table = _materialize(core_aten_decompositions())
    wanted = _resolve_packets(EXTRA_DECOMPOSITIONS) + list(extra or [])
    table.update(get_decompositions(wanted))

    keep_native = _packets(_resolve_packets(NATIVE_OPS)) | _packets(native_ops or [])
    for op in list(table.keys()):
        packet = op.overloadpacket if isinstance(op, torch._ops.OpOverload) else op
        if packet in keep_native:
            del table[op]
    return table
