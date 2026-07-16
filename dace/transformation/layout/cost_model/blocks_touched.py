# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Average new memory blocks touched per iteration -- the layout-sensitive term of the memory cost.

The LogP/LogGP parameters (L, G) are properties of the hardware; this is the property of the ACCESS
that the layout transformations actually move. A request of ``n`` bytes has a fixed ``n``, but
Permute/Pad/Block change how many BLOCKS those bytes span, and the block count is what the model
pays for (one message per block: a cache line on the CPU, a coalesced sector on the GPU). This is the
Delta term of the SC26 cost table.

For a perfectly nested loop over an array with affine index, when a loop parameter ``p`` steps by its
stride, the byte address moves by ``stride(p)``. If that stride is at least one block, the step lands
on a fresh block (fraction 1); if it is a sub-block stride, consecutive steps SHARE a block and only
``stride/block`` of a new block is touched per step. Summed over the nest and divided by the
iteration count, that is the average new blocks per iteration.

The fraction is computed for EVERY loop dimension, not only the innermost: a blocked or AoSoA layout
puts a sub-block stride on an OUTER loop, and crediting reuse only on the innermost would make that
layout look no better than a strided one. The model assumes separable strides (each dimension maps to
a disjoint block range, which well-formed array layouts satisfy); densely overlapping strides are out
of scope.

The block size is a parameter, so one function serves both devices: 64 B cache line on the CPU, 32 B
sector for GPU global-memory coalescing. It is given in ELEMENTS (block bytes / dtype bytes), matching
the element strides.

ACCURACY: the per-dimension fraction is a CONTINUOUS approximation of the integer block count -- it
uses ``(extent - 1) * stride / block`` where the true count applies ``ceil`` at block granularity.
Checked against a brute-force traversal oracle, it converges to the exact block-transaction count as
extents grow (2D row-major -> 1/8, transpose -> 1), but OVERCOUNTS for small tiles: a contiguous 4x4
fp64 tile is 16 contiguous elements = 2 blocks, yet the independent per-dimension fractions cannot see
that the inner runs combine into one contiguous span, and report ~4. The RANKING is preserved
(contiguous always scores below scattered), so choosing a layout is sound; absolute transaction counts
for sub-block tiles are not exact. The exact streaming/coalescing count over the innermost contiguous
run is the next refinement, and ``tests/.../cost_model_blocks_touched_test.py`` carries the oracle it
must match.
"""
from typing import Dict, List

import dace
import sympy as sp

from dace.symbolic import int_floor, pystr_to_symbolic


def average_blocks_touched(
    state: dace.SDFGState,
    loop_ranges: List[Dict[str, dace.subsets.Range]],  # outer-to-inner: param -> range
    access_subsets: Dict[str, dace.subsets.Subset],  # array -> accessed subset
    block_size: int,  # transfer granularity in ELEMENTS (cache line / GPU sector)
) -> Dict[str, sp.Basic]:
    """``{array: average new blocks touched per iteration}`` (symbolic).

    ``loop_ranges`` lists the nest outer-to-inner; the last entry is the innermost loop.
    """
    sdfg = state.sdfg

    # Flatten the nest to a single param -> range map, preserving outer-to-inner order.
    params: List[str] = []
    ranges: Dict[str, dace.subsets.Range] = {}
    for nest in loop_ranges:
        for param, rng in nest.items():
            if param in ranges:
                raise ValueError(f"loop parameter {param!r} appears in more than one nest level")
            ranges[param] = rng
            params.append(param)

    # Iteration count of each loop: floor((end - begin) / step) + 1. int_floor, not '/', because this
    # is a COUNT -- C division would truncate a rational extent the wrong way.
    extents = {}
    for param in params:
        begin, end, step = ranges[param]
        extents[param] = int_floor(pystr_to_symbolic(end) - pystr_to_symbolic(begin),
                                   pystr_to_symbolic(step)) + 1
    total_iters = sp.Mul(*[extents[p] for p in params]) if params else sp.Integer(1)

    results: Dict[str, sp.Basic] = {}
    for arr, subset in access_subsets.items():
        if arr not in sdfg.arrays or not isinstance(subset, dace.subsets.Range):
            continue

        strides = sdfg.arrays[arr].strides
        index = [pystr_to_symbolic(rb) for rb, _, _ in subset.ranges]
        addr = sum(idx * pystr_to_symbolic(st) for idx, st in zip(index, strides))

        total_new = sp.Integer(1)  # the first iteration always touches a new block
        for depth, param in enumerate(params):
            psym = pystr_to_symbolic(param)
            step = pystr_to_symbolic(ranges[param][2])
            # Byte-address movement when this loop steps once.
            stride = sp.simplify(addr.subs(psym, psym + step) - addr)
            # Fraction of a NEW block per step: a sub-block stride shares blocks between steps. A
            # genuine rational in [0, 1] -- NOT int_floor, which would zero out all sub-block reuse.
            frac = sp.Min(1, sp.Abs(stride) / block_size)
            # Each outer-loop combination visits this loop afresh.
            outer = sp.Mul(*[extents[params[k]] for k in range(depth)]) if depth else sp.Integer(1)
            total_new += outer * (extents[param] - 1) * frac

        results[arr] = sp.simplify(total_new / total_iters)  # a rational average, '/' is exact here

    return results
