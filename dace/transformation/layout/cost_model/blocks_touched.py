# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Average new memory blocks touched per iteration -- the layout-sensitive cost term (overcounts small tiles, but layout ranking is preserved)."""
from typing import Dict, List

import dace
import numpy
import sympy as sp

from dace.symbolic import int_floor, pystr_to_symbolic, simplify


def average_blocks_touched(
        state: dace.SDFGState,
        loop_ranges: List[Dict[str, dace.subsets.Range]],  # outer-to-inner: param -> range
        access_subsets: Dict[str, dace.subsets.Subset],  # array -> accessed subset
        block_size: int,  # transfer granularity in ELEMENTS (cache line / GPU sector)
) -> Dict[str, sp.Basic]:
    """``{array: average new blocks touched per iteration}`` (symbolic); loop_ranges is outer-to-inner."""
    sdfg = state.sdfg

    # flatten nest to a param -> range map, outer-to-inner order preserved
    params: List[str] = []
    ranges: Dict[str, dace.subsets.Range] = {}
    for nest in loop_ranges:
        for param, rng in nest.items():
            if param in ranges:
                raise ValueError(f"loop parameter {param!r} appears in more than one nest level")
            ranges[param] = rng
            params.append(param)

    # iteration count = floor((end - begin) / step) + 1; int_floor not '/' since this is a COUNT
    extents = {}
    for param in params:
        begin, end, step = ranges[param]
        extent = int_floor(pystr_to_symbolic(end) - pystr_to_symbolic(begin), pystr_to_symbolic(step)) + 1
        # an empty level makes total_iters 0, and the per-iteration average below divides by it (-> zoo)
        if extent.is_number and extent <= 0:
            raise ValueError(f"loop parameter {param!r} has range {ranges[param]}, i.e. {extent} iterations; "
                             f"a per-iteration block average is undefined for an empty nest -- skip the nest "
                             f"instead of costing it")
        extents[param] = extent
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
            # byte-address movement per step
            stride = sp.simplify(addr.subs(psym, psym + step) - addr)
            # affine in psym <=> the step delta is free of psym; '//' and '%' indices are not, and would
            # leak the loop variable into the "per-iteration" cost, crashing the later float()
            if psym in stride.free_symbols:
                raise ValueError(f"array {arr!r}: index {subset} is not affine in loop parameter {param!r} "
                                 f"(e.g. '//' or '%'), so the per-step address delta {stride} still depends "
                                 f"on {param!r} and the block count is not constant per iteration. Replay the "
                                 f"materialized access sequence with blocks_touched.replayed_blocks_touched "
                                 f"and pass the chosen bound via replayed_counts={{'{arr}': "
                                 f"(messages_per_iter, sectors_per_iter)}}.")
            # fraction of a new block per step; rational in [0,1], NOT int_floor (would zero sub-block reuse)
            frac = sp.Min(1, sp.Abs(stride) / block_size)
            # each outer-loop combination revisits this loop afresh
            outer = sp.Mul(*[extents[params[k]] for k in range(depth)]) if depth else sp.Integer(1)
            total_new += outer * (extents[param] - 1) * frac

        average = simplify(total_new / total_iters)  # a rational average, '/' is exact here
        # The per-step guard above only sees one parameter at a time, so a cross-parameter index like
        # A[i*j] clears it twice and still leaks both into the average. Catch it on the result, where
        # the caller's float() would otherwise fail naming neither the array nor the index.
        leaked = sorted({str(s) for s in average.free_symbols} & set(params))
        if leaked:
            raise ValueError(f"array {arr!r}: index {subset} is not affine in {leaked} together (e.g. a "
                             f"product of two loop parameters), so the per-iteration block average "
                             f"{average} still depends on them. Replay the materialized access sequence "
                             f"with blocks_touched.replayed_blocks_touched and pass the chosen bound via "
                             f"replayed_counts={{'{arr}': (messages_per_iter, sectors_per_iter)}}.")
        results[arr] = average

    return results


def replayed_blocks_touched(indices, block_elems: int):
    """Blocks-per-access bounds for indirect access ``A[idx[i]]``, by replaying idx: streaming (upper) vs distinct (lower)."""
    if block_elems < 1:
        raise ValueError(f"block_elems must be >= 1, got {block_elems}")
    idx = numpy.asarray(indices)
    if idx.ndim != 1:
        raise ValueError(f"indices must be a 1-D access sequence, got shape {idx.shape}; flatten in "
                         "TRAVERSAL order first -- the streaming bound depends on it")
    if idx.size == 0:
        return 0.0, 0.0
    # Plain `//` on purpose: idx is a concrete numpy array of recorded accesses, so this is
    # elementwise integer division, not a symbolic expression. int_floor would sympify it.
    blocks = idx // block_elems
    streaming = (1 + int(numpy.count_nonzero(blocks[1:] != blocks[:-1]))) / idx.size
    distinct = int(numpy.unique(blocks).size) / idx.size
    return streaming, distinct
