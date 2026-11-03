# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Coarsen a kernel whose WCR scatters into a small array, so the block-private copy codegen keeps of that array pays.

A kernel scattering into a small target (``h[idx[i]] += w[i]``, ``s[j] += a[i, j]``) folds per block into a
shared-memory copy and commits it with one global atomic per element (see
:func:`~dace.codegen.targets.cpu.shared_scatter_target`). That trades the per-thread global atomics for ``m`` per
block, which only pays when a block covers many more elements than the target has: the kernel is coarsened along
its outermost parameter until a block covers ``COVERAGE`` times the target.
"""

import math
from typing import Any

from dace import SDFG, properties, symbolic
from dace.sdfg import nodes
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel
from dace.transformation.passes.gpu_specialization.coarsen_kernel_threads import CoarsenKernelThreads

#: Elements a block covers per element of the target it commits.
COVERAGE = 16
#: Most elements of the coarsened parameter one thread takes.
MAX_FACTOR = 64


@properties.make_properties
class CoarsenScatterKernels(ppl.Pass):
    """Coarsen every kernel with a privatizable scatter target (module docstring)."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        from dace.codegen.targets.cpu import SHARED_SCATTER_ELEMENTS, shared_scatter_target

        kernels = [(s, n) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and is_kernel(s, n)]
        coarsened = 0
        for state, kernel in kernels:
            if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
                continue
            params = set(kernel.map.params)
            targets = [
                state.sdfg.arrays[e.data.data]
                for e in state.in_edges(state.exit_node(kernel))
                if e.data.wcr is not None
                and (e.data.dynamic or {str(sym) for sym in e.data.subset.free_symbols} & params)
                and shared_scatter_target(state.sdfg, state, kernel, e.data.data, state.sdfg.arrays[e.data.data])
            ]
            if not targets:
                continue
            # A symbolic size is privatized up to the shared capacity, so the block is sized for that
            size = max(
                SHARED_SCATTER_ELEMENTS
                if symbolic.issymbolic(t.total_size, state.sdfg.constants)
                else int(t.total_size)
                for t in targets
            )
            block = math.prod(int(b) for b in (kernel.map.gpu_block_size or [256]))
            factor = min(MAX_FACTOR, math.ceil(COVERAGE * size / block))
            if factor < 2:
                continue
            CoarsenKernelThreads().coarsen(state, kernel, {kernel.map.params[0]: factor})
            coarsened += 1
        return coarsened or None
