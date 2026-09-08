# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Read-only query helpers used by the vectorization pipeline.

These helpers do not mutate the SDFG; they extract access subsets used by the
emission and prep passes.
"""
from typing import Dict, Optional

import dace

from dace.transformation.passes.vectorization.utils.subsets import an_side_subset


def collect_element_write_subsets(state: dace.SDFGState) -> Optional[Dict[str, dace.subsets.Range]]:
    """Return ``{arr_name: subset}`` for every element-wise write in ``state``.

    A write is element-wise iff the subset written on the AccessNode's side has
    ``num_elements_exact() == 1``. Multiple writes to the same array keep
    only the last subset seen.

    The written region is read with :func:`~dace.transformation.passes.vectorization.utils.
    subsets.an_side_subset`, NOT with ``edge.data.subset``. On a plain AN-to-AN copy the memlet
    is oriented on the SOURCE, so ``subset`` names the region READ and the destination's is in
    ``other_subset``. ``delta[i] -> _scan_in_out[i - 1]`` then reported ``i`` as the write, and
    the ITE rewrite built from it wrote one cell past the intended one -- shifting the whole
    scan by one and running off the end of the buffer.

    :param state: State to inspect.
    :returns: Mapping of array name to its element-wise write subset, or
        ``None`` if any in-edge to an AccessNode is not element-wise.
    """
    out: Dict[str, dace.subsets.Range] = {}
    for n in state.nodes():
        if not isinstance(n, dace.nodes.AccessNode):
            continue
        for e in state.in_edges(n):
            if e.data.data is None:
                continue
            try:
                written = an_side_subset(e, n, state.sdfg, state)
                if written.num_elements_exact() != 1:
                    return None
            except Exception:
                return None
            out[n.data] = written
    return out
