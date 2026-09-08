# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Read-only query helpers used by the vectorization pipeline.

These helpers do not mutate the SDFG; they extract access subsets used by the
emission and prep passes.
"""
from typing import Dict, Optional

import dace


def collect_element_write_subsets(state: dace.SDFGState) -> Optional[Dict[str, dace.subsets.Range]]:
    """Return ``{arr_name: subset}`` for every element-wise write in ``state``.

    A write is element-wise iff its memlet subset has
    ``num_elements_exact() == 1``. Multiple writes to the same array keep
    only the last subset seen.

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
            # The DESTINATION side, which is what a write subset means. ``subset`` describes
            # whichever side ``data`` names, and on an access-node-to-access-node copy that is the
            # SOURCE: ``delta[i] -> _scan_in_out`` carries ``delta``'s subset while the write lands
            # at ``_scan_in_out[i - 1]``. Read as the write, it dropped the offset, and the ITE that
            # replaced the copy wrote one element past its buffer.
            subset = e.data.dst_subset
            if subset is None:
                return None
            try:
                if subset.num_elements_exact() != 1:
                    return None
            except Exception:
                return None
            out[n.data] = subset
    return out
