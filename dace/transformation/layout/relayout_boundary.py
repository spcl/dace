# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Insert relayout states on line-graph boundaries, one `LayoutChange` node per changed array."""
from typing import Dict, List, Tuple

from dace import SDFG, SDFGState
from dace.libraries.layout import add_layout_change


def relayout_on_boundary(sdfg: SDFG,
                         dst_state: SDFGState,
                         changes: Dict[str, Tuple[str, List]],
                         make_transient: bool = True) -> SDFGState:
    """Insert a relayout state before `dst_state`; `changes` = {in_array: (out_array, ops)}."""
    if not changes:
        raise ValueError("relayout_on_boundary: empty change set -- refuse to insert a dead state")
    boundary = sdfg.add_state_before(dst_state,
                                     label=f"relayout_before_{dst_state.label}",
                                     is_start_block=(sdfg.start_block is dst_state))
    for in_name in sorted(changes):
        out_name, ops = changes[in_name]
        add_layout_change(sdfg, boundary, in_name, out_name, ops)
        sdfg.arrays[out_name].transient = make_transient
    return boundary
