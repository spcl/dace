# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Insert relayout states on line-graph boundaries (GLOBAL_LAYOUT_DESIGN.md, task A4).

A layout TRAJECTORY changes an array's layout between two kernel states; the change is materialized
as its own state on the boundary, holding one ``LayoutChange`` node per changed array (parallel
nodes, one state). The node carries the layout-algebra op sequence; lowering (pure copy map /
TensorTranspose) is chosen later by ``select_layout_lowering``. Descriptor changes -- blocking
changes shape -- are owned by ``add_layout_change``, which creates the laid-out output descriptor
from the op sequence.

The inserted state is recognized by ``line_graph.is_relayout_state`` and takes no kernel position.
Downstream states are NOT rewritten here -- pointing the consumers at the relaid array is task A5's
segment rewrite.
"""
from typing import Dict, List, Tuple

from dace import SDFG, SDFGState
from dace.libraries.layout import add_layout_change


def relayout_on_boundary(sdfg: SDFG,
                         dst_state: SDFGState,
                         changes: Dict[str, Tuple[str, List]],
                         make_transient: bool = True) -> SDFGState:
    """Insert a relayout state on the boundary BEFORE ``dst_state``.

    :param sdfg: the (line-graph shaped) SDFG.
    :param dst_state: the kernel state the relaid arrays feed; the new state takes over its
                      incoming edge (a line graph has exactly one).
    :param changes: ``{in_array: (out_array, ops)}`` -- one parallel ``LayoutChange`` per entry; the
                    ``out_array`` descriptor is created from the op sequence (blocking changes its
                    shape).
    :param make_transient: the relaid copies are internal segment storage by default; pass False to
                           expose one as a program output.
    :return: the inserted boundary state.
    """
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
