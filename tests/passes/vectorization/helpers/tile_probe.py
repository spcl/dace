# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural probes shared by the K-dim tile-descent unit tests.

These classifiers inspect a lowered SDFG to count raw tasklets that still touch
tile-shaped data (the K-dim "tile-only" invariant check). They live here rather
than in the production pipeline because only the tests consume them.
"""
from typing import Tuple

import dace


def descriptor_is_tile_or_broadcast(desc, widths: Tuple[int, ...]) -> bool:
    """ONE-aware classify: ``desc`` is a tile (full ``widths``) or broadcast-tile
    (each dim = tile width ``w_d`` or broadcast marker ``dace.symbolic.ONE`` /
    literal ``1``), with >=1 real (non-broadcast) width dim.

    CLASSIFY ONLY, never reshapes a memlet. ``(W, ONE)`` / ``(ONE, W)`` broadcast
    shapes live only on ``TileLoad`` / ``TileStore`` index connectors; collapsing
    to ``(W,)`` in a memlet trips DaCe's subset-dimensionality validator (such
    shapes only passed to tile load/store). Treats the ``ONE`` marker as the int
    ``1`` it stands for without touching the memlet.

    Rank gate: a scalar ``(1,)`` bridge in a K=1 ``__tile_k1_tail`` body is rank
    1, so vs a rank-2 ``widths`` it is correctly NOT a tile (scalar-load -> scalar
    chains stay python tasklets).

    :param desc: Data descriptor (``dace.data.Array`` / ``Scalar`` / ...).
    :param widths: Per-tile-dim widths, innermost-last.
    :returns: True iff ``desc`` is a tile or broadcast-tile of rank ``len(widths)``.
    """
    import sympy
    from dace.symbolic import ONE
    if not isinstance(desc, dace.data.Array):
        return False
    shape = tuple(desc.shape)
    if len(shape) != len(widths):
        return False
    n_real = 0
    for s, w in zip(shape, widths):
        try:
            is_w = bool(dace.symbolic.simplify(s - w) == 0)
        except Exception:  # noqa: BLE001 -- symbolic simplification may refuse
            is_w = (s == w)
        is_one_marker = isinstance(s, sympy.Basic) and ONE in s.free_symbols
        try:
            is_lit_one = bool(dace.symbolic.simplify(s - 1) == 0)
        except Exception:  # noqa: BLE001
            is_lit_one = (s == 1)
        if is_w:
            n_real += 1
        elif is_one_marker or is_lit_one:
            continue  # broadcast dim -- the ``ONE`` marker, i.e. width 1
        else:
            return False
    return n_real >= 1


def tasklet_reads_or_writes_tile(state: dace.SDFGState, tasklet: dace.nodes.Tasklet, widths: Tuple[int, ...]) -> bool:
    """True iff any in/out edge of ``tasklet`` carries a tile / broadcast-tile
    descriptor (see :func:`descriptor_is_tile_or_broadcast`).

    K-dim tile-only invariant: tile-shaped values flow ONLY through tile lib
    nodes, never raw tasklets. A raw tasklet touching a tile = unlowered residue
    (real failure); a pure-scalar tasklet (e.g. the scalar ``__tile_k1_tail``
    remainder) is legit, NOT counted.
    """
    sdfg = state.sdfg
    for edge in list(state.in_edges(tasklet)) + list(state.out_edges(tasklet)):
        if edge.data is None or edge.data.data is None:
            continue
        desc = sdfg.arrays.get(edge.data.data)
        if desc is not None and descriptor_is_tile_or_broadcast(desc, widths):
            return True
    return False
