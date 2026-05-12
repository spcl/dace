# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared iteration helpers for the vectorization pipeline.

The "walk every memlet of array X in this SDFG" pattern is hand-rolled
in many places throughout the vectorization helpers
(``drop_dims``, ``offset_memlets``, ``replace_memlet_expression``,
``replace_all_access_subsets``, the inline loop in
``prepare_vectorized_array``, ...). Each copy independently has to
remember the same shape — walk all states, walk every edge, filter on
``edge.data.data == dataname`` — and each forgetting one piece is a
recurring class of bug.

``walk_memlets_of`` centralizes the state-level walk. Helpers that also
touch interstate edges / loop or conditional-block code blocks keep
their additional walks adjacent to this one.
"""
from typing import Iterator, Tuple

import dace
from dace import SDFGState
from dace.memlet import Memlet
from dace.sdfg.graph import Edge


def walk_memlets_of(sdfg: dace.SDFG, dataname: str) -> Iterator[Tuple[SDFGState, Edge[Memlet]]]:
    """Yield ``(state, edge)`` for every state-level edge whose memlet
    targets ``dataname``.

    Recurses into nested SDFGs via ``sdfg.all_states()``. Skips edges
    whose memlet has no data attached (``edge.data.data is None``).

    Args:
        sdfg: The SDFG whose state-level edges to walk.
        dataname: The array / scalar name to filter on.

    Yields:
        ``(state, edge)`` pairs for every match in iteration order.
    """
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None and edge.data.data == dataname:
                yield state, edge
