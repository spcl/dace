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


def assert_no_lane_memlet_reads(sdfg: dace.SDFG, vector_width: int) -> None:
    """Verifier for the masked-remainder contract (locked decision 2026-05-14).

    For every map in ``sdfg`` (recursively) that has an ``_iter_mask`` array
    allocated in scope, every memlet inside that map's scope subgraph must
    have been collapsed by the lane-fanout / intrinsic-detect passes — no
    per-lane subset references (i.e. no ``_laneid_<i>`` symbol in any subset
    expression) may remain. Such references would lower to unconditional
    scalar memlet reads in codegen and fault on inactive lanes when the
    surrounding map runs the masked-remainder iteration.

    Raises ``RuntimeError`` on the first uncollapsed per-lane memlet,
    naming the offending edge so the caller can flip
    ``lower_to_intrinsics=True`` or fall back to ``remainder_strategy="scalar"``.

    ``vector_width`` is accepted for symmetry / future use; the check is
    currently width-agnostic (just looks for the ``_laneid_`` symbol).
    """
    from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme

    for state in sdfg.all_states():
        has_iter_mask = any(name == "_iter_mask" or name.startswith("_iter_mask_") for name in state.sdfg.arrays)
        if not has_iter_mask:
            continue
        for e in state.edges():
            if e.data.data is None:
                continue
            subset_str = str(e.data.subset)
            if LaneIdScheme.SUFFIX in subset_str:
                raise RuntimeError(f"assert_no_lane_memlet_reads: map with _iter_mask in scope "
                                   f"({state.label}) still has a per-lane memlet on '{e.data.data}' "
                                   f"with subset {subset_str}. Lane fanout failed to collapse to "
                                   f"an intrinsic — would fault on inactive lanes. Set "
                                   f"`lower_to_intrinsics=True` on VectorizeCPU, or use "
                                   f"`remainder_strategy='scalar'` for this kernel.")

    for s in sdfg.all_sdfgs_recursive():
        if s is sdfg:
            continue
        # Re-walk children: each NSDFG may have its own _iter_mask scope.
        for state in s.states():
            has_iter_mask = any(name == "_iter_mask" or name.startswith("_iter_mask_") for name in state.sdfg.arrays)
            if not has_iter_mask:
                continue
            for e in state.edges():
                if e.data.data is None:
                    continue
                subset_str = str(e.data.subset)
                if LaneIdScheme.SUFFIX in subset_str:
                    raise RuntimeError(f"assert_no_lane_memlet_reads: nested SDFG '{s.name}' with "
                                       f"_iter_mask in scope still has a per-lane memlet on "
                                       f"'{e.data.data}' in state '{state.label}' with subset "
                                       f"{subset_str}.")
