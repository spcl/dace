# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared memlet/iteration walk helpers for the vectorization pipeline."""
from typing import Iterator, Tuple

import dace
from dace import SDFGState
from dace.memlet import Memlet
from dace.sdfg.graph import Edge


def walk_memlets_of(sdfg: dace.SDFG, dataname: str) -> Iterator[Tuple[SDFGState, Edge[Memlet]]]:
    """
    Yield ``(state, edge)`` for every state-level edge whose memlet targets ``dataname``.

    Recurses into nested SDFGs via ``sdfg.all_states()`` and skips edges
    whose memlet has no data attached.

    :param sdfg: The SDFG whose state-level edges to walk.
    :param dataname: The array / scalar name to filter on.
    :returns: Iterator of ``(state, edge)`` pairs in iteration order.
    """
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None and edge.data.data == dataname:
                yield state, edge


def assert_no_lane_memlet_reads(sdfg: dace.SDFG, vector_width: int) -> None:
    """
    Verify no per-lane memlet survived the masked-remainder lane collapse.

    For every SDFG state with an ``_iter_mask`` array in scope, no memlet
    subset may reference a ``_laneid_<i>`` symbol; such references would
    lower to unconditional scalar reads that fault on inactive lanes.

    :param sdfg: The SDFG to verify (recursively, including nested SDFGs).
    :param vector_width: Accepted for symmetry; the check is width-agnostic.
    :raises RuntimeError: On the first uncollapsed per-lane memlet, naming
        the offending edge.
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
