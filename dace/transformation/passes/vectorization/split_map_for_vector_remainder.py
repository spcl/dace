# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``SplitMapForVectorRemainder``, P2 vectorization-prep. For every
innermost step-1 map whose range is not provably divisible by
``vector_width``, split the map into:

- a **main map** that strides by ``vector_width`` and is the vectorisation
  target,
- a **remainder map** scheduled sequentially after the main map covering
  the leftover trailing iterations.

Two modes for the remainder:

- ``"scalar"`` (default), step 1, marked ``ScheduleType.Sequential`` so
  the vectorizer skips it. Today's standard postamble shape.
- ``"masked"``, step ``vector_width``, the remainder body keeps the same
  vector body and relies on P3 to wire an ``_iter_mask`` connector that
  zeroes out the out-of-range lanes. Until P3 lands the mask is missing,
  so masked mode is a stub that allocates the right map ranges but does
  not attach the mask yet, downstream emission must error if it sees a
  step-``W`` remainder without the connector.

Reuses ``dace.transformation.helpers.replicate_scope`` (proven on the
legacy ``dataflow/vectorization.py`` postamble path).
"""
from typing import Optional

import dace
from dace import properties, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import replicate_scope
from dace.transformation.passes.vectorization.vectorization_utils import is_innermost_map


@properties.make_properties
class SplitMapForVectorRemainder(ppl.Pass):
    """Split innermost step-1 maps into a vectorisable main map plus a
    trailing remainder map. See module docstring for the two modes."""

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    mode = properties.Property(dtype=str, default="scalar", allow_none=False,
                               desc="``scalar`` for a step-1 sequential remainder, ``masked`` for step-W "
                               "remainder that P3 will attach the iter-mask connector to")

    def __init__(self, vector_width: int = 8, mode: str = "scalar"):
        super().__init__()
        self.vector_width = vector_width
        self.mode = mode

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        if self.mode not in ("scalar", "masked"):
            raise ValueError(f"SplitMapForVectorRemainder.mode must be 'scalar' or 'masked', got {self.mode!r}")

        W = self.vector_width
        applied = 0
        # Snapshot the map list up front, splitting mutates the state graph
        # and we do not want to iterate over the just-added remainder map.
        for n, g in [(n, g) for n, g in sdfg.all_nodes_recursive()
                     if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)]:
            if not is_innermost_map(g, n):
                continue
            if not self._split(g, n, W):
                continue
            applied += 1
        return applied or None

    def _split(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry, W: int) -> bool:
        # Only handle a step-1 innermost range. Multi-dim maps split on the
        # innermost dim, but skip if that dim does not have step 1.
        if not map_entry.map.range.ranges:
            return False
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return False

        # Skip if the trip count is provably a multiple of W (nothing to do)
        # or provably smaller than W (the main map would be empty so splitting
        # only adds noise, leave the original map for downstream handling).
        trip = symbolic.simplify(ub - lb + 1)
        try:
            if bool((trip % W).simplify() == 0):
                return False
            if bool((trip < W).simplify()):
                return False
        except Exception:
            pass

        # New trailing-edge for the main map. main runs lb..main_end inclusive
        # with step 1 (the vector body is W-wide; emission then takes W-strided
        # care of it). The remainder runs main_end+1..ub.
        main_end = lb + ((trip // W) * W) - 1
        rem_start = main_end + 1

        # Replicate the entire scope (entry + body + exit) for the remainder.
        scope_view = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
        rem_scope = replicate_scope(state.sdfg, state, scope_view)

        # Tighten the main map's innermost range and the remainder's.
        map_entry.map.range[-1] = (lb, main_end, 1)
        rem_step = 1 if self.mode == "scalar" else W
        rem_scope.entry.map.range[-1] = (rem_start, ub, rem_step)

        if self.mode == "scalar":
            # Scalar remainder must not be vectorised by the downstream pass.
            rem_scope.entry.map.schedule = dace.dtypes.ScheduleType.Sequential
        return True
