# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split innermost maps into a vectorisable main map plus trailing remainder.

Every innermost step-1 map not provably divisible by ``vector_width`` -> main map
(vectorisation target) + remainder map over leftover trailing iters. Remainder = sequential
scalar tail (``scalar`` mode) or masked step-W body a later pass attaches the iteration-mask
connector to (``masked`` mode).
"""
from typing import Optional

import dace
from dace import properties, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import replicate_scope
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map


@properties.make_properties
class SplitMapForVectorRemainder(ppl.Pass):
    """Split innermost step-1 maps into a vectorisable main map plus a
    trailing remainder map. See module docstring for the two modes."""

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    mode = properties.Property(dtype=str,
                               default="scalar",
                               allow_none=False,
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
        """Split every eligible innermost map into a main map plus a remainder.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of maps split, or ``None`` if none.
        :raises ValueError: If ``mode`` is not ``scalar`` or ``masked``.
        """
        if self.mode not in ("scalar", "masked"):
            raise ValueError(f"SplitMapForVectorRemainder.mode must be 'scalar' or 'masked', got {self.mode!r}")

        W = self.vector_width
        applied = 0
        # Snapshot up front: splitting mutates the graph; don't iterate over the
        # just-added remainder map.
        for n, g in [(n, g) for n, g in sdfg.all_nodes_recursive()
                     if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)]:
            if not is_vectorizable_map(g, n):
                continue
            if not self._split(g, n, W):
                continue
            applied += 1
        return applied or None

    def _split(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry, W: int) -> bool:
        """Split one innermost map into a main map and a remainder map.

        :param state: The state containing the map.
        :param map_entry: The innermost map entry to split.
        :param W: Vector width used for the main map and divisibility check.
        :returns: ``True`` if the map was split, ``False`` if skipped.
        """
        # Only step-1 innermost range. Multi-dim maps split on the innermost dim.
        if not map_entry.map.range.ranges:
            return False
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return False

        # Skip if trip provably multiple of W (nothing to do) or provably < W
        # (main map empty -> splitting only adds noise; leave original).
        trip = symbolic.simplify(ub - lb + 1)
        try:
            if bool((trip % W).simplify() == 0):
                return False
            if bool((trip < W).simplify()):
                return False
        except Exception:
            pass

        # main runs lb..main_end step 1 (vector body W-wide; emission handles the W-stride);
        # remainder runs main_end+1..ub.
        #
        # ``int_floor`` (NOT sympy ``//``) so C++ codegen emits correct integer division. sympy
        # simplifies ``(LEN_1D - 1) // 8`` to ``floor(LEN_1D/8 - 1/8)``, printed as
        # ``(LEN_1D / 8) - (1 / 8)``; in C++ ``1 / 8`` == 0, so bound becomes ``LEN_1D / 8`` not
        # ``(LEN_1D - 1) / 8``. For LEN_1D=64 range=63 main loop runs one extra W-tile (index 63
        # not 55), overwriting pre-loop scalar writes to ``a[LEN_1D - 1]`` (TSVC s2244 failure).
        main_end = lb + (symbolic.int_floor(trip, W) * W) - 1
        rem_start = main_end + 1

        # Replicate the whole scope (entry + body + exit) for the remainder.
        scope_view = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
        rem_scope = replicate_scope(state.sdfg, state, scope_view)

        # Tighten main + remainder innermost ranges. Both modes emit a step-1 length-R remainder;
        # only schedule + label marker differ:
        # - ``scalar``: Sequential schedule -> vectorizer skips it (plain scalar tail loop).
        # - ``masked``: default schedule + ``__masked_rem`` marker -> P3 (GenerateIterationMask)
        #   attaches ``_iter_mask``, vectorizer tiles to step-W trip-1 (MapTiling
        #   ``divides_evenly`` hint), masked emitter (C.2-b) routes body tasklets to their
        #   ``_masked`` variant so trailing OOB lanes are gated.
        map_entry.map.range[-1] = (lb, main_end, 1)
        rem_scope.entry.map.range[-1] = (rem_start, ub, 1)

        # Remainder = trailing tail, never the parallel hot path. Main keeps its default schedule
        # (P2 retightens range only) -> parallelism unaffected.
        if self.mode == "scalar":
            # Sequential: off parallel path; ``vectorize.py`` skips Sequential maps.
            rem_scope.entry.map.schedule = dace.dtypes.ScheduleType.Sequential
        else:  # "masked"
            # ``__masked_rem`` marker; schedule stays default so vectorizer still tiles to step-W.
            rem_scope.entry.map.label = rem_scope.entry.map.label + "__masked_rem"
        return True
