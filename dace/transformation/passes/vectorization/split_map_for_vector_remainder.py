# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split innermost maps into a vectorisable main map plus a trailing remainder.

For every innermost step-1 map whose range is not provably divisible by
``vector_width``, the map is split into a main map (the vectorisation
target) and a remainder map covering the leftover trailing iterations.
The remainder is either a sequential scalar tail (``scalar`` mode) or a
masked step-``vector_width`` body that relies on a later pass to attach
the iteration-mask connector (``masked`` mode).
"""
from typing import Optional

import dace
from dace import properties, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import replicate_scope
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map


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
        """Initialize the pass.

        :param vector_width: SIMD lane count; the main map's vector width.
        :param mode: ``scalar`` for a sequential step-1 remainder, ``masked``
            for a step-``vector_width`` remainder masked by a later pass.
        """
        super().__init__()
        self.vector_width = vector_width
        self.mode = mode

    def modifies(self) -> ppl.Modifies:
        """Return the set of SDFG elements this pass may modify."""
        return ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Return whether the pass should run again after modifications."""
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
        """Split one innermost map into a main map and a remainder map.

        :param state: The state containing the map.
        :param map_entry: The innermost map entry to split.
        :param W: Vector width used for the main map and divisibility check.
        :returns: ``True`` if the map was split, ``False`` if skipped.
        """
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
        #
        # Use ``dace.symbolic.int_floor`` (NOT sympy's ``//``) so the C++
        # codegen emits correct integer division.  sympy simplifies
        # ``(LEN_1D - 1) // 8`` to ``floor(LEN_1D/8 - 1/8)`` which the
        # codegen prints as ``(LEN_1D / 8) - (1 / 8)`` — in C++ integer
        # division ``1 / 8`` is 0, so the main bound becomes ``LEN_1D / 8``
        # instead of ``(LEN_1D - 1) / 8``.  For ``LEN_1D=64, range=63`` the
        # main loop then iterates one extra W-tile (up to index 63 instead
        # of stopping at 55), overwriting pre-loop scalar writes to
        # ``a[LEN_1D - 1]`` (TSVC s2244 failure).
        main_end = lb + (symbolic.int_floor(trip, W) * W) - 1
        rem_start = main_end + 1

        # Replicate the entire scope (entry + body + exit) for the remainder.
        scope_view = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
        rem_scope = replicate_scope(state.sdfg, state, scope_view)

        # Tighten the main map's innermost range and the remainder's. Both
        # ``scalar`` and ``masked`` modes emit a step-1 length-R remainder;
        # the only difference is the schedule + label marker:
        #
        # - ``scalar``: Sequential schedule so the vectorizer leaves it alone
        #   (codegen emits a plain scalar tail loop).
        # - ``masked``: default schedule + ``__masked_rem`` label marker so
        #   ``GenerateIterationMask`` (P3) attaches an ``_iter_mask`` to the
        #   body and the vectorizer tiles the map to step-W trip-1 via
        #   MapTiling's ``divides_evenly`` hint. The masked emitter (C.2-b)
        #   then routes every body tasklet to its ``_masked`` runtime variant
        #   so the trailing OOB lanes are gated.
        map_entry.map.range[-1] = (lb, main_end, 1)
        rem_scope.entry.map.range[-1] = (rem_start, ub, 1)

        # The remainder is the trailing tail, never the parallel hot
        # path. The main / vectorised map keeps its default schedule
        # (P2 only retightens its range, never its schedule) so its
        # parallelism is unaffected.
        if self.mode == "scalar":
            # Scalar remainder is forced Sequential: this both keeps it
            # off the parallel path AND makes the downstream vectorizer
            # skip it (``vectorize.py`` skips Sequential maps) so it
            # stays a plain scalar tail loop.
            rem_scope.entry.map.schedule = dace.dtypes.ScheduleType.Sequential
        else:  # "masked"
            # The masked remainder is NOT a scalar tail — it is a single
            # masked W-wide iteration the vectorizer must still tile to
            # step-W. It therefore cannot be ``Sequential`` (the
            # vectorizer skips Sequential maps); the ``__masked_rem``
            # label drives P3 + the masked emitter instead.
            rem_scope.entry.map.label = rem_scope.entry.map.label + "__masked_rem"
        return True
