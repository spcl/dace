# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MarkTileDims`` — validation-only pass that picks K innermost
parameters per inner map and constructs a :class:`TileDimSpec` per
candidate.

Runs as the first per-map analysis step in the v2 orchestrator. Loud
failure on any inner map that cannot be K-dim tiled (degenerate dim,
step != 1, fewer than K params, etc.) so the orchestrator's error
points at the map that needs attention rather than downstream
masked-tail emission failing in a confusing way.
"""
from typing import Dict, List, Optional, Tuple

import dace
from dace import dtypes, properties, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


_RUNTIME_GUARD_STATE_LABEL = "_tile_runtime_check"


@properties.make_properties
class MarkTileDims(ppl.Pass):
    """Validate and record the K innermost tiled dims per inner map.

    For every innermost map in the SDFG: take its last ``K`` parameters
    (where ``K = len(widths)``), check step == 1 and trip > 1 on each,
    then build a :class:`TileDimSpec` recording the iter-vars, widths
    and original exclusive upper bounds. The pass returns the
    ``{MapEntry: TileDimSpec}`` map for downstream passes that need it.

    No SDFG mutation. Failures raise loudly (``NotImplementedError``)
    with the offending map identified by its label.
    """

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    skip_ineligible = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, ineligible maps are silently dropped from the result instead of "
        "raising. Default is loud failure so the orchestrator surfaces the problem.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, ), skip_ineligible: bool = False):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param skip_ineligible: When True, soft-skip ineligible inner
            maps; default raises ``NotImplementedError``.
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"MarkTileDims: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)
        self.skip_ineligible = skip_ineligible

    def modifies(self) -> ppl.Modifies:
        """Pass is read-only.

        :returns: ``ppl.Modifies.Nothing``.
        """
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Validation pass; no re-apply is needed.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _classify_one(self, map_entry: MapEntry,
                      runtime_constraints: List[Tuple[str, int, str]]) -> Optional[TileDimSpec]:
        """Build a :class:`TileDimSpec` for ``map_entry`` if eligible.

        :param map_entry: The candidate inner map entry.
        :param runtime_constraints: Out-param. Each symbolic ``trip >= W``
            constraint is appended as ``(trip_str, W, map_label)`` so the
            caller can emit one combined runtime guard at SDFG entry.
        :returns: The spec when the K innermost params each have
            step == 1 and a non-degenerate trip; ``None`` otherwise.
        :raises NotImplementedError: When ``skip_ineligible`` is
            ``False`` and the map is ineligible.
        """
        # ``__tile_k1_tail`` tail maps are pinned to K=1 widths=(1,)
        # regardless of the orchestrator-level widths; the postamble is
        # always a single-lane scalar-tile remainder over the innermost
        # iter var only.
        widths = (1, ) if map_entry.map.label.endswith(TILE_K1_TAIL_MARKER) else tuple(self.widths)
        K = len(widths)
        params = list(map_entry.map.params)
        ranges = list(map_entry.map.range.ranges)
        if len(params) < K:
            return self._fail_or_skip(f"map {map_entry.label!r} has only {len(params)} params (< K={K})")
        iter_vars = tuple(params[-K:])
        slice_ranges = ranges[-K:]
        global_ubs = []
        for (lb, ub, step), iv in zip(slice_ranges, iter_vars):
            if step != 1 and str(step) != "1":
                return self._fail_or_skip(
                    f"map {map_entry.label!r} dim {iv!r} has step {step!r}; v2 requires step == 1")
            trip_expr = symbolic.simplify(ub - lb + 1)
            # Map iter-var ``iv`` is the ``-K + idx``-th param; pair it with
            # ``widths[idx]`` (innermost-last alignment) so a too-small
            # trip is checked against the right width.
            dim_idx = iter_vars.index(iv)
            tile_w = int(widths[dim_idx])
            try:
                trip_int = int(trip_expr)
                if trip_int <= 1:
                    return self._fail_or_skip(f"map {map_entry.label!r} dim {iv!r} has degenerate trip {trip_int} "
                                              f"(must be > 1); flatten the map first")
                if trip_int < tile_w:
                    # Statically too small for a single tile: skip vectorization
                    # for this map entirely. Downstream passes leave the map
                    # alone, codegen sequentialises (and, if small, unrolls)
                    # the iteration. This is the CLOUDSC ``klon, 5, 5`` pattern
                    # — the tiny inner dims are never tiled; only the outer
                    # ``klon``-wide dim gets a tile spec.
                    return None
            except (TypeError, ValueError):
                # Trip is symbolic — we cannot decide at compile time.
                # The pipeline ASSUMES trip >= W for the inner tiled dim
                # (the 99.99% case in real workloads). Record a runtime
                # constraint; the caller emits one combined ``__builtin_trap``
                # guard at SDFG entry so a misuse fails loudly at runtime
                # rather than producing wrong results.
                #
                # Skip when the trip expression is a derived split-shape
                # (contains ``int_floor`` / ``floor`` / ``Mod`` / ...): those
                # belong to interior (``__tile_main``) and slab maps minted by
                # :class:`SplitMapForTileRemainder`. Their trips are
                # ``W*floor(N,W)`` (interior, redundant) and ``N mod W`` (slab,
                # always < W by construction). The original pre-split map's
                # trip ``N`` is what the runtime assumption is about; the guard
                # gets emitted for that one.
                if _is_simple_symbolic_trip(trip_expr):
                    runtime_constraints.append((symbolic.symstr(trip_expr), tile_w,
                                                f"{map_entry.label} dim {iv}"))
            global_ubs.append(str(ub + 1))
        return TileDimSpec(
            iter_vars=iter_vars,
            widths=widths,
            global_ubs=tuple(global_ubs),
        )

    def _fail_or_skip(self, msg: str) -> Optional[TileDimSpec]:
        """Either raise or return ``None`` based on ``skip_ineligible``.

        :param msg: Diagnostic message included in the raised error.
        :returns: ``None`` when ``skip_ineligible`` is True.
        :raises NotImplementedError: When ``skip_ineligible`` is False.
        """
        if self.skip_ineligible:
            return None
        raise NotImplementedError(f"MarkTileDims: {msg}")

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[Dict[MapEntry, TileDimSpec]]:
        """Walk every innermost map and record the K-dim spec.

        :param sdfg: SDFG to analyze.
        :param _: Pipeline-results placeholder (unused).
        :returns: ``{MapEntry: TileDimSpec}`` for every eligible inner
            map; ``None`` when no candidate matched.
        :raises NotImplementedError: When an inner map is ineligible
            and ``skip_ineligible`` is False.
        """
        specs: Dict[MapEntry, TileDimSpec] = {}
        runtime_constraints: List[Tuple[str, int, str]] = []
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry):
                continue
            if not isinstance(g, dace.SDFGState):
                continue
            if not is_innermost_map(g, n):
                continue
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: stays scalar
                continue
            spec = self._classify_one(n, runtime_constraints)
            if spec is not None:
                specs[n] = spec
        if runtime_constraints:
            _emit_runtime_trip_guard(sdfg, runtime_constraints)
        assert_invariant(no_memlet_dim_mismatch(sdfg), "MarkTileDims",
                         "memlet subset and other_subset have matching dimensionality")
        return specs or None


def _is_simple_symbolic_trip(trip_expr) -> bool:
    """Return ``True`` when ``trip_expr`` references only free symbols and
    integer constants — i.e. an original pre-split trip like ``N`` or
    ``M - K``. Return ``False`` for derived split-shape trips that involve
    ``int_floor`` / ``floor`` / ``ceiling`` / ``Mod`` (interior + slab
    trips minted by :class:`SplitMapForTileRemainder`).

    A pure free-symbol trip is what the ``N >= W`` runtime assumption is
    actually about; derived split trips are mechanically constructed and
    do not encode a user-facing assumption.
    """
    import sympy
    try:
        funcs = trip_expr.atoms(sympy.Function)
    except AttributeError:
        return True  # int or plain symbol — simple
    return not funcs


def _emit_runtime_trip_guard(sdfg: dace.SDFG, constraints: List[Tuple[str, int, str]]) -> None:
    """Emit one side-effect tasklet at the SDFG entry that ``__builtin_trap``s
    when any symbolic ``trip < W`` constraint fails at runtime.

    The check enforces the K-dim pipeline's design assumption (``N >= W`` on
    every inner tiled dim). When the trip is statically known, the rule is
    enforced at compile time in :meth:`_classify_one`. When the trip is
    symbolic and depends only on SDFG-level free symbols, this guard catches
    a misuse at runtime instead of producing wrong results.

    Constraints depending on inner-scope vars (outer map iterators, etc.)
    are dropped — they are not visible at SDFG entry so cannot be checked
    there. Idempotent: if the guard state already exists from a prior
    invocation, nothing new is emitted.

    :param sdfg: The SDFG to instrument.
    :param constraints: ``[(trip_expr_str, W, map_label), ...]`` collected
        by the classifier for each symbolic inner-dim trip.
    """
    # Restrict to constraints whose free names are PURE SDFG symbols. Scalar
    # AccessNodes (kfdia, kidia, ...) are passed as data, not as scope-visible
    # C variables, so a zero-connector tasklet referencing them would need
    # input edges — out of scope for this entry guard. Skip those constraints
    # silently; the assumption falls back to "trusted".
    sdfg_syms = set(sdfg.symbols.keys())
    visible = []
    for trip_str, W, label in constraints:
        free = symbolic.symlist(symbolic.pystr_to_symbolic(trip_str)).keys()
        if free and all(str(s) in sdfg_syms for s in free):
            visible.append((trip_str, W, label))
    if not visible:
        return

    # Idempotent: don't double-prepend.
    for state in sdfg.states():
        if state.label == _RUNTIME_GUARD_STATE_LABEL:
            return

    # Build the check code WITHOUT inline comments — DaCe's CodeBlock
    # symbol scanner tokenizes the whole string (including ``// ...``) and
    # any identifier that matches an SDFG symbol would get added to the
    # function arglist (e.g. an outer ``i`` iter-var symbol). The map label
    # is kept in the tasklet name for diagnostics instead.
    check_lines = [f'if (!(({trip_str}) >= {W})) {{ __builtin_trap(); }}' for trip_str, W, _ in visible]
    code = "\n".join(check_lines)
    diag = "__".join(label.replace(" ", "_") for _, _, label in visible)[:80]
    guard_state = sdfg.add_state_before(sdfg.start_block, label=_RUNTIME_GUARD_STATE_LABEL, is_start_block=True)
    guard_state.add_tasklet(
        name=f"tile_runtime_trip_guard__{diag}" if diag else "tile_runtime_trip_guard",
        inputs={},
        outputs={},
        code=code,
        language=dtypes.Language.CPP,
    )
