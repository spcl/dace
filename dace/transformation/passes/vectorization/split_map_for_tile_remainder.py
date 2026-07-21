# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitMapForTileRemainder`` — peel a K-dim tile map into a divisible interior
plus masked boundary slabs (``masked_tail`` strategy).

K-slab loop-peeling decomposition (K+1 regions, NOT the 2**K Cartesian corner
split). Interior = every tiled dim tightened to the largest multiple-of-W bound.
Slab for dim ``d``: dims ``< d`` at interior extent, dim ``d`` = trailing tail,
dims ``> d`` at full extent. Slabs pairwise disjoint + interior tile the whole
space -> a tile is in a slab iff any tiled dim is partial. Corner absorption:
each slab claims exactly one dim's tail over the full extent of later dims,
which is why K slabs suffice instead of 2**K - 1 Cartesian cells.

Interior marked ``__tile_main`` -> GenerateTileIterationMask skips its mask,
descent/emit lower it has_mask=False (perf fast path). Each slab keeps its mask.

K=2 layout (aligned bounds ``A_d = floor(N_d / W) * W``)::

            dim1 ->
           [0 : A1)        [A1 : N1)
          +---------------+--------+
    [0:A0)|   INTERIOR    | slab 1 |   slab_1 = [0:A0, A1:N1]
     dim0 | (mask-free)   |        |   (dim1 tail, dim0 INTERIOR height)
          +---------------+--------+
   [A0:N0)|     slab 0  (full dim1 width)  |   slab_0 = [A0:N0, 0:N1]
          +-------------------------------+   (dim0 tail, dim1 FULL width)

``slab_0`` takes dim0's tail across the *full* dim1 extent (absorbs the
both-dims-partial corner); ``slab_1`` takes dim1's tail only over dim0's
*interior* height (corner already slab_0's).

Runs BEFORE :class:`MarkTileDims` (new boundary maps get tagged/tiled like the
original) and on step-1 maps (before :class:`StrideMapByTileWidths`). A dim
provably divisible by ``W`` is not split -> a fully-divisible map yields just
the mask-free interior, no remainder.
"""
from typing import Optional, Tuple

import dace
from dace import properties, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import replicate_scope
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)

# Label suffix marking the fully-in-bounds interior a tile-remainder split
# produces. GenerateTileIterationMask sees it -> skips mask -> has_mask=False.
TILE_MAIN_MARKER = "__tile_main"

# Label suffix: boundary region runs as a plain step-1 scalar loop
# (scalar_postamble tail). Every tile prep pass (MarkTileDims /
# GenerateTileIterationMask / StrideMapByTileWidths / InsertTileLoadStore /
# ConvertTaskletsToTileOps) skips this suffix -> tail keeps its original scalar
# body: not tiled, strided, or masked.
SCALAR_TAIL_MARKER = "__scalar_tail"

# Label suffix: boundary region flows through the tile-op pipeline at K=1
# widths=(1,) — single-lane "scalar tile" remainder. Every tile prep pass treats
# it as tile-main pinned K=1 w=1: stride 1 (no W-stride), no mask, body rewritten
# to tile ops (TileBinop/TileLoad/TileStore at one lane). Uniform remainder
# emission when opted in via ``scalar_remainder_emit="tile"`` on the orchestrator.
TILE_K1_TAIL_MARKER = "__tile_k1_tail"


@properties.make_properties
class SplitMapForTileRemainder(ppl.Pass):
    """Split each K-dim tile map into a divisible interior + masked boundary
    regions, marking the interior ``__tile_main``. See module docstring."""

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    tail_mode = properties.Property(
        dtype=str,
        allow_none=False,
        default="masked",
        desc="Boundary-region handling: 'masked' (W-strided masked slabs, the "
        "masked_tail strategy), 'scalar' (step-1 scalar-loop slabs marked "
        "__scalar_tail, the scalar_postamble strategy), or 'tile_k1' (step-1 "
        "tile-op slabs at widths=(1,) marked __tile_k1_tail, the K=0/single-lane "
        "tile-op variant of the scalar_postamble strategy).",
    )

    assume_even = properties.Property(
        dtype=bool,
        default=False,
        desc="Assume every tiled map extent is an exact multiple of its width, so there is "
        "NO remainder: mark every eligible map ``__tile_main`` (mask-free) WITHOUT peeling a "
        "boundary slab. The single strided ``0:N:W`` map covers the whole range. Used by the "
        "GPU path, which emits no remainder loop; the caller guarantees the even extent.",
    )

    range_check = properties.Property(
        dtype=bool,
        default=True,
        desc="Under ``assume_even``, guard the even-extent assumption at runtime: for every tiled "
        "dim whose extent is not PROVABLY a multiple of its width, emit a host-side side-effect "
        "tasklet that checks ``extent % W == 0`` before the kernel and, on violation, writes to "
        "stderr and traps (``abort``) rather than silently reading/writing out of bounds. A "
        "provably-divisible extent needs no check. Ignored when ``assume_even`` is False (the "
        "remainder is peeled, so no assumption to guard).",
    )

    def __init__(self,
                 widths: Tuple[int, ...] = (8, ),
                 tail_mode: str = "masked",
                 assume_even: bool = False,
                 range_check: bool = True):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param tail_mode: ``"masked"`` (W-strided masked slabs), ``"scalar"``
            (step-1 scalar-loop slabs marked :data:`SCALAR_TAIL_MARKER`), or
            ``"tile_k1"`` (step-1 tile-op slabs at ``widths=(1,)`` marked
            :data:`TILE_K1_TAIL_MARKER`, the single-lane tile-op variant).
        :param assume_even: ``True`` -> mark every eligible map ``__tile_main``
            without splitting (whole extent assumed divisible, no remainder).
        :param range_check: Under ``assume_even``, emit a host-side runtime
            ``extent % W == 0`` guard (stderr + ``abort`` on violation) for every
            not-provably-divisible tiled dim. No effect when ``assume_even`` is False.
        :raises ValueError: If ``widths`` length not in ``{1, 2, 3}`` or
            ``tail_mode`` invalid.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"SplitMapForTileRemainder: widths length {len(widths)} not in {{1, 2, 3}}")
        if tail_mode not in ("masked", "scalar", "tile_k1"):
            raise ValueError(f"SplitMapForTileRemainder: tail_mode {tail_mode!r} not in "
                             f"{{'masked', 'scalar', 'tile_k1'}}")
        self.widths = list(widths)
        self.tail_mode = tail_mode
        self.assume_even = assume_even
        self.range_check = range_check

    def modifies(self) -> ppl.Modifies:
        """Pass replicates scopes and retightens ranges.

        :returns: ``ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Scopes``.
        """
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Runs once.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _provably_divisible(self, lb, ub, W: int) -> bool:
        """Whether dim ``[lb:ub]`` is provably a whole number of tiles.

        Only a provably-divisible dim needs no split (all no-mask interior,
        ``has_mask=False``). Every other dim -> (optionally empty) interior +
        w-mask remainder tile. SHORT dim (``trip < W``) -> EMPTY interior + one
        masked tile over the whole dim (mask ``l < trip``); ``trip == 0`` =
        all-false-mask no-op.

        :param lb: Inclusive lower bound.
        :param ub: Inclusive upper bound.
        :param W: Tile width.
        :returns: ``True`` iff trip is provably a multiple of ``W``.
        """
        trip = symbolic.simplify(ub - lb + 1)
        try:
            return bool((trip % W).simplify() == 0)
        except Exception:  # noqa: BLE001 - non-decidable symbolic trip -> split
            return False

    def _trip_class(self, lb, ub, W: int) -> str:
        """Classify a tiled dim's extent against width ``W``.

        ``'divisible'``   -- provably a whole number of tiles (constant OR symbolic like ``4*M``).
        ``'below'``       -- provably ``< W``: too small to tile, keep the map scalar.
        ``'nondivisible'``-- provably not a whole multiple of ``W`` (a constant ``>= W``, or a
                             symbolic extent whose remainder reduces to a nonzero constant, e.g.
                             ``4*M + 1``): a provable ``assume_even`` violation (rerun with
                             ``assume_even=False``).
        ``'symbolic'``    -- non-decidable extent: guard the assumption at runtime.
        """
        if self._provably_divisible(lb, ub, W):  # constant or symbolic (``4*M % 4 == 0``)
            return 'divisible'
        trip = symbolic.simplify(ub - lb + 1)
        try:
            t = int(trip)
        except (TypeError, ValueError):
            # Symbolic, not provably divisible: a remainder that reduces to a nonzero CONSTANT is a
            # provable violation; an undecidable remainder falls through to a runtime guard.
            try:
                if int((trip % W).simplify()) != 0:
                    return 'nondivisible'
            except (TypeError, ValueError):
                pass
            return 'symbolic'
        return 'below' if t < W else 'nondivisible'

    def _split(self, state: dace.SDFGState, map_entry: MapEntry, K: int) -> bool:
        """Peel ``map_entry``'s K innermost dims into interior + K slabs.

        See module docstring for the K-slab decomposition. Interior =
        ``map_entry`` itself tightened on every dim, marked ``__tile_main``.
        Each slab = fresh ``replicate_scope`` copy of the interior-so-far with
        dim ``d`` set to its tail (dims ``> d`` still full — not yet tightened).

        :param state: State holding the map.
        :param map_entry: Innermost map entry to peel (becomes the interior).
        :param K: Number of tiled (innermost) dims.
        :returns: ``True`` if interior marked (always, when map has >= K dims);
            ``False`` if map too small.
        """
        ranges = list(map_entry.map.range.ranges)
        if len(ranges) < K:
            return False
        tiled_dims = list(range(len(ranges) - K, len(ranges)))
        # ``assume_even``: caller guarantees every tiled extent is a multiple of W,
        # so no boundary -> skip peel, mark whole map ``__tile_main`` (mask-free).
        if self.assume_even:
            classes = []
            for d, W in zip(tiled_dims, self.widths):
                lb, ub, _ = map_entry.map.range[d]
                classes.append((self._trip_class(lb, ub, W), d, W, lb, ub))
            # A provably-too-small dim (extent < W) cannot be tiled with no remainder to cover it
            # -> keep the WHOLE map scalar. ``MarkTileDims`` refuses the same dim under
            # ``assume_even``, so the two passes agree (no strided-map/scalar-body desync). Takes
            # precedence over a nondivisible sibling dim: an untiled map is never wrong. On the
            # masked / scalar-tail paths below there IS a remainder, so a short dim is peeled into
            # an empty interior plus one masked tile and stays tiled.
            if any(c == 'below' for c, *_ in classes):
                return False
            for c, d, W, lb, ub in classes:
                if c == 'nondivisible':
                    # Provable violation of the caller's even-extent guarantee: fail loudly at
                    # transform time (a runtime guard would only abort once the kernel launches).
                    raise ValueError(f"SplitMapForTileRemainder: map {map_entry.map.label!r} dim {d} has extent "
                                     f"{symbolic.simplify(ub - lb + 1)}, which is provably not a multiple of tile "
                                     f"width {W} that assume_even requires. Rerun with assume_even=False to peel a "
                                     f"masked remainder.")
                # Non-decidable extent: record a host-side runtime guard (extent % W == 0 and
                # extent >= W). A provably-divisible extent needs no check.
                if c == 'symbolic' and self.range_check:
                    self._range_checks.append((state.sdfg, symbolic.simplify(ub - lb + 1), int(W)))
            if not map_entry.map.label.endswith(TILE_MAIN_MARKER):
                map_entry.map.label = map_entry.map.label + TILE_MAIN_MARKER
            return True
        for d, W in zip(tiled_dims, self.widths):
            lb, ub, step = map_entry.map.range[d]
            if self._provably_divisible(lb, ub, W):
                continue
            trip = symbolic.simplify(ub - lb + 1)
            # ``int_floor`` (not ``//``) -> C++ codegen emits integer division
            # (see SplitMapForVectorRemainder for the full rationale).
            main_end = lb + (symbolic.int_floor(trip, W) * W) - 1
            # Slab for dim ``d`` = copy of interior-so-far with dim d set to tail.
            # Replicate BEFORE tightening dim d on the interior.
            scope_view = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
            slab = replicate_scope(state.sdfg, state, scope_view)
            slab.entry.map.range[d] = (main_end + 1, ub, step)
            map_entry.map.range[d] = (lb, main_end, step)
            # ``scalar`` mode: mark slab -> tile prep passes skip it -> stays a
            # plain step-1 scalar loop with the original body.
            if self.tail_mode == "scalar" and not slab.entry.map.label.endswith(SCALAR_TAIL_MARKER):
                slab.entry.map.label = slab.entry.map.label + SCALAR_TAIL_MARKER
            # ``tile_k1`` mode: mark slab -> treated as tile-main pinned K=1
            # widths=(1,): step 1, no mask, body lowered to single-lane tile ops.
            elif self.tail_mode == "tile_k1" and not slab.entry.map.label.endswith(TILE_K1_TAIL_MARKER):
                slab.entry.map.label = slab.entry.map.label + TILE_K1_TAIL_MARKER
        # Interior fully tiled on every dim (skipped = divisible, peeled =
        # tightened to multiple of W) -> mark mask-free.
        if not map_entry.map.label.endswith(TILE_MAIN_MARKER):
            map_entry.map.label = map_entry.map.label + TILE_MAIN_MARKER
        return True

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Split every eligible innermost K-dim map into interior + remainders.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of maps split, or ``None`` if none.
        """
        K = len(self.widths)
        applied = 0
        # Collector for assume_even runtime guards, filled by ``_split`` and drained below.
        self._range_checks = []
        # Snapshot up front: splitting mutates the graph; must not re-split a
        # freshly replicated remainder map.
        eligible = [
            (n, g) for n, g in sdfg.all_nodes_recursive()
            if isinstance(n, MapEntry) and isinstance(g, dace.SDFGState) and is_vectorizable_map(
                g, n, len(self.widths)) and len(n.map.params) >= K and not n.map.label.endswith(TILE_MAIN_MARKER)
            and not n.map.label.endswith(SCALAR_TAIL_MARKER) and not n.map.label.endswith(TILE_K1_TAIL_MARKER)
        ]
        for n, g in eligible:
            if self._split(g, n, K):
                applied += 1
        if self.assume_even and self.range_check and self._range_checks:
            self._emit_range_checks()
        if applied:
            # ``replicate_scope`` deep-copies body NestedSDFGs without registering
            # the clone in ``cfg_list``; rebuild so later passes (and
            # ``expand_library_nodes``) resolve the new nested CFGs.
            sdfg.reset_cfg_list()
        assert_invariant(no_memlet_dim_mismatch(sdfg), "SplitMapForTileRemainder",
                         "memlet subset and other_subset have matching dimensionality")
        return applied or None

    def _emit_range_checks(self) -> None:
        """Emit the host-side even-extent guards recorded during ``assume_even`` splitting.

        One guard state is prepended (as the new start block) to each SDFG that owns a checked
        map, so the check runs before that SDFG's kernels. Every distinct ``(extent, width)``
        pair becomes a side-effect CPP tasklet that writes to stderr and ``abort``\\ s when the
        even-extent assumption is violated at runtime -- ``extent`` is not a whole multiple of
        ``width`` OR ``extent < width`` (a symbolic extent that turns out shorter than one tile) --
        turning a silent out-of-bounds tile access into a loud, deterministic failure that names
        the fix (rerun with ``assume_even=False``). Host-side, so it guards CPU and GPU alike.
        ``side_effects=True`` keeps the guard from being fused into a neighbour or eliminated as
        dead code.
        """
        by_sdfg = {}
        for owner, extent, width in self._range_checks:
            by_sdfg.setdefault(owner, set()).add((symbolic.symstr(extent, cpp_mode=True), width))
        for owner, checks in by_sdfg.items():
            guard = owner.add_state_before(owner.start_block, label="tile_even_range_check", is_start_block=True)
            for extent_c, width in sorted(checks):
                code = (f'if ((long long)({extent_c}) % {width} != 0 || (long long)({extent_c}) < {width}) {{\n'
                        f'    fprintf(stderr, "DaCe tile vectorization: extent %lld violates assume_even '
                        f'(must be a multiple of tile width {width} and >= {width}); rerun with '
                        f'assume_even=False.\\n", (long long)({extent_c}));\n'
                        f'    abort();\n'
                        f'}}')
                tasklet = guard.add_tasklet(name="tile_range_check",
                                            inputs={},
                                            outputs={},
                                            code=code,
                                            language=dace.dtypes.Language.CPP)
                tasklet.side_effects = True
