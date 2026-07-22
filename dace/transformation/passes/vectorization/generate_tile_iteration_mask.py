# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GenerateTileIterationMask`` — allocate the K-dim ``_tile_iter_mask``
transient and the producing :class:`TileMaskGen` lib node inside every
K-dim eligible inner-map outer scope.

The mask lives directly in the parent state (between ``MapEntry`` and
the body) as a register transient, so downstream :class:`ConvertTaskletsToTileOps`
can wire it into every lib node without crossing a NestedSDFG boundary.
"""
from typing import Dict, Optional, Tuple

import dace
from dace import properties
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.libraries.tileops import TileMaskGen
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER, TILE_MAIN_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map
from dace.transformation.passes.vectorization.utils.mask_scaffold import (prepend_dominating_init_state,
                                                                          thread_symbols_into_nsdfg)
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch,
                                                                            tile_mask_gen_dominates_consumers)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _mask_array_name_for(parent_sdfg: dace.SDFG) -> str:
    """Build a per-map mask array name, unique within the SDFG.

    Several inner maps can coexist in one state (e.g. jacobi2d's B-update and
    A-update); each needs its OWN mask transient + access node in its OWN scope,
    else ConvertTaskletsToTileOps would wire one map's mask into another's (disjoint) scope.

    :param parent_sdfg: SDFG the mask array is added to.
    :returns: ``"_tile_iter_mask"`` for the first map, then
        ``"_tile_iter_mask_1"``, ... — the first name not in ``parent_sdfg.arrays``.
    """
    base = TileNameScheme.ITER_MASK
    if base not in parent_sdfg.arrays:
        return base
    idx = 1
    while f"{base}_{idx}" in parent_sdfg.arrays:
        idx += 1
    return f"{base}_{idx}"


@properties.make_properties
class GenerateTileIterationMask(ppl.Pass):
    """Attach a K-dim iteration mask to every K-dim eligible inner map.

    For each inner map: adds ``_tile_iter_mask : bool[widths]`` (a
    Register transient) and prepends a :class:`TileMaskGen` lib node
    inside the map scope that writes the mask. The mask is consumed by
    every downstream :class:`TileLoad` / :class:`TileBinop` /
    :class:`TileStore` placed inside the same scope.

    Idempotent — re-running on an already-masked map is a no-op.
    """

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"GenerateTileIterationMask: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        """Adds arrays + lib nodes -> Everything."""
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Idempotent -> False."""
        return False

    def _spec_for(self, map_entry: MapEntry) -> TileDimSpec:
        """Rebuild a :class:`TileDimSpec` from a map's last K params.

        :param map_entry: Inner map entry.
        :returns: A fresh :class:`TileDimSpec` covering the K innermost
            dims; ``global_ubs[k]`` is ``str(ub_k + 1)`` (exclusive).
        """
        K = len(self.widths)
        params = list(map_entry.map.params)
        ranges = list(map_entry.map.range.ranges)
        iter_vars = tuple(params[-K:])
        global_ubs = tuple(str(r[1] + 1) for r in ranges[-K:])
        return TileDimSpec(iter_vars=iter_vars, widths=tuple(self.widths), global_ubs=global_ubs)

    def _attach_mask(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, map_entry: MapEntry,
                     spec: TileDimSpec) -> bool:
        """Add the mask transient + producer :class:`TileMaskGen` INSIDE the body NSDFG.

        Per design 6.5 / 6.7 + user direction 2026-06-10: the mask lives where
        the lib nodes consume it — inside the body NSDFG. The walker + converter
        detect the inner ``_tile_iter_mask`` AccessNode and wire ``has_mask=True``
        + ``_mask`` onto TileLoad / TileStore / Tile{Binop,Unop,ITE,Reduce}.

        :param parent_sdfg: SDFG owning ``parent_state``.
        :param parent_state: State holding the inner map.
        :param map_entry: Inner map entry.
        :param spec: Per-dim tile specification.
        :returns: ``True`` when a mask was added; ``False`` if the body NSDFG
            already has a ``TileMaskGen`` (idempotent per-map).
        """
        scope = parent_state.all_nodes_between(map_entry, parent_state.exit_node(map_entry)) or set()
        # The body NSDFG should exist inside the scope (NestInnermost runs first).
        body_nsdfgs = [n for n in scope if isinstance(n, dace.nodes.NestedSDFG)]
        if not body_nsdfgs:
            # Defensive: unnested kernels shouldn't occur under walker-primary pipeline.
            return False
        body_nsdfg = body_nsdfgs[0]
        inner_sdfg = body_nsdfg.sdfg
        # Idempotency: skip if the body already has a TileMaskGen.
        for inner_state in inner_sdfg.states():
            if any(isinstance(n, TileMaskGen) for n in inner_state.nodes()):
                return False
        # Pick a unique name in the inner SDFG's arrays.
        mask_name = TileNameScheme.ITER_MASK
        if mask_name in inner_sdfg.arrays:
            idx = 1
            while f"{mask_name}_{idx}" in inner_sdfg.arrays:
                idx += 1
            mask_name = f"{mask_name}_{idx}"
        inner_sdfg.add_array(
            mask_name,
            list(spec.widths),
            dace.bool_,
            storage=dace.dtypes.StorageType.Register,
            transient=True,
        )

        # Mask must be GENERATED in a state that DOMINATES every masked consumer.
        # A flat body is one state, but a data-dependent ``if`` (TileITE) body has
        # several (``compute_then``/``compute_else``/``apply_ITE``); a producer in
        # a non-dominating branch would let other branches read ``_tile_iter_mask``
        # uninitialized (flaky). Prepend a dedicated start state so the mask
        # dominates the whole body (shared with GenerateIterationMask's
        # ``_iter_mask_init``).
        def _build_mask(init_state: dace.SDFGState) -> None:
            mask_node = TileMaskGen(
                name="_tile_iter_mask_gen",
                widths=spec.widths,
                iter_vars=spec.iter_vars,
                global_ubs=spec.global_ubs,
            )
            init_state.add_node(mask_node)
            mask_access = init_state.add_access(mask_name)
            subset = ", ".join(f"0:{w}" for w in spec.widths)
            init_state.add_edge(mask_node, "_o", mask_access, None, dace.Memlet(f"{mask_name}[{subset}]"))

        prepend_dominating_init_state(inner_sdfg, "_tile_mask_init", _build_mask)
        # ``global_ubs`` reference outer-scope loop bounds (e.g. ``kfdia``). A
        # bound symbol the body NSDFG doesn't otherwise use (unlike a shape symbol
        # like ``klev``) must be threaded in, else the TileMaskGen tasklet fails to
        # compile (``'kfdia' was not declared``).
        import dace.symbolic as _sym
        ub_syms = {str(s) for ub in spec.global_ubs for s in _sym.pystr_to_symbolic(str(ub)).free_symbols}
        thread_symbols_into_nsdfg(inner_sdfg, body_nsdfg, ub_syms, parent_sdfg)
        return True

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Optional[Dict]) -> Optional[int]:
        """Walk every innermost map and attach the mask to its scope.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Reads ``"MarkTileDims"`` when present.
        :returns: Number of maps with a fresh mask, or ``None`` if none.
        """
        specs: Optional[Dict[MapEntry, TileDimSpec]] = None
        if pipeline_results and "MarkTileDims" in pipeline_results:
            specs = pipeline_results["MarkTileDims"]
        attached = 0
        K = len(self.widths)
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry) or not isinstance(g, dace.SDFGState):
                continue
            if not is_vectorizable_map(g, n, len(self.widths)):
                continue
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: no mask
                continue
            # ``__tile_k1_tail`` postamble: K=1 widths=(1,), runs element by
            # element. Every iteration is in bounds by construction — no mask.
            if n.map.label.endswith(TILE_K1_TAIL_MARKER):
                continue
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < K:
                continue
            # The all-main interior region of a ``masked_tail`` split is fully
            # in bounds on every tiled dim — skip the mask so the descent / emit
            # lower it with ``has_mask=False`` (the fast path).
            if n.map.label.endswith(TILE_MAIN_MARKER):
                continue
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            if self._attach_mask(g.sdfg, g, n, spec):
                attached += 1
        assert_invariant(no_memlet_dim_mismatch(sdfg), "GenerateTileIterationMask",
                         "memlet subset and other_subset have matching dimensionality")
        assert_invariant(tile_mask_gen_dominates_consumers(sdfg), "GenerateTileIterationMask",
                         "every TileMaskGen lives in its SDFG start block (dominates masked consumers)")
        return attached or None
