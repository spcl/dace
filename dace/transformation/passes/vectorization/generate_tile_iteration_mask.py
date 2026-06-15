# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GenerateTileIterationMask`` — allocate the K-dim ``_tile_iter_mask``
transient and the producing :class:`TileMaskGen` lib node inside every
K-dim eligible inner-map outer scope.

The mask lives directly in the parent state (between ``MapEntry`` and
the body) as a register transient, so downstream :class:`EmitTileOps`
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
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                             no_memlet_dim_mismatch)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _mask_array_name_for(parent_sdfg: dace.SDFG) -> str:
    """Build a per-map mask array name, unique within the SDFG.

    Several inner maps can coexist in one state (e.g. jacobi2d's
    B-update and A-update); each needs its OWN mask transient + access
    node in its OWN scope, otherwise EmitTileOps would wire one map's
    mask into another map's (disjoint) scope.

    :param parent_sdfg: SDFG the mask array is added to.
    :returns: ``"_tile_iter_mask"`` for the first map, then
        ``"_tile_iter_mask_1"``, ``"_tile_iter_mask_2"``, ... — the
        first name not already present in ``parent_sdfg.arrays``.
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
        """Pass adds arrays and lib nodes.

        :returns: ``ppl.Modifies.Everything``.
        """
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Idempotent — runs once.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
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
        """Add the mask transient + the producer :class:`TileMaskGen` INSIDE the body NSDFG.

        Per design 6.5 / 6.7 + user direction 2026-06-10: the mask lives WHERE the lib nodes
        consume it -- inside the body NSDFG. The walker and converter then detect the
        inner ``_tile_iter_mask`` AccessNode and wire ``has_mask=True`` + ``_mask`` onto
        TileLoad / TileStore / Tile{Binop, Unop, ITE, Reduce} lib nodes.

        :param parent_sdfg: SDFG owning ``parent_state``.
        :param parent_state: State holding the inner map.
        :param map_entry: Inner map entry.
        :param spec: Per-dim tile specification.
        :returns: ``True`` when a mask was added; ``False`` if the body NSDFG already has
            a ``TileMaskGen`` in its inner state (idempotent per-map).
        """
        scope = parent_state.all_nodes_between(map_entry, parent_state.exit_node(map_entry)) or set()
        # The body NSDFG should exist inside the scope (NestInnermost runs first).
        body_nsdfgs = [n for n in scope if isinstance(n, dace.nodes.NestedSDFG)]
        if not body_nsdfgs:
            # Fall back to outer-scope emission for any kernels that didn't get nested
            # (defensive: should not happen under the walker-primary pipeline).
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
        # Insert TileMaskGen + access node into the body NSDFG's first SDFGState.
        # When the body has nested CFG structure (``insert_copies`` /
        # ``fuse_overlapping_loads`` may produce this), ``start_state`` returns a
        # ControlFlowRegion -- which can't host AccessNodes / library nodes. Walk
        # the flat ``states()`` iterator (which already flattens CFG regions to
        # their leaf SDFGStates) and pick the first one.
        inner_state = next(iter(inner_sdfg.states()))
        mask_node = TileMaskGen(
            name="_tile_iter_mask_gen",
            widths=spec.widths,
            iter_vars=spec.iter_vars,
            global_ubs=spec.global_ubs,
        )
        inner_state.add_node(mask_node)
        mask_access = inner_state.add_access(mask_name)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        inner_state.add_edge(
            mask_node,
            "_o",
            mask_access,
            None,
            dace.Memlet(f"{mask_name}[{subset}]"),
        )
        # The mask's per-dim upper bounds (``global_ubs``) reference outer-scope
        # symbols (loop bounds such as ``kfdia``). A bound symbol the body NSDFG
        # does not otherwise use -- i.e. not an array-shape symbol like ``klev`` --
        # is absent from both the inner SDFG's symbol table AND the NestedSDFG's
        # ``symbol_mapping``, so the generated body function never receives it and
        # the TileMaskGen tasklet fails to compile (``'kfdia' was not declared``).
        # Thread every ``global_ubs`` free symbol into the body NSDFG: declare it on
        # the inner SDFG and map it (identity) from the same-named parent symbol.
        import dace.symbolic as _sym
        for ub in spec.global_ubs:
            for s in _sym.pystr_to_symbolic(str(ub)).free_symbols:
                sname = str(s)
                if sname not in inner_sdfg.symbols:
                    dtype = parent_sdfg.symbols[sname] if sname in parent_sdfg.symbols else dace.dtypes.int64
                    inner_sdfg.add_symbol(sname, dtype)
                if sname not in body_nsdfg.symbol_mapping:
                    body_nsdfg.symbol_mapping[sname] = _sym.pystr_to_symbolic(sname)
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
            if not is_innermost_map(g, n):
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
        return attached or None
