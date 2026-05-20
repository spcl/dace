# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GenerateTileIterationMask`` — allocate the K-dim ``_tile_iter_mask``
transient and the producing :class:`TileMaskGen` lib node inside every
K-dim eligible inner-map outer scope.

The mask lives directly in the parent state (between ``MapEntry`` and
the body) as a register transient, so downstream :class:`EmitTileOps`
can wire it into every lib node without crossing a NestedSDFG boundary.
"""
from typing import Dict, List, Optional, Tuple

import dace
from dace import properties, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.libraries.tileops import TileMaskGen
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _mask_array_name_for(map_entry: MapEntry) -> str:
    """Build a per-map mask array name to keep names distinct when
    several inner maps coexist in the same SDFG.

    :param map_entry: Inner map entry the mask is being attached to.
    :returns: ``"_tile_iter_mask"`` for the first map, ``"<base>_<n>"``
        for subsequent ones.
    """
    return TileNameScheme.ITER_MASK


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

    def __init__(self, widths: Tuple[int, ...] = (8,)):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(
                f"GenerateTileIterationMask: widths length {len(widths)} not in {{1, 2, 3}}"
            )
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

    def _attach_mask(self,
                     parent_sdfg: dace.SDFG,
                     parent_state: dace.SDFGState,
                     map_entry: MapEntry,
                     spec: TileDimSpec) -> bool:
        """Add the mask transient + the producer :class:`TileMaskGen`
        inside the map scope.

        :param parent_sdfg: SDFG owning ``parent_state``.
        :param parent_state: State holding the inner map.
        :param map_entry: Inner map entry.
        :param spec: Per-dim tile specification.
        :returns: ``True`` when a mask was added; ``False`` if a
            ``_tile_iter_mask`` array already exists (idempotent).
        """
        mask_name = _mask_array_name_for(map_entry)
        if mask_name in parent_sdfg.arrays:
            return False
        parent_sdfg.add_array(
            mask_name,
            list(spec.widths),
            dace.bool_,
            storage=dace.dtypes.StorageType.Register,
            transient=True,
        )
        mask_node = TileMaskGen(
            name="_tile_iter_mask_gen",
            widths=spec.widths,
            iter_vars=spec.iter_vars,
            global_ubs=spec.global_ubs,
        )
        parent_state.add_node(mask_node)
        mask_access = parent_state.add_access(mask_name)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        parent_state.add_edge(
            mask_node,
            "_o",
            mask_access,
            None,
            dace.Memlet(f"{mask_name}[{subset}]"),
        )
        parent_state.add_nedge(map_entry, mask_node, dace.Memlet())
        parent_state.add_nedge(mask_access, parent_state.exit_node(map_entry), dace.Memlet())
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
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < K:
                continue
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            if self._attach_mask(g.sdfg, g, n, spec):
                attached += 1
        return attached or None
