# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GenerateTileIterationMask`` — allocate the K-dim ``_tile_iter_mask``
transient and the producing :class:`TileMaskGen` lib node inside every
K-dim eligible inner-map body NSDFG.

The pass assumes :class:`NestInnermostMapBodyIntoNSDFG` has already
nested each inner map's body in a NestedSDFG; it then prepends a
single ``_tile_iter_mask_init`` state to the body that runs the
``TileMaskGen`` lib node, exposing ``_tile_iter_mask`` as a body-local
register tile that downstream tile-op lib nodes consume.
"""
from typing import Dict, List, Optional, Tuple

import dace
from dace import properties, symbolic
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.transformation import pass_pipeline as ppl
from dace.libraries.tileops import TileMaskGen
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.map_predicates import (
    get_single_nsdfg_inside_map,
    is_innermost_map,
)
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


@properties.make_properties
class GenerateTileIterationMask(ppl.Pass):
    """Attach a K-dim iteration mask to every K-dim eligible inner map.

    For each inner map whose body is a single NestedSDFG (precondition
    enforced by :class:`NestInnermostMapBodyIntoNSDFG`), the pass:

    1. Adds a ``_tile_iter_mask`` transient (shape ``widths``, dtype
       ``bool_``, storage ``Register``) to the inner SDFG.
    2. Prepends a state running a :class:`TileMaskGen` lib node that
       writes the mask via the per-dim ANY-OOB conjunction
       ``(iter_var_k + l_k < global_ub_k)``.
    3. Threads the outer ``iter_vars`` and any free symbols inside the
       ``global_ubs`` expressions into the inner SDFG's symbol table
       and the parent ``NestedSDFG``'s ``symbol_mapping``.

    Idempotent — re-running on an already-masked body is a no-op.
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
        """Pass adds states and arrays inside body NSDFGs.

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

    def _ensure_symbols_in_nsdfg(self,
                                 nsdfg_node: NestedSDFG,
                                 names: List[str]) -> None:
        """Make every name in ``names`` visible inside ``nsdfg_node.sdfg``.

        Adds missing symbols to the inner SDFG's symbol table and to
        ``nsdfg_node.symbol_mapping`` so the parent-scope value flows
        in. Uses the parent SDFG's declared dtype when present;
        defaults to ``int64`` otherwise (matches the existing 1D
        ``GenerateIterationMask`` convention).

        :param nsdfg_node: NestedSDFG node receiving the symbols.
        :param names: Symbol names to ensure.
        """
        inner = nsdfg_node.sdfg
        parent = nsdfg_node.sdfg.parent_sdfg
        for name in names:
            if not name or not name.isidentifier():
                continue
            if name not in inner.symbols:
                dtype = parent.symbols.get(name, dace.int64) if parent is not None else dace.int64
                inner.add_symbol(name, dtype)
            if name not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[name] = name

    def _attach_mask(self,
                     nsdfg_node: NestedSDFG,
                     spec: TileDimSpec) -> bool:
        """Allocate the mask + the producer ``TileMaskGen`` inside the
        body NSDFG.

        :param nsdfg_node: Body NestedSDFG node.
        :param spec: Per-dim tile specification.
        :returns: ``True`` when a mask was added; ``False`` if a
            ``_tile_iter_mask`` was already present (idempotent).
        """
        inner = nsdfg_node.sdfg
        if TileNameScheme.ITER_MASK in inner.arrays:
            return False
        inner.add_array(
            TileNameScheme.ITER_MASK,
            list(spec.widths),
            dace.bool_,
            storage=dace.dtypes.StorageType.Register,
            transient=True,
        )
        names_to_thread: List[str] = list(spec.iter_vars)
        for ub in spec.global_ubs:
            for sym in symbolic.symlist(symbolic.pystr_to_symbolic(ub)).values():
                names_to_thread.append(str(sym))
        self._ensure_symbols_in_nsdfg(nsdfg_node, names_to_thread)
        prep = inner.add_state("_tile_iter_mask_init", is_start_block=True)
        mask_node = TileMaskGen(
            name="_tile_iter_mask_gen",
            widths=spec.widths,
            iter_vars=spec.iter_vars,
            global_ubs=spec.global_ubs,
        )
        prep.add_node(mask_node)
        out_access = prep.add_access(TileNameScheme.ITER_MASK)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        prep.add_edge(
            mask_node,
            "_o",
            out_access,
            None,
            dace.Memlet(f"{TileNameScheme.ITER_MASK}[{subset}]"),
        )
        return True

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Optional[Dict]) -> Optional[int]:
        """Walk every innermost map and attach the mask to its body.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: When the orchestrator ran
            :class:`MarkTileDims` earlier, the spec dict is fetched
            from here under the key ``"MarkTileDims"``; otherwise the
            spec is rebuilt from the map's last K params.
        :returns: Number of bodies that received a fresh mask, or
            ``None`` if none.
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
            nsdfg_node = get_single_nsdfg_inside_map(g, n)
            if nsdfg_node is None:
                raise NotImplementedError(
                    f"GenerateTileIterationMask: map {n.label!r} body is not a single "
                    f"NestedSDFG; run NestInnermostMapBodyIntoNSDFG first."
                )
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            if self._attach_mask(nsdfg_node, spec):
                attached += 1
        return attached or None
