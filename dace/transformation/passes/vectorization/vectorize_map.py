# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vectorize one map and nothing else.

:class:`VectorizeMap` nests the scope of the map it is given into a nested SDFG of its own and runs
:class:`~dace.transformation.passes.vectorization.vectorize_multi_dim.VectorizeMultiDim` on that nested SDFG alone,
so every other map of the program keeps its scalar body. The vectorizer's contract applies to the map: it is in
canonical form, and its innermost maps are the ones tiled.
"""

from typing import Any

import dace
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_multi_dim import (
    VectorizeCPUMultiDim,
    VectorizeGPUMultiDim,
    VectorizeMultiDim,
)

__all__ = ["VectorizeMap"]


class VectorizeMap(ppl.Pass):
    """Vectorize the map ``entry`` of ``state`` by ``config`` (the device picks the CPU or the GPU vectorizer).

    :param config: The vectorizer configuration, as :class:`VectorizeMultiDim` takes it.
    :param state: The state that holds the map.
    :param entry: The map to vectorize; it must be top-level in ``state``.
    """

    CATEGORY: str = "Vectorization"

    def __init__(self, config: VectorizeConfig, state: SDFGState, entry: nodes.MapEntry) -> None:
        super().__init__()
        self.config = config
        self.state = state
        self.entry = entry

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def vectorizer(self) -> VectorizeMultiDim:
        """The device's vectorizer for :attr:`config`."""
        if self.config.device is dace.DeviceType.GPU:
            return VectorizeGPUMultiDim(self.config)
        return VectorizeCPUMultiDim(self.config)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]) -> nodes.NestedSDFG:
        """Nest the map and vectorize the nest.

        :returns: The nested SDFG node that now holds the vectorized map.
        :raises ValueError: The map is not top-level in its state, or not in ``sdfg``.
        """
        if self.state.sdfg is not sdfg:
            raise ValueError(f"VectorizeMap: the state of map {self.entry.map.label!r} is not in {sdfg.label!r}")
        if self.state.entry_node(self.entry) is not None:
            raise ValueError(f"VectorizeMap: map {self.entry.map.label!r} is inside another map; name the outer one")
        scope = self.state.scope_subgraph(self.entry, include_entry=True, include_exit=True)
        nest = nest_state_subgraph(sdfg, self.state, scope, name=f"{self.entry.map.label}_vectorized")
        self.vectorizer().apply_pass(nest.sdfg, {})
        return nest
