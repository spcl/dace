# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict
from dace import SDFG, InterstateEdge, properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.vectorization_utils import *


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveVectorMaps(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, sdfg: SDFG):
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, nodes.MapEntry) and n.map.label.startswith("vectorloop_"):
                remove_map(n, g)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        self._apply(sdfg)
        sdfg.validate()
