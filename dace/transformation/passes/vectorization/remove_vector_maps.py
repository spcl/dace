# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that removes the residual scaffolding vectorization maps."""
from typing import Any, Dict
from dace import SDFG, properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils import *


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveVectorMaps(ppl.Pass):
    """Remove the ``vectorloop_`` scaffolding maps left after lane lowering."""

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        """Report the SDFG elements this pass may change."""
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Report whether this pass should run again after other passes."""
        return False

    def depends_on(self):
        """Report the passes this pass depends on."""
        return {}

    def _apply(self, sdfg: SDFG):
        """Remove every ``vectorloop_`` map in ``sdfg`` and nested SDFGs.

        :param sdfg: SDFG to transform in place.
        """
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, nodes.MapEntry) and n.map.label.startswith("vectorloop_"):
                remove_map(n, g)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        """Run the pass over ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes.
        """
        self._apply(sdfg)
        sdfg.validate()
