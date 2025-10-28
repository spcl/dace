# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
from typing import Any, Dict, Optional, Set, Union
from dace import SDFG
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil
import dace.sdfg.construction_utils as cutil


@transformation.explicit_cf_compatible
class DuplicateAllMemletsSharingSingleMapOutConnector(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _apply(self, sdfg: SDFG):
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    # Check if inner most map
                    maps_inside = {
                        node
                        for node in state.all_nodes_between(node, state.exit_node(node))
                        if isinstance(node, dace.nodes.MapEntry)
                    }
                    if len(maps_inside) > 0:
                        continue

                    cutil.duplicate_memlets_sharing_single_in_connector(state, node)

                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply(node.sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        self._apply(sdfg)
        return None
