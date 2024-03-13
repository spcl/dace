# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, Set, Type, Union

import sympy

from dace import dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.scope import ScopeTree
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.symbolic import SymExpr, symbol
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.map_dim_shuffle import MapDimShuffle
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap


@properties.make_properties
class MoveSequentialIntoParallel(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Parallelization'

    def __init__(self):
        self._non_analyzable_loops = set()
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {}

    def _check_permute_doacross(self, state: SDFGState, scope: ScopeTree) -> None:
        if (scope.entry is not None and isinstance(scope.entry, nodes.MapEntry) and
            scope.entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross):
            offset_dims = []
            for edge in state.scope_subgraph(scope.entry).edges():
                memlet: Memlet = edge.data
                if memlet.schedule == dtypes.MemletScheduleType.Doacross_Sink:
                    for i, offs in enumerate(memlet.doacross_dependency_offset):
                        if not isinstance(offs, (sympy.Symbol, symbol)) and isinstance(offs, (sympy.Basic, SymExpr)):
                            offset_dims.append(i)
                    break
            new_perm = []
            for i, r in enumerate(scope.entry.map.params):
                if i not in offset_dims:
                    new_perm.append(r)
            for i in offset_dims:
                new_perm.append(scope.entry.map.params[i])

            if new_perm != scope.entry.map.params:
                map_dim_shuffle = MapDimShuffle()
                map_dim_shuffle.setup_match(state.sdfg, state.sdfg.cfg_id, state.block_id,
                                            { MapDimShuffle.map_entry: state.node_id(scope.entry) }, 0)
                map_dim_shuffle.parameters = new_perm
                map_dim_shuffle.apply(state, state.sdfg)

        for child in scope.children:
            self._check_permute_doacross(state, child)

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Any:
        """
        TODO
        """
        results = {}

        for sdfg in top_sdfg.all_sdfgs_recursive():
            sdfg.apply_transformations_repeated([MoveLoopIntoMap])
            sdfg.simplify()
            sdfg.reset_cfg_list()
            
            for state in sdfg.states():
                root_scope = state.scope_tree()[None]
                for child in root_scope.children:
                    self._check_permute_doacross(state, child)

        return results

