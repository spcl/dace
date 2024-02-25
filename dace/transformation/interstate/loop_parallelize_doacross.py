# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, List, Optional, Set
from dace.memlet import Memlet
from dace.properties import make_properties
from dace.sdfg import utils as sdutils
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import transformation as xf
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis.loop_analysis import LoopCarryDependencyAnalysis

@make_properties
class LoopParallelizeDoacross(xf.MultiStateTransformation):

    loop = xf.PatternNode(LoopRegion)

    _loop_carry_dependencies_dict: Optional[Dict[int, Dict[LoopRegion, Dict[Memlet, Set[Memlet]]]]]

    def __init__(self, loop_carry_dependencies: Optional[Dict[int, Dict[LoopRegion,
                                                                        Dict[Memlet, Set[Memlet]]]]] = None) -> None:
        self._loop_carry_dependencies_dict = loop_carry_dependencies
        super().__init__()

    @classmethod
    def expressions(cls) -> List[SubgraphView]:
        return [sdutils.node_path_graph(cls.loop)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: sdutils.SDFG,
                       permissive: bool = False) -> bool:
        # The loop must have a single state as a body, and that single state may not contain more than one nested
        # map scope in the first level.
        loop = self.loop
        body = loop.nodes()
        if len(body) != 1:
            return False
        body_state = body[0]
        if not isinstance(body_state, SDFGState):
            return False
        scope_tree = body_state.scope_tree()

        parent_scope = scope_tree[None]
        if (len(parent_scope.children) > 1):
            return False
        elif (len(parent_scope.children) == 1):
            if len(parent_scope.children[0].children) > 0:
                return False
        return True

    def apply(self, graph: ControlFlowRegion | SDFGState, sdfg: xf.SDFG) -> None:
        if self._loop_carry_dependencies_dict is None:
            analysis_pass = Pipeline([LoopCarryDependencyAnalysis()])
            res = {}
            analysis_pass.apply_pass(sdfg, res)
            self._loop_carry_dependencies_dict = res[LoopCarryDependencyAnalysis.__name__]
        sdfg_id = self.loop.sdfg.cfg_id
        loop_deps = self._loop_carry_dependencies_dict[sdfg_id][self.loop]

        has_carry_dep = False
        for _, writes in loop_deps.items():
            if len(writes) > 0:
                has_carry_dep = True
        if not has_carry_dep:
            # No dependencies, this does not need to be parallelized via do-across.
            return

        print('match found that can be applied')

