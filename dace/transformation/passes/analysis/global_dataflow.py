# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Set, Union
from dace import properties
from dace import transformation
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.sdfg.graph import Edge
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.simplification.structs import LowerStructViews


@properties.make_properties
@transformation.explicit_cf_compatible
class BuildGlobalDataflowProxyGraph(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If access nodes were modified, reapply
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.CFG

    def _process_state(self, state: SDFGState, result: SDFGState):
        sdfg = state.sdfg
        for dn in state.data_nodes():
            if '.' not in dn.data and dn.data not in result.sdfg.arrays:
                result.sdfg.add_datadesc(dn.data, sdfg.arrays[dn.data])
            for ie in state.in_edges(dn):
                mtree = state.memlet_tree(ie)
                for leaf in mtree.leaves():
                    if isinstance(leaf.src, nd.AccessNode):
                        result.add_edge(leaf.src, leaf.src_conn, ie.dst, ie.dst_conn, ie.data)
                    else:
                        for prev_ie in state.in_edges(leaf.src):
                            prev_mtree = state.memlet_tree(prev_ie)
                            t_edge = prev_mtree.root().edge
                            if t_edge.data.is_empty():
                                continue
                            t_memlet = Memlet()
                            t_memlet.data = t_edge.data.data
                            t_memlet.subset = t_edge.data.subset
                            t_memlet.other_subset = (ie.data.subset if ie.data.other_subset is None
                                                     else ie.data.other_subset)
                            if not isinstance(t_edge.src, nd.AccessNode):
                                if t_edge.data.is_empty():
                                    successor = state.successors(t_edge.src)
                                    if successor:
                                        for succ in successor:
                                            result.add_edge(succ, None, ie.dst, ie.dst_conn, t_memlet)
                                    elif ie.dst not in result:
                                        result.add_node(ie.dst)
                                else:
                                    ...
                            else:
                                result.add_edge(t_edge.src, t_edge.src_conn, ie.dst, ie.dst_conn, t_memlet)

    def _process_cfg(self, cfg: ControlFlowRegion, result: SDFGState,
                     processed_elements: Set[Union[ControlFlowBlock, Edge[InterstateEdge]]]):
        start_block = cfg.start_block
        if start_block not in processed_elements:
            if isinstance(start_block, SDFGState):
                self._process_state(start_block, result)
            elif isinstance(start_block, ControlFlowRegion):
                # TODO: process meta
                self._process_cfg(start_block, result, processed_elements)
            elif isinstance(start_block, ConditionalBlock):
                # TODO: process meta
                for c, branch in start_block.branches:
                    self._process_cfg(branch, result, processed_elements)

        for isedge in cfg.edge_bfs(cfg.start_block):
            if isedge not in processed_elements:
                # TODO: process isedge
                pass
            block = isedge.dst
            if block not in processed_elements:
                processed_elements.add(block)
                if isinstance(block, SDFGState):
                    self._process_state(block, result)
                elif isinstance(block, ControlFlowRegion):
                    # TODO: process meta
                    self._process_cfg(block, result, processed_elements)
                elif isinstance(block, ConditionalBlock):
                    # TODO: process meta
                    for c, branch in block.branches:
                        self._process_cfg(branch, result, processed_elements)

    def _process_sdfg(self, sdfg: SDFG, result: SDFGState, processed_elements: Set[Union[ControlFlowBlock,
                                                                                         Edge[InterstateEdge]]]):
        self._process_cfg(sdfg, result, processed_elements)

    def _preprocess_sdfg(self, sdfg: SDFG):
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in sd.all_control_flow_regions():
                if isinstance(cfg, LoopRegion):
                    pass
                elif isinstance(cfg, ConditionalBlock):
                    pass

                for isedge in cfg.edges():
                    if not isedge.data.is_unconditional() or isedge.data.assignments:
                        #lift_state = cfg.add_state('edge_lift_state')
                        ...

    def apply_pass(self, top_sdfg: SDFG, _) -> Any:
        """
        TODO
        """
        sdfg_copy = top_sdfg
        #sdfg_copy = SDFG.from_json(top_sdfg.to_json())
        #ppl.Pipeline([LowerStructViews()]).apply_pass(sdfg_copy, {})
        #sdfg_copy.simplify()
        result = SDFG(sdfg_copy.name + '_GlobalDataflowProxyGraph')
        main_state = result.add_state('state', is_start_block=True)

        processed_elements = set()
        self._preprocess_sdfg(sdfg_copy)
        self._process_sdfg(sdfg_copy, main_state, processed_elements)

        return result
