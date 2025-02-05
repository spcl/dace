# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Optional
from dace import properties
from dace import transformation
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, ReturnBlock, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.propagation import MemletPropagation, MemletPropResultT


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class BuildGlobalDataflowProxyGraph(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Analysis'

    with_transients = properties.Property(dtype=bool, default=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If access nodes were modified, reapply
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.CFG

    def depends_on(self):
        return { MemletPropagation }

    def _process_state(self, state: SDFGState, result: SDFGState):
        # TODO: Process interstate edge reads.
        sdfg = state.sdfg
        for dn in state.data_nodes():
            if '.' not in dn.data:
                if dn.data not in result.sdfg.arrays:
                    result.sdfg.add_datadesc(dn.data, sdfg.arrays[dn.data])
            else:
                root_data = dn.root_data
                if root_data not in result.sdfg.arrays:
                    result.sdfg.add_datadesc(root_data, sdfg.arrays[root_data])

            for ie in state.in_edges(dn):
                mtree = state.memlet_tree(ie)
                for leaf in mtree.leaves():
                    if isinstance(leaf.src, nd.AccessNode):
                        result.add_edge(leaf.src, leaf.src_conn, ie.dst, ie.dst_conn, ie.data)
                    else:
                        prev_iedges = state.in_edges(leaf.src)
                        if len(prev_iedges) > 0:
                            for prev_ie in prev_iedges:
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
                        else:
                            # Originates from a tasklet or something similar with no inputs - so a simple access node
                            # will do the trick for now.
                            result.add_node(dn)

    def _connect_block_backwards(self, block: ControlFlowBlock, result: SDFGState,
                                 start: Optional[ControlFlowBlock] = None,
                                 use_conditional: bool = False) -> None:
        pivot: ControlFlowBlock
        if start is None:
            pivot = block
        else:
            pivot = start
        cfg = pivot.parent_graph

        while pivot is not None:
            predecessors = cfg.predecessors(pivot)
            if len(predecessors) > 1:
                raise NotImplementedError('Building a proxy graph for global dataflow is only possible if the ' +
                                          'SDFG\'s control flow was fully lifted. A branch was detected in ' +
                                          cfg.label + ', so this is not the case for the given SDFG ' +
                                          cfg.sdfg.name)
            if len(predecessors) == 0:
                break
            pivot = predecessors[0]
            for read_data in block._certain_reads_moredata:
                if read_data in pivot._certain_writes_moredata:
                    read_entry = block._certain_reads_moredata[read_data]
                    write_entry = pivot._certain_writes_moredata[read_data]
                    if read_entry.memlet.subset.intersects(write_entry.memlet.subset):
                        if len(read_entry.accesses) > 1 or len(write_entry.accesses) > 1:
                            ...
                        for read_access in read_entry.accesses:
                            if not isinstance(read_access, tuple) or not isinstance(read_access[1], nd.AccessNode):
                                ...
                            for write_access in write_entry.accesses:
                                if not isinstance(write_access, tuple) or not isinstance(write_access[1], nd.AccessNode):
                                    ...
                                target_memlet = Memlet()
                                target_memlet.data = read_data
                                target_memlet.subset = write_access[0].data.subset
                                target_memlet.other_subset = read_access[0].data.subset
                                if use_conditional:
                                    # Hack: Add some WCR to make it appear dashed.
                                    target_memlet.wcr = 'lambda a, b: a + b'
                                src = result.node_by_guid(write_access[1].guid)
                                if src is None:
                                    result.add_node(write_access[1])
                                    src = write_access[1]
                                dst = result.node_by_guid(read_access[1].guid)
                                if dst is None:
                                    result.add_node(read_access[1])
                                    dst = read_access[1]
                                result.add_edge(src, None, dst, None, target_memlet)
                elif read_data in pivot._possible_writes_moredata:
                    read_entry = block._certain_reads_moredata[read_data]
                    write_entry = pivot._possible_writes_moredata[read_data]
                    if read_entry.memlet.subset.intersects(write_entry.memlet.subset):
                        if len(read_entry.accesses) > 1 or len(write_entry.accesses) > 1:
                            ...
                        for read_access in read_entry.accesses:
                            if not isinstance(read_access, tuple) or not isinstance(read_access[1], nd.AccessNode):
                                ...
                            for write_access in write_entry.accesses:
                                if not isinstance(write_access, tuple) or not isinstance(write_access[1], nd.AccessNode):
                                    ...
                                target_memlet = Memlet()
                                target_memlet.data = read_data
                                target_memlet.subset = write_access[0].data.subset
                                target_memlet.other_subset = read_access[0].data.subset
                                # Force conditional
                                target_memlet.wcr = 'lambda a, b: a + b'
                                src = result.node_by_guid(write_access[1].guid)
                                if src is None:
                                    result.add_node(write_access[1])
                                    src = write_access[1]
                                dst = result.node_by_guid(read_access[1].guid)
                                if dst is None:
                                    result.add_node(read_access[1])
                                    dst = read_access[1]
                                result.add_edge(src, None, dst, None, target_memlet)

    def _process_conditional(self, conditional: ConditionalBlock, result: SDFGState,
                             global_prop: Dict[int, MemletPropResultT]) -> None:
        for _, branch in conditional.branches:
            self._process_cfg(branch, result, global_prop)

            if branch._certain_reads_moredata:
                self._connect_block_backwards(branch, result, start=conditional, use_conditional=True)

    def _process_loop(self, loop: LoopRegion, result: SDFGState, global_prop: Dict[int, MemletPropResultT]) -> None:
        self._process_cfg(loop, result, global_prop)
        loop_carry_deps = loop_analysis.get_loop_carry_dependencies(loop, certain_only=True)
        if loop_carry_deps:
            for input_dep in loop_carry_deps.keys():
                output_dep = loop_carry_deps[input_dep]
                if len(input_dep.accesses) > 1 or len(output_dep.accesses) > 1:
                    ...
                for read_access in input_dep.accesses:
                    if not isinstance(read_access, tuple) or not isinstance(read_access[1], nd.AccessNode):
                        ...
                    for write_access in output_dep.accesses:
                        if not isinstance(write_access, tuple) or not isinstance(write_access[1], nd.AccessNode):
                            ...
                        target_memlet = Memlet()
                        target_memlet.data = input_dep.memlet.data
                        target_memlet.subset = write_access[0].data.subset
                        target_memlet.other_subset = read_access[0].data.subset
                        src = result.node_by_guid(write_access[1].guid)
                        if src is None:
                            result.add_node(write_access[1])
                            src = write_access[1]
                        dst = result.node_by_guid(read_access[1].guid)
                        if dst is None:
                            result.add_node(read_access[1])
                            dst = read_access[1]
                        result.add_edge(src, None, dst, None, target_memlet)

    def _process_cfg(self, cfg: ControlFlowRegion, result: SDFGState,
                     global_prop: Dict[int, MemletPropResultT]) -> None:
        for block in cfg_analysis.blockorder_topological_sort(cfg, recursive=False, ignore_nonstate_blocks=False):
            if isinstance(block, SDFGState):
                self._process_state(block, result)
            elif isinstance(block, LoopRegion):
                self._process_loop(block, result, global_prop)
            elif isinstance(block, ControlFlowRegion):
                self._process_cfg(block, result, global_prop)
            elif isinstance(block, ConditionalBlock):
                self._process_conditional(block, result, global_prop)
            else:
                ...

            if block._certain_reads_moredata:
                self._connect_block_backwards(block, result)

    def _process_sdfg(self, sdfg: SDFG, result: SDFGState, global_prop: Dict[int, MemletPropResultT]) -> None:
        self._process_cfg(sdfg, result, global_prop)

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict[str, Any]) -> SDFG:
        """
        TODO
        """
        has_return = False
        for blk in top_sdfg.all_control_flow_blocks():
            if isinstance(blk, ReturnBlock):
                has_return = True
        if has_return:
            # NOTE: This may be possible to add though.
            raise NotImplementedError('Building a global dataflow proxy graph is not implemented for graphs' +
                                      ' which may early-exit with returns.')

        propagation_results = pipeline_res[MemletPropagation.__name__]
        sdfg = top_sdfg
        result = SDFG(sdfg.name + '_GlobalDataflowProxyGraph')
        main_state = result.add_state('state', is_start_block=True)

        self._process_sdfg(sdfg, main_state, propagation_results)

        if not self.with_transients:
            self._filter_non_transients_out(main_state)

        return result

    def _filter_non_transients_out(self, result: SDFGState) -> None:
        for dn in result.data_nodes():
            if dn.desc(result.sdfg).transient:
                # First, remove self edges.
                for e in result.all_edges(dn):
                    if e.src is e.dst:
                        result.remove_edge(e)
                # Second, connect predecessors to successors, adjusting subsets and data for memlets accordingly.
                for ie in result.in_edges(dn):
                    for oe in result.out_edges(dn):
                        target_memlet = Memlet()
                        target_memlet.data = ie.src.data
                        if ie.data.data == dn.data and ie.data.other_subset is not None:
                            target_memlet.subset = ie.data.other_subset
                        else:
                            target_memlet.subset = ie.data.subset
                        if oe.data.data == dn.data and oe.data.other_subset is not None:
                            target_memlet.other_subset = oe.data.other_subset
                        else:
                            target_memlet.other_subset = oe.data.subset
                        if len(result.edges_between(ie.src, oe.dst)) > 0:
                            ...
                        else:
                            result.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, target_memlet)
                result.remove_node(dn)
