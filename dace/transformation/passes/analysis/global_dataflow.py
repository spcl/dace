# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from dace import data as dt
from dace import dtypes, properties
from dace import transformation
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.sdfg import SDFG, memlets_in_ast
from dace.sdfg.state import (ConditionalBlock, ControlFlowBlock, ControlFlowRegion, GlobalDepDataRecord, LoopRegion,
                             ReturnBlock, SDFGState)
from dace.subsets import Range
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


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class BuildLayeredDataflowGraphs(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Analysis'

    with_transients = properties.Property(dtype=bool, default=True)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If access nodes were modified, reapply
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.CFG

    def _process_cfg(self, cfg: ControlFlowRegion) -> SDFG:
        proxy_graph = SDFG('DataflowProxy_' + cfg.label)
        graph = proxy_graph.add_state(is_start_block=True)
        sdfg = cfg.sdfg
        arrays_and_symbols = copy.deepcopy(sdfg.arrays)
        for k, v in sdfg.symbols.items():
            arrays_and_symbols[k] = v

        def add_container_if_not_exists(cont: str):
            if '.' in cont:
                root_container = cont.split('.')[0]
                if root_container not in proxy_graph.arrays:
                    proxy_graph.add_datadesc(root_container, sdfg.arrays[root_container])
            elif cont not in proxy_graph.arrays:
                if cont in sdfg.symbols:
                    if cont not in proxy_graph.symbols:
                        proxy_graph.add_symbol(cont, sdfg.symbols[cont])
                elif cont in sdfg.arrays:
                    proxy_graph.add_datadesc(cont, sdfg.arrays[cont])
                else:
                    if cont not in proxy_graph.symbols:
                        # TODO: needs a better solution for the datatype.
                        proxy_graph.add_symbol(cont, dtypes.int32)

        def process_read(read_container: str, repo: Dict[str, GlobalDepDataRecord],
                         read_nodes: Dict[str, nd.AccessNode], tasklet: nd.Tasklet, certain: bool = True):
            add_container_if_not_exists(read_container)
            if read_container not in read_nodes:
                read_nodes[read_container] = graph.add_access(read_container)

            read_memlet = Memlet()
            read_memlet.data = read_container
            if repo is not None:
                read_memlet.subset = repo[read_container].subset
                read_memlet.dynamic = repo[read_container].dynamic
                read_memlet.volume = repo[read_container].volume
            else:
                read_memlet.subset = Range.from_string('0')
                read_memlet.dynamic = False
                read_memlet.volume = 1
            if not certain:
                read_memlet.wcr = 'lambda a: a'
            graph.add_edge(read_nodes[read_container], None, tasklet, read_container, read_memlet)

        def process_write(write_container: str, repo: Dict[str, GlobalDepDataRecord],
                          write_nodes: Dict[str, nd.AccessNode], tasklet: nd.Tasklet, certain: bool = True):
            add_container_if_not_exists(write_container)
            if write_container not in write_nodes:
                write_nodes[write_container] = graph.add_access(write_container)

            write_memlet = Memlet()
            write_memlet.data = write_container
            if repo is not None:
                write_memlet.subset = repo[write_container].subset
                write_memlet.dynamic = repo[write_container].dynamic
                write_memlet.volume = repo[write_container].volume
            else:
                write_memlet.subset = Range.from_string('0')
                write_memlet.dynamic = False
                write_memlet.volume = 1
            if not certain:
                write_memlet.wcr = 'lambda a: a'
            graph.add_edge(tasklet, write_container, write_nodes[write_container], None, write_memlet)

        available_read_nodes = {}

        for block in cfg_analysis.blockorder_topological_sort(cfg, recursive=False, ignore_nonstate_blocks=False):
            all_inputs = set(block.certain_reads.keys()).union(set(block.possible_reads.keys()))
            all_outputs = set(block.certain_writes.keys()).union(set(block.possible_writes.keys()))
            symbol_reads = set()
            if isinstance(block, LoopRegion):
                ignore_sym = set([block.loop_variable])
                symbol_reads |= block.loop_condition.get_free_symbols(ignore_sym)
                if block.init_statement:
                    symbol_reads |= block.init_statement.get_free_symbols(ignore_sym)
                if block.update_statement:
                    symbol_reads |= block.update_statement.get_free_symbols(ignore_sym)
            elif isinstance(block, ConditionalBlock):
                for cond, _ in block.branches:
                    if cond is not None:
                        symbol_reads |= cond.get_free_symbols()
            all_inputs.update(symbol_reads)
            if len(all_inputs) > 0 or len(all_outputs) > 0:
                # Abuse the tasklet code for the GUID of the corresponding block.
                tasklet = graph.add_tasklet(block.label, all_inputs, all_outputs, block.guid, dtypes.Language.CPP)

                read_nodes = {}
                for k, v in available_read_nodes.items():
                    read_nodes[k] = v
                write_nodes = {}
                for read_container in block.certain_reads.keys():
                    process_read(read_container, block.certain_reads, read_nodes, tasklet)
                for read_container in block.possible_reads.keys():
                    if read_container in block.certain_reads:
                        continue
                    process_read(read_container, block.possible_reads, read_nodes, tasklet, False)
                for write_container in block.certain_writes.keys():
                    process_write(write_container, block.certain_writes, write_nodes, tasklet)
                for write_container in block.possible_writes.keys():
                    if write_container in block.certain_writes:
                        continue
                    process_write(write_container, block.possible_writes, write_nodes, tasklet, False)

                if isinstance(block, LoopRegion):
                    ignore_sym = set([block.loop_variable])
                    for sym in block.loop_condition.get_free_symbols(ignore_sym):
                        process_read(sym, None, read_nodes, tasklet)
                    if block.init_statement:
                        for sym in block.init_statement.get_free_symbols(ignore_sym):
                            process_read(sym, None, read_nodes, tasklet)
                    if block.update_statement:
                        certain = block.update_before_condition
                        for sym in block.update_statement.get_free_symbols(ignore_sym):
                            process_read(sym, None, read_nodes, tasklet, certain)
                elif isinstance(block, ConditionalBlock):
                    for i, (cond, _) in enumerate(block.branches):
                        if cond is not None:
                            certain = i == 0
                            for sym in cond.get_free_symbols():
                                process_read(sym, None, read_nodes, tasklet, certain)

                for k, v in read_nodes.items():
                    available_read_nodes[k] = v
                for k, v in write_nodes.items():
                    available_read_nodes[k] = v

                if isinstance(block, LoopRegion):
                    # Annotate loop carry dependencies.
                    carry_deps = loop_analysis.get_loop_carry_dependencies(block)
                    for dat in carry_deps:
                        for dep_read, dep_write in carry_deps[dat]:
                            target_memlet = Memlet(data=dat, subset=dep_write.subset, other_subset=dep_read.subset,
                                                   volume=dep_read.volume, dynamic=dep_read.dynamic)
                            graph.add_edge(write_nodes[dat], None, read_nodes[dat], None, target_memlet)

            edge_read_nodes = {}
            for k, v in available_read_nodes.items():
                edge_read_nodes[k] = v
            edge_write_nodes = {}
            for oedge in cfg.out_edges(block):
                uncond = oedge.data.is_unconditional()
                reads: List[Memlet] = []
                writes: List[str] = []
                if not uncond:
                    for read_memlet in memlets_in_ast(oedge.data.condition.code[0], arrays_and_symbols):
                        reads.append(read_memlet)
                for k, v in oedge.data.assignments.items():
                    for read_memlet in memlets_in_ast(ast.parse(v), arrays_and_symbols):
                        reads.append(read_memlet)
                    writes.append(k)
                if len(reads) > 0 or len(writes) > 0:
                    # Abuse the tasklet code for the GUID of the corresponding interstate edge.
                    tasklet = graph.add_tasklet('isedge_' + oedge.data.label, {}, {},
                                                oedge.data.guid, dtypes.Language.CPP)
                    for read in reads:
                        if read.data not in tasklet.in_connectors:
                            tasklet.add_in_connector(read.data)
                        process_read(read.data, None, edge_read_nodes, tasklet, uncond)
                    for write in writes:
                        if write not in tasklet.out_connectors:
                            tasklet.add_out_connector(write)
                        process_write(write, None, edge_write_nodes, tasklet, uncond)
            for k, v in edge_read_nodes.items():
                available_read_nodes[k] = v
            for k, v in edge_write_nodes.items():
                available_read_nodes[k] = v

        return proxy_graph

    def apply_pass(self, sdfg, pipeline_results) -> Dict[int, SDFG]:
        result = {}

        sdfg.reset_cfg_list()
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, ConditionalBlock):
                continue
            result[cfg.cfg_id] = self._process_cfg(cfg)

        return result
