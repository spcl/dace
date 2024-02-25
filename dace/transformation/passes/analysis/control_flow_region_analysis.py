# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from dace import SDFG, SDFGState, properties
from dace.memlet import Memlet
from dace.sdfg import nodes, propagation
from dace.sdfg.analysis.writeset_underapproximation import (
    UnderapproximateWrites, UnderapproximateWritesDictT)
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.scope import ScopeTree
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import AccessRanges, ControlFlowBlockReachability

@properties.make_properties
class StateDataDependence(ppl.Pass):
    """
    Analyze the input dependencies and the underapproximated outputs of states.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets)

    def depends_on(self):
        return {UnderapproximateWrites, AccessRanges}

    def _gather_reads_scope(self, state: SDFGState, scope: ScopeTree,
                            writes: Dict[str, List[Tuple[Memlet, nodes.AccessNode]]],
                            not_covered_reads: Set[Memlet], scope_ranges: Dict[str, Range]):
        scope_nodes = state.scope_children()[scope.entry]
        data_nodes_in_scope: Set[nodes.AccessNode] = set([n for n in scope_nodes if isinstance(nodes.AccessNode)])
        if scope.entry is not None:
            # propagate
            pass

        for anode in data_nodes_in_scope:
            for oedge in state.out_edges(anode):
                if not oedge.data.is_empty():
                    root_edge = state.memlet_tree(oedge).root().edge
                    read_subset = root_edge.data.src_subset
                    covered = False
                    for [write, to] in writes[anode.data]:
                        if write.subset.covers_precise(read_subset) and nx.has_path(state.nx, to, anode):
                            covered = True
                            break
                    if not covered:
                        not_covered_reads.add(root_edge.data)

    def _state_get_deps(self, state: SDFGState,
                        underapproximated_writes: UnderapproximateWritesDictT) -> Tuple[Set[Memlet], Set[Memlet]]:
        # Collect underapproximated write memlets.
        writes: Dict[str, List[Tuple[Memlet, nodes.AccessNode]]] = defaultdict(lambda: [])
        for anode in state.data_nodes():
            for iedge in state.in_edges(anode):
                if not iedge.data.is_empty():
                    root_edge = state.memlet_tree(iedge).root().edge
                    if root_edge in underapproximated_writes['approximation']:
                        writes[anode.data].append([underapproximated_writes['approximation'][root_edge], anode])
                    else:
                        writes[anode.data].append([root_edge.data, anode])

        # Go over (overapproximated) reads and check if they are covered by writes.
        not_covered_reads: List[Tuple[MultiConnectorEdge[Memlet], Memlet]] = []
        for anode in state.data_nodes():
            for oedge in state.out_edges(anode):
                if not oedge.data.is_empty():
                    root_edge = state.memlet_tree(oedge).root().edge
                    read_subset = root_edge.data.src_subset
                    covered = False
                    for [write, to] in writes[anode.data]:
                        if write.subset.covers_precise(read_subset) and nx.has_path(state.nx, to, anode):
                            covered = True
                            break
                    if not covered:
                        not_covered_reads.append([root_edge, root_edge.data])
        # Make sure all reads are propagated if they happen inside maps. We do not need to do this for writes, because
        # it is already taken care of by the write underapproximation analysis pass.
        self._recursive_propagate_reads(state, state.scope_tree()[None], not_covered_reads)

        write_set = set()
        for data in writes:
            for memlet, _ in writes[data]:
                write_set.add(memlet)

        read_set = set()
        for reads in not_covered_reads:
            read_set.add(reads[1])

        return read_set, write_set

    def _recursive_propagate_reads(self, state: SDFGState, scope: ScopeTree,
                                   read_edges: Set[Tuple[MultiConnectorEdge[Memlet], Memlet]]):
        for child in scope.children:
            self._recursive_propagate_reads(state, child, read_edges)

        if scope.entry is not None:
            if isinstance(scope.entry, nodes.MapEntry):
                for read_tuple in read_edges:
                    read_edge, read_memlet = read_tuple
                    for param in scope.entry.map.params:
                        if param in read_memlet.free_symbols:
                            aligned_memlet = propagation.align_memlet(state, read_edge, True)
                            propagated_memlet = propagation.propagate_memlet(state, aligned_memlet, scope.entry, True)
                            read_tuple[1] = propagated_memlet

    def apply_pass(self, top_sdfg: SDFG,
                   pipeline_results: Dict[str, Any]) -> Dict[int, Dict[SDFGState, Tuple[Set[Memlet], Set[Memlet]]]]:
        """
        :return: For each SDFG, a dictionary mapping states to sets of their input and output memlets.
        """

        results = defaultdict(lambda: defaultdict(lambda: [set(), set()]))

        underapprox_writes_dict: Dict[int, Any] = pipeline_results[UnderapproximateWrites.__name__]
        for sdfg in top_sdfg.all_sdfgs_recursive():
            uapprox_writes = underapprox_writes_dict[sdfg.cfg_id]
            for state in sdfg.states():
                input_dependencies, output_dependencies = self._state_get_deps(state, uapprox_writes)
                results[sdfg.cfg_id][state] = [input_dependencies, output_dependencies]

        return results


@properties.make_properties
class CFGDataDependence(ppl.Pass):
    """
    Analyze the input dependencies and the underapproximated outputs of control flow graphs / regions.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return {StateDataDependence, ControlFlowBlockReachability}

    def _recursive_get_deps_region(self, cfg: ControlFlowRegion,
                                   results: Dict[int, Tuple[Dict[str, Set[Memlet]], Dict[str, Set[Memlet]]]],
                                   state_deps: Dict[int, Dict[SDFGState, Tuple[Set[Memlet], Set[Memlet]]]],
                                   cfg_reach: Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]]
                                   ) -> Tuple[Dict[str, Set[Memlet]], Dict[str, Set[Memlet]]]:
        # Collect all individual reads and writes happening inside the region.
        region_reads: Dict[str, List[Tuple[Memlet, ControlFlowBlock]]] = defaultdict(list)
        region_writes: Dict[str, List[Tuple[Memlet, ControlFlowBlock]]] = defaultdict(list)
        for node in cfg.nodes():
            if isinstance(node, SDFGState):
                for read in state_deps[node.sdfg.cfg_id][node][0]:
                    region_reads[read.data].append([read, node])
                for write in state_deps[node.sdfg.cfg_id][node][1]:
                    region_writes[write.data].append([write, node])
            elif isinstance(node, ControlFlowRegion):
                sub_reads, sub_writes = self._recursive_get_deps_region(node, results, state_deps, cfg_reach)
                for data in sub_reads:
                    for read in sub_reads[data]:
                        region_reads[data].append([read, node])
                for data in sub_writes:
                    for write in sub_writes[data]:
                        region_writes[data].append([write, node])

        # Through reachability analysis, check which writes cover which reads.
        # TODO: make sure this doesn't cover up reads if we have a cycle in the CFG.
        not_covered_reads: Dict[str, Set[Memlet]] = defaultdict(set)
        for data in region_reads:
            for read, read_block in region_reads[data]:
                covered = False
                for write, write_block in region_writes[data]:
                    if (write.subset.covers_precise(read.src_subset) and write_block is not read_block and
                        nx.has_path(cfg.nx, write_block, read_block)):
                        covered = True
                        break
                if not covered:
                    not_covered_reads[data].add(read)

        write_set: Dict[str, Set[Memlet]] = defaultdict(set)
        for data in region_writes:
            for memlet, _ in region_writes[data]:
                write_set[data].add(memlet)

        results[cfg.cfg_id] = [not_covered_reads, write_set]

        return not_covered_reads, write_set

    def apply_pass(self, top_sdfg: SDFG,
                   pipeline_res: Dict[str, Any]) -> Dict[int, Tuple[Dict[str, Set[Memlet]], Dict[str, Set[Memlet]]]]:
        """
        :return: For each SDFG, a dictionary mapping states to sets of their input and output memlets.
        """

        results = defaultdict(lambda: defaultdict(lambda: [defaultdict(set), defaultdict(set)]))

        state_deps_dict = pipeline_res[StateDataDependence.__name__]
        cfb_reachability_dict = pipeline_res[ControlFlowBlockReachability.__name__]
        for sdfg in top_sdfg.all_sdfgs_recursive():
            self._recursive_get_deps_region(sdfg, results, state_deps_dict, cfb_reachability_dict)

        return results
