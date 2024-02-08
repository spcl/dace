# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from dace import SDFG, SDFGState, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.writeset_underapproximation import (
    UnderapproximateWrites, UnderapproximateWritesDictT)
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import ControlFlowBlockReachability


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
        return {UnderapproximateWrites}

    def _state_get_deps(self, state: SDFGState,
                        underapproximated_writes: UnderapproximateWritesDictT) -> Tuple[Set[Memlet], Set[Memlet]]:
        # Collect underapproximated write memlets.
        writes: Dict[str, List[Tuple[Memlet, nodes.AccessNode]]] = defaultdict(lambda: [])
        for anode in state.data_nodes():
            for iedge in state.in_edges(anode):
                if not iedge.data.is_empty():
                    root_edge = state.memlet_tree(iedge).root().edge
                    writes[anode.data].append([underapproximated_writes['approximation'][root_edge], anode])

        # Go over (overapproximated) reads and check if they are covered by writes.
        not_covered_reads = set()
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
                        not_covered_reads.add(root_edge.data)

        write_set = set()
        for data in writes:
            for memlet, _ in writes[data]:
                write_set.add(memlet)

        return not_covered_reads, write_set

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

    def _recursive_get_deps_region(self, cfg: ControlFlowRegion, results: Dict[int, Tuple[Set[Memlet], Set[Memlet]]],
                                   state_deps: Dict[int, Dict[SDFGState, Tuple[Set[Memlet], Set[Memlet]]]],
                                   cfg_reach: Dict[int, Dict[ControlFlowBlock, Set[ControlFlowBlock]]]
                                   ) -> Tuple[Set[Memlet], Set[Memlet]]:
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
                for read in sub_reads:
                    region_reads[read.data].append([read, node])
                for write in sub_writes:
                    region_writes[write.data].append([write, node])

        # Through reachability analysis, check which writes cover which reads.
        not_covered_reads: Set[Memlet] = set()
        for data in region_reads:
            for read, read_block in region_reads[data]:
                covered = False
                for write, write_block in region_writes[data]:
                    if write.subset.covers_precise(read.src_subset) and nx.has_path(cfg.nx, write_block, read_block):
                        covered = True
                        break
                if not covered:
                    not_covered_reads.add(read)

        write_set: Set[Memlet] = set()
        for data in region_writes:
            for memlet, _ in region_writes[data]:
                write_set.add(memlet)

        results[cfg.cfg_id] = [not_covered_reads, write_set]

        return not_covered_reads, write_set

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict[str, Any]) -> Dict[int, Tuple[Set[Memlet], Set[Memlet]]]:
        """
        :return: For each SDFG, a dictionary mapping states to sets of their input and output memlets.
        """

        results = defaultdict(lambda: [set(), set()])

        state_deps_dict = pipeline_res[StateDataDependence.__name__]
        cfb_reachability_dict = pipeline_res[ControlFlowBlockReachability.__name__]
        for sdfg in top_sdfg.all_sdfgs_recursive():
            self._recursive_get_deps_region(sdfg, results, state_deps_dict, cfb_reachability_dict)

        return results
