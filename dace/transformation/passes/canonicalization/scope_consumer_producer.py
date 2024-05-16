# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import networkx as nx
from dace import properties
from dace.memlet import Memlet
from dace.sdfg import nodes, propagation
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import LoopRegion, SDFGState
from dace.subsets import Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation import helpers as xfh


@properties.make_properties
class ScopeConsumerProducerCanonicalization(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Canonicalization'

    def __init__(self):
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets)

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {}

    def apply_pass(self, top_sdfg: SDFG,
                   _: Dict[str, Any]) -> Optional[Dict[int, Dict[SDFGState, Tuple[Set[str], Set[str]]]]]:
        """
        TODO
        """
        results: Dict[int, Dict[SDFGState, Tuple[Set[str], Set[str]]]] = {}

        did_something = False
        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            cfg_res: Dict[SDFGState, Tuple[Set[str], Set[str]]] = defaultdict(lambda: [set(), set()])
            for state in cfg.nodes():
                if not isinstance(state, SDFGState):
                    continue

                scopes = state.scope_children()
                for leaf in state.scope_leaves():
                    if leaf.entry is not None:
                        in_nodes: Dict[str, nodes.AccessNode] = dict()
                        out_nodes: Dict[str, nodes.AccessNode] = dict()
                        for ie in state.in_edges(leaf.entry):
                            if isinstance(ie.src, nodes.AccessNode):
                                in_nodes[ie.src.data] = ie.src
                        for ie in state.out_edges(leaf.exit):
                            if isinstance(ie.dst, nodes.AccessNode):
                                out_nodes[ie.dst.data] = ie.dst

                        scope_nodes = scopes[leaf.entry]
                        for node in scope_nodes:
                            if isinstance(node, nodes.AccessNode):
                                source = None
                                sink = None
                                if node.data in in_nodes:
                                    source = in_nodes[node.data]
                                if node.data in out_nodes:
                                    sink = out_nodes[node.data]
                                read_redirected, write_redirected = xfh.make_map_internal_read_external(state.sdfg,
                                                                                                        state,
                                                                                                        leaf.entry,
                                                                                                        node, source)

                                #if (node in state.nodes() and (state.out_degree(node) == 0 or
                                #    all([e.data.data is None for e in state.out_edges(node)]))):
                                #    a_write_redir, _ = xfh.make_map_internal_write_external(state.sdfg, state,
                                #                                                            leaf.exit, node, sink)
                                #    write_redirected = write_redirected or a_write_redir

                                if read_redirected:
                                    cfg_res[state][0].add(node.data)
                                    did_something = True
                                if write_redirected:
                                    cfg_res[state][1].add(node.data)
                                    did_something = True
            results[cfg.cfg_id] = cfg_res

        return results if did_something else None


@properties.make_properties
class ScopeIntermediateAccessesCanonicalization(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Canonicalization'

    def __init__(self):
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets)

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {}

    def apply_pass(self, top_sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[int, Dict[str, Set[str]]]]:
        """
        TODO
        """
        results: Dict[int, Dict[str, Set[str]]] = {}

        did_something = False
        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            sdfg = cfg if isinstance(cfg, SDFG) else cfg.sdfg
            sdfg_id = sdfg.cfg_id
            sdfg_res = defaultdict(lambda: defaultdict(set))
            for state in cfg.nodes():
                if not isinstance(state, SDFGState):
                    continue

                scopes = state.scope_children()
                for leaf in state.scope_leaves():
                    if leaf.entry is not None:
                        scope_writes: Dict[str, Set[MultiConnectorEdge[Memlet]]] = defaultdict(set)
                        elimination_candidates: List[Tuple[nodes.AccessNode, Subset]] = []

                        for iedge in state.in_edges(leaf.exit):
                            if iedge.data.data is not None:
                                scope_writes[iedge.data.data].add(iedge)

                        scope_nodes = scopes[leaf.entry]
                        for node in scope_nodes:
                            if isinstance(node, nodes.AccessNode):
                                iedges = state.in_edges(node)
                                cover_subset = None
                                if (len(iedges) == 1 and iedges[0].data is not None and
                                    iedges[0].data.data is not None and iedges[0].data.volume == 1):
                                    cover_subset = iedges[0].data.dst_subset or iedges[0].data.subset
                                for iedge in iedges:
                                    if iedge.data.data is not None:
                                        scope_writes[node.data].add(iedge)

                                all_covered = True
                                if cover_subset is None:
                                    all_covered = False
                                else:
                                    for oedge in state.out_edges(node):
                                        if (oedge.data is not None and oedge.data.data == node.data and
                                            not cover_subset.covers_precise(oedge.data.src_subset)):
                                            all_covered = False
                                            break
                                        elif oedge.data.data != node.data:
                                            # This indicates a copy to a different data container - meaning we cannot
                                            # safely eliminate this container no matter what.
                                            all_covered = False
                                            break
                                if all_covered:
                                    elimination_candidates.append([node, cover_subset])

                        for node, subset in elimination_candidates:
                            if node.data in scope_writes:
                                writes = scope_writes[node.data]
                                # If there are not at least two writes to that data container, there is no need for
                                # splitting.
                                if len(writes) > 1:
                                    for write in writes:
                                        write_subset = write.data.dst_subset or write.data.subset
                                        if (write.dst is not node and nx.has_path(state.nx, node, write.dst) and
                                            write.data.volume == 1 and write_subset.covers_precise(subset)):
                                            # We can split this write off into a thread-local variable.
                                            desc = node.desc(sdfg)
                                            tmp_name, _ = sdfg.add_temp_transient([1], dtype=desc.dtype)
                                            tmp_access = state.add_access(tmp_name)
                                            for iedge in state.in_edges(node):
                                                new_memlet = Memlet(data=tmp_name)
                                                state.add_edge(iedge.src, iedge.src_conn, tmp_access, None,
                                                               new_memlet)
                                                state.remove_edge(iedge)
                                            for oedge in state.out_edges(node):
                                                new_memlet = Memlet(data=tmp_name, other_subset=oedge.data.dst_subset)
                                                state.add_edge(tmp_access, None, oedge.dst, oedge.dst_conn,
                                                               new_memlet)
                                                state.remove_edge(oedge)
                                            state.remove_node(node)

                                            sdfg_res[state][node.data].add(tmp_name)
                                            did_something = True
                                            break
            results[sdfg_id] = sdfg_res

        return results if did_something else None
