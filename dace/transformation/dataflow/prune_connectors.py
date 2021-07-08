# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from os import stat
from typing import Any, AnyStr, Dict, Set, Tuple, Union
import networkx as nx
from networkx.exception import NodeNotFound
import re

import dace
from dace import dtypes, registry, SDFG, SDFGState, symbolic
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils
from dace.sdfg.analysis import cfg


@registry.autoregister_params(singlestate=True, strict=True)
class PruneConnectors(pm.Transformation):
    """ Removes unused connectors from nested SDFGs, as well as their memlets
        in the outer scope, replacing them with empty memlets if necessary.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def expressions():
        return [utils.node_path_graph(PruneConnectors.nsdfg)]

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:

        nsdfg = graph.node(candidate[PruneConnectors.nsdfg])

        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - read_set
        prune_out = nsdfg.out_connectors.keys() - write_set

        # Add WCR outputs to "do not prune" input list
        for e in graph.out_edges(nsdfg):
            if e.data.wcr is not None and e.src_conn in prune_in:
                if (graph.in_degree(
                        next(
                            iter(graph.in_edges_by_connector(
                                nsdfg, e.src_conn))).src) > 0):
                    prune_in.remove(e.src_conn)
        if len(prune_in) > 0 or len(prune_out) > 0:
            return True

        return False

    def apply(self, sdfg: SDFG) -> Union[Any, None]:

        state = sdfg.node(self.state_id)
        nsdfg = self.nsdfg(sdfg)

        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - read_set
        prune_out = nsdfg.out_connectors.keys() - write_set

        # Detect which nodes are used, so we can delete unused nodes after the
        # connectors have been pruned
        all_data_used = read_set | write_set

        # Add WCR outputs to "do not prune" input list
        for e in state.out_edges(nsdfg):
            if e.data.wcr is not None and e.src_conn in prune_in:
                if (state.in_degree(
                        next(
                            iter(state.in_edges_by_connector(
                                nsdfg, e.src_conn))).src) > 0):
                    prune_in.remove(e.src_conn)
        to_reconnect_inp = set()
        for conn in prune_in:
            for e in state.in_edges_by_connector(nsdfg, conn):
                for e2 in state.in_edges(e.src):
                    to_reconnect_inp.add(e2.src)
                state.remove_memlet_path(e, remove_orphans=True)

        to_reconnect_out = set()
        for conn in prune_out:
            for e in state.out_edges_by_connector(nsdfg, conn):
                for e2 in state.out_edges(e.dst):
                    to_reconnect_out.add(e2.dst)
                state.remove_memlet_path(e, remove_orphans=True)

        for conn in prune_in:
            if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
                # If the data is now unused, we can purge it from the SDFG
                nsdfg.sdfg.remove_data(conn)
        for conn in prune_out:
            if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
                # If the data is now unused, we can purge it from the SDFG
                nsdfg.sdfg.remove_data(conn)

        G = helpers.simplify_state(state)
        for src in to_reconnect_inp:
            if src not in state.nodes():
                continue  # Removed orphan access node
            has_path = False
            try:
                has_path = nx.has_path(G, src, nsdfg)
            except NodeNotFound:
                has_path = nx.has_path(state.nx, src, nsdfg)
            if not has_path:
                state.add_nedge(src, nsdfg, dace.Memlet())
        for dst in to_reconnect_out:
            if dst not in state.nodes():
                continue  # Removed orphan access node
            has_path = False
            try:
                has_path = nx.has_path(G, nsdfg, dst)
            except NodeNotFound:
                has_path = nx.has_path(state.nx, nsdfg, dst)
            if not has_path:
                state.add_nedge(nsdfg, dst, dace.Memlet())


@registry.autoregister_params(singlestate=True, strict=True)
class PruneSymbols(pm.Transformation):
    """ 
    Removes unused symbol mappings from nested SDFGs, as well as internal
    symbols if necessary.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def expressions():
        return [utils.node_path_graph(PruneSymbols.nsdfg)]

    @staticmethod
    def _candidates(nsdfg: nodes.NestedSDFG) -> Set[str]:
        candidates = set(nsdfg.symbol_mapping.keys())
        if len(candidates) == 0:
            return set()

        for desc in nsdfg.sdfg.arrays.values():
            candidates -= set(map(str, desc.free_symbols))

        ignore = set()
        for nstate in cfg.stateorder_topological_sort(nsdfg.sdfg):
            state_syms = nstate.free_symbols

            # Try to be conservative with C++ tasklets
            for node in nstate.nodes():
                if (isinstance(node, nodes.Tasklet)
                        and node.language is dtypes.Language.CPP):
                    for candidate in candidates:
                        if re.findall(r'\b%s\b' % re.escape(candidate),
                                      node.code.as_string):
                            state_syms.add(candidate)

            # Any symbol used in this state is considered used
            candidates -= (state_syms - ignore)
            if len(candidates) == 0:
                return set()

            # Any symbol that is set in all outgoing edges is ignored from
            # this point
            local_ignore = None
            for e in nsdfg.sdfg.out_edges(nstate):
                # Look for symbols in condition
                candidates -= (set(
                    map(str, symbolic.symbols_in_ast(e.data.condition.code[0])))
                               - ignore)

                for assign in e.data.assignments.values():
                    candidates -= (symbolic.free_symbols_and_functions(assign) -
                                   ignore)

                if local_ignore is None:
                    local_ignore = set(e.data.assignments.keys())
                else:
                    local_ignore &= e.data.assignments.keys()
            if local_ignore is not None:
                ignore |= local_ignore

        return candidates

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:

        nsdfg: nodes.NestedSDFG = graph.node(candidate[PruneSymbols.nsdfg])

        if len(PruneSymbols._candidates(nsdfg)) > 0:
            return True

        return False

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        nsdfg = self.nsdfg(sdfg)

        candidates = PruneSymbols._candidates(nsdfg)
        for candidate in candidates:
            del nsdfg.symbol_mapping[candidate]

            # If not used in SDFG, remove from symbols as well
            if helpers.is_symbol_unused(nsdfg.sdfg, candidate):
                nsdfg.sdfg.remove_symbol(candidate)
