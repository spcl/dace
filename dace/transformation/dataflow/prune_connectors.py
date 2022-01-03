# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from os import stat
from typing import Any, AnyStr, Dict, Set, Tuple, Union
import re

from dace import dtypes, registry, SDFG, SDFGState, symbolic, properties
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils
from dace.sdfg.analysis import cfg


@properties.make_properties
class PruneConnectors(pm.SingleStateTransformation, pm.DataflowCoarseningTransformation):
    """ Removes unused connectors from nested SDFGs, as well as their memlets
        in the outer scope, replacing them with empty memlets if necessary.

        Optionally: after pruning, removes the unused containers from parent SDFG.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    remove_unused_containers = properties.Property(dtype=bool,
                                                   default=False,
                                                   desc='If True, remove unused containers from parent SDFG.')

    @staticmethod
    def expressions():
        return [utils.node_path_graph(PruneConnectors.nsdfg)]

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:

        nsdfg = graph.node(candidate[PruneConnectors.nsdfg])

        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - read_set
        prune_out = nsdfg.out_connectors.keys() - write_set

        # Take into account symbol mappings
        strs = tuple(nsdfg.symbol_mapping.values())
        syms = tuple(symbolic.pystr_to_symbolic(s) for s in strs)
        symnames = tuple(s.name if hasattr(s, 'name') else '' for s in syms)
        for conn in list(prune_in):
            if conn in syms or conn in symnames:
                prune_in.remove(conn)

        # Add WCR outputs to "do not prune" input list
        for e in graph.out_edges(nsdfg):
            if e.data.wcr is not None and e.src_conn in prune_in:
                if (graph.in_degree(next(iter(graph.in_edges_by_connector(nsdfg, e.src_conn))).src) > 0):
                    prune_in.remove(e.src_conn)
        has_before = all(
            graph.in_degree(graph.memlet_path(e)[0].src) > 0 for e in graph.in_edges(nsdfg) if e.dst_conn in prune_in)
        has_after = all(
            graph.out_degree(graph.memlet_path(e)[-1].dst) > 0 for e in graph.out_edges(nsdfg)
            if e.src_conn in prune_out)
        if has_before and has_after:
            return False
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
                if (state.in_degree(next(iter(state.in_edges_by_connector(nsdfg, e.src_conn))).src) > 0):
                    prune_in.remove(e.src_conn)
        do_not_prune = set()
        for conn in prune_in:
            if any(
                    state.in_degree(state.memlet_path(e)[0].src) > 0 for e in state.in_edges(nsdfg)
                    if e.dst_conn == conn):
                do_not_prune.add(conn)
                continue
            for e in state.in_edges_by_connector(nsdfg, conn):
                state.remove_memlet_path(e, remove_orphans=True)

        for conn in prune_out:
            if any(
                    state.out_degree(state.memlet_path(e)[-1].dst) > 0 for e in state.out_edges(nsdfg)
                    if e.src_conn == conn):
                do_not_prune.add(conn)
                continue
            for e in state.out_edges_by_connector(nsdfg, conn):
                state.remove_memlet_path(e, remove_orphans=True)

        for conn in prune_in:
            if conn in nsdfg.sdfg.arrays and conn not in all_data_used and conn not in do_not_prune:
                # If the data is now unused, we can purge it from the SDFG
                nsdfg.sdfg.remove_data(conn)
        for conn in prune_out:
            if conn in nsdfg.sdfg.arrays and conn not in all_data_used and conn not in do_not_prune:
                # If the data is now unused, we can purge it from the SDFG
                nsdfg.sdfg.remove_data(conn)

        if self.remove_unused_containers:
            # Remove unused containers from parent SDFGs
            containers = list(sdfg.arrays.keys())
            for name in containers:
                s = nsdfg.sdfg
                while s.parent_sdfg:
                    s = s.parent_sdfg
                    try:
                        s.remove_data(name)
                    except ValueError:
                        break


class PruneSymbols(pm.SingleStateTransformation, pm.DataflowCoarseningTransformation):
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
                if (isinstance(node, nodes.Tasklet) and node.language is dtypes.Language.CPP):
                    for candidate in candidates:
                        if re.findall(r'\b%s\b' % re.escape(candidate), node.code.as_string):
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
                candidates -= (set(map(str, symbolic.symbols_in_ast(e.data.condition.code[0]))) - ignore)

                for assign in e.data.assignments.values():
                    candidates -= (symbolic.free_symbols_and_functions(assign) - ignore)

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
                       permissive: bool = False) -> bool:

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
