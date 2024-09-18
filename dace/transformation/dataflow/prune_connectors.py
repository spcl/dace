# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Set, Tuple
import re

from dace import dtypes, SDFG, SDFGState, symbolic, properties, data as dt
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils
from dace.sdfg.analysis import cfg
from dace.sdfg.state import StateSubgraphView


@properties.make_properties
class PruneConnectors(pm.SingleStateTransformation):
    """
    Removes unused connectors from nested SDFGs, as well as their memlets in the outer scope.

    The transformation will not apply if this would remove all inputs and outputs.
    Optionally: after pruning, removes the unused containers from parent SDFG.

    Args:
        remove_unused_containers: If `True` the transformation will remove _all_ containers
            That are not used from the SDFG, the function will recursively go up.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    remove_unused_containers = properties.Property(dtype=bool,
                                                   default=False,
                                                   desc='If True, remove unused containers from parent SDFG (recursively).')

    @classmethod
    def expressions(cls):
        return [utils.node_path_graph(cls.nsdfg)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:

        prune_in, prune_out = self._get_prune_sets(graph)
        if not prune_in and not prune_out:
            return False

        return True

    def _get_prune_sets(self, state: SDFGState) -> Tuple[Set[str], Set[str]]:
        """Computes the set of the input and output connectors that can be removed.

        Returns:
            A tuple of two sets, the first set contains the name of all input
            connectors that can be removed and the second the name of all output
            connectors that can be removed.
        """
        nsdfg = self.nsdfg

        # Note the implementation of `read_and_write_sets()` filters array
        #  that fully written and read from the read set and only includes
        #  them in the write set. Thus we have to assume that every write
        #  is also a read to compensate for this.
        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - (read_set | write_set)
        prune_out = nsdfg.out_connectors.keys() - write_set

        # Take into account symbol mappings
        strs = tuple(nsdfg.symbol_mapping.keys())
        syms = set(symbolic.pystr_to_symbolic(s) for s in strs)
        symnames = set(s.name for s in syms if hasattr(s, 'name'))

        # If the connector is an array argument, it is handled by read/write sets,
        #  now we have to handle symbols. In the simplest one we should be able
        #  to use `nsdfg.sdfg.used_symbols(False)` however, for some reason this
        #  does not work. Thus we are using `nsdfg.sdfg.symbols` as an over
        #  approximation.
        used_symbols = nsdfg.sdfg.used_symbols(False).union(nsdfg.sdfg.symbols.keys())

        for conn in prune_in.copy():
            if conn in syms or conn in symnames or conn in used_symbols:
                prune_in.remove(conn)

        return prune_in, prune_out

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg = self.nsdfg

        # Determine which connectors can be removed.
        prune_in, prune_out = self._get_prune_sets(state)

        # Fission subgraph around nsdfg into its own state to avoid data races
        nsdfg_state = helpers.state_fission_after(state, nsdfg)

        # Detect which nodes are used, so we can delete unused nodes after the
        # connectors have been pruned
        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        all_data_used = read_set | write_set

        for conn in prune_in:
            for e in nsdfg_state.in_edges_by_connector(nsdfg, conn):
                nsdfg_state.remove_memlet_path(e, remove_orphans=True)

        for conn in prune_out:
            for e in nsdfg_state.out_edges_by_connector(nsdfg, conn):
                nsdfg_state.remove_memlet_path(e, remove_orphans=True)

        for conn in prune_in:
            if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
                # If the data is now unused, we can purge it from the SDFG
                nsdfg.sdfg.remove_data(conn)
        for conn in prune_out:
            if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
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


class PruneSymbols(pm.SingleStateTransformation):
    """ 
    Removes unused symbol mappings from nested SDFGs, as well as internal
    symbols if necessary.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [utils.node_path_graph(cls.nsdfg)]

    @staticmethod
    def _candidates(nsdfg: nodes.NestedSDFG) -> Set[str]:
        candidates = set(nsdfg.symbol_mapping.keys())
        if len(candidates) == 0:
            return set()

        for desc in nsdfg.sdfg.arrays.values():
            candidates -= set(map(str, desc.free_symbols))

        ignore = set()
        for nstate in cfg.blockorder_topological_sort(nsdfg.sdfg):
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
            for e in nstate.parent_graph.out_edges(nstate):
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

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:

        nsdfg: nodes.NestedSDFG = self.nsdfg

        if len(PruneSymbols._candidates(nsdfg)) > 0:
            return True

        return False

    def apply(self, graph: SDFGState, sdfg: SDFG):
        nsdfg = self.nsdfg

        candidates = PruneSymbols._candidates(nsdfg)
        for candidate in candidates:
            del nsdfg.symbol_mapping[candidate]

            # If not used in SDFG, remove from symbols as well
            if helpers.is_symbol_unused(nsdfg.sdfg, candidate):
                nsdfg.sdfg.remove_symbol(candidate)


class PruneUnusedOutputs(pm.SingleStateTransformation):
    """ 
    Removes unused symbol mappings from nested SDFGs, as well as internal
    symbols if necessary.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [utils.node_path_graph(cls.nsdfg)]

    @classmethod
    def _candidates(cls, nsdfg: nodes.NestedSDFG) -> Tuple[Set[str], Set[Tuple[SDFGState, nodes.AccessNode]]]:
        # Start with all non-transient arrays
        candidates = set(conn for conn in nsdfg.out_connectors.keys())
        candidate_nodes: Set[Tuple[SDFGState, nodes.AccessNode]] = set()

        # Remove candidates that are used more than once in the outer SDFG
        state = nsdfg.sdfg.parent
        sdfg = nsdfg.sdfg.parent_sdfg
        for e in state.out_edges(nsdfg):
            if e.data.is_empty():
                continue
            outer_desc = sdfg.arrays[e.data.data]
            if isinstance(outer_desc, dt.View):
                candidates.remove(e.src_conn)
                continue
            if not outer_desc.transient:
                candidates.remove(e.src_conn)
                continue
            if not isinstance(state.memlet_path(e)[-1].dst, nodes.AccessNode):
                candidates.remove(e.src_conn)
                continue

            all_access_nodes = [(s, n) for s in sdfg.nodes() for n in s.data_nodes() if n.data == e.data.data]
            if len(all_access_nodes) > 1:
                candidates.remove(e.src_conn)
                continue
            if all_access_nodes[0][0].out_degree(all_access_nodes[0][1]) > 0:
                candidates.remove(e.src_conn)
                continue

        if not candidates:
            return set(), set()

        # Remove candidates that are used in the nested SDFG
        for nstate in nsdfg.sdfg.states():
            for node in nstate.data_nodes():
                if node.data in candidates:
                    # If used in nested SDFG
                    if nstate.out_degree(node) > 0:
                        candidates.remove(node.data)
                        continue
                    # If a result of a code node
                    if any(not isinstance(nstate.memlet_path(e)[0].src, nodes.AccessNode)
                           for e in nstate.in_edges(node)):
                        candidates.remove(node.data)
                        continue

                    # Add node for later use
                    candidate_nodes.add((nstate, node))

        # Any array that is used in interstate edges is removed
        for e in nsdfg.sdfg.all_interstate_edges():
            candidates -= (set(map(str, symbolic.symbols_in_ast(e.data.condition.code[0]))))
            for assign in e.data.assignments.values():
                candidates -= (symbolic.free_symbols_and_functions(assign))

        candidate_nodes = {n for n in candidate_nodes if n[1].data in candidates}

        return candidates, candidate_nodes

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        nsdfg: nodes.NestedSDFG = self.nsdfg
        candidates, _ = self._candidates(nsdfg)
        if len(candidates) > 0:
            return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg = self.nsdfg

        candidates, candidate_nodes = self._candidates(nsdfg)
        for outer_edge in state.out_edges(nsdfg):
            if outer_edge.src_conn in candidates:
                state.remove_memlet_path(outer_edge)
                sdfg.remove_data(outer_edge.data.data, validate=False)
        for nstate, node in candidate_nodes:
            for ie in nstate.in_edges(node):
                nstate.remove_memlet_path(ie)
        for cand in candidates:
            nsdfg.sdfg.remove_data(cand, validate=False)
