# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Set, Tuple
import re

from dace import dtypes, SDFG, SDFGState, symbolic, properties
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils
from dace.sdfg.analysis import cfg


@properties.make_properties
class PruneConnectors(pm.SingleStateTransformation):
    """
    Removes unused connectors from nested SDFGs, as well as their memlets in the outer scope.

    The transformation will not apply if this would remove all inputs and outputs.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [utils.node_path_graph(cls.nsdfg)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:

        prune_in, prune_out = self._get_prune_sets(graph)
        if not prune_in and not prune_out:
            return False

        # If the nested SDFG at global scope we must check if we can isolate it.
        if graph.scope_dict()[self.nsdfg] is None:
            if not helpers.isolate_nested_sdfg(state=graph, nsdfg_node=self.nsdfg, test_if_applicable=True):
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

        # From the input connectors (i.e. data container on the inside) remove
        #  all those that are not used for reading and from the output containers
        #  remove those that are not used fro reading.
        # NOTE: If a data container is used for reading and writing then only the
        #  output connector is retained, except the output is a WCR, then the input
        #  is also retained.
        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - read_set
        prune_out = nsdfg.out_connectors.keys() - write_set

        for e in state.out_edges(nsdfg):
            if e.data.wcr is not None and e.src_conn in prune_in:
                prune_in.remove(e.src_conn)

        # Same rule one level in. A WCR write IS an implicit read of its destination
        # (``a (CR: Sum)= x`` reads ``a``), but ``read_and_write_sets`` is pure dataflow and
        # reports the container write-only, so the accumulator lands in ``prune_in``. The loop
        # above only sees a WCR left on the OUTER boundary edge; a body that materialises the
        # read-modify-write on an INNER edge (tsvc ``vpv``: ``b[i] -> tasklet -> a[i] (CR: Sum)``
        # inside the map-body NestedSDFG) carries no outer WCR at all, and pruning ``a``'s input
        # silently drops the accumulated value -- ``a[i] += b[i]`` degrades to ``a[i] = b[i]``.
        if prune_in:
            for inner_state in nsdfg.sdfg.states():
                for e in inner_state.edges():
                    if e.data is not None and e.data.wcr is not None:
                        prune_in.discard(e.data.data)

        # A container can also be live SYMBOLICALLY rather than as dataflow: as an index inside a
        # memlet subset (tsvc_2_5 gather ``src[__tmp_r]``), inside a NODE (a tasklet whose code names
        # the container after an interstate assignment was folded into it -- ``__out = tmp1 * tmp2``
        # for an integer explicit-map elementwise kernel -- or a deeper NestedSDFG reading it), or in
        # an interstate-edge condition or assignment. ``read_and_write_sets`` records only the
        # container a memlet NAMES, never the symbols in its SUBSET or in node properties, so none of
        # these reaches it -- and ``SDFG.free_symbols`` does not catch it either, because while the
        # connector still exists the name is a DESCRIPTOR, not a symbol. It only turns into a free
        # symbol once ``apply`` removes the data, which is exactly the "Missing symbols on nested
        # SDFG" the validator then raises (nodes.py:809).
        if prune_in or prune_out:
            symbolic_uses: Set[str] = set()
            for inner_state in nsdfg.sdfg.states():
                for e in inner_state.edges():
                    if e.data is not None:
                        symbolic_uses |= {str(s) for s in e.data.free_symbols}
                for n in inner_state.nodes():
                    symbolic_uses |= {str(s) for s in n.free_symbols}
            for e in nsdfg.sdfg.all_interstate_edges():
                symbolic_uses |= {str(s) for s in e.data.free_symbols}
            prune_in -= symbolic_uses
            prune_out -= symbolic_uses

        return prune_in, prune_out

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg = self.nsdfg

        # Determine which connectors can be removed.
        prune_in, prune_out = self._get_prune_sets(state)

        # If the nested SDFG is at global scope, check if it can be isolated.
        if state.scope_dict()[nsdfg] is None:
            _, nsdfg_state, _ = helpers.isolate_nested_sdfg(state=state, nsdfg_node=nsdfg)
        else:
            nsdfg_state = state

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
                    candidates -= ((symbolic.free_symbols_and_functions(assign) | symbolic.arrays(assign)) - ignore)

                if local_ignore is None:
                    local_ignore = set(e.data.assignments.keys())
                else:
                    local_ignore &= e.data.assignments.keys()
            if local_ignore is not None:
                ignore |= local_ignore

        return candidates

    def _find_symbols_that_can_not_be_removed(self, sdfg: SDFG) -> Set[str]:
        """Find the set of symbols that can not be removed."""

        # The implementation of this function is based upon `dace.transformation.helpers.is_symbol_unused()`.
        #  However, instead of scanning the SDFG for every symbol, this function scans the SDFG
        #  once and then returns the collected set.
        # TODO: Investigate if this function can be replaced by a call to `used_symbols()`.
        #   See https://github.com/spcl/dace/pull/2080#discussion_r2226418881
        unremovable_symbols: Set[str] = set()

        for desc in sdfg.arrays.values():
            unremovable_symbols.update(map(str, desc.free_symbols))
        for state in sdfg.states():
            unremovable_symbols.update(state.free_symbols)
        for e in sdfg.all_interstate_edges():
            unremovable_symbols.update(e.data.free_symbols)
        return unremovable_symbols

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:

        nsdfg: nodes.NestedSDFG = self.nsdfg
        if len(PruneSymbols._candidates(nsdfg)) > 0:
            return True

        return False

    def apply(self, graph: SDFGState, sdfg: SDFG):
        nsdfg = self.nsdfg
        candidates = PruneSymbols._candidates(nsdfg)
        unremovable_symbols = self._find_symbols_that_can_not_be_removed(nsdfg.sdfg)

        for candidate in candidates:
            del nsdfg.symbol_mapping[candidate]

            # If not used in SDFG, remove from symbols as well
            if candidate not in unremovable_symbols:
                nsdfg.sdfg.remove_symbol(candidate)
