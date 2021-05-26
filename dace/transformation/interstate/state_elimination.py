# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State elimination transformations """

import copy
import networkx as nx
from typing import Dict, List, Set

from dace import data as dt, dtypes, registry, sdfg, symbolic
from dace.sdfg import nodes, SDFG, SDFGState, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.config import Config


@registry.autoregister_params(strict=True)
class EndStateElimination(transformation.Transformation):
    """
    End-state elimination removes a redundant state that has one incoming edge
    and no contents.
    """

    _end_state = sdfg.SDFGState()

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(EndStateElimination._end_state)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        state = graph.nodes()[candidate[EndStateElimination._end_state]]

        out_edges = graph.out_edges(state)
        in_edges = graph.in_edges(state)

        # If this is an end state, there are no outgoing edges
        if len(out_edges) != 0:
            return False

        # We only match end states with one source and no conditions
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]
        if not edge.data.is_unconditional():
            return False

        # Only empty states can be eliminated
        if state.number_of_nodes() > 0:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        state = graph.nodes()[candidate[EndStateElimination._end_state]]
        return state.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.subgraph[EndStateElimination._end_state]]
        sdfg.remove_node(state)


def _assignments_to_consider(sdfg, edge):
    assignments_to_consider = {}
    for var, assign in edge.data.assignments.items():
        as_symbolic = symbolic.pystr_to_symbolic(assign)
        # Assignments cannot access a data container
        if not symbolic.contains_sympy_functions(as_symbolic):  # via subscript
            # Assignments cannot use scalar values
            for sym in as_symbolic.free_symbols:
                if str(sym) in sdfg.arrays:
                    break
            else:
                assignments_to_consider[var] = assign
    return assignments_to_consider


@registry.autoregister_params(strict=True)
class StateAssignElimination(transformation.Transformation):
    """
    State assign elimination removes all assignments into the final state
    and subsumes the assigned value into its contents.
    """

    _end_state = sdfg.SDFGState()

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(StateAssignElimination._end_state)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        state = graph.nodes()[candidate[StateAssignElimination._end_state]]

        out_edges = graph.out_edges(state)
        in_edges = graph.in_edges(state)

        # We only match end states with one source and at least one assignment
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]

        assignments_to_consider = _assignments_to_consider(sdfg, edge)

        # No assignments to eliminate
        if len(assignments_to_consider) == 0:
            return False

        # If this is an end state, there are no other edges to consider
        if len(out_edges) == 0:
            return True

        # Otherwise, ensure the symbols are never set/used again in edges
        akeys = set(assignments_to_consider.keys())
        for e in sdfg.edges():
            if e is edge:
                continue
            if e.data.free_symbols & akeys:
                return False

        # If used in any state that is not the current one, fail
        for s in sdfg.nodes():
            if s is state:
                continue
            if s.free_symbols & akeys:
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        state = graph.nodes()[candidate[StateAssignElimination._end_state]]
        return state.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.subgraph[StateAssignElimination._end_state]]
        edge = sdfg.in_edges(state)[0]
        # Since inter-state assignments that use an assigned value leads to
        # undefined behavior (e.g., {m: n, n: m}), we can replace each
        # assignment separately.
        keys_to_remove = set()
        assignments_to_consider = _assignments_to_consider(sdfg, edge)
        for varname, assignment in assignments_to_consider.items():
            state.replace(varname, assignment)
            keys_to_remove.add(varname)

        for varname in keys_to_remove:
            # Remove assignments from edge
            del edge.data.assignments[varname]

            for e in sdfg.edges():
                if varname in e.data.free_symbols:
                    break
            else:
                # If removed assignment does not appear in any other edge,
                # remove symbol
                if varname in sdfg.symbols:
                    sdfg.remove_symbol(varname)


@registry.autoregister_params(singlestate=True)
class HoistState(transformation.Transformation):
    """ Move a state out of a nested SDFG """
    nsdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(HoistState.nsdfg)]

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate,
                       expr_index,
                       sdfg,
                       strict=False):
        nsdfg: nodes.NestedSDFG = graph.node(candidate[HoistState.nsdfg])

        # Must be a free nested SDFG
        if graph.entry_node(nsdfg) is not None:
            return False

        # Must have at least two states with a hoistable source state
        if nsdfg.sdfg.number_of_nodes() < 2:
            return False
        # Source state must not lead to more than one state or be conditional
        source_state = nsdfg.sdfg.start_state
        if nsdfg.sdfg.out_degree(source_state) != 1:
            return False
        nisedge = nsdfg.sdfg.out_edges(source_state)[0]
        if not nisedge.data.is_unconditional():
            return False

        # Keep all data descriptors to check for potential issues
        data_to_check: Set[str] = set()

        # Add data descriptors from interstate edge
        syms = nisedge.data.free_symbols
        for sym in syms:
            sym = str(sym)
            if sym in nsdfg.sdfg.arrays:
                if nsdfg.sdfg.arrays[sym].transient:  # Cannot keep transient
                    return False
                data_to_check.add(sym)

        # Add data descriptors from access nodes
        for dnode in source_state.data_nodes():
            data_to_check.add(dnode.data)
            desc = nsdfg.sdfg.arrays[dnode.data]
            # Cannot hoist state with transient
            if not isinstance(desc, dt.View) and desc.transient:
                return False

        # Nested SDFG surrounding edges must contain all of the array
        # TODO(later): Allow this case (with offsetting)
        outer_data_to_check: Set[str] = set()
        for e in graph.in_edges(nsdfg):
            if e.dst_conn in data_to_check:
                outer_data_to_check.add(e.data.data)
                if any(me != 0 for me in e.data.subset.min_element()):
                    return False
        for e in graph.out_edges(nsdfg):
            if e.src_conn in data_to_check:
                outer_data_to_check.add(e.data.data)
                if any(me != 0 for me in e.data.subset.min_element()):
                    return False

        # Data validity checks for descriptors in data_to_check:
        # 1. Path to nested SDFG must not go through descriptors,
        # 2. No other connected components can use descriptors.
        for dnode in graph.data_nodes():
            if dnode.data in outer_data_to_check:
                if nx.has_path(graph._nx, nsdfg, dnode):
                    # OK, has path from nsdfg to access node
                    continue
                if dnode in graph.predecessors(nsdfg):
                    # OK, a direct edge to nsdfg
                    continue
                if nx.has_path(graph._nx, dnode, nsdfg):
                    # NOT OK, some path goes through access node to SDFG,
                    # so state cannot safely be hoisted
                    return False
                # NOT OK, access node used independently from nsdfg
                return False

        return True

    def apply(self, sdfg: SDFG):
        nsdfg: nodes.NestedSDFG = self.nsdfg(sdfg)
        state = sdfg.node(self.state_id)

        new_state = sdfg.add_state_before(state)
        isedge = sdfg.edges_between(new_state, state)[0]

        # Find relevant symbol and data descriptor mapping
        mapping: Dict[str, str] = {}
        mapping.update({k: str(v) for k, v in nsdfg.symbol_mapping.items()})
        mapping.update({
            k: next(iter(state.in_edges_by_connector(nsdfg, k))).data.data
            for k in nsdfg.in_connectors
        })
        mapping.update({
            k: next(iter(state.out_edges_by_connector(nsdfg, k))).data.data
            for k in nsdfg.out_connectors
        })

        # Get internal state and interstate edge
        source_state = nsdfg.sdfg.start_state
        nisedge = nsdfg.sdfg.out_edges(source_state)[0]

        # Add state contents (nodes)
        new_state.add_nodes_from(source_state.nodes())

        # Replace data descriptors and symbols on state graph
        for node in source_state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in mapping:
                node.data = mapping[node.data]
        for edge in source_state.edges():
            edge.data.replace(mapping)
            if edge.data.data in mapping:
                edge.data.data = mapping[edge.data.data]

        # Add state contents (edges)
        for edge in source_state.edges():
            new_state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn,
                               edge.data)

        # Safe replacement of edge contents
        for k, v in mapping.items():
            nisedge.data.replace(k, '__dacesym_' + k, replace_keys=False)
        for k, v in mapping.items():
            nisedge.data.replace('__dacesym_' + k, v, replace_keys=False)

        # Add interstate edge
        for akey, aval in nisedge.data.assignments.items():
            # Map assignment to outer edge
            if akey not in sdfg.symbols and akey not in sdfg.arrays:
                newname = akey
            else:
                newname = nsdfg.label + '_' + akey

            isedge.data.assignments[newname] = aval

            # Add symbol to outer SDFG
            sdfg.add_symbol(newname, nsdfg.sdfg.symbols[akey])

            # Add symbol mapping to nested SDFG
            nsdfg.symbol_mapping[akey] = newname

        isedge.data.condition = nisedge.data.condition

        # Clean nested SDFG
        nsdfg.sdfg.remove_node(source_state)

        # Set new starting state
        nsdfg.sdfg.start_state = nsdfg.sdfg.node_id(nisedge.dst)
