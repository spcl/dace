# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State elimination transformations """

import networkx as nx
from typing import Dict, List, Set

from dace import data as dt, dtypes, registry, sdfg, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes, SDFG, SDFGState, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.sdfg.analysis import cfg


class EndStateElimination(transformation.MultiStateTransformation, transformation.SimplifyPass):
    """
    End-state elimination removes a redundant state that has one incoming edge
    and no contents.
    """

    end_state = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.end_state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        state = self.end_state

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

    def apply(self, _, sdfg):
        state = self.end_state
        # Handle orphan symbols (due to the deletion the incoming edge)
        edge = sdfg.in_edges(state)[0]
        sym_assign = edge.data.assignments.keys()
        sdfg.remove_node(state)
        # Remove orphan symbols
        for sym in sym_assign:
            if sym in sdfg.free_symbols:
                sdfg.remove_symbol(sym)


class StartStateElimination(transformation.MultiStateTransformation):
    """
    Start-state elimination removes a redundant state that has one outgoing edge
    and no contents. This transformation applies only to nested SDFGs.
    """

    start_state = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.start_state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        state = self.start_state

        # The transformation applies only to nested SDFGs
        if not graph.parent:
            return False

        # Only empty states can be eliminated
        if state.number_of_nodes() > 0:
            return False

        out_edges = graph.out_edges(state)
        in_edges = graph.in_edges(state)

        # If this is a start state, there are no incoming edges
        if len(in_edges) != 0:
            return False

        # We only match start states with one sink and no conditions
        if len(out_edges) != 1:
            return False
        edge = out_edges[0]
        if not edge.data.is_unconditional():
            return False
        # Assignments that make descriptors into symbols cannot be eliminated
        for assign in edge.data.assignments.values():
            if graph.arrays.keys() & symbolic.free_symbols_and_functions(assign):
                return False

        return True

    def apply(self, _, sdfg):
        state = self.start_state
        # Move assignments to the nested SDFG node's symbol mappings
        node = sdfg.parent_nsdfg_node
        edge = sdfg.out_edges(state)[0]
        for k, v in edge.data.assignments.items():
            node.symbol_mapping[k] = v
        sdfg.remove_node(state)


def _assignments_to_consider(sdfg, edge, is_constant=False):
    assignments_to_consider = {}
    for var, assign in edge.data.assignments.items():
        as_symbolic = symbolic.pystr_to_symbolic(assign)
        if isinstance(as_symbolic, bool):
            as_symbolic = symbolic.pystr_to_symbolic(as_symbolic)
        if is_constant and as_symbolic.free_symbols:
            continue
        # Assignments cannot access a data container
        if not symbolic.contains_sympy_functions(as_symbolic):  # via subscript
            # Assignments cannot use scalar values
            for sym in as_symbolic.free_symbols:
                if str(sym) in sdfg.arrays:
                    break
            else:
                assignments_to_consider[var] = assign
    return assignments_to_consider


class StateAssignElimination(transformation.MultiStateTransformation, transformation.SimplifyPass):
    """
    State assign elimination removes all assignments into the final state
    and subsumes the assigned value into its contents.
    """

    end_state = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.end_state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        state = self.end_state

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

    def apply(self, _, sdfg):
        state = self.end_state
        edge = sdfg.in_edges(state)[0]
        # Since inter-state assignments that use an assigned value leads to
        # undefined behavior (e.g., {m: n, n: m}), we can replace each
        # assignment separately.
        keys_to_remove = set()
        assignments_to_consider = _assignments_to_consider(sdfg, edge)
        for varname, assignment in assignments_to_consider.items():
            state.replace(varname, assignment)
            keys_to_remove.add(varname)

        repl_dict = {}

        for varname in keys_to_remove:
            # Remove assignments from edge
            del edge.data.assignments[varname]

            for e in sdfg.edges():
                if varname in e.data.free_symbols:
                    break
            else:
                # If removed assignment does not appear in any other edge,
                # replace and remove symbol
                if varname in sdfg.symbols:
                    sdfg.remove_symbol(varname)
                # if assignments_to_consider[varname] in sdfg.symbols:
                if varname in sdfg.free_symbols:
                    repl_dict[varname] = assignments_to_consider[varname]

        def _str_repl(s, d):
            for k, v in d.items():
                s.replace(str(k), str(v))

        if repl_dict:
            symbolic.safe_replace(repl_dict, lambda m: _str_repl(sdfg, m))


class ConstantPropagation(transformation.MultiStateTransformation):
    """
    Removes constant assignments in interstate edges and replaces them in successor states.
    """

    end_state = transformation.PatternNode(sdfg.SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.end_state)]

    def can_be_applied(self, graph, expr_index, sdfg: SDFG, permissive=False):
        state = self.end_state

        out_edges = graph.out_edges(state)
        in_edges = graph.in_edges(state)

        # We only match states with one source and at least one assignment
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]
        assignments_to_consider = _assignments_to_consider(sdfg, edge, True)

        # No assignments to eliminate
        if len(assignments_to_consider) == 0:
            return False

        # If this is an end state, there are no other edges to consider
        if len(out_edges) == 0:
            return True

        # Otherwise, ensure the symbols are never set/used again in edges
        akeys = set(assignments_to_consider.keys())
        for e in sdfg.bfs_edges(state):
            if e is edge:
                continue
            if e.data.assignments.keys() & akeys:
                return False

        return True

    def apply(self, _, sdfg: SDFG):
        state = self.end_state
        edge = sdfg.in_edges(state)[0]
        # Since inter-state assignments that use an assigned value leads to
        # undefined behavior (e.g., {m: n, n: m}), we can replace each
        # assignment separately.
        assignments_to_consider = _assignments_to_consider(sdfg, edge, True)

        def _str_repl(s, d, **kwargs):
            for k, v in d.items():
                s.replace(str(k), str(v), **kwargs)

        # Replace in state, and all successors
        symbolic.safe_replace(assignments_to_consider, lambda m: _str_repl(state, m))
        visited = {edge}
        for isedge in sdfg.bfs_edges(state):
            if isedge not in visited:
                symbolic.safe_replace(assignments_to_consider, lambda m: _str_repl(isedge.data, m, replace_keys=False))
                visited.add(isedge)
            if isedge.dst not in visited:
                symbolic.safe_replace(assignments_to_consider, lambda m: _str_repl(isedge.dst, m))
                visited.add(isedge.dst)

        repl_dict = {}

        for varname in assignments_to_consider.keys():
            # Remove assignments from edge
            del edge.data.assignments[varname]

            for e in sdfg.edges():
                if varname in e.data.free_symbols:
                    break
            else:
                # If removed assignment does not appear in any other edge,
                # replace and remove symbol
                if varname in sdfg.symbols:
                    sdfg.remove_symbol(varname)
                # if assignments_to_consider[varname] in sdfg.symbols:
                if varname in sdfg.free_symbols:
                    repl_dict[varname] = assignments_to_consider[varname]

        if repl_dict:
            symbolic.safe_replace(repl_dict, lambda m: _str_repl(sdfg, m))


def _alias_assignments(sdfg, edge):
    assignments_to_consider = {}
    for var, assign in edge.assignments.items():
        if assign in sdfg.symbols or (assign in sdfg.arrays and isinstance(sdfg.arrays[assign], dt.Scalar)):
            assignments_to_consider[var] = assign
    return assignments_to_consider


class SymbolAliasPromotion(transformation.MultiStateTransformation, transformation.SimplifyPass):
    """
    SymbolAliasPromotion moves inter-state assignments that create symbolic
    aliases to the previous inter-state edge according to the topological order.
    The purpose of this transformation is to iteratively move symbolic aliases
    together, so that true duplicates can be easily removed.
    """

    first_state = transformation.PatternNode(SDFGState)
    second_state = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_state, cls.second_state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        fstate = self.first_state
        sstate = self.second_state

        # For the topological order to be unambiguous:
        # 1. First state must have unique input edge.
        in_fedges = graph.in_edges(fstate)
        if len(in_fedges) != 1:
            return False
        in_edge = in_fedges[0].data
        # 2. There must be a unique edge from the first state to the second
        # one and no edge from the second state to the first one.
        edges = graph.edges_between(fstate, sstate)
        if len(edges) != 1:
            return False
        if len(graph.edges_between(sstate, fstate)) > 1:
            return False

        edge = edges[0].data
        in_edge = in_fedges[0].data

        to_consider = _alias_assignments(sdfg, edge)

        to_not_consider = set()
        for k, v in to_consider.items():
            # Remove symbols that are taking part in the edge's condition
            condsyms = [str(s) for s in edge.condition_sympy().free_symbols]
            if k in condsyms:
                to_not_consider.add(k)
            # Remove symbols that are set in the in_edge
            # with a different assignment
            if k in in_edge.assignments and in_edge.assignments[k] != v:
                to_not_consider.add(k)
            # Remove symbols whose assignment (RHS) is a symbol
            # and is set in the in_edge.
            if v in sdfg.symbols and v in in_edge.assignments:
                to_not_consider.add(k)
            # Remove symbols whose assignment (RHS) is a scalar
            # and is set in the first state.
            if v in sdfg.arrays and isinstance(sdfg.arrays[v], dt.Scalar):
                if any(isinstance(n, nodes.AccessNode) and n.data == v for n in fstate.nodes()):
                    to_not_consider.add(k)

        for k in to_not_consider:
            del to_consider[k]

        # No assignments to promote
        if len(to_consider) == 0:
            return False

        return True

    def apply(self, _, sdfg):
        fstate = self.first_state
        sstate = self.second_state

        edge = sdfg.edges_between(fstate, sstate)[0].data
        in_edge = sdfg.in_edges(fstate)[0].data

        to_consider = _alias_assignments(sdfg, edge)

        to_not_consider = set()
        for k, v in to_consider.items():
            # Remove symbols that are taking part in the edge's condition
            condsyms = [str(s) for s in edge.condition_sympy().free_symbols]
            if k in condsyms:
                to_not_consider.add(k)
            # Remove symbols that are set in the in_edge
            # with a different assignment
            if k in in_edge.assignments and in_edge.assignments[k] != v:
                to_not_consider.add(k)
            # Remove symbols whose assignment (RHS) is a symbol
            # and is set in the in_edge.
            if v in sdfg.symbols and v in in_edge.assignments:
                to_not_consider.add(k)
            # Remove symbols whose assignment (RHS) is a scalar
            # and is set in the first state.
            if v in sdfg.arrays and isinstance(sdfg.arrays[v], dt.Scalar):
                if any(isinstance(n, nodes.AccessNode) and n.data == v for n in fstate.nodes()):
                    to_not_consider.add(k)

        for k in to_not_consider:
            del to_consider[k]

        for k, v in to_consider.items():
            del edge.assignments[k]
            in_edge.assignments[k] = v


class HoistState(transformation.SingleStateTransformation):
    """ Move a state out of a nested SDFG """
    nsdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nsdfg)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg, permissive=False):
        nsdfg = self.nsdfg

        # Must be a free nested SDFG
        if graph.entry_node(nsdfg) is not None:
            return False

        # Must have two states with an empty source state.
        # Otherwise structured control flow (loop init states, for example)
        # may be broken.
        if not permissive:
            if nsdfg.sdfg.number_of_nodes() != 2:
                return False
            if nsdfg.sdfg.start_state.number_of_nodes() != 0:
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

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg: nodes.NestedSDFG = self.nsdfg

        new_state = sdfg.add_state_before(state)
        isedge = sdfg.edges_between(new_state, state)[0]

        # Find relevant symbol and data descriptor mapping
        mapping: Dict[str, str] = {}
        mapping.update({k: str(v) for k, v in nsdfg.symbol_mapping.items()})
        mapping.update({k: next(iter(state.in_edges_by_connector(nsdfg, k))).data.data for k in nsdfg.in_connectors})
        mapping.update({k: next(iter(state.out_edges_by_connector(nsdfg, k))).data.data for k in nsdfg.out_connectors})

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
            new_state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        # Safe replacement of edge contents
        def replfunc(m):
            for k, v in mapping.items():
                nisedge.data.replace(k, v, replace_keys=False)

        symbolic.safe_replace(mapping, replfunc)

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


class DeadStateElimination(transformation.MultiStateTransformation):
    """
    Dead state elimination removes an unreachable state and all of its dominated
    states.
    """

    end_state = transformation.PatternNode(sdfg.SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.end_state)]

    def can_be_applied(self, graph: SDFG, expr_index, sdfg: SDFG, permissive=False):
        state: SDFGState = self.end_state
        in_edges = graph.in_edges(state)

        # We only match end states with one source and at least one assignment
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]

        if edge.data.assignments:
            return False
        if edge.data.is_unconditional():
            return False

        # Evaluate condition
        scond = edge.data.condition_sympy()
        if scond == False:
            return True

        return False

    def apply(self, _, sdfg: SDFG):
        # Remove state and all dominated states
        state = self.end_state

        domset = cfg.all_dominators(sdfg)
        states_to_remove = {k for k, v in domset.items() if state in v}
        states_to_remove.add(state)
        sdfg.remove_nodes_from(states_to_remove)


class TrueConditionElimination(transformation.MultiStateTransformation, transformation.SimplifyPass):
    """
    If a state transition condition is always true, removes condition from edge.
    """

    state_a = transformation.PatternNode(sdfg.SDFGState)
    state_b = transformation.PatternNode(sdfg.SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.state_a, cls.state_b)]

    def can_be_applied(self, graph: SDFG, expr_index, sdfg: SDFG, permissive=False):
        a: SDFGState = self.state_a
        b: SDFGState = self.state_b
        # Directed graph has only one edge between two nodes
        edge = graph.edges_between(a, b)[0]

        if edge.data.is_unconditional():
            return False

        # Evaluate condition
        scond = edge.data.condition_sympy()
        if scond == True:
            return True

        return False

    def apply(self, _, sdfg: SDFG):
        a: SDFGState = self.state_a
        b: SDFGState = self.state_b
        edge = sdfg.edges_between(a, b)[0]
        edge.data.condition = CodeBlock("1")


class FalseConditionElimination(transformation.MultiStateTransformation):
    """
    If a state transition condition is always true, removes condition from edge.
    """

    state_a = transformation.PatternNode(sdfg.SDFGState)
    state_b = transformation.PatternNode(sdfg.SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.state_a, cls.state_b)]

    def can_be_applied(self, graph: SDFG, expr_index, sdfg: SDFG, permissive=False):
        a: SDFGState = self.state_a
        b: SDFGState = self.state_b

        in_edges = graph.in_edges(b)

        # Only apply in cases where DeadStateElimination wouldn't
        if len(in_edges) <= 1:
            return False

        # Directed graph has only one edge between two nodes
        edge = graph.edges_between(a, b)[0]

        if edge.data.assignments:
            return False
        if edge.data.is_unconditional():
            return False

        # Evaluate condition
        scond = edge.data.condition_sympy()
        if scond == False:
            return True

        return False

    def apply(self, _, sdfg: SDFG):
        a: SDFGState = self.state_a
        b: SDFGState = self.state_b
        edge = sdfg.edges_between(a, b)[0]
        sdfg.remove_edge(edge)
