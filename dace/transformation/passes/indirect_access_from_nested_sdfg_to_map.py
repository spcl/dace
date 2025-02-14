import dace
from dace import properties
from dace.transformation import pass_pipeline as ppl, transformation
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dace import SDFG, Memlet, SDFGState, data, dtypes, properties, InterstateEdge, SDFGState, properties
from dace.sdfg.graph import Edge

import copy

@properties.make_properties
@transformation.explicit_cf_compatible
class IndirectAccessFromNestedSDFGToMap(ppl.Pass):
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def can_preprocess(sdfg: SDFG, state:SDFGState,
                       map_entry: dace.nodes.MapEntry) -> bool:
        # All edges of kernel map directly go to a NestedSDFG
        dst_nodes = set()
        for oe in state.out_edges(map_entry):
            dst_nodes.add(oe.dst)
        #print(dst_nodes)
        if len(dst_nodes) != 1:
            return False
        dst_node = dst_nodes.pop()
        if not isinstance(dst_node, dace.nodes.NestedSDFG):
            return False

        # Need to have 2 states, transition to second state should assign the temporary accessess
        # If first staet is not empty, move them to the second state
        # all names
        nsdfg = dst_node.sdfg
        nested_start_state = nsdfg.start_state
        if nsdfg.out_degree(nested_start_state) != 1:
            return False


    def can_apply(sdfg: SDFG, state: SDFGState,
                  map_entry: dace.nodes.MapEntry,
                  names: Set[str]) -> bool:
        # All edges of kernel map directly go to a NestedSDFG
        dst_nodes = set()
        for oe in state.out_edges(map_entry):
            dst_nodes.add(oe.dst)

        if len(dst_nodes) != 1:
            return False
        dst_node = dst_nodes.pop()
        if isinstance(dst_node, dace.nodes.NestedSDFG):
            return False
        if len(dst_node.sdfg.states()) != 2:
            return False

        # Need to have 2 states, first state empty, transition to second state should assign
        # all names
        nsdfg = dst_node.sdfg
        start_state = nsdfg.start_state
        if len(start_state.nodes()) != 0 or nsdfg.out_degree(start_state) != 1:
            return False
        interstate_edge : dace.InterstateEdge = nsdfg.out_edges(start_state)[0].data
        #print(interstate_edge)
        if set(interstate_edge.assignments.keys()) != set(names):
            return False

        old_symbol_mapping = dict()
        new_symbol_mapping = dict()
        for assignment_dst, assignment_src in interstate_edge.assignments.items():
            #print(assignment_dst, assignment_src)
            assert assignment_src in nsdfg.arrays
            assert assignment_src in dst_node.in_connectors
            assert len(list(state.in_edges_by_connector(dst_node, assignment_src))) == 1
            new_symbol_mapping[assignment_dst] = list(state.in_edges_by_connector(dst_node, assignment_src))[0]
            old_symbol_mapping[assignment_dst] = assignment_src

        return True

    def preprocess_for_symbollify(sdfg : dace.SDFG, state : dace.SDFGState, map_entry : dace.nodes.MapEntry) -> typing.List[str]:
        names = []

        # All edges of kernel map directly go to a NestedSDFG
        dst_nodes = set()
        for oe in state.out_edges(map_entry):
            dst_nodes.add(oe.dst)
        #print(dst_nodes)
        assert len(dst_nodes) == 1
        dst_node = dst_nodes.pop()
        assert isinstance(dst_node, dace.nodes.NestedSDFG)

        # Need to have 2 states, transition to second state should assign the temporary accessess
        # If first staet is not empty, move them to the second state
        # all names
        nsdfg = dst_node.sdfg
        nested_start_state = nsdfg.start_state
        assert nsdfg.out_degree(nested_start_state) == 1
        interstate_edge : dace.InterstateEdge = nsdfg.out_edges(nested_start_state)[0].data
        nested_kernel_state  = nsdfg.out_edges(nested_start_state)[0].dst

        # Empty the first state
        nodes_to_add = set()
        edges_to_add = set()
        nested_kernel_start_nodes = set()
        nested_kernel_end_nodes = set()
        nested_start_start_ndoes = set()
        nested_start_end_nodes = set()
        node_mapping = dict()

        # Get start and end nodes
        for (start_nodes, end_nodes, _state) in [(nested_start_start_ndoes, nested_start_end_nodes, nested_start_state),
                                                (nested_kernel_start_nodes, nested_kernel_end_nodes, nested_kernel_state)]:
                for node in _state.nodes():
                    if _state.in_degree(node) == 0:
                        start_nodes.add(node)
                    if _state.out_degree(node) == 0:
                        end_nodes.add(node)

        # We need connect these type of nodes (if end node of previous state aligns with a start node then connect)
        subst = dict()
        for end_node in nested_start_end_nodes:
            assert isinstance(end_node, dace.nodes.AccessNode)
            assert (end_node.data in [n.data for n in nested_kernel_start_nodes if isinstance(n, dace.nodes.AccessNode)])
            for n in nested_kernel_start_nodes:
                if isinstance(n, dace.nodes.AccessNode):
                    if end_node.data == n.data:
                        subst[end_node.data] = n

        # Copy nodes from first state
        for node in nested_start_state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data in subst:
                node_mapping[node] = subst[node.data]
            else:
                newnode = copy.deepcopy(node)
                nodes_to_add.add(newnode)
                node_mapping[node] = newnode

        # Copy edges from first state, with correct substitutions (previous for loo p does it)
        #print(node_mapping)
        for edge in nested_start_state.edges():
            u, uc, v, vc, m = edge
            edges_to_add.add((node_mapping[u], uc, node_mapping[v], vc, m))

        for node in nodes_to_add:
            if node not in nested_kernel_state.nodes():
                nested_kernel_state.add_node(node)

        for edge in edges_to_add:
            if edge not in nested_kernel_state.edges():
                nested_kernel_state.add_edge(*edge)

        for node in nested_start_state.nodes():
            nested_start_state.remove_node(node)


        # Generate assignments, there is a problem
        # We might not have x = b, with b passed to the nested SDFG
        # but a = b + x, then we need to keep first map the statements to: x = x + b
        # and then replace all occurences of a with x + b, and make assignment x = x
        new_assignments = dict()
        repl_dict = dict()
        for assignment_dst, assignment_src in interstate_edge.assignments.items():
            if assignment_dst in dst_node.in_connectors:
                #print(assignment_dst, "in")
                names.append(str(assignment_dst))
            else:
                #print(assignment_dst, "out")
                #print(assignment_src, type(assignment_src), dace.symbolic.SymExpr(assignment_src))
                sym_expr = dace.symbolic.SymExpr(assignment_src)
                free_syms_in_in_conns = [sym for sym in sym_expr.free_symbols if (str(sym) in dst_node.in_connectors)]
                assert len(free_syms_in_in_conns) == 1
                #print(free_syms_in_in_conns)
                free_sym = free_syms_in_in_conns[0]
                new_assignments[str(free_sym)] = str(free_sym)
                repl_dict[str(assignment_dst)] = assignment_src

                names.append(str(free_sym))
        nsdfg.replace_dict(repldict=repl_dict)
        interstate_edge.assignments = new_assignments

        return names

    def symbollify(sdfg : dace.SDFG, state : dace.SDFGState, map_entry : dace.nodes.MapEntry, names : typing.List[str]):
        # All edges of kernel map directly go to a NestedSDFG
        dst_nodes = set()
        for oe in state.out_edges(map_entry):
            dst_nodes.add(oe.dst)
        #print(dst_nodes)
        assert len(dst_nodes) == 1
        dst_node = dst_nodes.pop()
        assert isinstance(dst_node, dace.nodes.NestedSDFG)

        # Need to have 2 states, first state empty, transition to second state should assign
        # all names
        nsdfg = dst_node.sdfg
        start_state = nsdfg.start_state
        assert len(start_state.nodes()) == 0 and nsdfg.out_degree(start_state) == 1
        interstate_edge : dace.InterstateEdge = nsdfg.out_edges(start_state)[0].data
        #print(interstate_edge)
        assert set(interstate_edge.assignments.keys()) == set(names)
        #print(nsdfg.symbols, nsdfg.arrays)
        old_symbol_mapping = dict()
        new_symbol_mapping = dict()
        for assignment_dst, assignment_src in interstate_edge.assignments.items():
            #print(assignment_dst, assignment_src)
            assert assignment_src in nsdfg.arrays
            assert assignment_src in dst_node.in_connectors
            assert len(list(state.in_edges_by_connector(dst_node, assignment_src))) == 1
            new_symbol_mapping[assignment_dst] = list(state.in_edges_by_connector(dst_node, assignment_src))[0]
            old_symbol_mapping[assignment_dst] = assignment_src

        #print(new_symbol_mapping)

        # For all in connectors that do not direct to the members, create access nodes
        outside_access_nodes = dict()

        for in_connector in dst_node.in_connectors:
            if in_connector not in old_symbol_mapping.values():
                #print("In conn", in_connector)
                # Find the corresponding array
                assert len(list(state.in_edges_by_connector(dst_node, in_connector))) == 1
                u, uc, v, vc, memlet = list(state.in_edges_by_connector(dst_node, in_connector))[0]
                if vc not in sdfg.arrays:
                    #print(f"{vc} not in sdfg.arrays")
                    nname = sdfg.add_datadesc(name=vc, datadesc=copy.deepcopy(nsdfg.arrays[vc]))
                    assert nname == vc

                # Since it is from a map to another map we do need a second access node.
                #naccess = state.add_access(vc)
                #state.add_edge(u, uc, naccess, None, copy.deepcopy(memlet))
                outside_access_nodes[in_connector] = (u, uc, copy.deepcopy(memlet))

        # Same for out connectors
        outside_exit_access_nodes = dict()
        for out_connector in dst_node.out_connectors:
            for out_connector in dst_node.out_connectors:
                #print("Out conn", out_connector)
                assert len(list(state.out_edges_by_connector(dst_node, out_connector))) == 1
                u, uc, v, vc, memlet = list(state.out_edges_by_connector(dst_node, out_connector))[0]
                if uc not in sdfg.arrays:
                    #print(f"{uc} not in sdfg.arrays")
                    nname = sdfg.add_datadesc(name=uc, datadesc=copy.deepcopy(nsdfg.arrays[uc]))
                    assert nname == uc
                outside_exit_access_nodes[out_connector] = (v, vc, copy.deepcopy(memlet))

        # Redirect the new access nodes to the map
        inner_map_entry, inner_map_exit = state.add_map(
            name="inner_kernel",
            ndrange={"__i":dace.subsets.Range([(0,0,1)])},
            schedule=dace.ScheduleType.Sequential,
            unroll=True
        )
        # Create nodes from outer to inner map
        for data_name, (src, src_conn, memlet) in outside_access_nodes.items():
            state.add_edge(src, src_conn, inner_map_entry, f"IN_{data_name}", copy.deepcopy(memlet))
            inner_map_entry.add_in_connector(f"IN_{data_name}")
            inner_map_entry.add_out_connector(f"OUT_{data_name}")

        # Create the symbols
        for name, edge in new_symbol_mapping.items():
            state.add_edge(edge.src, edge.src_conn, inner_map_entry, name, copy.deepcopy(edge.data))
            inner_map_entry.add_in_connector(name)

        # Create nodes from inner exit map to outer exit map
        for data_name, (dst, dst_conn, memlet) in outside_exit_access_nodes.items():
            state.add_edge(inner_map_exit, f"OUT_{data_name}", dst, dst_conn, copy.deepcopy(memlet))
            inner_map_exit.add_in_connector(f"IN_{data_name}")
            inner_map_exit.add_out_connector(f"OUT_{data_name}")

        # Now iterate through the second state
        # Get start and end nodes.
        # Connect start nodes to map entry
        # Connect end nodes to map exit
        main_kernel_state = nsdfg.out_edges(start_state)[0].dst

        # Copy all nodes outside nestd SDFG
        nodes_to_add = set()
        edges_to_add = set()
        start_nodes = set()
        end_nodes = set()
        node_mapping = dict()
        for node in main_kernel_state.nodes():
            newnode = copy.deepcopy(node)
            nodes_to_add.add(newnode)
            node_mapping[node] = newnode
            if main_kernel_state.in_degree(node) == 0:
                start_nodes.add(newnode)
            if main_kernel_state.out_degree(node) == 0:
                end_nodes.add(newnode)

        for edge in main_kernel_state.edges():
            u, uc, v, vc, m = edge
            edges_to_add.add((node_mapping[u], uc, node_mapping[v], vc, m))

        for node in nodes_to_add:
            if isinstance(node, dace.nodes.AccessNode) and node.data not in sdfg.arrays:
                sdfg.add_datadesc(name=node.data, datadesc=copy.deepcopy(nsdfg.arrays[node.data]))

            if node not in state.nodes():
                state.add_node(node)

            if node in start_nodes:
                ies = list(state.in_edges_by_connector(inner_map_entry, f"IN_{node.data}"))
                assert len(ies) == 1
                ie = ies[0]
                state.add_edge(inner_map_entry, f"OUT_{node.data}", node, None, copy.deepcopy(ie.data))

            if node in end_nodes:
                oes = list(state.out_edges_by_connector(inner_map_exit, f"OUT_{node.data}"))
                assert len(oes) == 1
                oe = oes[0]
                state.add_edge(node, None, inner_map_exit, f"IN_{node.data}", copy.deepcopy(oe.data))

        for edge in edges_to_add:
            if edge not in state.edges():
                state.add_edge(*edge)

        # Done, remove nested SDFG
        state.remove_node(dst_node)

    def apply_pass(
        self, sdfg: SDFG, _
    ) -> Optional[Set[Union[SDFGState, Edge[InterstateEdge]]]]:
        for s in sdfg.states():
            kernel_entries = [n for n in s.nodes() if isinstance(n, dace.nodes.MapEntry) and s.scope_dict()[n] is None]
            for n in kernel_entries:
                if isinstance(n, dace.nodes.MapEntry):
                    if self.can_preprocess(n):
                        names = self.preprocess_for_symbollify(sdfg, s, n)
                        assert self.can_apply(sdfg, s, n, names)
                        self.symbollify(sdfg, s, n, names)
                        sdfg.validate()