# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module contains classes that implement the OTF map fusion transformation.
"""
import copy
import sympy

from typing import List, Tuple

from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState, StateSubgraphView
from dace.sdfg import nodes as nds
from dace.memlet import Memlet
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import data as dt
from dace import dtypes
from dace import symbolic, nodes
from dace.properties import SymbolicProperty, make_properties, Property

from dace.transformation.dataflow.stream_transient import AccumulateTransient
from dace.transformation.dataflow.local_storage import OutLocalStorage, InLocalStorage


@make_properties
class OTFMapFusion(transformation.SingleStateTransformation):
    """
    Performs fusion of two maps by replicating the contents of the first into the second map
    until all the input dependencies (memlets) of the second one are met.
    """
    first_map_exit = transformation.PatternNode(nds.ExitNode)
    array = transformation.PatternNode(nds.AccessNode)
    second_map_entry = transformation.PatternNode(nds.EntryNode)

    identity = SymbolicProperty(desc="Identity value to set", default=None, allow_none=True)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # General conditions
        if not sdfg.arrays[self.array.data].transient:
            return False
        if graph.out_degree(self.first_map_exit) > 1:
            return False
        if graph.in_degree(self.array) > 1:
            return False

        # No non-transients in scope of first map
        first_map_entry = graph.entry_node(self.first_map_exit)
        subgraph = graph.scope_subgraph(first_map_entry, include_entry=True, include_exit=True)
        for dnode in subgraph.data_nodes():
            if not sdfg.arrays[dnode.data].transient:
                return False

        # Condition: Equations solvable (dims(first map) <= dims(second map))
        if len(first_map_entry.map.params) > len(self.second_map_entry.map.params):
            return False

        # Condition: Consumed is covered by produced data
        produce_edge = next(graph.edges_between(self.first_map_exit, self.array).__iter__())
        consume_edges = graph.edges_between(self.array, self.second_map_entry)
        for edge in consume_edges:
            read_memlet = edge.data
            write_memlet = produce_edge.data
            if not write_memlet.subset.covers(read_memlet.subset):
                return False

        # First memlets
        first_map_dims = set(self.first_map_exit.map.params)
        produce_memlets = {}
        for edge in graph.in_edges(self.first_map_exit):
            memlet = edge.data
            # Unique
            if memlet.data in produce_memlets:
                return False

            # Condition: Consecutive produce
            for access in memlet.subset:
                _, _, step = access
                # If we skip, we get undefined points in array
                if step != 1:
                    return False

            # Condition: Constant num_elements
            for dim_num_elements in memlet.subset.size_exact():
                syms = set(map(str, dim_num_elements.free_symbols))
                if len(syms.intersection(first_map_dims)) > 0:
                    return False

            # WCR
            if memlet.wcr is not None:
                # Condition: Real dimensions only
                real_dimensions = set()
                for access in memlet.subset:
                    start, _, _ = access
                    for sym in start.free_symbols:
                        real_dimensions.add(str(sym))

                # Map defines more dimensions than WCR memlet has
                if len(real_dimensions) == 0 or len(first_map_dims.difference(real_dimensions)) > 0:
                    return False

            produce_memlets[memlet.data] = memlet

        second_map_dims = set(self.second_map_entry.map.params)
        for edge in graph.out_edges(self.second_map_entry):
            memlet = edge.data
            if memlet.data not in produce_memlets:
                continue

            # Condition: Constant num_elements
            for dim_num_elements in memlet.subset.size_exact():
                if len(set(map(str, dim_num_elements.free_symbols)).intersection(second_map_dims)) > 0:
                    # Num elements in dimension depends on map
                    return False

            consume_subset = tuple(memlet.subset.ranges)

            produce_memlet = produce_memlets[memlet.data]
            produce_subset = tuple(produce_memlet.subset.ranges)
            param_mapping = OTFMapFusion.solve(first_map_entry.map.params, produce_subset,
                                               self.second_map_entry.map.params, consume_subset)

            if param_mapping is None:
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        intermediate_access_node = self.array
        first_map_exit = self.first_map_exit
        first_map_entry = graph.entry_node(first_map_exit)

        # Prepare: Make first and second map parameters disjoint
        # This avoids mutual matching: i -> j, j -> i
        subgraph = graph.scope_subgraph(first_map_entry, include_entry=True, include_exit=True)
        for param in first_map_entry.map.params:
            i = 0
            new_param = f"_i{i}"
            while new_param in self.second_map_entry.map.params or new_param in first_map_entry.map.params:
                i = i + 1
                new_param = f"_i{i}"

            advanced_replace(subgraph, param, new_param)

        # Prepare: Preemptively rename params defined by second map in scope of first
        # This avoids that local variables (e.g., in nested SDFG) have collisions with new map scope
        for param in self.second_map_entry.map.params:
            new_param = param + "_local"
            advanced_replace(subgraph, param, new_param)

        # Add local buffers for array-like OTFs
        for edge in graph.out_edges(self.second_map_entry):
            if edge.data is None or edge.data.data != intermediate_access_node.data:
                continue

            xform = InLocalStorage()
            xform._sdfg = sdfg
            xform.state_id = sdfg.node_id(graph)
            xform.node_a = edge.src
            xform.node_b = edge.dst
            xform.array = intermediate_access_node.data
            if xform.can_be_applied(graph, expr_index=0, sdfg=sdfg):
                InLocalStorage.apply_to(sdfg=sdfg,
                                        node_a=edge.src,
                                        node_b=edge.dst,
                                        options={"array": intermediate_access_node.data},
                                        save=False)

        for edge in graph.in_edges(first_map_exit):
            if edge.data is None or edge.data.data != intermediate_access_node.data:
                continue

            if edge.data.wcr is None:
                xform = OutLocalStorage()
                xform._sdfg = sdfg
                xform.state_id = sdfg.node_id(graph)
                xform.node_a = edge.src
                xform.node_b = edge.dst
                xform.array = intermediate_access_node.data
                if xform.can_be_applied(graph, expr_index=0, sdfg=sdfg):
                    OutLocalStorage.apply_to(sdfg=sdfg,
                                             node_a=edge.src,
                                             node_b=edge.dst,
                                             options={
                                                 "array": intermediate_access_node.data,
                                             },
                                             save=False)
            else:
                xform = AccumulateTransient()
                xform._sdfg = sdfg
                xform.state_id = sdfg.node_id(graph)
                xform.map_exit = edge.src
                xform.outer_map_exit = edge.dst
                xform.array = intermediate_access_node.data
                xform.identity = self.identity
                if xform.can_be_applied(graph, expr_index=0, sdfg=sdfg):
                    AccumulateTransient.apply_to(sdfg=sdfg,
                                                 map_exit=edge.src,
                                                 outer_map_exit=edge.dst,
                                                 array=intermediate_access_node.data,
                                                 options={
                                                     "array": intermediate_access_node.data,
                                                     "identity": self.identity
                                                 },
                                                 save=False)

        # Phase 1: Add new access nodes to second map
        for edge in graph.edges_between(intermediate_access_node, self.second_map_entry):
            graph.remove_edge_and_connectors(edge)

        connector_mapping = {}
        for edge in graph.in_edges(first_map_entry):
            new_in_connector = self.second_map_entry.next_connector(edge.dst_conn[3:])
            new_in_connector = "IN_" + new_in_connector
            if not self.second_map_entry.add_in_connector(new_in_connector):
                raise ValueError("Failed to add new in connector")

            memlet = copy.deepcopy(edge.data)
            graph.add_edge(edge.src, edge.src_conn, self.second_map_entry, new_in_connector, memlet)

            connector_mapping[edge.dst_conn] = new_in_connector

        # Phase 2: Match relevant memlets
        produce_memlets = {}
        for edge in graph.in_edges(first_map_exit):
            memlet = edge.data
            produce_memlets[memlet.data] = memlet

        # Group by same access scheme
        consume_memlets = {}
        for edge in graph.out_edges(self.second_map_entry):
            memlet = edge.data
            if memlet.data not in produce_memlets:
                continue

            if memlet.data not in consume_memlets:
                consume_memlets[memlet.data] = {}

            accesses = tuple(memlet.subset.ranges)
            if accesses not in consume_memlets[memlet.data]:
                consume_memlets[memlet.data][accesses] = []

            consume_memlets[memlet.data][accesses].append(edge)

            # And remove from second map
            self.second_map_entry.remove_out_connector(edge.src_conn)
            graph.remove_edge(edge)

        # Phase 3: OTF - copy content of first map for each memlet of second according to matches
        for array in consume_memlets:
            first_memlet = produce_memlets[array]
            first_accesses = tuple(first_memlet.subset.ranges)
            for second_accesses in consume_memlets[array]:
                # Step 1: Infer index access of second map to new inputs with respect to original first map
                mapping = OTFMapFusion.solve(first_map_entry.map.params, first_accesses,
                                             self.second_map_entry.map.params, second_accesses)

                # Step 2: Add Temporary buffer
                tmp_name = sdfg.temp_data_name()
                shape = first_memlet.subset.size_exact()
                tmp_name, tmp_desc = sdfg.add_array(tmp_name,
                                                    shape=shape,
                                                    dtype=sdfg.arrays[array].dtype,
                                                    transient=True,
                                                    find_new_name=True,
                                                    lifetime=dtypes.AllocationLifetime.Scope)
                tmp_access = graph.add_access(tmp_name)

                # Add edges from temporary buffer to second map's content
                for edge in consume_memlets[array][second_accesses]:
                    otf_memlet = Memlet.from_array(dataname=tmp_name, datadesc=tmp_desc, wcr=None)
                    graph.add_edge(tmp_access, None, edge.dst, edge.dst_conn, otf_memlet)

                # Step 3: Copy content of first map into second map
                otf_nodes = self._copy_first_map_contents(sdfg, graph, first_map_entry, first_map_exit)

                # Connect the nodes to the otf_scalar and second map entry
                for node in otf_nodes:
                    # Connect new OTF nodes to tmp_access for write
                    for edge in graph.edges_between(node, first_map_exit):
                        otf_memlet = Memlet.from_array(dataname=tmp_name, datadesc=tmp_desc, wcr=first_memlet.wcr)
                        graph.add_edge(edge.src, edge.src_conn, tmp_access, None, otf_memlet)
                        graph.remove_edge(edge)

                    # Connect new OTF nodes to second map entry for read
                    for edge in graph.edges_between(first_map_entry, node):
                        memlet = copy.deepcopy(edge.data)

                        in_connector = edge.src_conn.replace("OUT", "IN")
                        if in_connector in connector_mapping:
                            out_connector = connector_mapping[in_connector].replace("IN", "OUT")
                        else:
                            out_connector = edge.src_conn

                        if out_connector not in self.second_map_entry.out_connectors:
                            self.second_map_entry.add_out_connector(out_connector)

                        graph.add_edge(self.second_map_entry, out_connector, node, edge.dst_conn, memlet)
                        graph.remove_edge(edge)

                # Step 4: Rename all symbols of first map in copied content my matched symbol of second map
                otf_nodes.append(self.second_map_entry)
                otf_subgraph = StateSubgraphView(graph, otf_nodes)
                for param in mapping:
                    if isinstance(param, tuple):
                        # Constant intervals
                        continue

                    advanced_replace(otf_subgraph, str(param), str(mapping[param]))

        # Check if first_map is still consumed by some node
        if graph.out_degree(intermediate_access_node) == 0:
            del sdfg.arrays[intermediate_access_node.data]
            graph.remove_node(intermediate_access_node)

            subgraph = graph.scope_subgraph(first_map_entry, include_entry=True, include_exit=True)
            for dnode in subgraph.data_nodes():
                if dnode.data in sdfg.arrays:
                    del sdfg.arrays[dnode.data]

            obsolete_nodes = graph.all_nodes_between(first_map_entry,
                                                     first_map_exit) | {first_map_entry, first_map_exit}
            graph.remove_nodes_from(obsolete_nodes)

    def _copy_first_map_contents(self, sdfg: SDFG, graph: SDFGState, first_map_entry: nodes.MapEntry,
                                 first_map_exit: nodes.MapExit):
        inter_nodes = list(graph.all_nodes_between(first_map_entry, first_map_exit) - {first_map_entry})

        # Add new nodes
        new_inter_nodes = [copy.deepcopy(node) for node in inter_nodes]
        for node in new_inter_nodes:
            graph.add_node(node)

        id_map = {graph.node_id(old): graph.node_id(new) for old, new in zip(inter_nodes, new_inter_nodes)}

        # Rename transients
        tmp_map = {}
        for node in new_inter_nodes:
            if not isinstance(node, nodes.AccessNode) or node.data in tmp_map:
                continue

            new_name = sdfg.temp_data_name()
            desc = sdfg.arrays[node.data]
            sdfg.arrays[new_name] = copy.deepcopy(desc)
            tmp_map[node.data] = new_name

        for node in new_inter_nodes:
            if not isinstance(node, nodes.AccessNode) or node.data not in tmp_map:
                continue
            node.data = tmp_map[node.data]

        def map_node(node):
            return graph.node(id_map[graph.node_id(node)])

        def map_memlet(memlet):
            memlet = copy.deepcopy(memlet)
            memlet.data = tmp_map.get(memlet.data, memlet.data)
            return memlet

        for edge in graph.edges():
            if edge.src in inter_nodes or edge.dst in inter_nodes:
                src = map_node(edge.src) if edge.src in inter_nodes else edge.src
                dst = map_node(edge.dst) if edge.dst in inter_nodes else edge.dst
                edge_data = map_memlet(edge.data)
                graph.add_edge(src, edge.src_conn, dst, edge.dst_conn, edge_data)

        return new_inter_nodes

    @staticmethod
    def solve(first_params: List[str], write_accesses: Tuple, second_params: List[str], read_accesses: Tuple):
        """
        Infers the memory access for the write memlet given the
        location/parameters of the read access.

        Example:
        - Write memlet: ``A[i + 1, j]``
        - Read memlet: ``A[k, l]``
        - Infer: ``k -> i - 1, l - > j``

        :param first_params: parameters of the first_map_entry.
        :param write_accesses: subset ranges of the write memlet.
        :param second_params: parameters of the second map_entry.
        :param read_accesses: subset ranges of the read memlet.
        :return: mapping of parameters.
        """

        # Make sure that parameters of first_map_entry are different symbols than parameters of second_map_entry.
        first_params_subs = {}
        first_params_subs_ = {}
        for i, param in enumerate(first_params):
            s = symbolic.symbol(f'f_{i}')
            first_params_subs[param] = s
            first_params_subs_[s] = param

        second_params_subs = {}
        second_params_subs_ = {}
        for i, param in enumerate(second_params):
            s = symbolic.symbol(f's_{i}')
            second_params_subs[param] = s
            second_params_subs_[s] = param

        mapping = {}
        for write_access, read_access in zip(write_accesses, read_accesses):
            b0, e0, s0 = write_access
            if isinstance(e0, symbolic.SymExpr):
                return None

            b0 = symbolic.pystr_to_symbolic(b0)
            e0 = symbolic.pystr_to_symbolic(e0)
            s0 = symbolic.pystr_to_symbolic(s0)
            for param in first_params_subs:
                b0 = b0.subs(param, first_params_subs[param])
                e0 = e0.subs(param, first_params_subs[param])
                s0 = s0.subs(param, first_params_subs[param])

            b1, e1, s1 = read_access
            if isinstance(e1, symbolic.SymExpr):
                return None

            b1 = symbolic.pystr_to_symbolic(b1)
            e1 = symbolic.pystr_to_symbolic(e1)
            s1 = symbolic.pystr_to_symbolic(s1)
            for param in second_params_subs:
                b1 = b1.subs(param, second_params_subs[param])
                e1 = e1.subs(param, second_params_subs[param])
                s1 = s1.subs(param, second_params_subs[param])

            # Condition: num elements must match
            # Note: we already assumed that num elements is constant
            if (b0 - e0) != (b1 - e1):
                return None

            params_0 = b0.free_symbols.intersection(first_params_subs.values())
            params_1 = b1.free_symbols.intersection(second_params_subs.values())
            if len(params_0) == 0 and len(params_1) == 0:
                # Constants
                mapping[(b0, e0)] = (b1, e1)
            elif len(params_0) == 1 and len(params_1) <= 1:
                b_eq = sympy.Eq(b0, b1)
                e_eq = sympy.Eq(e0, e1)
                params = b0.free_symbols.union(e0.free_symbols)
                params = params.intersection(first_params_subs_.keys())
                sol = sympy.solve((b_eq, e_eq), params)
                if not sol:
                    return None

                for param, sub in sol.items():
                    if param in mapping:
                        return None

                    mapping[param] = sub
            else:
                return None

        # Translate back to original symbols
        solution = {}
        for param, sub in mapping.items():
            if isinstance(param, tuple):
                solution[param] = sub
            else:
                for param_, sub_ in second_params_subs.items():
                    sub = sub.subs(sub_, param_)

                solution[symbolic.pystr_to_symbolic(first_params_subs_[param])] = sub
        return solution


def advanced_replace(subgraph: StateSubgraphView, s: str, s_: str) -> None:
    subgraph.replace(s, s_)

    # .replace does not change map param definitions and nested SDFGs
    for node in subgraph.nodes():
        if isinstance(node, nodes.MapEntry):
            params = [s_ if p == s else p for p in node.map.params]
            node.map.params = params
        elif isinstance(node, nodes.NestedSDFG):
            for nsdfg in node.sdfg.all_sdfgs_recursive():
                nsdfg.replace(s, s_)
                for nstate in nsdfg.nodes():
                    for nnode in nstate.nodes():
                        if isinstance(nnode, nodes.MapEntry):
                            params = [s_ if p == s else p for p in nnode.map.params]
                            nnode.map.params = params
