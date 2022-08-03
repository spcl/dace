# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement the OTF map fusion transformation.
"""
import copy
import sympy

from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import nodes as nds
from dace.memlet import Memlet
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import data as dt


class OTFMapFusion(transformation.SingleStateTransformation):
    """ Performs fusion of two maps by replicating the contents of the first into the second map
        until all the input dependencies (memlets) of the second one are met.
    """
    first_map_exit = transformation.PatternNode(nds.ExitNode)
    array = transformation.PatternNode(nds.AccessNode)
    second_map_entry = transformation.PatternNode(nds.EntryNode)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Check intermediate nodes between both maps.
        for _, _, node, _, _ in graph.out_edges(self.first_map_exit):
            # Only map -> array -> map
            if not isinstance(node, nds.AccessNode):
                return False

            # Non-transient blocks removal of first map
            if not sdfg.arrays[node.data].transient:
                return False

            # Check that array is not co-produced by other parent map.
            producers = set(map(lambda edge: edge.src, graph.in_edges(node)))
            for prod in producers:
                if prod != self.first_map_exit:
                    return False

        # No WCR on first map
        for edge in graph.in_edges(self.first_map_exit):
            if edge.data.wcr is not None:
                return False

        # Only single write per array
        write_memlets = {}
        for edge in graph.in_edges(self.first_map_exit):
            memlet = edge.data
            array = memlet.data

            if array in write_memlets:
                return False

            write_memlets[array] = memlet

        # Memlets access must be inferable
        first_map_entry = graph.entry_node(self.first_map_exit)
        for edge in graph.out_edges(self.second_map_entry):
            read_memlet = edge.data
            if read_memlet.data not in write_memlets:
                continue

            write_memlet = write_memlets[read_memlet.data]
            param_subs = OTFMapFusion.solve(first_map_entry.map.params, write_memlet.subset.ranges,
                                            self.second_map_entry.map.params, read_memlet.subset.ranges)
            if param_subs is None:
                return False

        # Success
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        first_map_entry = graph.entry_node(self.first_map_exit)

        ### Prepare Access Nodes

        # Add edges for input of first map to second map
        for edge in graph.in_edges(first_map_entry):
            if self.second_map_entry.add_in_connector(edge.dst_conn + "_"):
                memlet = copy.deepcopy(edge.data)
                graph.add_edge(edge.src, edge.src_conn, self.second_map_entry, edge.dst_conn + "_", memlet)
            else:
                raise ValueError("Failed to connect")

        # Collect output nodes of first map
        intermediate_dnodes = set()
        for _, _, node, _, _ in graph.out_edges(self.first_map_exit):
            if not isinstance(node, nds.AccessNode):
                continue

            intermediate_dnodes.add(node)

        # Remove edges of these nodes to second map
        for dnode in intermediate_dnodes:
            for edge in graph.edges_between(dnode, self.second_map_entry):
                graph.remove_edge_and_connectors(edge)

        ### Re-wiring the memlets ###

        # Collect write-memlets of first map
        # One write per array
        write_memlets = {}
        for edge in graph.in_edges(self.first_map_exit):
            memlet = edge.data
            array = memlet.data

            # Only single write memlet per array
            assert array not in write_memlets
            write_memlets[array] = memlet

        # Collect read-memlets of second map
        # Group by same access scheme
        read_memlets = {}
        for edge in graph.out_edges(self.second_map_entry):
            read_memlet = edge.data
            array = read_memlet.data
            if array not in write_memlets:
                continue

            if array not in read_memlets:
                read_memlets[array] = {}

            accesses = read_memlet.subset.ranges
            accesses = tuple(accesses)
            if accesses not in read_memlets:
                read_memlets[array][accesses] = []

            read_memlets[array][accesses].append(edge)

        # OTF: Replace read memlet by copying full map content of corresponding write memlet
        for array in read_memlets:
            write_memlet = write_memlets[array]
            write_accesses = write_memlet.subset.ranges
            for read_accesses in read_memlets[array]:
                # Map read access to write access
                param_subs = OTFMapFusion.solve(first_map_entry.map.params, write_accesses,
                                                self.second_map_entry.map.params, read_accesses)

                new_nodes = self._copy_first_map_contents(sdfg, graph, first_map_entry)

                tmp_name = "__otf"
                tmp_name, _ = sdfg.add_scalar(tmp_name, sdfg.arrays[array].dtype, transient=True, find_new_name=True)
                tmp_access = graph.add_access(tmp_name)

                # Connect read-in memlets to tmp_access node
                read_memlet_group = read_memlets[array][read_accesses]
                for edge in read_memlet_group:
                    graph.add_edge(tmp_access, None, edge.dst, edge.dst_conn, Memlet(tmp_name))

                    self.second_map_entry.remove_out_connector(edge.src_conn)
                    graph.remove_edge(edge)

                # Connect new sub graph
                for node in new_nodes:
                    # Connect new OTF nodes to tmp_access for write
                    for edge in graph.edges_between(node, self.first_map_exit):
                        graph.add_edge(edge.src, edge.src_conn, tmp_access, None, Memlet(tmp_name))
                        graph.remove_edge(edge)

                    # Connect new OTF nodes to second map entry for read
                    for edge in graph.edges_between(first_map_entry, node):
                        memlet = copy.deepcopy(edge.data)

                        # Substitute accesses
                        ranges = []
                        for i, access in enumerate(memlet.subset.ranges):
                            b, e, s = access
                            b = sympy.sympify(b, locals=sdfg.constants)
                            e = sympy.sympify(e, locals=sdfg.constants)
                            for param, sub in param_subs[i].items():
                                b = b.subs(param, sub)
                                e = e.subs(param, sub)

                            ranges.append((b, e, s))
                        memlet.subset.ranges = ranges

                        self.second_map_entry.add_out_connector(edge.src_conn + "_")
                        graph.add_edge(self.second_map_entry, edge.src_conn + "_", node, edge.dst_conn, memlet)

                        graph.remove_edge(edge)

        # Check if first_map is still consumed by some node
        remove_first_map = True
        for dnode in intermediate_dnodes:
            remove_first_map = remove_first_map and (graph.out_degree(dnode) == 0)

        # Remove if not
        if remove_first_map:
            graph.remove_nodes_from(
                graph.all_nodes_between(first_map_entry, self.first_map_exit) | {first_map_entry, self.first_map_exit})

        # Remove isolated nodes. Find elegant solution
        for node in graph.nodes():
            if not isinstance(node, nds.AccessNode):
                continue

            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                graph.remove_node(node)

    def _copy_first_map_contents(self, sdfg, graph, first_map_entry):
        inter_nodes = list(graph.all_nodes_between(first_map_entry, self.first_map_exit) - {first_map_entry})
        new_inter_nodes = [copy.deepcopy(node) for node in inter_nodes]
        tmp_map = dict()
        for node in new_inter_nodes:
            if isinstance(node, nds.AccessNode):
                data = sdfg.arrays[node.data]
                if isinstance(data, dt.Scalar) and data.transient:
                    tmp_name = sdfg.temp_data_name()
                    sdfg.add_scalar(tmp_name, data.dtype, transient=True)
                    tmp_map[node.data] = tmp_name
                    node.data = tmp_name
            graph.add_node(node)
        id_map = {graph.node_id(old): graph.node_id(new) for old, new in zip(inter_nodes, new_inter_nodes)}

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
    def solve(first_params, write_accesses, second_params, read_accesses):
        """ Infers the memory access for the write memlet given the
            location/parameters of the read access.

            Example:
            - Write memlet: A[i + 1, j]
            - Read memlet: A[k, l]
            - Infer: k -> i - 1, l - > j

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
            s = sympy.symbols(f'f_{i}')
            first_params_subs[param] = s
            first_params_subs_[s] = param

        second_params_subs = {}
        second_params_subs_ = {}
        for i, param in enumerate(second_params):
            s = sympy.symbols(f's_{i}')
            second_params_subs[param] = s
            second_params_subs_[s] = param

        solutions = []
        for write_access, read_access in zip(write_accesses, read_accesses):
            b0, e0, s0 = write_access
            b0 = sympy.sympify(b0)
            e0 = sympy.sympify(e0)
            for param in first_params_subs:
                b0 = b0.subs(param, first_params_subs[param])
                e0 = e0.subs(param, first_params_subs[param])

            b1, e1, s1 = read_access
            b1 = sympy.sympify(b1)
            e1 = sympy.sympify(e1)
            for param in second_params_subs:
                b1 = b1.subs(param, second_params_subs[param])
                e1 = e1.subs(param, second_params_subs[param])

            # Step is constant
            s0 = int(s0)
            assert (s0 - s1) == 0

            if (b0 - b1) == 0 and (e0 - e1) == 0:
                # Trivial case
                sol_ = {b0: b1}
                solutions.append(sol_)
            else:
                b_eq = sympy.Eq(b0, b1)
                e_eq = sympy.Eq(e0, e1)
                params = b0.free_symbols.union(e0.free_symbols)
                sol_ = sympy.solve((b_eq, e_eq), params)
                if not sol_:
                    return None

                # Translate back to original symbols
                sol = {}
                for param, eq in sol_.items():
                    for param_, sub_ in second_params_subs.items():
                        eq = eq.subs(sub_, param_)

                    sol[first_params_subs_[param]] = eq

                solutions.append(sol)

        return solutions
