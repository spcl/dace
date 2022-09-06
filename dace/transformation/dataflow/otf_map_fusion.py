# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module contains classes that implement the OTF map fusion transformation.
"""
import copy
import sympy

from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import nodes as nds
from dace.memlet import Memlet
from dace.sdfg import utils as sdutil
from dace.subsets import Range
from dace.transformation import transformation
from dace import data as dt
from dace import dtypes
from dace import symbolic

from dace.frontend.operations import detect_reduction_type


class OTFMapFusion(transformation.SingleStateTransformation):
    """
    Performs fusion of two maps by replicating the contents of the first into the second map
    until all the input dependencies (memlets) of the second one are met.
    """
    first_map_exit = transformation.PatternNode(nds.ExitNode)
    array = transformation.PatternNode(nds.AccessNode)
    second_map_entry = transformation.PatternNode(nds.EntryNode)

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

        # Equation Solvability: First map must induce a non-ambiguous set of writes for all possible reads of second map.

        # dims(first map) <= dims(second map)
        first_map_entry = graph.entry_node(self.first_map_exit)
        if len(first_map_entry.map.params) > len(self.second_map_entry.map.params):
            return False

        # Write memlets are unique
        out_memlets = {}
        for edge in graph.in_edges(self.first_map_exit):
            memlet = edge.data

            # Limitation: Only scalars for now
            if memlet.num_elements() > 1:
                return False

            # In general, WCR requires initialization of the data with the identity of the reduction. In the special case of "0"-identity, we don't need to explicitly initialize but can set the "setzero" flag at creation of the data. This avoids adding states.
            if not memlet.wcr is None:
                red_type = detect_reduction_type(memlet.wcr)
                if red_type == dtypes.ReductionType.Custom:
                    return False

                dtype = sdfg.arrays[memlet.data].dtype
                identity = dtypes.reduction_identity(dtype, red_type)
                if identity != 0:
                    return False

                # We require that the parallel dims are separate from the reduce dims
                dims = len(first_map_entry.map.params)
                if len(memlet.subset) != dims:
                    return False

            # Unique choice which write memlet to pick for each read
            if memlet.data in out_memlets:
                return False
            out_memlets[memlet.data] = memlet

        in_memlets = {}
        for edge in graph.out_edges(self.second_map_entry):
            memlet = edge.data

            if not memlet.data in out_memlets:
                continue

            # Only fuse scalars
            if memlet.num_elements() > 1:
                return False

            if memlet.data not in in_memlets:
                in_memlets[memlet.data] = {}

            accesses = tuple(memlet.subset.ranges)
            if accesses not in in_memlets[memlet.data]:
                in_memlets[memlet.data][accesses] = []

            in_memlets[memlet.data][accesses].append(edge)

        for array in in_memlets:
            out_memlet = out_memlets[array]
            out_accesses = tuple(out_memlet.subset.ranges)
            for in_accesses in in_memlets[array]:
                param_mapping = OTFMapFusion.solve(first_map_entry.map.params, out_accesses,
                                                   self.second_map_entry.map.params, in_accesses)
                if param_mapping is None:
                    return False

        # Success
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        first_map_entry = graph.entry_node(self.first_map_exit)

        # Phase 1: Re-wire access nodes of maps

        # a. Collect out access nodes of first map
        intermediate_dnodes = set()
        for edge in graph.out_edges(self.first_map_exit):
            node = edge.dst
            if not isinstance(node, nds.AccessNode):
                continue

            intermediate_dnodes.add(node)

        # b. Remove edges of these nodes to second map
        for dnode in intermediate_dnodes:
            for edge in graph.edges_between(dnode, self.second_map_entry):
                graph.remove_edge_and_connectors(edge)

        # c. Add edges for in access nodes of first map to second map
        connector_mapping = {}
        for edge in graph.in_edges(first_map_entry):
            old_in_connector = edge.dst_conn
            if not edge.data is None and not edge.data.data is None:
                suffix = edge.data.data
            else:
                suffix = "OTF"

            new_in_connector = "IN_" + suffix
            if not self.second_map_entry.add_in_connector(new_in_connector):
                for n in range(2, 65536):
                    new_in_connector = "IN_" + suffix + "_" + str(n)
                    if self.second_map_entry.add_in_connector(new_in_connector):
                        break

            memlet = copy.deepcopy(edge.data)
            graph.add_edge(edge.src, edge.src_conn, self.second_map_entry, new_in_connector, memlet)

            connector_mapping[old_in_connector] = new_in_connector

        # Phase 2: Memlet matching
        # Collect memlet-array map

        out_memlets = {}
        for edge in graph.in_edges(self.first_map_exit):
            memlet = edge.data
            out_memlets[memlet.data] = memlet

        # Group by same access scheme
        in_memlets = {}
        for edge in graph.out_edges(self.second_map_entry):
            memlet = edge.data
            if memlet.data not in out_memlets:
                continue

            if memlet.data not in in_memlets:
                in_memlets[memlet.data] = {}

            accesses = tuple(memlet.subset.ranges)
            if accesses not in in_memlets[memlet.data]:
                in_memlets[memlet.data][accesses] = []

            in_memlets[memlet.data][accesses].append(edge)

            # And remove from second map
            self.second_map_entry.remove_out_connector(edge.src_conn)
            graph.remove_edge(edge)

        # Phase 3: OTF - copy content for each in memlet of second map
        for array in in_memlets:
            out_memlet = out_memlets[array]
            out_accesses = tuple(out_memlet.subset.ranges)
            for in_accesses in in_memlets[array]:
                # Add intermediate scalar for output of copied map content
                tmp_name = "__otf"
                tmp_name, _ = sdfg.add_scalar(tmp_name,
                                              sdfg.arrays[array].dtype,
                                              transient=True,
                                              find_new_name=True,
                                              lifetime=dtypes.AllocationLifetime.Scope)
                tmp_access = graph.add_access(tmp_name)
                tmp_access.setzero = True

                # Connect in memlets of second map to this scalar
                read_memlet_group = in_memlets[array][in_accesses]
                for edge in read_memlet_group:
                    graph.add_edge(tmp_access, None, edge.dst, edge.dst_conn, Memlet(tmp_name))

                # Solve index mapping between memlets
                param_mapping = OTFMapFusion.solve(first_map_entry.map.params, out_accesses,
                                                   self.second_map_entry.map.params, in_accesses)

                # Copy actual content
                first_map_inner_nodes = self._copy_first_map_contents(sdfg, graph, first_map_entry)

                # Prepare: Symbol collision
                # Second map defines symbols (params) and becomes the new outer scope of the copied nodes.
                # Thus, newly introduced symbols in this subgraph must be renamed first to avoid conflicts.
                # Note symbols of first map must be re-named according to the param mapping.
                for node in first_map_inner_nodes:
                    # TODO: More complex cases, e.g. map nests, states
                    if not isinstance(node, nds.MapEntry):
                        continue

                    # Rename map params
                    inner_map_subs = {}
                    inner_map_params = []
                    for param in node.map.params:
                        n = 0
                        r = f"r_{n}"
                        while r in self.second_map_entry.map.params or r in inner_map_params:
                            r = f"r_{n}"
                            n = n + 1

                        r = symbolic.symbol(r)
                        inner_map_subs[param] = r
                        inner_map_params.append(str(r))

                    node.map.params = inner_map_params

                    # Substitue range definitions by symbols for second map
                    ranges = []
                    for access in node.map.range:
                        b, e, s = access
                        b = symbolic.pystr_to_symbolic(b)
                        e = symbolic.pystr_to_symbolic(e)
                        s = symbolic.pystr_to_symbolic(s)

                        b = b.subs(param_mapping)
                        e = e.subs(param_mapping)
                        s = s.subs(param_mapping)

                        ranges.append((b, e, s))
                    node.map.range = Range(ranges)

                    # Re-name all memlets in subgraph
                    scope_subs = {**param_mapping, **inner_map_subs}
                    scope_subgraph = graph.scope_subgraph(node, graph.exit_node(node))
                    for edge in scope_subgraph.edges():
                        memlet = edge.data
                        # Substitute accesses
                        ranges = []
                        for access in memlet.subset.ranges:
                            b, e, s = access
                            b = symbolic.pystr_to_symbolic(b)
                            e = symbolic.pystr_to_symbolic(e)
                            s = symbolic.pystr_to_symbolic(s)

                            b = b.subs(scope_subs)
                            e = e.subs(scope_subs)
                            s = s.subs(scope_subs)

                            ranges.append((b, e, s))
                        memlet.subset = Range(ranges)

                # Now connect the nodes to the otf_scalar and second map entry
                for node in first_map_inner_nodes:
                    for edge in graph.out_edges(node):
                        memlet = edge.data
                        if memlet.wcr is not None and memlet.data == self.array.data:
                            tmp_memlet = Memlet(tmp_name)
                            tmp_memlet.wcr = memlet.wcr
                            tmp_memlet.src_subset = Range([(0, 0, 1)])

                            edge.data = tmp_memlet

                    # Connect new OTF nodes to tmp_access for write
                    for edge in graph.edges_between(node, self.first_map_exit):
                        graph.remove_edge(edge)
                        tmp_memlet = Memlet(tmp_name)
                        tmp_memlet.wcr = edge.data.wcr
                        if not tmp_memlet.wcr is None:
                            tmp_memlet.src_subset = Range([(0, 0, 1)])
                        graph.add_edge(edge.src, edge.src_conn, tmp_access, None, tmp_memlet)

                    # Connect new OTF nodes to second map entry for read
                    for edge in graph.edges_between(first_map_entry, node):
                        memlet = copy.deepcopy(edge.data)
                        if not memlet.wcr is None and memlet.data == self.array.data:
                            memlet.data = tmp_name

                        # Substitute accesses
                        ranges = []
                        for access in memlet.subset.ranges:
                            b, e, s = access
                            b = symbolic.pystr_to_symbolic(b)
                            e = symbolic.pystr_to_symbolic(e)
                            s = symbolic.pystr_to_symbolic(s)

                            b = b.subs(param_mapping)
                            e = e.subs(param_mapping)
                            s = s.subs(param_mapping)

                            ranges.append((b, e, s))

                        memlet.subset = Range(ranges)

                        in_connector = edge.src_conn.replace("OUT", "IN")
                        if in_connector in connector_mapping:
                            out_connector = connector_mapping[in_connector].replace("IN", "OUT")
                        else:
                            out_connector = edge.src_conn

                        if out_connector not in self.second_map_entry.out_connectors:
                            self.second_map_entry.add_out_connector(out_connector)

                        graph.add_edge(self.second_map_entry, out_connector, node, edge.dst_conn, memlet)
                        graph.remove_edge(edge)

        # Check if first_map is still consumed by some node
        remove_first_map = True
        for dnode in intermediate_dnodes:
            if graph.out_degree(dnode) > 0:
                remove_first_map = False
                break

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

            # Step is constant
            assert (s0 - s1) == 0
            if (b0 - b1) == 0 and (e0 - e1) == 0:
                if b0 in mapping:
                    return None

                # Trivial case
                mapping[b0] = b1
            else:
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

        # Translate back to original symbols
        solution = {}
        for param, sub in mapping.items():
            for param_, sub_ in second_params_subs.items():
                sub = sub.subs(sub_, param_)

            solution[first_params_subs_[param]] = sub

        solution = {symbolic.pystr_to_symbolic(k): v for k, v in solution.items()}
        return solution
