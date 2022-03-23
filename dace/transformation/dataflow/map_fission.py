# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Map Fission transformation. """

from copy import deepcopy as dcpy
from collections import defaultdict
from dace import registry, sdfg as sd, memlet as mm, subsets, data as dt
from dace.sdfg import nodes, graph as gr
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import OrderedDiGraph
from dace.symbolic import pystr_to_symbolic
from dace.transformation import transformation, helpers
from typing import List, Optional, Tuple


class MapFission(transformation.SingleStateTransformation):
    """ Implements the MapFission transformation.
        Map fission refers to subsuming a map scope into its internal subgraph,
        essentially replicating the map into maps in all of its internal
        components. This also extends the dimensions of "border" transient
        arrays (i.e., those between the maps), in order to retain program
        semantics after fission.

        There are two cases that match map fission:
        1. A map with an arbitrary subgraph with more than one computational
           (i.e., non-access) node. The use of arrays connecting the
           computational nodes must be limited to the subgraph, and non
           transient arrays may not be used as "border" arrays.
        2. A map with one internal node that is a nested SDFG, in which
           each state matches the conditions of case (1).

        If a map has nested SDFGs in its subgraph, they are not considered in
        the case (1) above, and MapFission must be invoked again on the maps
        with the nested SDFGs in question.
    """
    map_entry = transformation.PatternNode(nodes.EntryNode)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.map_entry),
            sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg),
        ]

    @staticmethod
    def _components(subgraph: gr.SubgraphView) -> List[Tuple[nodes.Node, nodes.Node]]:
        """
        Returns the list of tuples non-array components in this subgraph.
        Each element in the list is a 2 tuple of (input node, output node) of
        the component.
        """
        graph = (subgraph if isinstance(subgraph, sd.SDFGState) else subgraph.graph)
        schildren = subgraph.scope_children()
        ns = [(n, graph.exit_node(n)) if isinstance(n, nodes.EntryNode) else (n, n) for n in schildren[None]
              if isinstance(n, (nodes.CodeNode, nodes.EntryNode))]

        return ns

    @staticmethod
    def _border_arrays(sdfg, parent, subgraph):
        """ Returns a set of array names that are local to the fission
            subgraph. """
        nested = isinstance(parent, sd.SDFGState)
        schildren = subgraph.scope_children()
        subset = gr.SubgraphView(parent, schildren[None])
        if nested:
            return set(node.data for node in subset.nodes()
                       if isinstance(node, nodes.AccessNode) and sdfg.arrays[node.data].transient)
        else:
            return set(node.data for node in subset.nodes() if isinstance(node, nodes.AccessNode))

    @staticmethod
    def _internal_border_arrays(total_components, subgraphs):
        """ Returns the set of border arrays that appear between computational
            components (i.e., without sources and sinks). """
        inputs = set()
        outputs = set()

        for components, subgraph in zip(total_components, subgraphs):
            for component_in, component_out in components:
                for e in subgraph.in_edges(component_in):
                    if isinstance(e.src, nodes.AccessNode):
                        inputs.add(e.src.data)
                for e in subgraph.out_edges(component_out):
                    if isinstance(e.dst, nodes.AccessNode):
                        outputs.add(e.dst.data)

        return inputs & outputs

    @staticmethod
    def _outside_map(node, scope_dict, entry_nodes):
        """ Returns True iff node is not in any of the scopes spanned by
            entry_nodes. """
        while scope_dict[node] is not None:
            if scope_dict[node] in entry_nodes:
                return False
            node = scope_dict[node]
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_node = self.map_entry
        nsdfg_node = None

        # If the map is dynamic-ranged, the resulting border arrays would be
        # dynamically sized
        if sd.has_dynamic_map_inputs(graph, map_node):
            return False

        if expr_index == 0:  # Map with subgraph
            subgraphs = [graph.scope_subgraph(map_node, include_entry=False, include_exit=False)]
        else:  # Map with nested SDFG
            nsdfg_node = self.nested_sdfg
            # Make sure there are no other internal nodes in the map
            if len(set(e.dst for e in graph.out_edges(map_node))) > 1:
                return False
            subgraphs = list(nsdfg_node.sdfg.nodes())

        # Test subgraphs
        border_arrays = set()
        total_components = []
        for sg in subgraphs:
            components = self._components(sg)
            snodes = sg.nodes()
            # Test that the subgraphs have more than one computational component
            if expr_index == 0 and len(snodes) > 0 and len(components) <= 1:
                return False

            # Test that the components are connected by transients that are not
            # used anywhere else
            border_arrays |= self._border_arrays(nsdfg_node.sdfg if expr_index == 1 else sdfg,
                                                 sg if expr_index == 1 else graph, sg)
            total_components.append(components)

            # In nested SDFGs and subgraphs, ensure none of the border
            # values are non-transients
            for array in border_arrays:
                if expr_index == 0:
                    ndesc = sdfg.arrays[array]
                else:
                    ndesc = nsdfg_node.sdfg.arrays[array]

                if ndesc.transient is False:
                    return False

            # In subgraphs, make sure transients are not used/allocated
            # in other scopes or states
            if expr_index == 0:
                # Find all nodes not in subgraph
                not_subgraph = set(n.data for n in graph.nodes() if n not in snodes and isinstance(n, nodes.AccessNode))
                not_subgraph.update(
                    set(n.data for s in sdfg.nodes() if s != graph for n in s.nodes()
                        if isinstance(n, nodes.AccessNode)))

                for _, component_out in components:
                    for e in sg.out_edges(component_out):
                        if isinstance(e.dst, nodes.AccessNode):
                            if e.dst.data in not_subgraph:
                                return False

        # Fail if there are arrays inside the map that are not a direct
        # output of a computational component
        # TODO(later): Support this case? Ambiguous array sizes and memlets
        external_arrays = (border_arrays - self._internal_border_arrays(total_components, subgraphs))
        if len(external_arrays) > 0:
            return False

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        nsdfg_node: Optional[nodes.NestedSDFG] = None

        # Obtain subgraph to perform fission to
        if self.expr_index == 0:  # Map with subgraph
            subgraphs = [(graph, graph.scope_subgraph(map_entry, include_entry=False, include_exit=False))]
            parent = sdfg
        else:  # Map with nested SDFG
            nsdfg_node = self.nested_sdfg
            subgraphs = [(state, state) for state in nsdfg_node.sdfg.nodes()]
            parent = nsdfg_node.sdfg
        modified_arrays = set()

        # Get map information
        outer_map: nodes.Map = map_entry.map
        mapsize = outer_map.range.size()

        # Add new symbols from outer map to nested SDFG
        if self.expr_index == 1:
            map_syms = outer_map.range.free_symbols
            for edge in graph.out_edges(map_entry):
                if edge.data.data:
                    map_syms.update(edge.data.subset.free_symbols)
            for edge in graph.in_edges(map_exit):
                if edge.data.data:
                    map_syms.update(edge.data.subset.free_symbols)
            for sym in map_syms:
                symname = str(sym)
                if symname in outer_map.params:
                    continue
                if symname not in nsdfg_node.symbol_mapping.keys():
                    nsdfg_node.symbol_mapping[symname] = sym
                    nsdfg_node.sdfg.symbols[symname] = graph.symbols_defined_at(nsdfg_node)[symname]

            # Remove map symbols from nested mapping
            for name in outer_map.params:
                if str(name) in nsdfg_node.symbol_mapping:
                    del nsdfg_node.symbol_mapping[str(name)]
                if str(name) in nsdfg_node.sdfg.symbols:
                    del nsdfg_node.sdfg.symbols[str(name)]

        for state, subgraph in subgraphs:
            components = MapFission._components(subgraph)
            sources = subgraph.source_nodes()
            sinks = subgraph.sink_nodes()

            # Collect external edges
            if self.expr_index == 0:
                external_edges_entry = list(state.out_edges(map_entry))
                external_edges_exit = list(state.in_edges(map_exit))
            else:
                external_edges_entry = [
                    e for e in subgraph.edges()
                    if (isinstance(e.src, nodes.AccessNode) and not nsdfg_node.sdfg.arrays[e.src.data].transient)
                ]
                external_edges_exit = [
                    e for e in subgraph.edges()
                    if (isinstance(e.dst, nodes.AccessNode) and not nsdfg_node.sdfg.arrays[e.dst.data].transient)
                ]

            # Map external edges to outer memlets
            edge_to_outer = {}
            for edge in external_edges_entry:
                if self.expr_index == 0:
                    # Subgraphs use the corresponding outer map edges
                    path = state.memlet_path(edge)
                    eindex = path.index(edge)
                    edge_to_outer[edge] = path[eindex - 1]
                else:
                    # Nested SDFGs use the internal map edges of the node
                    outer_edge = next(e for e in graph.in_edges(nsdfg_node) if e.dst_conn == edge.src.data)
                    edge_to_outer[edge] = outer_edge

            for edge in external_edges_exit:
                if self.expr_index == 0:
                    path = state.memlet_path(edge)
                    eindex = path.index(edge)
                    edge_to_outer[edge] = path[eindex + 1]
                else:
                    # Nested SDFGs use the internal map edges of the node
                    outer_edge = next(e for e in graph.out_edges(nsdfg_node) if e.src_conn == edge.dst.data)
                    edge_to_outer[edge] = outer_edge

            # Collect all border arrays and code->code edges
            arrays = MapFission._border_arrays(nsdfg_node.sdfg if self.expr_index == 1 else sdfg, state, subgraph)
            scalars = defaultdict(list)
            for _, component_out in components:
                for e in subgraph.out_edges(component_out):
                    if isinstance(e.dst, nodes.CodeNode):
                        scalars[e.data.data].append(e)

            # Create new arrays for scalars
            for scalar, edges in scalars.items():
                desc = parent.arrays[scalar]
                del parent.arrays[scalar]
                name, newdesc = parent.add_transient(scalar,
                                                     mapsize,
                                                     desc.dtype,
                                                     desc.storage,
                                                     lifetime=desc.lifetime,
                                                     debuginfo=desc.debuginfo,
                                                     allow_conflicts=desc.allow_conflicts,
                                                     find_new_name=True)

                # Add extra nodes in component boundaries
                for edge in edges:
                    anode = state.add_access(name)
                    sbs = subsets.Range.from_string(','.join(outer_map.params))
                    # Offset memlet by map range begin (to fit the transient)
                    sbs.offset([r[0] for r in outer_map.range], True)
                    state.add_edge(edge.src, edge.src_conn, anode, None,
                                   mm.Memlet.simple(name, sbs, num_accesses=outer_map.range.num_elements()))
                    state.add_edge(anode, None, edge.dst, edge.dst_conn,
                                   mm.Memlet.simple(name, sbs, num_accesses=outer_map.range.num_elements()))
                    state.remove_edge(edge)

            # Add extra maps around components
            new_map_entries = []
            for component_in, component_out in components:
                me, mx = state.add_map(outer_map.label + '_fission', [(p, '0:1') for p in outer_map.params],
                                       outer_map.schedule,
                                       unroll=outer_map.unroll,
                                       debuginfo=outer_map.debuginfo)

                # Add dynamic input connectors
                for conn in map_entry.in_connectors:
                    if not conn.startswith('IN_'):
                        me.add_in_connector(conn)

                me.map.range = dcpy(outer_map.range)
                new_map_entries.append(me)

                # Reconnect edges through new map
                for e in state.in_edges(component_in):
                    state.add_edge(me, None, e.dst, e.dst_conn, dcpy(e.data))
                    # Reconnect inner edges at source directly to external nodes
                    if self.expr_index == 0 and e in external_edges_entry:
                        state.add_edge(edge_to_outer[e].src, edge_to_outer[e].src_conn, me, None,
                                       dcpy(edge_to_outer[e].data))
                    else:
                        state.add_edge(e.src, e.src_conn, me, None, dcpy(e.data))
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.in_degree(component_in) == 0:
                    state.add_edge(me, None, component_in, None, mm.Memlet())

                for e in state.out_edges(component_out):
                    state.add_edge(e.src, e.src_conn, mx, None, dcpy(e.data))
                    # Reconnect inner edges at sink directly to external nodes
                    if self.expr_index == 0 and e in external_edges_exit:
                        state.add_edge(mx, None, edge_to_outer[e].dst, edge_to_outer[e].dst_conn,
                                       dcpy(edge_to_outer[e].data))
                    else:
                        state.add_edge(mx, None, e.dst, e.dst_conn, dcpy(e.data))
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.out_degree(component_out) == 0:
                    state.add_edge(component_out, None, mx, None, mm.Memlet())
            # Connect other sources/sinks not in components (access nodes)
            # directly to external nodes
            if self.expr_index == 0:
                for node in sources:
                    if isinstance(node, nodes.AccessNode):
                        for edge in state.in_edges(node):
                            outer_edge = edge_to_outer[edge]
                            memlet = dcpy(edge.data)
                            memlet.subset = subsets.Range(outer_map.range.ranges + memlet.subset.ranges)
                            state.add_edge(outer_edge.src, outer_edge.src_conn, edge.dst, edge.dst_conn, memlet)

                for node in sinks:
                    if isinstance(node, nodes.AccessNode):
                        for edge in state.out_edges(node):
                            outer_edge = edge_to_outer[edge]
                            state.add_edge(edge.src, edge.src_conn, outer_edge.dst, outer_edge.dst_conn,
                                           dcpy(outer_edge.data))

            # Augment arrays by prepending map dimensions
            for array in arrays:
                if array in modified_arrays:
                    continue
                desc = parent.arrays[array]
                if isinstance(desc, dt.Scalar):  # Scalar needs to be augmented to an array
                    desc = dt.Array(desc.dtype, desc.shape, desc.transient, desc.allow_conflicts, desc.storage,
                                    desc.location, desc.strides, desc.offset, False, desc.lifetime,
                                    0, desc.debuginfo, desc.total_size, desc.start_offset)
                    parent.arrays[array] = desc
                for sz in reversed(mapsize):
                    desc.strides = [desc.total_size] + list(desc.strides)
                    desc.total_size = desc.total_size * sz

                desc.shape = mapsize + list(desc.shape)
                desc.offset = [0] * len(mapsize) + list(desc.offset)
                modified_arrays.add(array)

            # Fill scope connectors so that memlets can be tracked below
            state.fill_scope_connectors()

            # Correct connectors and memlets in nested SDFGs to account for
            # missing outside map
            if self.expr_index == 1:
                to_correct = ([(e, e.src) for e in external_edges_entry] + [(e, e.dst) for e in external_edges_exit])
                corrected_nodes = set()
                for edge, node in to_correct:
                    if isinstance(node, nodes.AccessNode):
                        if node in corrected_nodes:
                            continue
                        corrected_nodes.add(node)

                        outer_edge = edge_to_outer[edge]
                        desc = parent.arrays[node.data]

                        # Modify shape of internal array to match outer one
                        outer_desc = sdfg.arrays[outer_edge.data.data]
                        if not isinstance(desc, dt.Scalar):
                            desc.shape = outer_desc.shape
                        if isinstance(desc, dt.Array):
                            desc.strides = outer_desc.strides
                            desc.total_size = outer_desc.total_size

                        # Inside the nested SDFG, offset all memlets to include
                        # the offsets from within the map.
                        # NOTE: Relies on propagation to fix outer memlets
                        for internal_edge in state.all_edges(node):
                            for e in state.memlet_tree(internal_edge):
                                e.data.subset.offset(desc.offset, False)
                                e.data.subset = helpers.unsqueeze_memlet(e.data, outer_edge.data, desc=sdfg.arrays[e.data.data]).subset

                        # Only after offsetting memlets we can modify the
                        # overall offset
                        if isinstance(desc, dt.Array):
                            desc.offset = outer_desc.offset

            # Fill in memlet trees for border transients
            # NOTE: Memlet propagation should run to correct the outer edges
            for node in subgraph.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in arrays:
                    for edge in state.all_edges(node):
                        for e in state.memlet_tree(edge):
                            # Prepend map dimensions to memlet
                            e.data.subset = subsets.Range([(pystr_to_symbolic(d) - r[0], pystr_to_symbolic(d) - r[0], 1)
                                                           for d, r in zip(outer_map.params, outer_map.range)] +
                                                          e.data.subset.ranges)

        # If nested SDFG, reconnect nodes around map and modify memlets
        if self.expr_index == 1:
            for edge in graph.in_edges(map_entry):
                if not edge.dst_conn or not edge.dst_conn.startswith('IN_'):
                    continue

                # Modify edge coming into nested SDFG to include entire array
                desc = sdfg.arrays[edge.data.data]
                edge.data.subset = subsets.Range.from_array(desc)
                edge.data.num_accesses = edge.data.subset.num_elements()

                # Find matching edge inside map
                inner_edge = next(e for e in graph.out_edges(map_entry)
                                  if e.src_conn and e.src_conn[4:] == edge.dst_conn[3:])
                graph.add_edge(edge.src, edge.src_conn, nsdfg_node, inner_edge.dst_conn, dcpy(edge.data))

            for edge in graph.out_edges(map_exit):
                # Modify edge coming out of nested SDFG to include entire array
                desc = sdfg.arrays[edge.data.data]
                edge.data.subset = subsets.Range.from_array(desc)

                # Find matching edge inside map
                inner_edge = next(e for e in graph.in_edges(map_exit) if e.dst_conn[3:] == edge.src_conn[4:])
                graph.add_edge(nsdfg_node, inner_edge.src_conn, edge.dst, edge.dst_conn, dcpy(edge.data))

        # Remove outer map
        graph.remove_nodes_from([map_entry, map_exit])
