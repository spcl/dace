""" Map Fission transformation. """

from copy import deepcopy as dcpy
from collections import defaultdict
from dace import dtypes, registry, symbolic, sdfg as sd, memlet as mm, subsets
from dace.graph import nodes, nxutil, labeling
from dace.graph.graph import OrderedDiGraph
from dace.sdfg import replace
from dace.transformation import pattern_matching
from typing import List, Tuple


@registry.autoregister_params(singlestate=True)
class MapFission(pattern_matching.Transformation):
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
    _map_entry = nodes.EntryNode()
    _nested_sdfg = nodes.NestedSDFG("", OrderedDiGraph(), set(), set())

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(MapFission._map_entry, ),
            nxutil.node_path_graph(
                MapFission._map_entry,
                MapFission._nested_sdfg,
            )
        ]

    @staticmethod
    def _components(
            subgraph: sd.SubgraphView) -> List[Tuple[nodes.Node, nodes.Node]]:
        """
        Returns the list of tuples non-array components in this subgraph.
        Each element in the list is a 2 tuple of (input node, output node) of
        the component.
        """
        graph = (subgraph
                 if isinstance(subgraph, sd.SDFGState) else subgraph.graph)
        sdict = subgraph.scope_dict(node_to_children=True)
        ns = [(n, graph.exit_nodes(n)[0])
              if isinstance(n, nodes.EntryNode) else (n, n)
              for n in sdict[None]
              if isinstance(n, (nodes.CodeNode, nodes.EntryNode))]

        return ns

    @staticmethod
    def _border_arrays(components, subgraph, sources, sinks):
        """ Returns a set of array names that are local to the fission
            subgraph. """
        result = set()
        # In subgraphs, transient source/sink nodes (that do not come from
        # outside the map) are also border arrays
        use_sources_sinks = not isinstance(subgraph, sd.SDFGState)

        for component_in, component_out in components:
            for e in subgraph.in_edges(component_in):
                if (isinstance(e.src, nodes.AccessNode)
                        and (use_sources_sinks or e.src not in sources)):
                    result.add(e.src.data)
            for e in subgraph.out_edges(component_out):
                if (isinstance(e.dst, nodes.AccessNode)
                        and (use_sources_sinks or e.dst not in sinks)):
                    result.add(e.dst.data)

        return result

    @staticmethod
    def _internal_border_arrays(components, subgraph):
        """ Returns the set of border arrays that appear between computational
            components (i.e., without sources and sinks). """
        inputs = set()
        outputs = set()

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

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_node = graph.node(candidate[MapFission._map_entry])
        nsdfg_node = None
        if expr_index == 0:  # Map with subgraph
            subgraphs = [
                graph.scope_subgraph(
                    map_node, include_entry=False, include_exit=False)
            ]
        else:  # Map with nested SDFG
            nsdfg_node = graph.node(candidate[MapFission._nested_sdfg])
            # Make sure there are no other internal nodes in the map
            if len(set(e.dst for e in graph.out_edges(map_node))) > 1:
                return False
            subgraphs = list(nsdfg_node.sdfg.nodes())

        # Test subgraphs
        for sg in subgraphs:
            components = MapFission._components(sg)
            snodes = sg.nodes()
            sources = sg.source_nodes()
            sinks = sg.sink_nodes()
            # Test that the subgraphs have more than one computational component
            if len(snodes) > 0 and len(components) <= 1:
                return False

            # Test that the components are connected by transients that are not
            # used anywhere else
            border_arrays = MapFission._border_arrays(components, sg, sources,
                                                      sinks)

            # Fail if there are arrays inside the map that are not a direct
            # output of a computational component
            # TODO(later): Support this case? Ambiguous array sizes and memlets
            external_arrays = (
                border_arrays - MapFission._internal_border_arrays(
                    components, sg))
            if len(external_arrays) > 0:
                return False

            # 1. In nested SDFGs and subgraphs, ensure none of the border
            #    values are non-transients
            for array in border_arrays:
                if expr_index == 0:
                    ndesc = sdfg.arrays[array]
                else:
                    ndesc = nsdfg_node.sdfg.arrays[array]

                if ndesc.transient is False:
                    return False

            # 2. In subgraphs, make sure transients are not used/allocated
            #    in other scopes or states
            if expr_index == 0:
                # Find all nodes not in subgraph
                not_subgraph = set(
                    n.data for n in graph.nodes()
                    if n not in snodes and isinstance(n, nodes.AccessNode))
                not_subgraph.update(
                    set(n.data for s in sdfg.nodes() if s != graph
                        for n in s.nodes() if isinstance(n, nodes.AccessNode)))

                for _, component_out in components:
                    for e in sg.out_edges(component_out):
                        if isinstance(e.dst, nodes.AccessNode):
                            if e.dst.data in not_subgraph:
                                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.node(candidate[MapFission._map_entry])
        return map_entry.map.label

    def apply(self, sdfg: sd.SDFG):
        graph: sd.SDFGState = sdfg.nodes()[self.state_id]
        map_entry = graph.node(self.subgraph[MapFission._map_entry])
        map_exit = graph.exit_nodes(map_entry)[0]

        # Obtain subgraph to perform fission to
        if self.expr_index == 0:  # Map with subgraph
            subgraphs = [(graph,
                          graph.scope_subgraph(
                              map_entry,
                              include_entry=False,
                              include_exit=False))]
            parent = sdfg
        else:  # Map with nested SDFG
            nsdfg_node: nodes.NestedSDFG = graph.node(
                self.subgraph[MapFission._nested_sdfg])
            subgraphs = [(state, state) for state in nsdfg_node.sdfg.nodes()]
            parent = nsdfg_node.sdfg

        # Get map information
        outer_map: nodes.Map = map_entry.map
        mapsize = outer_map.range.size()

        for state, subgraph in subgraphs:
            components = MapFission._components(subgraph)
            sources = subgraph.source_nodes()
            sinks = subgraph.sink_nodes()

            # Collect external edges for source/sink nodes
            edge_to_outer = {}
            for node in sources:
                for edge in state.in_edges(node):
                    path = state.memlet_path(edge)
                    eindex = path.index(edge)
                    edge_to_outer[edge] = path[eindex - 1]
            for node in sinks:
                for edge in state.out_edges(node):
                    path = state.memlet_path(edge)
                    eindex = path.index(edge)
                    edge_to_outer[edge] = path[eindex + 1]

            # Collect all border arrays and code->code edges
            arrays = MapFission._border_arrays(components, subgraph, sources,
                                               sinks)
            scalars = defaultdict(list)
            for _, component_out in components:
                for e in subgraph.out_edges(component_out):
                    if isinstance(e.dst, nodes.CodeNode):
                        scalars[e.data.data].append(e)

            # Create new arrays for scalars
            for scalar, edges in scalars.items():
                desc = parent.arrays[scalar]
                name, newdesc = parent.add_temp_transient(
                    mapsize,
                    desc.dtype,
                    desc.storage,
                    toplevel=desc.toplevel,
                    debuginfo=desc.debuginfo,
                    allow_conflicts=desc.allow_conflicts)

                # Add extra nodes in component boundaries
                for edge in edges:
                    anode = state.add_access(name)
                    state.add_edge(
                        edge.src, edge.src_conn, anode, None,
                        mm.Memlet(
                            name, outer_map.range.num_elements(),
                            subsets.Range.from_string(','.join(
                                outer_map.params)), 1))
                    state.add_edge(
                        anode, None, edge.dst, edge.dst_conn,
                        mm.Memlet(
                            name, outer_map.range.num_elements(),
                            subsets.Range.from_string(','.join(
                                outer_map.params)), 1))
                    state.remove_edge(edge)

            # Add extra maps around components
            new_map_entries = []
            for component_in, component_out in components:
                me, mx = state.add_map(
                    outer_map.label + '_fission',
                    [(p, '0:1') for p in outer_map.params],
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
                    if component_in in sources:
                        state.add_edge(edge_to_outer[e].src,
                                       edge_to_outer[e].src_conn, me, None,
                                       dcpy(edge_to_outer[e].data))
                    else:
                        state.add_edge(e.src, e.src_conn, me, None,
                                       dcpy(e.data))
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.in_degree(component_in) == 0:
                    state.add_edge(me, None, component_in, None,
                                   mm.EmptyMemlet())

                for e in state.out_edges(component_out):
                    state.add_edge(e.src, e.src_conn, mx, None, dcpy(e.data))
                    # Reconnect inner edges at sink directly to external nodes
                    if component_out in sinks:
                        state.add_edge(mx, None, edge_to_outer[e].dst,
                                       edge_to_outer[e].dst_conn,
                                       dcpy(edge_to_outer[e].data))
                    else:
                        state.add_edge(mx, None, e.dst, e.dst_conn,
                                       dcpy(e.data))
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.out_degree(component_out) == 0:
                    state.add_edge(component_out, None, mx, None,
                                   mm.EmptyMemlet())
            # Connect other sources/sinks not in components (access nodes)
            # directly to external nodes
            for node in sources:
                if isinstance(node, nodes.AccessNode):
                    for edge in state.in_edges(node):
                        outer_edge = edge_to_outer[edge]
                        memlet = dcpy(edge.data)
                        memlet.subset = subsets.Range(outer_map.range.ranges +
                                                      memlet.subset.ranges)
                        state.add_edge(outer_edge.src, outer_edge.src_conn,
                                       edge.dst, edge.dst_conn, memlet)
            for node in sinks:
                if isinstance(node, nodes.AccessNode):
                    for edge in state.out_edges(node):
                        outer_edge = edge_to_outer[edge]
                        state.add_edge(edge.src, edge.src_conn, outer_edge.dst,
                                       outer_edge.dst_conn,
                                       dcpy(outer_edge.data))

            # Augment arrays by prepending map dimensions
            for array in arrays:
                desc = parent.arrays[array]
                for sz in reversed(mapsize):
                    desc.strides = [desc.total_size] + list(desc.strides)
                    desc.total_size = desc.total_size * sz

                desc.shape = mapsize + list(desc.shape)
                desc.offset = [0] * len(mapsize) + list(desc.offset)

            # Fill scope connectors so that memlets can be tracked below
            state.fill_scope_connectors()

            # TODO: nodes to outer transients broken in nested SDFGS

            # Fill in memlet trees for border transients
            # NOTE: Memlet propagation should run to correct the outer edges
            for node in subgraph.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in arrays:
                    for edge in state.all_edges(node):
                        for e in state.memlet_tree(edge):
                            # Prepend map dimensions to memlet
                            e.data.subset = subsets.Range(
                                [(d, d, 1) for d in outer_map.params] +
                                e.data.subset.ranges)

            # Re-introduce dynamic range map memlets
            for edge in sd.dynamic_map_inputs(graph, map_entry):
                if self.expr_index == 0:  # Subgraph
                    for me in new_map_entries:
                        state.add_edge(edge.src, edge.src_conn, me,
                                       edge.dst_conn, dcpy(edge.data))
                else:  # TODO: Add connectors to Nested SDFG
                    raise NotImplementedError

        # If nested SDFG, reconnect nodes around map
        if self.expr_index == 1:
            for edge in graph.in_edges(map_entry):
                if not edge.dst_conn or not edge.dst_conn.startswith('IN_'):
                    continue
                # Find matching edge inside map
                inner_edge = next(
                    e for e in graph.out_edges(map_entry)
                    if e.src_conn and e.src_conn[4:] == edge.dst_conn[3:])
                graph.add_edge(edge.src, edge.src_conn, nsdfg_node,
                               inner_edge.dst_conn, dcpy(edge.data))

            for edge in graph.out_edges(map_exit):
                # Find matching edge inside map
                inner_edge = next(
                    e for e in graph.in_edges(map_exit)
                    if e.dst_conn[3:] == edge.src_conn[4:])
                graph.add_edge(nsdfg_node, inner_edge.src_conn, edge.dst,
                               edge.dst_conn, dcpy(edge.data))

        # Remove outer map
        graph.remove_nodes_from([map_entry, map_exit])
