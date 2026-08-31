# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Map Fission transformation. """

from copy import deepcopy as dcpy
from collections import defaultdict
from functools import reduce
from dace import sdfg as sd, memlet as mm, subsets, data as dt
from dace.properties import CodeBlock
from dace.sdfg import nodes, graph as gr
from dace.sdfg import utils as sdutil
from dace.sdfg.propagation import propagate_memlets_state, propagate_subset
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.symbolic import pystr_to_symbolic
from dace.transformation import transformation, helpers
from typing import List, Optional, Tuple


def _substitute_map_range(subset: subsets.Range, params: List[str], rng: subsets.Range) -> Optional[subsets.Range]:
    """
    Replaces the map parameters in ``subset`` by the range of values they take.

    Memlet propagation falls back to a bounding box as soon as the map has a non-unit step, which
    turns a strided gather into a contiguous copy. Where every dimension is a point access that is
    affine in a single parameter with a positive integer multiplier, the exact strided range can be
    derived instead, which keeps the number of elements on both sides of a copy equal.

    :param subset: The subset to substitute into, expressed in terms of ``params``.
    :param params: The map parameters.
    :param rng: The range the map parameters iterate over.
    :return: The substituted subset, or ``None`` if it cannot be derived exactly.
    """
    symbolic_params = [pystr_to_symbolic(p) for p in params]
    result = []
    for rb, re, rs in subset.ndrange():
        rb, re, rs = (pystr_to_symbolic(v) for v in (rb, re, rs))
        used = [(i, p) for i, p in enumerate(symbolic_params) if p in (rb.free_symbols | re.free_symbols)]
        if not used:
            result.append((rb, re, rs))
            continue
        if len(used) > 1 or rb != re or rs != 1:
            return None
        pind, param = used[0]
        # Match an affine access ``mult * param + addition``
        poly = rb.as_poly(param)
        if poly is None or poly.degree() > 1:
            return None
        mult = poly.coeff_monomial(param)
        addition = poly.coeff_monomial(1)
        if not mult.is_Integer or mult <= 0:
            return None
        map_rb, map_re, map_rs = rng[pind]
        # The range's own end is kept rather than the last value the parameter actually takes. Both
        # describe the same set of elements, but this form makes the element count come out as the
        # map's own size expression, which symbolic comparisons against the augmented transient can
        # then match without having to reason about the ceiling division.
        result.append((mult * map_rb + addition, mult * map_re + addition, mult * map_rs))
    return subsets.Range(result)


@transformation.explicit_cf_compatible
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
    def _border_arrays(sdfg: sd.SDFG, parent, subgraph):
        """ Returns a set of array names that are local to the fission
            subgraph. """
        nested = isinstance(parent, sd.SDFGState)
        schildren = subgraph.scope_children()
        subset = gr.SubgraphView(parent, schildren[None])
        if nested:
            # Views are marked transient but do not own their storage: they alias a container that
            # already spans every iteration of the map. Giving them the per-iteration extent that
            # border transients get would grow the descriptor past the window its ``views`` edge
            # binds, so they are not border arrays.
            return set(node.data for node in subset.nodes() if isinstance(node, nodes.AccessNode)
                       and sdfg.arrays[node.data].transient and not isinstance(sdfg.arrays[node.data], dt.View))
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
            nsdfg_node = dcpy(self.nested_sdfg)
            # Make sure there are no other internal nodes in the map
            if len(set(e.dst for e in graph.out_edges(map_node))) > 1:
                return False

            # Get NestedSDFG control flow components
            nsdfg_node.sdfg.reset_cfg_list()

            # Fissioning a component across a conditional needs the branch
            # condition replicated into each fissioned map, currently not supported.
            if any(
                    isinstance(cfg, ConditionalBlock)
                    for cfg in nsdfg_node.sdfg.all_control_flow_regions(recursive=True)):
                return False

            if len(nsdfg_node.sdfg.nodes()) == 1:
                child = nsdfg_node.sdfg.nodes()[0]
                conditions: List[CodeBlock] = []
                if isinstance(child, LoopRegion):
                    conditions.append(child.loop_condition)
                elif isinstance(child, ConditionalBlock):
                    for c, _ in child.branches:
                        if c is not None:
                            conditions.append(c)
                for cond in conditions:
                    if any(p in cond.get_free_symbols() for p in map_node.map.params):
                        return False
                    for s in cond.get_free_symbols():
                        for e in graph.edges_by_connector(self.nested_sdfg, s):
                            if any(p in e.data.free_symbols for p in map_node.map.params):
                                return False
                    if any(p in cond.get_free_symbols() for p in map_node.map.params):
                        return False
            # Reject if any interstate edge inside the nested SDFG has an
            # assignment that depends on the map iterator, either directly or
            # through a nested-SDFG input connector whose incoming memlet
            # subset references a map parameter. Such assignments cannot be
            # safely hoisted out of the fissioned maps.
            map_params = set(map_node.map.params)
            inputs_dep_on_map = set()
            for e in graph.out_edges(map_node):
                if e.dst is self.nested_sdfg and e.dst_conn is not None and e.data.subset is not None:
                    if any(str(s) in map_params for s in e.data.subset.free_symbols):
                        inputs_dep_on_map.add(e.dst_conn)
            for ise in nsdfg_node.sdfg.all_interstate_edges():
                assign_free = set()
                for expr in ise.data.assignments.values():
                    try:
                        assign_free.update(str(s) for s in pystr_to_symbolic(expr).free_symbols)
                    except Exception:
                        pass
                if assign_free & map_params:
                    return False
                if assign_free & inputs_dep_on_map:
                    return False

            helpers.nest_sdfg_control_flow(nsdfg_node.sdfg)

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
                    set(n.data for s in sdfg.states() if s != graph for n in s.nodes()
                        if isinstance(n, nodes.AccessNode)))

                for _, component_out in components:
                    for e in sg.out_edges(component_out):
                        if isinstance(e.dst, nodes.AccessNode):
                            if e.dst.data in not_subgraph:
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
            helpers.nest_sdfg_control_flow(nsdfg_node.sdfg)
            subgraphs = [(state, state) for state in nsdfg_node.sdfg.nodes()]
            parent = nsdfg_node.sdfg
            parent_sdfg = parent.parent_sdfg
        modified_arrays = set()
        scalar_like_arrays = set()

        # Get map information
        outer_map: nodes.Map = map_entry.map
        # Border-transient extent equals the iteration count per dimension.
        # Memlets that index border transients are normalized to
        # `(p - iMin) / step` so the squeezed array remains in-bounds for
        # strided maps. Symbolic steps are assumed non-negative.
        mapsize = outer_map.range.size()
        squeezed_idx = [(pystr_to_symbolic(p) - iMin) / step
                        for p, (iMin, _iMax, step) in zip(outer_map.params, outer_map.range.ranges)]

        # Add new symbols from outer map to nested SDFG
        # Add new symbols also from the adjacent edge subsets and the data descriptors they carry.
        if self.expr_index == 1:
            map_syms = outer_map.range.free_symbols
            for edge in graph.out_edges(map_entry):
                if edge.data.data:
                    map_syms.update(edge.data.subset.free_symbols)
                if edge.data.data in parent_sdfg.arrays:
                    map_syms.update(parent_sdfg.arrays[edge.data.data].free_symbols)
            for edge in graph.in_edges(map_exit):
                if edge.data.data:
                    map_syms.update(edge.data.subset.free_symbols)
                if edge.data.data in parent_sdfg.arrays:
                    map_syms.update(parent_sdfg.arrays[edge.data.data].free_symbols)
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

            # Enclosing scope of the fissioned map (``None`` if top-level).
            # The new per-component fission maps live in this scope, so
            # boundary edges are rewired through it instead of through the about-to-be-removed map.
            parent_entry = state.entry_node(map_entry) if self.expr_index == 0 else None
            parent_exit = state.exit_node(parent_entry) if parent_entry is not None else None

            # For each boundary edge, record the edge just outside the
            # original map so the new per-component maps can reconnect there.
            # For a data edge that is its memlet-path neighbour
            # (``path[eindex - 1]`` at entry, ``path[eindex + 1]`` at exit).
            # An empty dependency edge has a one-element path and thus no
            # outside edge: store ``None``.
            edge_to_outer = {}
            edge_to_outer = {}
            for edge in external_edges_entry:
                if self.expr_index == 0:
                    # Subgraphs use the corresponding outer map edges
                    path = state.memlet_path(edge)
                    eindex = path.index(edge)
                    edge_to_outer[edge] = path[eindex - 1] if eindex > 0 else None
                else:
                    outer_edge = next((e for e in graph.in_edges(nsdfg_node) if e.dst_conn == edge.src.data), None)
                    if outer_edge is None:
                        outer_edge = next(e for e in graph.out_edges(nsdfg_node) if e.src_conn == edge.src.data)
                    edge_to_outer[edge] = outer_edge

            for edge in external_edges_exit:
                if self.expr_index == 0:
                    path = state.memlet_path(edge)
                    eindex = path.index(edge)
                    edge_to_outer[edge] = path[eindex + 1] if eindex + 1 < len(path) else None
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
                    sbs = subsets.Range([(idx, idx, 1) for idx in squeezed_idx])
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
                                       unroll=outer_map.unroll)

                # Add dynamic input connectors
                for conn in map_entry.in_connectors:
                    if not conn.startswith('IN_'):
                        me.add_in_connector(conn)

                me.map.range = dcpy(outer_map.range)
                new_map_entries.append(me)

                # Reconnect edges through new map
                conn_idx = 0
                for e in state.in_edges(component_in):
                    if e.data.data:
                        in_conn = f"IN_{conn_idx}"
                        out_conn = f"OUT_{conn_idx}"
                        conn_idx += 1
                        me.add_in_connector(in_conn)
                        me.add_out_connector(out_conn)
                    else:
                        in_conn = None
                        out_conn = None
                    state.add_edge(me, out_conn, e.dst, e.dst_conn, dcpy(e.data))
                    # Reconnect inner edges at source directly to external nodes
                    if self.expr_index == 0 and e in external_edges_entry:
                        outer = edge_to_outer[e]
                        if outer is None:
                            # Empty dependency edge: keep the
                            # new map inside the enclosing scope via an empty
                            # edge instead of aliasing the removed map_entry.
                            if parent_entry is not None:
                                state.add_edge(parent_entry, None, me, None, mm.Memlet())
                        else:
                            state.add_edge(outer.src, outer.src_conn, me, in_conn, dcpy(outer.data))
                    else:
                        state.add_edge(e.src, e.src_conn, me, in_conn, dcpy(e.data))
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.in_degree(component_in) == 0:
                    state.add_edge(me, None, component_in, None, mm.Memlet())

                conn_idx = 0
                for e in state.out_edges(component_out):
                    if e.data.data:
                        in_conn = f"IN_{conn_idx}"
                        out_conn = f"OUT_{conn_idx}"
                        conn_idx += 1
                        mx.add_in_connector(in_conn)
                        mx.add_out_connector(out_conn)
                    else:
                        in_conn = None
                        out_conn = None
                    state.add_edge(e.src, e.src_conn, mx, in_conn, dcpy(e.data))
                    # Reconnect inner edges at sink directly to external nodes
                    if self.expr_index == 0 and e in external_edges_exit:
                        outer = edge_to_outer[e]
                        if outer is None:
                            # Empty dependency edge: keep
                            # the new map inside the enclosing scope via an
                            # empty edge instead of aliasing the removed map_exit.
                            if parent_exit is not None:
                                state.add_edge(mx, None, parent_exit, None, mm.Memlet())
                        else:
                            state.add_edge(mx, out_conn, outer.dst, outer.dst_conn, dcpy(outer.data))
                    else:
                        state.add_edge(mx, out_conn, e.dst, e.dst_conn, dcpy(e.data))
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.out_degree(component_out) == 0:
                    state.add_edge(component_out, None, mx, None, mm.Memlet())
            # Connect other sources/sinks not in components (access nodes)
            # directly to external nodes. These edges end up outside the new Map scopes, which is
            # recorded so that the memlets filled in below do not refer to the map parameters.
            outside_border_edges = set()
            if self.expr_index == 0:
                for node in sources:
                    if isinstance(node, nodes.AccessNode):
                        for edge in state.in_edges(node):
                            outer_edge = edge_to_outer.get(edge)
                            if outer_edge is None:  # No outer feeder: nothing to rewire.
                                continue
                            # The added map dimensions are filled in below, where it is known which
                            # side of the memlet belongs to the augmented container and whether that
                            # container is scalar-like.
                            new_edge = state.add_edge(outer_edge.src, outer_edge.src_conn, edge.dst, edge.dst_conn,
                                                      dcpy(edge.data))
                            outside_border_edges.add(new_edge)

                for node in sinks:
                    if isinstance(node, nodes.AccessNode):
                        for edge in state.out_edges(node):
                            outer_edge = edge_to_outer.get(edge)
                            if outer_edge is None:  # No outer consumer: nothing to rewire.
                                continue
                            new_edge = state.add_edge(edge.src, edge.src_conn, outer_edge.dst, outer_edge.dst_conn,
                                                      dcpy(outer_edge.data))
                            outside_border_edges.add(new_edge)

            # Augment arrays by prepending map dimensions
            for array in arrays:
                if array in modified_arrays:
                    continue
                desc = parent.arrays[array]
                # Treat scalars and length-1 arrays as "scalar-like": their single
                # degenerate dimension is replaced by the map dims rather than
                # prepended, so the result is shape [extent] rather than
                # [extent, 1] (which produces zero-stride aliasing).
                scalar_like = (isinstance(desc, dt.Scalar)
                               or (isinstance(desc, dt.Array) and len(desc.shape) == 1 and desc.shape[0] == 1))
                if isinstance(desc, dt.Scalar):
                    desc = dt.Array(desc.dtype, desc.shape, desc.transient, desc.allow_conflicts, desc.storage,
                                    desc.location, desc.strides, desc.offset, False, desc.lifetime, 0, desc.debuginfo,
                                    desc.total_size, desc.start_offset)
                    parent.arrays[array] = desc

                if scalar_like:
                    desc.shape = list(mapsize)
                    strides = [1] * len(mapsize)
                    for i in range(len(mapsize) - 2, -1, -1):
                        strides[i] = strides[i + 1] * mapsize[i + 1]
                    desc.strides = strides
                    desc.total_size = reduce(lambda a, b: a * b, mapsize, 1)
                    desc.offset = [0] * len(mapsize)
                    scalar_like_arrays.add(array)
                else:
                    for sz in reversed(mapsize):
                        desc.strides = [desc.total_size] + list(desc.strides)
                        desc.total_size = desc.total_size * sz
                    desc.shape = list(mapsize) + list(desc.shape)
                    desc.offset = [0] * len(mapsize) + list(desc.offset)
                modified_arrays.add(array)

            # Fill scope connectors so that memlets can be tracked below
            state.fill_scope_connectors()

            # Correct connectors and memlets in nested SDFGs to account for
            # missing outside map
            if self.expr_index == 1:

                # NOTE: In the following scope dictionary, we mark the new MapEntries as existing in their own scope.
                # This makes it easier to detect edges that are outside the new Map scopes (after MapFission).
                scope_dict = state.scope_dict()
                for k, v in scope_dict.items():
                    if isinstance(k, nodes.MapEntry) and k in new_map_entries and v is None:
                        scope_dict[k] = k

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
                        # If the two descriptors are already identical, the nested SDFG has been
                        # integrated into its parent (see ``dace.sdfg.dealias``) and its memlets are
                        # expressed in the outer coordinate system already. Widening them against the
                        # outer edge would then apply the outer edge's offset a second time.
                        already_integrated = desc.is_equivalent(outer_desc)
                        if isinstance(desc, dt.Scalar):
                            parent.arrays[node.data] = dcpy(outer_desc)
                            desc = parent.arrays[node.data]
                            desc.transient = False
                        elif isinstance(desc, dt.Array):
                            desc.shape = outer_desc.shape
                            desc.strides = outer_desc.strides
                            desc.total_size = outer_desc.total_size

                        # Inside the nested SDFG, offset all memlets to include
                        # the offsets from within the map.
                        # NOTE: Relies on propagation to fix outer memlets
                        for internal_edge in state.all_edges(node):
                            for e in state.memlet_tree(internal_edge):
                                if not already_integrated:
                                    e.data.subset.offset(desc.offset, False)
                                    e.data.subset = helpers.unsqueeze_memlet(e.data, outer_edge.data).subset
                                # NOTE: If the edge is outside of the new Map scope, then try to propagate it. This is
                                # needed for edges directly connecting AccessNodes, because the standard memlet
                                # propagation will stop at the first AccessNode outside the Map scope. For example, see
                                # `test.transformations.mapfission_test.MapFissionTest.test_array_copy_outside_scope`.
                                if not (scope_dict[e.src] and scope_dict[e.dst]):
                                    outside_border_edges.add(e)
                                    new_subset = _substitute_map_range(e.data.subset, outer_map.params, outer_map.range)
                                    if new_subset is None:
                                        e.data = propagate_subset([e.data], desc, outer_map.params, outer_map.range)
                                    else:
                                        e.data.subset = new_subset

                        # Only after offsetting memlets we can modify the
                        # overall offset
                        if isinstance(desc, dt.Array):
                            desc.offset = outer_desc.offset

            # Fill in memlet trees for border transients
            # NOTE: Memlet propagation should run to correct the outer edges
            # NOTE: Edges rewired around the new Map scopes (see ``outside_border_edges``) sit
            # outside them and therefore cannot refer to the map parameters; they have to span the
            # whole extent that was added to the transient instead.
            full_map_ranges = [(0, sz - 1, 1) for sz in mapsize]
            for node in subgraph.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in arrays:
                    is_scalar_like = node.data in scalar_like_arrays
                    for edge in state.all_edges(node):
                        for e in state.memlet_tree(edge):
                            # Prepend map dimensions to memlet
                            # NOTE: Do this only for the subset corresponding to `node.data`. If the edge is copying
                            # to/from another AccessNode, the other data may not need extra dimensions. For example, see
                            # `test.transformations.mapfission_test.MapFissionTest.test_array_copy_outside_scope`.
                            if e in outside_border_edges:
                                map_ranges = full_map_ranges
                                # The external container on the other side of the copy did not gain
                                # any dimension, but its subset may still name the map parameters,
                                # which are out of scope here. Propagate it over the outer range.
                                if e.data.data != node.data and e.data.subset is not None:
                                    new_subset = _substitute_map_range(e.data.subset, outer_map.params, outer_map.range)
                                    if new_subset is None:
                                        new_subset = propagate_subset([e.data], parent.arrays[e.data.data],
                                                                      outer_map.params, outer_map.range).subset
                                    e.data.subset = new_subset
                            else:
                                map_ranges = [(idx, idx, 1) for idx in squeezed_idx]
                            if e.data.data == node.data:
                                if e.data.subset:
                                    if is_scalar_like:
                                        e.data.subset = subsets.Range(map_ranges)
                                    else:
                                        e.data.subset = subsets.Range(map_ranges + e.data.subset.ranges)
                            else:
                                if e.data.other_subset:
                                    if is_scalar_like:
                                        e.data.other_subset = subsets.Range(map_ranges)
                                    else:
                                        e.data.other_subset = subsets.Range(map_ranges + e.data.other_subset.ranges)

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
                for inner_edge in graph.out_edges_by_connector(map_entry, f"OUT_{edge.dst_conn[3:]}"):
                    graph.add_edge(edge.src, edge.src_conn, nsdfg_node, inner_edge.dst_conn, dcpy(edge.data))

            for edge in graph.out_edges(map_exit):
                # Modify edge coming out of nested SDFG to include entire array
                desc = sdfg.arrays[edge.data.data]
                edge.data.subset = subsets.Range.from_array(desc)

                # Find matching edge inside map
                for inner_edge in graph.in_edges_by_connector(map_exit, f"IN_{edge.src_conn[4:]}"):
                    graph.add_edge(nsdfg_node, inner_edge.src_conn, edge.dst, edge.dst_conn, dcpy(edge.data))

        # Remove outer map
        graph.remove_nodes_from([map_entry, map_exit])

        # NOTE: It is better to manually call memlet propagation here to ensure that all subsets are properly updated.
        # This can solve issues when, e.g., applying MapFission through `SDFG.apply_transformations_repeated`.
        propagate_memlets_state(sdfg, graph)
