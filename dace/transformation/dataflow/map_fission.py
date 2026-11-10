# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Map Fission transformation."""

from collections import defaultdict
from copy import deepcopy as dcpy
from functools import reduce

from dace import data as dt
from dace import memlet as mm
from dace import sdfg as sd
from dace import subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import graph as gr
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.propagation import propagate_memlets_state, propagate_subset
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.symbolic import pystr_to_symbolic
from dace.transformation import helpers, transformation


@transformation.explicit_cf_compatible
class MapFission(transformation.SingleStateTransformation):
    """Implements the MapFission transformation.
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
        # The pass explicitly calls memlet propagation on the fissioned subgraph,
        # so it does not need to be called again.
        return True

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.map_entry),
            sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg),
        ]

    @staticmethod
    def _components(subgraph: gr.SubgraphView) -> list[tuple[nodes.Node, nodes.Node]]:
        """
        Returns the list of tuples non-array components in this subgraph.
        Each element in the list is a 2 tuple of (input node, output node) of
        the component.
        """
        graph = subgraph if isinstance(subgraph, sd.SDFGState) else subgraph.graph
        schildren = subgraph.scope_children()
        ns = [
            (n, graph.exit_node(n)) if isinstance(n, nodes.EntryNode) else (n, n)
            for n in schildren[None]
            if isinstance(n, (nodes.CodeNode, nodes.EntryNode))
        ]

        return ns

    @staticmethod
    def reduction_nodes(
        graph: sd.SDFGState, nsdfg_node: nodes.NestedSDFG, map_exit: nodes.MapExit
    ) -> list[nodes.AccessNode]:
        """The per-iteration access nodes a nested-SDFG body writes and only the map exit reads, such as the
        ``nested SDFG -> access node -wcr-> map exit`` a reduction leaves the body through."""
        sdfg = graph.sdfg
        found = []
        for e in graph.out_edges(nsdfg_node):
            node = e.dst
            if not isinstance(node, nodes.AccessNode) or not sdfg.arrays[node.data].transient:
                continue
            if graph.in_degree(node) != 1 or any(o.dst is not map_exit for o in graph.out_edges(node)):
                continue
            if any(
                n is not node and isinstance(n, nodes.AccessNode) and n.data == node.data
                for state in sdfg.states()
                for n in state.nodes()
            ):
                continue
            found.append(node)
        return found

    @staticmethod
    def fold_reduction_nodes(graph: sd.SDFGState, nsdfg_node: nodes.NestedSDFG, map_exit: nodes.MapExit) -> None:
        """Write the map exit directly from the nested SDFG, with the memlet (and write-conflict resolution) the
        access node passed on, and drop the access node."""
        for node in MapFission.reduction_nodes(graph, nsdfg_node, map_exit):
            (feed,) = graph.in_edges(node)
            for e in graph.out_edges(node):
                graph.add_edge(nsdfg_node, feed.src_conn, map_exit, e.dst_conn, dcpy(e.data))
            graph.remove_node(node)
            del graph.sdfg.arrays[node.data]

    @staticmethod
    def _border_arrays(sdfg: sd.SDFG, parent, subgraph):
        """Returns a set of array names that are local to the fission
        subgraph."""
        nested = isinstance(parent, sd.SDFGState)
        schildren = subgraph.scope_children()
        subset = gr.SubgraphView(parent, schildren[None])
        if nested:
            # Views alias storage that already spans every iteration
            return {
                node.data
                for node in subset.nodes()
                if isinstance(node, nodes.AccessNode)
                and sdfg.arrays[node.data].transient
                and not isinstance(sdfg.arrays[node.data], dt.View)
            }
        else:
            return {node.data for node in subset.nodes() if isinstance(node, nodes.AccessNode)}

    @staticmethod
    def _internal_border_arrays(total_components, subgraphs):
        """Returns the set of border arrays that appear between computational
        components (i.e., without sources and sinks)."""
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
    def map_dimensions_lead(subset: subsets.Range, params: list[str]) -> bool:
        """Whether a copy whose other end is a widened border transient still pairs its dimensions in order.

        Fission prepends the map dimensions to the transient, so the other container's subset must index the map
        parameters, one per dimension and in map order, before any other dimension longer than one element.
        """
        tags: list[str] = []
        for dim, size in zip(subset.ranges, subset.size()):
            used = [p for p in params if p in symbolic.symlist(list(dim))]
            if len(used) > 1:
                return False
            if used:
                tags.append(used[0])
            elif size != 1:
                tags.append("")
        return tags[: len(params)] == params and all(tag == "" for tag in tags[len(params) :])

    @staticmethod
    def misordered_copies(
        state, edges, borders, params: list[str], boundary=(), placed=None
    ) -> list[gr.MultiConnectorEdge]:
        """The copies between a border transient and another container that would pair dimensions out of order once
        the transient is widened. ``edges`` are the candidate edges, ``boundary`` the fissioned map's own entry and
        exit (which lead to containers outside it), and ``placed(container, subset)`` gives the subset fission leaves a
        container with (a nested-SDFG connector takes its outer memlet's dimensions)."""
        found = []
        for e in edges:
            if e.data.is_empty() or not all(isinstance(n, nodes.AccessNode) or n in boundary for n in (e.src, e.dst)):
                continue
            ends = [(e.src, e.data.get_src_subset), (e.dst, e.data.get_dst_subset)]
            border = [n for n, _ in ends if isinstance(n, nodes.AccessNode) and n.data in borders]
            if len(border) != 1:
                continue
            ((other, subset_of),) = [(n, get) for n, get in ends if n is not border[0]]
            subset = subset_of(e, state)
            if subset is not None and placed is not None:
                subset = placed(other.data, subset)
            if subset is None or not MapFission.map_dimensions_lead(subset, params):
                found.append(e)
        return found

    @staticmethod
    def _copy_indices(state, edge) -> tuple[list, list, list] | None:
        """The ranges of an elementwise copy map for ``edge`` and the element each side reads or writes, or ``None``
        if the two subsets do not pair their non-unit dimensions in order."""
        src, dst = edge.data.get_src_subset(edge, state), edge.data.get_dst_subset(edge, state)
        if src is None or dst is None:
            return None
        src_sizes = [sz for sz in src.size() if sz != 1]
        if src_sizes != [sz for sz in dst.size() if sz != 1]:
            return None
        params = [f"__fission_copy_{i}" for i in range(len(src_sizes))]

        def element(subset):
            idx, k = [], 0
            for (begin, _end, step), size in zip(subset.ranges, subset.size()):
                if size == 1:
                    idx.append((begin, begin, 1))
                else:
                    i = begin + symbolic.pystr_to_symbolic(params[k]) * step
                    idx.append((i, i, 1))
                    k += 1
            return subsets.Range(idx)

        ranges = [(p, subsets.Range([(0, sz - 1, 1)])) for p, sz in zip(params, src_sizes)]
        return ranges, element(src), element(dst)

    @staticmethod
    def copy_as_map(state, edge) -> None:
        """Replaces the copy ``edge`` by an elementwise copy map, a component fission can replicate per iteration."""
        ranges, src_element, dst_element = MapFission._copy_indices(state, edge)
        path = state.memlet_path(edge)
        src_data, dst_data = path[0].src.data, path[-1].dst.data
        me, mx = state.add_map("fission_copy", {p: str(r) for p, r in ranges})
        tasklet = state.add_tasklet("fission_copy", {"__inp"}, {"__out"}, "__out = __inp")
        state.add_memlet_path(
            edge.src,
            me,
            tasklet,
            memlet=mm.Memlet(data=src_data, subset=src_element),
            src_conn=edge.src_conn,
            dst_conn="__inp",
        )
        state.add_memlet_path(
            tasklet,
            mx,
            edge.dst,
            memlet=mm.Memlet(data=dst_data, subset=dst_element, wcr=edge.data.wcr),
            src_conn="__out",
            dst_conn=edge.dst_conn,
        )
        state.remove_edge(edge)

    def _misordered_lifted_copies(self, graph, sdfg, nsdfg: sd.SDFG | None, states, border_arrays) -> list:
        """``(state, edge)`` for each copy fission would lift with its dimensions paired out of order."""
        map_node = self.map_entry
        params = list(map_node.map.params)
        if nsdfg is None:
            scope, boundary = graph.scope_dict(), (map_node, graph.exit_node(map_node))
            edges = [e for e in graph.scope_subgraph(map_node).edges() if scope[e.src] is map_node or e.src is map_node]
            return [(graph, e) for e in self.misordered_copies(graph, edges, border_arrays, params, boundary)]

        outer = {e.dst_conn: e.data for e in graph.in_edges(self.nested_sdfg)}
        outer |= {e.src_conn: e.data for e in graph.out_edges(self.nested_sdfg)}

        def placed(name: str, subset: subsets.Range) -> subsets.Range:
            if name not in outer or outer[name].is_empty():
                return subset
            inner_desc, outer_desc = nsdfg.arrays[name], sdfg.arrays[outer[name].data]
            if inner_desc.is_equivalent(outer_desc):
                return subset
            inner = mm.Memlet(data=name, subset=dcpy(subset))
            inner.subset.offset(inner_desc.offset, False)
            return helpers.unsqueeze_memlet(inner, outer[name]).subset

        found = []
        for st in states:
            scope = st.scope_dict()
            edges = [e for e in st.edges() if scope[e.src] is None]
            found += [(st, e) for e in self.misordered_copies(st, edges, border_arrays, params, placed=placed)]
        return found

    @staticmethod
    def remove_iteration_slice_views(nsdfg: sd.SDFG, params: list[str]) -> bool:
        """Folds every view in ``nsdfg`` whose window moves with the map parameters into the memlets that use it, as
        fission widens the containers it reaches but not such a window. Returns whether all of them were folded."""
        from dace.transformation.dataflow.redundant_array import RemoveSliceView  # Avoid import loop

        for state in nsdfg.states():
            for node in list(state.data_nodes()):
                if not isinstance(node.desc(nsdfg), dt.View):
                    continue
                view_edge = sdutil.get_view_edge(state, node)
                if view_edge is None or not set(params) & set(map(str, view_edge.data.free_symbols)):
                    continue
                if not RemoveSliceView.can_be_applied_to(nsdfg, view=node):
                    return False
                RemoveSliceView.apply_to(nsdfg, view=node, verify=False)
        return True

    @staticmethod
    def widen_integrated_connectors(nsdfg_node: nodes.NestedSDFG, state, widened: dict, idx: list, ptypes: dict):
        """Widens the connectors of ``nsdfg_node`` that share the coordinates of a container fission widened.

        ``widened`` maps each widened container to ``(old descriptor, new descriptor, scalar-like)``. A connector whose
        descriptor equals the old one indexes the container directly, so it takes the new descriptor and every memlet
        inside gets the map indices ``idx`` in front, recursively.
        """
        inner_sdfg = nsdfg_node.sdfg
        inner_widened = {}
        for e in state.all_edges(nsdfg_node):
            conn = e.dst_conn if e.dst is nsdfg_node else e.src_conn
            if e.data.data not in widened or conn is None or conn not in inner_sdfg.arrays or conn in inner_widened:
                continue
            old, new, scalar_like = widened[e.data.data]
            inner = inner_sdfg.arrays[conn]
            if inner.transient or not inner.is_equivalent(old):
                continue
            desc = dcpy(new)
            desc.transient = False
            inner_sdfg.arrays[conn] = desc
            inner_widened[conn] = (inner, desc, scalar_like)
        if not inner_widened:
            return
        for p, ptype in ptypes.items():
            if p not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[p] = p
            if p not in inner_sdfg.symbols:
                inner_sdfg.add_symbol(p, ptype)

        def widen(subset: subsets.Range, scalar_like: bool) -> subsets.Range:
            prefix = [(i, i, 1) for i in idx]
            return subsets.Range(prefix if scalar_like else prefix + subset.ranges)

        for inner_state in inner_sdfg.states():
            for e in inner_state.edges():
                if e.data.is_empty():
                    continue
                if e.data.data in inner_widened:
                    e.data.subset = widen(e.data.subset, inner_widened[e.data.data][2])
                elif e.data.other_subset is not None:
                    path = inner_state.memlet_path(e)
                    ends = {n.data for n in (path[0].src, path[-1].dst) if isinstance(n, nodes.AccessNode)}
                    other = ends & set(inner_widened)
                    if other:
                        e.data.other_subset = widen(e.data.other_subset, inner_widened[other.pop()][2])
            for node in inner_state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    MapFission.widen_integrated_connectors(node, inner_state, inner_widened, idx, ptypes)

    @staticmethod
    def _outside_map(node, scope_dict, entry_nodes):
        """Returns True iff node is not in any of the scopes spanned by
        entry_nodes."""
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
            # Make sure there are no other internal nodes in the map, but the access nodes it leaves through
            if len({e.dst for e in graph.out_edges(map_node)}) > 1:
                return False
            map_exit = graph.exit_node(map_node)
            body = set(graph.scope_children()[map_node]) - {map_exit, self.nested_sdfg}
            if body - set(self.reduction_nodes(graph, self.nested_sdfg, map_exit)):
                return False

            # Get NestedSDFG control flow components
            nsdfg_node.sdfg.reset_cfg_list()

            # Fissioning a component across a conditional needs the branch
            # condition replicated into each fissioned map, currently not supported.
            if any(
                isinstance(cfg, ConditionalBlock) for cfg in nsdfg_node.sdfg.all_control_flow_regions(recursive=True)
            ):
                return False

            if len(nsdfg_node.sdfg.nodes()) == 1:
                child = nsdfg_node.sdfg.nodes()[0]
                conditions: list[CodeBlock] = []
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
            if not self.remove_iteration_slice_views(nsdfg_node.sdfg, list(map_node.map.params)):
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
            border_arrays |= self._border_arrays(
                nsdfg_node.sdfg if expr_index == 1 else sdfg, sg if expr_index == 1 else graph, sg
            )
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
                not_subgraph = {n.data for n in graph.nodes() if n not in snodes and isinstance(n, nodes.AccessNode)}
                not_subgraph.update(
                    {n.data for s in sdfg.states() if s != graph for n in s.nodes() if isinstance(n, nodes.AccessNode)}
                )

                for _, component_out in components:
                    for e in sg.out_edges(component_out):
                        if isinstance(e.dst, nodes.AccessNode):
                            if e.dst.data in not_subgraph:
                                return False

        # A copy between a border transient and another container leaves the new maps whole; widening the transient
        # by the map dimensions in front must not pair them with other dimensions of that container. ``apply`` turns
        # such a copy into an elementwise copy map, which needs the two subsets to pair their dimensions in order
        nsdfg = nsdfg_node.sdfg if expr_index == 1 else None
        for state, e in self._misordered_lifted_copies(graph, sdfg, nsdfg, subgraphs, border_arrays):
            if self._copy_indices(state, e) is None:
                return False

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        nsdfg_node: nodes.NestedSDFG | None = None

        # Obtain subgraph to perform fission to
        if self.expr_index == 0:  # Map with subgraph
            subgraphs = [(graph, graph.scope_subgraph(map_entry, include_entry=False, include_exit=False))]
            parent = sdfg
        else:  # Map with nested SDFG
            nsdfg_node = self.nested_sdfg
            MapFission.fold_reduction_nodes(graph, nsdfg_node, map_exit)
            # Removing a node renumbers the state: bind the match to the nodes themselves
            self.subgraph.update({MapFission.map_entry: map_entry, MapFission.nested_sdfg: nsdfg_node})
            helpers.nest_sdfg_control_flow(nsdfg_node.sdfg)
            MapFission.remove_iteration_slice_views(nsdfg_node.sdfg, list(map_entry.map.params))
            subgraphs = [(state, state) for state in nsdfg_node.sdfg.nodes()]
            parent = nsdfg_node.sdfg
            parent_sdfg = parent.parent_sdfg
        # Copies fission cannot lift without pairing dimensions out of order are replicated per iteration instead
        states = [state for state, _ in subgraphs]
        border_arrays = set()
        for state, subgraph in subgraphs:
            border_arrays |= MapFission._border_arrays(
                parent if self.expr_index == 1 else sdfg, state if self.expr_index == 1 else graph, subgraph
            )
        nsdfg = parent if self.expr_index == 1 else None
        misordered = self._misordered_lifted_copies(graph, sdfg, nsdfg, states, border_arrays)
        for state, e in misordered:
            MapFission.copy_as_map(state, e)
        if misordered and self.expr_index == 0:
            subgraphs = [(graph, graph.scope_subgraph(map_entry, include_entry=False, include_exit=False))]

        modified_arrays = set()
        scalar_like_arrays = set()
        widened = {}  # Container -> (descriptor before widening, after, scalar-like)

        # Get map information
        outer_map: nodes.Map = map_entry.map
        # Border-transient extent equals the iteration count per dimension.
        # Memlets that index border transients are normalized to
        # `(p - iMin) / step` so the squeezed array remains in-bounds for
        # strided maps. Symbolic steps are assumed non-negative.
        mapsize = outer_map.range.size()
        squeezed_idx = [
            (pystr_to_symbolic(p) - iMin) / step
            for p, (iMin, _iMax, step) in zip(outer_map.params, outer_map.range.ranges)
        ]

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
            # A border container may also feed a component, so it need not be a source or sink of the subgraph
            top_access = [n for n in subgraph.scope_children()[None] if isinstance(n, nodes.AccessNode)]

            # Collect external edges
            if self.expr_index == 0:
                external_edges_entry = list(state.out_edges(map_entry))
                external_edges_exit = list(state.in_edges(map_exit))
            else:
                external_edges_entry = [
                    e
                    for e in subgraph.edges()
                    if (isinstance(e.src, nodes.AccessNode) and not nsdfg_node.sdfg.arrays[e.src.data].transient)
                ]
                external_edges_exit = [
                    e
                    for e in subgraph.edges()
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

            # Collect all border arrays
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
                name, newdesc = parent.add_transient(
                    scalar,
                    mapsize,
                    desc.dtype,
                    desc.storage,
                    lifetime=desc.lifetime,
                    debuginfo=desc.debuginfo,
                    allow_conflicts=desc.allow_conflicts,
                    find_new_name=True,
                )

                # Add extra nodes in component boundaries
                for edge in edges:
                    anode = state.add_access(name)
                    sbs = subsets.Range([(idx, idx, 1) for idx in squeezed_idx])
                    state.add_edge(
                        edge.src,
                        edge.src_conn,
                        anode,
                        None,
                        mm.Memlet.simple(name, sbs, num_accesses=outer_map.range.num_elements()),
                    )
                    state.add_edge(
                        anode,
                        None,
                        edge.dst,
                        edge.dst_conn,
                        mm.Memlet.simple(name, sbs, num_accesses=outer_map.range.num_elements()),
                    )
                    state.remove_edge(edge)

            # Add extra maps around components
            new_map_entries = []
            for component_in, component_out in components:
                me, mx = state.add_map(
                    outer_map.label + "_fission",
                    [(p, "0:1") for p in outer_map.params],
                    outer_map.schedule,
                    unroll=outer_map.unroll,
                )

                # Add dynamic input connectors
                for conn in map_entry.in_connectors:
                    if not conn.startswith("IN_"):
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
                        written = state.add_edge(mx, out_conn, e.dst, e.dst_conn, dcpy(e.data))
                        # A reduction the removed map performed is now performed by the new map, inside
                        if self.expr_index == 1 and e in external_edges_exit and edge_to_outer[e].data.wcr:
                            for tree_edge in state.memlet_tree(written):
                                tree_edge.data.wcr = edge_to_outer[e].data.wcr
                    state.remove_edge(e)
                # Empty memlet edge in nested SDFGs
                if state.out_degree(component_out) == 0:
                    state.add_edge(component_out, None, mx, None, mm.Memlet())
            # Connect access nodes not in components directly to external nodes
            outside_border_edges = set()
            if self.expr_index == 0:
                for node in top_access:
                    for edge in state.in_edges(node):
                        outer_edge = edge_to_outer.get(edge)
                        if outer_edge is None:  # No outer feeder: nothing to rewire.
                            continue
                        # The added map dimensions are filled in below
                        new_edge = state.add_edge(
                            outer_edge.src, outer_edge.src_conn, edge.dst, edge.dst_conn, dcpy(edge.data)
                        )
                        outside_border_edges.add(new_edge)
                    for edge in state.out_edges(node):
                        outer_edge = edge_to_outer.get(edge)
                        if outer_edge is None:  # No outer consumer: nothing to rewire.
                            continue
                        new_edge = state.add_edge(
                            edge.src, edge.src_conn, outer_edge.dst, outer_edge.dst_conn, dcpy(outer_edge.data)
                        )
                        outside_border_edges.add(new_edge)

            # Augment arrays by prepending map dimensions
            for array in arrays:
                if array in modified_arrays:
                    continue
                desc = parent.arrays[array]
                old_desc = dcpy(desc)
                # Treat scalars and length-1 arrays as "scalar-like": their single
                # degenerate dimension is replaced by the map dims rather than
                # prepended, so the result is shape [extent] rather than
                # [extent, 1] (which produces zero-stride aliasing).
                scalar_like = isinstance(desc, dt.Scalar) or (
                    isinstance(desc, dt.Array) and len(desc.shape) == 1 and desc.shape[0] == 1
                )
                if isinstance(desc, dt.Scalar):
                    desc = dt.Array(
                        desc.dtype,
                        desc.shape,
                        desc.transient,
                        desc.allow_conflicts,
                        desc.storage,
                        desc.location,
                        desc.strides,
                        desc.offset,
                        False,
                        desc.lifetime,
                        0,
                        desc.debuginfo,
                        desc.total_size,
                        desc.start_offset,
                    )
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
                widened[array] = (old_desc, desc, scalar_like)

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

                to_correct = [(e, e.src) for e in external_edges_entry] + [(e, e.dst) for e in external_edges_exit]
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
                        # An integrated nested SDFG already uses the outer coordinate system, up to the ``offset``
                        # it may keep to name the origin of its index space (e.g., one-based indices)
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
                                    # Propagate only the subset of `node.data`; the other side of a copy is kept
                                    other_subset = e.data.other_subset
                                    e.data.other_subset = None
                                    e.data = propagate_subset([e.data], desc, outer_map.params, outer_map.range)
                                    e.data.other_subset = other_subset

                        # Only after offsetting memlets we can modify the
                        # overall offset. The memlets of an integrated nested SDFG were not offset, and still
                        # address the container through its own offset
                        if isinstance(desc, dt.Array) and not already_integrated:
                            desc.offset = outer_desc.offset

            # Fill in memlet trees for border transients
            # NOTE: Memlet propagation should run to correct the outer edges
            # NOTE: Edges rewired outside the new Map scopes cannot refer to the map parameters
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
                                # The external subset may still name the out-of-scope map parameters
                                if (
                                    e.data.data != node.data
                                    and e.data.subset is not None
                                    and set(outer_map.params) & set(map(str, e.data.subset.free_symbols))
                                ):
                                    e.data.subset = propagate_subset(
                                        [e.data], parent.arrays[e.data.data], outer_map.params, outer_map.range
                                    ).subset
                            else:
                                map_ranges = [(idx, idx, 1) for idx in squeezed_idx]
                            if e.data.data == node.data:
                                if e.data.subset:
                                    if is_scalar_like:
                                        e.data.subset = subsets.Range(map_ranges)
                                    else:
                                        e.data.subset = subsets.Range(map_ranges + e.data.subset.ranges)
                            elif e.data.other_subset:
                                if is_scalar_like:
                                    e.data.other_subset = subsets.Range(map_ranges)
                                else:
                                    e.data.other_subset = subsets.Range(map_ranges + e.data.other_subset.ranges)

        # A connector sharing the coordinates of a widened container is widened alike; one selecting a window of it
        # becomes a view of it
        for state in parent.states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG) and any(
                    e.data.data in modified_arrays for e in state.all_edges(node)
                ):
                    ptypes = {p: state.symbols_defined_at(node)[p] for p in outer_map.params}
                    MapFission.widen_integrated_connectors(node, state, widened, squeezed_idx, ptypes)
                    node.integrate_into_parent()

        # If nested SDFG, reconnect nodes around map and modify memlets
        if self.expr_index == 1:
            for edge in graph.in_edges(map_entry):
                if edge.data.is_empty():
                    # an ordering edge carries no data: it now orders the nested SDFG that replaces the map
                    graph.add_nedge(edge.src, nsdfg_node, dcpy(edge.data))
                    continue
                if not edge.dst_conn or not edge.dst_conn.startswith("IN_"):
                    continue

                # Modify edge coming into nested SDFG to include entire array
                desc = sdfg.arrays[edge.data.data]
                edge.data.subset = subsets.Range.from_array(desc)
                edge.data.num_accesses = edge.data.subset.num_elements()

                # Find matching edge inside map
                for inner_edge in graph.out_edges_by_connector(map_entry, f"OUT_{edge.dst_conn[3:]}"):
                    graph.add_edge(edge.src, edge.src_conn, nsdfg_node, inner_edge.dst_conn, dcpy(edge.data))

            for edge in graph.out_edges(map_exit):
                if edge.data.is_empty():
                    graph.add_nedge(nsdfg_node, edge.dst, dcpy(edge.data))
                    continue
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
