"""Backend #2: the rustworkx-backed compat shim. Wraps a rustworkx.PyDiGraph so DaCe's
scratch-graph call sites (dace.graphlib.DiGraph()/MultiDiGraph()) get real acceleration while
looking, to their callers, like the real-networkx graphs they replace: arbitrary
(possibly-unhashable) Python objects as node identities, dict-shaped attribute payloads
accessed and mutated the same way networkx's attribute dicts are (add_node(n, **attr),
add_edge(u, v, **attr), G[u][v]['x'] += y).

rustworkx itself is imported lazily, only inside the functions/methods that actually touch it,
so `import dace.graphlib` never requires rustworkx to be installed -- only selecting
backend='rustworkx' does (see dace/config_schema.yml's graph:backend description).

No mixed backends: under backend='rustworkx', a REAL networkx graph (e.g. reached via an
SDFG/state's .nx/._nx escape hatch) is not hardcoded to run on networkx just because of its
Python type -- every algorithm below that doesn't need a callback/special adjacency protocol
coerces it to a temporary rustworkx handle on the fly and runs accelerated (see _coerce()).
Two confirmed rustworkx gaps stay on real networkx regardless of backend, since there is no
accelerated implementation to lower to: transitive closure (handled here via to_networkx(), a
single documented conversion helper, not a per-call shim pattern repeated everywhere) and
directed s-t max-flow/min-cut (handled in dace.graphlib.algorithms.flow.edmondskarp, which
reuses this same helper).
"""
import copy

from networkx.exception import NetworkXError, NetworkXNoCycle, NetworkXNoPath, NetworkXUnfeasible, NodeNotFound


class NodeIndexMap:
    """Bidirectional node-object <-> rustworkx integer index map. Matches real networkx's actual
    node-identity semantics: a real dict, keyed by hash+equality (e.g. two separately-constructed
    but equal tuples like (0, i) are the SAME node -- state_fusion.py's find_fused_components
    relies on exactly this), with an id(node)-keyed fallback ONLY for genuinely unhashable nodes
    (DaCe graphs are known to contain some, e.g. list-backed constructs -- see
    dace.sdfg.graph.OrderedDiGraph.has_cycles' hashability note). For DaCe's own node objects
    (Tasklet, AccessNode, ...), which don't override __hash__/__eq__, hash+equality already IS
    identity, so this is a strict superset of the old id()-only behavior, not a behavior change
    for that case -- it only fixes the value-comparable-node case (ints, strings, tuples...),
    which is exactly what real networkx supports and DaCe's own code (e.g. state_fusion.py's
    bipartite (group, index) tuple nodes) relies on.
    """

    def __init__(self):
        self.obj_to_idx = {}
        self.id_to_idx = {}
        self.idx_to_obj = {}

    def add(self, node, index):
        try:
            self.obj_to_idx[node] = index
        except TypeError:
            self.id_to_idx[id(node)] = index
        self.idx_to_obj[index] = node

    def remove(self, node):
        try:
            index = self.obj_to_idx.pop(node)
        except TypeError:
            index = self.id_to_idx.pop(id(node))
        del self.idx_to_obj[index]

    def index_of(self, node):
        try:
            return self.obj_to_idx[node]
        except TypeError:
            return self.id_to_idx[id(node)]

    def node_at(self, index):
        return self.idx_to_obj[index]

    def __contains__(self, node):
        try:
            return node in self.obj_to_idx
        except TypeError:
            return id(node) in self.id_to_idx


class AdjacencyView:
    """G[u] -> AdjacencyView(u); G[u][v] -> the (u, v) edge's attribute dict, by reference (as
    confirmed empirically: rustworkx's get_edge_data returns the stored object itself, not a
    copy), so G[u][v]['attr'] += x mutates the real edge data like networkx's adjacency view."""

    def __init__(self, handle, u):
        self._handle = handle
        self._u = u

    def __getitem__(self, v):
        return self._handle.get_edge_payload(self._u, v)

    def __contains__(self, v):
        return self._handle.has_edge(self._u, v)


class NodeView:
    """`G.nodes` supports both bare iteration (`for n in G.nodes:`) and being called
    (`G.nodes()`, `G.nodes(data=True)`), matching real networkx.DiGraph.nodes' dual
    property/callable-view shape."""

    def __init__(self, handle):
        self._handle = handle

    def _list(self):
        # Node insertion order, NOT rustworkx's own index order: PyDiGraph recycles indices
        # freed by remove_node (confirmed empirically -- add 'a','b','c', remove 'b', add 'd' ->
        # 'd' lands at 'b's old index, between 'a' and 'c' in node_indices() order). idx_to_obj
        # is a plain dict populated in NodeIndexMap.add()/.remove() call order, so its own
        # iteration order already matches real networkx's del/reinsert-preserving _node dict
        # semantics exactly -- no separate order-tracking structure needed.
        return list(self._handle._index.idx_to_obj.values())

    def __call__(self, data=False):
        if not data:
            return self._list()
        return [(n, self._handle._rx.get_node_data(self._handle._index.index_of(n))) for n in self._list()]

    def __iter__(self):
        return iter(self._list())

    def __len__(self):
        return self._handle.number_of_nodes()

    def __contains__(self, node):
        return node in self._handle

    def __getitem__(self, node):
        # G.nodes[n] -> that node's attribute dict, by reference (matches real
        # networkx.DiGraph.nodes[n]; used e.g. by dace/transformation/passes/pattern_matching.py's
        # node_pred(digraph.nodes[nid], pnode)).
        return self._handle._rx.get_node_data(self._handle._index.index_of(node))


class EdgeView:
    """`G.edges`, same dual property/callable-view shape as NodeView above."""

    def __init__(self, handle):
        self._handle = handle

    def _list(self):
        # Grouped by source node in node-insertion order, then by edge-insertion order within
        # each node -- matches real networkx.DiGraph.edges' actual iteration order (its adjacency
        # dict is keyed by node-insertion order, with each node's own out-edges insertion-ordered
        # within it). rustworkx's own PyDiGraph.edge_list() is flat GLOBAL insertion order
        # (confirmed empirically to NOT group by source node the way networkx does), and its
        # per-node out_edges() is LIFO -- reverse of insertion order (confirmed empirically: new
        # edges are prepended, so reversing always recovers true insertion order, including after
        # a remove+re-add). Silently using either raw form previously caused a real divergence:
        # dace/transformation/passes/pattern_matching.py's order-sensitive match selection picked
        # a different (individually valid, but not networkx-matching) match under rustworkx,
        # observed via tests/transformations/double_buffering_test.py.
        result = []
        for u_idx, u in self._handle._index.idx_to_obj.items():
            for _, v_idx, _ in reversed(list(self._handle._rx.out_edges(u_idx))):
                result.append((u, self._handle._index.node_at(v_idx)))
        return result

    def __call__(self, data=False):
        pairs = self._list()
        if not data:
            return pairs
        return [(u, v, self._handle.get_edge_payload(u, v)) for u, v in pairs]

    def __iter__(self):
        return iter(self._list())

    def __len__(self):
        return self._handle.number_of_edges()

    def __getitem__(self, key):
        # G.edges[u, v] -> that edge's attribute dict, by reference (matches real
        # networkx.DiGraph.edges[u, v]; used e.g. by pattern_matching.py's
        # edge_pred(digraph.edges[u, v], nxpattern.edges[pedge])). Python desugars `edges[u, v]`
        # to a single `(u, v)` tuple argument here, same as the explicit-tuple form.
        u, v = key
        return self._handle.get_edge_payload(u, v)


class RustworkxGraphHandle:
    """A dace.graphlib.DiGraph()/MultiDiGraph() built under backend='rustworkx'."""

    graphlib_backend = None  # bound to INSTANCE at the bottom of this module

    def __init__(self, multigraph=False):
        import rustworkx
        self.multigraph = multigraph
        self._rx = rustworkx.PyDiGraph(multigraph=multigraph)
        self._index = NodeIndexMap()

    # -- construction / mutation, mirrors networkx.DiGraph's own method signatures ----------

    def add_node(self, node_for_adding, **attr):
        if node_for_adding in self._index:
            self._rx.get_node_data(self._index.index_of(node_for_adding)).update(attr)
            return
        idx = self._rx.add_node(dict(attr))
        self._index.add(node_for_adding, idx)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)
        ui, vi = self._index.index_of(u_of_edge), self._index.index_of(v_of_edge)
        return self._rx.add_edge(ui, vi, dict(attr))

    def remove_node(self, node):
        self._rx.remove_node(self._index.index_of(node))
        self._index.remove(node)

    def remove_edge(self, u, v):
        self._rx.remove_edge(self._index.index_of(u), self._index.index_of(v))

    def add_nodes_from(self, nodes_for_adding, **attr):
        # matches networkx.DiGraph.add_nodes_from: items may be a plain node or a (node, attr
        # dict) 2-tuple; the dict (if present) is merged over the function-level **attr defaults.
        for n in nodes_for_adding:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                node, node_attr = n
                merged = dict(attr)
                merged.update(node_attr)
                self.add_node(node, **merged)
            else:
                self.add_node(n, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        # matches networkx.DiGraph.add_edges_from: items may be a (u, v) 2-tuple or a
        # (u, v, attr dict) 3-tuple, dict merged over the function-level **attr defaults.
        for e in ebunch_to_add:
            if len(e) == 3:
                u, v, edge_attr = e
                merged = dict(attr)
                merged.update(edge_attr)
                self.add_edge(u, v, **merged)
            else:
                u, v = e
                self.add_edge(u, v, **attr)

    def remove_nodes_from(self, nodes):
        # matches networkx.DiGraph.remove_nodes_from: silently ignores nodes not in the graph.
        for n in list(nodes):
            if n in self._index:
                self.remove_node(n)

    def remove_edges_from(self, ebunch):
        # matches networkx.DiGraph.remove_edges_from: silently ignores edges not in the graph.
        for e in ebunch:
            u, v = e[0], e[1]
            if self.has_edge(u, v):
                self.remove_edge(u, v)

    # -- queries ------------------------------------------------------------------------------

    def has_node(self, node):
        return node in self._index

    def has_edge(self, u, v):
        if u not in self._index or v not in self._index:
            return False
        return self._rx.has_edge(self._index.index_of(u), self._index.index_of(v))

    def get_edge_payload(self, u, v):
        return self._rx.get_edge_data(self._index.index_of(u), self._index.index_of(v))

    @property
    def nodes(self):
        return NodeView(self)

    @property
    def edges(self):
        return EdgeView(self)

    # NOTE: rustworkx's in_edges/out_edges/successor_indices/predecessor_indices all return
    # results in LIFO (reverse-of-insertion) order, confirmed empirically -- reversed() below
    # recovers real networkx's actual insertion-order iteration (see EdgeView._list's docstring
    # for the full explanation and the real bug this fixes).

    def in_edges(self, node):
        idx = self._index.index_of(node)
        return [(self._index.node_at(u), self._index.node_at(v)) for u, v, _ in reversed(list(self._rx.in_edges(idx)))]

    def out_edges(self, node):
        idx = self._index.index_of(node)
        return [(self._index.node_at(u), self._index.node_at(v)) for u, v, _ in reversed(list(self._rx.out_edges(idx)))]

    def successors(self, node):
        idx = self._index.index_of(node)
        return (self._index.node_at(i) for i in reversed(list(self._rx.successor_indices(idx))))

    def predecessors(self, node):
        idx = self._index.index_of(node)
        return (self._index.node_at(i) for i in reversed(list(self._rx.predecessor_indices(idx))))

    def in_degree(self, node):
        return self._rx.in_degree(self._index.index_of(node))

    def out_degree(self, node):
        return self._rx.out_degree(self._index.index_of(node))

    def reverse(self, copy=True):
        # matches networkx.DiGraph.reverse's copy=True default (a new graph, edges reversed);
        # copy=False (in-place reversal) isn't implemented -- no caller in dace/ needs it.
        if not copy:
            raise NotImplementedError('RustworkxGraphHandle.reverse(copy=False) is not supported')
        result = RustworkxGraphHandle(multigraph=self.multigraph)
        for node in self.nodes():
            result.add_node(node, **self._rx.get_node_data(self._index.index_of(node)))
        for u, v in self.edges():
            result.add_edge(v, u, **self.get_edge_payload(u, v))
        return result

    def number_of_nodes(self):
        return self._rx.num_nodes()

    def number_of_edges(self):
        return self._rx.num_edges()

    def __len__(self):
        return self._rx.num_nodes()

    def __iter__(self):
        return iter(self.nodes())

    def __contains__(self, node):
        return node in self._index

    def __getitem__(self, u):
        return AdjacencyView(self, u)

    # -- deepcopy: required by the GraphBackend protocol contract (protocol.py) -- rebuilt from
    # scratch rather than trusting PyDiGraph's own copy/pickle support (unverified), so
    # correctness depends only on documented rustworkx primitives (add_node/add_edge) and
    # Python's own copy.deepcopy contract for the payload objects, which we control. Uses the
    # caller's `memo`, not a fresh one, so cross-references (a node shared by two edges, or by
    # some external structure sharing the same deepcopy operation, e.g. SDFG.__deepcopy__'s own
    # generic per-attribute sweep) resolve to the identical copied instance -- see dace.graphlib
    # package docs and the implementation plan for why this specific property is load-bearing.

    def __deepcopy__(self, memo):
        result = RustworkxGraphHandle(multigraph=self.multigraph)
        for node in self.nodes():
            attr = self._rx.get_node_data(self._index.index_of(node))
            result.add_node(copy.deepcopy(node, memo), **copy.deepcopy(attr, memo))
        for u, v in self.edges():
            payload = self.get_edge_payload(u, v)
            result.add_edge(copy.deepcopy(u, memo), copy.deepcopy(v, memo), **copy.deepcopy(payload, memo))
        return result


def _index_of(G, node, exc_type, message):
    """G._index.index_of() raises a raw KeyError for a missing node; real networkx raises a
    specific, documented exception type per algorithm (NodeNotFound for has_path/
    shortest_path_length/all_simple_paths' source, NetworkXError for immediate_dominators/
    descendants/ancestors/dfs_edges) -- several dace call sites (e.g.
    dace/transformation/dataflow/redundant_array.py's `except NodeNotFound:`) actually catch
    these, so matching the real type, not just the real VALUE, is required."""
    if node not in G._index:
        raise exc_type(message)
    return G._index.index_of(node)


def _coerce(G):
    """No mixed backends (see resolve.backend_for): under backend='rustworkx', algorithm calls
    on a REAL networkx graph (e.g. reached via an SDFG/state's .nx/._nx escape hatch) still run
    the accelerated implementation, not the reference one. If G is already a
    RustworkxGraphHandle this is zero-copy; if G is a real networkx graph, a temporary handle is
    built on the fly, used for this one call, and discarded -- only the result, translated back
    to the ORIGINAL node objects via the same NodeIndexMap machinery used everywhere else in
    this module, escapes this function.

    Known performance characteristic, not yet addressed: this conversion is O(V+E) and uncached,
    so a caller that makes many algorithm calls against the same unchanged live graph (e.g. a
    loop of has_path checks against one SDFGState) currently re-pays the full conversion cost
    every single call, which can be *slower* than just running real networkx's own algorithm
    directly for small/cheap queries. A correct cache would need invalidation hooks wired into
    every mutator of dace.sdfg.graph.OrderedDiGraph (add_node/add_edge/remove_node/remove_edge/
    reverse) to stay safe -- exactly the invasive, higher-risk "dual-storage sync across ~15
    mutator methods" change the architecture deliberately avoided for this phase (see the
    dace.graphlib package docs). Do not paper over this with a heuristic cache (e.g. keyed on
    node/edge counts) -- that can silently serve stale results for a same-count topology change
    and would be a correctness bug, not a performance shortcut. Revisit with real profiling data.
    """
    if isinstance(G, RustworkxGraphHandle):
        return G
    handle = RustworkxGraphHandle(multigraph=G.is_multigraph())
    for node, attr in G.nodes(data=True):
        handle.add_node(node, **attr)
    for u, v, attr in G.edges(data=True):
        handle.add_edge(u, v, **attr)
    return handle


def to_networkx(G):
    """Converts a RustworkxGraphHandle to a real, temporary networkx.DiGraph/MultiDiGraph.
    Used only for the two algorithms rustworkx has no native equivalent for: transitive closure
    (below) and max-flow/min-cut (dace.graphlib.algorithms.flow.edmondskarp). Both are cold
    paths (run once per call, not in a hot per-node loop), so the conversion cost is bounded and
    proportionate -- this is deliberately one named, documented, tested conversion, not the
    per-call-copy-shim-everywhere pattern the BoostX/DaCeX prior art warns against.
    """
    import networkx
    result = networkx.MultiDiGraph() if G.multigraph else networkx.DiGraph()
    for node in G.nodes():
        result.add_node(node, **G._rx.get_node_data(G._index.index_of(node)))
    for u, v in G.edges():
        result.add_edge(u, v, **G.get_edge_payload(u, v))
    return result


def _transitive_closure_via_networkx(G, dag_only):
    import networkx
    nxg = to_networkx(G)
    closure = networkx.transitive_closure_dag(nxg) if dag_only else networkx.transitive_closure(nxg)
    result = RustworkxGraphHandle(multigraph=G.multigraph)
    for node in closure.nodes():
        result.add_node(node)
    for u, v in closure.edges():
        result.add_edge(u, v)
    return result


class RustworkxDiGraphMatcher:
    """Backend-specific implementation behind graphlib.isomorphism.DiGraphMatcher when both
    input graphs are rustworkx-backed. Matches networkx's DiGraphMatcher shape (constructed
    with (G1, G2, node_match, edge_match), iterated via .subgraph_isomorphisms_iter()) so
    pattern_matching.py needs no changes beyond the import swap."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        self._G1, self._G2 = G1, G2
        self._node_match, self._edge_match = node_match, edge_match

    def subgraph_isomorphisms_iter(self):
        import rustworkx
        node_matcher = (lambda a, b: self._node_match(a, b)) if self._node_match else None
        edge_matcher = (lambda a, b: self._edge_match(a, b)) if self._edge_match else None
        mappings = rustworkx.digraph_vf2_mapping(self._G1._rx,
                                                  self._G2._rx,
                                                  node_matcher=node_matcher,
                                                  edge_matcher=edge_matcher,
                                                  subgraph=True,
                                                  induced=True)
        for mapping in mappings:
            yield {self._G1._index.node_at(a): self._G2._index.node_at(b) for a, b in mapping.items()}


class RustworkxBackend:
    name = 'rustworkx'

    def new_digraph(self):
        return RustworkxGraphHandle(multigraph=False)

    def new_multidigraph(self):
        return RustworkxGraphHandle(multigraph=True)

    def has_path(self, G, source, target):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NodeNotFound, f'Source {source} is not in G')
        target_idx = _index_of(G, target, NodeNotFound, f'Target {target} is not in G')
        return rustworkx.has_path(G._rx, source_idx, target_idx)

    def immediate_dominators(self, G, start):
        import rustworkx
        G = _coerce(G)
        start_idx = _index_of(G, start, NetworkXError, f'{start} is not in G')
        idom = rustworkx.immediate_dominators(G._rx, start_idx)
        return {G._index.node_at(k): G._index.node_at(v) for k, v in idom.items()}

    def weakly_connected_components(self, G):
        # lazy generator, matching real networkx.weakly_connected_components (each component is
        # still an eager set, same as real networkx -- only the outer sequence is lazy).
        import rustworkx
        G = _coerce(G)
        for comp in rustworkx.weakly_connected_components(G._rx):
            yield {G._index.node_at(i) for i in comp}

    def topological_sort(self, G):
        # lazy generator, matching real networkx.topological_sort. rustworkx raises its own
        # rustworkx.DAGHasCycle on a cyclic graph where real networkx raises NetworkXUnfeasible
        # -- translated here so `except nx.NetworkXUnfeasible:` call sites (e.g. dace/library.py's
        # dependency-graph sort) work identically either backend, including when the cycle is
        # only discovered partway through consuming the generator.
        import rustworkx
        G = _coerce(G)
        try:
            for i in rustworkx.topological_sort(G._rx):
                yield G._index.node_at(i)
        except rustworkx.DAGHasCycle as e:
            raise NetworkXUnfeasible(str(e)) from e

    def simple_cycles(self, G):
        # lazy generator, matching real networkx.simple_cycles.
        import rustworkx
        G = _coerce(G)
        for cycle in rustworkx.simple_cycles(G._rx):
            yield [G._index.node_at(i) for i in cycle]

    def find_cycle(self, G, source=None):
        # networkx's `source` accepts a single node OR a list of nodes (tried in turn until a
        # cycle is found); rustworkx's digraph_find_cycle only accepts a single index, so a
        # list is looped here. Note: unlike real networkx, a `source=None` search on a
        # *disconnected* graph only searches rustworkx's own arbitrarily-chosen starting
        # component, not exhaustively all of them -- the only caller today
        # (dace.sdfg.graph.DiGraph.has_cycles, confirmed dead code with no live callers) always
        # passes an explicit, non-empty source list, so this doesn't affect anything reachable.
        import rustworkx
        G = _coerce(G)
        sources = list(source) if isinstance(source, (list, tuple, set)) else [source]
        for src in sources:
            # unlike the other methods here, real networkx does NOT raise for a missing/None
            # source in find_cycle -- it just finds no cycle from it. Match that: skip rather
            # than raise, don't call _index_of.
            if src is not None and src not in G._index:
                continue
            src_idx = G._index.index_of(src) if src is not None else None
            edges = rustworkx.digraph_find_cycle(G._rx, src_idx)
            if edges:
                return [(G._index.node_at(u), G._index.node_at(v)) for u, v in edges]
        raise NetworkXNoCycle('No cycle found.')

    def descendants(self, G, source):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NetworkXError, f'The node {source} is not in the digraph.')
        return {G._index.node_at(i) for i in rustworkx.descendants(G._rx, source_idx)}

    def ancestors(self, G, source):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NetworkXError, f'The node {source} is not in the digraph.')
        return {G._index.node_at(i) for i in rustworkx.ancestors(G._rx, source_idx)}

    def all_simple_paths(self, G, source, target):
        # lazy generator, matching real networkx.all_simple_paths -- including its asymmetry: a
        # missing SOURCE raises NodeNotFound, but a missing TARGET just yields no paths.
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NodeNotFound, f'source node {source} not in graph')
        if target not in G._index:
            return
        target_idx = G._index.index_of(target)
        paths = rustworkx.digraph_all_simple_paths(G._rx, source_idx, target_idx)
        for path in paths:
            yield [G._index.node_at(i) for i in path]

    def transitive_closure(self, G):
        # No rustworkx equivalent exists at all (confirmed absent) -- not a "lowering"
        # candidate, so a real-networkx input runs directly on real networkx with no pointless
        # round-trip conversion; only an already-rustworkx-native handle goes through the
        # documented to_networkx()-based fallback.
        if isinstance(G, RustworkxGraphHandle):
            return _transitive_closure_via_networkx(G, dag_only=False)
        import networkx
        return networkx.transitive_closure(G)

    def transitive_closure_dag(self, G):
        if isinstance(G, RustworkxGraphHandle):
            return _transitive_closure_via_networkx(G, dag_only=True)
        import networkx
        return networkx.transitive_closure_dag(G)

    def dfs_edges(self, G, source=None):
        # lazy generator, matching real networkx.dfs_edges.
        import rustworkx
        G = _coerce(G)
        if source is not None:
            src_idx = _index_of(G, source, NetworkXError, f'The node {source} is not in the digraph.')
        else:
            src_idx = None
        for u, v in rustworkx.digraph_dfs_edges(G._rx, src_idx):
            yield (G._index.node_at(u), G._index.node_at(v))

    def shortest_path_length(self, G, source, target):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NodeNotFound, f'Source {source} is not in G')
        target_idx = _index_of(G, target, NodeNotFound, f'Target {target} is not in G')
        lengths = rustworkx.dijkstra_shortest_path_lengths(G._rx,
                                                             source_idx,
                                                             edge_cost_fn=lambda _: 1.0,
                                                             goal=target_idx)
        if target_idx not in lengths:
            raise NetworkXNoPath(f'No path between {source} and {target}.')
        return int(lengths[target_idx])

    def isomorphism_matcher(self, G1, G2, node_match=None, edge_match=None):
        # VF2 with match callbacks is natively supported by rustworkx (node_matcher=/
        # edge_matcher=), so this *could* also coerce+lower -- not done here because the only
        # real caller (pattern_matching.py) always builds both graphs via graphlib.DiGraph()
        # already, so G1/G2 are already rustworkx-native by the time this runs; no `.nx`-derived
        # graph is ever handed to this method in practice.
        return RustworkxDiGraphMatcher(_coerce(G1), _coerce(G2), node_match, edge_match)


INSTANCE = RustworkxBackend()
RustworkxGraphHandle.graphlib_backend = INSTANCE
