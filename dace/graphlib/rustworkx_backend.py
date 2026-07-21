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
import functools

import networkx

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

    # id_to_idx holds the ORIGINAL process's id() values, which mean nothing after unpickling --
    # every unhashable node would then miss its entry (has_node False, remove_node KeyError).
    # Drop it on the way out and rebuild from idx_to_obj, which pickles faithfully.
    def __getstate__(self):
        return {'obj_to_idx': self.obj_to_idx, 'idx_to_obj': self.idx_to_obj}

    def __setstate__(self, state):
        self.obj_to_idx = state['obj_to_idx']
        self.idx_to_obj = state['idx_to_obj']
        self.id_to_idx = {}
        for index, node in self.idx_to_obj.items():
            try:
                hash(node)
            except TypeError:
                self.id_to_idx[id(node)] = index

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

    # networkx's adjacency view iterates its neighbours (`for v in G[u]`); without this Python
    # falls back to the integer-index protocol and __getitem__(0) raises a confusing KeyError.
    def __iter__(self):
        return self._handle.successors(self._u)

    def keys(self):
        return list(self._handle.successors(self._u))

    def __len__(self):
        return len(list(self._handle.successors(self._u)))


class NeighborMapView:
    """`G.pred` / `G.succ`: subscript a node to get its neighbours, matching networkx's
    adjacency mapping shape closely enough for the real call sites (`len(cfg.nx.pred[u])`,
    `for v in cfg.nx.pred[u]` in dace/sdfg/analysis/cfg.py)."""

    def __init__(self, handle, direction):
        self._handle = handle
        self._direction = direction

    def _neighbors(self, node):
        if self._direction == 'pred':
            return list(self._handle.predecessors(node))
        return list(self._handle.successors(node))

    def __getitem__(self, node):
        return self._neighbors(node)

    def __contains__(self, node):
        return node in self._handle

    def __iter__(self):
        return iter(self._handle.nodes())

    def __len__(self):
        return self._handle.number_of_nodes()


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
        # bulk-fetch all payloads once (handle.node_payloads_by_index) instead of a per-node
        # get_node_data round-trip; idx_to_obj order is exactly _list's node order.
        payloads = self._handle.node_payloads_by_index()
        return [(node, payloads[idx]) for idx, node in self._handle._index.idx_to_obj.items()]

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
        if not data:
            return self._list()
        # read each edge's payload inline from the same out_edges pass (handle.edges_with_payload)
        # rather than a per-edge get_edge_payload round-trip -- also keeps each parallel multigraph
        # edge's own payload, which get_edge_payload/get_edge_data collapse to a single one.
        return list(self._handle.edges_with_payload())

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
            # add_edge calls this twice with no attrs for every edge; skipping the no-op
            # get_node_data round-trip drops 2 FFI calls per edge on every construction path.
            if attr:
                self._rx.get_node_data(self._index.index_of(node_for_adding)).update(attr)
            return
        idx = self._rx.add_node(dict(attr))
        self._index.add(node_for_adding, idx)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)
        ui, vi = self._index.index_of(u_of_edge), self._index.index_of(v_of_edge)
        # Re-adding an existing edge MERGES attributes in networkx (add_edge(a,b,w=1) then
        # add_edge(a,b,z=2) leaves {'w':1,'z':2}); rustworkx's add_edge would replace the
        # payload outright and silently drop 'w'. Multigraphs are exempt -- there a repeat
        # add_edge is a genuinely new parallel edge, not an update of the existing one.
        if not self.multigraph and self._rx.has_edge(ui, vi):
            self._rx.get_edge_data(ui, vi).update(attr)
            return None
        return self._rx.add_edge(ui, vi, dict(attr))

    def remove_node(self, node):
        self._rx.remove_node(self.node_index_or_raise(node))
        self._index.remove(node)

    def remove_edge(self, u, v, key=None):
        import rustworkx
        # Multigraph callers pass back the key add_edge returned, to name ONE parallel edge --
        # dace.sdfg.graph's OrderedMultiDiGraph stores it as edge.key. Our key is rustworkx's
        # own globally-unique edge index (networkx numbers per (u, v) instead, but callers only
        # ever round-trip the value, never assume its numbering), so it removes by index.
        if key is not None:
            try:
                self._rx.remove_edge_from_index(key)
                return
            except (rustworkx.NoEdgeBetweenNodes, IndexError):
                raise NetworkXError(f'The edge {u}-{v} with key {key} is not in the graph.')
        ui, vi = self.node_index_or_raise(u), self.node_index_or_raise(v)
        try:
            self._rx.remove_edge(ui, vi)
        except rustworkx.NoEdgeBetweenNodes:
            # NoEdgeBetweenNodes is not a NetworkXError subclass, so `except NetworkXError:`
            # handlers would miss it entirely.
            raise NetworkXError(f'The edge {u}-{v} is not in the graph.')

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

    # Small pieces of the networkx graph API that callers reach for on any graph-like object;
    # without them an otherwise-valid call site dies with AttributeError only under this backend.
    @property
    def pred(self):
        return NeighborMapView(self, 'pred')

    @property
    def succ(self):
        return NeighborMapView(self, 'succ')

    @property
    def adj(self):
        return NeighborMapView(self, 'succ')

    def is_multigraph(self):
        return self.multigraph

    def is_directed(self):
        return True

    def get_edge_data(self, u, v, default=None):
        if not self.has_edge(u, v):
            return default
        return self.get_edge_payload(u, v)

    # -- bulk accessors: one native rustworkx call for the whole node/edge set, so rebuild/dump
    # loops don't pay a per-element index_of + get_node_data / get_edge_payload round-trip into rust
    # (the antipattern that made __deepcopy__ slower than plain networkx). Shared by __deepcopy__,
    # reverse, to_networkx and the data=True views below.

    def node_payloads_by_index(self):
        """{index: payload} for every node, from one bulk rx.node_indices()/rx.nodes() pair (same
        order). Zip against idx_to_obj (node objects, in insertion order) to recover (node, payload)
        -- the index keys are NOT ascending after remove/re-add, hence a map rather than a bare zip."""
        return dict(zip(self._rx.node_indices(), self._rx.nodes()))

    def edges_with_payload(self):
        """(u, v, payload) for every edge, in real networkx .edges() order (grouped by source node
        in insertion order, then that node's own edge-insertion order -- see EdgeView._list for why
        the order is load-bearing). The payload comes straight from rx.out_edges, which already
        carries it, so there's no second get_edge_payload round-trip per edge -- and each PARALLEL
        multigraph edge keeps its own payload (get_edge_data returns just one of them for all)."""
        for u_idx, u in self._index.idx_to_obj.items():
            for _, v_idx, payload in reversed(list(self._rx.out_edges(u_idx))):
                yield u, self._index.node_at(v_idx), payload

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
        # networkx returns an empty view for a node that isn't there rather than raising.
        if node not in self._index:
            return []
        idx = self._index.index_of(node)
        return [(self._index.node_at(u), self._index.node_at(v)) for u, v, _ in reversed(list(self._rx.in_edges(idx)))]

    def out_edges(self, node):
        if node not in self._index:
            return []
        idx = self._index.index_of(node)
        return [(self._index.node_at(u), self._index.node_at(v)) for u, v, _ in reversed(list(self._rx.out_edges(idx)))]

    def node_index_or_raise(self, node):
        """index_of() raises a bare KeyError for an unknown node, but networkx raises
        NetworkXError here -- and real call sites catch that specific type (e.g.
        dace/transformation/dataflow/redundant_array.py's `except NetworkXError:` around
        successors()), so a KeyError sails straight past the handler."""
        if node not in self._index:
            raise NetworkXError(f'The node {node} is not in the digraph.')
        return self._index.index_of(node)

    def unique_neighbors(self, indices):
        """rustworkx returns one entry per EDGE, so a multigraph yields the same neighbour once
        per parallel edge; networkx yields each neighbour once. Leaving the duplicates in is not
        merely cosmetic -- transformation/helpers.py's simplify_state and transient_reuse.py both
        run `for p in predecessors: for c in successors: add_edge(p, c)`, which squares the
        duplicate count every round and blows up exponentially (a real state measured 131072
        edges and 20s under rustworkx vs 1 edge and 0.7ms under networkx).

        dict.fromkeys keeps first-seen order, so the reversed()-recovered insertion order that
        EdgeView._list documents is preserved. Falls back to id() for unhashable nodes, the same
        way NodeIndexMap does.
        """
        nodes = [self._index.node_at(i) for i in reversed(list(indices))]
        try:
            return list(dict.fromkeys(nodes))
        except TypeError:
            seen, unique = set(), []
            for n in nodes:
                if id(n) not in seen:
                    seen.add(id(n))
                    unique.append(n)
            return unique

    def successors(self, node):
        # Materialized, not a generator: node_at() resolved lazily would alias a recycled index
        # if the graph is mutated mid-iteration, silently yielding a node that was never a
        # neighbour (networkx raises RuntimeError there instead).
        return iter(self.unique_neighbors(self._rx.successor_indices(self.node_index_or_raise(node))))

    # networkx.DiGraph.neighbors is successors (out-neighbors); dace/autodiff/analysis.py calls
    # it on a transitive_closure() result, which is a handle under this backend.
    def neighbors(self, node):
        return self.successors(node)

    def predecessors(self, node):
        return iter(self.unique_neighbors(self._rx.predecessor_indices(self.node_index_or_raise(node))))

    def in_degree(self, node):
        # networkx returns an (empty) degree view rather than raising for an absent node.
        if node not in self._index:
            return 0
        return self._rx.in_degree(self._index.index_of(node))

    def out_degree(self, node):
        if node not in self._index:
            return 0
        return self._rx.out_degree(self._index.index_of(node))

    def reverse(self, copy=True):
        # matches networkx.DiGraph.reverse's copy=True default (a new graph, edges reversed);
        # copy=False (in-place reversal) isn't implemented -- no caller in dace/ needs it.
        if not copy:
            raise NotImplementedError('RustworkxGraphHandle.reverse(copy=False) is not supported')
        result = RustworkxGraphHandle(multigraph=self.multigraph)
        payloads = self.node_payloads_by_index()
        result.build_bulk(((node, payloads[idx]) for idx, node in self._index.idx_to_obj.items()),
                          ((v, u, payload) for u, v, payload in self.edges_with_payload()))
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
        # Bulk-fetch node payloads and read edge payloads inline (see node_payloads_by_index /
        # edges_with_payload) instead of the old per-node index_of + get_node_data and per-edge
        # get_edge_payload round-trips into rust. Walk idx_to_obj in insertion order so the copy's
        # node/edge order stays faithful (see NodeView._list / EdgeView._list). copy.deepcopy(...,
        # memo) is still called on every node and payload -- the unavoidable, load-bearing part: the
        # shared memo keeps cross-references (a node shared by two edges, or by some outer structure
        # deepcopied in the same sweep, e.g. SDFG.__deepcopy__) resolved to the identical copy.
        payloads = self.node_payloads_by_index()
        result.build_bulk(((copy.deepcopy(node, memo), copy.deepcopy(payloads[idx], memo))
                           for idx, node in self._index.idx_to_obj.items()),
                          ((copy.deepcopy(u, memo), copy.deepcopy(v, memo), copy.deepcopy(payload, memo))
                           for u, v, payload in self.edges_with_payload()))
        return result

    def build_bulk(self, nodes_with_payload, edges_with_payload):
        """Fill an EMPTY handle in two rustworkx calls instead of V+E.

        add_node/add_edge each cross into Rust per element; add_nodes_from/add_edges_from take the
        whole batch at once (and return indices in argument order, so NodeIndexMap stays faithful
        to insertion order -- the property NodeView._list depends on). Only valid on an empty
        handle with no duplicate nodes, which is exactly what every conversion path here has.
        """
        nodes_with_payload = list(nodes_with_payload)
        indices = self._rx.add_nodes_from([payload for _, payload in nodes_with_payload])
        for (node, _), index in zip(nodes_with_payload, indices):
            self._index.add(node, index)
        self._rx.add_edges_from([(self._index.index_of(u), self._index.index_of(v), payload)
                                 for u, v, payload in edges_with_payload])


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
    # Attributes go in by dict update, not **kwargs: networkx allows any hashable attribute key
    # (G.nodes[n][7] = ...), and **attr would die with "keywords must be strings".
    # dict(attr) not **attr: networkx permits any hashable attribute key (G.nodes[n][7] = ...),
    # which **kwargs would reject with "keywords must be strings".
    handle.build_bulk(((node, dict(attr)) for node, attr in G.nodes(data=True)),
                      ((u, v, dict(attr)) for u, v, attr in G.edges(data=True)))
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
    payloads = G.node_payloads_by_index()
    for idx, node in G._index.idx_to_obj.items():
        result.add_node(node, **payloads[idx])
    for u, v, payload in G.edges_with_payload():
        result.add_edge(u, v, **payload)
    return result


def _transitive_closure_via_networkx(G, dag_only):
    import networkx
    nxg = to_networkx(G)
    closure = networkx.transitive_closure_dag(nxg) if dag_only else networkx.transitive_closure(nxg)
    result = RustworkxGraphHandle(multigraph=G.multigraph)
    # networkx.transitive_closure starts from G.copy(), so it keeps node and edge attributes --
    # rebuilding with bare add_node/add_edge would silently drop every one of them.
    result.build_bulk(((node, dict(attr)) for node, attr in closure.nodes(data=True)),
                      ((u, v, dict(attr)) for u, v, attr in closure.edges(data=True)))
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


def run_natively_on_networkx(networkx_func):
    """For an algorithm whose own cost is O(V+E), reuse a real networkx graph as-is instead of
    converting it to a temporary rustworkx one.

    _coerce is itself O(V+E), but in Python-level FFI calls, whereas networkx's own traversals are
    tuned and usually early-exit. Measured on a chain graph, the CONVERSION ALONE costs 4.7x
    (V=3000) to 8.1x (V=200) what networkx's whole has_path call costs -- so for a linear
    algorithm, coercing can never pay off, no matter how fast the Rust side is. Only super-linear
    algorithms can amortize a conversion (all_simple_paths and simple_cycles are exponential, VF2
    isomorphism likewise), and those keep coercing.

    This only affects graphs that ARE already real networkx -- i.e. reached through an SDFG or
    state's .nx/._nx, whose storage is real networkx unconditionally by design. A graph built
    natively through graphlib.DiGraph() is a RustworkxGraphHandle and still runs fully in Rust,
    with no conversion anywhere; that is where the backend's speedup actually comes from.
    """

    def decorator(method):

        @functools.wraps(method)
        def wrapper(self, G, *args, **kwargs):
            if isinstance(G, networkx.Graph):
                return networkx_func(G, *args, **kwargs)
            return method(self, G, *args, **kwargs)

        return wrapper

    return decorator


class RustworkxBackend:
    name = 'rustworkx'

    def new_digraph(self):
        return RustworkxGraphHandle(multigraph=False)

    def new_multidigraph(self):
        return RustworkxGraphHandle(multigraph=True)

    @run_natively_on_networkx(networkx.has_path)
    def has_path(self, G, source, target):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NodeNotFound, f'Source {source} is not in G')
        target_idx = _index_of(G, target, NodeNotFound, f'Target {target} is not in G')
        # networkx counts the trivial length-0 path, so has_path(G, x, x) is True for any node in
        # G regardless of cycles; rustworkx.has_path returns False for source == target. Matching
        # networkx here is load-bearing: the symbol-write-scopes analysis' _find_dominating_write
        # probes a loop guard's self-reachability this way, and a False silently swaps the SSA
        # symbol suffixes assigned to two otherwise-identical loops.
        if source_idx == target_idx:
            return True
        return rustworkx.has_path(G._rx, source_idx, target_idx)

    @run_natively_on_networkx(networkx.immediate_dominators)
    def immediate_dominators(self, G, start):
        import rustworkx
        G = _coerce(G)
        start_idx = _index_of(G, start, NetworkXError, f'{start} is not in G')
        idom = rustworkx.immediate_dominators(G._rx, start_idx)
        return {G._index.node_at(k): G._index.node_at(v) for k, v in idom.items()}

    @run_natively_on_networkx(networkx.weakly_connected_components)
    def weakly_connected_components(self, G):
        # lazy generator, matching real networkx.weakly_connected_components (each component is
        # still an eager set, same as real networkx -- only the outer sequence is lazy).
        import rustworkx
        G = _coerce(G)
        for comp in rustworkx.weakly_connected_components(G._rx):
            yield {G._index.node_at(i) for i in comp}

    @run_natively_on_networkx(networkx.topological_sort)
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

    @run_natively_on_networkx(networkx.find_cycle)
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

    @run_natively_on_networkx(networkx.descendants)
    def descendants(self, G, source):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NetworkXError, f'The node {source} is not in the digraph.')
        return {G._index.node_at(i) for i in rustworkx.descendants(G._rx, source_idx)}

    @run_natively_on_networkx(networkx.ancestors)
    def ancestors(self, G, source):
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NetworkXError, f'The node {source} is not in the digraph.')
        return {G._index.node_at(i) for i in rustworkx.ancestors(G._rx, source_idx)}

    def all_simple_paths(self, G, source, target):
        # lazy generator, matching real networkx.all_simple_paths. networkx only tolerates a
        # missing target when it is iterable (it does `set(target)`, so a container of targets);
        # for a plain non-iterable node -- which every DaCe IR node is -- it raises NodeNotFound
        # just like a missing source.
        import rustworkx
        G = _coerce(G)
        source_idx = _index_of(G, source, NodeNotFound, f'source node {source} not in graph')
        if target not in G._index:
            try:
                iter(target)
            except TypeError:
                raise NodeNotFound(f'target node {target} not in graph')
            return
        target_idx = G._index.index_of(target)
        # networkx counts the trivial length-0 path, so source == target yields exactly [[source]]
        # (same convention as has_path); rustworkx's own call would return nothing.
        if source_idx == target_idx:
            yield [source]
            return
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

    @run_natively_on_networkx(networkx.dfs_edges)
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

    @run_natively_on_networkx(networkx.shortest_path_length)
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
