# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Backend #2: rustworkx-backed compat shim. Wraps a rustworkx.PyDiGraph behind networkx's
DiGraph API: arbitrary (possibly-unhashable) node objects, dict-shaped attribute payloads,
mutable `G[u][v]['x'] += y`.

rustworkx is resolved lazily through rustworkx_module(), so `import dace.graphlib` never requires
it installed. Transitive closure and directed s-t max-flow/min-cut have no rustworkx equivalent and
always run on real networkx via to_networkx().
"""
import copy
import functools
import inspect
import types
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import networkx

#: A node's or an edge's networkx-style attribute dict. Keys need not be strings.
Payload = Dict[Any, Any]

#: Cache for rustworkx_module(); populated on first use.
RUSTWORKX_MODULE = None


def rustworkx_module() -> types.ModuleType:
    """
    Imports rustworkx on first use and caches it for every later call.

    :return: The ``rustworkx`` module.
    """
    global RUSTWORKX_MODULE
    if RUSTWORKX_MODULE is None:
        # Optional dependency: a module-level import would break `import dace.graphlib` wherever
        # rustworkx is not installed.
        import rustworkx
        RUSTWORKX_MODULE = rustworkx
    return RUSTWORKX_MODULE


class NodeIndexMap:
    """
    Bidirectional node-object <-> rustworkx integer index map.

    Keyed by hash+equality like networkx's own node dict (two separately-built but equal tuples are
    the SAME node -- state_fusion.py's find_fused_components relies on this), with an id()-keyed
    fallback only for genuinely unhashable nodes, which DaCe graphs do contain.
    """

    def __init__(self) -> None:
        self.obj_to_idx: Dict[Any, int] = {}
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_obj: Dict[int, Any] = {}

    def __getstate__(self) -> Dict[str, Any]:
        # id_to_idx holds this process's id() values, meaningless after unpickling -- drop it and
        # rebuild from idx_to_obj, which pickles faithfully.
        return {'obj_to_idx': self.obj_to_idx, 'idx_to_obj': self.idx_to_obj}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.obj_to_idx = state['obj_to_idx']
        self.idx_to_obj = state['idx_to_obj']
        self.id_to_idx = {}
        for index, node in self.idx_to_obj.items():
            try:
                hash(node)
            except TypeError:
                self.id_to_idx[id(node)] = index

    def add(self, node: Any, index: int) -> None:
        try:
            self.obj_to_idx[node] = index
        except TypeError:
            self.id_to_idx[id(node)] = index
        self.idx_to_obj[index] = node

    def remove(self, node: Any) -> None:
        try:
            index = self.obj_to_idx.pop(node)
        except TypeError:
            index = self.id_to_idx.pop(id(node))
        del self.idx_to_obj[index]

    def index_of(self, node: Any) -> int:
        try:
            return self.obj_to_idx[node]
        except TypeError:
            return self.id_to_idx[id(node)]

    def node_at(self, index: int) -> Any:
        return self.idx_to_obj[index]

    def __contains__(self, node: Any) -> bool:
        try:
            return node in self.obj_to_idx
        except TypeError:
            return id(node) in self.id_to_idx


class AdjacencyView:
    """G[u] -> AdjacencyView(u); G[u][v] -> the edge's attribute dict BY REFERENCE (rustworkx's
    get_edge_data hands back the stored object), so G[u][v]['attr'] += x mutates the real edge."""

    def __init__(self, handle: 'RustworkxGraphHandle', u: Any) -> None:
        self._handle = handle
        self._u = u

    def __getitem__(self, v: Any) -> Payload:
        return self._handle.get_edge_payload(self._u, v)

    def __contains__(self, v: Any) -> bool:
        return self._handle.has_edge(self._u, v)

    def __iter__(self) -> Iterator[Any]:
        # networkx's adjacency view iterates its neighbours (`for v in G[u]`); without this Python
        # falls back to integer indexing and __getitem__(0) raises a confusing KeyError.
        return self._handle.successors(self._u)

    def keys(self) -> List[Any]:
        return list(self._handle.successors(self._u))

    def __len__(self) -> int:
        return len(list(self._handle.successors(self._u)))


class NeighborMapView:
    """`G.pred` / `G.succ`: subscript a node to get its neighbours, matching networkx's adjacency
    mapping shape for the real call sites (`len(cfg.nx.pred[u])`, `for v in cfg.nx.pred[u]`)."""

    def __init__(self, handle: 'RustworkxGraphHandle', direction: str) -> None:
        self._handle = handle
        self._direction = direction

    def __getitem__(self, node: Any) -> List[Any]:
        if self._direction == 'pred':
            return list(self._handle.predecessors(node))
        return list(self._handle.successors(node))

    def __contains__(self, node: Any) -> bool:
        return node in self._handle

    def __iter__(self) -> Iterator[Any]:
        return iter(self._handle.nodes())

    def __len__(self) -> int:
        return self._handle.number_of_nodes()


class NodeView:
    """`G.nodes` works both bare (`for n in G.nodes:`) and called (`G.nodes(data=True)`),
    matching networkx.DiGraph.nodes' dual property/callable-view shape."""

    def __init__(self, handle: 'RustworkxGraphHandle') -> None:
        self._handle = handle

    def _list(self) -> List[Any]:
        # Node INSERTION order, not rustworkx's index order: PyDiGraph recycles indices freed by
        # remove_node. idx_to_obj is in add()/remove() call order, which already matches networkx's
        # del/reinsert-preserving _node dict semantics.
        return list(self._handle._index.idx_to_obj.values())

    def __call__(self, data: bool = False) -> Union[List[Any], List[Tuple[Any, Payload]]]:
        if not data:
            return self._list()
        return list(self._handle.nodes_with_payload())

    def __iter__(self) -> Iterator[Any]:
        return iter(self._list())

    def __len__(self) -> int:
        return self._handle.number_of_nodes()

    def __contains__(self, node: Any) -> bool:
        return node in self._handle

    def __getitem__(self, node: Any) -> Payload:
        # G.nodes[n] -> that node's attribute dict, by reference, like networkx's.
        return self._handle._rx.get_node_data(self._handle._index.index_of(node))


class EdgeView:
    """`G.edges`, same dual property/callable-view shape as NodeView."""

    def __init__(self, handle: 'RustworkxGraphHandle') -> None:
        self._handle = handle

    def _list(self) -> List[Tuple[Any, Any]]:
        return [(u, v) for u, v, _ in self._handle.edges_with_payload()]

    def __call__(self, data: bool = False) -> Union[List[Tuple[Any, Any]], List[Tuple[Any, Any, Payload]]]:
        if not data:
            return self._list()
        # Payloads read inline from the same out_edges pass, which keeps each parallel multigraph
        # edge's own payload; get_edge_data would collapse them to a single one.
        return list(self._handle.edges_with_payload())

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        return iter(self._list())

    def __len__(self) -> int:
        return self._handle.number_of_edges()

    def __getitem__(self, key: Tuple[Any, Any]) -> Payload:
        # G.edges[u, v] -> that edge's attribute dict, by reference, like networkx's. Python passes
        # `edges[u, v]` as a single (u, v) tuple, same as the explicit-tuple form.
        u, v = key
        return self._handle.get_edge_payload(u, v)


class RustworkxGraphHandle:
    """A dace.graphlib.DiGraph()/MultiDiGraph() built under backend='rustworkx'."""

    graphlib_backend = None  # bound to INSTANCE at the bottom of this module

    def __init__(self, multigraph: bool = False) -> None:
        self.multigraph = multigraph
        self._rx = rustworkx_module().PyDiGraph(multigraph=multigraph)
        self._index = NodeIndexMap()
        # {rustworkx edge index: networkx-style multigraph key}. Callers store a key and hand it
        # back later (OrderedMultiDiGraph keeps it as edge.key). rustworkx's own edge index cannot
        # serve as that name -- any rebuild (__deepcopy__ regroups edges by source) renumbers it,
        # so remove_edge(u, v, key) would delete an unrelated edge. These keys are per-(u, v)
        # counters like networkx's, carried through every rebuild.
        self._edge_keys: Dict[int, Any] = {}

    # -- construction / mutation, mirrors networkx.DiGraph's own method signatures ----------

    def add_node(self, node_for_adding: Any, **attr: Any) -> None:
        if node_for_adding in self._index:
            # add_edge calls this twice with no attrs per edge; skipping the no-op round-trip
            # drops 2 FFI calls per edge.
            if attr:
                self._rx.get_node_data(self._index.index_of(node_for_adding)).update(attr)
            return
        self._index.add(node_for_adding, self._rx.add_node(dict(attr)))

    def add_edge(self, u_of_edge: Any, v_of_edge: Any, **attr: Any) -> Optional[Any]:
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)
        ui, vi = self._index.index_of(u_of_edge), self._index.index_of(v_of_edge)
        # Re-adding an existing edge MERGES attributes in networkx; rustworkx's add_edge would
        # replace the payload outright. Multigraphs are exempt -- there a repeat add_edge is a
        # genuinely new parallel edge, not an update.
        if not self.multigraph and self._rx.has_edge(ui, vi):
            self._rx.get_edge_data(ui, vi).update(attr)
            return None
        key = self.next_edge_key(ui, vi)
        self._edge_keys[self._rx.add_edge(ui, vi, dict(attr))] = key
        # networkx.DiGraph.add_edge returns None; only MultiDiGraph returns a key.
        return key if self.multigraph else None

    def next_edge_key(self, ui: int, vi: int) -> Any:
        """
        A key not already naming a parallel (ui, vi) edge, by networkx's own rule: start at the
        pair's current edge count and step up until free. Not always the lowest free value, and
        matching the rule (not just uniqueness) keeps stored keys comparable across backends.

        :param ui: Source node index.
        :param vi: Destination node index.
        :return: The key to give the next (ui, vi) edge.
        """
        if not self.multigraph:
            return 0
        taken = {self._edge_keys[i] for i in self._rx.edge_indices_from_endpoints(ui, vi)}
        key = len(taken)
        while key in taken:
            key += 1
        return key

    def edge_index_for_key(self, ui: int, vi: int, key: Any) -> Optional[int]:
        """
        Finds the edge carrying a caller-held multigraph key.

        :param ui: Source node index.
        :param vi: Destination node index.
        :param key: The networkx-style multigraph key.
        :return: The rustworkx edge index currently holding (ui, vi, key), or None.
        """
        for idx in self._rx.edge_indices_from_endpoints(ui, vi):
            if self._edge_keys.get(idx) == key:
                return idx
        return None

    def remove_node(self, node: Any) -> None:
        # rustworkx drops a node's incident edges with it, but their keys must go too, or a
        # later edge recycling that index inherits a stale key.
        idx = self.node_index_or_raise(node)
        for edge_idx in self._rx.incident_edges(idx, all_edges=True):
            self._edge_keys.pop(edge_idx, None)
        self._rx.remove_node(idx)
        self._index.remove(node)

    def remove_edge(self, u: Any, v: Any, key: Optional[Any] = None) -> None:
        rustworkx = rustworkx_module()
        ui, vi = self.node_index_or_raise(u), self.node_index_or_raise(v)
        # Resolve a caller-held multigraph key through the (u, v) pair, never as a rustworkx edge
        # index -- see __init__'s _edge_keys note on why the index is not a stable edge name.
        if key is not None:
            idx = self.edge_index_for_key(ui, vi, key)
            if idx is None:
                raise networkx.NetworkXError(f'The edge {u}-{v} with key {key} is not in the graph.')
            self._rx.remove_edge_from_index(idx)
            del self._edge_keys[idx]
            return
        try:
            # networkx removes the LAST-added parallel edge when no key is given; rustworkx picks
            # its own, so name the same one explicitly.
            candidates = self._rx.edge_indices_from_endpoints(ui, vi)
            if not candidates:
                raise rustworkx.NoEdgeBetweenNodes
            idx = max(candidates)
            self._rx.remove_edge_from_index(idx)
            self._edge_keys.pop(idx, None)
        except rustworkx.NoEdgeBetweenNodes:
            # NoEdgeBetweenNodes is not a NetworkXError subclass, so `except NetworkXError:`
            # handlers would miss it.
            raise networkx.NetworkXError(f'The edge {u}-{v} is not in the graph.')

    def add_nodes_from(self, nodes_for_adding: Iterable[Any], **attr: Any) -> None:
        # matches networkx: items may be a plain node or a (node, attr dict) 2-tuple, the dict
        # merged over the function-level **attr defaults.
        for n in nodes_for_adding:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                node, node_attr = n
                merged = dict(attr)
                merged.update(node_attr)
                self.add_node(node, **merged)
            else:
                self.add_node(n, **attr)

    def add_edges_from(self, ebunch_to_add: Iterable[Tuple], **attr: Any) -> None:
        # matches networkx: items may be a (u, v) 2-tuple or a (u, v, attr dict) 3-tuple, the
        # dict merged over the function-level **attr defaults.
        for e in ebunch_to_add:
            if len(e) == 3:
                u, v, edge_attr = e
                merged = dict(attr)
                merged.update(edge_attr)
                self.add_edge(u, v, **merged)
            else:
                u, v = e
                self.add_edge(u, v, **attr)

    def remove_nodes_from(self, nodes: Iterable[Any]) -> None:
        # matches networkx: silently ignores nodes not in the graph.
        for n in list(nodes):
            if n in self._index:
                self.remove_node(n)

    def remove_edges_from(self, ebunch: Iterable[Tuple]) -> None:
        # matches networkx: silently ignores edges not in the graph.
        for e in ebunch:
            u, v = e[0], e[1]
            if self.has_edge(u, v):
                self.remove_edge(u, v)

    # -- queries ------------------------------------------------------------------------------

    def has_node(self, node: Any) -> bool:
        return node in self._index

    def has_edge(self, u: Any, v: Any) -> bool:
        if u not in self._index or v not in self._index:
            return False
        return self._rx.has_edge(self._index.index_of(u), self._index.index_of(v))

    def get_edge_payload(self, u: Any, v: Any) -> Payload:
        return self._rx.get_edge_data(self._index.index_of(u), self._index.index_of(v))

    # Bits of the networkx graph API callers reach for on any graph-like object; without them a
    # valid call site dies with AttributeError only under this backend.
    @property
    def pred(self) -> NeighborMapView:
        return NeighborMapView(self, 'pred')

    @property
    def succ(self) -> NeighborMapView:
        return NeighborMapView(self, 'succ')

    @property
    def adj(self) -> NeighborMapView:
        return NeighborMapView(self, 'succ')

    def is_multigraph(self) -> bool:
        return self.multigraph

    def is_directed(self) -> bool:
        return True

    def get_edge_data(self, u: Any, v: Any, default: Any = None) -> Any:
        if not self.has_edge(u, v):
            return default
        return self.get_edge_payload(u, v)

    # -- index translation and bulk accessors: one native rustworkx call for the whole node/edge
    # set, so rebuild/dump loops don't pay a per-element round-trip into rust. Shared by
    # __deepcopy__, reverse, to_networkx and the views above.

    def nodes_at(self, indices: Iterable[int]) -> List[Any]:
        """
        Translates rustworkx node indices back to the node objects they name.

        :param indices: Rustworkx node indices.
        :return: The node objects, in the order the indices were given.
        """
        node_at = self._index.node_at
        return [node_at(i) for i in indices]

    def nodes_with_payload(self) -> Iterator[Tuple[Any, Payload]]:
        """
        Yields (node, payload) per node in insertion order, from one bulk node_indices()/nodes()
        pair. Indexed by a map, not a bare zip against idx_to_obj: the rustworkx indices are NOT
        ascending after a remove/re-add.

        :return: Generator of (node object, attribute dict).
        """
        payloads = dict(zip(self._rx.node_indices(), self._rx.nodes()))
        for idx, node in self._index.idx_to_obj.items():
            yield node, payloads[idx]

    def edges_with_payload(self) -> Iterator[Tuple[Any, Any, Payload]]:
        """
        Yields (u, v, payload) per edge, grouped by source node in node-insertion order and then
        by edge-insertion order within each node -- networkx.DiGraph.edges' own order. rustworkx's
        edge_list() is flat GLOBAL insertion order and its per-node out_edges() is LIFO, so reverse
        it. The order is load-bearing: pattern_matching.py's match selection is order-sensitive.

        The payload comes straight from out_edges, so each PARALLEL multigraph edge keeps its own
        (get_edge_data returns just one of them for all).

        :return: Generator of (source node, destination node, attribute dict).
        """
        for u_idx, u in self._index.idx_to_obj.items():
            for _, v_idx, payload in reversed(list(self._rx.out_edges(u_idx))):
                yield u, self._index.node_at(v_idx), payload

    def edges_with_payload_and_keys(self) -> Iterator[Tuple[Any, Any, Payload, Any]]:
        """
        As edges_with_payload, plus each edge's multigraph key, so a rebuild carries keys across
        unchanged. out_edge_indices is reversed like out_edges so index i lines up with edge i.

        :return: Generator of (source node, destination node, attribute dict, multigraph key).
        """
        for u_idx, u in self._index.idx_to_obj.items():
            edges = reversed(list(self._rx.out_edges(u_idx)))
            indices = reversed(list(self._rx.out_edge_indices(u_idx)))
            for (_, v_idx, payload), edge_idx in zip(edges, indices):
                yield u, self._index.node_at(v_idx), payload, self._edge_keys.get(edge_idx, 0)

    @property
    def nodes(self) -> NodeView:
        return NodeView(self)

    @property
    def edges(self) -> EdgeView:
        return EdgeView(self)

    def endpoints_of(self, rx_edges: Iterable[Tuple[int, int, Payload]]) -> List[Tuple[Any, Any]]:
        """
        Translates rustworkx (u index, v index, payload) triples to (u, v) node-object pairs.

        :param rx_edges: Edge triples from one of rustworkx's per-node accessors, which return
                         LIFO (reverse-of-insertion) order; they are reversed here to recover
                         networkx's insertion-order iteration.
        :return: The (source, destination) node pairs.
        """
        node_at = self._index.node_at
        return [(node_at(u), node_at(v)) for u, v, _ in reversed(list(rx_edges))]

    def in_edges(self, node: Any) -> List[Tuple[Any, Any]]:
        # networkx returns an empty view for a missing node rather than raising.
        if node not in self._index:
            return []
        return self.endpoints_of(self._rx.in_edges(self._index.index_of(node)))

    def out_edges(self, node: Any) -> List[Tuple[Any, Any]]:
        if node not in self._index:
            return []
        return self.endpoints_of(self._rx.out_edges(self._index.index_of(node)))

    def node_index_or_raise(self, node: Any) -> int:
        """
        index_of() raises a bare KeyError for an unknown node, but networkx raises NetworkXError
        and real call sites catch that specific type (e.g. redundant_array.py), so a KeyError would
        sail straight past the handler.

        :param node: The node to look up.
        :return: Its rustworkx node index.
        """
        return index_of_or_raise(self, node, networkx.NetworkXError, f'The node {node} is not in the digraph.')

    def unique_neighbors(self, indices: Iterable[int]) -> List[Any]:
        """
        Deduplicates translated neighbour indices, keeping first-seen order.

        rustworkx returns one entry per EDGE, so a multigraph repeats a neighbour once per parallel
        edge; networkx yields each once. Duplicates are not cosmetic: helpers.py's simplify_state
        and transient_reuse.py run `for p in preds: for c in succs: add_edge(p, c)`, which squares
        the duplicate count every round.

        :param indices: Neighbour node indices, in rustworkx's LIFO order.
        :return: The neighbour node objects, deduplicated, in insertion order.
        """
        nodes = self.nodes_at(reversed(list(indices)))
        try:
            # dict.fromkeys keeps first-seen (insertion) order; id() fallback for unhashable nodes,
            # like NodeIndexMap.
            return list(dict.fromkeys(nodes))
        except TypeError:
            seen, unique = set(), []
            for n in nodes:
                if id(n) not in seen:
                    seen.add(id(n))
                    unique.append(n)
            return unique

    def successors(self, node: Any) -> Iterator[Any]:
        # Materialized, not lazy: node_at() resolved later would alias a recycled index if the
        # graph is mutated mid-iteration, silently yielding a node that was never a neighbour.
        return iter(self.unique_neighbors(self._rx.successor_indices(self.node_index_or_raise(node))))

    def neighbors(self, node: Any) -> Iterator[Any]:
        # networkx.DiGraph.neighbors is successors (out-neighbors).
        return self.successors(node)

    def predecessors(self, node: Any) -> Iterator[Any]:
        return iter(self.unique_neighbors(self._rx.predecessor_indices(self.node_index_or_raise(node))))

    def in_degree(self, node: Any) -> Union[int, List]:
        # An absent node gives networkx an empty degree VIEW, not 0, and `[] == 0` is False --
        # returning a literal 0 makes validation.py's `in_degree(s) == 0 and out_degree(s) == 0`
        # unreachable-state check fire on a perfectly valid graph.
        if node not in self._index:
            return []
        return self._rx.in_degree(self._index.index_of(node))

    def out_degree(self, node: Any) -> Union[int, List]:
        if node not in self._index:
            return []
        return self._rx.out_degree(self._index.index_of(node))

    def reverse(self, copy: bool = True) -> 'RustworkxGraphHandle':
        # matches networkx's copy=True default; in-place reversal isn't needed by any dace caller.
        if not copy:
            raise NotImplementedError('RustworkxGraphHandle.reverse(copy=False) is not supported')
        result = RustworkxGraphHandle(multigraph=self.multigraph)
        result.build_bulk(self.nodes_with_payload(), ((v, u, payload) for u, v, payload in self.edges_with_payload()))
        return result

    def number_of_nodes(self) -> int:
        return self._rx.num_nodes()

    def number_of_edges(self) -> int:
        return self._rx.num_edges()

    def __len__(self) -> int:
        return self._rx.num_nodes()

    def __iter__(self) -> Iterator[Any]:
        return iter(self.nodes())

    def __contains__(self, node: Any) -> bool:
        return node in self._index

    def __getitem__(self, u: Any) -> AdjacencyView:
        return AdjacencyView(self, u)

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'RustworkxGraphHandle':
        # Required by the GraphBackend protocol (protocol.py). Rebuilt from scratch rather than
        # trusting PyDiGraph's own copy support, so correctness rests only on documented rustworkx
        # primitives plus Python's deepcopy contract for the payloads.
        result = RustworkxGraphHandle(multigraph=self.multigraph)
        # Walks nodes in insertion order so the copy's node/edge order stays faithful. Uses the
        # CALLER's memo, not a fresh one, so cross-references (a node shared by two edges, or by an
        # outer structure deepcopied in the same sweep, e.g. SDFG.__deepcopy__) resolve to the
        # identical copied instance.
        result.build_bulk(
            ((copy.deepcopy(node, memo), copy.deepcopy(payload, memo)) for node, payload in self.nodes_with_payload()),
            ((copy.deepcopy(u, memo), copy.deepcopy(v, memo), copy.deepcopy(payload, memo), key)
             for u, v, payload, key in self.edges_with_payload_and_keys()))
        return result

    def build_bulk(self, nodes_with_payload: Iterable[Tuple[Any, Payload]],
                   edges_with_payload: Iterable[Tuple]) -> None:
        """
        Fills an EMPTY handle in two rustworkx calls instead of V+E.

        add_nodes_from/add_edges_from return indices in argument order, so NodeIndexMap stays
        faithful to insertion order (what NodeView._list depends on). Only valid on an empty
        handle with no duplicate nodes.

        :param nodes_with_payload: (node, attribute dict) per node.
        :param edges_with_payload: (u, v, payload) or (u, v, payload, key) per edge; the key form
                                   PRESERVES caller-visible multigraph keys across a rebuild
                                   (__deepcopy__ uses it), the 3-tuple form mints them fresh.
        """
        nodes_with_payload = list(nodes_with_payload)
        indices = self._rx.add_nodes_from([payload for _, payload in nodes_with_payload])
        for (node, _), index in zip(nodes_with_payload, indices):
            self._index.add(node, index)

        # Keys for the 3-tuple form are minted here, not via next_edge_key: the edges are not in
        # the graph yet, so it would see no parallel edges and hand every one of them key 0.
        minted = {}
        endpoints = []
        for edge in edges_with_payload:
            u, v, payload = edge[0], edge[1], edge[2]
            ui, vi = self._index.index_of(u), self._index.index_of(v)
            if len(edge) == 4:
                key = edge[3]
            else:
                key = minted.get((ui, vi), 0)
            minted[(ui, vi)] = max(minted.get((ui, vi), 0), key) + 1
            endpoints.append((ui, vi, payload, key))

        edge_indices = self._rx.add_edges_from([(ui, vi, payload) for ui, vi, payload, _ in endpoints])
        for (_, _, _, key), edge_idx in zip(endpoints, edge_indices):
            self._edge_keys[edge_idx] = key


def index_of_or_raise(G: RustworkxGraphHandle, node: Any, exc_type: type, message: str) -> int:
    """
    index_of() raises a raw KeyError for a missing node; networkx raises a specific type per
    algorithm (NodeNotFound vs NetworkXError), and dace call sites catch those types.

    :param G: The handle to look the node up in.
    :param node: The node to look up.
    :param exc_type: Exception type to raise if the node is absent.
    :param message: Message for that exception.
    :return: The node's rustworkx index.
    """
    if node not in G._index:
        raise exc_type(message)
    return G._index.index_of(node)


def _coerce(G: Any) -> RustworkxGraphHandle:
    """
    No mixed backends: a REAL networkx graph (reached via an SDFG/state's .nx/._nx) still runs the
    accelerated implementation. Zero-copy if G is already a handle; otherwise a temporary handle is
    built, used for one call and discarded, with results translated back to the ORIGINAL node
    objects.

    Known, unaddressed: the conversion is O(V+E) and uncached, so many calls against one unchanged
    graph re-pay it and can be slower than plain networkx. A correct cache would need invalidation
    hooks in every OrderedDiGraph mutator; do NOT approximate one with a heuristic (e.g. keyed on
    node/edge counts) -- that serves stale results on a same-count topology change.

    :param G: A RustworkxGraphHandle or a real networkx graph.
    :return: A handle holding the same nodes, edges and attributes.
    """
    if isinstance(G, RustworkxGraphHandle):
        return G
    handle = RustworkxGraphHandle(multigraph=G.is_multigraph())
    # dict(attr) not **attr: networkx permits any hashable attribute key (G.nodes[n][7] = ...),
    # which **kwargs would reject with "keywords must be strings".
    handle.build_bulk(((node, dict(attr)) for node, attr in G.nodes(data=True)),
                      ((u, v, dict(attr)) for u, v, attr in G.edges(data=True)))
    return handle


def to_networkx(G: RustworkxGraphHandle) -> networkx.DiGraph:
    """
    Converts a handle to a real, temporary networkx graph. Used only for the two algorithms
    rustworkx has no equivalent for: transitive closure (below) and max-flow/min-cut
    (algorithms.flow.edmondskarp). Both are cold paths, so the conversion cost is bounded.

    :param G: The handle to convert.
    :return: An equivalent networkx MultiDiGraph or DiGraph.
    """
    result = networkx.MultiDiGraph() if G.multigraph else networkx.DiGraph()
    for node, payload in G.nodes_with_payload():
        result.add_node(node, **payload)
    for u, v, payload in G.edges_with_payload():
        result.add_edge(u, v, **payload)
    return result


def _transitive_closure_via_networkx(G: RustworkxGraphHandle, dag_only: bool) -> RustworkxGraphHandle:
    """
    Runs networkx's transitive closure on a temporary conversion of ``G``.

    :param G: The handle to close over.
    :param dag_only: Use the cheaper DAG-only algorithm.
    :return: A new handle holding the closure.
    """
    nxg = to_networkx(G)
    closure = networkx.transitive_closure_dag(nxg) if dag_only else networkx.transitive_closure(nxg)
    result = RustworkxGraphHandle(multigraph=G.multigraph)
    # networkx.transitive_closure starts from G.copy(), so it keeps node/edge attributes --
    # rebuilding with bare add_node/add_edge would drop them.
    result.build_bulk(((node, dict(attr)) for node, attr in closure.nodes(data=True)),
                      ((u, v, dict(attr)) for u, v, attr in closure.edges(data=True)))
    return result


class RustworkxDiGraphMatcher:
    """rustworkx VF2 behind networkx's DiGraphMatcher shape ((G1, G2, node_match, edge_match)
    plus .subgraph_isomorphisms_iter()), so pattern_matching.py needs only the import swap."""

    def __init__(self,
                 G1: RustworkxGraphHandle,
                 G2: RustworkxGraphHandle,
                 node_match: Optional[Callable[[Payload, Payload], bool]] = None,
                 edge_match: Optional[Callable[[Payload, Payload], bool]] = None) -> None:
        self._G1, self._G2 = G1, G2
        self._node_match, self._edge_match = node_match, edge_match

    def subgraph_isomorphisms_iter(self) -> Iterator[Dict[Any, Any]]:
        node_matcher = (lambda a, b: self._node_match(a, b)) if self._node_match else None
        edge_matcher = (lambda a, b: self._edge_match(a, b)) if self._edge_match else None
        mappings = rustworkx_module().digraph_vf2_mapping(self._G1._rx,
                                                          self._G2._rx,
                                                          node_matcher=node_matcher,
                                                          edge_matcher=edge_matcher,
                                                          subgraph=True,
                                                          induced=True)
        node_at_1, node_at_2 = self._G1._index.node_at, self._G2._index.node_at
        for mapping in mappings:
            yield {node_at_1(a): node_at_2(b) for a, b in mapping.items()}


def run_natively_on_networkx(networkx_func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    For an algorithm whose own cost is O(V+E), reuses a real networkx graph as-is instead of
    converting it to a temporary rustworkx one. _coerce is itself O(V+E) but in Python-level FFI
    calls, so for a linear algorithm coercing can never pay off; only super-linear algorithms
    (all_simple_paths, simple_cycles, VF2) can amortize a conversion and those keep coercing.

    :param networkx_func: The networkx implementation to run on a real networkx graph.
    :return: A decorator for the backend method handling handles.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(method)
        def wrapper(self, G, *args, **kwargs):
            if isinstance(G, networkx.Graph):
                return networkx_func(G, *args, **kwargs)
            return method(self, G, *args, **kwargs)

        return wrapper

    return decorator


def coerce_to_handle(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Passes the method its graph argument as a RustworkxGraphHandle, converting a real networkx one
    on the way in (see _coerce). Composes under run_natively_on_networkx, which short-circuits
    before this ever runs.

    :param method: A backend method taking (self, G, ...).
    :return: The wrapped method.
    """
    if inspect.isgeneratorfunction(method):

        @functools.wraps(method)
        def wrapper(self, G, *args, **kwargs):
            # `yield from`, so the conversion stays as lazy as it was inside the generator body.
            yield from method(self, _coerce(G), *args, **kwargs)
    else:

        @functools.wraps(method)
        def wrapper(self, G, *args, **kwargs):
            return method(self, _coerce(G), *args, **kwargs)

    return wrapper


class RustworkxBackend:
    name = 'rustworkx'

    def new_digraph(self) -> RustworkxGraphHandle:
        return RustworkxGraphHandle(multigraph=False)

    def new_multidigraph(self) -> RustworkxGraphHandle:
        return RustworkxGraphHandle(multigraph=True)

    @run_natively_on_networkx(networkx.has_path)
    @coerce_to_handle
    def has_path(self, G: RustworkxGraphHandle, source: Any, target: Any) -> bool:
        source_idx = index_of_or_raise(G, source, networkx.NodeNotFound, f'Source {source} is not in G')
        target_idx = index_of_or_raise(G, target, networkx.NodeNotFound, f'Target {target} is not in G')
        # networkx counts the trivial length-0 path, so has_path(G, x, x) is True for any node in
        # G; rustworkx.has_path returns False. symbol-write-scopes' _find_dominating_write probes
        # a loop guard's self-reachability this way, and a False swaps two loops' SSA suffixes.
        if source_idx == target_idx:
            return True
        return rustworkx_module().has_path(G._rx, source_idx, target_idx)

    @run_natively_on_networkx(networkx.immediate_dominators)
    @coerce_to_handle
    def immediate_dominators(self, G: RustworkxGraphHandle, start: Any) -> Dict[Any, Any]:
        start_idx = index_of_or_raise(G, start, networkx.NetworkXError, f'{start} is not in G')
        idom = rustworkx_module().immediate_dominators(G._rx, start_idx)
        return dict(zip(G.nodes_at(idom.keys()), G.nodes_at(idom.values())))

    @run_natively_on_networkx(networkx.weakly_connected_components)
    @coerce_to_handle
    def weakly_connected_components(self, G: RustworkxGraphHandle) -> Iterator[Set[Any]]:
        # lazy generator like real networkx; each component is still an eager set, as there too.
        for comp in rustworkx_module().weakly_connected_components(G._rx):
            yield set(G.nodes_at(comp))

    @run_natively_on_networkx(networkx.topological_sort)
    @coerce_to_handle
    def topological_sort(self, G: RustworkxGraphHandle) -> Iterator[Any]:
        # Kahn-by-generation in Python, not rustworkx.topological_sort: a DAG has many valid orders
        # and rustworkx picks a different one than networkx. Not cosmetic -- state_fusion.py and
        # sdfg_nesting.py do `next(n for n in order if ...)`, so a different order builds a
        # structurally different SDFG. Ties must break by INSERTION order, which rustworkx's
        # topological_generations does not do (it orders by node index). Both O(V+E).
        in_degree = {}
        frontier = []
        for node in G.nodes():
            degree = G.in_degree(node)
            if degree:
                in_degree[node] = degree
            else:
                frontier.append(node)
        emitted = 0
        while frontier:
            generation, frontier = frontier, []
            for node in generation:
                emitted += 1
                yield node
                # One decrement per EDGE, not per neighbour: in_degree counts edges while
                # successors() dedups parallel ones, so a per-neighbour decrement leaves a
                # parallel-edge target above zero and the whole graph looks cyclic.
                for _, child in G.out_edges(node):
                    in_degree[child] -= 1
                    if not in_degree[child]:
                        del in_degree[child]
                        frontier.append(child)
        if emitted != G.number_of_nodes():
            # rustworkx would raise DAGHasCycle here; `except nx.NetworkXUnfeasible:` call sites
            # must work under either backend.
            raise networkx.NetworkXUnfeasible('Graph contains a cycle or graph changed during iteration')

    @coerce_to_handle
    def simple_cycles(self, G: RustworkxGraphHandle) -> Iterator[List[Any]]:
        # lazy generator like real networkx.
        for cycle in rustworkx_module().simple_cycles(G._rx):
            yield G.nodes_at(cycle)

    @run_natively_on_networkx(networkx.find_cycle)
    @coerce_to_handle
    def find_cycle(self, G: RustworkxGraphHandle, source: Any = None) -> List[Tuple[Any, Any]]:
        # networkx's `source` takes a node OR a list of nodes tried in turn; rustworkx takes a
        # single index, so a list is looped here. Unlike networkx, a source=None search on a
        # disconnected graph only searches rustworkx's own starting component; the sole caller
        # always passes an explicit, non-empty source list.
        sources = list(source) if isinstance(source, (list, tuple, set)) else [source]
        for src in sources:
            # unlike the other methods here, networkx does NOT raise for a missing/None source in
            # find_cycle -- it just finds no cycle from it, so skip rather than raise.
            if src is not None and src not in G._index:
                continue
            src_idx = G._index.index_of(src) if src is not None else None
            edges = rustworkx_module().digraph_find_cycle(G._rx, src_idx)
            if edges:
                node_at = G._index.node_at
                return [(node_at(u), node_at(v)) for u, v in edges]
        raise networkx.NetworkXNoCycle('No cycle found.')

    @run_natively_on_networkx(networkx.descendants)
    @coerce_to_handle
    def descendants(self, G: RustworkxGraphHandle, source: Any) -> Set[Any]:
        source_idx = index_of_or_raise(G, source, networkx.NetworkXError, f'The node {source} is not in the digraph.')
        return set(G.nodes_at(rustworkx_module().descendants(G._rx, source_idx)))

    @run_natively_on_networkx(networkx.ancestors)
    @coerce_to_handle
    def ancestors(self, G: RustworkxGraphHandle, source: Any) -> Set[Any]:
        source_idx = index_of_or_raise(G, source, networkx.NetworkXError, f'The node {source} is not in the digraph.')
        return set(G.nodes_at(rustworkx_module().ancestors(G._rx, source_idx)))

    @coerce_to_handle
    def all_simple_paths(self, G: RustworkxGraphHandle, source: Any, target: Any) -> Iterator[List[Any]]:
        # lazy generator like real networkx. networkx only tolerates a missing target when it is
        # iterable (a container of targets); for a plain node it raises NodeNotFound, like a
        # missing source.
        source_idx = index_of_or_raise(G, source, networkx.NodeNotFound, f'source node {source} not in graph')
        if target not in G._index:
            try:
                iter(target)
            except TypeError:
                raise networkx.NodeNotFound(f'target node {target} not in graph')
            return
        target_idx = G._index.index_of(target)
        # networkx counts the trivial length-0 path, so source == target yields exactly
        # [[source]] (same convention as has_path); rustworkx would return nothing.
        if source_idx == target_idx:
            yield [source]
            return
        for path in rustworkx_module().digraph_all_simple_paths(G._rx, source_idx, target_idx):
            yield G.nodes_at(path)

    def transitive_closure(self, G: Any) -> Any:
        # No rustworkx equivalent at all, so a real-networkx input runs directly on networkx with
        # no pointless round-trip; only a native handle goes through the to_networkx() fallback.
        if isinstance(G, RustworkxGraphHandle):
            return _transitive_closure_via_networkx(G, dag_only=False)
        return networkx.transitive_closure(G)

    def transitive_closure_dag(self, G: Any) -> Any:
        if isinstance(G, RustworkxGraphHandle):
            return _transitive_closure_via_networkx(G, dag_only=True)
        return networkx.transitive_closure_dag(G)

    @run_natively_on_networkx(networkx.dfs_edges)
    @coerce_to_handle
    def dfs_edges(self, G: RustworkxGraphHandle, source: Any = None) -> Iterator[Tuple[Any, Any]]:
        # lazy generator like real networkx.
        if source is not None:
            src_idx = index_of_or_raise(G, source, networkx.NetworkXError, f'The node {source} is not in the digraph.')
        else:
            src_idx = None
        node_at = G._index.node_at
        for u, v in rustworkx_module().digraph_dfs_edges(G._rx, src_idx):
            yield (node_at(u), node_at(v))

    @run_natively_on_networkx(networkx.shortest_path_length)
    @coerce_to_handle
    def shortest_path_length(self, G: RustworkxGraphHandle, source: Any, target: Any) -> int:
        source_idx = index_of_or_raise(G, source, networkx.NodeNotFound, f'Source {source} is not in G')
        target_idx = index_of_or_raise(G, target, networkx.NodeNotFound, f'Target {target} is not in G')
        lengths = rustworkx_module().dijkstra_shortest_path_lengths(G._rx,
                                                                    source_idx,
                                                                    edge_cost_fn=lambda _: 1.0,
                                                                    goal=target_idx)
        if target_idx not in lengths:
            raise networkx.NetworkXNoPath(f'No path between {source} and {target}.')
        return int(lengths[target_idx])

    def isomorphism_matcher(self,
                            G1: Any,
                            G2: Any,
                            node_match: Optional[Callable[[Payload, Payload], bool]] = None,
                            edge_match: Optional[Callable[[Payload, Payload], bool]] = None) -> RustworkxDiGraphMatcher:
        # _coerce is a no-op in practice: the only real caller (pattern_matching.py) builds both
        # graphs via graphlib.DiGraph(), so they are already rustworkx-native.
        return RustworkxDiGraphMatcher(_coerce(G1), _coerce(G2), node_match, edge_match)


INSTANCE = RustworkxBackend()
RustworkxGraphHandle.graphlib_backend = INSTANCE
