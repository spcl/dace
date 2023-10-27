# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" File containing DaCe-serializable versions of graphs, nodes, and edges. """

from collections import deque, OrderedDict
import itertools
import networkx as nx
from dace.dtypes import deduplicate
import dace.serialize
from typing import Any, Callable, Generic, Iterable, List, Sequence, TypeVar, Union


class NodeNotFoundError(Exception):
    pass


class EdgeNotFoundError(Exception):
    pass


T = TypeVar('T')
NodeT = TypeVar('NodeT')
EdgeT = TypeVar('EdgeT')


@dace.serialize.serializable
class Edge(Generic[T]):
    def __init__(self, src, dst, data: T):
        self._src = src
        self._dst = dst
        self._data: T = data

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def data(self) -> T:
        return self._data

    @data.setter
    def data(self, new_data: T):
        self._data = new_data

    def __iter__(self):
        yield self._src
        yield self._dst
        yield self._data

    def to_json(self, parent_graph):
        memlet_ret = self.data.to_json()
        ret = {
            'type': type(self).__name__,
            'attributes': {
                'data': memlet_ret
            },
            'src': str(parent_graph.node_id(self.src)),
            'dst': str(parent_graph.node_id(self.dst)),
        }

        return ret

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Edge":
            raise TypeError("Invalid data type")

        ret = Edge(json_obj['src'], json_obj['dst'], dace.serialize.from_json(json_obj['attributes']['data'], context))

        return ret

    @staticmethod
    def __len__():
        return 3

    def reverse(self):
        self._src, self._dst = self._dst, self._src


@dace.serialize.serializable
class MultiEdge(Edge, Generic[T]):
    def __init__(self, src, dst, data: T, key):
        super(MultiEdge, self).__init__(src, dst, data)
        self._key = key

    @property
    def key(self):
        return self._key


@dace.serialize.serializable
class MultiConnectorEdge(MultiEdge, Generic[T]):
    def __init__(self, src, src_conn: str, dst, dst_conn: str, data: T, key):
        super(MultiConnectorEdge, self).__init__(src, dst, data, key)
        self._src_conn = src_conn
        self._dst_conn = dst_conn

    def to_json(self, parent_graph):
        ret = super().to_json(parent_graph)

        ret['dst_connector'] = self.dst_conn
        ret['src_connector'] = self.src_conn

        ret['type'] = "MultiConnectorEdge"

        return ret

    @staticmethod
    def from_json(json_obj, context=None):

        sdfg = context['sdfg_state']
        if sdfg is None:
            raise Exception("parent_graph must be defined for this method")
        data = dace.serialize.from_json(json_obj['attributes']['data'], context)
        src_nid = json_obj['src']
        dst_nid = json_obj['dst']

        dst = sdfg.nodes()[int(dst_nid)]
        src = sdfg.nodes()[int(src_nid)]

        dst_conn = json_obj['dst_connector']
        src_conn = json_obj['src_connector']

        # Auto-create key (used when uniquely identifying networkx multigraph
        # edges)
        ret = MultiConnectorEdge(src, src_conn, dst, dst_conn, data, None)

        return ret

    @property
    def src_conn(self):
        return self._src_conn

    @src_conn.setter
    def src_conn(self, val):
        self._src_conn = val

    @property
    def dst_conn(self):
        return self._dst_conn

    @dst_conn.setter
    def dst_conn(self, val):
        self._dst_conn = val

    @property
    def data(self) -> T:
        return self._data

    @data.setter
    def data(self, new_data: T):
        self._data = new_data

    def __iter__(self):
        yield self._src
        yield self._src_conn
        yield self._dst
        yield self._dst_conn
        yield self._data

    @staticmethod
    def __len__():
        return 5

    def __repr__(self):
        return f"{self.src}:{self.src_conn}  -({self.data})->  {self.dst}:{self.dst_conn}"


@dace.serialize.serializable
class Graph(Generic[NodeT, EdgeT]):
    def _not_implemented_error(self):
        return NotImplementedError("Not implemented for " + str(type(self)))

    def to_json(self):
        ret = {
            'type': type(self).__name__,
            'attributes': dace.serialize.all_properties_to_json(self),
            'nodes': [n.to_json(self) for n in self.nodes()],
            'edges': [e.to_json(self) for e in self.edges()],
        }
        return ret

    @property
    def nx(self):
        """ Returns a networkx version of this graph if available. """
        raise TypeError("No networkx version exists for this graph type")

    def nodes(self) -> Iterable[NodeT]:
        """Returns an iterable to internal graph nodes."""
        raise self._not_implemented_error()

    def edges(self) -> Iterable[Edge[EdgeT]]:
        """Returns an iterable to internal graph edges."""
        raise self._not_implemented_error()

    def in_edges(self, node: NodeT) -> Iterable[Edge[EdgeT]]:
        """Returns an iterable to Edge objects."""
        raise self._not_implemented_error()

    def out_edges(self, node: NodeT) -> Iterable[Edge[EdgeT]]:
        """Returns an iterable to Edge objects."""
        raise self._not_implemented_error()

    def __getitem__(self, node: NodeT) -> Iterable[NodeT]:
        """ Returns an iterable to neighboring nodes. """
        return (e.dst for e in self.out_edges(node))

    def all_edges(self, *nodes: NodeT) -> Iterable[Edge[EdgeT]]:
        """Returns an iterable to incoming and outgoing Edge objects."""
        result = set()
        for node in nodes:
            result.update(self.in_edges(node))
            result.update(self.out_edges(node))
        return list(result)

    def add_node(self, node: NodeT):
        """Adds node to the graph."""
        raise self._not_implemented_error()

    def add_nodes_from(self, node_list: Sequence[NodeT]):
        """Adds nodes from an iterable to the graph"""
        for node in node_list:
            self.add_node(node)

    def node_id(self, node: NodeT) -> int:
        """Returns a numeric ID that corresponds to the node index in the
           internal graph representation (unique)."""
        for i, n in enumerate(self.nodes()):
            if node == n:
                return i
        raise NodeNotFoundError(node)

    def edge_id(self, edge: Edge[EdgeT]) -> int:
        """Returns a numeric ID that corresponds to the edge index in the
           internal graph representation (unique)."""
        for i, e in enumerate(self.edges()):
            if edge == e:
                return i
        raise EdgeNotFoundError(edge)

    def add_edge(self, source: NodeT, destination: NodeT, data: EdgeT):
        """Adds an edge to the graph containing the specified data.
        Returns the added edge."""
        raise self._not_implemented_error()

    def remove_node(self, node: NodeT):
        """Removes the specified node."""
        raise self._not_implemented_error()

    def remove_nodes_from(self, node_list: Sequence[NodeT]):
        """Removes the nodes specified in an iterable."""
        for node in node_list:
            self.remove_node(node)

    def remove_edge(self, edge: Edge[EdgeT]):
        """Removes the specified Edge object."""
        raise self._not_implemented_error()

    def edges_between(self, source: NodeT, destination: NodeT) -> Iterable[Edge[EdgeT]]:
        """Returns all edges that connect source and destination directly"""
        raise self._not_implemented_error()

    def predecessors(self, node: NodeT) -> Iterable[NodeT]:
        """Returns an iterable of nodes that have edges leading to the passed
        node"""
        return deduplicate([e.src for e in self.in_edges(node)])

    def successors(self, node: NodeT) -> Iterable[NodeT]:
        """Returns an iterable of nodes that have edges leading to the passed
        node"""
        return deduplicate([e.dst for e in self.out_edges(node)])

    def neighbors(self, node: NodeT) -> Iterable[NodeT]:
        return itertools.chain(self.predecessors(node), self.successors(node))

    def in_degree(self, node: NodeT) -> int:
        """Returns the number of incoming edges to the specified node."""
        raise self._not_implemented_error()

    def out_degree(self, node: NodeT) -> int:
        """Returns the number of outgoing edges from the specified node."""
        raise self._not_implemented_error()

    def degree(self, node: NodeT) -> int:
        """Returns the number of edges connected to/from the specified node."""
        return self.in_degree(node) + self.out_degree(node)

    def number_of_nodes(self) -> int:
        """Returns the total number of nodes in the graph."""
        raise self._not_implemented_error()

    def number_of_edges(self) -> int:
        """Returns the total number of edges in the graph."""
        raise self._not_implemented_error()

    def is_directed(self) -> bool:
        raise self._not_implemented_error()

    def is_multigraph(self) -> bool:
        raise self._not_implemented_error()

    def __iter__(self) -> Iterable[NodeT]:
        return iter(self.nodes())

    def __len__(self) -> int:
        """ Returns the total number of nodes in the graph (nx compatibility)"""
        return self.number_of_nodes()

    def bfs_edges(self, node: Union[NodeT, Sequence[NodeT]], reverse: bool = False) -> Iterable[Edge[EdgeT]]:
        """Returns a generator over edges in the graph originating from the
        passed node in BFS order"""
        if isinstance(node, (tuple, list)):
            queue = deque(node)
        else:
            queue = deque([node])
        visited = set()
        while len(queue) > 0:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            edges = (self.out_edges(node) if not reverse else self.in_edges(node))
            for e in edges:
                next_node = e.dst if not reverse else e.src
                if next_node not in visited:
                    queue.append(next_node)
                yield e

    def dfs_edges(self,
                  source: Union[NodeT, Sequence[NodeT]],
                  condition: Callable[[NodeT, NodeT, Any], bool] = None) -> Iterable[Edge[EdgeT]]:
        """Traverse a graph (DFS) with an optional condition to filter out nodes
        """
        if isinstance(source, list): nodes = source
        else: nodes = [source]
        visited = set()
        for start in nodes:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, self.out_edges(start).__iter__())]
            while stack:
                parent, children = stack[-1]
                try:
                    e = next(children)
                    to_yield = condition is None or condition(e.src, e.dst, e.data)
                    if e.dst not in visited:
                        visited.add(e.dst)
                        if to_yield:
                            stack.append((e.dst, self.out_edges(e.dst).__iter__()))
                    if to_yield:
                        yield e
                except StopIteration:
                    stack.pop()

    def source_nodes(self) -> List[NodeT]:
        """Returns nodes with no incoming edges."""
        return [n for n in self.nodes() if self.in_degree(n) == 0]

    def sink_nodes(self) -> List[NodeT]:
        """Returns nodes with no outgoing edges."""
        return [n for n in self.nodes() if self.out_degree(n) == 0]

    def topological_sort(self, source: NodeT = None) -> Sequence[NodeT]:
        """Returns nodes in topological order iff the graph contains exactly
        one node with no incoming edges."""
        if source is not None:
            sources = [source]
        else:
            sources = self.source_nodes()
            if len(sources) == 0:
                sources = [self.nodes()[0]]
                #raise RuntimeError("No source nodes found")
            if len(sources) > 1:
                sources = [self.nodes()[0]]
                #raise RuntimeError("Multiple source nodes found")
        seen = OrderedDict()  # No OrderedSet in Python
        queue = deque(sources)
        while len(queue) > 0:
            node = queue.popleft()
            seen[node] = None
            for e in self.out_edges(node):
                succ = e.dst
                if succ not in seen:
                    seen[succ] = None
                    queue.append(succ)
        return seen.keys()

    def all_simple_paths(self,
                         source_node: NodeT,
                         dest_node: NodeT,
                         as_edges: bool = False) -> Iterable[Sequence[Union[Edge[EdgeT], NodeT]]]:
        """ 
        Finds all simple paths (with no repeating nodes) from ``source_node``
        to ``dest_node``.

        :param source_node: Node to start from.
        :param dest_node: Node to end at.
        :param as_edges: If True, returns list of edges instead of nodes.
        """
        if as_edges:
            for path in map(nx.utils.pairwise, nx.all_simple_paths(self._nx, source_node, dest_node)):
                yield [Edge(e[0], e[1], self._nx.edges[e]['data']) for e in path]
        else:
            return nx.all_simple_paths(self._nx, source_node, dest_node)

    def all_nodes_between(self, begin: NodeT, end: NodeT) -> Sequence[NodeT]:
        """Finds all nodes between begin and end. Returns None if there is any
           path starting at begin that does not reach end."""
        to_visit = [begin]
        seen = set()
        while len(to_visit) > 0:
            n = to_visit.pop()
            if n == end:
                continue  # We've reached the end node
            if n in seen:
                continue  # We've already visited this node
            if n != begin:
                seen.add(n)
            # Keep chasing all paths to reach the end node
            node_out_edges = self.out_edges(n)
            if len(node_out_edges) == 0:
                # We traversed to the end without finding the end
                return set()
            for e in node_out_edges:
                next_node = e.dst
                if next_node != end and next_node not in seen:
                    to_visit.append(next_node)
        return seen


@dace.serialize.serializable
class SubgraphView(Graph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    def __init__(self, graph: Graph[NodeT, EdgeT], subgraph_nodes: Sequence[NodeT]):
        super().__init__()
        self._graph = graph
        self._subgraph_nodes = list(sorted(subgraph_nodes, key=lambda n: graph.node_id(n)))

    def nodes(self) -> Sequence[NodeT]:
        return self._subgraph_nodes

    def edges(self) -> List[Edge[EdgeT]]:
        return [e for e in self._graph.edges() if e.src in self._subgraph_nodes and e.dst in self._subgraph_nodes]

    def in_edges(self, node: NodeT) -> List[Edge[EdgeT]]:
        if node not in self._subgraph_nodes:
            raise NodeNotFoundError

        return [e for e in self._graph.in_edges(node) if e.src in self._subgraph_nodes]

    def out_edges(self, node: NodeT) -> List[Edge[EdgeT]]:
        if node not in self._subgraph_nodes:
            raise NodeNotFoundError

        return [e for e in self._graph.out_edges(node) if e.dst in self._subgraph_nodes]

    def add_node(self, node):
        raise PermissionError

    def add_nodes_from(self, node_list):
        raise PermissionError

    def node_id(self, node: NodeT) -> int:
        if node not in self._subgraph_nodes:
            raise NodeNotFoundError
        return self._graph.node_id(node)

    def add_edge(self, source, destination, data):
        raise PermissionError

    def remove_node(self, node):
        raise PermissionError

    def remove_nodes_from(self, node_list):
        raise PermissionError

    def remove_edge(self, edge):
        raise PermissionError

    def edges_between(self, source, destination):
        if source not in self._subgraph_nodes or \
           destination not in self._subgraph_nodes:
            raise NodeNotFoundError
        return self._graph.edges_between(source, destination)

    def in_degree(self, node):
        return len(self.in_edges(node))

    def out_degree(self, node):
        return len(self.out_edges(node))

    def number_of_nodes(self):
        return len(self._subgraph_nodes)

    def number_of_edges(self):
        return len(self.edges())

    def is_directed(self):
        return self._graph.is_directed()

    def is_multigraph(self):
        return self._graph.is_multigraph()

    @property
    def graph(self):
        return self._graph


@dace.serialize.serializable
class DiGraph(Graph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    def __init__(self):
        super().__init__()
        self._nx = nx.DiGraph()

    def nodes(self):
        return self._nx.nodes()

    @staticmethod
    def _from_nx(edge):
        return Edge(edge[0], edge[1], edge[2]["data"])

    def edges(self):
        return [DiGraph._from_nx(e) for e in self._nx.edges()]

    def in_edges(self, node):
        return [DiGraph._from_nx(e) for e in self._nx.in_edges()]

    def out_edges(self, node):
        return [DiGraph._from_nx(e) for e in self._nx.out_edges()]

    def add_node(self, node):
        return self._nx.add_node(node)

    def add_edge(self, source, destination, data):
        return self._nx.add_edge(source, destination, data=data)

    def remove_node(self, node):
        self._nx.remove_node(node)

    def remove_edge(self, edge):
        self._nx.remove_edge(edge[0], edge[1])

    def in_degree(self, node):
        return self._nx.in_degree(node)

    def out_degree(self, node):
        return self._nx.out_degree(node)

    def number_of_nodes(self):
        return self._nx.number_of_nodes()

    def number_of_edges(self):
        return self._nx.number_of_edges()

    def is_directed(self):
        return True

    def is_multigraph(self):
        return False

    def edges_between(self, source, destination):
        return [e for e in self.out_edges(source) if e.dst == destination]

    def find_cycles(self):
        return nx.simple_cycles(self._nx)
    
    def has_cycles(self) -> bool:
        try:
            nx.find_cycle(self._nx, self.source_nodes())
            return True
        except nx.NetworkXNoCycle:
            return False


class MultiDiGraph(DiGraph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    def __init__(self):
        super().__init__()
        self._nx = nx.MultiDiGraph()

    @staticmethod
    def _from_nx(edge) -> MultiEdge[EdgeT]:
        return MultiEdge(edge[0], edge[1], edge[3]["data"], edge[2])

    def add_edge(self, source: NodeT, destination: NodeT, data: EdgeT):
        key = self._nx.add_edge(source, destination, data=data)
        return source, destination, data, key

    def remove_edge(self, edge: MultiEdge[EdgeT]):
        self._nx.remove_edge(edge[0], edge[1], edge.key)

    def is_multigraph(self) -> bool:
        return True


class MultiDiConnectorGraph(MultiDiGraph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _from_nx(edge):
        return MultiConnectorEdge(edge[0], edge[3]["src_conn"], edge[1], edge[3]["dst_conn"], edge[3]["data"], edge[2])

    def add_edge(self, source: NodeT, src_connector: str, destination: NodeT, dst_connector: str, data: EdgeT):
        key = self._nx.add_edge(source, destination, data=data, src_conn=src_connector, dst_conn=dst_connector)
        return source, src_connector, destination, dst_connector, data, key

    def remove_edge(self, edge: MultiConnectorEdge[EdgeT]):
        self._nx.remove_edge(edge[0], edge[1], edge.key)

    def is_multigraph(self) -> bool:
        return True


@dace.serialize.serializable
class OrderedDiGraph(Graph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    """ Directed graph where nodes and edges are returned in the order they
        were added. """
    def __init__(self):
        self._nx = nx.DiGraph()
        # {node: ({in edge: None}, {out edges: None})}
        self._nodes = OrderedDict()
        # {(src, dst): edge}
        self._edges = OrderedDict()

    @property
    def nx(self):
        return self._nx

    def node(self, id: int) -> NodeT:
        try:
            return next(n for i, n in enumerate(self._nodes.keys()) if i == id)
        except StopIteration:
            raise NodeNotFoundError

    def node_id(self, node: NodeT) -> int:
        try:
            return next(i for i, n in enumerate(self._nodes.keys()) if n is node)
        except StopIteration:
            raise NodeNotFoundError(node)

    def nodes(self) -> List[NodeT]:
        return list(self._nodes.keys())

    def edges(self) -> List[Edge[EdgeT]]:
        return list(self._edges.values())

    def in_edges(self, node: NodeT) -> List[Edge[EdgeT]]:
        return list(self._nodes[node][0].values())

    def out_edges(self, node: NodeT) -> List[Edge[EdgeT]]:
        return list(self._nodes[node][1].values())

    def add_node(self, node: NodeT):
        if node in self._nodes:
            raise RuntimeError("Duplicate node added")
        self._nodes[node] = (OrderedDict(), OrderedDict())
        self._nx.add_node(node)

    def add_edge(self, src: NodeT, dst: NodeT, data: EdgeT = None):
        t = (src, dst)
        if t in self._edges:
            raise RuntimeError("Duplicate edge added")
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)
        edge = Edge(src, dst, data)
        self._edges[t] = edge
        self._nodes[src][1][t] = edge
        self._nodes[dst][0][t] = edge
        self._nx.add_edge(src, dst, data=data)
        return edge

    def remove_node(self, node: NodeT):
        try:
            for edge in itertools.chain(self.in_edges(node), self.out_edges(node)):
                self.remove_edge(edge)
            del self._nodes[node]
            self._nx.remove_node(node)
        except KeyError:
            pass

    def remove_edge(self, edge: Edge[EdgeT]):
        src = edge.src
        dst = edge.dst
        t = (src, dst)
        self._nx.remove_edge(src, dst)
        del self._nodes[src][1][t]
        del self._nodes[dst][0][t]
        del self._edges[t]

    def in_degree(self, node):
        return self._nx.in_degree(node)

    def out_degree(self, node):
        return self._nx.out_degree(node)

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)

    def is_directed(self):
        return True

    def is_multigraph(self):
        return False

    def find_cycles(self):
        return nx.simple_cycles(self._nx)
    
    def has_cycles(self) -> bool:
        try:
            nx.find_cycle(self._nx, self.source_nodes())
            return True
        except nx.NetworkXNoCycle:
            return False

    def edges_between(self, source: NodeT, destination: NodeT) -> List[Edge[EdgeT]]:
        if (source, destination) in self._edges:
            return [self._edges[(source, destination)]]
        if source not in self.nodes(): return []
        return [e for e in self.out_edges(source) if e.dst == destination]

    def reverse(self):
        """Reverses source and destination of all edges in the graph"""
        raise self._not_implemented_error()


class OrderedMultiDiGraph(OrderedDiGraph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    """ Directed multigraph where nodes and edges are returned in the order
        they were added. """
    def __init__(self):
        self._nx = nx.MultiDiGraph()
        # {node: ({in edge: edge}, {out edge: edge})}
        self._nodes = OrderedDict()
        # {edge: edge}
        self._edges = OrderedDict()

    def add_edge(self, src: NodeT, dst: NodeT, data: EdgeT) -> MultiEdge[EdgeT]:
        key = self._nx.add_edge(src, dst, data=data)
        edge = MultiEdge(src, dst, data, key)
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)
        self._nodes[src][1][edge] = edge
        self._nodes[dst][0][edge] = edge
        self._edges[edge] = edge
        return edge

    def remove_edge(self, edge: MultiEdge[EdgeT]):
        del self._edges[edge]
        del self._nodes[edge.src][1][edge]
        del self._nodes[edge.dst][0][edge]
        self._nx.remove_edge(edge.src, edge.dst, edge.key)

    def in_edges(self, node) -> List[MultiEdge[EdgeT]]:
        return super().in_edges(node)

    def out_edges(self, node) -> List[MultiEdge[EdgeT]]:
        return super().out_edges(node)

    def edges_between(self, source: NodeT, destination: NodeT) -> List[MultiEdge[EdgeT]]:
        return super().edges_between(source, destination)

    def reverse(self) -> None:
        self._nx.reverse(False)
        for e in self._edges.keys():
            e.reverse()
        for n, (in_edges, out_edges) in self._nodes.items():
            self._nodes[n] = (out_edges, in_edges)

    def is_multigraph(self) -> bool:
        return True


class OrderedMultiDiConnectorGraph(OrderedMultiDiGraph[NodeT, EdgeT], Generic[NodeT, EdgeT]):
    """ Directed multigraph with node connectors (SDFG states), where nodes
        and edges are returned in the order they were added. """
    def __init__(self):
        super().__init__()

    def add_edge(self, src: NodeT, src_conn: str, dst: NodeT, dst_conn: str, data: EdgeT) -> MultiConnectorEdge[EdgeT]:
        key = self._nx.add_edge(src, dst, data=data, src_conn=src_conn, dst_conn=dst_conn)
        edge = MultiConnectorEdge(src, src_conn, dst, dst_conn, data, key)
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)
        self._nodes[src][1][edge] = edge
        self._nodes[dst][0][edge] = edge
        self._edges[edge] = edge
        return edge

    def add_nedge(self, src: NodeT, dst: NodeT, data: EdgeT) -> MultiConnectorEdge[EdgeT]:
        """ Adds an edge without (value=None) connectors. """
        return self.add_edge(src, None, dst, None, data)

    def remove_edge(self, edge: MultiConnectorEdge[EdgeT]):
        del self._edges[edge]
        del self._nodes[edge.src][1][edge]
        del self._nodes[edge.dst][0][edge]
        self._nx.remove_edge(edge.src, edge.dst, edge.key)

    def reverse(self) -> None:
        self._nx.reverse(False)
        for e in self._edges.keys():
            e.reverse()
        for n, (in_edges, out_edges) in self._nodes.items():
            self._nodes[n] = (out_edges, in_edges)

    def in_edges(self, node) -> List[MultiConnectorEdge[EdgeT]]:
        return super().in_edges(node)

    def out_edges(self, node) -> List[MultiConnectorEdge[EdgeT]]:
        return super().out_edges(node)

    def edges_between(self, source: NodeT, destination: NodeT) -> List[MultiConnectorEdge[EdgeT]]:
        return super().edges_between(source, destination)

    def is_multigraph(self) -> bool:
        return True
