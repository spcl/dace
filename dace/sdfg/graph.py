# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" File containing DaCe-serializable versions of graphs, nodes, and edges. """

from collections import deque, OrderedDict
import itertools
import networkx as nx
from dace.dtypes import deduplicate
import dace.serialize
from typing import Any, List


class NodeNotFoundError(Exception):
    pass


class EdgeNotFoundError(Exception):
    pass


@dace.serialize.serializable
class Edge(object):
    def __init__(self, src, dst, data):
        self._src = src
        self._dst = dst
        self._data = data

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def data(self):
        return self._data

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

        ret = Edge(
            json_obj['src'], json_obj['dst'],
            dace.serialize.from_json(json_obj['attributes']['data'], context))

        return ret

    @staticmethod
    def __len__():
        return 3

    def reverse(self):
        self._src, self._dst = self._dst, self._src


@dace.serialize.serializable
class MultiEdge(Edge):
    def __init__(self, src, dst, data, key):
        super(MultiEdge, self).__init__(src, dst, data)
        self._key = key

    @property
    def key(self):
        return self._key


@dace.serialize.serializable
class MultiConnectorEdge(MultiEdge):
    def __init__(self, src, src_conn, dst, dst_conn, data, key):
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

    def __iter__(self):
        yield self._src
        yield self._src_conn
        yield self._dst
        yield self._dst_conn
        yield self._data

    @staticmethod
    def __len__():
        return 5


@dace.serialize.serializable
class Graph(object):
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

    def nodes(self):
        """Returns an iterable to internal graph nodes."""
        raise self._not_implemented_error()

    def edges(self):
        """Returns an iterable to internal graph edges."""
        raise self._not_implemented_error()

    def in_edges(self, node):
        """Returns an iterable to Edge objects."""
        raise self._not_implemented_error()

    def out_edges(self, node):
        """Returns an iterable to Edge objects."""
        raise self._not_implemented_error()

    def __getitem__(self, node):
        """ Returns an iterable to neighboring nodes. """
        return (e.dst for e in self.out_edges(node))

    def all_edges(self, *nodes):
        """Returns an iterable to incoming and outgoing Edge objects."""
        result = set()
        for node in nodes:
            result.update(self.in_edges(node))
            result.update(self.out_edges(node))
        return list(result)

    def add_node(self, node):
        """Adds node to the graph."""
        raise self._not_implemented_error()

    def add_nodes_from(self, node_list):
        """Adds nodes from an iterable to the graph"""
        for node in node_list:
            self.add_node(node)

    def node_id(self, node):
        """Returns a numeric ID that corresponds to the node index in the
           internal graph representation (unique)."""
        for i, n in enumerate(self.nodes()):
            if node == n:
                return i
        raise NodeNotFoundError(node)

    def edge_id(self, edge):
        """Returns a numeric ID that corresponds to the edge index in the
           internal graph representation (unique)."""
        for i, e in enumerate(self.edges()):
            if edge == e:
                return i
        raise EdgeNotFoundError(edge)

    def add_edge(self, source, destination, data):
        """Adds an edge to the graph containing the specified data.
        Returns the added edge."""
        raise self._not_implemented_error()

    def remove_node(self, node):
        """Removes the specified node."""
        raise self._not_implemented_error()

    def remove_nodes_from(self, node_list):
        """Removes the nodes specified in an iterable."""
        for node in node_list:
            self.remove_node(node)

    def remove_edge(self, edge):
        """Removes the specified Edge object."""
        raise self._not_implemented_error()

    def edges_between(self, source, destination):
        """Returns all edges that connect source and destination directly"""
        raise self._not_implemented_error()

    def predecessors(self, node):
        """Returns an iterable of nodes that have edges leading to the passed
        node"""
        return deduplicate([e.src for e in self.in_edges(node)])

    def successors(self, node):
        """Returns an iterable of nodes that have edges leading to the passed
        node"""
        return deduplicate([e.dst for e in self.out_edges(node)])

    def neighbors(self, node):
        return itertools.chain(self.predecessors(node), self.successors(node))

    def in_degree(self, node) -> int:
        """Returns the number of incoming edges to the specified node."""
        raise self._not_implemented_error()

    def out_degree(self, node) -> int:
        """Returns the number of outgoing edges from the specified node."""
        raise self._not_implemented_error()

    def degree(self, node) -> int:
        """Returns the number of edges connected to/from the specified node."""
        return self.in_degree(node) + self.out_degree(node)

    def number_of_nodes(self):
        """Returns the total number of nodes in the graph."""
        raise self._not_implemented_error()

    def number_of_edges(self):
        """Returns the total number of edges in the graph."""
        raise self._not_implemented_error()

    def is_directed(self):
        raise self._not_implemented_error()

    def is_multigraph(self):
        raise self._not_implemented_error()

    def __iter__(self):
        return iter(self.nodes())

    def __len__(self):
        """ Returns the total number of nodes in the graph (nx compatibility)"""
        return self.number_of_nodes()

    def bfs_edges(self, node, reverse=False):
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
            edges = (self.out_edges(node)
                     if not reverse else self.in_edges(node))
            for e in edges:
                next_node = e.dst if not reverse else e.src
                if next_node not in visited:
                    queue.append(next_node)
                yield e

    def dfs_edges(self, source, condition=None):
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
                    if e.dst not in visited:
                        visited.add(e.dst)
                        if condition is None or condition(e.src, e.dst, e.data):
                            yield e
                            stack.append(
                                (e.dst, self.out_edges(e.dst).__iter__()))
                except StopIteration:
                    stack.pop()

    def source_nodes(self) -> List[Any]:
        """Returns nodes with no incoming edges."""
        return [n for n in self.nodes() if self.in_degree(n) == 0]

    def sink_nodes(self) -> List[Any]:
        """Returns nodes with no outgoing edges."""
        return [n for n in self.nodes() if self.out_degree(n) == 0]

    def topological_sort(self, source=None):
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

    def all_simple_paths(self, source_node, dest_node, as_edges=False):
        """ 
        Finds all simple paths (with no repeating nodes) from source_node
        to dest_node.
        :param source_node: Node to start from.
        :param dest_node: Node to end at.
        :param as_edges: If True, returns list of edges instead of nodes.
        """
        if as_edges:
            for path in map(
                    nx.utils.pairwise,
                    nx.all_simple_paths(self._nx, source_node, dest_node)):
                yield [
                    Edge(e[0], e[1], self._nx.edges[e]['data']) for e in path
                ]
        else:
            return nx.all_simple_paths(self._nx, source_node, dest_node)

    def all_nodes_between(self, begin, end):
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
            seen.add(n)
            # Keep chasing all paths to reach the end node
            node_out_edges = self.out_edges(n)
            if len(node_out_edges) == 0:
                # We traversed to the end without finding the end
                return None
            for e in node_out_edges:
                next_node = e.dst
                if next_node != end and next_node not in seen:
                    to_visit.append(next_node)
        return seen


@dace.serialize.serializable
class SubgraphView(Graph):
    def __init__(self, graph, subgraph_nodes):
        super().__init__()
        self._graph = graph
        self._subgraph_nodes = subgraph_nodes

    def nodes(self):
        return self._subgraph_nodes

    def edges(self):
        return [
            e for e in self._graph.edges()
            if e.src in self._subgraph_nodes and e.dst in self._subgraph_nodes
        ]

    def in_edges(self, node):
        if node not in self._subgraph_nodes:
            raise NodeNotFoundError

        return [
            e for e in self._graph.in_edges(node)
            if e.src in self._subgraph_nodes
        ]

    def out_edges(self, node):
        if node not in self._subgraph_nodes:
            raise NodeNotFoundError

        return [
            e for e in self._graph.out_edges(node)
            if e.dst in self._subgraph_nodes
        ]

    def add_node(self, node):
        raise PermissionError

    def add_nodes_from(self, node_list):
        raise PermissionError

    def node_id(self, node):
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
class DiGraph(Graph):
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


class MultiDiGraph(DiGraph):
    def __init__(self):
        super().__init__()
        self._nx = nx.MultiDiGraph()

    @staticmethod
    def _from_nx(edge):
        return MultiEdge(edge[0], edge[1], edge[3]["data"], edge[2])

    def add_edge(self, source, destination, data):
        key = self._nx.add_edge(source, destination, data=data)
        return source, destination, data, key

    def remove_edge(self, edge):
        self._nx.remove_edge(edge[0], edge[1], edge.key)

    def is_multigraph(self):
        return True


class MultiDiConnectorGraph(MultiDiGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _from_nx(edge):
        return MultiConnectorEdge(edge[0], edge[3]["src_conn"], edge[1],
                                  edge[3]["dst_conn"], edge[3]["data"], edge[2])

    def add_edge(self, source, src_connector, destination, dst_connector, data):
        key = self._nx.add_edge(source,
                                destination,
                                data=data,
                                src_conn=src_connector,
                                dst_conn=dst_connector)
        return source, src_connector, destination, dst_connector, data, key

    def remove_edge(self, edge):
        self._nx.remove_edge(edge[0], edge[1], edge.key)

    def is_multigraph(self):
        return True


@dace.serialize.serializable
class OrderedDiGraph(Graph):
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

    def node(self, id):
        return list(self._nodes.keys())[id]

    def nodes(self):
        return list(self._nodes.keys())

    def edges(self):
        return list(self._edges.values())

    def in_edges(self, node):
        return list(self._nodes[node][0].values())

    def out_edges(self, node):
        return list(self._nodes[node][1].values())

    def add_node(self, node):
        if node in self._nodes:
            raise RuntimeError("Duplicate node added")
        self._nodes[node] = (OrderedDict(), OrderedDict())
        self._nx.add_node(node)

    def add_edge(self, src, dst, data):
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
        return self._nx.add_edge(src, dst, data=data)

    def remove_node(self, node):
        for edge in itertools.chain(self.in_edges(node), self.out_edges(node)):
            self.remove_edge(edge)
        del self._nodes[node]
        self._nx.remove_node(node)

    def remove_edge(self, edge):
        src = edge.src
        dst = edge.dst
        t = (src, dst)
        self._nx.remove_edge(src, dst)
        del self._nodes[src][1][t]
        del self._nodes[dst][0][t]
        del self._edges[t]

    def in_degree(self, node):
        return len(self._nodes[node][0])

    def out_degree(self, node):
        return len(self._nodes[node][1])

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

    def edges_between(self, source, destination):
        if source not in self.nodes(): return []
        return [e for e in self.out_edges(source) if e.dst == destination]

    def reverse(self):
        """Reverses source and destination of all edges in the graph"""
        raise self._not_implemented_error()


class OrderedMultiDiGraph(OrderedDiGraph):
    """ Directed multigraph where nodes and edges are returned in the order
        they were added. """
    def __init__(self):
        self._nx = nx.MultiDiGraph()
        # {node: ({in edge: edge}, {out edge: edge})}
        self._nodes = OrderedDict()
        # {edge: edge}
        self._edges = OrderedDict()

    def add_edge(self, src, dst, data):
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

    def remove_edge(self, edge):
        del self._edges[edge]
        del self._nodes[edge.src][1][edge]
        del self._nodes[edge.dst][0][edge]
        self._nx.remove_edge(edge.src, edge.dst, edge.key)

    def reverse(self):
        self._nx.reverse(False)
        for e in self._edges.keys():
            e.reverse()
        for n, (in_edges, out_edges) in self._nodes.items():
            self._nodes[n] = (out_edges, in_edges)

    def is_multigraph(self):
        return True


class OrderedMultiDiConnectorGraph(OrderedMultiDiGraph):
    """ Directed multigraph with node connectors (SDFG states), where nodes
        and edges are returned in the order they were added. """
    def __init__(self):
        super().__init__()

    def add_edge(self, src, src_conn, dst, dst_conn, data):
        key = self._nx.add_edge(src,
                                dst,
                                data=data,
                                src_conn=src_conn,
                                dst_conn=dst_conn)
        edge = MultiConnectorEdge(src, src_conn, dst, dst_conn, data, key)
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)
        self._nodes[src][1][edge] = edge
        self._nodes[dst][0][edge] = edge
        self._edges[edge] = edge
        return edge

    def add_nedge(self, src, dst, data):
        """ Adds an edge without (value=None) connectors. """
        return self.add_edge(src, None, dst, None, data)

    def remove_edge(self, edge):
        del self._edges[edge]
        del self._nodes[edge.src][1][edge]
        del self._nodes[edge.dst][0][edge]
        self._nx.remove_edge(edge.src, edge.dst, edge.key)

    def reverse(self):
        self._nx.reverse(False)
        for e in self._edges.keys():
            e.reverse()
        for n, (in_edges, out_edges) in self._nodes.items():
            self._nodes[n] = (out_edges, in_edges)

    def is_multigraph(self):
        return True
