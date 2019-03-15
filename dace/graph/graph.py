""" Graph and multigraph implementations for DaCe. """

from collections import deque, OrderedDict
import itertools
import networkx as nx
from dace.types import deduplicate


class NodeNotFoundError(Exception):
    pass


class EdgeNotFoundError(Exception):
    pass


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

    def toJSON(self, indent=0):
        if self._data is None:
            return "null"
        return self._data.toJSON(indent)

    @staticmethod
    def __len__():
        return 3

    def reverse(self):
        self._src, self._dst = self._dst, self._src


class MultiEdge(Edge):
    def __init__(self, src, dst, data, key):
        super(MultiEdge, self).__init__(src, dst, data)
        self._key = key

    def toJSON(self, indent=0):
        # we loose the key here, what is that even?
        if self._data is None:
            return "null"
        return self._data.toJSON(indent)

    @property
    def key(self):
        return self._key


class MultiConnectorEdge(MultiEdge):
    def __init__(self, src, src_conn, dst, dst_conn, data, key):
        super(MultiConnectorEdge, self).__init__(src, dst, data, key)
        self._src_conn = src_conn
        self._dst_conn = dst_conn

    def toJSON(self, indent=0):
        # we lose the key here, what is that even?
        return ('%s' % ("null"
                        if self._data is None else self._data.toJSON(indent)))

    @property
    def src_conn(self):
        return self._src_conn

    @property
    def src_connector(self):
        return self._src_conn

    @property
    def dst_conn(self):
        return self._dst_conn

    @property
    def dst_connector(self):
        return self._dst_conn

    def __iter__(self):
        yield self._src
        yield self._src_conn
        yield self._dst
        yield self._dst_conn
        yield self._data

    @staticmethod
    def __len__():
        return 5


class Graph(object):
    def _not_implemented_error(self):
        return NotImplementedError("Not implemented for " + str(type(self)))

    def toJSON(self, indent=0):
        json = " " * indent + "{\n"
        indent += 2
        json += " " * indent + "\"type\": \"" + type(self).__name__ + "\",\n"
        json += " " * indent + "\"nodes\": [\n"
        indent += 2
        for n in self.nodes():
            json += " " * indent + "{\n"
            indent += 2
            json += " " * indent + "\"id\" : \"" + str(
                self.node_id(n)) + "\",\n"
            json += " " * indent + "\"attributes\" : " + n.toJSON(indent) + "\n"
            indent -= 2
            if n == self.nodes()[-1]:
                json += " " * indent + "}\n"
            else:
                json += " " * indent + "},\n"
        indent -= 2
        json += " " * indent + "],\n"

        json += " " * indent + "\"edges\": [\n"
        for e in self.edges():
            json += " " * indent + "{\n"
            indent += 2
            json += " " * indent + "\"src\" : \"" + str(self.node_id(
                e.src)) + "\",\n"
            if isinstance(e, MultiConnectorEdge):
                json += " " * indent + '"src_connector" : "%s",\n' % e.src_conn
            json += " " * indent + "\"dst\" : \"" + str(self.node_id(
                e.dst)) + "\",\n"
            if isinstance(e, MultiConnectorEdge):
                json += " " * indent + '"dst_connector" : "%s",\n' % e.dst_conn
            json += " " * indent + "\"attributes\" : " + e.toJSON(indent) + "\n"
            indent -= 2
            if e == self.edges()[-1]:
                json += " " * indent + "}\n"
            else:
                json += " " * indent + "},\n"
        indent -= 2
        json += " " * indent + "]\n"
        json += " " * indent + "}\n"
        return json

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
        """Returns a numeric node ID that corresponds to the node index in the
           internal graph representation (unique)."""
        for i, n in enumerate(self.nodes()):
            if node == n:
                return i
        raise NodeNotFoundError(node)

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

    def in_degree(self, node):
        """Returns the number of incoming edges to the specified node."""
        raise self._not_implemented_error()

    def out_degree(self, node):
        """Returns the number of outgoing edges from the specified node."""
        raise self._not_implemented_error()

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

    def dfs_edges(G, source, condition=None):
        """Traverse a graph (DFS) with an optional condition to filter out nodes
        """
        if isinstance(source, list): nodes = source
        else: nodes = [source]
        visited = set()
        for start in nodes:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, G.out_edges(start).__iter__())]
            while stack:
                parent, children = stack[-1]
                try:
                    e = next(children)
                    if e.dst not in visited:
                        visited.add(e.dst)
                        if condition is None or condition(
                                e.src, e.dst, e.data):
                            yield e
                            stack.append((e.dst,
                                          G.out_edges(e.dst).__iter__()))
                except StopIteration:
                    stack.pop()

    def source_nodes(self):
        """Returns nodes with no incoming edges."""
        return [n for n in self.nodes() if self.in_degree(n) == 0]

    def sink_nodes(self):
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

    def all_simple_paths(self, source_node, dest_node):
        """ Finds all simple paths (with no repeating nodes) from source_node
            to dest_node """
        return nx.all_simple_paths(self._nx, source_node, dest_node)


class SubgraphView(Graph):
    def __init__(self, graph, subgraph_nodes):
        self._graph = graph
        self._subgraph_nodes = subgraph_nodes
        self._parallel_parent = None

    def is_parallel(self):
        return self._parallel_parent != None

    def set_parallel_parent(self, parallel_parent):
        self._parallel_parent = parallel_parent

    def get_parallel_parent(self):
        return self._parallel_parent

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


class DiGraph(Graph):
    def __init__(self):
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
        self._nx = nx.MultiDiGraph()

    @staticmethod
    def _from_nx(edge):
        return MultiEdge(edge[0], edge[1], edge[3]["data"], edge[2])

    def add_edge(self, source, destination, data):
        key = self._nx.add_edge(source, destination, data=data)
        return (source, destination, data, key)

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
                                  edge[3]["dst_conn"], edge[3]["data"],
                                  edge[2])

    def add_edge(self, source, src_connector, destination, dst_connector,
                 data):
        key = self._nx.add_edge(
            source,
            destination,
            data=data,
            src_conn=src_connector,
            dst_conn=dst_connector)
        return (source, src_connector, destination, dst_connector, data, key)

    def remove_edge(self, edge):
        self._nx.remove_edge(edge[0], edge[1], edge.key)

    def is_multigraph(self):
        return True


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
        key = self._nx.add_edge(
            src, dst, data=data, src_conn=src_conn, dst_conn=dst_conn)
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
