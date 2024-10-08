import ast
from itertools import chain

import dace.subsets
from dace import transformation, SDFGState, SDFG, Memlet
from dace.sdfg import nodes
from dace.sdfg.nodes import Tasklet, ExitNode
from dace.transformation.dataflow import MapFusion


class ConstAssignmentMapFusion(MapFusion):
    first_map_exit = transformation.PatternNode(nodes.ExitNode)
    array = transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.PatternNode(nodes.EntryNode)

    # NOTE: `expression()` is inherited.

    @staticmethod
    def consistent_const_assignment_table(graph, en, ex) -> tuple[bool, dict]:
        table = {}
        for n in graph.all_nodes_between(en, ex):
            # Each of the nodes in this map must be...
            if not isinstance(n, Tasklet):
                # ...a tasklet...
                return False, table
            if len(n.code.code) != 1 or not isinstance(n.code.code[0], ast.Assign):
                # ...that assigns...
                return False, table
            op = n.code.code[0]
            if not isinstance(op.value, ast.Constant) or len(op.targets) != 1:
                # ...a constant to a single target.
                return False, table
            const = op.value.value
            for oe in graph.out_edges(n):
                dst = oe.data
                dst_arr = oe.data.data
                if dst_arr in table and table[dst_arr] != const:
                    # A target array can appear multiple times, but it must always be consistently assigned.
                    return False, table
                table[dst] = const
                table[dst_arr] = const
        return True, table

    def map_nodes(self, graph: SDFGState):
        return (graph.entry_node(self.first_map_exit), self.first_map_exit,
                self.second_map_entry, graph.exit_node(self.second_map_entry))

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        first_entry, first_exit, second_entry, second_exit = self.map_nodes(graph)
        # TODO(pratyai): Make a better check for map compatibility.
        if first_entry.map.range != second_entry.map.range or first_entry.map.schedule != second_entry.map.schedule:
            # TODO(pratyai): Make it so that a permutation of the ranges, or even an union of the ranges will work.
            return False

        # Both maps must have consistent constant assignment for the target arrays.
        is_const_assignment, assignments = self.consistent_const_assignment_table(graph, first_entry, first_exit)
        if not is_const_assignment:
            return False
        is_const_assignment, further_assignments = self.consistent_const_assignment_table(graph, second_entry,
                                                                                          second_exit)
        if not is_const_assignment:
            return False
        for k, v in further_assignments.items():
            if k in assignments and v != assignments[k]:
                return False
            assignments[k] = v
        return True

    @staticmethod
    def track_access_nodes(graph: SDFGState, first_exit: ExitNode, second_exit: ExitNode):
        # Track all the access nodes that will survive the purge.
        access_nodes, remove_nodes = {}, set()
        dst_nodes = set(e.dst for e in chain(graph.out_edges(first_exit), graph.out_edges(second_exit)))
        for n in dst_nodes:
            if n.data in access_nodes:
                remove_nodes.add(n)
            else:
                access_nodes[n.data] = n
        for n in remove_nodes:
            assert n.data in access_nodes
            assert access_nodes[n.data] != n
        return access_nodes, remove_nodes

    @staticmethod
    def make_equivalent_connections(first_exit: ExitNode, second_exit: ExitNode):
        # Set up the extra connections on the first node.
        conn_map = {}
        for c, v in second_exit.in_connectors.items():
            assert c.startswith('IN_')
            cbase = c.removeprefix('IN_')
            sc = first_exit.next_connector(cbase)
            conn_map[f"IN_{cbase}"] = f"IN_{sc}"
            conn_map[f"OUT_{cbase}"] = f"OUT_{sc}"
            first_exit.add_in_connector(f"IN_{sc}", dtype=v)
            first_exit.add_out_connector(f"OUT_{sc}", dtype=v)
        for c, v in second_exit.out_connectors.items():
            assert c in conn_map
        return conn_map

    def apply(self, graph: SDFGState, sdfg: SDFG):
        first_entry, first_exit, second_entry, second_exit = self.map_nodes(graph)

        # By now, we know that the two maps are compatible, not reading anything, and just blindly writing constants
        # _consistently_.
        is_const_assignment, assignments = self.consistent_const_assignment_table(graph, first_entry, first_exit)
        assert is_const_assignment

        # Track all the access nodes that will survive the purge.
        access_nodes, remove_nodes = self.track_access_nodes(graph, first_exit, second_exit)

        # Set up the extra connections on the first node.
        conn_map = self.make_equivalent_connections(first_exit, second_exit)

        # Redirect outgoing edges from exit nodes that are going to be invalidated.
        for e in graph.out_edges(first_exit):
            array_name = e.dst.data
            assert array_name in access_nodes
            if access_nodes[array_name] != e.dst:
                graph.add_memlet_path(first_exit, access_nodes[array_name], src_conn=e.src_conn, dst_conn=e.dst_conn,
                                      memlet=Memlet(str(e.data)))
                graph.remove_edge(e)
        for e in graph.out_edges(second_exit):
            array_name = e.dst.data
            assert array_name in access_nodes
            graph.add_memlet_path(first_exit, access_nodes[array_name], src_conn=conn_map[e.src_conn],
                                  dst_conn=e.dst_conn, memlet=Memlet(str(e.data)))
            graph.remove_edge(e)

        # Move the tasklets from the second map into the first map.
        second_tasklets = graph.all_nodes_between(second_entry, second_exit)
        for t in second_tasklets:
            for e in graph.in_edges(t):
                graph.add_memlet_path(first_entry, t, memlet=Memlet())
                graph.remove_edge(e)
            for e in graph.out_edges(t):
                graph.add_memlet_path(e.src, first_exit, src_conn=e.src_conn, dst_conn=conn_map[e.dst_conn],
                                      memlet=Memlet(str(e.data)))
                graph.remove_edge(e)

        # Redirect any outgoing edges from the nodes to be removed through their surviving counterparts.
        for n in remove_nodes:
            for e in graph.out_edges(n):
                if e.dst != second_entry:
                    alt_n = access_nodes[n.data]
                    memlet = Memlet(str(e.data)) if not e.data.is_empty() else Memlet()
                    graph.add_memlet_path(alt_n, e.dst, src_conn=e.src_conn, dst_conn=e.dst_conn, memlet=memlet)
            graph.remove_node(n)
        graph.remove_node(second_entry)
        graph.remove_node(second_exit)
