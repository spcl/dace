import ast
from itertools import chain
from typing import Optional

from dace import transformation, SDFGState, SDFG, Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import OrderedDiGraph
from dace.sdfg.nodes import Tasklet, ExitNode, MapEntry, MapExit, NestedSDFG, Node
from dace.sdfg.state import ControlFlowBlock
from dace.sdfg.utils import node_path_graph
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import StateFusionExtended


def unique_map_node(graph: SDFGState) -> Optional[tuple[MapEntry, MapExit]]:
    all_nodes = list(graph.all_nodes_recursive())
    en: list[MapEntry] = [n for n, _ in all_nodes if isinstance(n, MapEntry)]
    ex = [n for n, _ in all_nodes if isinstance(n, MapExit)]
    if len(en) != 1 or len(ex) != 1:
        return None
    return en[0], ex[0]


def floating_nodes_graph(*args):
    g = OrderedDiGraph()
    for n in args:
        g.add_node(n)
    return g


class ConstAssignmentMapFusion(MapFusion):
    """
    Fuses two maps within a state, where each map:
    1. Either assigns consistent constant values to elements of one or more data arrays.
        - Consisency: The values must be the same for all elements in a data array (in both maps). But different data
          arrays are allowed to have different values.
    2. Or assigns constant values as described earlier, but _conditionally_. The condition must only depend on the map
       Parameters.

    Further conditions:
    1. Range compatibility: The two map must have the exact same range.
       # TODO(pratyai): Generalize this in `compatible_range()`.
    2. The maps must have one of the following patterns.
        - Exists a path like: MapExit -> AccessNode -> MapEntry
        - Neither map is dependent on the other. I.e. There is no dependency path between them.
    """
    first_map_entry = transformation.PatternNode(nodes.EntryNode)
    first_map_exit = transformation.PatternNode(nodes.ExitNode)
    array = transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.PatternNode(nodes.EntryNode)
    second_map_exit = transformation.PatternNode(nodes.ExitNode)

    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry),
                floating_nodes_graph(cls.first_map_entry, cls.first_map_exit, cls.second_map_entry,
                                     cls.second_map_exit)]

    @staticmethod
    def consistent_branch_const_assignment_table(graph: Node) -> tuple[bool, dict]:
        """
        If the graph consists of only conditional consistent constant assignments, produces a table mapping data arrays
        and memlets to their consistent constant assignments. See the class docstring for what is considered consistent.
        """
        table = {}
        # Basic premise check.
        if not isinstance(graph, NestedSDFG):
            return False, table
        graph: SDFG = graph.sdfg
        if not isinstance(graph, ControlFlowBlock):
            return False, table

        # Must have exactly 3 nodes, and exactly one of them a source, another a sink.
        src, snk = graph.source_nodes(), graph.sink_nodes()
        if len(graph.nodes()) != 3 or len(src) != 1 or len(snk) != 1:
            return False, table
        src, snk = src[0], snk[0]
        body = set(graph.nodes()) - {src, snk}
        if len(body) != 1:
            return False, table
        body = list(body)[0]

        # Must have certain structure of outgoing edges.
        src_eds = list(graph.out_edges(src))
        if len(src_eds) != 2 or any(e.data.is_unconditional() or e.data.assignments for e in src_eds):
            return False, table
        tb, el = src_eds
        if tb.dst != body:
            tb, el = el, tb
        if tb.dst != body or el.dst != snk:
            return False, table
        body_eds = list(graph.out_edges(body))
        if len(body_eds) != 1 or body_eds[0].dst != snk or not body_eds[0].data.is_unconditional() or body_eds[
            0].data.assignments:
            return False, table

        # Branch conditions must depend only on the loop variables.
        for b in [tb, el]:
            cond = b.data.condition
            for c in cond.code:
                used = set([ast_node.id for ast_node in ast.walk(c) if isinstance(ast_node, ast.Name)])
                if not used.issubset(graph.free_symbols):
                    return False, table

        # Body must have only constant assignments.
        for n, _ in body.all_nodes_recursive():
            # Each tasklet in this box...
            if not isinstance(n, Tasklet):
                continue
            if len(n.code.code) != 1 or not isinstance(n.code.code[0], ast.Assign):
                # ...must assign...
                return False, table
            op = n.code.code[0]
            if not isinstance(op.value, ast.Constant) or len(op.targets) != 1:
                # ...a constant to a single target.
                return False, table
            const = op.value.value
            for oe in body.out_edges(n):
                dst = oe.data
                dst_arr = oe.data.data
                if dst_arr in table and table[dst_arr] != const:
                    # A target array can appear multiple times, but it must always be consistently assigned.
                    return False, table
                table[dst] = const
                table[dst_arr] = const
        return True, table

    @staticmethod
    def consistent_const_assignment_table(graph: SDFGState, en: MapEntry, ex: MapExit) -> tuple[bool, dict]:
        """
        If the graph consists of only (conditional or unconditional) consistent constant assignments, produces a table
        mapping data arrays and memlets to their consistent constant assignments. See the class docstring for what is
        considered consistent.
        """
        table = {}
        for n in graph.all_nodes_between(en, ex):
            if isinstance(n, NestedSDFG):
                # First handle the case of conditional constant assignment.
                is_branch_const_assignment, internal_table = ConstAssignmentMapFusion.consistent_branch_const_assignment_table(n)
                if not is_branch_const_assignment:
                    return False, table
                for oe in graph.out_edges(n):
                    dst = oe.data
                    dst_arr = oe.data.data
                    if dst_arr in table and table[dst_arr] != internal_table[oe.src_conn]:
                        # A target array can appear multiple times, but it must always be consistently assigned.
                        return False, table
                    table[dst] = internal_table[oe.src_conn]
                    table[dst_arr] = internal_table[oe.src_conn]
            else:
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
        """Return the entry and exit nodes of the relevant maps as a tuple: entry_1, exit_1, entry_2, exit_2."""
        return (graph.entry_node(self.first_map_exit), self.first_map_exit,
                self.second_map_entry, graph.exit_node(self.second_map_entry))

    @staticmethod
    def compatible_range(first_entry: MapEntry, second_entry: MapEntry) -> bool:
        """Decide if the two ranges are compatible. See the class docstring for what is considered compatible."""
        if first_entry.map.schedule != second_entry.map.schedule:
            # If the two maps are not to be scheduled on the same device, don't fuse them.
            return False
        if first_entry.map.range != second_entry.map.range:
            # TODO(pratyai): Make it so that a permutation of the ranges, or even an union of the ranges will work.
            return False
        return True

    def no_dependency_pattern(self, graph: SDFGState, sdfg: SDFG, permissive: bool = False) -> bool:
        """Decide if the two maps are independent of each other."""
        first_entry, first_exit, second_entry, second_exit = self.map_nodes(graph)
        if first_entry != self.first_map_entry or second_exit != self.second_map_exit:
            return False
        if graph.all_nodes_between(first_exit, second_entry) or graph.all_nodes_between(second_exit, first_entry):
            return False
        return True

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        assert expr_index in (0, 1)
        if expr_index == 1:
            # Test the rest of the second pattern in the `expressions()`.
            if not self.no_dependency_pattern(graph, sdfg, permissive):
                return False

        first_entry, first_exit, second_entry, second_exit = self.map_nodes(graph)
        if not self.compatible_range(first_entry, second_entry):
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
    def track_access_nodes(graph: SDFGState, first_exit: ExitNode, second_exit: ExitNode) -> tuple[dict, set]:
        """
        Track all the access nodes that will survive after cleaning up duplicates. Returns a tuple with:
        1. A map: the underlying data array -> the surviving access node.
        2. A set of access nodes to be removed, i.e. has a duplicate in the map described earlier.
        """
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
    def make_equivalent_connectors(first_exit: ExitNode, second_exit: ExitNode):
        """
        Create the additional connectors in the first exit node that matches the second exit node (which will be removed
        later).
        """
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

        # Keep track in case loop variables are named differently.
        param_map = {p2: p1 for p1, p2 in zip(first_entry.map.params, second_entry.map.params)}

        # Track all the access nodes that will survive the purge.
        access_nodes, remove_nodes = self.track_access_nodes(graph, first_exit, second_exit)

        # Set up the extra connections on the first node.
        conn_map = self.make_equivalent_connectors(first_exit, second_exit)

        # Redirect outgoing edges from exit nodes that are going to be invalidated.
        for e in graph.out_edges(first_exit):
            array_name = e.dst.data
            assert array_name in access_nodes
            if access_nodes[array_name] != e.dst:
                graph.add_memlet_path(first_exit, access_nodes[array_name], src_conn=e.src_conn, dst_conn=e.dst_conn,
                                      memlet=Memlet.from_memlet(e.data))
                graph.remove_edge(e)
        for e in graph.out_edges(second_exit):
            array_name = e.dst.data
            assert array_name in access_nodes
            graph.add_memlet_path(first_exit, access_nodes[array_name], src_conn=conn_map[e.src_conn],
                                  dst_conn=e.dst_conn, memlet=Memlet.from_memlet(e.data))
            graph.remove_edge(e)

        # Move the tasklets from the second map into the first map.
        second_tasklets = graph.all_nodes_between(second_entry, second_exit)
        for t in second_tasklets:
            for e in graph.in_edges(t):
                graph.add_memlet_path(first_entry, t, memlet=Memlet.from_memlet(e.data))
                graph.remove_edge(e)
            for e in graph.out_edges(t):
                e_data = e.data
                e_data.subset.replace(param_map)
                graph.add_memlet_path(e.src, first_exit, src_conn=e.src_conn, dst_conn=conn_map[e.dst_conn],
                                      memlet=Memlet.from_memlet(e.data))
                graph.remove_edge(e)

        # Redirect any outgoing edges from the nodes to be removed through their surviving counterparts.
        for n in remove_nodes:
            for e in graph.out_edges(n):
                if e.dst != second_entry:
                    alt_n = access_nodes[n.data]
                    memlet = Memlet.from_memlet(e.data)
                    graph.add_memlet_path(alt_n, e.dst, src_conn=e.src_conn, dst_conn=e.dst_conn, memlet=memlet)
            graph.remove_node(n)
        graph.remove_node(second_entry)
        graph.remove_node(second_exit)


class ConstAssignmentStateFusion(StateFusionExtended):
    """
    If two consecutive states are such that
    1. Each state has just one _constant assigment map_ (see the docstring of `ConstAssignmentMapFusion`).
    2. If those two maps were in the same state `ConstAssignmentMapFusion` would fuse them.
    then fuse the two states.
    """
    first_state = transformation.PatternNode(SDFGState)
    second_state = transformation.PatternNode(SDFGState)

    # NOTE: `expression()` is inherited.

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # All the basic rules apply.
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        st0, st1 = self.first_state, self.second_state

        # Moreover, each state must contain just one constant assignment map.
        for st in [st0, st1]:
            en_ex = unique_map_node(st)
            if not en_ex:
                return False
            en, ex = en_ex
            if len(st.in_edges(en)) != 0:
                return False
            is_const_assignment, assignments = ConstAssignmentMapFusion.consistent_const_assignment_table(st, en, ex)
            if not is_const_assignment:
                return False

        # Moreover, both states' ranges must be compatible.
        if not ConstAssignmentMapFusion.compatible_range(unique_map_node(st0)[0], unique_map_node(st1)[0]):
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        # First, fuse the two states.
        super().apply(graph, sdfg)
        sdfg.validate()
        sdfg.apply_transformations_repeated(ConstAssignmentMapFusion)
        sdfg.validate()
