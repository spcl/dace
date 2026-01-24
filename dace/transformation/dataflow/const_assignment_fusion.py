import ast
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Optional, Union, Tuple

from dace import transformation, SDFGState, SDFG, Memlet, ScheduleType, subsets
from dace.properties import make_properties, Property
from dace.sdfg.graph import OrderedDiGraph
from dace.sdfg.nodes import Tasklet, ExitNode, MapEntry, MapExit, NestedSDFG, Node, EntryNode, AccessNode
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, StateSubgraphView
from dace.subsets import Range
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import StateFusionExtended


def _unique_top_level_map_node(graph: SDFGState) -> Optional[Tuple[MapEntry, MapExit]]:
    all_top_nodes = [n for n, s in graph.scope_dict().items() if s is None]
    if not all(isinstance(n, (MapEntry, AccessNode)) for n in all_top_nodes):
        return None
    en: list[MapEntry] = [n for n in all_top_nodes if isinstance(n, MapEntry)]
    ex: list[MapExit] = [graph.exit_node(n) for n in all_top_nodes if isinstance(n, MapEntry)]
    if len(en) != 1 or len(ex) != 1:
        return None
    return en[0], ex[0]


def _floating_nodes_graph(*args):
    g = OrderedDiGraph()
    for n in args:
        g.add_node(n)
    return g


def _consistent_branch_const_assignment_table(graph: Node) -> Tuple[bool, dict]:
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
        if not _is_constant_or_numerical_literal(op.value) or len(op.targets) != 1:
            # ...a constant to a single target.
            return False, table
        const = _value_of_constant_or_numerical_literal(op.value)
        for oe in body.out_edges(n):
            dst = oe.data
            dst_arr = oe.data.data
            if dst_arr in table and table[dst_arr] != const:
                # A target array can appear multiple times, but it must always be consistently assigned.
                return False, table
            table[dst] = const
            table[dst_arr] = const
    return True, table


def _is_constant_or_numerical_literal(n: ast.Expr):
    """Work around the API differences between Python versions (e.g., 3.7 and 3.12)"""
    return isinstance(n, (ast.Constant, ast.Num))


def _value_of_constant_or_numerical_literal(n: ast.Expr):
    """Work around the API differences between Python versions (e.g., 3.7 and 3.12)"""
    return n.value if isinstance(n, ast.Constant) else n.n


def _consistent_const_assignment_table(graph: SDFGState, en: MapEntry, ex: MapExit) -> Tuple[bool, dict]:
    """
    If the graph consists of only (conditional or unconditional) consistent constant assignments, produces a table
    mapping data arrays and memlets to their consistent constant assignments. See the class docstring for what is
    considered consistent.
    """
    table = {}
    for n in graph.all_nodes_between(en, ex):
        if isinstance(n, NestedSDFG):
            # First handle the case of conditional constant assignment.
            is_branch_const_assignment, internal_table = _consistent_branch_const_assignment_table(n)
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
        elif isinstance(n, MapEntry):
            is_const_assignment, internal_table = _consistent_const_assignment_table(graph, n, graph.exit_node(n))
            if not is_const_assignment:
                return False, table
            for k, v in internal_table.items():
                if k in table and v != table[k]:
                    return False, table
                internal_table[k] = v
        elif isinstance(n, MapExit):
            pass  # Handled with `MapEntry`
        else:
            # Each of the nodes in this map must be...
            if not isinstance(n, Tasklet):
                # ...a tasklet...
                return False, table
            if len(n.code.code) != 1 or not isinstance(n.code.code[0], ast.Assign):
                # ...that assigns...
                return False, table
            op = n.code.code[0]
            if not _is_constant_or_numerical_literal(op.value) or len(op.targets) != 1:
                # ...a constant to a single target.
                return False, table
            const = _value_of_constant_or_numerical_literal(op.value)
            for oe in graph.out_edges(n):
                dst = oe.data
                dst_arr = oe.data.data
                if dst_arr in table and table[dst_arr] != const:
                    # A target array can appear multiple times, but it must always be consistently assigned.
                    return False, table
                table[dst] = const
                table[dst_arr] = const
    return True, table


def _removeprefix(c: str, p: str):
    """Since `str.removeprefix()` wasn't added until Python 3.9"""
    if not c.startswith(p):
        return c
    return c[len(p):]


def _add_equivalent_connectors(dst: Union[EntryNode, ExitNode], src: Union[EntryNode, ExitNode]):
    """
    Create the additional connectors in the first exit node that matches the second exit node (which will be removed
    later).
    """
    conn_map = defaultdict()
    for c, v in src.in_connectors.items():
        assert c.startswith('IN_')
        cbase = _removeprefix(c, 'IN_')
        sc = dst.next_connector(cbase)
        conn_map[f"IN_{cbase}"] = f"IN_{sc}"
        conn_map[f"OUT_{cbase}"] = f"OUT_{sc}"
        dst.add_in_connector(f"IN_{sc}", dtype=v)
        dst.add_out_connector(f"OUT_{sc}", dtype=v)
    for c, v in src.out_connectors.items():
        assert c in conn_map
    return conn_map


def _connector_counterpart(c: Union[str, None]) -> Union[str, None]:
    """If it's an input connector, find the corresponding output connector, and vice versa."""
    if c is None:
        return None
    assert isinstance(c, str)
    if c.startswith('IN_'):
        return f"OUT_{_removeprefix(c, 'IN_')}"
    elif c.startswith('OUT_'):
        return f"IN_{_removeprefix(c, 'OUT_')}"
    return None


def _consolidate_empty_dependencies(graph: SDFGState, first_entry: MapEntry, second_entry: MapEntry):
    """
    Remove all the incoming edges of the two maps and add empty edges from the union of the access nodes they
    depended on before.

    Preconditions:
    1. All the incoming edges of the two maps must be from an access node and empty (i.e. have existed
    only for synchronization).
    2. The two maps must be constistent const assignments (see the class docstring for what is considered
    consistent).
    """
    # First, construct a table of the dependencies.
    table = {}
    for en in [first_entry, second_entry]:
        for e in graph.in_edges(en):
            assert e.data.is_empty()
            assert e.src_conn is None and e.dst_conn is None
            if not isinstance(e.src, AccessNode):
                continue
            if e.src.data not in table:
                table[e.src.data] = e.src
            elif table[e.src.data] in graph.bfs_nodes(e.src):
                # If this copy of the node is above the copy we've seen before, use this one instead.
                table[e.src.data] = e.src
            graph.remove_edge(e)
    # Then, if we still have so that any of the map _writes_ to these nodes, we want to just create fresh copies to
    # avoid cycles.
    alt_table = {}
    for k, v in table.items():
        if v in graph.bfs_nodes(first_entry) or v in graph.bfs_nodes(second_entry):
            alt_v = deepcopy(v)
            graph.add_node(alt_v)
            alt_table[k] = alt_v
        else:
            alt_table[k] = v
    # Finally, these nodes should be depended on by _both_ maps.
    for en in [first_entry, second_entry]:
        for n in alt_table.values():
            graph.add_nedge(n, en, Memlet())


def _consolidate_written_nodes(graph: SDFGState, first_exit: MapExit, second_exit: MapExit):
    """
    If the two maps write to the same underlying data array through two access nodes, replace those edges'
    destination with a single shared copy.

    Precondition:
    1. The two maps must not depend on each other through an access node, which should be taken care of already by
    `consolidate_empty_dependencies()`.
    2. The two maps must be constistent const assignments (see the class docstring for what is considered
    consistent).
    """
    # First, construct tables of the surviving and all written access nodes.
    surviving_nodes, all_written_nodes = {}, set()
    for ex in [first_exit, second_exit]:
        for e in graph.out_edges(ex):
            assert not e.data.is_empty()
            assert e.src_conn is not None and ((e.dst_conn is None) == isinstance(e.dst, AccessNode))
            if not isinstance(e.dst, AccessNode):
                continue
            all_written_nodes.add(e.dst)
            if e.dst.data not in surviving_nodes:
                surviving_nodes[e.dst.data] = e.dst
            elif e.dst in graph.bfs_nodes(surviving_nodes[e.dst.data]):
                # If this copy of the node is above the copy we've seen before, use this one instead.
                surviving_nodes[e.dst.data] = e.dst
    # Then, redirect all the edges toward the surviving copies of the destination access nodes.
    for n in all_written_nodes:
        for e in graph.in_edges(n):
            assert e.src in [first_exit, second_exit]
            assert e.dst_conn is None
            graph.add_edge(e.src, e.src_conn, surviving_nodes[e.dst.data], e.dst_conn, Memlet.from_memlet(e.data))
            graph.remove_edge(e)
        for e in graph.out_edges(n):
            assert e.src_conn is None
            graph.add_edge(surviving_nodes[e.src.data], e.src_conn, e.dst, e.dst_conn, Memlet.from_memlet(e.data))
            graph.remove_edge(e)
    # Finally, cleanup the orphan nodes.
    for n in all_written_nodes:
        if graph.degree(n) == 0:
            graph.remove_node(n)


def _consume_map_exactly(graph: SDFGState, dst: Tuple[MapEntry, MapExit], src: Tuple[MapEntry, MapExit]):
    """
    Transfer the entirety of `src` map's body into `dst` map. Only possible when the two maps' ranges are identical.
    """
    dst_en, dst_ex = dst
    src_en, src_ex = src

    assert all(e.data.is_empty() for e in graph.in_edges(src_en))
    cmap = _add_equivalent_connectors(dst_en, src_en)
    for e in graph.in_edges(src_en):
        graph.add_edge(e.src, e.src_conn, dst_en, cmap.get(e.dst_conn), Memlet.from_memlet(e.data))
        graph.remove_edge(e)
    for e in graph.out_edges(src_en):
        graph.add_edge(dst_en, cmap.get(e.src_conn), e.dst, e.dst_conn, Memlet.from_memlet(e.data))
        graph.remove_edge(e)

    cmap = _add_equivalent_connectors(dst_ex, src_ex)
    for e in graph.in_edges(src_ex):
        graph.add_edge(e.src, e.src_conn, dst_ex, cmap.get(e.dst_conn), Memlet.from_memlet(e.data))
        graph.remove_edge(e)
    for e in graph.out_edges(src_ex):
        graph.add_edge(dst_ex, cmap.get(e.src_conn), e.dst, e.dst_conn, Memlet.from_memlet(e.data))
        graph.remove_edge(e)

    graph.remove_node(src_en)
    graph.remove_node(src_ex)


def _consume_map_with_grid_strided_loop(graph: SDFGState, dst: Tuple[MapEntry, MapExit],
                                        src: Tuple[MapEntry, MapExit]):
    """
    Transfer the entirety of `src` map's body into `dst` map, guarded behind a _grid-strided_ loop.
    Prerequisite: `dst` map's range must cover `src` map's range in entirety. Statically checking this may not
    always be possible.
    """
    dst_en, dst_ex = dst
    src_en, src_ex = src

    def range_for_grid_stride(r, val, bound):
        r = list(r)
        r[0] = val
        r[1] = bound - 1
        r[2] = bound
        return tuple(r)

    gsl_ranges = [range_for_grid_stride(rd, p, rs[1] + 1)
                  for p, rs, rd in zip(dst_en.map.params, src_en.map.range.ranges, dst_en.map.range.ranges)]
    gsl_params = [f"gsl_{p}" for p in dst_en.map.params]
    en, ex = graph.add_map(graph.sdfg._find_new_name('gsl'),
                           ndrange={k: v for k, v in zip(gsl_params, gsl_ranges)},
                           schedule=ScheduleType.Sequential)
    _consume_map_exactly(graph, (en, ex), src)

    assert all(e.data.is_empty() for e in graph.in_edges(en))
    cmap = _add_equivalent_connectors(dst_en, en)
    for e in graph.in_edges(en):
        graph.add_edge(e.src, e.src_conn, dst_en, cmap.get(e.dst_conn), Memlet.from_memlet(e.data))
        graph.add_edge(dst_en, cmap.get(e.src_conn), e.dst, e.dst_conn, Memlet.from_memlet(e.data))
        graph.remove_edge(e)

    cmap = _add_equivalent_connectors(dst_ex, ex)
    for e in graph.out_edges(ex):
        graph.add_edge(e.src, e.src_conn, dst_ex, _connector_counterpart(cmap.get(e.src_conn)),
                       Memlet.from_memlet(e.data))
        graph.add_edge(dst_ex, cmap.get(e.src_conn), e.dst, e.dst_conn, Memlet.from_memlet(e.data))
        graph.remove_edge(e)
    if len(graph.in_edges(en)) == 0:
        graph.add_nedge(dst_en, en, Memlet())
    if len(graph.out_edges(ex)) == 0:
        graph.add_nedge(ex, dst_ex, Memlet())


def _fused_range(r1: Range, r2: Range) -> Optional[Range]:
    if r1 == r2:
        return r1
    if len(r1) != len(r2):
        return None
    r = []
    bb = subsets.union(r1, r2).ndrange()
    for i in range(len(r1)):
        if r1.strides()[i] != r2.strides()[i]:
            return None
        if r1.strides()[i] == 1:
            r.append(bb[i])
        elif r1.ranges[i] == r2.ranges[i]:
            r.append(bb[i])
        else:
            return None
    return r


def _maps_have_compatible_ranges(first_entry: MapEntry, second_entry: MapEntry, use_grid_strided_loops: bool) -> bool:
    """Decide if the two ranges are compatible. See the class docstring for what is considered compatible."""
    if first_entry.map.schedule != second_entry.map.schedule:
        # If the two maps are not to be scheduled on the same device, don't fuse them.
        return False
    if len(first_entry.map.range) != len(second_entry.map.range):
        # If it's not even possible to take component-wise union of the two map's range, don't fuse them.
        # TODO(pratyai): Make it so that a permutation of the ranges, or even an union of the ranges will work.
        return False
    if not use_grid_strided_loops:
        # If we don't use grid-strided loops, the two maps' ranges must be identical.
        if first_entry.map.range != second_entry.map.range:
            return False
    if first_entry.map.schedule == ScheduleType.Sequential:
        # For _grid-strided loops_, fuse them only when their ranges are _exactly_ the same. I.e., never put them
        # behind another layer of grid-strided loop.
        if first_entry.map.range != second_entry.map.range:
            return False
    return True


@make_properties
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
    first_map_entry = transformation.PatternNode(MapEntry)
    second_map_entry = transformation.PatternNode(MapEntry)

    use_grid_strided_loops = Property(dtype=bool, default=False,
                                      desc='Set to use grid strided loops to use two maps with non-idential ranges.')

    @classmethod
    def expressions(cls):
        # Take any two maps, then check that _every_ path from the first map to second map has exactly one access node
        # in the middle and the second edge of the path is empty.
        return [_floating_nodes_graph(cls.first_map_entry, cls.second_map_entry)]

    def _map_nodes(self, graph: SDFGState):
        """Return the entry and exit nodes of the relevant maps as a tuple: entry_1, exit_1, entry_2, exit_2."""
        return (self.first_map_entry, graph.exit_node(self.first_map_entry),
                self.second_map_entry, graph.exit_node(self.second_map_entry))

    def _no_dependency_pattern(self, graph: SDFGState) -> bool:
        """Decide if the two maps are independent of each other."""
        first_entry, first_exit, second_entry, second_exit = self._map_nodes(graph)
        all_in_edges = list(chain(graph.in_edges(first_entry), graph.in_edges(second_entry)))
        all_out_edges = list(chain(graph.out_edges(first_exit), graph.out_edges(second_exit)))

        # The analysis is too difficult to continue (so just reject independence to err on the side of caution), when...
        if graph.scope_dict()[first_entry] != graph.scope_dict()[second_entry]:
            # ... the two maps are not even on the same scope (so analysing the connectivity is difficult).
            return False
        if not all(isinstance(n, AccessNode) for n in chain(graph.all_nodes_between(first_exit, second_entry),
                                                            graph.all_nodes_between(second_exit, first_entry))):
            # ... there are non-AccessNodes between the two maps (also difficult to analyse).
            return False
        if any(not isinstance(e.src, (MapExit, AccessNode)) for e in all_in_edges):
            # ... either map has incoming edges from a node that is not an AccessNode or a MapExit (also difficult).
            return False
        if any(not isinstance(e.dst, (MapEntry, AccessNode)) for e in all_out_edges):
            # ... either map has outgoing edges to a node that is not an AccessNode or a MapEntry (also difficult).
            return False

        if any(not e.data.is_empty() for e in all_in_edges):
            # If any of the maps are reading anything, then it isn't independent.
            return False

        return True

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # Test the rest of the second pattern in the `expressions()`.
        if not self._no_dependency_pattern(graph):
            return False

        first_entry, first_exit, second_entry, second_exit = self._map_nodes(graph)
        if not _maps_have_compatible_ranges(first_entry, second_entry,
                                            use_grid_strided_loops=self.use_grid_strided_loops):
            return False

        # Both maps must have consistent constant assignment for the target arrays.
        is_const_assignment, assignments = _consistent_const_assignment_table(graph, first_entry, first_exit)
        if not is_const_assignment:
            return False
        is_const_assignment, further_assignments = _consistent_const_assignment_table(graph, second_entry, second_exit)
        if not is_const_assignment:
            return False
        for k, v in further_assignments.items():
            if k in assignments and v != assignments[k]:
                return False
            assignments[k] = v
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        first_entry, first_exit, second_entry, second_exit = self._map_nodes(graph)

        # By now, we know that the two maps are compatible, not reading anything, and just blindly writing constants
        # _consistently_.
        is_const_assignment, assignments = _consistent_const_assignment_table(graph, first_entry, first_exit)
        assert is_const_assignment

        # Rename in case loop variables are named differently.
        nodes_to_update = {n for n in graph.all_nodes_between(second_entry, second_exit)} | {second_entry, second_exit}
        view = StateSubgraphView(graph, list(nodes_to_update))
        view.replace_dict({p2: p1 for p1, p2 in zip(first_entry.map.params, second_entry.map.params)})

        # Consolidate the incoming dependencies of the two maps.
        _consolidate_empty_dependencies(graph, first_entry, second_entry)

        # Consolidate the written access nodes of the two maps.
        _consolidate_written_nodes(graph, first_exit, second_exit)

        # If the ranges are identical, then simply fuse the two maps. Otherwise, use grid-strided loops.
        assert _fused_range(first_entry.map.range, second_entry.map.range) is not None
        en, ex = graph.add_map(sdfg._find_new_name('map_fusion_wrapper'),
                               ndrange={k: v for k, v in zip(first_entry.map.params,
                                                             _fused_range(first_entry.map.range,
                                                                          second_entry.map.range))},
                               schedule=first_entry.map.schedule)
        if first_entry.map.range == second_entry.map.range:
            for cur_en, cur_ex in [(first_entry, first_exit), (second_entry, second_exit)]:
                _consume_map_exactly(graph, (en, ex), (cur_en, cur_ex))
        elif self.use_grid_strided_loops:
            assert ScheduleType.Sequential not in [first_entry.map.schedule, second_entry.map.schedule]
            for cur_en, cur_ex in [(first_entry, first_exit), (second_entry, second_exit)]:
                if en.map.range == cur_en.map.range:
                    _consume_map_exactly(graph, (en, ex), (cur_en, cur_ex))
                else:
                    _consume_map_with_grid_strided_loop(graph, (en, ex), (cur_en, cur_ex))

        # Cleanup: remove duplicate empty dependencies.
        seen = set()
        for e in graph.in_edges(en):
            assert e.data.is_empty()
            if e.src not in seen:
                seen.add(e.src)
            else:
                graph.remove_edge(e)


@make_properties
class ConstAssignmentStateFusion(StateFusionExtended):
    """
    If two consecutive states are such that
    1. Each state has just one _constant assigment map_ (see the docstring of `ConstAssignmentMapFusion`).
    2. If those two maps were in the same state `ConstAssignmentMapFusion` would fuse them.
    then fuse the two states.
    """
    first_state = transformation.PatternNode(SDFGState)
    second_state = transformation.PatternNode(SDFGState)

    use_grid_strided_loops = Property(dtype=bool, default=False,
                                      desc='Set to use grid strided loops to use two maps with non-idential ranges.')

    # NOTE: `expression()` is inherited.

    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # All the basic rules apply.
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        st0, st1 = self.first_state, self.second_state

        # Moreover, the states together must contain a consistent constant assignment map.
        assignments = {}
        for st in [st0, st1]:
            en_ex = _unique_top_level_map_node(st)
            if not en_ex:
                return False
            en, ex = en_ex
            if any(not e.data.is_empty() for e in st.in_edges(en)):
                return False
            is_const_assignment, further_assignments = _consistent_const_assignment_table(st, en, ex)
            if not is_const_assignment:
                return False
            for k, v in further_assignments.items():
                if k in assignments and v != assignments[k]:
                    return False
                assignments[k] = v

        # Moreover, both states' ranges must be compatible.
        if not _maps_have_compatible_ranges(_unique_top_level_map_node(st0)[0], _unique_top_level_map_node(st1)[0],
                                            use_grid_strided_loops=self.use_grid_strided_loops):
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        # First, fuse the two states.
        super().apply(graph, sdfg)
        sdfg.validate()
        # Then, fuse the maps inside.
        sdfg.apply_transformations_repeated(ConstAssignmentMapFusion,
                                            options={'use_grid_strided_loops': self.use_grid_strided_loops})
        sdfg.validate()
