# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Move a loop that is a map's whole body outside the map: the inverse of ``MoveLoopIntoMap``."""

from dace import sdfg as sd
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg import state as cf
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.transformation import helpers, transformation
from dace.transformation.passes import move_if_into_loop


def free_names(expression: str) -> set[str]:
    return {str(s) for s in symbolic.pystr_to_symbolic(expression).free_symbols}


def carried_across_iterations(loop: cf.LoopRegion, body: sd.SDFG) -> str | None:
    """What keeps a value from one iteration of ``loop`` to the next inside ``body``, if anything does.

    Once the loop is outside the map, every iteration calls ``body`` afresh, so its transients and symbols start over.
    """
    assigned = {name for edge in loop.all_interstate_edges() for name in edge.data.assignments}
    for edge in loop.all_interstate_edges():
        for name, rhs in edge.data.assignments.items():
            if free_names(rhs) & assigned:
                return f"symbol {name} is carried across iterations"
    for name, desc in body.arrays.items():
        states = [s for s in loop.states() if any(n.data == name for n in s.data_nodes())]
        if desc.transient and states and (len(states) > 1 or name in move_if_into_loop.upward_exposed_reads(states[0])):
            return f"transient {name} is carried across iterations"
    return None


def crowded(state: sd.SDFGState, entry: nodes.MapEntry) -> list[nodes.Node]:
    """The top-level nodes of ``state`` other than ``entry``'s map and access nodes."""
    exit_node = state.exit_node(entry)
    return [
        n
        for n in state.scope_children()[None]
        if n is not entry and n is not exit_node and not isinstance(n, nodes.AccessNode)
    ]


def isolate_map(state: sd.SDFGState, entry: nodes.MapEntry) -> sd.SDFGState:
    """The state holding ``entry``'s map and its access nodes alone: what the map reads from is split off into a state
    before it and the rest stays in ``state``, after it. ``state`` itself when it holds nothing else."""
    if not crowded(state, entry):
        return state
    reached: set[nodes.Node] = set()
    sdutil.find_upstream_nodes(node_to_start=entry, state=state, seen=reached)
    upstream = [n for n in crowded(state, entry) if n in reached]
    if upstream:
        helpers.state_fission(SubgraphView(state, upstream))
    if not crowded(state, entry):
        return state
    isolated = helpers.state_fission(SubgraphView(state, [entry]))
    assert not crowded(isolated, entry), "only access nodes lie between the map and its upstream"
    return isolated


@transformation.explicit_cf_compatible
class MapLoopInterchange(transformation.SingleStateTransformation):
    """``map i: for t: B`` -> ``for t: map i: B``, for a map whose body is a nested SDFG holding only the loop.

    Map iterations are independent, so running every map iteration per loop iteration keeps the order each one
    sees. The move is legal when the loop's trip count is the same for every map iteration, nothing leaves an
    iteration early, and nothing inside the body carries a value between loop iterations. The loop repeats the map's
    state, so a state holding more than the map is split first: what the map reads from runs in a state before it,
    the rest in a state after it.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def refusal(self, state: sd.SDFGState) -> str | None:
        """Why the loop cannot move outside the map, or ``None``."""
        entry, body = self.map_entry, self.nested_sdfg
        blocks = body.sdfg.nodes()
        if len(blocks) != 1 or not isinstance(blocks[0], cf.LoopRegion):
            return "the body is not exactly one loop"
        loop = blocks[0]
        if any(e.dst is not body for e in state.out_edges(entry)):
            return "the map holds more than the nested SDFG"
        exit_node = state.exit_node(entry)
        if any(not isinstance(e.src, nodes.AccessNode) for e in state.in_edges(entry)) or any(
            not isinstance(e.dst, nodes.AccessNode) for e in state.out_edges(exit_node)
        ):
            return "the map exchanges data with a node other than an access node; its state cannot be split around it"
        var = loop.loop_variable
        outer = state.sdfg
        if var in entry.map.params or var in outer.symbols or var in outer.arrays:
            return f"the loop variable {var} is already defined outside the map"
        statements = [c for c in (loop.init_statement, loop.loop_condition, loop.update_statement) if c is not None]
        for name in sorted({str(s) for code in statements for s in code.get_free_symbols()} - {var}):
            if (
                name in entry.map.params
                or name in entry.in_connectors
                or name in body.sdfg.arrays
                or str(body.symbol_mapping.get(name)) != name
            ):
                return f"the loop bound {name} varies across map iterations"
        if any(
            isinstance(b, (cf.BreakBlock, cf.ContinueBlock, cf.ReturnBlock)) for b in loop.all_control_flow_blocks()
        ):
            return "the loop leaves an iteration early"
        return carried_across_iterations(loop, body.sdfg)

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return self.refusal(graph) is None

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        entry, body = self.map_entry, self.nested_sdfg  # read before the split moves them and stales their ids
        state = isolate_map(graph, entry)
        loop = body.sdfg.nodes()[0]
        var = loop.loop_variable
        dtype = loop.new_symbols({})[var]
        # splice the loop body into the nested SDFG, which then runs one iteration per call
        blocks, edges, start = list(loop.nodes()), list(loop.edges()), loop.start_block
        body.sdfg.remove_node(loop)
        for block in blocks:
            loop.remove_node(block)
            body.sdfg.add_node(block, is_start_block=block is start)
        for edge in edges:
            body.sdfg.add_edge(edge.src, edge.dst, edge.data)
        # the loop variable is a free symbol of the body now, bound by the outer loop
        if var not in body.sdfg.symbols:
            body.sdfg.add_symbol(var, dtype)
        body.symbol_mapping[var] = symbolic.symbol(var, dtype)
        # wrap the map's state in the loop
        parent = state.parent_graph
        ins, outs = parent.in_edges(state), parent.out_edges(state)
        parent.add_node(loop, is_start_block=parent.start_block is state, ensure_unique_name=True)
        for edge in ins:
            parent.add_edge(edge.src, loop, edge.data)
        for edge in outs:
            parent.add_edge(loop, edge.dst, edge.data)
        parent.remove_node(state)
        loop.add_node(state, is_start_block=True)
        root = sdfg.root_sdfg
        sdutil.set_nested_sdfg_parent_references(root)
        root.reset_cfg_list()
