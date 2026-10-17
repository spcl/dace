# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""State fusion transformation"""

import networkx as nx

from dace import data as dt
from dace import memlet, properties, sdfg, subsets
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.sdfg.validation import InvalidSDFGEdgeError, validate_state
from dace.transformation import transformation
from dace.transformation.interstate.state_fusion import StateFusion


# Helper class for finding connected component correspondences
class CCDesc:
    def __init__(
        self,
        first_input_nodes: list[nodes.AccessNode],
        first_output_nodes: list[nodes.AccessNode],
        second_input_nodes: list[nodes.AccessNode],
        second_output_nodes: list[nodes.AccessNode],
    ) -> None:
        self.first_inputs = {n.data for n in first_input_nodes}
        self.first_input_nodes = first_input_nodes
        self.first_outputs = {n.data for n in first_output_nodes}
        self.first_output_nodes = first_output_nodes
        self.second_inputs = {n.data for n in second_input_nodes}
        self.second_input_nodes = second_input_nodes
        self.second_outputs = {n.data for n in second_output_nodes}
        self.second_output_nodes = second_output_nodes


def top_level_nodes(state: SDFGState):
    return state.scope_children()[None]


def is_start_block(graph, block) -> bool:
    """``graph.start_block is block``, without the getter's raise on a region with several unpinned sources."""
    try:
        return graph.start_block is block
    except ValueError:
        return False


def in_state_order(node_iter, order: dict[nodes.Node, int]) -> list[nodes.Node]:
    """Nodes sorted by their position in ``state.nodes()`` (``order``). Nodes hash by ``id()``, so set order
    differs between runs, and several decisions below are first-match-wins over such sets."""
    return sorted(node_iter, key=order.__getitem__)


def read_consumers(state: SDFGState, owner_sdfg: sdfg.SDFG, node: nodes.AccessNode) -> list[nodes.Node]:
    """The nodes reading ``node`` in ``state``. A View is a pointer, so its read happens where the view is consumed."""
    consumers: list[nodes.Node] = []
    pending = [e.dst for e in state.out_edges(node)]
    for consumer in pending:
        if (
            isinstance(consumer, nodes.AccessNode)
            and isinstance(owner_sdfg.arrays.get(consumer.data), dt.View)
            and state.out_degree(consumer) > 0
        ):
            pending.extend(e.dst for e in state.out_edges(consumer))
        else:
            consumers.append(consumer)
    return consumers


@transformation.explicit_cf_compatible
@properties.make_properties
class StateFusionExtended(transformation.MultiStateTransformation):
    """Implements the state-fusion transformation extended to fuse states with RAW and WAW dependencies.
    An empty memlet is used to represent a dependency between two subgraphs with RAW and WAW dependencies.
    The merge is made by identifying the source in the first state and the sink in the second state,
    and linking the bottom of the appropriate source subgraph in the first state with the top of the
    appropriate sink subgraph in the second state.

    State-fusion takes two states that are connected through a single edge,
    and fuses them into one state. If permissive, also applies if potential memory
    access hazards are created.
    """

    first_state = transformation.PatternNode(sdfg.SDFGState)
    second_state = transformation.PatternNode(sdfg.SDFGState)

    strict_validate = properties.Property(
        dtype=bool, default=False, desc="Run ``sdfg.validate()`` after ``apply``, on top of the fused-state check."
    )

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_state, cls.second_state)]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ``(kind, first_nodes, second_nodes)`` happens-before edges recorded by the match, wired by ``apply``
        self._connections_to_make = []
        self._matched_permissive = False

    @staticmethod
    def find_fused_components(first_cc_input, first_cc_output, second_cc_input, second_cc_output) -> list[CCDesc]:
        # Make a bipartite graph out of the first and second components
        g = nx.DiGraph()
        g.add_nodes_from((0, i) for i in range(len(first_cc_output)))
        g.add_nodes_from((1, i) for i in range(len(second_cc_output)))
        # Find matching nodes in second state
        for i, cc1 in enumerate(first_cc_output):
            outnames1 = {n.data for n in cc1}
            for j, cc2 in enumerate(second_cc_input):
                inpnames2 = {n.data for n in cc2}
                if len(outnames1 & inpnames2) > 0:
                    g.add_edge((0, i), (1, j))

        # Construct result out of connected components of the bipartite graph, sorted: their order is the
        # order in which happens-before edges are recorded
        result = []
        for cc in sorted(sorted(cc) for cc in nx.weakly_connected_components(g)):
            input1, output1, input2, output2 = [], [], [], []
            for gind, cind in cc:
                if gind == 0:
                    input1 += first_cc_input[cind]
                    output1 += first_cc_output[cind]
                else:
                    input2 += second_cc_input[cind]
                    output2 += second_cc_output[cind]
            result.append(CCDesc(input1, output1, input2, output2))

        return result

    # The overlap and ordering checks are StateFusion's
    memlets_intersect = staticmethod(StateFusion.memlets_intersect)
    has_path = StateFusion.has_path
    _check_all_paths = StateFusion._check_all_paths

    def _check_paths(self, first_state, second_state, match_nodes, nodes_first, nodes_second, first_read, second_read):
        # Ordered when one match node is reached from every first-state node and its second-state partner
        # reaches every second-state node; any such match proves it, whichever position it sits in (s253)
        for match, second_match in match_nodes.items():
            if all(nx.has_path(first_state._nx, n, match) for n in nodes_first) and all(
                nx.has_path(second_state._nx, second_match, n) for n in nodes_second
            ):
                return True
        # Not ordered by dataflow. Only an actual overlap is a hazard.
        return not StateFusionExtended.memlets_intersect(
            first_state, nodes_first, first_read, second_state, nodes_second, second_read
        )

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # We keep it alive, such that `apply()` can use it later. A refused match must not leave a
        # half-filled recorder behind.
        self._connections_to_make.clear()
        self._matched_permissive = permissive
        if self._check_fusible(graph, sdfg, permissive):
            return True
        self._connections_to_make.clear()
        return False

    def _check_fusible(self, graph, sdfg, permissive: bool) -> bool:
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state

        out_edges = graph.out_edges(first_state)
        in_edges = graph.in_edges(first_state)

        # First state must have only one output edge (with dst the second
        # state).
        if len(out_edges) != 1:
            return False
        # If both states have more than one incoming edge, some control flow
        # may become ambiguous
        if len(in_edges) > 1 and graph.in_degree(second_state) > 1:
            return False
        # The interstate edge must not have a condition.
        if not out_edges[0].data.is_unconditional():
            return False
        # The interstate edge may have assignments, as long as there are input
        # edges to the first state that can absorb them.
        if out_edges[0].data.assignments:
            if not in_edges:
                return False
            # Fail if symbol is set before the state to fuse
            new_assignments = set(out_edges[0].data.assignments.keys())
            if any((new_assignments & set(e.data.assignments.keys())) for e in in_edges):
                return False
            # Fail if symbol is used in the dataflow of that state
            if len(new_assignments & first_state.free_symbols) > 0:
                return False
            # Fail if assignments have free symbols that are updated in the
            # first state
            freesyms = out_edges[0].data.free_symbols
            if freesyms and any(
                n.data in freesyms
                for n in first_state.nodes()
                if isinstance(n, nodes.AccessNode) and first_state.in_degree(n) > 0
            ):
                return False
            # Fail if symbols assigned on the first edge are free symbols on the
            # second edge
            symbols_used = set(out_edges[0].data.free_symbols)
            for e in in_edges:
                if e.data.assignments.keys() & symbols_used:
                    return False
                # Also fail in the inverse; symbols assigned on the second edge are free symbols on the first edge
                if new_assignments & set(e.data.free_symbols):
                    return False

        # There can be no state that have output edges pointing to both the
        # first and the second state. Such a case will produce a multi-graph.
        for src, _, _ in in_edges:
            for _, dst, _ in graph.out_edges(src):
                if dst == second_state:
                    return False

        # A side effect (I/O, a callback, a stateful library call) is ordered by the interstate edge, not by
        # dataflow; fused, it could be reordered or run concurrently. ``NestedSDFG.has_side_effects`` recurses.
        for state in (first_state, second_state):
            if any(isinstance(n, nodes.CodeNode) and n.has_side_effects(sdfg) for n in state.nodes()):
                return False

        if not permissive:
            # Strict mode that inhibits state fusion if Python callbacks are involved
            if Config.get_bool("frontend", "dont_fuse_callbacks"):
                for node in first_state.data_nodes() + second_state.data_nodes():
                    if node.data == "__pystate":
                        return False

            first_scope = first_state.scope_dict()
            top_writes: dict[str, int] = {}
            scoped_writes: set[str] = set()
            for n in first_state.data_nodes():
                if first_state.in_degree(n) == 0:
                    continue
                if first_scope[n] is None:
                    top_writes[n.data] = top_writes.get(n.data, 0) + 1
                else:
                    scoped_writes.add(n.data)
            # A container the first state writes only inside a scope has no top-level node the second state's
            # access could merge with or be ordered after (crc16's ``c_0``)
            if (scoped_writes - top_writes.keys()) & {n.data for n in second_state.data_nodes()}:
                return False
            # A transient with two top-level producers in the first state that the second state writes again:
            # the common-data-node merge could bind a reader to the wrong producer, which no ordering edge
            # repairs (nbody's ``__rdo0``)
            for n in top_level_nodes(second_state):
                if not isinstance(n, nodes.AccessNode) or second_state.in_degree(n) == 0:
                    continue
                desc = sdfg.arrays[n.data]
                if desc.transient and not isinstance(desc, dt.View) and top_writes.get(n.data, 0) >= 2:
                    return False

            # If second state has other input edges, there might be issues
            # Exceptions are when none of the states contain dataflow, unless
            # the first state is an initial state (in which case the new initial
            # state would be ambiguous).
            first_in_edges = graph.in_edges(first_state)
            second_in_edges = graph.in_edges(second_state)
            if (not second_state.is_empty() or not first_state.is_empty() or len(first_in_edges) == 0) and len(
                second_in_edges
            ) != 1:
                return False

            # Get connected components, in state order (see `in_state_order`)
            first_order = {n: i for i, n in enumerate(first_state.nodes())}
            second_order = {n: i for i, n in enumerate(second_state.nodes())}
            first_cc = sorted(
                (in_state_order(cc, first_order) for cc in nx.weakly_connected_components(first_state._nx)),
                key=lambda cc: first_order[cc[0]],
            )
            second_cc = sorted(
                (in_state_order(cc, second_order) for cc in nx.weakly_connected_components(second_state._nx)),
                key=lambda cc: second_order[cc[0]],
            )

            # Find source/sink (data) nodes
            top1, top2 = top_level_nodes(first_state), top_level_nodes(second_state)
            first_input = in_state_order(
                [n for n in first_state.source_nodes() if isinstance(n, nodes.AccessNode)], first_order
            )
            first_input_set = set(first_input)
            first_output = in_state_order(
                [n for n in top1 if isinstance(n, nodes.AccessNode) and n not in first_input_set], first_order
            )
            second_input = in_state_order(
                [n for n in second_state.source_nodes() if isinstance(n, nodes.AccessNode)], second_order
            )
            second_input_set = set(second_input)
            second_output = in_state_order(
                [n for n in top2 if isinstance(n, nodes.AccessNode) and n not in second_input_set], second_order
            )
            first_output_set, second_output_set = set(first_output), set(second_output)

            def by_data(node_list) -> dict[str, list[nodes.AccessNode]]:
                result: dict[str, list[nodes.AccessNode]] = {}
                for n in node_list:
                    result.setdefault(n.data, []).append(n)
                return result

            first_input_by_data = by_data(first_input)
            first_output_by_data = by_data(first_output)
            second_output_by_data = by_data(second_output)

            # WCR is a read-modify-write: a second-state ``a(+)= ...`` implicitly reads the value the first
            # state wrote, through no edge (covariance ``mean[:] = 0; mean(+)= ...``). A transient scalar seed
            # fuses, with every second-state source leading to the accumulation ordered after the seed writer;
            # any other seeded container keeps its states apart.
            first_written: dict[str, list] = {}
            first_writers: dict[str, list[nodes.AccessNode]] = {}
            for n in first_output:
                for e in first_state.in_edges(n):
                    if not e.data.is_empty():
                        ss = e.data.get_dst_subset(e, first_state) or e.data.subset
                        if ss is not None:
                            first_written.setdefault(n.data, []).append(ss)
                            first_writers.setdefault(n.data, []).append(n)
            second_sources = second_state.source_nodes()
            for e in second_state.edges() if first_written else ():
                if e.data.wcr is None or e.data.data not in first_written:
                    continue
                wsub = e.data.get_dst_subset(e, second_state) or e.data.subset
                if wsub is not None and all(subsets.intersects(wsub, fs) is False for fs in first_written[e.data.data]):
                    continue
                seed = sdfg.arrays[e.data.data]
                accumulated = second_state.memlet_path(e)[-1].dst
                if (
                    not (isinstance(seed, dt.Scalar) and seed.transient)
                    or not isinstance(accumulated, nodes.AccessNode)
                    or accumulated not in top2
                    or not any(
                        nx.has_path(second_state._nx, i, accumulated) for i in second_sources if i is not accumulated
                    )
                ):
                    return False
                self._connections_to_make.append(
                    ("seed", list(dict.fromkeys(first_writers[e.data.data])), [accumulated])
                )

            # WAR (anti-dependency): the first state reads ``d`` and the second overwrites an overlapping
            # element. The read happens at the reader's consumer, so every consumer is ordered before the
            # second-state writer by a happens-before edge, recorded here and wired in ``apply``. An empty
            # in-edge (a prior fusion's ordering edge) is not a write.
            def has_data_write(n):
                return any(not e.data.is_empty() for e in first_state.in_edges(n))

            first_out_data = {n.data for n in first_output if has_data_write(n)}
            first_read_subsets: dict[str, list] = {}
            first_readers: dict[str, list[nodes.AccessNode]] = {}
            for rn in first_state.data_nodes():
                if first_scope[rn] is not None:
                    continue
                for re in first_state.out_edges(rn):
                    # A single-sided copy memlet names only the destination
                    ss = re.data.get_src_subset(re, first_state) or re.data.get_dst_subset(re, first_state)
                    if ss is not None:
                        first_read_subsets.setdefault(rn.data, []).append(ss)
                        first_readers.setdefault(rn.data, []).append(rn)

            def ordering_endpoints(readers) -> list[nodes.Node] | None:
                """Where each read completes. A scope entry stands in for its exit (an edge out of an entry
                would pull the second state into the scope); a library node for its outputs (it counts its
                edges). ``None``: a scope without an exit, refuse."""
                endpoints = []
                for rn in readers:
                    for cons in read_consumers(first_state, sdfg, rn):
                        if isinstance(cons, nodes.EntryNode):
                            cons = first_state.exit_node(cons)
                            if cons is None:
                                return None
                        elif isinstance(cons, nodes.LibraryNode):
                            endpoints.extend([se.dst for se in first_state.out_edges(cons)] or [cons])
                            continue
                        endpoints.append(cons)
                return list(dict.fromkeys(endpoints))

            def ordered_by_existing_dataflow(consumers, we) -> bool:
                # The write's value flows from a second-state source the first state produces, and every
                # consumer of the read reaches that producer
                if not consumers:
                    return False
                for a in nx.ancestors(second_state._nx, we.src) | {we.src}:
                    if not isinstance(a, nodes.AccessNode) or a.data not in first_out_data:
                        continue
                    if second_state.in_degree(a) != 0:
                        continue  # re-written inside the second state: not a merge point
                    producers = [n for n in first_output_by_data[a.data] if has_data_write(n)]
                    if producers and all(any(nx.has_path(first_state._nx, c, p) for p in producers) for c in consumers):
                        return True
                return False

            for wnode in second_output:
                if wnode.data not in first_read_subsets:
                    continue
                consumers = None
                for we in second_state.in_edges(wnode):
                    wsub = we.data.get_dst_subset(we, second_state) or we.data.get_src_subset(we, second_state)
                    if wsub is None:
                        continue
                    if consumers is None:
                        consumers = ordering_endpoints(first_readers[wnode.data])
                        if consumers is None:
                            return False
                    if ordered_by_existing_dataflow(consumers, we):
                        continue
                    if any(subsets.intersects(wsub, rs) is not False for rs in first_read_subsets[wnode.data]):
                        if consumers:
                            self._connections_to_make.append(("war", consumers, [wnode]))
                        break

            # Find source/sink (data) nodes by connected component
            first_cc_input = [[n for n in cc if n in first_input_set] for cc in first_cc]
            first_cc_output = [[n for n in cc if n in first_output_set] for cc in first_cc]
            second_cc_input = [[n for n in cc if n in second_input_set] for cc in second_cc]
            second_cc_output = [[n for n in cc if n in second_output_set] for cc in second_cc]

            # Apply transformation in case all paths to the second state's
            # nodes go through the same access node, which implies sequential
            # behavior in SDFG semantics.
            first_output_names = {node.data for node in first_output}
            second_input_names = {node.data for node in second_input}

            # If any second input appears more than once, fail
            if len(second_input) > len(second_input_names):
                return False

            # If any first output that is an input to the second state
            # appears in more than one CC, fail
            matches = first_output_names & second_input_names
            for match in matches:
                if sum(1 for cc in first_cc_output if any(n.data == match for n in cc)) > 1:
                    return False

            # Recreate fused connected component correspondences, and then
            # check for hazards
            resulting_ccs: list[CCDesc] = StateFusionExtended.find_fused_components(
                first_cc_input, first_cc_output, second_cc_input, second_cc_output
            )

            # The leaf (topological) instance of each name in the first state
            first_leaf_by_data: dict[str, nodes.AccessNode] = {}
            for x in reversed(list(nx.topological_sort(first_state._nx))):
                if isinstance(x, nodes.AccessNode) and x.data not in first_leaf_by_data:
                    first_leaf_by_data[x.data] = x

            # Check for data races
            for fused_cc in resulting_ccs:
                # Write-Write hazard - data is output of both first and second
                # states, without a read in between
                write_write_candidates = (fused_cc.first_outputs & fused_cc.second_outputs) - fused_cc.second_inputs

                # Those nodes will be the connection points upon fusion
                cc_second_inputs = in_state_order(fused_cc.second_input_nodes, second_order)
                match_nodes: dict[nodes.AccessNode, nodes.AccessNode] = {
                    first_leaf_by_data[match]: next(n for n in cc_second_inputs if n.data == match)
                    for match in sorted(fused_cc.first_outputs & fused_cc.second_inputs)
                }

                # If we have potential candidates, check if there is a
                # path from the first write to the second write (in that
                # case, there is no hazard):
                for cand in sorted(write_write_candidates):
                    nodes_first = first_output_by_data.get(cand, [])
                    nodes_second = second_output_by_data.get(cand, [])

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    if not self._check_paths(
                        first_state, second_state, match_nodes, nodes_first, nodes_second, False, False
                    ):
                        return False
                # End of write-write hazard check

                first_inout = sorted(fused_cc.first_inputs | fused_cc.first_outputs)
                for other_cc in resulting_ccs:
                    # NOTE: Special handling for `other_cc is fused_cc`
                    if other_cc is fused_cc:
                        # Checking for potential Read-Write data races
                        for d in first_inout:
                            if d in other_cc.second_outputs:
                                nodes_second = second_output_by_data.get(d, [])
                                # Read-Write race
                                if d in fused_cc.first_inputs:
                                    nodes_first = first_input_by_data.get(d, [])
                                else:
                                    nodes_first = []
                                for n2 in nodes_second:
                                    for e in second_state.in_edges(n2):
                                        path = second_state.memlet_path(e)
                                        src = path[0].src
                                        if src in second_input_set and src.data in fused_cc.first_outputs:
                                            for n1 in fused_cc.first_output_nodes:
                                                if n1.data == src.data:
                                                    for n0 in nodes_first:
                                                        if not nx.has_path(first_state._nx, n0, n1):
                                                            return False
                                # Read-write hazard where an access node is connected
                                # to more than one output at once: (a) -> (b)  |  (d) -> [code] -> (d)
                                #                                     \-> (c)  |
                                # in the first state, and the same memory is inout in the second state
                                # All paths need to lead to `src`
                                if not self._check_all_paths(
                                    first_state, second_state, match_nodes, nodes_first, nodes_second, True, False
                                ):
                                    return False
                                # Same-cc write-after-read is ordered by the WAR edges recorded above

                        continue
                    # If an input/output of a connected component in the first
                    # state is an output of another connected component in the
                    # second state, we have a potential data race (Read-Write
                    # or Write-Write). Read-Write is ordered by the WAR edges recorded above.
                    for d in first_inout:
                        if d in other_cc.second_outputs and d in fused_cc.first_outputs:
                            # Write-Write race: the second write must stay last, so order it after the first
                            nodes_first = first_output_by_data.get(d, [])
                            nodes_second = second_output_by_data.get(d, [])
                            if StateFusionExtended.memlets_intersect(
                                first_state, nodes_first, False, second_state, nodes_second, False
                            ):
                                self._connections_to_make.append(("waw", nodes_first, nodes_second))
                    # End of data race check

                # Read-after-write dependencies: if there is an output of the
                # second state that is an input of the first, ensure all paths
                # from the input of the first state lead to the output.
                # Otherwise, there may be a RAW due to topological sort or
                # concurrency.
                second_inout = (fused_cc.first_inputs | fused_cc.first_outputs) & fused_cc.second_outputs
                for inout in sorted(second_inout):
                    nodes_first = [n for n in match_nodes if n.data == inout]
                    if any(first_state.out_degree(n) > 0 for n in nodes_first):
                        return False

                    # If we have potential candidates, check if there is a
                    # path from the first read to the second write (in that
                    # case, there is no hazard):
                    nodes_first = in_state_order(
                        {n for n in fused_cc.first_input_nodes + fused_cc.first_output_nodes if n.data == inout},
                        first_order,
                    )
                    nodes_second = in_state_order(
                        {n for n in fused_cc.second_output_nodes if n.data == inout}, second_order
                    )

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    if not self._check_paths(
                        first_state, second_state, match_nodes, nodes_first, nodes_second, True, False
                    ):
                        return False

                # End of read-write hazard check

                # Read-after-write dependencies: if there is more than one first
                # output with the same data, make sure it can be unambiguously
                # connected to the second state
                if len(fused_cc.first_output_nodes) > len(fused_cc.first_outputs):
                    for inpnode in fused_cc.second_input_nodes:
                        found = None
                        for outnode in fused_cc.first_output_nodes:
                            if outnode.data != inpnode.data:
                                continue
                            if StateFusionExtended.memlets_intersect(
                                first_state, [outnode], False, second_state, [inpnode], True
                            ):
                                # If found more than once, either there is a
                                # path from one to another or it is ambiguous
                                if found is not None:
                                    if nx.has_path(first_state.nx, outnode, found):
                                        # Found is a descendant, continue
                                        continue
                                    elif nx.has_path(first_state.nx, found, outnode):
                                        # New node is a descendant, set as found
                                        found = outnode
                                    else:
                                        # No path: ambiguous match
                                        return False
                                found = outnode

        return True

    def apply(self, graph: ControlFlowRegion, sdfg):
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state

        # This will populate `self._connections_to_make`, in the mode the match was made in. A refusal leaves
        # it incomplete, and wiring part of the happens-before edges would drop an ordering.
        if not self.can_be_applied(graph, self.expr_index, sdfg, permissive=self._matched_permissive):
            return

        # Remove interstate edge(s)
        edges = graph.edges_between(first_state, second_state)
        for edge in edges:
            if edge.data.assignments:
                for src, dst, other_data in graph.in_edges(first_state):
                    other_data.assignments.update(edge.data.assignments)
            graph.remove_edge(edge)

        # Special case 1: first state is empty
        if first_state.is_empty():
            was_start = is_start_block(graph, first_state)
            sdutil.change_edge_dest(graph, first_state, second_state)
            graph.remove_node(first_state)
            if was_start and len(graph.source_nodes()) != 1:
                graph.start_block = graph.node_id(second_state)
            return

        # Special case 2: second state is empty
        if second_state.is_empty():
            was_start = is_start_block(graph, second_state)
            sdutil.change_edge_src(graph, second_state, first_state)
            sdutil.change_edge_dest(graph, second_state, first_state)
            graph.remove_node(second_state)
            if was_start and len(graph.source_nodes()) != 1:
                graph.start_block = graph.node_id(first_state)
            return

        # Normal case: both states are not empty

        # Find source/sink (data) nodes
        top2 = top_level_nodes(second_state)
        second_sources = set(second_state.source_nodes())

        # NOTE: We exclude Views from the process of merging common data nodes because it may lead to double edges.
        second_mid = [
            x
            for x in list(nx.topological_sort(second_state._nx))
            if isinstance(x, nodes.AccessNode)
            and second_state.out_degree(x) > 0
            and not isinstance(sdfg.arrays[x.data], dt.View)
        ]

        # Merge second state to first state
        # First keep a backup of the topological sorted order of the nodes
        sdict = first_state.scope_dict()
        order = [
            x
            for x in reversed(list(nx.topological_sort(first_state._nx)))
            if isinstance(x, nodes.AccessNode) and sdict[x] is None
        ]
        # A seeded accumulation is ordered by its ``seed`` entry; its WAW edges would land on second-state
        # sources that may merge upstream of the seed and close a cycle (ludcmp ``w``)
        seeded = {n for kind, _, second_nodes in self._connections_to_make if kind == "seed" for n in second_nodes}
        for node in second_state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                # update parent information
                node.sdfg.parent = first_state

            # The node could have been added when adding connections by add_nedge hence the need to check
            if node not in first_state.nodes():
                first_state.add_node(node)

            # Wire the recorded happens-before edges onto every second-state source leading to the hazard node
            for kind, first_nodes, second_nodes in self._connections_to_make:
                if node not in second_nodes or (kind == "waw" and node in seeded):
                    continue
                for i in top2:
                    if i not in second_sources or (i is not node and not nx.has_path(second_state._nx, i, node)):
                        continue
                    if kind == "seed" and isinstance(i, nodes.AccessNode):
                        # A source AccessNode may merge upstream of the seed writer (ludcmp ``w = y[i]``); order
                        # its consumers instead, or ``i`` itself for a library node (it counts its inputs)
                        for j in first_nodes:
                            for k in dict.fromkeys(e.dst for e in second_state.out_edges(i)):
                                if not nx.has_path(second_state._nx, k, node):
                                    continue
                                target = i if isinstance(k, nodes.LibraryNode) else k
                                if not first_state.edges_between(j, target):
                                    first_state.add_nedge(j, target, memlet.Memlet())
                        continue
                    for j in first_nodes:
                        first_state.add_nedge(j, i, memlet.Memlet())
        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        top = top_level_nodes(first_state)
        waw_flagged = {n for kind, _, second_nodes in self._connections_to_make if kind == "waw" for n in second_nodes}

        # Merge common (data) nodes
        merged_nodes = set()
        for node in second_mid:
            # merge only top level nodes, skip everything else
            if node not in top2:
                continue

            candidates = [x for x in order if x.data == node.data and x in top and x not in merged_nodes]
            source_node = first_state.in_degree(node) == 0

            # If not source node, try to connect every memlet-intersecting candidate
            if not source_node:
                for cand in candidates:
                    # A flagged write-write pair stays two nodes, ordered by the edge wired above
                    if node in waw_flagged and first_state.in_degree(cand) > 0:
                        continue
                    if StateFusionExtended.memlets_intersect(first_state, [cand], False, second_state, [node], True):
                        if nx.has_path(first_state._nx, cand, node):  # Do not create cycles
                            continue
                        sdutil.change_edge_src(first_state, cand, node)
                        sdutil.change_edge_dest(first_state, cand, node)
                        first_state.remove_node(cand)
                        merged_nodes.add(cand)
                continue

            if len(candidates) == 0:
                continue
            elif len(candidates) == 1:
                n = candidates[0]
            else:
                # Choose first candidate that intersects memlets
                for cand in candidates:
                    if StateFusionExtended.memlets_intersect(first_state, [cand], False, second_state, [node], True):
                        n = cand
                        break
                else:
                    # No node intersects, use topologically-last node
                    n = candidates[0]

            sdutil.change_edge_src(first_state, node, n)
            sdutil.change_edge_dest(first_state, node, n)
            first_state.remove_node(node)
            merged_nodes.add(n)

        # Redirect edges and remove second state
        was_start = is_start_block(graph, second_state)
        sdutil.change_edge_src(graph, second_state, first_state)
        graph.remove_node(second_state)
        if was_start and len(graph.source_nodes()) != 1:
            graph.start_block = graph.node_id(first_state)

        # Technically unneeded, but better to keep track.
        self._connections_to_make.clear()

        self._post_apply_check(first_state, sdfg)

    def _post_apply_check(self, fused_state: SDFGState, sdfg) -> None:
        """Validate the fused state, so a bad merge fails here instead of in codegen.

        :raises InvalidSDFGEdgeError: on an invalid edge, e.g. a ``memlet.data`` naming neither endpoint.
        """
        try:
            validate_state(fused_state, sdfg=sdfg, initialized_transients=set(sdfg.arrays))
        except InvalidSDFGEdgeError as ex:
            raise InvalidSDFGEdgeError(
                f"StateFusionExtended produced an invalid edge: {ex.message}", ex.sdfg, ex.state_id, ex.edge_id
            ) from ex
        if self.strict_validate:
            sdfg.validate()
