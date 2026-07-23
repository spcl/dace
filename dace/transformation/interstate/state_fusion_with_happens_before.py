# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

from typing import Dict, List

import networkx as nx

from dace import data as dt, properties, sdfg, subsets, memlet
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.sdfg.validation import InvalidSDFGEdgeError
from dace.transformation import transformation


# Helper class for finding connected component correspondences
class CCDesc:

    def __init__(self, first_input_nodes: List[nodes.AccessNode], first_output_nodes: List[nodes.AccessNode],
                 second_input_nodes: List[nodes.AccessNode], second_output_nodes: List[nodes.AccessNode]) -> None:
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


def in_state_order(state: SDFGState, node_iter) -> List[nodes.Node]:
    """Deterministic order for a collection of nodes of ``state``.

    SDFG nodes hash by ``id()``, so iterating a ``set`` of them yields a different order on
    every run (and a ``set`` of *names* varies too, since string hashing is salted per
    process). That matters here because several decisions are first-match-wins over such a
    collection -- which second-state node a match binds to, which match ``_check_paths``
    stops at -- and because the order in which the happens-before edges are recorded is the
    order in which they are inserted into the fused state, which downstream passes see.
    Ordering by position in the state makes all of it reproducible.
    """
    # ``state.node_id`` is a linear scan (graph.py), so ``key=state.node_id`` is O(n) PER
    #  element -- O(n^2) across a call, and this runs several times per ``can_be_applied`` inside
    #  the StateFusionExtended fixpoint (measured: the dominant cost on cloudsc-sized states).
    #  ``node_id`` is exactly the index into ``state.nodes()``, so a one-shot position map gives
    #  the identical order in O(n + m log m).
    order = {node: i for i, node in enumerate(state.nodes())}
    return sorted(node_iter, key=order.__getitem__)


@transformation.explicit_cf_compatible
@properties.make_properties
class StateFusionExtended(transformation.MultiStateTransformation):
    """ Implements the state-fusion transformation extended to fuse states with RAW and WAW dependencies.
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
        dtype=bool,
        default=False,
        desc=("Run ``sdfg.validate()`` after ``apply`` (paranoid debug mode). When ``False`` (the default in hot "
              "paths), only a cheap structural check on the fused state runs -- catches the historical bug class "
              "(orphan nodes, mismatched ``memlet.data``) without paying the full-SDFG-validate cost. The cheap "
              "check is *always* on; this knob only adds the full validate on top."),
    )

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_state, cls.second_state)]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connections_to_make = []

    @property
    def connections_to_make(self):
        return self._connections_to_make

    @staticmethod
    def find_fused_components(first_cc_input, first_cc_output, second_cc_input, second_cc_output) -> List[CCDesc]:
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

        # Construct result out of connected components of the bipartite graph. Both the
        #  components and their contents are sorted: they are `set`s of `(graph, index)`
        #  tuples, so their natural iteration order is not reproducible, and it decides the
        #  order in which the happens-before edges below are recorded.
        result = []
        for cc in sorted((sorted(cc) for cc in nx.weakly_connected_components(g))):
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

    @staticmethod
    def memlets_intersect(graph_a: SDFGState, group_a: List[nodes.AccessNode], inputs_a: bool, graph_b: SDFGState,
                          group_b: List[nodes.AccessNode], inputs_b: bool) -> bool:
        """
        Performs an all-pairs check for subset intersection on two
        groups of nodes. If group intersects or result is indeterminate,
        returns True as a precaution.

        :param graph_a: The graph in which the first set of nodes reside.
        :param group_a: The first set of nodes to check.
        :param inputs_a: If True, checks inputs of the first group.
        :param graph_b: The graph in which the second set of nodes reside.
        :param group_b: The second set of nodes to check.
        :param inputs_b: If True, checks inputs of the second group.
        :return: True if subsets intersect or result is indeterminate.
        """
        # Set traversal functions
        src_subset = lambda e: (e.data.src_subset if e.data.src_subset is not None else e.data.dst_subset)
        dst_subset = lambda e: (e.data.dst_subset if e.data.dst_subset is not None else e.data.src_subset)
        if inputs_a:
            edges_a = [e for n in group_a for e in graph_a.out_edges(n)]
            subset_a = src_subset
        else:
            edges_a = [e for n in group_a for e in graph_a.in_edges(n)]
            subset_a = dst_subset
        if inputs_b:
            edges_b = [e for n in group_b for e in graph_b.out_edges(n)]
            subset_b = src_subset
        else:
            edges_b = [e for n in group_b for e in graph_b.in_edges(n)]
            subset_b = dst_subset

        # Simple all-pairs check
        for ea in edges_a:
            for eb in edges_b:
                result = subsets.intersects(subset_a(ea), subset_b(eb))
                if result is True or result is None:
                    return True
        return False

    def has_path(self, first_state: SDFGState, second_state: SDFGState,
                 match_nodes: Dict[nodes.AccessNode, nodes.AccessNode], node_a: nodes.Node, node_b: nodes.Node) -> bool:
        """ Check for paths between the two states if they are fused. """
        for match_a, match_b in match_nodes.items():
            if nx.has_path(first_state._nx, node_a, match_a) and nx.has_path(second_state._nx, match_b, node_b):
                return True
        return False

    def _check_all_paths(self, first_state: SDFGState, second_state: SDFGState,
                         match_nodes: Dict[nodes.AccessNode, nodes.AccessNode], nodes_first: List[nodes.AccessNode],
                         nodes_second: List[nodes.AccessNode], first_read: bool, second_read: bool) -> bool:
        for node_a in nodes_first:
            succ_a = first_state.successors(node_a)
            for node_b in nodes_second:
                if all(self.has_path(first_state, second_state, match_nodes, sa, node_b) for sa in succ_a):
                    return True
        # Path not found, check memlets
        if StateFusionExtended.memlets_intersect(first_state, nodes_first, first_read, second_state, nodes_second,
                                                 second_read):
            return False
        return True

    def _check_paths(self, first_state: SDFGState, second_state: SDFGState,
                     match_nodes: Dict[nodes.AccessNode, nodes.AccessNode], nodes_first: List[nodes.AccessNode],
                     nodes_second: List[nodes.AccessNode], first_read: bool, second_read: bool) -> bool:
        # The hazard is covered when the ordering the interstate edge used to give us is
        #  reproduced by dataflow once the states are merged: a first-state node reaches a
        #  second-state node through a node that exists in both, i.e. through one of the
        #  match nodes that the merge collapses into one.
        #
        # NOTE: This used to stop at the FIRST match that had a path in the first state and
        #  decide on that one alone. The verdict then depended on which element a `set`
        #  happened to yield first, so the very same SDFG fused on one run and not on the
        #  next (TSVC `s253`). It also let one ordered node stand in for its unordered
        #  siblings.
        #
        # So: try EVERY candidate and take the first that proves ordering. The verdict is an
        #  ``any`` over candidates, hence independent of the order they are visited in -- which
        #  is what the s253 flake actually needed. (``match_nodes`` is built from a ``sorted``
        #  intersection above, so its order is stable too; this does not rely on that.)
        #
        # ⛔ Deliberately NOT strengthened to "every (first, second) pair is connected by some
        #  match". That property is stronger than the hazard being covered, and pricing it costs
        #  a BFS per (first_node, second_node, match) triple -- measured 14.8s vs 0.1ms at 160
        #  nodes, which stalled the cloudsc ``pretreat`` phase for over an hour. The weaker
        #  predicate below is the pre-existing semantics, and anything it fails to prove falls
        #  through to the ``memlets_intersect`` check rather than being waved through.
        for match, second_match in match_nodes.items():
            # ``all`` on BOTH sides: a match only covers the hazard if EVERY first-state node
            #  reaches it and EVERY second-state node is reached from its partner. Using ``any``
            #  on the first side is unsound for >= 2 writers -- a sibling writer that does not
            #  reach the match is then left unordered against the second-state write, yet the
            #  fusion is declared safe (silent write-write miscompile). ``all`` == ``any`` for a
            #  single node, so the s253 case this scan fixes is unchanged; a genuinely
            #  multi-writer case that no single match covers falls through to
            #  ``memlets_intersect`` and is rejected on overlap, never waved through.
            if not all(nx.has_path(first_state._nx, first_node, match) for first_node in nodes_first):
                continue
            if all(nx.has_path(second_state._nx, second_match, second_node) for second_node in nodes_second):
                return True

        # Not ordered by dataflow. Only an actual overlap is a hazard.
        return not StateFusionExtended.memlets_intersect(first_state, nodes_first, first_read, second_state,
                                                         nodes_second, second_read)

    @staticmethod
    def state_has_side_effect_node(state: SDFGState, sdfg) -> bool:
        """Whether ``state`` carries a node whose execution has side effects (a Tasklet with
        ``side_effects`` / a callback call, a side-effecting library node, or a nested SDFG that
        contains one). Such a node relies on the ORDER the interstate edge imposes between the two
        states; fusing collapses that to intra-state dataflow order, under which a side-effect node
        with no data dependence tying it to its neighbours could execute in any order (or
        concurrently in a parallel scope), reordering or dropping the effect."""
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet) and node.has_side_effects(sdfg):
                return True
            if isinstance(node, nodes.LibraryNode) and node.has_side_effects:
                return True
            if isinstance(node, nodes.NestedSDFG):
                for nested_state in node.sdfg.states():
                    if StateFusionExtended.state_has_side_effect_node(nested_state, node.sdfg):
                        return True
        return False

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state
        # We keep it alive, such that `apply()` can use it later.
        self.connections_to_make.clear()

        # Do not fuse states that carry a side-effect node: state fusion preserves only dataflow
        # order, so a side effect (I/O, a trap guard, a stateful library / MPI call) whose order
        # is guaranteed by the interstate edge -- not by a data dependence -- would be reordered
        # or run concurrently once the two states become one.
        if (StateFusionExtended.state_has_side_effect_node(first_state, sdfg)
                or StateFusionExtended.state_has_side_effect_node(second_state, sdfg)):
            return False

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
            if freesyms and any(n.data in freesyms for n in first_state.nodes()
                                if isinstance(n, nodes.AccessNode) and first_state.in_degree(n) > 0):
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

        if not permissive:
            # Strict mode that inhibits state fusion if Python callbacks are involved
            if Config.get_bool('frontend', 'dont_fuse_callbacks'):
                for node in (first_state.data_nodes() + second_state.data_nodes()):
                    if node.data == '__pystate':
                        return False

            # Library nodes and nested SDFGs carry dependencies fusion cannot see.
            # Tasklet callbacks are governed by dont_fuse_callbacks above.
            for state in (first_state, second_state):
                for node in state.nodes():
                    if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)) and node.has_side_effects(sdfg):
                        return False

            # Reused-transient ambiguity: when BOTH states write into the same
            # non-view transient that already carries more than one top-level
            # producer AccessNode in the first state, the two chains feed
            # distinct values into aliased instances of one name, and the
            # merge's common-data-node collapse cross-binds a reader to the
            # wrong producer. Unlike a missing happens-before edge, a
            # wrong-value binding cannot be repaired by an ordering edge, so
            # refuse. Pinned by nbody: the reused mean transient ``__rdo0``
            # accumulates several component writes across successive fusions and
            # the KE reduction ends up reading the wrong one.
            first_scope = first_state.scope_dict()
            first_producers: Dict[str, int] = {}
            for n in first_state.data_nodes():
                if first_scope[n] is None and first_state.in_degree(n) > 0:
                    first_producers[n.data] = first_producers.get(n.data, 0) + 1
            second_scope = second_state.scope_dict()
            second_writes = {
                n.data
                for n in second_state.data_nodes() if second_scope[n] is None and second_state.in_degree(n) > 0
            }
            for dname in second_writes:
                desc = sdfg.arrays.get(dname)
                if desc is None or not desc.transient or isinstance(desc, dt.View):
                    continue
                if first_producers.get(dname, 0) >= 2:
                    return False

            # If second state has other input edges, there might be issues
            # Exceptions are when none of the states contain dataflow, unless
            # the first state is an initial state (in which case the new initial
            # state would be ambiguous).
            first_in_edges = graph.in_edges(first_state)
            second_in_edges = graph.in_edges(second_state)
            if ((not second_state.is_empty() or not first_state.is_empty() or len(first_in_edges) == 0)
                    and len(second_in_edges) != 1):
                return False

            # Get connected components. `weakly_connected_components` hands back `set`s, so
            #  both the components and their contents are put in state order, see
            #  `in_state_order()`.
            first_cc = sorted(
                (in_state_order(first_state, cc) for cc in nx.weakly_connected_components(first_state._nx)),
                key=lambda cc: first_state.node_id(cc[0]))
            second_cc = sorted(
                (in_state_order(second_state, cc) for cc in nx.weakly_connected_components(second_state._nx)),
                key=lambda cc: second_state.node_id(cc[0]))

            # Find source/sink (data) nodes, again in state order rather than in `set` order.
            top1, top2 = top_level_nodes(first_state), top_level_nodes(second_state)
            first_input = in_state_order(first_state,
                                         [n for n in first_state.source_nodes() if isinstance(n, nodes.AccessNode)])
            first_input_set = set(first_input)
            first_output = in_state_order(
                first_state, [n for n in top1 if isinstance(n, nodes.AccessNode) and n not in first_input_set])
            first_output_set = set(first_output)
            second_input = in_state_order(second_state,
                                          [n for n in second_state.source_nodes() if isinstance(n, nodes.AccessNode)])
            second_input_set = set(second_input)
            second_output = in_state_order(
                second_state, [n for n in top2 if isinstance(n, nodes.AccessNode) and n not in second_input_set])
            second_output_set = set(second_output)

            # WCR is a read-modify-write: an accumulate ``a(+)= ...`` implicitly READS
            # the prior (seed) value of ``a``. If the FIRST state WRITES an array the
            # SECOND state then WCR-accumulates (overlapping subset), fusing the two
            # states into one drops the seed->accumulate ordering -- the implicit seed
            # read is not an edge, so a later MapFusion / topological reorder can run the
            # accumulate before the seed init, zeroing the result (covariance /
            # correlation ``mean[:] = 0.0; mean(+)= data[...]``). The states must stay
            # ordered, so refuse the fusion (the seed read is a genuine RAW dependency).
            first_written: Dict[str, List] = {}
            for n in first_output:
                for e in first_state.in_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        ss = e.data.get_dst_subset(e, first_state) or e.data.subset
                        if ss is not None:
                            first_written.setdefault(n.data, []).append(ss)
            for e in second_state.edges():
                if e.data is None or e.data.wcr is None or e.data.data not in first_written:
                    continue
                wsub = e.data.get_dst_subset(e, second_state) or e.data.subset
                if wsub is None or any(subsets.intersects(wsub, fs) is not False for fs in first_written[e.data.data]):
                    return False

            # Write-after-read into a sink: a second-state write to an element the
            # first state reads, landing on a pure sink (no outgoing edges), cannot be
            # ordered after that read by a dependency edge (the edge would have to
            # target the read), so it is refused for now -- per write edge, and
            # exempting writes whose value flows from data the first state produced (a
            # ``read -> ... -> write`` path orders them safely).
            #
            # ``first_out_data`` for the ``flows_from_first`` exemption must
            # only include AccessNodes whose in-edges are REAL data writes.
            # Empty-memlet edges added by a PRIOR fusion's ``connections_to_make``
            # encode happens-before, not data flow, so an empty-edge-only
            # AccessNode is a happens-before sink, not a producer; counting it
            # in ``first_out_data`` falsely exempts a downstream WAR. Pinned by
            # the peeled-pattern test (peeled chain's empty edge to remainder's
            # ``B`` source previously fooled the check into accepting the
            # racy merged+remainder fusion).
            def _has_data_write(n):
                ies = first_state.in_edges(n)
                return any(e.data is not None and not e.data.is_empty() for e in ies)

            first_out_data = {n.data for n in first_output if _has_data_write(n)}
            # Top-level first-state readers of each array (node + read subset). Used to
            # order first-state reads BEFORE a second-state overwrite (WAR anti-dep).
            first_read_subsets: Dict[str, List] = {}
            first_readers: Dict[str, List[nodes.AccessNode]] = {}
            first_scope = first_state.scope_dict()
            for rn in first_state.data_nodes():
                if first_scope[rn] is not None:
                    continue
                for re in first_state.out_edges(rn):
                    # Fall back to the other side, as ``memlets_intersect`` does: a
                    # single-sided memlet on an AccessNode->AccessNode copy (e.g.
                    # ``Memlet('Tm[0:8]')`` on ``A -> Tm``, naming only the dst) leaves
                    # ``src_subset`` None, and without the fallback the first-state READ of
                    # ``A`` would never be recorded and the WAR silently missed.
                    ss = re.data.get_src_subset(re, first_state)
                    if ss is None:
                        ss = re.data.get_dst_subset(re, first_state)
                    if ss is not None:
                        first_read_subsets.setdefault(rn.data, []).append(ss)
                        first_readers.setdefault(rn.data, []).append(rn)

            # WAR (write-after-read / anti-dependency): the first state READS ``d`` and the
            # second state WRITES an overlapping element of ``d``. The read must see the OLD
            # value, so every first-state reader of ``d`` must be ordered BEFORE the
            # second-state writer. Since ``d`` is a false (anti) dependency -- no data flows
            # read->write -- a happens-before empty edge (first-reader -> second-writer) is
            # the correct fix; no node merge. Recorded here, wired in ``apply``.
            #
            # Exemption: when the write VALUE already flows from first-produced data, the
            # existing dataflow orders the read before the write -- add no edge. This must be
            # a PATH property, not a name match: the exempting ancestor has to be a
            # second-state SOURCE (only those merge with the first state's producer, which is
            # what creates the ordering path) whose data the first state really writes, AND
            # every first-state reader of ``d`` must actually reach that producer. A global
            # name-only test wrongly exempts (a) a read sitting in a different first-state
            # connected component than the producer, and (b) a transient the SECOND state
            # itself re-writes -- both silent WAR miscompiles. ``first_out_data`` counts only
            # REAL data writers; an empty-memlet happens-before sink from a prior fusion is
            # not a producer (peeled-pattern regression).
            def _ordered_by_existing_dataflow(readers, we) -> bool:
                for a in (nx.ancestors(second_state._nx, we.src) | {we.src}):
                    if not isinstance(a, nodes.AccessNode) or a.data not in first_out_data:
                        continue
                    if second_state.in_degree(a) != 0:
                        continue  # re-written inside the second state: not a merge point
                    producers = [n for n in first_output if n.data == a.data and _has_data_write(n)]
                    if producers and all(any(nx.has_path(first_state._nx, r, p) for p in producers) for r in readers):
                        return True
                return False

            for wnode in second_output:
                if wnode.data not in first_read_subsets:
                    continue
                for we in second_state.in_edges(wnode):
                    wsub = we.data.get_dst_subset(we, second_state)
                    if wsub is None:
                        wsub = we.data.get_src_subset(we, second_state)
                    if wsub is None:
                        continue
                    if _ordered_by_existing_dataflow(first_readers[wnode.data], we):
                        continue
                    if any(subsets.intersects(wsub, rs) is not False for rs in first_read_subsets[wnode.data]):
                        # The anti-dependency is on the READ ITSELF, which happens at the
                        # reader's CONSUMER (the tasklet / map that reads through the access
                        # node), not at the access node (a pure source node has no
                        # computation and is always "ready"). Ordering only the access node
                        # before the write leaves the consumer a free sibling that codegen
                        # may still schedule after the overwrite. So order every consumer of
                        # every first-state reader of ``d`` before the second-state writer.
                        #
                        # When the consumer is a scope ENTRY (the read feeds a Map/Consume),
                        # the ordering endpoint must be the matching EXIT, not the entry: an
                        # edge out of an entry node is *inside* that scope, so it would drag
                        # the whole second-state subgraph into the map scope (scope_dict
                        # recurses through every successor of an EntryNode) -- a silent
                        # miscompile that ``validate()`` does not catch. The exit also orders
                        # the map's internal reads, which is exactly what the WAR needs.
                        consumers = []
                        for rn in first_readers[wnode.data]:
                            for e in first_state.out_edges(rn):
                                cons = e.dst
                                if isinstance(cons, nodes.EntryNode):
                                    cons = first_state.exit_node(cons)
                                consumers.append(cons)
                        consumers = list(dict.fromkeys(consumers))
                        if consumers:
                            self.connections_to_make.append(('war', consumers, [wnode]))
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
                cc_appearances = 0
                for cc in first_cc_output:
                    if len([n for n in cc if n.data == match]) > 0:
                        cc_appearances += 1
                if cc_appearances > 1:
                    return False

            # Recreate fused connected component correspondences, and then
            # check for hazards
            resulting_ccs: List[CCDesc] = StateFusionExtended.find_fused_components(first_cc_input, first_cc_output,
                                                                                    second_cc_input, second_cc_output)

            if len(resulting_ccs) > 1:
                # Declared side effects would race across parallel components.
                for state in (first_state, second_state):
                    for node in state.nodes():
                        if isinstance(node, nodes.Tasklet) and node.side_effects:
                            return False

            # Check for data races
            for fused_cc in resulting_ccs:
                # Write-Write hazard - data is output of both first and second
                # states, without a read in between
                write_write_candidates = ((fused_cc.first_outputs & fused_cc.second_outputs) - fused_cc.second_inputs)

                # Find the leaf (topological) instances of the matches
                order = [
                    x for x in reversed(list(nx.topological_sort(first_state._nx)))
                    if isinstance(x, nodes.AccessNode) and x.data in fused_cc.first_outputs
                ]
                # Those nodes will be the connection points upon fusion. Both the names and
                #  the second-state candidates are ordered: this is a first-match-wins bind
                #  and `_check_paths()` stops at the first match that has a path, so an
                #  arbitrary order makes the verdict differ between runs.
                match_nodes: Dict[nodes.AccessNode, nodes.AccessNode] = {
                    next(n for n in order if n.data == match):
                    next(n for n in in_state_order(second_state, fused_cc.second_input_nodes) if n.data == match)
                    for match in sorted(fused_cc.first_outputs & fused_cc.second_inputs)
                }

                # If we have potential candidates, check if there is a
                # path from the first write to the second write (in that
                # case, there is no hazard):
                for cand in sorted(write_write_candidates):
                    nodes_first = [n for n in first_output if n.data == cand]
                    nodes_second = [n for n in second_output if n.data == cand]

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    if not self._check_paths(first_state, second_state, match_nodes, nodes_first, nodes_second, False,
                                             False):
                        return False
                # End of write-write hazard check

                first_inout = fused_cc.first_inputs | fused_cc.first_outputs
                for other_cc in resulting_ccs:
                    # NOTE: Special handling for `other_cc is fused_cc`
                    if other_cc is fused_cc:
                        # Checking for potential Read-Write data races
                        for d in sorted(first_inout):
                            if d in other_cc.second_outputs:
                                nodes_second = [n for n in second_output if n.data == d]
                                # Read-Write race
                                if d in fused_cc.first_inputs:
                                    nodes_first = [n for n in first_input if n.data == d]
                                else:
                                    nodes_first = []
                                for n2 in nodes_second:
                                    for e in second_state.in_edges(n2):
                                        path = second_state.memlet_path(e)
                                        src = path[0].src
                                        if src in second_input and src.data in fused_cc.first_outputs:
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
                                if not self._check_all_paths(first_state, second_state, match_nodes, nodes_first,
                                                             nodes_second, True, False):
                                    return False

                                # Same-cc write-after-read (first reads ``d``, second writes
                                # it) is handled uniformly by the consolidated WAR recorder
                                # above: it adds a first-reader -> second-writer happens-before
                                # edge (or exempts it when the write value flows from
                                # first-produced data). No reject here.

                        continue
                    # If an input/output of a connected component in the first
                    # state is an output of another connected component in the
                    # second state, we have a potential data race (Read-Write
                    # or Write-Write)
                    for d in sorted(first_inout):
                        if d in other_cc.second_outputs:
                            # Check for intersection (if None, fusion is ok)
                            nodes_second = [n for n in second_output if n.data == d]
                            # Cross-cc write-after-read (first reads ``d``, second writes it)
                            # is handled uniformly by the consolidated WAR recorder above
                            # (first-reader -> second-writer happens-before edge).
                            # Write-Write race (output dependency): both states write ``d``.
                            # Last writer (second) wins. Keep the two writer nodes distinct
                            # and add a happens-before edge first-write -> second-write so a
                            # later reorder cannot flip the last writer.
                            if d in fused_cc.first_outputs:
                                nodes_first = [n for n in first_output if n.data == d]
                                if StateFusionExtended.memlets_intersect(first_state, nodes_first, False, second_state,
                                                                         nodes_second, False):
                                    self.connections_to_make.append(('waw', nodes_first, nodes_second))
                    # End of data race check

                # Read-after-write dependencies: if there is an output of the
                # second state that is an input of the first, ensure all paths
                # from the input of the first state lead to the output.
                # Otherwise, there may be a RAW due to topological sort or
                # concurrency.
                second_inout = ((fused_cc.first_inputs | fused_cc.first_outputs) & fused_cc.second_outputs)
                for inout in sorted(second_inout):
                    nodes_first = [n for n in match_nodes if n.data == inout]
                    if any(first_state.out_degree(n) > 0 for n in nodes_first):
                        return False

                    # If we have potential candidates, check if there is a
                    # path from the first read to the second write (in that
                    # case, there is no hazard):
                    nodes_first = in_state_order(
                        first_state,
                        {n
                         for n in fused_cc.first_input_nodes + fused_cc.first_output_nodes if n.data == inout})
                    nodes_second = in_state_order(second_state,
                                                  {n
                                                   for n in fused_cc.second_output_nodes if n.data == inout})

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    if not self._check_paths(first_state, second_state, match_nodes, nodes_first, nodes_second, True,
                                             False):
                        return False

                # End of read-write hazard check

                # Read-after-write dependencies: if there is more than one first
                # output with the same data, make sure it can be unambiguously
                # connected to the second state
                if (len(fused_cc.first_output_nodes) > len(fused_cc.first_outputs)):
                    for inpnode in fused_cc.second_input_nodes:
                        found = None
                        for outnode in fused_cc.first_output_nodes:
                            if outnode.data != inpnode.data:
                                continue
                            if StateFusionExtended.memlets_intersect(first_state, [outnode], False, second_state,
                                                                     [inpnode], True):
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

        # This will populate `self.connections_to_make`.
        self.can_be_applied(graph, 0, sdfg)

        # Remove interstate edge(s)
        edges = graph.edges_between(first_state, second_state)
        for edge in edges:
            if edge.data.assignments:
                for src, dst, other_data in graph.in_edges(first_state):
                    other_data.assignments.update(edge.data.assignments)
            graph.remove_edge(edge)

        # Special case 1: first state is empty
        if first_state.is_empty():
            sdutil.change_edge_dest(graph, first_state, second_state)
            graph.remove_node(first_state)
            if graph.start_block == first_state:
                graph.start_block = graph.node_id(second_state)
            return

        # Special case 2: second state is empty
        if second_state.is_empty():
            sdutil.change_edge_src(graph, second_state, first_state)
            sdutil.change_edge_dest(graph, second_state, first_state)
            graph.remove_node(second_state)
            if graph.start_block == second_state:
                graph.start_block = graph.node_id(first_state)
            return

        # Normal case: both states are not empty

        # Find source/sink (data) nodes
        first_input = [node for node in first_state.source_nodes() if isinstance(node, nodes.AccessNode)]
        first_output = [node for node in first_state.sink_nodes() if isinstance(node, nodes.AccessNode)]

        top2 = top_level_nodes(second_state)

        # first input = first input - first output
        first_input = [
            node for node in first_input if next((x for x in first_output if x.data == node.data), None) is None
        ]

        # NOTE: We exclude Views from the process of merging common data nodes because it may lead to double edges.
        second_mid = [
            x for x in list(nx.topological_sort(second_state._nx)) if isinstance(x, nodes.AccessNode)
            and second_state.out_degree(x) > 0 and not isinstance(sdfg.arrays[x.data], dt.View)
        ]

        # Merge second state to first state
        # First keep a backup of the topological sorted order of the nodes
        sdict = first_state.scope_dict()
        order = [
            x for x in reversed(list(nx.topological_sort(first_state._nx)))
            if isinstance(x, nodes.AccessNode) and sdict[x] is None
        ]
        for node in second_state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                # update parent information
                node.sdfg.parent = first_state

            #The node could have been added when adding connections by add_nedge hence the need to check
            if node not in first_state.nodes():
                first_state.add_node(node)

            # Wire the happens-before edges ``can_be_applied`` recorded for the false
            # dependencies -- WAW (first-write -> second-write) and WAR (first-read ->
            # second-write). The edge lands on the second-state SOURCE ``i`` that leads to
            # the hazard node, so the first-state endpoint is ordered before the whole
            # second-state chain producing / overwriting the datum. (RAW, a TRUE data
            # dependency, is handled by the common-data-node merge below, not here.)
            for _kind, first_nodes, second_nodes in self.connections_to_make:
                if node in second_nodes:
                    for i in top2:
                        if i not in second_state.source_nodes():
                            continue
                        # Plain reachability. (Do NOT use ``all_nodes_between``: it returns an
                        # EMPTY SET -- not None -- as soon as its DFS hits any sink that is not
                        # ``node``, so a second-state source that fans out to another chain made
                        # the guard False and the ordering edge was silently never wired. That
                        # is the most common topology there is, which left both WAR and WAW
                        # connections unhonoured.)
                        if i is not node and not nx.has_path(second_state._nx, i, node):
                            continue
                        for j in first_nodes:
                            if j in first_state.nodes():
                                first_state.add_nedge(j, i, memlet.Memlet())
        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        top = top_level_nodes(first_state)

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
                # WAW guard: refuse the same-name merge when ``can_be_applied``
                # flagged a write-write hazard for this (cand, node) data via
                # ``connections_to_make`` (cross-CC sibling writes that need
                # happens-before ordering, not last-writer-wins collapse). The
                # empty memlet added above wires the ordering; leaving both
                # AccessNodes distinct is what preserves correctness when the two
                # chains write different source data to the same shared scalar.
                # Pinned by compound_nest's per-iteration ``arr_index`` (TSVC-
                # shape sibling tasklets where both writers feed disjoint ``arr``
                # subsets).
                node_waw_flagged = any(node in second_nodes
                                       for kind, first_nodes, second_nodes in self.connections_to_make if kind == 'waw')
                for cand in candidates:
                    if node_waw_flagged and first_state.in_degree(cand) > 0:
                        continue
                    if StateFusionExtended.memlets_intersect(first_state, [cand], False, second_state, [node], True):
                        if nx.has_path(first_state._nx, cand, node):  # Do not create cycles
                            continue
                        sdutil.change_edge_src(first_state, cand, node)
                        sdutil.change_edge_dest(first_state, cand, node)
                        first_state.remove_node(cand)
                        # Record the merged-away node so a later ``second_mid`` node with the same
                        # data does not re-select it: ``candidates`` (above) filters on the stale
                        # ``order`` / ``top`` snapshots, which still contain this now-removed node.
                        # Without this, ``memlets_intersect`` re-queries ``first_state.in_edges(cand)``
                        # on a deleted node -> ``KeyError`` (cloudsc gpu_scc's many same-named
                        # ``__t0_split_*`` from SplitTasklets). Mirrors the ``merged_nodes.add(n)``
                        # in the general same-name-merge branch below.
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
        sdutil.change_edge_src(graph, second_state, first_state)
        graph.remove_node(second_state)
        if graph.start_block == second_state:
            graph.start_block = graph.node_id(first_state)

        # Technically unneeded, but better to keep track.
        self.connections_to_make.clear()

        # Post-apply structural check: never let a buggy merge leave an
        # invalid SDFG behind. Always run the cheap focused check; the
        # ``strict_validate`` knob additionally runs full ``sdfg.validate()``
        # for paranoid debug runs. Pinned by the s118-class regression where
        # the merger left an edge whose ``memlet.data`` referenced a node
        # that was no longer in the fused state's data flow.
        self._post_apply_check(first_state, sdfg)

    def _post_apply_check(self, fused_state: SDFGState, owner_sdfg) -> None:
        """Cheap structural validation focused on the failure modes the
        merger historically produced.

        For every edge in ``fused_state`` whose memlet carries a data
        reference, require that ``memlet.data`` match either ``src.data``
        or ``dst.data`` when those endpoints are ``AccessNode`` instances.
        Mirror the rule the SDFG validator at ``validation.py:750`` uses,
        but evaluated on this one fused state -- O(|edges|) instead of
        a full-SDFG walk.

        Tasklet/MapEntry/MapExit/NestedSDFG endpoints are skipped: the
        connector-name vs memlet-data alignment for those is enforced by
        higher-level passes (codegen, schedule inference) and is not the
        bug class this check is hunting.

        :raises InvalidSDFGEdgeError: when an offending edge is found.
        """
        ssdfg = fused_state.sdfg
        state_id = ssdfg.node_id(fused_state) if fused_state in ssdfg.nodes() else 0

        # Acyclicity net: the WAR/WAW happens-before edges all point first-state ->
        # second-state and RAW merges are guarded against creating a path cycle, so the
        # fused state must stay a DAG. A cyclic state is unschedulable; fail loud here
        # rather than emit a silently reordered kernel. Cheap: one DAG test.
        if not nx.is_directed_acyclic_graph(fused_state._nx):
            raise InvalidSDFGEdgeError(
                'StateFusionExtended produced a cyclic state: the RAW/WAR/WAW ordering edges '
                'cannot be linearized (unsolvable dependency combination)', ssdfg, state_id, None)

        for eid, e in enumerate(fused_state.edges()):
            if e.data is None or e.data.is_empty() or e.data.data is None:
                continue
            name = e.data.data
            # Resolve the memlet-PATH endpoints, exactly as the authoritative
            # validator does (validation.py:715-718): a write-through edge
            # ``inner_producer -> MapExit`` has ``memlet.data`` naming the OUTER
            # array at the end of the path, not the immediate scratch source.
            # Checking the immediate ``e.src``/``e.dst`` here false-flagged those
            # valid edges (MapFusionVertical's shared-intermediate write-through
            # on nbody / vadv).
            try:
                path = fused_state.memlet_path(e)
                path_src, path_dst = path[0].src, path[-1].dst
            except Exception:
                path_src, path_dst = e.src, e.dst
            src_an = path_src if isinstance(path_src, nodes.AccessNode) else None
            dst_an = path_dst if isinstance(path_dst, nodes.AccessNode) else None
            # Structures: memlet.data is the structure's root, member access
            # via connector. The full SDFG validator special-cases this; we
            # skip rather than risk a false positive.
            for an in (src_an, dst_an):
                if an is not None and isinstance(ssdfg.arrays.get(an.data), dt.Structure):
                    break
            else:
                if src_an is None and dst_an is None:
                    continue
                # Mirror the SDFG validator's edge rule (validation.py:750):
                # memlet.data must match SRC's AccessNode data/conn, or
                # DST's. A copy between two AccessNodes ``A[i] -> B[0]`` is
                # recorded with ``memlet.data == 'A'`` (one side); the other
                # side lives in ``other_subset``. So the check is a logical
                # OR across the two endpoints, not an AND.
                src_match = (src_an is not None and (name == src_an.data or name == e.src_conn))
                dst_match = (dst_an is not None and (name == dst_an.data or name == e.dst_conn))
                if not (src_match or dst_match):
                    raise InvalidSDFGEdgeError(
                        f"StateFusionExtended produced an invalid edge: memlet.data={name!r} "
                        f"does not match src={getattr(src_an, 'data', None)!r} or "
                        f"dst={getattr(dst_an, 'data', None)!r}",
                        ssdfg,
                        state_id,
                        eid,
                    )

        if self.strict_validate:
            ssdfg.validate()
