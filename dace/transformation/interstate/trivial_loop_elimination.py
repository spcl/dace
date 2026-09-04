# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Eliminates trivial loop """

from dace import sdfg as sd, symbolic
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import helpers, transformation
from dace.transformation.passes.analysis import loop_analysis


@transformation.explicit_cf_compatible
class TrivialLoopElimination(transformation.MultiStateTransformation):
    """
    Eliminates loops that provably run at most once: a single-iteration loop is spliced into its
    parent with the iterator substituted, a ZERO-trip loop is deleted outright (its body must never
    run -- splicing it in would fabricate an iteration, polybench nussinov's ``for j = N; j < N``
    writing ``table[N-1, N]`` one column past the end).
    """

    loop = transformation.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def is_zero_trip(self) -> bool:
        """Provably zero iterations: the first condition check already fails. Undecidable is False
        (the sound direction -- an undecidable loop is neither spliced nor deleted)."""
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)
        if start is None or end is None or stride is None:
            return False
        try:
            if stride > 0:
                return bool(start > end)
            return bool(start < end)
        except Exception:
            return False

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Check if this is a for-loop with known range.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)
        if start is None or end is None or stride is None:
            return False

        # The loop must provably run AT MOST once. ``get_loop_end`` returns the INCLUSIVE last
        # iteration value, so that means no second iteration; whether the single candidate
        # iteration happens at all (splice) or not (delete) is :meth:`is_zero_trip`'s call in
        # ``apply``. An undecidable comparison lands in the ``except`` below and refuses.
        try:
            # No second iteration: reject a loop that runs two or more times.
            if stride > 0 and start + stride < end + 1:
                return False
            if stride < 0 and start + stride > end - 1:
                return False
            # Decidability gate: whether the single candidate iteration runs at all must be
            # provable -- True means zero-trip (``apply`` DELETES the loop), False means exactly
            # one iteration (``apply`` splices it). Undecidable raises into the ``except`` and
            # refuses: such a loop could be neither spliced (might be zero-trip) nor deleted
            # (might run).
            bool(start > end) if stride > 0 else bool(start < end)
        except:
            # if the relation can't be determined it's not a trivial loop
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # Obtain iteration variable, range and stride
        itervar = self.loop.loop_variable
        start = loop_analysis.get_init_assignment(self.loop)

        if self.is_zero_trip():
            # The body never runs: delete the loop outright. What remains of the region is its
            # exit binding -- a for-loop that fails its first condition check still executed the
            # init, so downstream reads of the iterator see ``start``. Bound on its own edge so the
            # init expression reads pre-edge values, exactly as the loop would have evaluated it.
            head = graph.add_state(self.loop.label + '_zero_trip', is_start_block=graph.start_block is self.loop)
            tail = graph.add_state(self.loop.label + '_zero_trip_exit')
            for e in graph.in_edges(self.loop):
                graph.add_edge(e.src, head, e.data)
            graph.add_edge(head, tail, InterstateEdge(assignments={itervar: str(start)}))
            for e in graph.out_edges(self.loop):
                graph.add_edge(tail, e.dst, e.data)
            graph.remove_node(self.loop)
            return

        # ``replace`` (``ControlGraphView.replace``, state.py) hand-walks ``nodes()`` / ``edges()`` and
        # never routes through ``replace_dict``, so it silently misses a ``ConditionalBlock``'s branch
        # CONDITIONS (they live in ``_branches``, not ``nodes()``) and a nested ``LoopRegion``'s own
        # init / condition / update -- neither class overrides ``replace``. Those keep naming the
        # eliminated iterator: harmless while a peel/fission sibling still binds the name, but dangling
        # the instant ``UniqueLoopIterators`` renames it (``SDFG.arglist`` -> ``KeyError``, polybench
        # nussinov). ``replace_dict`` is the override-aware path. ``replace_keys=False`` leaves the
        # about-to-be-removed loop's own ``loop_variable`` alone; nested loops keep their own iterators
        # because those names are not in the replacement map.
        self.loop.replace_dict({itervar: str(start)}, symrepl={symbolic.symbol(itervar): start}, replace_keys=False)

        # Reparent the loop's blocks into the parent graph. A loop body is its own name scope, so a
        # label that was unique inside the loop can already be taken in the destination: sibling loops
        # cloned from one original (loop peeling's ``L_p0``/``L_p1``, LoopFission's ``L_fis0``/``L_fis1``)
        # each carry a body block of the SAME name, and eliminating the second one lands it next to the
        # first. Rename on arrival -- ``ensure_unique_name`` is what every other reparenting site uses
        # (``move_if_into_loop``, ``move_loop_invariant_if_up``, ``fuse_loops``) -- so the graph is never
        # left holding two blocks with one name. Every block is added explicitly here, up front: the
        # edge loop below would otherwise auto-add the non-start blocks (``OrderedDiGraph.add_edge``),
        # which bypasses the unique naming entirely. Edges are wired by object reference, so relabelling
        # is safe. ``start_block`` goes first to keep the parent's node order as it was.
        spliced = [self.loop.start_block] + [b for b in self.loop.nodes() if b is not self.loop.start_block]
        for block in spliced:
            graph.add_node(block, ensure_unique_name=True)
        for e in graph.in_edges(self.loop):
            graph.add_edge(e.src, self.loop.start_block, e.data)
        sink = graph.add_state(self.loop.label + '_sink')
        for n in self.loop.sink_nodes():
            graph.add_edge(n, sink, InterstateEdge())
        for e in graph.out_edges(self.loop):
            graph.add_edge(sink, e.dst, e.data)
        for e in self.loop.edges():
            graph.add_edge(e.src, e.dst, e.data)

        # Remove loop and if necessary also the loop variable.
        graph.remove_node(self.loop)
        if itervar in sdfg.symbols and helpers.is_symbol_unused(sdfg, itervar):
            sdfg.remove_symbol(itervar)

        # The substitution above can turn a body guard CONSTANT (a boundary split's single-trip
        # segment under ``if i < end``, spliced at ``i = end`` -- cloudsc), and the dead branch
        # then holds memlets that read as statically out of bounds. Resolve those trivial branches
        # now, scoped to the spliced blocks plus one level of the parent (for a conditional that
        # was spliced in directly) -- not a whole-graph pass.
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf  # avoid import loop
        lift = LiftTrivialIf()
        for block in spliced:
            if isinstance(block, ControlFlowRegion) and block.parent_graph is graph:
                lift.lift_in_region(block)
        lift.lift_in_region(graph, recursive=False)
