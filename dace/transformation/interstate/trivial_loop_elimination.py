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
    Eliminates loops with a single loop iteration.
    """

    loop = transformation.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Check if this is a for-loop with known range.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)
        if start is None or end is None or stride is None:
            return False

        # Check if this is a trivial loop: it must run EXACTLY once. ``get_loop_end``
        # returns the INCLUSIVE last iteration value, so a single-iteration loop is one
        # whose second iteration is past the end AND whose first iteration happens at all.
        try:
            # No second iteration: reject a loop that runs two or more times.
            if stride > 0 and start + stride < end + 1:
                return False
            if stride < 0 and start + stride > end - 1:
                return False
            # A first iteration: reject a ZERO-trip loop, whose body must never run.
            # Inlining it would fabricate an iteration the loop never executes -- polybench
            # nussinov's ``for j in range(i+1, N)`` degenerates to ``for j = N; j < N`` at the
            # peeled ``i = N-1``, and splicing that body in writes ``table[N-1, N]``, one column
            # past the end of ``table[N, N]``. An undecidable comparison lands in the ``except``
            # below and refuses, which is the sound direction.
            if stride > 0 and start > end:
                return False
            if stride < 0 and start < end:
                return False
        except:
            # if the relation can't be determined it's not a trivial loop
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # Obtain iteration variable, range and stride
        itervar = self.loop.loop_variable
        start = loop_analysis.get_init_assignment(self.loop)

        # ``replace`` (``ControlGraphView.replace``, state.py) hand-walks ``nodes()`` / ``edges()`` and
        # never routes through ``replace_dict``, so it silently misses a ``ConditionalBlock``'s branch
        # CONDITIONS (they live in ``_branches``, not ``nodes()``) and a nested ``LoopRegion``'s own
        # init / condition / update -- neither class overrides ``replace``. Those keep naming the
        # eliminated iterator: harmless while a peel/fission sibling still binds the name, but dangling
        # the instant ``UniqueLoopIterators`` renames it (``SDFG.arglist`` -> ``KeyError``, polybench
        # nussinov). ``replace_dict`` is the override-aware path. ``replace_keys=False`` leaves the
        # about-to-be-removed loop's own ``loop_variable`` alone; nested loops keep their own iterators
        # because those names are not in the replacement map.
        self.loop.replace_dict({itervar: str(start)},
                               symrepl={symbolic.symbol(itervar): start},
                               replace_keys=False)

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
        for block in [self.loop.start_block] + [b for b in self.loop.nodes() if b is not self.loop.start_block]:
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
