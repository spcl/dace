# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Eliminates trivial loop """

from dace import sdfg as sd
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

        # Check if this is a trivial loop.
        try:
            if stride > 0 and start + stride < end + 1:
                return False
            if stride < 0 and start + stride > end - 1:
                return False
        except:
            # if the relation can't be determined it's not a trivial loop
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # Obtain iteration variable, range and stride
        itervar = self.loop.loop_variable
        start = loop_analysis.get_init_assignment(self.loop)

        self.loop.replace(itervar, start)

        # Add the loop contents to the parent graph.
        graph.add_node(self.loop.start_block)
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
