# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Eliminates loop overwriting data containers"""

from dace import sdfg as sd
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ConditionalBlock
from dace.transformation import transformation, helpers
from dace.transformation.passes.analysis import loop_analysis
from dace.sdfg.sdfg import InterstateEdge
from dace import symbolic


@transformation.explicit_cf_compatible
class LoopOverwriteElimination(transformation.MultiStateTransformation):
    """
    Eliminates loops which write to the same location in each iteration by replacing the loop with the last iteration.
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

        # Check if any continue, break, or return statements are present in the loop.
        if self.loop.has_break or self.loop.has_continue or self.loop.has_return:
            return False

        # Find symbols depending on the loop variable
        sym_deps = {}
        for edge in self.loop.all_interstate_edges():
            for k, v in edge.data.assignments.items():
                sym_expr = symbolic.pystr_to_symbolic(v)
                if k not in sym_deps:
                    sym_deps[k] = set()
                str_set = set([str(s) for s in sym_expr.free_symbols])
                sym_deps[k].update(str_set)

        itervar_dep_syms = set()
        changed = True
        while changed:
            changed = False
            for k, v in sym_deps.items():
                if k in itervar_dep_syms:
                    continue
                if self.loop.loop_variable in v or itervar_dep_syms.intersection(v):
                    itervar_dep_syms.add(k)
                    changed = True
        itervar_dep_syms.add(self.loop.loop_variable)

        # Every write needs to be independent of the loop index.
        for state in self.loop.all_states():
            for dn in state.data_nodes():
                for e in state.in_edges(dn):
                    # If pointers are involved or it's not an overwrite, give up
                    if e.data.dynamic or e.data.wcr is not None:
                        return False

                    dst_subset = e.data.get_dst_subset(e, state)
                    for rb, re, _ in dst_subset.ndrange():
                        str_set = set(
                            [str(s) for s in rb.free_symbols.union(re.free_symbols)]
                        )
                        if itervar_dep_syms.intersection(str_set):
                            return False

        # If an data container is written and read, the read cannot have the same index as the write.

        # No conditional edge may depend on the loop variable.
        for edge in self.loop.all_interstate_edges():
            if itervar_dep_syms.intersection(edge.data.condition.get_free_symbols()):
                return False

        # No conditional block may depend on the loop variable.
        for node in self.loop.nodes():
            if not isinstance(node, ConditionalBlock):
                continue
            for cond, _ in node.branches:
                if itervar_dep_syms.intersection(cond.get_free_symbols()):
                    return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)

        last_iteration = start + (end - start) // stride * stride
        itervar = self.loop.loop_variable

        # Rewrite each occurence of the loop variable in the loop body
        self.loop.replace(itervar, last_iteration)

        # Add the loop contents to the parent graph.
        graph.add_node(self.loop.start_block)
        for e in graph.in_edges(self.loop):
            graph.add_edge(e.src, self.loop.start_block, e.data)
        sink = graph.add_state(self.loop.label + "_sink")
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
