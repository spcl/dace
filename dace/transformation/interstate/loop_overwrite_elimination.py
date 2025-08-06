# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Eliminates loop overwriting data containers"""

from dace import sdfg as sd
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ConditionalBlock, SDFGState
from dace.transformation import transformation, helpers
from dace.transformation.passes.analysis import loop_analysis
from dace.sdfg.sdfg import InterstateEdge
from dace import symbolic, nodes
from dace.subsets import intersects
import copy


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

        # Cached lists
        cfbs = set(self.loop.all_control_flow_blocks())
        states = set([s for s in cfbs if isinstance(s, SDFGState)])
        interstate_edges = set(e for s in cfbs for e in s.parent_graph.in_edges(s) + s.parent_graph.out_edges(s))

        # Find symbols depending on the loop variable
        sym_deps = {}
        for edge in interstate_edges:
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

        # Find Loop's unique data
        read_set, write_set = self.loop.read_and_write_sets()
        unique_set = set()
        for name in read_set | write_set:
            if not sdfg.arrays[name].transient:
                continue
            found = False
            for state in sdfg.states():
                if state in states:
                    continue
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode) and node.data == name:
                        found = True
                        break
            if not found:
                unique_set.add(name)

        # All the uniuque data needs to be written and read in the same index, otherwise there might be a loop-carried dependency.
        for state in states:
            for dn in state.data_nodes():
                if dn.data not in unique_set:
                    continue
                read_subsets = set()
                write_subsets = set()
                for e in state.out_edges(dn):
                    # If pointers are involved or it's not an overwrite, give up
                    if e.data.dynamic or e.data.wcr is not None:
                        return False
                    read_subsets.add(e.data.get_src_subset(e, state))
                for e in state.in_edges(dn):
                    # If pointers are involved or it's not an overwrite, give up
                    if e.data.dynamic or e.data.wcr is not None:
                        return False
                    write_subsets.add(e.data.get_dst_subset(e, state))
                if any(not intersects(rs, ws) for rs in read_subsets for ws in write_subsets):
                    return False

        # Every write needs to be independent of the loop index.
        write_subsets = {}
        for state in states:
            for dn in state.data_nodes():
                if dn.data in unique_set:
                    continue
                for e in state.in_edges(dn):
                    # If pointers are involved or it's not an overwrite, give up
                    if e.data.dynamic or e.data.wcr is not None:
                        return False

                    dst_subset = e.data.get_dst_subset(e, state)
                    for rb, re, _ in dst_subset.ndrange():
                        str_set = set([str(s) for s in rb.free_symbols.union(re.free_symbols)])
                        if itervar_dep_syms.intersection(str_set):
                            return False

                    if dn.data not in write_subsets:
                        write_subsets[dn.data] = set()
                    write_subsets[dn.data].add(dst_subset)

        # If an data container is written and read, the last read cannot be the same index as the write, because there is a loop-carried dependency then.
        for state in states:
            for dn in state.data_nodes():
                if dn.data in unique_set or dn.data not in write_subsets:
                    continue
                for e in state.out_edges(dn):
                    # If pointers are involved or it's not an overwrite, give up
                    if e.data.dynamic or e.data.wcr is not None:
                        return False

                    src_subset = copy.deepcopy(e.data.get_src_subset(e, state))
                    src_subset.replace({self.loop.loop_variable: end})
                    # None of write_subsets should lie within the new subset
                    if any(intersects(ws_ss, src_subset) for ws_ss in write_subsets[dn.data]):
                        return False

        # No conditional edge may depend on the loop variable.
        for edge in interstate_edges:
            if itervar_dep_syms.intersection(edge.data.condition.get_free_symbols()):
                return False

        # No conditional block or loop may depend on the loop variable.
        for cfb in cfbs:
            if isinstance(cfb, ConditionalBlock):
                for cond, _ in cfb.branches:
                    if itervar_dep_syms.intersection(cond.get_free_symbols()):
                        return False
            elif isinstance(cfb, LoopRegion):
                if (itervar_dep_syms.intersection(cfb.init_statement.get_free_symbols())
                        or itervar_dep_syms.intersection(cfb.loop_condition.get_free_symbols())
                        or itervar_dep_syms.intersection(cfb.update_statement.get_free_symbols())):
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
