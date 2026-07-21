# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import ast
import copy
from typing import List, Optional, Union

from dace import dtypes, sdfg as sd, symbolic
from dace.properties import Property, make_properties
from dace.sdfg import InterstateEdge, utils as sdutil
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowRegion, LoopRegion, SDFGState
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


@make_properties
@xf.explicit_cf_compatible
class LoopUnroll(xf.MultiStateTransformation):
    """ Unrolls a for-loop into multiple individual control flow regions """

    loop = xf.PatternNode(LoopRegion)

    count = Property(
        dtype=int,
        default=0,
        desc='Number of iterations to unroll, or zero for all iterations (loop must be constant-sized for 0)',
    )

    inline_iterations = Property(dtype=bool,
                                 default=True,
                                 desc="Whether or not to inline individual iterations' CFGs after unrolling")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        # If loop stride is not specialized or constant-sized, fail
        if symbolic.issymbolic(step, sdfg.constants):
            return False
        # A zero stride never advances the iterate, so there is no finite unrolling to emit.
        if symbolic.evaluate(step, sdfg.constants) == 0:
            return False
        # If loop range diff is not constant-sized, fail
        if symbolic.issymbolic(end - start, sdfg.constants):
            return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg):
        # Loop must be fully unrollable for now.
        if self.count != 0:
            raise NotImplementedError  # TODO(later)

        # Obtain loop information
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)

        try:
            stride = symbolic.evaluate(stride, sdfg.constants)
            loop_diff = int(symbolic.evaluate(end - start, sdfg.constants))
        except TypeError:
            raise TypeError('Loop difference and strides cannot be symbolic.')

        # ``get_loop_end`` reports the INCLUSIVE last value of the iterate, so the iterate offsets are
        # ``0, stride, 2*stride, ...`` up to and including ``loop_diff``. A Python ``range`` takes an
        # EXCLUSIVE bound, which therefore has to sit one unit PAST ``loop_diff`` in the direction of
        # travel: above it while ascending, but BELOW it while descending. Hardcoding ``+1`` here left a
        # descending loop's bound two units short of where it belongs and silently truncated the last
        # ``ceil(2 / |stride|)`` iterations (2 of them at stride -1, 1 at stride -2). A loop that runs
        # zero times still yields an empty range under either sign: the bound then lies on the wrong
        # side of the start for the given stride.
        offsets = range(0, loop_diff + (1 if stride > 0 else -1), stride)

        # A start-block loop has no in-edges, so the unrolled chain would inherit none and leave the
        # parent with an ambiguous start. Prepend an empty ``pre -> loop`` start state so the loop is a
        # normal interior block; ``pre`` stays the single start and is absorbed by the following fusion.
        # Guard on the ITERATION COUNT, not on the sign of ``loop_diff``, which is negative for every
        # descending loop that does have iterations; a zero-trip loop instead gets its own start state
        # below.
        if len(offsets) > 0 and graph.start_block is self.loop:
            pre_state = graph.add_state(self.loop.label + '_unroll_pre', is_start_block=True)
            graph.add_edge(pre_state, self.loop, sd.InterstateEdge())

        # Create states for loop subgraph
        # A state is returned as a replacement when the loop body is empty
        unrolled_iterations: List[Union[ControlFlowRegion, SDFGState]] = []
        for position, i in enumerate(offsets):
            # Instantiate loop contents as a new control flow region with iterate value.
            # `position` (0, 1, 2, ...) is instantiate_loop_iteration's label-safety fallback,
            # NOT `i` itself: for a decrementing loop (negative stride) `i` walks 0, -1, -2, ...
            # and is just as unsafe to embed in a name as a symbolic iterate value would be.
            current_index = start + i
            iteration_region = self.instantiate_loop_iteration(graph, self.loop, current_index, position)
            iteration_region.replace_dict({self.loop.loop_variable: current_index}, replace_keys=True)
            iteration_region.replace_meta_accesses({self.loop.loop_variable: symbolic.symstr(current_index)})

            # Connect iterations with unconditional edges
            if len(unrolled_iterations) > 0:
                assert unrolled_iterations[-1] in graph.nodes()
                assert iteration_region in graph.nodes()
                graph.add_edge(unrolled_iterations[-1], iteration_region, sd.InterstateEdge())
            unrolled_iterations.append(iteration_region)

        if len(unrolled_iterations) == 0:
            s = graph.add_state(label="empty_unroll", is_start_block=True)
            unrolled_iterations.append(s)

        if unrolled_iterations:
            for ie in graph.in_edges(self.loop):
                assert ie.src in graph.nodes()
                assert unrolled_iterations[0] in graph.nodes()
                graph.add_edge(ie.src, unrolled_iterations[0], ie.data)
            for oe in graph.out_edges(self.loop):
                assert unrolled_iterations[-1] in graph.nodes()
                assert oe.dst in graph.nodes()
                graph.add_edge(unrolled_iterations[-1], oe.dst, oe.data)

        # The loop is never the start block here (a start-block loop got an empty ``pre`` predecessor
        # above), so removing it cannot orphan the parent's start -- no start-block re-designation needed.
        graph.remove_node(self.loop)

        if self.inline_iterations:
            for it in unrolled_iterations:
                # SDFGState does not have an inline attribute
                if isinstance(it, SDFGState):
                    continue
                it.inline()

    def instantiate_loop_iteration(self,
                                   graph: ControlFlowRegion,
                                   loop: LoopRegion,
                                   value: symbolic.SymbolicType,
                                   index: int,
                                   label_suffix: Optional[str] = None) -> ControlFlowRegion:
        it_label = loop.label + '_' + loop.loop_variable + (label_suffix
                                                            if label_suffix is not None else symbolic.symstr(value))
        if not dtypes.validate_name(it_label):
            # A concrete (non-symbolic) iterate value can still render into an invalid
            # identifier -- e.g. a negative int's ``str()`` contains a bare ``-``
            # (confirmed on a real CloudSC loop counting down: label
            # ``for_1260_jn-1`` rejected by validate_name). The enumeration index is
            # always a small non-negative int, always identifier-safe; fall back to it
            # rather than trying to enumerate every character ``str(value)`` could ever
            # produce.
            it_label = loop.label + '_' + loop.loop_variable + str(index)
        iteration_region = ControlFlowRegion(it_label, graph.sdfg, graph)

        # ``ensure_unique_name``: the label is derived from the loop label + iterate value, which is
        # NOT unique when several sibling loops share a label (e.g. many deepcopies of one inner loop
        # left by unrolling an enclosing loop). Without this, every such loop emits the same
        # ``<label>_<var><value>`` iteration regions into one parent -- a "multiple blocks with the
        # same name" validation failure. On inline the unique region label prefixes its promoted
        # children too, so the fix carries through both the inline and no-inline paths.
        graph.add_node(iteration_region, ensure_unique_name=True)

        block_map = {}

        for block in loop.nodes():
            # A deepcopy copies a block ~9x faster than a to/from JSON round-trip (measured on a
            # CloudSC loop body: 0.38ms vs 3.37ms). ``ControlFlowBlock.__deepcopy__`` detaches the
            # parent SDFG on the whole subtree, where the old JSON path carried it in via the
            # deserialization context. Restore the two pointers deepcopy drops, before
            # ``replace_dict`` dereferences them:
            #  * every control-flow block's ``.sdfg``. ``all_control_flow_blocks`` (non-recursive)
            #    reaches nested regions at any depth but stops at nested-SDFG boundaries, whose
            #    states must keep pointing at their own SDFG.
            #  * each DIRECT nested SDFG's ``parent_sdfg``: the block's own memo had no reference to
            #    the outer SDFG, so ``SDFG.__deepcopy__`` nulled it (``parent`` and
            #    ``parent_nsdfg_node`` survived via the memo). Deeper nested SDFGs, whose parents
            #    were inside the memo, keep their own SDFG. Mirrors ``SDFGState.add_node``.
            new_block = copy.deepcopy(block)
            cfg_blocks = [new_block]
            if isinstance(new_block, AbstractControlFlowRegion):
                cfg_blocks.extend(new_block.all_control_flow_blocks())
            for cfg_block in cfg_blocks:
                cfg_block.sdfg = graph.sdfg
                if isinstance(cfg_block, SDFGState):
                    for nsdfg_node in cfg_block.nodes():
                        if isinstance(nsdfg_node, NestedSDFG):
                            nsdfg_node.sdfg.parent_sdfg = graph.sdfg
            assert block not in block_map
            block_map[block] = new_block
            new_block.replace_dict({loop.loop_variable: value})
            iteration_region.add_node(new_block, is_start_block=(block is loop.start_block))

        for edge in loop.edges():
            src = block_map[edge.src]
            dst = block_map[edge.dst]
            # Replace conditions in subgraph edges
            data = copy.deepcopy(edge.data)
            iteration_region.add_edge(src, dst, data)

        # A raw str() on a symbolic expression can misrender it (e.g. an operator-function like
        # ``__right_shift`` prints as its sympy class name, not the operator) -- symstr renders
        # the DaCe-printable form, so the substituted code stays parseable either way.
        value_str = symbolic.symstr(value)

        # Replace occurences of the loop variables on all interstate edges
        for edge, parent_graph in iteration_region.all_edges_recursive():  # Recursion needed for nested SDFGs
            if isinstance(edge.data, InterstateEdge):
                src = edge.src
                dst = edge.dst
                assert src in parent_graph.nodes()
                assert dst in parent_graph.nodes()
                if not edge.data.is_unconditional():
                    ASTFindReplace({loop.loop_variable: value_str}).visit(edge.data.condition)

                new_assignments = dict()
                for k, v in edge.data.assignments.items():
                    k_ast = ast.parse(k)
                    v_ast = ast.parse(v)
                    ASTFindReplace({loop.loop_variable: value_str}).visit(k_ast)
                    ASTFindReplace({loop.loop_variable: value_str}).visit(v_ast)
                    new_assignments[ast.unparse(k_ast)] = ast.unparse(v_ast)
                edge.data.assignments = new_assignments

        for node in iteration_region.all_nodes_recursive():
            if isinstance(node, NestedSDFG):
                if loop.loop_variable in node.symbol_mapping:
                    node.symbol_mapping[loop.loop_variable] = ASTFindReplace({
                        loop.loop_variable: value_str
                    }).visit(node.symbol_mapping[loop.loop_variable])
                if loop.loop_variable in node.symbol_mapping:
                    del node.symbol_mapping[loop.loop_variable]

        graph.reset_cfg_list()
        return iteration_region
