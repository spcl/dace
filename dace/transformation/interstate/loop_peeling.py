# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop peeling transformation """

import ast
import copy
import sympy as sp
from typing import List, Optional, Union

from dace import sdfg as sd
from dace import symbolic
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, SDFGState
from dace.frontend.python.astutils import ASTFindReplace
from dace.properties import Property, make_properties, CodeBlock
from dace.symbolic import pystr_to_symbolic
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible


@make_properties
@explicit_cf_compatible
class LoopPeeling(LoopUnroll):
    """
    Splits the first `count` iterations of loop into multiple, separate control flow regions (one per iteration).
    """

    begin = Property(
        dtype=bool,
        default=True,
        desc='If True, peels loop from beginning (first `count` iterations), otherwise peels last `count` iterations.',
    )

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        return True

    def _modify_cond(self, condition, var, step):
        condition = pystr_to_symbolic(condition.as_string)
        itersym = pystr_to_symbolic(var)
        # Find condition by matching expressions
        end: Optional[sp.Expr] = None
        a = sp.Wild('a')
        op = ''
        match = condition.match(itersym < a)
        if match:
            op = '<'
            end = match[a] - self.count * step
        if end is None:
            match = condition.match(itersym <= a)
            if match:
                op = '<='
                end = match[a] - self.count * step
        if end is None:
            match = condition.match(itersym > a)
            if match:
                op = '>'
                end = match[a] - self.count * step
        if end is None:
            match = condition.match(itersym >= a)
            if match:
                op = '>='
                end = match[a] - self.count * step
        if len(op) == 0:
            raise ValueError('Cannot match loop condition for peeling')

        res = str(itersym) + op + str(end)
        return res

    def _instantiate_peeled_iteration(self, graph: ControlFlowRegion, value: symbolic.SymbolicType,
                                      label_suffix: Optional[str]) -> ControlFlowBlock:
        """Instantiate one peeled iteration of ``self.loop`` directly into
        ``graph``, choosing the FLATTEST representation that still preserves
        the body's control flow:

        - **Single-state body** -> clone the body's nodes/edges into a fresh
          ``SDFGState`` added directly to ``graph`` (no wrapping
          ``ControlFlowRegion``). Downstream ``StateFusionExtended`` then
          owns ordering between the peeled state and the next block via
          interstate edges, the same way it handles any other adjacent
          ``SDFGState`` pair.

        - **Multi-state body** -> clone the body's blocks into a fresh
          ``ControlFlowRegion`` in ``graph`` (the previous behavior). The
          body's internal interstate edges must move as a unit because
          flattening them into the parent would lose the body-local
          control flow.

        In both cases the loop variable is substituted by ``value`` (the
        concrete iteration value) on tasklets, memlets, interstate-edge
        conditions/assignments, and NestedSDFG ``symbol_mapping`` entries.
        Cloning uses ``copy.deepcopy`` on each node and an
        ``{old_node: new_node}`` map to remap edge endpoints, so symbolic
        references inside memlets are preserved exactly.

        :param graph: The parent control-flow region (the loop's container).
        :param value: The concrete iteration value for this peeled iter.
        :param label_suffix: A disambiguator appended to the peeled-iter
                             label (the iteration index when symbolic).
        :returns: The new ``SDFGState`` (single-state body) or
                  ``ControlFlowRegion`` (multi-state body) added to
                  ``graph``.
        """
        loop = self.loop
        loop_var = loop.loop_variable
        suffix = label_suffix if label_suffix is not None else str(value)
        it_label = f'{loop.label}_{loop_var}{suffix}'

        body_blocks = list(loop.nodes())
        single_state = (len(body_blocks) == 1 and isinstance(body_blocks[0], SDFGState))

        if single_state:
            src_state: SDFGState = body_blocks[0]
            new_state = graph.add_state(it_label)
            node_map = {}
            for n in src_state.nodes():
                nn = copy.deepcopy(n)
                node_map[n] = nn
                new_state.add_node(nn)
            for e in src_state.edges():
                new_state.add_edge(node_map[e.src], e.src_conn, node_map[e.dst], e.dst_conn, copy.deepcopy(e.data))
            self._replace_loop_var_in_block(new_state, loop_var, value)
            graph.reset_cfg_list()
            return new_state

        # Multi-state body: keep the CFR wrapping so internal interstate
        # edges live with the iter, but use the same deepcopy-and-remap
        # pattern as the single-state case.
        new_cfr = ControlFlowRegion(it_label, graph.sdfg, graph)
        graph.add_node(new_cfr)
        block_map = {}
        for b in body_blocks:
            nb = copy.deepcopy(b)
            block_map[b] = nb
            new_cfr.add_node(nb, is_start_block=(b is loop.start_block))
        for e in loop.edges():
            new_cfr.add_edge(block_map[e.src], block_map[e.dst], copy.deepcopy(e.data))
        for b in new_cfr.nodes():
            self._replace_loop_var_in_block(b, loop_var, value)
        for e, _parent in new_cfr.all_edges_recursive():
            if isinstance(e.data, sd.InterstateEdge):
                self._replace_loop_var_in_iedge(e.data, loop_var, value)
        graph.reset_cfg_list()
        return new_cfr

    @staticmethod
    def _replace_loop_var_in_block(block: ControlFlowBlock, loop_var: str, value: symbolic.SymbolicType) -> None:
        """Substitute ``loop_var`` for ``value`` inside ``block`` (state-level
        ``replace_dict`` covers tasklets/memlets/connectors, and walks any
        nested SDFGs' ``symbol_mapping``)."""
        block.replace_dict({loop_var: value})
        for n in block.all_nodes_recursive() if hasattr(block, 'all_nodes_recursive') else []:
            if isinstance(n, NestedSDFG):
                if loop_var in n.symbol_mapping:
                    n.symbol_mapping[loop_var] = ASTFindReplace({loop_var: str(value)
                                                                 }).visit(n.symbol_mapping[loop_var])
                if loop_var in n.symbol_mapping:
                    del n.symbol_mapping[loop_var]

    @staticmethod
    def _replace_loop_var_in_iedge(iedge: sd.InterstateEdge, loop_var: str, value: symbolic.SymbolicType) -> None:
        """Substitute ``loop_var`` for ``value`` on an interstate edge
        (condition + assignments)."""
        if not iedge.is_unconditional():
            ASTFindReplace({loop_var: str(value)}).visit(iedge.condition)
        new_assignments = {}
        for k, v in iedge.assignments.items():
            k_ast = ast.parse(k)
            v_ast = ast.parse(v)
            ASTFindReplace({loop_var: str(value)}).visit(k_ast)
            ASTFindReplace({loop_var: str(value)}).visit(v_ast)
            new_assignments[ast.unparse(k_ast)] = ast.unparse(v_ast)
        iedge.assignments = new_assignments

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # Obtain loop information
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        stride = loop_analysis.get_loop_stride(self.loop)
        is_symbolic = any([symbolic.issymbolic(r) for r in (start, end)])

        if self.begin:
            peeled_iterations: List[ControlFlowBlock] = []
            for i in range(self.count):
                current_index = start + (i * stride)
                is_symbolic |= symbolic.issymbolic(current_index)
                it = self._instantiate_peeled_iteration(graph, current_index, str(i) if is_symbolic else None)

                if len(peeled_iterations) > 0:
                    graph.add_edge(peeled_iterations[-1], it, sd.InterstateEdge())
                peeled_iterations.append(it)

            if peeled_iterations:
                for ie in graph.in_edges(self.loop):
                    graph.add_edge(ie.src, peeled_iterations[0], ie.data)
                    graph.remove_edge(ie)
                graph.add_edge(peeled_iterations[-1], self.loop, sd.InterstateEdge())

                new_start = symbolic.evaluate(start + (self.count * stride), sdfg.constants)
                self.loop.init_statement = CodeBlock(f'{self.loop.loop_variable} = {new_start}')
        else:
            peeled_iterations: List[ControlFlowBlock] = []
            for i in reversed(range(self.count)):
                # Anchor the iterate value on the loop end (never the loop
                # variable) so back-peeled iterations don't leak a loop-defined
                # symbol past the loop. ``i`` counts down, so
                # ``end - (count - 1 - i) * stride`` yields ascending values
                # that pick up where the shortened loop stops.
                current_index = end - (self.count - 1 - i) * stride
                is_symbolic |= symbolic.issymbolic(current_index)
                it = self._instantiate_peeled_iteration(graph, current_index, str(i) if is_symbolic else None)

                if len(peeled_iterations) > 0:
                    graph.add_edge(it, peeled_iterations[-1], sd.InterstateEdge())
                peeled_iterations.append(it)

            if peeled_iterations:
                for oe in graph.out_edges(self.loop):
                    graph.add_edge(peeled_iterations[0], oe.dst, oe.data)
                    graph.remove_edge(oe)
                graph.add_edge(self.loop, peeled_iterations[-1], sd.InterstateEdge())

                new_cond = CodeBlock(self._modify_cond(self.loop.loop_condition, self.loop.loop_variable, stride))
                self.loop.loop_condition = new_cond
