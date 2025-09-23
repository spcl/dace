# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from typing import Dict, Optional, Set
from sympy.printing.pycode import pycode

from dace import SDFG
from dace import properties
from dace import Union
from dace import ControlFlowRegion
from dace.properties import Property
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation

from dace.sdfg.nodes import CodeBlock
import ast
import re


@properties.make_properties
@transformation.explicit_cf_compatible
class OffsetLoopsAndMaps(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    offset_expr = Property(dtype=dace.symbolic.SymExpr, default=dace.symbolic.SymExpr(0))
    begin_expr = Property(dtype=dace.symbolic.SymExpr, default=dace.symbolic.SymExpr(0))
    # To avoid Invalid type "int" for property offset_expr: expected SymExpr
    int_offset_expr = Property(dtype=int, default=0)
    int_begin_expr = Property(dtype=int, default=0)
    check_begin = Property(dtype=bool, default=False)

    def __init__(self, offset_expr: Union[dace.symbolic.SymExpr, dace.symbolic.symbol, int],
                 begin_expr: Union[dace.symbolic.SymExpr, dace.symbolic.symbol, int]):

        def _cast(expr):
            if isinstance(expr, int):
                return dace.symbolic.SymExpr(expr)
            elif isinstance(expr, dace.symbolic.symbol):
                return dace.symbolic.SymExpr(str(expr))
            else:
                if not isinstance(expr, dace.symbolic.SymExpr):
                    raise ValueError(f"Input {expr} to OffsetLoopsAndMaps is not a SymExpr | Symbol | Integer")
                else:
                    return expr

        if begin_expr is not None:
            begin_expr = _cast(begin_expr)
            self.check_begin = True
            if isinstance(begin_expr, int):
                self.int_begin_expr = begin_expr
            else:
                self.begin_expr = begin_expr
        else:
            self.check_begin = False

        offset_expr = _cast(offset_expr)

        # To avoid Invalid type "int" for property offset_expr: expected SymExpr
        if isinstance(offset_expr, int):
            self.int_offset_expr = offset_expr
        else:
            self.offset_expr = offset_expr

    def _get_offset_expr(self):
        if self.int_offset_expr != 0:
            return self.int_offset_expr
        else:
            return pycode(self.offset_expr)

    def _get_begin_expr(self):
        if self.check_begin is False:
            return None
        else:
            if self.int_begin_expr != 0:
                return self.int_begin_expr
            else:
                return pycode(self.begin_expr)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.CFG | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _create_new_memlet(self, edge_data, repldict: Dict[str, str]) -> dace.memlet.Memlet:
        """Create a new memlet with substituted subset ranges."""
        if edge_data is None:
            return None

        new_range_list = [(b.subs(repldict), e.subs(repldict), s.subs(repldict)) for b, e, s in edge_data.subset]
        new_range = dace.subsets.Range(new_range_list)
        return dace.memlet.Memlet(data=edge_data.data, subset=new_range)

    def _update_edge_if_changed(self, state, edge, new_memlet):
        """Update edge if the new memlet is different from the current one."""
        if new_memlet and new_memlet != edge.data:
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _process_memlets_in_edges(self, state, edges, repldict: Dict[str, str]):
        """Process memlets in a collection of edges."""
        for edge in edges:
            new_memlet = self._create_new_memlet(edge.data, repldict)
            self._update_edge_if_changed(state, edge, new_memlet)

    def _process_nested_sdfgs(self, state, repldict: Dict[str, str]):
        """Recursively process nested SDFGs in state nodes."""
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                self._repl_memlets_recursive(node.sdfg, repldict)

    def _repl_memlets_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]):
        """Recursively replace memlets in all states of a control flow region."""
        for state in cfg.all_states():
            self._process_memlets_in_edges(state, state.edges(), repldict)
            self._process_nested_sdfgs(state, repldict)

    def _repl_memlets_recursive_on_edge_list(self, state, edges, repldict: Dict[str, str]):
        """Replace memlets on a specific list of edges."""
        self._process_memlets_in_edges(state, edges, repldict)
        self._process_nested_sdfgs(state, repldict)

    def _repl_interstate_edges_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]):
        """Recursively replace interstate edges in control flow region."""
        # Replace interstate edges
        for edge in cfg.all_interstate_edges():
            edge.data.replace_dict(repldict)

        # Process nested SDFGs
        for state in cfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._repl_interstate_edges_recursive(node.sdfg, repldict)

    def _repl_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]):
        """Replace both interstate edges and memlets recursively."""
        self._repl_interstate_edges_recursive(cfg, repldict)
        self._repl_memlets_recursive(cfg, repldict)
        # TODO:
        # Implement for tasklets in case the loop variable is used as a symbol inside tasklet code

    def _add_to_rhs(self, expr: str, add_expr) -> str:
        """Add an expression to the right-hand side of a comparison."""
        try:
            tree = ast.parse(expr, mode="eval")
            comparison = tree.body

            if not isinstance(comparison, ast.Compare) or not comparison.comparators:
                raise ValueError("Expression must be a comparison with at least one comparator")

            # Modify the first comparator by adding the expression
            original_rhs = comparison.comparators[0]
            add_expr_ast = ast.parse(str(add_expr), mode="eval").body

            comparison.comparators[0] = ast.BinOp(left=original_rhs, op=ast.Add(), right=add_expr_ast)

            # Convert back to string and simplify
            str_expr = ast.unparse(tree)
            sym_expr = dace.symbolic.SymExpr(str_expr).simplify()
            return pycode(sym_expr)

        except Exception as e:
            raise ValueError(f"Failed to process expression '{expr}': {e}")

    def _apply(self, cfg: dace.ControlFlowRegion):
        for node in cfg.nodes():
            if isinstance(node, LoopRegion):
                node.init_statement
                node.loop_condition
                # The begin expression matches apply offset
                init_lhs, init_rhs = node.init_statement.as_string.split("=")
                if self._get_begin_expr() is None or dace.symbolic.SymExpr(init_rhs) == self._get_begin_expr():
                    new_init_statement = pycode(
                        dace.symbolic.SymExpr(f"(({init_rhs}) + {self._get_offset_expr()})").simplify())
                    node.init_statement = CodeBlock(f"{init_lhs} = {new_init_statement}")
                    new_loop_condition = self._add_to_rhs(node.loop_condition.as_string, self._get_offset_expr())
                    node.loop_condition = CodeBlock(new_loop_condition)

                    # Try normalize after update
                    node.normalize()

                    repldict = {node.loop_variable: f"({node.loop_variable} - {self._get_offset_expr()})"}
                    self._repl_recursive(node, repldict)
            elif isinstance(node, dace.SDFGState):
                state = node
                for state_node in state.nodes():
                    if isinstance(state_node, dace.nodes.MapEntry):
                        has_matches = False
                        new_range_list = []
                        repldict = dict()
                        for (b, e, s), param in zip(state_node.map.range, state_node.map.params):
                            if self._get_begin_expr() is None or b == self._get_begin_expr():
                                has_matches = True
                                new_range_list.append((b + self._get_offset_expr(), e + self._get_offset_expr(), s))
                                repldict[param] = f"({param} - {self._get_offset_expr()})"
                            else:
                                new_range_list.append((b, e, s))

                        if has_matches:
                            new_range = dace.subsets.Range(new_range_list)
                            state_node.map.range = new_range
                            nodes_between = state.all_nodes_between(state_node, state.exit_node(state_node))
                            edges_between = state.all_edges(*nodes_between)
                            self._repl_memlets_recursive_on_edge_list(state, edges_between, repldict)
                            # TODO:
                            # Implement for tasklets in case the loop variable is used as a symbol inside tasklet code

        for node in cfg.nodes():
            if isinstance(node, ControlFlowRegion):
                self._apply(node)
            if isinstance(node, ConditionalBlock):
                for _, body in node.branches:
                    self._apply(body)

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        # Do it for LoopRegions and Maps
        self._apply(sdfg)
