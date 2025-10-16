# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import dace
from typing import Dict, List, Optional, Set
import sympy
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


def _get_expr_from_str(expr: str) -> dace.symbolic.SymExpr:
    try:
        parsed_expr = sympy.sympify(expr, evaluate=False)
    except Exception as e:
        parsed_expr = dace.symbolic.SymExpr(expr)
    return parsed_expr


@properties.make_properties
@transformation.explicit_cf_compatible
class OffsetLoopsAndMaps(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    offset_expr = Property(dtype=str, default="0")
    begin_expr = Property(dtype=str, default="0")
    do_not_check_begin = Property(dtype=bool, default=False)
    convert_leq_to_lt = Property(dtype=bool, default=True)
    normalize_loops = Property(dtype=bool, default=False)

    def __init__(self,
                 offset_expr: str,
                 begin_expr: Union[str, None],
                 convert_leq_to_lt: bool = True,
                 normalize_loops: bool = False):
        self.offset_expr = offset_expr
        if begin_expr is None:
            self.do_not_check_begin = True
        else:
            self.begin_expr = begin_expr
        self.convert_leq_to_lt = convert_leq_to_lt
        self.normalize_loops = normalize_loops

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
        # Using symbols might create problems due to having different symbol objects with same symbols
        new_range_list = [(b.subs(repldict), e.subs(repldict), s.subs(repldict)) for b, e, s in edge_data.subset]
        new_range_str = ", ".join(f"{b}:{e+1}:{s}" for b, e, s in new_range_list)
        return dace.memlet.Memlet(expr=f"{edge_data.data}[{new_range_str}]")

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

    def _repl_tasklets_recursive(self, cfg: ControlFlowRegion, repldict):

        def _token_replace(code: str, src: str, dst: str) -> str:
            # Split while keeping delimiters
            tokens = re.split(r'(\s+|[()\[\]])', code)

            # Replace tokens that exactly match src
            tokens = [dst if token.strip() == src else token for token in tokens]

            # Recombine everything
            return ''.join(tokens).strip()

        for state in cfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    code = node.code
                    code_str = copy.deepcopy(node.code.as_string)
                    if code.language == dace.dtypes.Language.Python:
                        # Can raise exceptions if you have stuff like AND in the expression
                        try:
                            symexpr = dace.symbolic.SymExpr(code_str.split(" = ")[-1].strip())
                            symexpr = symexpr.subs(repldict)
                            code_str = code_str.split(" = ")[0].strip() + " = " + pycode(symexpr)
                        except Exception as e:
                            code_str = copy.deepcopy(node.code.as_string)
                            for k, v in repldict.items():
                                code_str = _token_replace(code_str, k, v)
                    else:
                        for k, v in repldict.items():
                            code_str = _token_replace(code_str, k, v)
                    node.code = CodeBlock(code_str, code.language)

                if isinstance(node, dace.nodes.NestedSDFG):
                    self._repl_interstate_edges_recursive(node.sdfg, repldict)

    def _repl_tasklets_recursive_from_node_list(self, state: dace.SDFGState, nodes: List[dace.nodes.Node], repldict):

        def _token_replace(code: str, src: str, dst: str) -> str:
            # Split while keeping delimiters
            tokens = re.split(r'(\s+|[()\[\]])', code)

            # Replace tokens that exactly match src
            tokens = [dst if token.strip() == src else token for token in tokens]

            # Recombine everything
            return ''.join(tokens).strip()

        for node in nodes:
            if isinstance(node, dace.nodes.Tasklet):
                code = node.code
                code_str = copy.deepcopy(node.code.as_string)
                if code.language == dace.dtypes.Language.Python:
                    # Can raise exceptions if you have stuff like AND in the expression
                    try:
                        symexpr = dace.symbolic.SymExpr(code_str.split(" = ")[-1].strip())
                        symexpr = symexpr.subs(repldict)
                        code_str = code_str.split(" = ")[0].strip() + " = " + pycode(symexpr)
                    except Exception as e:
                        code_str = copy.deepcopy(node.code.as_string)
                        for k, v in repldict.items():
                            code_str = _token_replace(code_str, k, v)
                else:
                    for k, v in repldict.items():
                        code_str = _token_replace(code_str, k, v)
                node.code = CodeBlock(code_str, code.language)

            if isinstance(node, dace.nodes.NestedSDFG):
                self._repl_interstate_edges_recursive(node.sdfg, repldict)

    def _repl_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]):
        """Replace both interstate edges and memlets recursively."""
        self._repl_interstate_edges_recursive(cfg, repldict)
        self._repl_memlets_recursive(cfg, repldict)
        # TODO:
        # Implement for tasklets in case the loop variable is used as a symbol inside tasklet code
        self._repl_tasklets_recursive(cfg, repldict)

    def _add_to_rhs(self, expr: str, add_expr: dace.symbolic.SymExpr, sdfg: dace.SDFG) -> str:
        """Add an expression to the right-hand side of a comparison."""
        #try:
        tree = ast.parse(expr, mode="eval")
        comparison = tree.body

        if not isinstance(comparison, ast.Compare) or not comparison.comparators:
            raise ValueError("Expression must be a comparison with at least one comparator")

        # Modify the first comparator by adding the expression
        original_rhs = comparison.comparators[0]
        add_expr_ast = ast.parse(pycode(add_expr), mode="eval").body

        comparison.comparators[0] = ast.BinOp(left=original_rhs, op=ast.Add(), right=add_expr_ast)

        # Convert back to string and simplify
        str_expr = ast.unparse(tree)
        sym_expr = _get_expr_from_str(str_expr)
        return pycode(sym_expr)

        #except Exception as e:
        #    raise ValueError(f"Failed to process expression '{expr}': {e}")

    def _apply(self, cfg: dace.ControlFlowRegion):
        for node in cfg.nodes():
            if isinstance(node, LoopRegion):
                node.init_statement
                node.loop_condition
                # The begin expression matches apply offset
                init_lhs, init_rhs = node.init_statement.as_string.split("=")
                if self.do_not_check_begin or _get_expr_from_str(init_rhs) == _get_expr_from_str(
                        self.begin_expr) or str(init_rhs) == str(self.begin_expr):
                    init_expr_str = f"(({init_rhs}) + {self.offset_expr})"
                    init_expr = _get_expr_from_str(init_expr_str)
                    new_init_statement = pycode(init_expr)
                    node.init_statement = CodeBlock(f"{init_lhs} = {new_init_statement}")
                    new_loop_condition = self._add_to_rhs(node.loop_condition.as_string,
                                                          _get_expr_from_str(self.offset_expr), cfg.sdfg)
                    node.loop_condition = CodeBlock(new_loop_condition)

                    # Try normalize after update
                    if self.normalize_loops:
                        node.normalize()

                    v = f"({node.loop_variable} - {_get_expr_from_str(self.offset_expr)})"
                    if "- -" in v:
                        v = v.replace("- -", "+ ")
                    repldict = {node.loop_variable: v}

                    self._repl_recursive(node, repldict)
            elif isinstance(node, dace.SDFGState):
                state = node
                for state_node in state.nodes():
                    if isinstance(state_node, dace.nodes.MapEntry):
                        has_matches = False
                        new_range_list = []
                        repldict = dict()
                        for (b, e, s), param in zip(state_node.map.range, state_node.map.params):
                            if self.do_not_check_begin is None or b == _get_expr_from_str(
                                    self.begin_expr) or str(b) == str(self.begin_expr):
                                has_matches = True
                                new_range_list.append((b + _get_expr_from_str(self.offset_expr),
                                                       e + _get_expr_from_str(self.offset_expr), s))
                                repldict[param] = f"({param} - {pycode(_get_expr_from_str(self.offset_expr))})"
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
                            self._repl_tasklets_recursive_from_node_list(state, nodes_between, repldict)

        for node in cfg.nodes():
            if isinstance(node, ControlFlowRegion):
                self._apply(node)
            if isinstance(node, ConditionalBlock):
                for _, body in node.branches:
                    self._apply(body)

    def _split_expr_str_opt_rhs(self, expr_str: str, op_to_split: str) -> str:
        exprs = expr_str.split(op_to_split)
        if len(exprs) != 2:
            return expr_str
        lhs, rhs = exprs[0], exprs[1]
        lhs = lhs.strip()
        rhs = rhs.strip()

        # Fix brackets
        opens = lhs.count("(")
        exits = lhs.count(")")
        rhs = "(" * (opens - exits) + rhs
        expr_str = lhs + op_to_split + sympy.pycode(dace.symbolic.SymExpr(rhs).simplify()) + (")" * (opens - exits))
        return expr_str

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        # Do it for LoopRegions and Maps
        self._apply(sdfg)
        sdfg.validate()

        # Simplify <= loop conditions to use < if set
        if self.convert_leq_to_lt:
            for n, g in sdfg.all_nodes_recursive():
                if isinstance(n, LoopRegion) and n.loop_condition.language == dace.dtypes.Language.Python:
                    expr = dace.symbolic.SymExpr(n.loop_condition.as_string)
                    if isinstance(expr, sympy.core.relational.Relational) and isinstance(expr, sympy.LessThan):
                        lhs, rhs = expr.lhs, expr.rhs
                        n.loop_condition = CodeBlock(sympy.pycode(sympy.StrictLessThan(lhs, rhs + 1)))

                    # Simplify only the rhs do this by splitting the expression from "<" and ( with the number of opened ( from left
                    # Then simplify it and add back
                    expr_str = self._split_expr_str_opt_rhs(n.loop_condition.as_string, " < ")
                    n.loop_condition = CodeBlock(expr_str)
        sdfg.validate()

        # Try to simplify loop init statements, expressions such as ((-1) + 1)
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, LoopRegion) and n.init_statement.language == dace.dtypes.Language.Python:
                try:
                    expr_str = self._split_expr_str_opt_rhs(n.init_statement.as_string, " = ")
                except Exception as e:
                    print(str(e))
                    expr_str = n.init_statement.as_string
                n.init_statement = CodeBlock(expr_str)
        sdfg.validate()

        return None
