# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Dict, Optional, Set
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
import dace.sdfg.utils as sdutil
from sympy import re, symbols, simplify
from sympy.core.relational import Relational
import re


def _safe_simplify(expr: dace.symbolic.SymExpr) -> dace.symbolic.SymExpr:
    # If it’s a relational (<, >, <=, >=, ==, !=)
    if isinstance(expr, Relational):
        lhs = simplify(expr.lhs)
        rhs = simplify(expr.rhs)
        # Recreate the relation with the same operator
        return expr.func(lhs, rhs)
    else:
        return simplify(expr)


def _get_expr_from_str(expr: str) -> dace.symbolic.SymExpr:
    try:
        parsed_expr = dace.symbolic.SymExpr(expr)
    except Exception as e:
        raise Exception(f"Parsing expression ({expr}) failed with error: {e}")
    return parsed_expr


def _split_and_repl(rhs: str, old_var: str, new_expr: str):
    tokens = re.split(r'([() ])', rhs)
    tokens = [t for t in tokens if t.strip() != '']
    ntokens = [t if t != old_var else new_expr for t in tokens]
    new_code = " ".join(ntokens)
    return new_code


def _token_based_repl_in_tasklet(tasklet: dace.nodes.Tasklet, old_var: str, new_expr: str) -> None:
    assert len(tasklet.code.as_string.split(
        "=")) == 2, f"Multiple assignments in a tasklet is not supported by loop-offsetting transformation currently"
    lhs, rhs = tasklet.code.as_string.split("=")
    if tasklet.language == dace.dtypes.Language.Python:
        # Try so simplify
        stripped_rhs = rhs.strip()
        # If tasklet has something liek dace.float64(5) this will fail
        try:
            sym_expr = dace.symbolic.SymExpr(stripped_rhs).subs({old_var: new_expr})
            simplified_expr = _safe_simplify(sym_expr)
            py_code = pycode(simplified_expr)
            tasklet.code = CodeBlock(lhs.strip() + " = " + py_code, tasklet.language)
        except Exception as e:
            new_code = _split_and_repl(rhs, old_var, new_expr)
            tasklet.code = CodeBlock(lhs.strip() + " = " + new_code, tasklet.language)
    else:
        new_code = _split_and_repl(rhs, old_var, new_expr)
        tasklet.code = CodeBlock(lhs.strip() + " = " + new_code, tasklet.language)


def safe_subs(expr, repldict):
    if hasattr(expr, "subs"):
        return expr.subs(repldict)
    else:
        return expr


@properties.make_properties
@transformation.explicit_cf_compatible
class OffsetLoopsAndMaps(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    offset_expr = Property(dtype=str, default="0", allow_none=False)
    begin_expr = Property(dtype=str, default=None, allow_none=True)

    def __init__(self, offset_expr: str, begin_expr: Union[str, None]):
        self.offset_expr = offset_expr
        self.begin_expr = begin_expr

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
        new_range_list = [(safe_subs(b, repldict), safe_subs(e, repldict), safe_subs(s, repldict))
                          for b, e, s in edge_data.subset]
        new_range_str = ", ".join(f"{b}:{e+1}:{s}" for b, e, s in new_range_list)
        if edge_data.other_subset is not None:
            new_other_range_list = [(safe_subs(b, repldict), safe_subs(e, repldict), safe_subs(s, repldict))
                                    for b, e, s in edge_data.other_subset]
            new_other_range_str = ", ".join(f"{b}:{e+1}:{s}" for b, e, s in new_other_range_list)
            return dace.memlet.Memlet(expr=f"{edge_data.data}[{new_range_str}] -> [{new_other_range_str}]")
        else:
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

    def _repl_use_in_tasklets_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]):
        for state in cfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    for p, repl_expr_str in repldict.items():
                        _token_based_repl_in_tasklet(node, p, repl_expr_str)
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._repl_use_in_tasklets_recursive(node.sdfg, repldict)

    def _repl_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]):
        """Replace both interstate edges and memlets recursively."""
        self._repl_interstate_edges_recursive(cfg, repldict)
        self._repl_memlets_recursive(cfg, repldict)
        self._repl_use_in_tasklets_recursive(cfg, repldict)

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
        # Fix some sign and binary op clashes
        str_expr = str_expr.replace("+ -", "- ").replace("- -", "+ ")
        sym_expr = _safe_simplify(_get_expr_from_str(str_expr))
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
                if self.begin_expr is None or _get_expr_from_str(init_rhs) == _get_expr_from_str(self.begin_expr):
                    init_expr_str = f"(({init_rhs}) + {self.offset_expr})"
                    init_expr = _get_expr_from_str(init_expr_str).simplify()
                    new_init_statement = pycode(init_expr)
                    node.init_statement = CodeBlock(f"{init_lhs} = {new_init_statement}")
                    new_loop_condition = self._add_to_rhs(node.loop_condition.as_string,
                                                          _get_expr_from_str(self.offset_expr), cfg.sdfg)
                    node.loop_condition = CodeBlock(new_loop_condition)

                    repldict = {node.loop_variable: f"({node.loop_variable} - {_get_expr_from_str(self.offset_expr)})"}
                    self._repl_recursive(node, repldict)
            elif isinstance(node, dace.SDFGState):
                state = node
                for state_node in state.nodes():
                    if isinstance(state_node, dace.nodes.MapEntry):
                        has_matches = False
                        new_range_list = []
                        repldict = dict()
                        for (b, e, s), param in zip(state_node.map.range, state_node.map.params):
                            if self.begin_expr is None or str(b) == self.begin_expr:
                                if str(b) == self.begin_expr:
                                    assert b == _get_expr_from_str(self.begin_expr)
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
                            # For tasklets in case the loop variable is used as a symbol inside tasklet code
                            for node in nodes_between:
                                if isinstance(node, dace.nodes.Tasklet):
                                    for p, repl_expr_str in repldict.items():
                                        _token_based_repl_in_tasklet(node, p, repl_expr_str)
                                if isinstance(node, dace.nodes.NestedSDFG):
                                    self._repl_use_in_tasklets_recursive(node.sdfg, repldict)

        for node in cfg.nodes():
            if isinstance(node, ControlFlowRegion):
                self._apply(node)
            elif isinstance(node, ConditionalBlock):
                for _, body in node.branches:
                    self._apply(body)
            else:
                assert isinstance(node, dace.SDFGState)

    def _find_scalar_write_expr_sets(self, sdfg: dace.SDFG) -> Dict[str, Set[str]]:
        write_expr_dict = dict()
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode) and isinstance(sdfg.sdfg.arrays[node.data],
                                                                          dace.data.Scalar):
                    scalar_name = node.data
                    if scalar_name not in write_expr_dict:
                        write_expr_dict[scalar_name] = set()
                    for ie in state.in_edges(node):
                        if not isinstance(ie.src, dace.nodes.Tasklet):
                            write_expr_dict[scalar_name].add("?")
                        else:
                            code_as_string = ie.src.code.as_string
                            if ie.src.language == dace.dtypes.Language.Python:
                                if len(code_as_string.split("=")) != 2:
                                    write_expr_dict[scalar_name].add("?")
                                rhs = code_as_string.split("=")[1]
                                try:
                                    sym_expr = dace.symbolic.SymExpr(rhs.strip())
                                    expr_set = {str(s) for s in sym_expr.free_symbols}
                                    if any(s in ie.src.in_connectors for s in expr_set):
                                        write_expr_dict[scalar_name].add("?")
                                    else:
                                        if len(expr_set) == 1 and pycode(sym_expr) == next(iter(expr_set)).strip():
                                            write_expr_dict[scalar_name] = write_expr_dict[scalar_name].union(expr_set)
                                        else:
                                            write_expr_dict[scalar_name] = write_expr_dict[scalar_name].union("?")
                                except Exception as e:
                                    write_expr_dict[scalar_name] = write_expr_dict[scalar_name].union("?")
                            else:
                                write_expr_dict[scalar_name].add("?")
        return write_expr_dict

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        self._apply(sdfg)
        sdfg.validate()

        # If used for its intention we might get scalars that are the form:
        # tmp = i (instead of i + some_expr)
        # This means, if this scalar is always written this value we can specialize it
        scalar_write_exprs = self._find_scalar_write_expr_sets(sdfg)
        for scalar_name, expr_set in scalar_write_exprs.items():
            if len(expr_set) == 1:
                expr = expr_set.pop()
                if expr != "?":
                    #print(f"Specialize {scalar_name} to {expr} as the write expression set is detected to: {expr}")
                    for state in sdfg.all_states():
                        for node in state.nodes():
                            if isinstance(node, dace.nodes.AccessNode) and node.data == scalar_name and isinstance(
                                    state.sdfg.arrays[node.data], dace.data.Scalar):
                                for ie in state.in_edges(node):
                                    if isinstance(ie.src, dace.nodes.Tasklet):
                                        # Access set not being "?" means it is a symbolic expression and not from in connectors
                                        assert state.in_degree(ie.src) == 0
                                        state.remove_node(ie.src)
                                for oe in state.out_edges(node):
                                    assert isinstance(oe.dst, dace.nodes.Tasklet)
                                    oe.dst.remove_in_connector(oe.dst_conn)
                                    oe.dst.code = CodeBlock(oe.dst.code.as_string.replace(oe.dst_conn, expr),
                                                            oe.dst.language)
                                    state.remove_edge(oe)
                                state.remove_node(node)
                    sdutil.specialize_scalar(sdfg, scalar_name, expr)
                else:
                    #print(f"Do not specialize {scalar_name} as the write expression set is detected as unknown: {expr}")
                    pass
            else:
                #print(f"Do not specialize {scalar_name} as the write expression set cardinality is not 1: {expr_set}")
                pass

        sdfg.validate()

        dace.sdfg.propagation.propagate_memlets_sdfg(sdfg)
        return None
