# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import dace
from typing import Dict, Iterable, List, Optional
import sympy
from sympy.printing.pycode import pycode
from dace import SDFG
from dace import properties
from dace import Union
from dace import ControlFlowRegion
from dace.properties import Property
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.symbolic import symstr
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg.nodes import CodeBlock
import ast


def _get_expr_from_str(expr: str) -> dace.symbolic.SymExpr:
    try:
        parsed_expr = sympy.sympify(expr, evaluate=False)
    except Exception as e:
        parsed_expr = dace.symbolic.SymExpr(expr)
    return parsed_expr


def create_new_memlet(edge_data: dace.memlet.Memlet, repldict: Dict[str, str]) -> Optional[dace.memlet.Memlet]:
    """Copy ``edge_data`` with ``repldict`` substituted into BOTH of its subsets.

    A two-sided (copy) memlet names one array in ``.data`` and keeps the OTHER side's
    indexing in ``.other_subset``; both sides are indexed by the enclosing scope's
    parameters, so a scope rewrite has to reach both. ``Memlet.replace`` does exactly
    that (plus the volume) via a collision-safe two-phase substitution, so a copy of
    the memlet is handed to it rather than rebuilding one from a subset string --
    which also keeps ``wcr`` / ``dynamic`` / ``is_data_src``, all dropped by a
    ``Memlet(expr=...)`` round-trip.

    :param edge_data: The memlet to rewrite (not modified).
    :param repldict: Symbol name -> replacement expression, as source strings.
    :returns: The rewritten memlet, or ``None`` if there is no subset to rewrite.
    """
    if edge_data is None or edge_data.subset is None:
        return None
    # Pre-sympify the values with dace's own parser: handing ``.subs`` plain strings makes
    # sympy re-parse them with vanilla ``sympify``, which mis-resolves a bare symbol name
    # colliding with a sympy builtin (``N`` is ``sympy.N``, the numeric-eval function).
    sym_repldict = {k: dace.symbolic.pystr_to_symbolic(v) for k, v in repldict.items()}
    new_memlet = copy.deepcopy(edge_data)
    new_memlet.replace(sym_repldict)
    return new_memlet


def update_edge_if_changed(state: dace.SDFGState, edge, new_memlet: Optional[dace.memlet.Memlet]) -> None:
    """Re-attach ``edge`` carrying ``new_memlet`` if that differs from what it carries now."""
    if new_memlet and new_memlet != edge.data:
        state.remove_edge(edge)
        state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)


def process_memlets_in_edges(state: dace.SDFGState, edges: Iterable, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into the memlet of every edge in ``edges``."""
    for edge in edges:
        update_edge_if_changed(state, edge, create_new_memlet(edge.data, repldict))


def repl_memlets_recursive(cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into every memlet of ``cfg``, descending into nested SDFGs."""
    for state in cfg.all_states():
        process_memlets_in_edges(state, state.edges(), repldict)
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_memlets_recursive(node.sdfg, repldict)


def repl_interstate_edges_recursive(cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into every inter-state edge of ``cfg``, descending into nested SDFGs."""
    for edge in [] if isinstance(cfg, dace.SDFGState) else cfg.all_interstate_edges():
        edge.data.replace_dict(repldict)

    for state in [cfg] if isinstance(cfg, dace.SDFGState) else cfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_interstate_edges_recursive(node.sdfg, repldict)


def token_replace_dict(code: str, repldict: Dict[str, str]) -> str:
    """Replace whole tokens of ``code`` that exactly match a key of ``repldict``."""
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', code)

    # Replace tokens that exactly match src
    tokens = [repldict[token.strip()] if token.strip() in repldict else token for token in tokens]

    # Recombine everything
    return ''.join(tokens).strip()


def repl_tasklets_on_node_list(node_list: Iterable[dace.nodes.Node], repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into the body of every tasklet in ``node_list``."""
    # Pre-sympify once (see ``create_new_memlet`` for why): vanilla ``.subs`` on a plain
    # ``{str: str}`` dict mis-resolves a symbol name colliding with a sympy builtin.
    sym_repldict = {dace.symbolic.pystr_to_symbolic(k): dace.symbolic.pystr_to_symbolic(v) for k, v in repldict.items()}
    for node in node_list:
        if isinstance(node, dace.nodes.Tasklet):
            code = node.code
            code_str = copy.deepcopy(node.code.as_string)
            if code.language == dace.dtypes.Language.Python:
                # Can raise exceptions if you have stuff like AND in the expression
                try:
                    symexpr = dace.symbolic.SymExpr(code_str.split(" = ")[-1].strip())
                    symexpr = symexpr.subs(sym_repldict)
                    code_str = code_str.split(" = ")[0].strip() + " = " + pycode(symexpr, allow_unknown_functions=True)
                except Exception as e:
                    code_str = copy.deepcopy(node.code.as_string)
                    code_str = token_replace_dict(code_str, repldict)
            else:
                code_str = token_replace_dict(code_str, repldict)
            node.code = CodeBlock(code_str, code.language)


def repl_tasklets_recursive(cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into every tasklet of ``cfg``, descending into nested SDFGs."""
    for state in [cfg] if isinstance(cfg, dace.SDFGState) else cfg.all_states():
        repl_tasklets_on_node_list(state.nodes(), repldict)
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_tasklets_recursive(node.sdfg, repldict)


def repl_for_regions_recursive(root: ControlFlowRegion, cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into every ``LoopRegion`` header under ``cfg`` except ``root``'s own."""
    for node in [] if isinstance(cfg, dace.SDFGState) else cfg.all_control_flow_regions():
        if node == root:
            continue
        if isinstance(node, LoopRegion):
            # TODO: do it better (try sympy subs)
            # A while-shaped region carries no init / update statement -- rewrite what it does carry.
            if node.loop_condition is not None:
                node.loop_condition = CodeBlock(token_replace_dict(node.loop_condition.as_string, repldict),
                                                node.loop_condition.language)
            if node.init_statement is not None:
                node.init_statement = CodeBlock(token_replace_dict(node.init_statement.as_string, repldict),
                                                node.init_statement.language)
            if node.update_statement is not None:
                node.update_statement = CodeBlock(token_replace_dict(node.update_statement.as_string, repldict),
                                                  node.update_statement.language)

    for state in [] if isinstance(cfg, dace.SDFGState) else cfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_for_regions_recursive(root, node.sdfg, repldict)


def repl_if_blocks_recursive(cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` into every ``ConditionalBlock`` branch condition under ``cfg``."""
    for node in [] if isinstance(cfg, dace.SDFGState) else cfg.all_control_flow_regions():
        if isinstance(node, ConditionalBlock):
            for i, (cond, body) in enumerate(node.branches):
                ncond = None
                if cond is not None:
                    # TODO: do it better (try sympy subs)
                    code_str = token_replace_dict(cond.as_string, repldict)
                    ncond = CodeBlock(code_str, cond.language)
                    node.branches[i] = (ncond, body)

    for state in [] if isinstance(cfg, dace.SDFGState) else cfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_if_blocks_recursive(node.sdfg, repldict)


def repl_recursive(cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
    """Substitute ``repldict`` everywhere inside ``cfg``: inter-state edges, memlets, tasklets,
    nested loop headers and branch conditions. ``cfg``'s OWN loop header is left alone."""
    repl_interstate_edges_recursive(cfg, repldict)
    repl_memlets_recursive(cfg, repldict)
    repl_tasklets_recursive(cfg, repldict)
    repl_for_regions_recursive(cfg, cfg, repldict)
    repl_if_blocks_recursive(cfg, repldict)


def add_to_rhs(expr: str, add_expr: dace.symbolic.SymExpr) -> str:
    """Add ``add_expr`` to the right-hand side of the comparison ``expr``.

    :param expr: A comparison, as source (e.g. ``i < N``).
    :param add_expr: The expression to add to the first comparator.
    :returns: The rewritten comparison, as source.
    :raises ValueError: If ``expr`` is not a comparison.
    """
    tree = ast.parse(expr, mode="eval")
    comparison = tree.body

    if not isinstance(comparison, ast.Compare) or not comparison.comparators:
        raise ValueError("Expression must be a comparison with at least one comparator")

    # Modify the first comparator by adding the expression
    original_rhs = comparison.comparators[0]
    add_expr_ast = ast.parse(symstr(add_expr), mode="eval").body

    comparison.comparators[0] = ast.BinOp(left=original_rhs, op=ast.Add(), right=add_expr_ast)

    # Convert back to string and simplify
    str_expr = ast.unparse(tree)
    sym_expr = _get_expr_from_str(str_expr)
    return symstr(sym_expr)


@properties.make_properties
@transformation.explicit_cf_compatible
class OffsetLoopsAndMaps(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    offset_expr = Property(dtype=str, default="0")
    begin_expr = Property(dtype=str, default="0")
    do_not_check_begin = Property(dtype=bool, default=False)
    convert_leq_to_lt = Property(dtype=bool, default=True)
    normalize_loops = Property(dtype=bool, default=False)
    squeeze = Property(dtype=bool, default=False)

    def __init__(self,
                 offset_expr: str,
                 begin_expr: Union[str, None],
                 convert_leq_to_lt: bool = True,
                 normalize_loops: bool = False,
                 squeeze: bool = False):
        self.offset_expr = offset_expr
        if begin_expr is None:
            self.do_not_check_begin = True
        else:
            self.begin_expr = begin_expr
        self.convert_leq_to_lt = convert_leq_to_lt
        self.normalize_loops = normalize_loops
        self.squeeze = squeeze

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.CFG | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    # The substitution machinery below is shared with ``NormalizeLoopAndMapOrigin``, which drives it
    # per scope instead of over the whole SDFG; it lives at module level so both callers reach it
    # without touching another pass's privates. These methods stay as the pass-facing spelling.

    def _create_new_memlet(self, edge_data: dace.memlet.Memlet, repldict: Dict[str,
                                                                               str]) -> Optional[dace.memlet.Memlet]:
        """Create a new memlet with substituted subset ranges."""
        return create_new_memlet(edge_data, repldict)

    def _update_edge_if_changed(self, state, edge, new_memlet) -> None:
        """Update edge if the new memlet is different from the current one."""
        update_edge_if_changed(state, edge, new_memlet)

    def _process_memlets_in_edges(self, state, edges, repldict: Dict[str, str]) -> None:
        """Process memlets in a collection of edges."""
        process_memlets_in_edges(state, edges, repldict)

    def _repl_memlets_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
        """Recursively replace memlets in all states of a control flow region."""
        repl_memlets_recursive(cfg, repldict)

    def _repl_memlets_on_edge_list(self, state, edges, repldict: Dict[str, str]) -> None:
        """Replace memlets on a specific list of edges."""
        process_memlets_in_edges(state, edges, repldict)

    def _repl_memlets_on_edge_list_recursive(self, state, edges, repldict: Dict[str, str]) -> None:
        """Replace memlets on a specific list of edges."""
        process_memlets_in_edges(state, edges, repldict)

        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_memlets_recursive(node.sdfg, repldict)

    def _repl_interstate_edges_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
        """Recursively replace interstate edges in control flow region."""
        repl_interstate_edges_recursive(cfg, repldict)

    def _repl_tasklets_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
        repl_tasklets_recursive(cfg, repldict)

    def _token_replace_dict(self, code: str, repldict: Dict[str, str]) -> str:
        return token_replace_dict(code, repldict)

    def _repl_tasklets_on_node_list(self, state: dace.SDFGState, nodes: List[dace.nodes.Node],
                                    repldict: Dict[str, str]) -> None:
        repl_tasklets_on_node_list(nodes, repldict)

    def _repl_tasklets_recursive_from_node_list(self, state: dace.SDFGState, nodes: List[dace.nodes.Node],
                                                repldict: Dict[str, str]) -> None:
        repl_tasklets_on_node_list(nodes, repldict)

        for node in nodes:
            if isinstance(node, dace.nodes.NestedSDFG):
                repl_tasklets_recursive(node.sdfg, repldict)

    def _token_match(self, code: str, src: str) -> bool:
        # Split while keeping delimiters
        tokens = re.split(r'(\s+|[()\[\]])', code)
        # Replace tokens that exactly match src
        tokens = [token for token in tokens if token.strip() == src]

        # Recombine everything
        return len(tokens) != 0

    def _repl_for_regions_recursive(self, root: ControlFlowRegion, cfg: ControlFlowRegion, repldict: Dict[str,
                                                                                                          str]) -> None:
        repl_for_regions_recursive(root, cfg, repldict)

    def _repl_if_blocks_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
        repl_if_blocks_recursive(cfg, repldict)

    def _repl_recursive(self, cfg: ControlFlowRegion, repldict: Dict[str, str]) -> None:
        """Replace both interstate edges and memlets recursively."""
        repl_recursive(cfg, repldict)

    def _add_to_rhs(self, expr: str, add_expr: dace.symbolic.SymExpr, sdfg: dace.SDFG) -> str:
        """Add an expression to the right-hand side of a comparison."""
        return add_to_rhs(expr, add_expr)

    def _apply(self, cfg: dace.ControlFlowRegion) -> int:
        """Offset every matching loop / map in ``cfg`` and everything nested below it.

        :param cfg: The region to rewrite in place.
        :returns: Number of loop regions and map entries that were offset.
        """
        offset_count = 0
        for node in cfg.nodes():
            if isinstance(node, LoopRegion):
                # A while-shaped region carries no counter to offset (``init_statement`` /
                # ``update_statement`` are ``None``) -- leave it alone and keep walking.
                if node.init_statement is None or node.loop_condition is None:
                    continue
                # The begin expression matches apply offset
                init_lhs, init_rhs = node.init_statement.as_string.split("=")
                if self.do_not_check_begin or _get_expr_from_str(init_rhs) == _get_expr_from_str(
                        self.begin_expr) or str(init_rhs) == str(self.begin_expr):
                    init_expr_str = f"(({init_rhs}) + {self.offset_expr})"
                    init_expr = _get_expr_from_str(init_expr_str)
                    new_init_statement = symstr(init_expr)
                    node.init_statement = CodeBlock(f"{init_lhs} = {new_init_statement}")
                    new_loop_condition = self._add_to_rhs(node.loop_condition.as_string,
                                                          _get_expr_from_str(self.offset_expr), cfg.sdfg)
                    node.loop_condition = CodeBlock(new_loop_condition)

                    # Try normalize after update
                    if self.normalize_loops:
                        node.normalize()

                    # Parenthesize the offset before subtracting: a bare f-string interpolation of a
                    # multi-term offset (e.g. ``2 - N``) would produce ``(j - 2 - N)``, which parses as
                    # ``j - 2 - N`` instead of the intended ``j - (2 - N) = j - 2 + N`` -- silently wrong
                    # for any compound (non-single-token) offset expression.
                    v = f"({node.loop_variable} - ({_get_expr_from_str(self.offset_expr)}))"
                    repldict = {node.loop_variable: v}

                    self._repl_recursive(node, repldict)
                    offset_count += 1
            elif isinstance(node, dace.SDFGState):
                state = node
                for state_node in state.nodes():
                    if isinstance(state_node, dace.nodes.MapEntry):
                        has_matches = False
                        new_range_list = []
                        repldict = dict()
                        multipliers = []
                        for (b, e, s), param in zip(state_node.map.range, state_node.map.params):
                            if self.do_not_check_begin or b == _get_expr_from_str(self.begin_expr) or str(b) == str(
                                    self.begin_expr):
                                has_matches = True

                                b_expr = dace.symbolic.SymExpr(
                                    symstr(b) + " + " + symstr(_get_expr_from_str(self.offset_expr))).simplify()
                                e_expr = dace.symbolic.SymExpr(
                                    symstr(e) + " + " + symstr(_get_expr_from_str(self.offset_expr))).simplify()
                                s_expr = dace.symbolic.SymExpr(symstr(s)).simplify()
                                prev_s_expr = s_expr
                                if self.squeeze:
                                    loop_len = e_expr + 1 - b_expr
                                    loop_step = s_expr
                                    if isinstance(loop_len / loop_step, (int, sympy.Number)):
                                        multiplier = dace.symbolic.SymExpr(int(loop_len / loop_step))
                                        multipliers.append(multiplier)
                                        assert b_expr == 0
                                        e_expr = b_expr + multiplier - 1
                                        s_expr = dace.symbolic.SymExpr(1)

                                new_range_list.append((b_expr, e_expr, s_expr))

                                # Parenthesize the offset before subtracting (see the matching fix in
                                # the ``LoopRegion`` branch above): unparenthesized, a compound offset
                                # (e.g. ``2 - N``) silently flips the sign of its trailing terms.
                                offset_str = symstr(_get_expr_from_str(self.offset_expr))
                                if self.squeeze:
                                    repldict[param] = f"(({param} * {prev_s_expr}) - ({offset_str}))"
                                else:
                                    repldict[param] = f"({param} - ({offset_str}))"
                            else:
                                new_range_list.append((b, e, s))

                        if has_matches:
                            new_range = dace.subsets.Range(new_range_list)
                            state_node.map.range = new_range
                            nodes_between = state.all_nodes_between(state_node, state.exit_node(state_node))
                            edges_between = state.all_edges(*nodes_between)
                            self._repl_memlets_on_edge_list(state, edges_between, repldict)
                            self._repl_tasklets_on_node_list(state, nodes_between, repldict)
                            for n in nodes_between:
                                if isinstance(n, dace.nodes.NestedSDFG):
                                    self._repl_recursive(n.sdfg, repldict)
                            offset_count += 1

        for node in cfg.nodes():
            if isinstance(node, ConditionalBlock):
                for _, body in node.branches:
                    offset_count += self._apply(body)
            elif isinstance(node, ControlFlowRegion):
                # Covers LoopRegion and any other control flow region.
                offset_count += self._apply(node)
            elif isinstance(node, dace.SDFGState):
                # Descend into NestedSDFGs so their maps/loops get
                # offset too -- not just the ones at the enclosing
                # SDFG's top level (e.g. loop bodies that ``LoopToMap``
                # promoted into a ``loop_body`` nested SDFG).
                for state_node in node.nodes():
                    if isinstance(state_node, dace.nodes.NestedSDFG):
                        offset_count += self._apply(state_node.sdfg)
            # Other control flow blocks (break/continue/return) hold no
            # states or regions to offset and are skipped.
        return offset_count

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
        expr_str = lhs + op_to_split + symstr(dace.symbolic.SymExpr(rhs).simplify()) + (")" * (opens - exits))
        return expr_str

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[int]:
        """Shift every matching loop / map in ``sdfg`` by ``offset_expr``.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results of prior passes in the pipeline, forwarded to the unit-copy sub-pass.
        :returns: Number of rewrites made (unit copies split + loops/maps offset + loop expressions simplified),
                  or ``None`` if nothing matched.
        """
        # Cross-array unit copies are split into ``_out = _in`` assign tasklets first --
        # the documented contract of ``InsertAssignTaskletsForUnitCopies``, matching the
        # canonicalize pipeline's ordering. (``create_new_memlet`` now substitutes
        # ``other_subset`` as well, so this is a shape choice, no longer a workaround.)
        from dace.transformation.passes.insert_unit_copy_assign_tasklets import (
            InsertAssignTaskletsForUnitCopies, )
        unit_copies = InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, pipeline_results or {})

        # Do it for LoopRegions and Maps
        rewrites = self._apply(sdfg)
        rewrites += unit_copies or 0
        sdfg.validate()

        # Simplify <= loop conditions to use < if set
        if self.convert_leq_to_lt:
            for n, g in sdfg.all_nodes_recursive():
                if (isinstance(n, LoopRegion) and n.loop_condition is not None
                        and n.loop_condition.language == dace.dtypes.Language.Python):
                    old_condition = n.loop_condition.as_string
                    expr = dace.symbolic.SymExpr(n.loop_condition.as_string)
                    if isinstance(expr, sympy.core.relational.Relational) and isinstance(expr, sympy.LessThan):
                        lhs, rhs = expr.lhs, expr.rhs
                        n.loop_condition = CodeBlock(symstr(sympy.StrictLessThan(lhs, rhs + 1)))

                    # Simplify only the rhs do this by splitting the expression from "<" and ( with the number of opened ( from left
                    # Then simplify it and add back
                    expr_str = self._split_expr_str_opt_rhs(n.loop_condition.as_string, " < ")
                    n.loop_condition = CodeBlock(expr_str)
                    if n.loop_condition.as_string != old_condition:
                        rewrites += 1
        sdfg.validate()

        # Try to simplify loop init statements, expressions such as ((-1) + 1)
        for n, g in sdfg.all_nodes_recursive():
            if (isinstance(n, LoopRegion) and n.init_statement is not None
                    and n.init_statement.language == dace.dtypes.Language.Python):
                old_init = n.init_statement.as_string
                try:
                    expr_str = self._split_expr_str_opt_rhs(n.init_statement.as_string, " = ")
                except Exception as e:
                    print(str(e))
                    expr_str = n.init_statement.as_string
                n.init_statement = CodeBlock(expr_str)
                if n.init_statement.as_string != old_init:
                    rewrites += 1
        sdfg.validate()

        return rewrites or None
