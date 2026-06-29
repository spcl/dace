# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Simplification pass that removes ``ConditionalBlock`` nodes whose condition is provably constant."""
import ast
import re
import dace
from typing import Any, Dict, Optional, Set, Union
from dace import SDFG, ControlFlowRegion
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.sdfg import ConditionalBlock
from dace.sdfg.state import ControlFlowBlock
from dace.transformation.helpers import move_branch_cfg_up_discard_conditions
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil
import sympy
from sympy import pycode


@transformation.explicit_cf_compatible
class LiftTrivialIf(ppl.Pass):
    """Remove ``ConditionalBlock`` nodes whose condition is provably constant.

    A condition is provably constant either statically (it evaluates to a literal,
    e.g. ``1 < 2``) or *over the iteration range of an enclosing loop*: a guard
    ``i == 0`` is a contradiction once the loop runs ``i`` in ``[2, N-1]``, and its
    negation ``not(i == 0)`` is then a tautology. Detecting these iteration
    tautologies/contradictions lets the pass collapse the boundary guards a loop
    peel leaves behind (``if i == N-1`` in a ``[0, N-2]`` remainder, the
    special-cased arms of an if/elif chain) so the remainder is clean affine code.

    Handles two shapes: a single-branch ``if`` (drop or replace with an empty
    branch depending on the truth value) and an ``if/else`` pair (drop the side
    that is unreachable). Runs to a fixed point at each region level and recurses
    into nested SDFGs.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return []

    def _make_unique_names(self, sdfg: dace.SDFG):
        """Relabel every block so labels are unique across the whole SDFG.

        Lifting a branch up splices its blocks into the parent region, which can
        collide with existing labels; uniquifying first keeps the result valid.

        :param sdfg: SDFG whose blocks are relabeled in place.
        """
        all_blocks = {
            n
            for n, _ in sdfg.all_nodes_recursive()
            if isinstance(n, dace.SDFGState) or isinstance(n, ControlFlowRegion) or isinstance(n, ControlFlowBlock)
        }
        all_labels: Set[str] = set()
        for n in all_blocks:
            new_label = dace.utils.find_new_name(n.label, all_labels)
            all_labels.add(new_label)
            n.label = new_label

    def _trivial_cond_check(self, code: CodeBlock, val: bool) -> bool:
        """Whether ``code`` provably evaluates to the constant truth value ``val``.

        :param code: Branch condition to inspect.
        :param val: Truth value to test the condition against.
        :returns: ``True`` only if the condition statically reduces to ``val``.
        """
        if code.language != dace.dtypes.Language.Python:
            return False

        # Primary: pystr_to_symbolic already handles Python and/or/not and
        # comparison operators. We require a concrete literal back -- bool() of
        # an unevaluated sympy expression (e.g. ``A[0]`` -> Function(0)) is
        # truthy, which would mis-classify dynamic conditions as trivial.
        try:
            expr = symbolic.pystr_to_symbolic(code.as_string)
            result = symbolic.evaluate(expr, symbols={})
            if isinstance(result, (bool, int, sympy.Integer)) or result in (sympy.true, sympy.false):
                return bool(result) is val
        except Exception:
            pass

        # Fallback: Some SDFGs (e.g. Fortran frontend) produce nested comparisons like
        # ``(a == 1) == 0`` that sympy refuses to compare bool against an int.
        # Try as best effort to rewrite boolean ops/literals to arithmetic over 0/1 and let
        # SymExpr.simplify reduce it.
        try:
            tokens = re.split(r'(\s+|[()\[\]])', code.as_string)
            replacements = {"True": "1", "False": "0", "and": "*", "or": "+"}
            rewritten = " ".join(replacements.get(t.strip(), t.strip()) for t in tokens).strip()
            simplified = dace.symbolic.SymExpr(rewritten).simplify()
            result = symbolic.evaluate(dace.symbolic.SymExpr(pycode(simplified)), symbols={})
            if isinstance(result, (bool, int, sympy.Integer)) or result in (sympy.true, sympy.false):
                return bool(result) is val
        except Exception:
            pass
        return False

    def _trivially_true(self, code: CodeBlock, cfb: Optional[ConditionalBlock] = None) -> bool:
        if self._trivial_cond_check(code, True):
            return True
        return cfb is not None and self._range_verdict(code, cfb) == 'true'

    def _trivially_false(self, code: CodeBlock, cfb: Optional[ConditionalBlock] = None) -> bool:
        if self._trivial_cond_check(code, False):
            return True
        return cfb is not None and self._range_verdict(code, cfb) == 'false'

    def _loop_iter_ranges(self, cfb: ConditionalBlock):
        """The ``(loop_variable, start, end)`` iteration range of every
        ``LoopRegion`` enclosing ``cfb``, innermost first. ``start`` / ``end`` are
        the inclusive bounds under normal (no-break) termination."""
        from dace.sdfg.state import LoopRegion
        from dace.transformation.passes.analysis import loop_analysis
        ranges = []
        graph = cfb.parent_graph
        seen = set()
        while graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            if isinstance(graph, LoopRegion) and graph.loop_variable:
                start = loop_analysis.get_init_assignment(graph)
                end = loop_analysis.get_loop_end(graph)
                if start is not None and end is not None:
                    ranges.append((graph.loop_variable, start, end))
            graph = getattr(graph, 'parent_graph', None)
        return ranges

    @staticmethod
    def _cmp_verdict(opname: str, c, start, end) -> str:
        """Whether ``i <opname> c`` is ``'true'`` / ``'false'`` for every ``i`` in
        ``[start, end]``, or ``'unknown'`` when the bounds are too symbolic to
        decide. ``c``, ``start`` and ``end`` are symbolic; a verdict is only
        returned when the deciding difference reduces to a concrete number."""

        def num(x):
            s = symbolic.simplify(x)
            return s if s.is_number else None

        def pos(x):  # provably x > 0
            n = num(x)
            return n is not None and n > 0

        def nonneg(x):  # provably x >= 0
            n = num(x)
            return n is not None and n >= 0

        def single_point(x_lo, x_hi, target):  # provably start == end == target
            return nonneg(x_hi - x_lo) and not pos(x_hi - x_lo) and num(x_lo - target) == 0

        if opname == 'Eq':  # i == c
            if pos(start - c) or pos(c - end):
                return 'false'
            if single_point(start, end, c):
                return 'true'
        elif opname == 'NotEq':  # i != c
            if pos(start - c) or pos(c - end):
                return 'true'
            if single_point(start, end, c):
                return 'false'
        elif opname == 'Lt':  # i < c
            if pos(c - end):
                return 'true'
            if nonneg(start - c):
                return 'false'
        elif opname == 'LtE':  # i <= c
            if nonneg(c - end):
                return 'true'
            if pos(start - c):
                return 'false'
        elif opname == 'Gt':  # i > c
            if pos(start - c):
                return 'true'
            if nonneg(c - end):
                return 'false'
        elif opname == 'GtE':  # i >= c
            if nonneg(start - c):
                return 'true'
            if pos(c - end):
                return 'false'
        return 'unknown'

    def _range_verdict(self, code: CodeBlock, cfb: ConditionalBlock) -> str:
        """Whether guard ``code`` is a tautology (``'true'``) or contradiction
        (``'false'``) over an enclosing loop's iteration range, else ``'unknown'``.
        Recognizes ``ivar <cmp> C`` / ``C <cmp> ivar`` (with a loop-invariant ``C``)
        and ``not(<compare>)``, matching the compared variable to the enclosing
        ``LoopRegion`` whose iterator it is."""
        if code is None or code.language != dace.dtypes.Language.Python:
            return 'unknown'
        ranges = self._loop_iter_ranges(cfb)
        if not ranges:
            return 'unknown'
        try:
            node = code.code[0]
        except (AttributeError, IndexError, TypeError):
            return 'unknown'
        if isinstance(node, ast.Expr):
            node = node.value
        negate = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        if negate:
            node = node.operand
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return 'unknown'
        left, op, right = node.left, node.ops[0], node.comparators[0]
        opname = type(op).__name__

        def name_of(n):
            return n.id if isinstance(n, ast.Name) else None

        ivar_to_range = {iv: (s, e) for iv, s, e in ranges}
        lname, rname = name_of(left), name_of(right)
        if lname in ivar_to_range and rname not in ivar_to_range:
            ivar, other = lname, right
        elif rname in ivar_to_range and lname not in ivar_to_range:
            ivar, other = rname, left
            opname = {'Lt': 'Gt', 'LtE': 'GtE', 'Gt': 'Lt', 'GtE': 'LtE'}.get(opname, opname)
        else:
            return 'unknown'
        try:
            c = symbolic.pystr_to_symbolic(ast.unparse(other))
        except Exception:
            return 'unknown'
        if symbolic.pystr_to_symbolic(ivar) in c.free_symbols:
            return 'unknown'  # C must be loop-invariant
        start, end = ivar_to_range[ivar]
        verdict = self._cmp_verdict(opname, c, start, end)
        if negate and verdict != 'unknown':
            verdict = 'false' if verdict == 'true' else 'true'
        return verdict

    @staticmethod
    def _is_conjunction(expr) -> bool:
        # ``pystr_to_symbolic`` yields dace's own conjunction operator (whose
        # ``func`` is named ``AND``), not ``sympy.And``; accept either.
        return isinstance(expr, sympy.And) or getattr(getattr(expr, 'func', None), '__name__', '') == 'AND'

    def _flatten_conjuncts(self, expr) -> list:
        """All conjuncts of a (possibly nested) conjunction, flattened.

        Conditional fusion nests the cartesian product as
        ``((c1 and c2) and c3)``; flattening exposes every atom so an atom
        and its negation can be found regardless of nesting depth.
        """
        if not self._is_conjunction(expr):
            return [expr]
        out = []
        for arg in expr.args:
            out.extend(self._flatten_conjuncts(arg))
        return out

    def _unsatisfiable(self, code: CodeBlock) -> bool:
        """Whether ``code`` is a contradiction even with free symbols: a
        (possibly nested) conjunction containing an atom and its negation
        (the ``c and not c`` combinations conditional fusion's cartesian
        product emits for identical guards). Such a branch never executes,
        so dropping it is value-preserving.
        """
        if code is None or code.language != dace.dtypes.Language.Python:
            return False
        try:
            expr = symbolic.pystr_to_symbolic(code.as_string)
        except Exception:
            return False
        if not self._is_conjunction(expr):
            return False
        conjuncts = self._flatten_conjuncts(expr)
        for i, ci in enumerate(conjuncts):
            neg = sympy.Not(ci)
            if any(j != i and neg == cj for j, cj in enumerate(conjuncts)):
                return True
        return False

    def _detect_and_remove_top_level_trivial_ifs(self, graph: Union[ControlFlowRegion, SDFG]) -> int:
        """Process the conditionals directly in ``graph`` (one level, no recursion).

        :param graph: Region or SDFG whose top-level conditionals are simplified.
        :returns: Number of branches/conditionals removed.
        """
        cfb_to_rm_cfg_to_keep = set()
        rmed_count = 0
        for cfb in graph.nodes():
            if isinstance(cfb, ConditionalBlock):
                # Longer if/elif/.../else chains (beyond the simple if and
                # if-else the cases below handle): drop branches that can
                # never hold -- a literal-false condition or a contradiction
                # such as ``c and not c``. A branch that never executes is
                # value-preserving to remove, and this collapses the
                # cartesian product a conditional-fusion emits for identical
                # guards back to the minimal chain. Gated to >2 branches so
                # the simple if / if-else handling below is untouched.
                if len(cfb.branches) > 2:
                    for cnd, cfg in list(cfb.branches):
                        if len(cfb.branches) <= 2:
                            break
                        if cnd is not None and (self._trivially_false(cnd, cfb) or self._unsatisfiable(cnd)):
                            cfb.remove_branch(cfg)
                            rmed_count += 1

                # Supported variants:
                # 1. if (cond) where cond is always true
                # 2. if (cond) else ()
                # 2.1 where cond is always true
                # 2.2 cond is always false
                conditions_and_cfgs = cfb.branches
                if len(conditions_and_cfgs) == 1:
                    cond, cfg = conditions_and_cfgs[0]
                    if self._trivially_true(cond, cfb):
                        cfb_to_rm_cfg_to_keep.add((cfb, cfg))
                    elif self._trivially_false(cond, cfb):
                        _cfg = ControlFlowRegion(label=f"empty_cfg_of_{cfb.label}", sdfg=cfb.sdfg, parent=cfb)
                        _cfg.add_state(label="empty_placholder", is_start_block=True)
                        cfb.add_branch(condition=None, branch=_cfg)
                        cfb_to_rm_cfg_to_keep.add((cfb, _cfg))
                elif len(conditions_and_cfgs) == 2:
                    cond1, cfg1 = conditions_and_cfgs[0]
                    cond2, cfg2 = conditions_and_cfgs[1]
                    # Either one of them must be none
                    if cond1 is not None and cond2 is not None:
                        continue
                    (not_none_cond, not_none_cfg), (none_cond, none_cfg) = (((cond1, cfg1),
                                                                             (cond2, cfg2)) if cond1 is not None else
                                                                            ((cond2, cfg2), (cond1, cfg1)))

                    if self._trivially_true(not_none_cond, cfb):  # 2.1
                        cfb_to_rm_cfg_to_keep.add((cfb, not_none_cfg))
                    elif self._trivially_false(not_none_cond, cfb):  # 2.2
                        cfb_to_rm_cfg_to_keep.add((cfb, none_cfg))

        for cfb, cfg in cfb_to_rm_cfg_to_keep:
            move_branch_cfg_up_discard_conditions(cfb, cfg)
            assert cfb not in graph.nodes()
            rmed_count += 1

        sdutil.set_nested_sdfg_parent_references(graph.sdfg)
        graph.sdfg.reset_cfg_list()

        return rmed_count

    def _detect_trivial_ifs_and_rm_cfg(self, graph: Union[ControlFlowRegion, SDFG]) -> int:
        """Simplify trivial conditionals in ``graph`` and all blocks nested below it.

        :param graph: Region or SDFG to simplify recursively.
        :returns: Total number of branches/conditionals removed.
        """
        # Removing a conditional can expose a new trivial one at top level, so iterate to a fixed point.
        rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
        local_rmed_count = rmed_count
        while local_rmed_count > 0:
            local_rmed_count = self._detect_and_remove_top_level_trivial_ifs(graph)
            rmed_count += local_rmed_count

        # Descend one more level into the nested control flow blocks.
        for node in graph.all_control_flow_blocks():
            local_rmed_count = self._detect_and_remove_top_level_trivial_ifs(node)
            rmed_count += local_rmed_count

        # Recurse in to nSDFGs
        for state in graph.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    rmed_count += self._detect_trivial_ifs_and_rm_cfg(node.sdfg)

        return rmed_count

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        # Start with top level nodes and continue further to ensure a trivial if within another trivial if
        # can be processed correctly
        self._make_unique_names(sdfg)
        sdfg.reset_cfg_list()
        self._detect_trivial_ifs_and_rm_cfg(sdfg)
        sdfg.reset_cfg_list()
        return None
