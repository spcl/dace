# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run ``sympy.simplify`` over every symbolic expression in an SDFG and
re-render via DaCe's sympy printer (``dace.symbolic.symstr``).

Scope:
  - ``Memlet.subset`` and ``Memlet.other_subset`` (rewritten as a new
    ``subsets.Range`` with each ``(begin, end, step)`` simplified)
  - ``Memlet.volume``
  - ``MapEntry.map.range`` (same per-bound simplification as memlet
    subsets)
  - Interstate edges: each assignment's RHS and the edge's condition,
    walked via ``all_interstate_edges`` so ``LoopRegion`` bodies and
    ``ConditionalBlock`` branches are reached too.
  - Control-flow ``CodeBlock``\\s themselves:
    ``LoopRegion.{init_statement, loop_condition, update_statement}`` and
    each ``ConditionalBlock`` branch's guard.
  - Every nested SDFG, recursively.

Range/subset bounds also get two targeted rewrites ``sympy.simplify`` does
not perform on its own.

``IfExpr(a < b, b, a) -> Max(a, b)`` (and the ``>``/``<=``/``>=`` and
``Min`` variants). ``IfExpr`` is DaCe's own ``sympy.Function``
(``dace.symbolic.IfExpr``), *not* a ``sympy.Piecewise``, so sympy never
recognises the select-the-larger idiom. ICON's ``get_indices_e`` inlines as
``IfExpr(i_startidx_in < 1, 1, i_startidx_in)``, which is exactly
``Max(1, i_startidx_in)``; recovering the ``Max`` both shortens the range
and feeds the offset push below.

``Max(a, b) + c -> Max(a + c, b + c)`` (and ``Min``, and the value branches
of an ``IfExpr``). Addition is order-preserving, so shifting every argument
of a Max/Min by the same additive term shifts the result by that term --
this holds for any term, not just constants. For ``IfExpr`` the same
rewrite is plain distributivity of addition over a branch selection, since
the predicate is an independent expression; the offset is therefore pushed
into the two value arguments only and never into the predicate.
Frontend-emitted ranges commonly look like ``Max(3, n - 2) - 1``; pushing
the offset in gives ``Max(2, n - 3)``, which is both shorter and more
likely to structurally match an equivalent range in a sibling map,
unblocking fusion.

A simplification is only applied if the rendered string actually changes,
so the returned counter reflects the number of expressions rewritten,
not merely inspected.
"""
from typing import Any

import sympy

from dace import SDFG, symbolic, subsets as subs
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl


class SimplifyExpressions(ppl.Pass):
    """``sympy.simplify`` every expression reachable from the SDFG."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Edges | ppl.Modifies.States))

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        count = simplify_expressions(sdfg)
        return count if count > 0 else None


def simplify_expressions(sdfg: SDFG) -> int:
    """Functional entry point. Returns the number of expressions rewritten."""
    count = 0
    for g in _all_sdfgs(sdfg):
        # ``symstr`` renders ``sympy.Function(name)(args)`` as ``name[args]``
        # iff the name appears in the passed set. Without it, we would
        # rewrite ``arr[i]`` accesses as ``arr(i)`` function calls and break
        # generated code.
        arrs = frozenset(g.arrays.keys())
        for state in g.all_states():
            for e in state.edges():
                count += _simplify_memlet(e.data)
            for n in state.nodes():
                if isinstance(n, nodes.MapEntry):
                    count += _simplify_map_range(n.map)

        for e in g.all_interstate_edges():
            for k, v in list(e.data.assignments.items()):
                new = _try_simplify_str(v, arrs)
                if new is not None and new != v:
                    e.data.assignments[k] = new
                    count += 1
            if e.data.condition is not None:
                cs = e.data.condition.as_string
                new = _try_simplify_str(cs, arrs)
                if new is not None and new != cs:
                    e.data.condition = CodeBlock(new, e.data.condition.language)
                    count += 1

        for block in g.all_control_flow_blocks():
            if isinstance(block, LoopRegion):
                for attr in ("init_statement", "loop_condition", "update_statement"):
                    cb = getattr(block, attr, None)
                    if cb is None:
                        continue
                    new = _try_simplify_codeblock(cb.as_string, arrs, attr)
                    if new is not None and new != cb.as_string:
                        setattr(block, attr, CodeBlock(new, cb.language))
                        count += 1
            if isinstance(block, ConditionalBlock):
                for i, (cond, body) in enumerate(block.branches):
                    if cond is None:
                        continue
                    new = _try_simplify_str(cond.as_string, arrs)
                    if new is not None and new != cond.as_string:
                        block.branches[i] = (CodeBlock(new, cond.language), body)
                        count += 1
    return count


def _try_simplify_codeblock(text: str, arrs, kind: str):
    """LoopRegion statements aren't plain expressions -- ``init_statement``
    and ``update_statement`` are assignments (``i = expr``), while
    ``loop_condition`` is a comparison. Parse/simplify only the RHS or
    the whole condition respectively."""
    if kind in ("init_statement", "update_statement") and "=" in text:
        lhs, _, rhs = text.partition("=")
        new_rhs = _try_simplify_str(rhs.strip(), arrs)
        if new_rhs is None:
            return None
        return f"{lhs.strip()} = {new_rhs}"
    return _try_simplify_str(text, arrs)


def _all_sdfgs(sdfg: SDFG):
    yield sdfg
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            yield n.sdfg


def _simplify_map_range(map_: nodes.Map) -> int:
    new_range = _try_simplify_range(map_.range)
    if new_range is not None and str(new_range) != str(map_.range):
        map_.range = new_range
        return 1
    return 0


def _simplify_memlet(m) -> int:
    if m is None:
        return 0
    count = 0
    new_subset = _try_simplify_range(m.subset)
    if new_subset is not None and str(new_subset) != str(m.subset):
        m.subset = new_subset
        count += 1
    new_other = _try_simplify_range(m.other_subset)
    if new_other is not None and str(new_other) != str(m.other_subset):
        m.other_subset = new_other
        count += 1
    if m.volume is not None:
        try:
            new_vol = symbolic.simplify(m.volume)
            if str(new_vol) != str(m.volume):
                m.volume = new_vol
                count += 1
        except Exception:
            pass
    return count


def _try_simplify_range(r):
    if not isinstance(r, subs.Range):
        return None
    try:
        new_ranges = [(_push_offset_into_selection(symbolic.simplify(b)),
                       _push_offset_into_selection(symbolic.simplify(e)),
                       _push_offset_into_selection(symbolic.simplify(s))) for b, e, s in r.ndrange()]
    except Exception:
        return None
    return subs.Range(new_ranges)


def _value_arg_positions(expr: sympy.Basic) -> tuple[int, ...] | None:
    """Positions of ``expr``'s arguments that carry a selected value, or
    ``None`` if ``expr`` does not select one. ``Max``/``Min`` select among all
    of their arguments; DaCe's ``IfExpr(predicate, if_true, if_false)`` selects
    among the last two only -- its first argument is a predicate, never a
    value, so an offset must never reach it."""
    if isinstance(expr, (sympy.Max, sympy.Min)):
        return tuple(range(len(expr.args)))
    if isinstance(expr, symbolic.IfExpr):
        return (1, 2)
    return None


def _ifexpr_to_minmax(expr: sympy.Basic) -> sympy.Basic:
    """``IfExpr(a < b, b, a) -> Max(a, b)``, ``IfExpr(a < b, a, b) -> Min(a, b)``,
    and the ``>``/``<=``/``>=`` variants. Requires the two value branches to be
    exactly the two comparison operands, so the branch that runs is determined
    entirely by which operand is larger."""
    if not isinstance(expr, symbolic.IfExpr):
        return expr
    pred, if_true, if_false = expr.args
    if getattr(pred, 'rel_op', None) not in ('<', '<=', '>', '>='):
        return expr
    if if_true == if_false or {if_true, if_false} != {pred.lhs, pred.rhs}:
        return expr
    true_takes_rhs = if_true == pred.rhs
    true_means_lhs_smaller = pred.rel_op in ('<', '<=')
    picks_larger = true_takes_rhs == true_means_lhs_smaller
    return (sympy.Max if picks_larger else sympy.Min)(pred.lhs, pred.rhs)


def _hoist_minmax_over_ifexpr(expr: sympy.Basic) -> sympy.Basic:
    """``IfExpr(p, k, Max(k, b)) -> Max(k, IfExpr(p, k, b))``, plus the ``Min`` and
    swapped-branch variants. Exact: on the branch that yields ``k`` the wrapper
    collapses (``Max(k, k) == k``), and on the other branch it is the original
    selection. Fortran ``MERGE(1, MAX(1, i), jb /= i_startblk)`` lands in this shape;
    lifting the ``Max`` out of the branch lets range analysis see it, which it
    cannot do while it is buried inside an ``IfExpr`` arm."""
    if not isinstance(expr, symbolic.IfExpr):
        return expr
    pred, if_true, if_false = expr.args
    for outer, inner, swapped in ((if_true, if_false, False), (if_false, if_true, True)):
        if not isinstance(inner, (sympy.Max, sympy.Min)) or len(inner.args) != 2:
            continue
        if outer not in inner.args:
            continue
        other = inner.args[1] if inner.args[0] == outer else inner.args[0]
        arms = (other, outer) if swapped else (outer, other)
        return inner.func(outer, symbolic.IfExpr(pred, *arms))
    return expr


def _push_offset_into_selection(expr: sympy.Basic) -> sympy.Basic:
    """Collapse ``IfExpr`` comparisons to ``Max``/``Min`` bottom-up, then push a
    shared additive offset into the value arguments of a selection node:
    ``Max(a, b) + c -> Max(a + c, b + c)``, likewise ``Min``, and
    ``IfExpr(p, a, b) + c -> IfExpr(p, a + c, b + c)``. Only triggers when
    exactly one selection term shares an ``Add`` with everything else -- sums of
    two or more are left untouched."""
    if expr.args:
        expr = expr.func(*(_push_offset_into_selection(a) for a in expr.args))
    expr = _ifexpr_to_minmax(expr)
    expr = _hoist_minmax_over_ifexpr(expr)
    if isinstance(expr, sympy.Add):
        selections = [a for a in expr.args if _value_arg_positions(a) is not None]
        if len(selections) == 1:
            sel = selections[0]
            rest = sympy.Add(*(a for a in expr.args if a is not sel))
            positions = _value_arg_positions(sel)
            return sel.func(*(_push_offset_into_selection(a + rest) if i in positions else a
                              for i, a in enumerate(sel.args)))
    return expr


def _try_simplify_str(expr: str, arrayexprs=None):
    try:
        parsed = symbolic.pystr_to_symbolic(expr)
    except Exception:
        return None
    try:
        simplified = symbolic.simplify(parsed)
    except Exception:
        return None
    try:
        return symbolic.symstr(simplified, arrayexprs=arrayexprs)
    except Exception:
        return None
