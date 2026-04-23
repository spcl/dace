# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run ``sympy.simplify`` over every symbolic expression in an SDFG and
re-render via DaCe's sympy printer (``dace.symbolic.symstr``).

Scope:
  - ``Memlet.subset`` and ``Memlet.other_subset`` (rewritten as a new
    ``subsets.Range`` with each ``(begin, end, step)`` simplified)
  - ``Memlet.volume``
  - Interstate edges: each assignment's RHS and the edge's condition,
    walked via ``all_interstate_edges`` so ``LoopRegion`` bodies and
    ``ConditionalBlock`` branches are reached too.
  - Control-flow ``CodeBlock``\\s themselves:
    ``LoopRegion.{init_statement, loop_condition, update_statement}`` and
    each ``ConditionalBlock`` branch's guard.
  - Every nested SDFG, recursively.

A simplification is only applied if the rendered string actually changes,
so the returned counter reflects the number of expressions rewritten,
not merely inspected.
"""
from typing import Any, Dict, Optional

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

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
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
        new_ranges = [(symbolic.simplify(b),
                       symbolic.simplify(e),
                       symbolic.simplify(s)) for b, e, s in r.ndrange()]
    except Exception:
        return None
    return subs.Range(new_ranges)


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
