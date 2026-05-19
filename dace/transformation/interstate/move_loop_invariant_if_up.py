# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoist a loop-invariant guarding conditional out of its enclosing loop.

The inverse of ``MoveIfIntoLoop``: ``for k: { prep*; if c: body }`` becomes
``prep*; if c: { for k: body }`` when ``c`` (and the interstate-edge symbol
assignments it depends on) is provably invariant w.r.t. the loop. Applied to
a fixpoint, so an innermost invariant guard sifts all the way up through a
stack of nested loops (a mix of loops and maps -- maps are lowered to
``LoopRegion`` s by the canonicalize pipeline before this runs, so operating
on ``LoopRegion`` covers both).

Conservative -- only fires when:

* the loop body is a linear chain whose only non-empty block is the
  single-branch (no-else) ``ConditionalBlock``; empty boundary states are
  allowed and are dropped (they carry no effect, so the loop is a genuine
  no-op when ``c`` is false -> hoisting is value-preserving: ``c`` true =>
  the loop runs the body every iteration as before; ``c`` false => the loop
  did nothing observable before and is simply not entered after);
* the condition does not reference the loop variable nor any data/symbol
  written inside the loop, *except* symbols an interstate edge of the chain
  assigns from a provably-invariant expression -- those assignments are
  hoisted out *with* the guard so it still sees them;
* there are no other (non-invariant) interstate-edge assignments in the
  chain.

Anything else is a no-op. Linear-CFG assumption: the loop body is a plain
linear chain; blocks may have parents (the pass recurses into all regions).
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg import nodes
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation


def _free(expr: str) -> set:
    """Free symbol names of a condition / assignment expression string."""
    try:
        import dace.symbolic as sym
        return {str(s) for s in sym.pystr_to_symbolic(expr).free_symbols}
    except Exception:
        return set()


def _written(region) -> set:
    """Data containers + interstate-assigned symbols written inside ``region``."""
    w = set()
    for e in region.all_interstate_edges():
        w |= set(e.data.assignments.keys())
    for st in region.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                w.add(n.data)
    return w


def _linear_order(region: ControlFlowRegion) -> Optional[List]:
    """Blocks of ``region`` in order iff it is a plain linear chain of
    unconditional edges (assignments allowed -- they carry condition prep);
    else ``None``."""
    blocks = list(region.nodes())
    edges = list(region.edges())
    if not blocks or len(edges) != len(blocks) - 1:
        return None
    for e in edges:
        if e.data.condition.as_string not in ('1', 'True', '(1)'):
            return None
    succ = {e.src: e.dst for e in edges}
    order = [region.start_block]
    while order[-1] in succ:
        order.append(succ[order[-1]])
    return order if len(order) == len(blocks) else None


def _is_empty_state(b) -> bool:
    return isinstance(b, SDFGState) and b.number_of_nodes() == 0


def _match(sdfg: SDFG):
    """Find a ``LoopRegion`` whose body is ``[empty*; if c; empty*]`` with a
    loop-invariant condition (plus a hoistable invariant assignment chain).

    :returns: ``(loop, cond_block, cond, hoist_assignments)`` or ``None``.
    """
    for loop in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(loop, LoopRegion):
            continue
        order = _linear_order(loop)
        if order is None:
            continue
        cbs = [b for b in order if isinstance(b, ConditionalBlock)]
        non_empty = [b for b in order if not _is_empty_state(b)]
        if len(cbs) != 1 or non_empty != cbs:
            continue  # the only non-empty body block must be the guard
        cb = cbs[0]
        if len(cb.branches) != 1:
            continue
        cond, branch = cb.branches[0]
        if cond is None or not isinstance(branch, ControlFlowRegion):
            continue

        lvar = str(loop.loop_variable)
        loop_w = _written(loop)
        # Interstate-edge assignments on the loop's own chain. Each must be
        # provably invariant (no loop var, RHS free of loop-written data) to
        # be hoistable; the guard may only depend on such hoistable symbols.
        chain_assignments: List[Tuple[str, str]] = []
        bad = False
        for e in loop.edges():
            for lhs, rhs in e.data.assignments.items():
                rf = _free(rhs)
                if lvar in rf or (rf & loop_w) or lhs in loop_w - set(e.data.assignments):
                    bad = True
                    break
                chain_assignments.append((lhs, rhs))
            if bad:
                break
        if bad:
            continue

        assigned = {lhs for lhs, _ in chain_assignments}
        cfree = {str(s) for s in cond.get_free_symbols()}
        # The guard must be invariant: no loop variable, and no dependence on
        # loop-written data/symbols other than the hoistable chain symbols.
        if lvar in cfree:
            continue
        if (cfree & loop_w) - assigned:
            continue
        return loop, cb, cond, chain_assignments
    return None


@transformation.explicit_cf_compatible
class MoveLoopInvariantIfUp(ppl.Pass):
    """Hoist a loop-invariant guarding conditional out of its loop (fixpoint).

    The inverse of ``MoveIfIntoLoop``. Repeatedly applied so an innermost
    invariant guard sifts all the way up through nested loops. The
    interstate-edge symbol-assignment chain the condition depends on is
    hoisted with it; emptied boundary states are dropped.
    """
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Hoist invariant guards out of their loops until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of guards hoisted, or ``None`` if none.
        """
        count = 0
        while True:
            m = _match(sdfg)
            if m is None:
                break
            self._move(*m)
            count += 1
        if count:
            set_nested_sdfg_parent_references(sdfg)
        return count or None

    @staticmethod
    def _move(loop: LoopRegion, cb: ConditionalBlock, cond: CodeBlock, hoist_assignments: List[Tuple[str, str]]):
        """Replace ``loop`` with ``[assign chain]; if cond: { loop' }`` where
        ``loop'`` is ``loop`` with its body reduced to the guard's branch
        body (empty boundary states and the now-hoisted guard removed)."""
        parent = loop.parent_graph
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop

        cond, branch = copy.deepcopy(cb).branches[0]

        # loop' = a copy of the loop whose body is exactly the guard's body.
        new_loop = copy.deepcopy(loop)
        for b in list(new_loop.nodes()):
            new_loop.remove_node(b)
        bb = list(branch.nodes())
        be = list(branch.edges())
        bstart = branch.start_block
        for b in bb:
            new_loop.add_node(b, ensure_unique_name=True)
        for e in be:
            new_loop.add_edge(e.src, e.dst, copy.deepcopy(e.data))
        new_loop.start_block = new_loop.node_id(bstart)

        # if cond: { loop' }
        inner = ControlFlowRegion(label=f"{loop.label}_body")
        inner.add_node(new_loop, is_start_block=True, ensure_unique_name=True)
        outer_cb = ConditionalBlock(label=f"{loop.label}_guard")
        outer_cb.add_branch(CodeBlock(cond.as_string), inner)

        parent.add_node(outer_cb, ensure_unique_name=True)
        # Hoisted invariant assignments go on the edge entering the guard.
        hoisted = {lhs: rhs for lhs, rhs in hoist_assignments}
        for e in in_edges:
            data = copy.deepcopy(e.data)
            data.assignments.update(hoisted)
            parent.add_edge(e.src, outer_cb, data)
        if not in_edges and hoisted:
            pre = parent.add_state(f"{loop.label}_hoist")
            parent.add_edge(pre, outer_cb, InterstateEdge(assignments=hoisted))
            if is_start:
                parent.start_block = parent.node_id(pre)
                is_start = False
        for e in out_edges:
            parent.add_edge(outer_cb, e.dst, copy.deepcopy(e.data))
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(loop)
        if is_start:
            parent.start_block = parent.node_id(outer_cb)
