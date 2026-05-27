# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Push a loop-invariant guarding conditional into the loop body.

The ``LoopRegion`` analogue of ``MoveIfIntoMap``: ``if c: { prep; for i:
body }`` becomes ``for i: { if c: { prep; body } }`` so later fission/fusion
can cross the conditional. Applied to a fixpoint, so a stack of nested
guards (``if c1: if c2: for i: ...``) is pushed in one level at a time --
including conditions the frontend materializes via loop-invariant prep
states and interstate-edge assignments. Conservative: it only fires on a
linear ``[prep..., loop]`` branch with a provably loop-invariant condition
and prep (no dependence on the loop variable, no clobber of loop-written
data); anything else is a no-op.

Free-state (imperfect-nest) case
--------------------------------
A frontend imperfect nest -- ``if c: { for j: body1 ; s }`` where ``s`` is a
bare ``SDFGState`` sibling of the loop, not its prep -- does not match the
``[prep..., one loop]`` shape. For such a heterogeneous branch body (a linear
chain of ``LoopRegion`` and bare ``SDFGState`` blocks, with at least one of
each) the guard is *distributed*: every bare state is wrapped in a trivial
single-iteration ``LoopRegion`` so the whole body is loops, then a copy of
the guard is pushed into each loop's body. This drops nothing -- the entire
subgraph stays guarded, one duplicated guard per sibling -- and is value-
preserving for a single-branch (no-else) guard whose condition is invariant
w.r.t. every sibling loop: guard true => every sibling runs (as before),
guard false => none runs (as before). The single-iteration wrappers are
spliced back out by the canonicalize pipeline's ``untrivialize`` stage
(``TrivialLoopElimination``) before ``LoopToMap``. Because the guard always
sits inside a loop (a real sibling loop, or the trivial wrapper of a bare
sibling), no top-level ``ConditionalBlock`` is ever produced -- so the path
fires regardless of nesting level (including a top-level guarded body).

Interstate-edge assignments on the branch chain are **not** duplicated into
the per-sibling guards, so they would execute unconditionally -- incorrect.
The pass therefore refuses (no-op) on any region carrying an interstate-edge
assignment and emits a loud warning so the dropped opportunity is visible.
"""
import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import dace.symbolic
from dace import SDFG
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg import nodes
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation


def _written(region: ControlFlowRegion) -> set:
    """Symbols/data written inside a region (interstate assigns + data writes)."""
    w = set()
    for e in region.all_interstate_edges():
        w |= set(e.data.assignments.keys())
    for st in region.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0:
                w.add(n.data)
    return w


def _linear_order(region: ControlFlowRegion) -> Optional[List]:
    """Blocks of ``region`` in order iff it is a plain linear chain
    (unconditional edges; assignments are allowed -- they carry condition
    prep); else ``None``."""
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


def _free(expr: str) -> set:
    """Free symbol names of a Python/condition expression string."""
    try:
        return {str(s) for s in dace.symbolic.pystr_to_symbolic(expr).free_symbols}
    except Exception:
        return set()


def _match(sdfg: SDFG) -> Optional[Tuple[ConditionalBlock, CodeBlock, ControlFlowRegion]]:
    """Find a ``ConditionalBlock`` guarding (loop-invariant prep then) a loop.

    :returns: ``(cond_block, condition, region)`` or ``None``.
    """
    for cb in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(cb, ConditionalBlock) or len(cb.branches) != 1:
            continue
        cond, region = cb.branches[0]
        if cond is None or not isinstance(region, ControlFlowRegion):
            continue  # single branch, real condition, no else
        order = _linear_order(region)
        if order is None or not isinstance(order[-1], LoopRegion):
            continue
        loop = order[-1]
        prep = order[:-1]
        if any(not isinstance(p, SDFGState) for p in prep):
            continue
        lvar = str(loop.loop_variable)
        loop_w = _written(loop)
        cfree = {str(s) for s in cond.get_free_symbols()}
        if lvar in cfree or (cfree & loop_w):
            continue  # condition loop-invariant
        # Prep states + the chain's edge assignments must be loop-invariant:
        # not depend on the loop var, not clobber loop-written data. They must
        # also not *produce* anything the loop's own bounds consume: _move sinks
        # the prep into the loop body, so a prep that computes a bound (e.g. a
        # materialized ``LEN_1D_minus_k = LEN_1D - k`` feeding ``i < LEN_1D_minus_k``)
        # would leave that bound uninitialized at the first condition check.
        bound_syms = _loop_bound_symbols(loop)
        bad = False
        for p in prep:
            acc = {n.data for n in p.nodes() if isinstance(n, nodes.AccessNode)}
            if lvar in acc or (_state_writes(p) & loop_w) or (_state_writes(p) & bound_syms):
                bad = True
        for e in region.edges():
            for lhs, rhs in e.data.assignments.items():
                if lhs in loop_w or lhs in bound_syms or lvar in _free(rhs) or (_free(rhs) & loop_w):
                    bad = True
        if bad:
            continue
        return cb, cond, region
    return None


def _state_writes(st: SDFGState) -> set:
    """Data containers written in ``st``."""
    return {n.data for n in st.nodes() if isinstance(n, nodes.AccessNode) and st.in_degree(n) > 0}


def _loop_bound_symbols(loop: LoopRegion) -> set:
    """Names referenced by ``loop``'s bound expressions (init/condition/update),
    excluding the loop variable itself.

    The loop's trip count is decided from these before the body runs, so any
    name here must already hold its value at loop entry. Used to refuse sinking
    a prep that *produces* a bound name into the body (which would leave the
    first condition check reading an uninitialized value)."""
    names = set()
    for stmt in (loop.init_statement, loop.loop_condition, loop.update_statement):
        if stmt is not None:
            names |= {str(s) for s in stmt.get_free_symbols()}
    names.discard(str(loop.loop_variable))
    return names


def _match_imperfect(sdfg: SDFG) -> Optional[Tuple[ConditionalBlock, CodeBlock, ControlFlowRegion]]:
    """Find a ``ConditionalBlock`` guarding a *heterogeneous* body: a linear
    chain of ``LoopRegion`` and bare ``SDFGState`` blocks with at least one of
    each (a frontend imperfect nest, e.g. ``if c: { for j: body1 ; s }``).

    Distinct from :func:`_match`, which only takes ``[prep..., one loop]``;
    here the guard is duplicated into every sibling (bare states first wrapped
    in a trivial single-iteration loop, so the duplicated guard sits *inside*
    that loop -- never a top-level ``ConditionalBlock``; the wrapper is spliced
    out later by the ``untrivialize`` stage). The condition must be invariant
    w.r.t. every sibling loop and unclobbered by any sibling bare state. A
    region carrying an interstate-edge assignment is a no-op (the assignment
    cannot be moved under the per-sibling guards, so it would run
    unconditionally) and a loud warning is emitted.

    :returns: ``(cond_block, condition, region)`` or ``None``.
    """
    for cb in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(cb, ConditionalBlock) or len(cb.branches) != 1:
            continue
        cond, region = cb.branches[0]
        if cond is None or not isinstance(region, ControlFlowRegion):
            continue
        order = _linear_order(region)
        if order is None:
            continue
        if any(not isinstance(b, (LoopRegion, SDFGState)) for b in order):
            continue
        loops = [b for b in order if isinstance(b, LoopRegion)]
        states = [b for b in order if isinstance(b, SDFGState)]
        if not loops or not states:
            continue  # heterogeneous only; the pure cases are _match's job
        # Leave the existing ``[prep states..., exactly one trailing loop]``
        # fast path entirely to _match (it places prep *inside* that loop).
        if len(loops) == 1 and order[-1] is loops[0] and all(isinstance(b, SDFGState) for b in order[:-1]):
            continue
        if any(e.data.assignments for e in region.edges()):
            warnings.warn("\n" + "!" * 78 + "\n"
                          "MoveIfIntoLoop: refusing to distribute guard "
                          f"{cond.as_string!r} over an imperfect nest in "
                          f"{cb.label!r}: the branch carries an interstate-edge "
                          "assignment that cannot be moved under the per-sibling "
                          "guards and would execute UNCONDITIONALLY. Left unchanged.\n" + "!" * 78,
                          stacklevel=2)
            continue  # conservative: no inter-block interstate assignments
        cfree = {str(s) for s in cond.get_free_symbols()}
        # The guard's truth must be the same for every sibling: it must not
        # depend on any sibling loop's variable / written data, nor on data a
        # sibling bare state writes.
        if any(str(lp.loop_variable) in cfree or (cfree & _written(lp)) for lp in loops):
            continue
        if any(cfree & _state_writes(st) for st in states):
            continue
        return cb, cond, region
    return None


def _guarded_loop(loop: LoopRegion, cond: CodeBlock) -> LoopRegion:
    """Return a copy of ``loop`` whose body is ``if cond: <original body>``."""
    lp = copy.deepcopy(loop)
    body_blocks = list(lp.nodes())
    body_edges = list(lp.edges())
    body_start = lp.start_block
    inner = ControlFlowRegion(label=f"{lp.label}_g")
    for b in body_blocks:
        lp.remove_node(b)
    for b in body_blocks:
        inner.add_node(b, ensure_unique_name=True)
    for e in body_edges:
        inner.add_edge(e.src, e.dst, copy.deepcopy(e.data))
    inner.start_block = inner.node_id(body_start)
    icb = ConditionalBlock(label=f"{lp.label}_if")
    icb.add_branch(CodeBlock(cond.as_string), inner)
    lp.add_node(icb, is_start_block=True, ensure_unique_name=True)
    return lp


def _trivial_guarded_loop(state: SDFGState, cond: CodeBlock) -> LoopRegion:
    """Wrap ``state`` in a trivial single-iteration loop whose body is
    ``if cond: <state>`` (the free-state perfect-nesting wrapper)."""
    inner = ControlFlowRegion(label=f"{state.label}_g")
    inner.add_node(copy.deepcopy(state), is_start_block=True)
    icb = ConditionalBlock(label=f"{state.label}_if")
    icb.add_branch(CodeBlock(cond.as_string), inner)
    tv = f"__triv_{state.label}"
    triv = LoopRegion(f"{state.label}_triv", f"{tv} < 1", tv, f"{tv} = 0", f"{tv} = {tv} + 1")
    triv.add_node(icb, is_start_block=True)
    return triv


@transformation.explicit_cf_compatible
class MoveIfIntoLoop(ppl.Pass):
    """Push a loop-invariant guarding conditional into the loop body (fixpoint).

    Handles both the ``[prep..., one loop]`` shape (the loop is hoisted to the
    conditional's position; its body becomes ``if c: { prep; body }``) and the
    heterogeneous imperfect-nest shape (each bare-state sibling is wrapped in a
    trivial single-iteration loop and a copy of the guard is pushed into every
    sibling loop). Both are value-preserving for a single-branch no-else guard
    with a sibling-invariant condition.
    """
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Repeatedly push guards into their loops until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of guards moved, or ``None`` if none.
        """
        count = 0
        while True:
            m = _match(sdfg)
            if m is not None:
                self._move(*m)
                count += 1
                continue
            m = _match_imperfect(sdfg)
            if m is not None:
                self._move_imperfect(*m)
                count += 1
                continue
            break
        if count:
            # _move / _move_imperfect deepcopy + re-add blocks; any nested
            # SDFG carried along keeps stale parent references until repaired.
            set_nested_sdfg_parent_references(sdfg)
        return count or None

    @staticmethod
    def _move(cb: ConditionalBlock, cond: CodeBlock, region: ControlFlowRegion):
        """Hoist the loop to ``cb``'s position; the new loop body is
        ``if cond: { prep ; original-loop-body }`` with all condition-prep
        states and their interstate-edge assignments preserved."""
        parent = cb.parent_graph
        in_edges = list(parent.in_edges(cb))
        out_edges = list(parent.out_edges(cb))
        is_start = parent.start_block is cb

        # Work on a copy of the branch region; splice the loop's body in
        # place of the loop so the region becomes  prep... -> body...
        rc = copy.deepcopy(region)
        order = _linear_order(rc)
        loop_c = order[-1]
        edge_into_loop = next((e for e in rc.edges() if e.dst is loop_c), None)

        lb_blocks = list(loop_c.nodes())
        lb_start = loop_c.start_block
        lb_edges = list(loop_c.edges())
        for b in lb_blocks:
            loop_c.remove_node(b)
        rc.remove_node(loop_c)
        for b in lb_blocks:
            rc.add_node(b, ensure_unique_name=True)
        for e in lb_edges:
            rc.add_edge(e.src, e.dst, copy.deepcopy(e.data))
        if edge_into_loop is not None:
            rc.add_edge(edge_into_loop.src, lb_start, copy.deepcopy(edge_into_loop.data))
            rc.start_block = rc.node_id(order[0])
        else:
            rc.start_block = rc.node_id(lb_start)

        inner_cb = ConditionalBlock(label=f"{loop_c.label}_if")
        inner_cb.add_branch(CodeBlock(cond.as_string), rc)

        new_loop = copy.deepcopy(loop_c)
        for b in list(new_loop.nodes()):
            new_loop.remove_node(b)
        new_loop.add_node(inner_cb, is_start_block=True, ensure_unique_name=True)

        parent.add_node(new_loop, ensure_unique_name=True)
        for e in in_edges:
            parent.add_edge(e.src, new_loop, copy.deepcopy(e.data))
        for e in out_edges:
            parent.add_edge(new_loop, e.dst, copy.deepcopy(e.data))
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(cb)
        if is_start:
            parent.start_block = parent.node_id(new_loop)

    @staticmethod
    def _move_imperfect(cb: ConditionalBlock, cond: CodeBlock, region: ControlFlowRegion):
        """Replace ``cb`` with the branch body's blocks in order, each carrying
        its own copy of the guard: a sibling ``LoopRegion`` becomes a loop
        whose body is ``if cond: <body>``; a bare ``SDFGState`` becomes a
        trivial single-iteration loop whose body is ``if cond: <state>``."""
        parent = cb.parent_graph
        in_edges = list(parent.in_edges(cb))
        out_edges = list(parent.out_edges(cb))
        is_start = parent.start_block is cb

        order = _linear_order(copy.deepcopy(region))
        units = [(_guarded_loop(b, cond) if isinstance(b, LoopRegion) else _trivial_guarded_loop(b, cond))
                 for b in order]

        for u in units:
            parent.add_node(u, ensure_unique_name=True)
        for a, b in zip(units, units[1:]):
            parent.add_edge(a, b, InterstateEdge())
        for e in in_edges:
            parent.add_edge(e.src, units[0], copy.deepcopy(e.data))
        for e in out_edges:
            parent.add_edge(units[-1], e.dst, copy.deepcopy(e.data))
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(cb)
        if is_start:
            parent.start_block = parent.node_id(units[0])
