# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Closed-form substitution for simple induction-variable loops.

A loop body of the shape ``for i in range(start, end, stride): accum = accum OP const``
(where ``OP`` is ``+`` or ``*`` and ``const`` is a numeric literal) is a scalar
recurrence with a closed form. Following Aho/Lam/Sethi/Ullman (the "red dragon
book", Ch. 9.6 -- induction variables and scalar evolution):

* ``accum = accum + c`` over ``N`` iters  ->  ``accum_N = accum_init + c*N``
* ``accum = accum * c`` over ``N`` iters  ->  ``accum_N = accum_init * c**N``

Eliminating the loop turns an ``O(N)`` recurrence into ``O(1)`` straight-line
code. Cross-reference: LLVM's ``IndVarSimplify`` pass, especially the
``llvm/test/Transforms/IndVarSimplify/closed-form-*.ll`` family of tests.

Scope today (kept narrow on purpose so the pass is provably correct):

* single-state body with a single tasklet,
* tasklet code = ``__out = __in1 OP const`` (or symmetric),
* read input traces back (through one optional slice-transient) to the same
  accumulator slot the tasklet's output eventually reaches,
* both the read and write subsets are loop-invariant (``a[i] = a[i] * 0.5`` is
  a per-element map, NOT an IV; rejected),
* stride ``== 1`` (extending to symbolic strides is a follow-up; the closed
  form is the same except trip count is ``(end-start)/stride+1``).

Out of scope for this implementation (potential follow-ups):

* derived IVs (``j = a*i + b`` from a basic IV ``i``) -- needs the IV graph
  from Ch. 9.6, then strength-reduction or closed-form substitution;
* loop-invariant non-literal operand (``s = s * k[0]`` with ``k[0]`` invariant)
  -- straightforward extension to the constant-detection branch;
* multi-statement bodies where the IV is one of several updates -- needs the
  "hoist IV out, leave the rest in the loop" split (TODO);
* non-commutative WCR ops -- our substitution assumes commutativity, which
  holds for ``+`` and ``*`` but would not for general WCR.

Eliminating the loop turns an O(N) recurrence into O(1) straight-line code,
and runs BEFORE LoopToReduce / LoopToMap so the IV-eligible loop never gets
mis-classified as a fold or a parallel map. The TSVC kernel ``s317``
(``q[0] *= 0.99`` for ``LEN_1D//2`` iters) is the canonical hit.
"""
import ast
from typing import Optional, Set, Tuple

from dace import SDFG, dtypes, nodes, properties, symbolic
from dace.sdfg import SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_to_reduce import _chase_forward_to_accum, _one_elem, _uses

#: AST binop type -> closed-form template ``(init, c, n) -> str``.
_CLOSED_FORM = {
    ast.Add: lambda init, c, n: f"(({init}) + ({c}) * ({n}))",
    ast.Mult: lambda init, c, n: f"(({init}) * (({c}) ** ({n})))",
}


class _UnwrapTypecasts(ast.NodeTransformer):
    """Strip ``dace.<typeclass>(x)`` calls -- the frontend's defensive type casts
    around symbolic operands (e.g. ``__in1 + dace.float64(step)``) -- by
    replacing each such call with its single argument. Identity semantics for IV
    pattern matching; the codegen still emits the cast from the original tasklet
    body, only this pass's analysis treats it as a no-op.
    """
    from dace import dtypes as _dtypes
    _TYPECAST_NAMES = set(_dtypes.TYPECLASS_STRINGS)

    def visit_Call(self, node):
        self.generic_visit(node)
        # Match ``dace.<typeclass>(x)``: ``func`` is Attribute(value=Name('dace'), attr=typeclass)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'dace' and node.func.attr in self._TYPECAST_NAMES and len(node.args) == 1
                and not node.keywords):
            return node.args[0]
        return node


@properties.make_properties
@xf.explicit_cf_compatible
class InductionVariableSubstitution(ppl.Pass):
    """Eliminate a single-tasklet ``acc = acc OP const`` loop via closed form."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        # Fixed point: substituting a primary IV frees the symbols derived from
        # it. e.g. TSVC s128 -- substituting ``j := j + 2`` rewrites ``k := j + 1``
        # to ``k := 2*i`` (a pure loop-var expression), which the next round's
        # ``_try_substitute_derived_symbol`` then folds into the ``b[k]`` gathers.
        # Each substitution strictly removes one carried symbol / loop, so this
        # terminates; the cap is a runaway backstop.
        count = 0
        for _ in range(1000):
            progressed = False
            # ``SDFG.free_symbols`` re-derives itself from the whole graph on EVERY access -- it is a
            # property, not a cached one (~0.5s on CloudSC). The invariance checks below consult it once
            # per candidate symbol, which made it 96% of this pass's runtime (163s of 169s over 500
            # calls). It only changes when the SDFG does, and every mutation below restarts this round,
            # so hoisting it here is exact. ``materialize_loop_exit_symbols`` already threads it the
            # same way for the same reason.
            sdfg_free_symbols = sdfg.free_symbols
            for node, parent in list(sdfg.all_nodes_recursive()):
                if not isinstance(node, LoopRegion):
                    continue
                # An early-exit loop (``break`` / ``continue`` targeting THIS loop) has a
                # data-dependent trip count: the IV's value at exit is NOT the counted final
                # value, and body reads of the IV must stay per-iteration. Closed-form /
                # exit-value substitution would corrupt them (e.g. rewriting a break guard's
                # ``d_idx = d[i]`` to ``d[N-1]``). Skip -- the split passes can't handle these
                # loops either; the early-exit lift runs before this stage instead.
                if _loop_has_break_or_continue(node):
                    continue
                # (1) whole-loop collapse ``acc = acc OP const`` -> closed form
                #     (eliminates the loop; the exponentiation collapse s317);
                # (2) an interstate-edge recurrence ``sym := sym + step`` -> closed
                #     form in the body (keeps the loop; the counter-IV shape);
                # (3) a symbol defined purely by a loop-var expression -> inline it
                #     (the derived symbol a primary-IV substitution just freed);
                # (4) an IV incremented identically in every branch of a body
                #     conditional -> hoist it out so (2) can then close it (s124).
                if (_try_substitute(parent, node, sdfg, sdfg_free_symbols)
                        or _try_substitute_iedge_iv(parent, node, sdfg, sdfg_free_symbols)
                        or _try_substitute_derived_symbol(parent, node, sdfg, sdfg_free_symbols)
                        or _hoist_branch_uniform_iv(parent, node, sdfg, sdfg_free_symbols)):
                    count += 1
                    progressed = True
                    break  # SDFG mutated -> restart the scan on fresh node list
            if not progressed:
                break
        return count or None


def _loop_has_break_or_continue(loop: LoopRegion) -> bool:
    """True if ``loop`` has a ``break`` / ``continue`` targeting it.

    A ``BreakBlock`` / ``ContinueBlock`` targets the innermost enclosing loop, so a break
    inside a nested ``LoopRegion`` belongs to that inner loop -- descend through conditional
    branches and non-loop regions, but not into nested loops.
    """
    stack = list(loop.nodes())
    while stack:
        blk = stack.pop()
        if isinstance(blk, (BreakBlock, ContinueBlock)):
            return True
        if isinstance(blk, LoopRegion):
            continue  # a break inside a nested loop targets that loop, not this one
        if isinstance(blk, ConditionalBlock):
            for _, branch in blk.branches:
                stack.extend(branch.nodes())
        elif isinstance(blk, ControlFlowRegion):
            stack.extend(blk.nodes())
    return False


def _try_substitute(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: Set[str]) -> bool:
    """Return True if the loop matched and was replaced; False otherwise."""
    info = _extract_iv(loop, sdfg, sdfg_free_symbols)
    if info is None:
        return False
    accum, accum_subset, op_type, const_val, trip_count = info

    # The closed-form RHS reads the seed via the tasklet's ``__in`` connector,
    # NOT via a bare ``accum[subset]`` expression -- the SDFG dataflow is what
    # actually wires the read.
    closed = _CLOSED_FORM[op_type]("__in", const_val, symbolic.symstr(trip_count))

    _replace_loop_with_closed_form(parent, loop, accum, accum_subset, closed, sdfg)
    return True


def _is_loop_invariant_symbol(name: str, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: Set[str]) -> bool:
    """Whether ``name`` refers to an SDFG symbol/constant that the loop does not
    redefine in its body (so its value is stable across iterations).

    Accepts: SDFG symbols and constants, with the loop variable explicitly excluded,
    and with the symbol not appearing as the LHS of any interstate-edge assignment
    inside the loop's body.

    ``sdfg_free_symbols`` is ``sdfg.free_symbols`` precomputed by the caller: that property walks the
    whole SDFG on every access, so it is passed in rather than recomputed per candidate symbol.
    """
    if name == loop.loop_variable:
        return False
    if name not in sdfg.symbols and name not in sdfg.constants and name not in sdfg_free_symbols:
        return False
    # The loop must not assign to ``name`` on any of its body interstate edges --
    # otherwise it would not be loop-invariant.
    for e in loop.edges():
        if e.data.assignments and name in e.data.assignments:
            return False
    return True


def _extract_iv(loop: LoopRegion, sdfg: SDFG,
                sdfg_free_symbols: Set[str]) -> Optional[Tuple[str, str, type, object, object]]:
    """Pattern-match the loop body. Returns ``(accum_name, accum_subset_str, ast.BinOp_type, const_val, trip_count)``
    or ``None``.

    ``const_val`` may be a Python ``int`` / ``float`` (numeric literal) OR a string
    naming a loop-invariant SDFG symbol -- in the symbolic case the closed form
    ``init + c*N`` / ``init * c**N`` keeps ``c`` as a name and is materialised in
    the post-loop tasklet by the codegen's symbol-binding path.
    """
    if not loop.loop_variable:
        return None
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None

    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return None
    state = blocks[0]

    tasklet = None
    for n in state.nodes():
        if isinstance(n, nodes.Tasklet):
            if tasklet is not None:
                return None
            tasklet = n
        elif not isinstance(n, nodes.AccessNode):
            return None
    if tasklet is None or tasklet.code.language != dtypes.Language.Python:
        return None

    try:
        tree = ast.parse((tasklet.code.as_string or "").strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    # Strip the frontend's defensive ``dace.<typeclass>(...)`` casts before pattern
    # matching, so ``__in1 + dace.float64(step)`` matches identically to ``__in1 + step``.
    rhs = _UnwrapTypecasts().visit(tree.body[0].value)
    if not isinstance(rhs, ast.BinOp) or type(rhs.op) not in _CLOSED_FORM:
        return None

    # Identify the carrier side (a bare ``ast.Name`` whose id is one of the
    # tasklet's INPUT CONNECTORS) and the increment side (anything else, as long
    # as it is loop-invariant). The classic shape is ``__in1 + 2.5`` (Constant on
    # the other side); the symbolic-frontend shape is ``__in1 + step`` with
    # ``step`` a SDFG symbol, which lifts the same way.
    in_connector_names = set(tasklet.in_connectors.keys())

    def _is_carrier(node):
        return isinstance(node, ast.Name) and node.id in in_connector_names

    if _is_carrier(rhs.left) and not _is_carrier(rhs.right):
        var_conn, other = rhs.left.id, rhs.right
    elif _is_carrier(rhs.right) and not _is_carrier(rhs.left):
        var_conn, other = rhs.right.id, rhs.left
    else:
        return None

    # The increment expression must not reference the loop variable; every
    # ``ast.Name`` in it must be a loop-invariant SDFG symbol / constant or a
    # built-in DaCe binding (``dace.float64`` etc.). Constants are inherently OK.
    if isinstance(other, ast.Constant):
        if not isinstance(other.value, (int, float)):
            return None
        const_val = other.value
    else:
        for sub in ast.walk(other):
            if isinstance(sub, ast.Name):
                if sub.id == loop.loop_variable:
                    return None
                # Allow SDFG symbols / constants / known dtype-cast roots; reject
                # any other name (which would imply a connector or loop-local var).
                if (sub.id not in sdfg.symbols and sub.id not in sdfg.constants and sub.id not in sdfg_free_symbols
                        and sub.id != 'dace' and not hasattr(__import__('builtins'), sub.id)):
                    return None
                if (sub.id in sdfg.symbols or sub.id in sdfg.constants or sub.id
                        in sdfg_free_symbols) and not _is_loop_invariant_symbol(sub.id, loop, sdfg, sdfg_free_symbols):
                    return None
        # Render the expression back to a source string for the closed form. The
        # later tasklet body splices it directly, so ``dace.float64(step)`` stays
        # ``dace.float64(step)`` and codegen resolves ``step`` via the symbol-binding path.
        const_val = ast.unparse(other)

    in_edges = [e for e in state.in_edges(tasklet) if e.data is not None and not e.data.is_empty()]
    out_edges = [e for e in state.out_edges(tasklet) if e.data is not None and not e.data.is_empty()]
    if len(in_edges) != 1 or len(out_edges) != 1:
        return None
    (in_edge, ) = in_edges
    (write_edge, ) = out_edges
    if in_edge.dst_conn != var_conn:
        return None
    if not isinstance(write_edge.dst, nodes.AccessNode):
        return None

    write_subset = write_edge.data.subset
    if _one_elem(write_subset) != 1:
        return None
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    if _uses(write_subset, loop_var_sym):
        return None

    final_accum, final_subset = _chase_forward_to_accum(state, sdfg, write_edge.dst, write_subset)
    if final_accum not in sdfg.arrays:
        return None
    # The eventual write subset must also be loop-invariant. Otherwise the
    # write hits a different slot each iteration (e.g. ``a[i] = a[i] * 0.5`` --
    # a per-element map, not a loop-carried accumulator IV).
    if _uses(final_subset, loop_var_sym):
        return None

    # Trace the input back to the same accumulator slot.
    src = in_edge.src
    if not isinstance(src, nodes.AccessNode):
        return None
    desc = sdfg.arrays.get(src.data)
    if desc is None:
        return None
    if desc.transient and len(state.in_edges(src)) == 1:
        pred = state.in_edges(src)[0]
        if not isinstance(pred.src, nodes.AccessNode) or pred.data is None or pred.data.subset is None:
            return None
        src_name = pred.src.data
        src_subset = pred.data.subset
    else:
        src_name = src.data
        src_subset = in_edge.data.subset
    if src_name != final_accum or str(src_subset) != str(final_subset):
        return None

    # Trip count = (end - start) // stride + 1 (loop_analysis.get_loop_end is inclusive).
    trip_count = symbolic.simplify((end - start) // stride + 1)

    return final_accum, str(final_subset), type(rhs.op), const_val, trip_count


def _replace_loop_with_closed_form(parent: ControlFlowRegion, loop: LoopRegion, accum_name: str, accum_subset: str,
                                   closed_form: str, sdfg: SDFG) -> None:
    """Swap ``loop`` for a state whose tasklet writes the closed form back to ``accum_name[accum_subset]``."""
    from dace import memlet as mm

    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))

    new_state = parent.add_state(loop.label + "_iv_closed", is_start_block=was_start)
    accum_r = new_state.add_read(accum_name)
    accum_w = new_state.add_write(accum_name)
    tasklet = new_state.add_tasklet(loop.label + "_iv_closed_tlt", {"__in"}, {"__out"}, f"__out = {closed_form}")
    new_state.add_edge(accum_r, None, tasklet, "__in", mm.Memlet(data=accum_name, subset=accum_subset))
    new_state.add_edge(tasklet, "__out", accum_w, None, mm.Memlet(data=accum_name, subset=accum_subset))

    for e in in_edges:
        parent.add_edge(e.src, new_state, e.data)
    for e in out_edges:
        parent.add_edge(new_state, e.dst, e.data)
    parent.remove_node(loop)


# -----------------------------------------------------------------------------
# Iedge-based IV substitution (multi-statement bodies)
# -----------------------------------------------------------------------------


def _symbol_updated_in_other_loop(sdfg: SDFG, loop: LoopRegion, sym_name: str) -> bool:
    """Whether ``sym_name`` is assigned on a direct interstate edge of any
    ``LoopRegion`` other than ``loop`` -- i.e. it is a counter shared with a
    nested or enclosing loop (TSVC s126). Such a symbol has no per-loop closed
    form, so the caller refuses to substitute it."""
    for region in sdfg.all_control_flow_regions():
        if not isinstance(region, LoopRegion) or region is loop:
            continue
        for e in region.edges():
            if sym_name in (e.data.assignments or {}):
                return True
    return False


def _hoist_branch_uniform_iv(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG,
                             sdfg_free_symbols: Set[str]) -> bool:
    """Hoist an IV increment that EVERY branch of a body ``ConditionalBlock``
    performs identically (``sym := sym + step`` on all paths) out of the
    conditional, so the branches share one increment on a single iedge.

    The increment lands BEFORE the conditional when the branches read ``sym``
    post-increment, and AFTER it when they read ``sym`` pre-increment -- whichever
    keeps the in-branch uses seeing exactly the value the sequential body gave them
    (see the ``side`` dispatch below).

    This is a structural enabler, not itself a substitution: after the hoist the
    increment is a plain between-blocks iedge that :func:`_try_substitute_iedge_iv`
    (next fixed-point round) closes. TSVC ``s124`` -- ``j += 1`` in BOTH the ``if``
    and the ``else``, before the ``a[j]`` writes -- becomes ``j = i`` so ``a[j]`` is
    the parallel ``a[i]``; the mirrored read-before-increment body closes the same
    way with the pre-increment offset.
    Requires an exhaustive conditional (an ``else`` branch): with an implicit
    fall-through some path skips the increment and the hoist would be unsound.
    """
    import dace
    for cb in [b for b in loop.nodes() if isinstance(b, ConditionalBlock)]:
        conds = [c for c, _ in cb.branches]
        branches = [br for _, br in cb.branches]
        if len(branches) < 2 or all(c is not None for c in conds):
            continue  # no else branch -> a path skips the increment -> unsound to hoist

        def branch_increments(br):
            incs = {}
            for e in br.edges():
                for lhs, rhs in (e.data.assignments or {}).items():
                    try:
                        delta = symbolic.simplify(symbolic.pystr_to_symbolic(rhs) - symbolic.pystr_to_symbolic(lhs))
                    except Exception:
                        continue
                    if getattr(delta, 'is_number', False):
                        incs.setdefault(lhs, []).append((e, delta))
            return incs

        per = [branch_increments(br) for br in branches]
        common = set.intersection(*[set(p) for p in per]) if per else set()
        for sym in common:
            if sym == loop.loop_variable or (sym not in sdfg.symbols and sym not in sdfg_free_symbols):
                continue
            if any(len(p[sym]) != 1 for p in per):
                continue  # a branch increments sym more than once
            steps = {p[sym][0][1] for p in per}
            if len(steps) != 1:
                continue  # branches disagree on the step
            step = next(iter(steps))
            branch_edges = {id(p[sym][0][0]) for p in per}
            if any(sym in (e2.data.assignments or {}) and id(e2) not in branch_edges
                   for e2 in loop.all_interstate_edges()):
                continue  # sym also written outside the per-branch increments (incl. nested) -> not clean
            # Soundness: the hoist moves each branch's increment to a single iedge
            # OUTSIDE the conditional, and WHICH SIDE it lands on is dictated by where
            # the branches read ``sym``. Blocks outside ``cb`` are unaffected either way
            # (both positions keep the increment between the pre-cb and post-cb blocks),
            # so only the in-branch uses decide:
            #
            # * ``'after'`` -- every branch increments before it reads (s124: ``j += 1``
            #   precedes ``a[j]``), so the uses want the POST-increment value: hoist to a
            #   single iedge BEFORE the conditional.
            # * ``'before'`` -- every branch reads before it increments (``a[j] = ...;
            #   j += 1``), so the uses want the PRE-increment value: hoist to a single
            #   iedge AFTER the conditional. Same value seen in-branch, and the increment
            #   is again a plain between-blocks iedge for the closed form (which then
            #   picks the pre-increment ``body_offset = norm_iter``).
            #
            #
            # A branch with NO use of ``sym`` (``'unused'``) is indifferent -- both
            # positions give it the same (unread) value -- so it does not vote; the
            # branches that DO read decide. If no branch reads at all, either position is
            # correct and we take ``'after'``.
            #
            # Only genuine ambiguity refuses: branches disagreeing on the side, or a
            # branch straddling its own increment (``None``).
            sides = {_consistent_use_side(br, p[sym][0][0], sym) for br, p in zip(branches, per)} - {'unused'}
            if not sides:
                side = 'after'
            elif sides in ({'after'}, {'before'}):
                (side, ) = sides
            else:
                continue
            # Strip the increment from each branch, then plant one iedge on the side the
            # branches' uses demand.
            for p in per:
                e, _ = p[sym][0]
                assigns = dict(e.data.assignments)
                assigns.pop(sym, None)
                e.data.assignments = assigns
            hoist = loop.add_state(cb.label + '_iv_hoist')
            new_rhs = symbolic.symstr(symbolic.pystr_to_symbolic(sym) + step)
            if side == 'after':
                was_start = loop.start_block is cb
                for ie in list(loop.in_edges(cb)):
                    loop.add_edge(ie.src, hoist, ie.data)
                    loop.remove_edge(ie)
                loop.add_edge(hoist, cb, dace.InterstateEdge(assignments={sym: new_rhs}))
                if was_start:
                    loop.start_block = loop.node_id(hoist)
            else:
                for oe in list(loop.out_edges(cb)):
                    loop.add_edge(hoist, oe.dst, oe.data)
                    loop.remove_edge(oe)
                loop.add_edge(cb, hoist, dace.InterstateEdge(assignments={sym: new_rhs}))
            return True
    return False


def _consistent_use_side(loop: LoopRegion, iv_edge, sym_name: str) -> Optional[str]:
    """Whether every USE of ``sym_name`` in the loop body executes on ONE side of
    the IV increment ``iv_edge`` (``sym := sym + step``).

    Returns ``'before'`` if all uses run before the increment (pre-increment: the
    body sees ``sym_init + norm_iter * step``), ``'after'`` if all run after it
    (post-increment: ``... + (norm_iter + 1) * step``), ``'unused'`` if the body
    contains NO use at all (both sides are vacuously correct -- the caller is free
    to pick either), or ``None`` if the uses straddle both sides, which genuinely
    needs per-block offsets and is the only case that must refuse.

    ``'unused'`` and ``None`` are deliberately distinct: conflating them would turn
    "either answer is right" into "no answer exists" and refuse a liftable loop.

    This generalizes the TOP / BOTTOM shape check to an IV increment sitting
    *between* content blocks (TSVC ``s128``: ``k := j + 1`` before ``j := j + 2``
    -- the only ``j`` use is the ``k`` iedge, which precedes the increment). Sides
    are decided by reverse/forward reachability from the increment's endpoints;
    the loop body is a DAG (the back-edge is the region boundary, not a body
    edge), so reachability is exact.
    """
    before = set(sdutil.dfs_conditional(loop, sources=[iv_edge.src], reverse=True))
    before.add(iv_edge.src)
    after = set(sdutil.dfs_conditional(loop, sources=[iv_edge.dst]))
    after.add(iv_edge.dst)

    saw_before = saw_after = False
    # Block uses: any body block that reads ``sym`` -- a state (memlet / tasklet)
    # OR a nested region (a ConditionalBlock whose branches read ``sym``, e.g. the
    # ``a[j]`` writes in s124). ``free_symbols`` is defined for every block type.
    for b in loop.nodes():
        if sym_name in {str(s) for s in b.free_symbols}:
            if b in before:
                saw_before = True
            elif b in after:
                saw_after = True
            else:
                return None  # neither strictly before nor after the increment
    # Interstate-edge uses (RHS assignments / condition), excluding the IV edge.
    for e in loop.edges():
        if e is iv_edge or sym_name not in {str(s) for s in e.data.free_symbols}:
            continue
        if e.dst in before:  # the edge completes at-or-before the increment's source
            saw_before = True
        elif e.src in after:  # the edge starts at-or-after the increment's destination
            saw_after = True
        else:
            return None
    if saw_before and saw_after:
        return None  # straddles the increment -> per-block offsets needed (unsupported)
    if saw_before:
        return 'before'
    if saw_after:
        return 'after'
    return 'unused'  # no body use -> either offset reproduces the (absent) reads


def _preloop_symbol_value(parent: ControlFlowRegion, loop: LoopRegion, sym_name: str):
    """The value ``sym_name`` holds on ENTRY to ``loop``, or ``None`` if it is not
    locally provable.

    Sound because it only answers when EVERY edge entering ``loop`` assigns
    ``sym_name`` the same expression -- then whichever path reached the loop, that
    is the entry value. No in-edges (``loop`` starts ``parent``), a path that does
    not assign it, or paths that disagree all mean "unknown", and the caller must
    then refuse rather than guess.
    """
    in_edges = list(parent.in_edges(loop))
    if not in_edges:
        return None
    vals = set()
    for e in in_edges:
        rhs = (e.data.assignments or {}).get(sym_name)
        if rhs is None:
            return None
        try:
            vals.add(symbolic.pystr_to_symbolic(rhs))
        except Exception:
            return None
    if len(vals) != 1:
        return None
    (val, ) = vals
    if loop.loop_variable in {str(s) for s in val.free_symbols}:
        return None  # references the loop variable, which is undefined before the loop
    return val


def _try_substitute_derived_symbol(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG,
                                   sdfg_free_symbols: Set[str]) -> bool:
    """Substitute a symbol defined *purely* by a loop-variable expression.

    A body iedge ``sym := f(loop_var, <loop-invariant symbols>)`` with NO
    self-reference (``sym`` not in ``f``) makes ``sym`` a plain derived quantity,
    not a recurrence -- every iteration's value is a closed-form function of the
    loop variable. Inline it: replace ``sym`` with ``f`` throughout the body and
    drop the defining iedge.

    This is the second half of the fixed-point (see :meth:`apply_pass`): once
    :func:`_try_substitute_iedge_iv` turns a primary IV ``j`` into a constant, a
    derived ``k := j + 1`` becomes ``k := 2*i`` -- now a pure loop-var expression
    this catches, folding it into the ``b[k]`` gathers so ``LoopToMap`` can
    parallelize (TSVC s128).

    Uses of ``sym`` may sit on EITHER side of the definition; the side picks which
    closed form the body reads (see the ``side`` dispatch below):

    * uses AFTER the definition -- or no use at all -- read this iteration's ``f(i)``;
    * uses BEFORE it read what the previous iteration left, ``f(i - stride)``, which
      is a closed form too -- but only from the second iteration on. The FIRST
      iteration reads whatever ``sym`` held on ENTRY to the loop, so this side is
      sound exactly when that entry value is provably ``f(start - stride)``
      (:func:`_preloop_symbol_value`). ``k = 0; for i: a[i] = b[k]; k = i + 1`` is the
      canonical hit: entry ``0 == f(-1)``, so ``b[k]`` folds to the parallel ``b[i]``.
      An entry value that disagrees (or is not locally provable) is refused -- the
      lagged closed form would then mispredict exactly the first iteration.

    Uses STRADDLING the definition need per-block offsets and are refused.
    """
    if not loop.loop_variable:
        return False
    loop_var = loop.loop_variable
    for e in loop.edges():
        if e.data.condition.as_string not in ('1', 'True', '(1)') or len(e.data.assignments) != 1:
            continue
        ((sym, rhs), ) = e.data.assignments.items()
        if sym == loop_var or (sym not in sdfg.symbols and sym not in sdfg_free_symbols):
            continue
        try:
            rhs_expr = symbolic.pystr_to_symbolic(rhs)
        except Exception:
            continue
        free = {str(s) for s in rhs_expr.free_symbols}
        if sym in free:
            continue  # self-reference -> a recurrence, not a derived symbol
        if not all(s == loop_var or _is_loop_invariant_symbol(s, loop, sdfg, sdfg_free_symbols) for s in free):
            continue  # depends on another loop-carried symbol -> substitute that first (fixed point)
        if any(oe is not e and sym in (oe.data.assignments or {}) for oe in loop.all_interstate_edges()):
            continue  # sym written elsewhere (incl. a NESTED loop, s141) -> not a clean single definition
        side = _consistent_use_side(loop, e, sym)
        if side in ('after', 'unused'):
            body_expr = rhs_expr  # uses (if any) follow the definition -> this iteration's value
        elif side == 'before':
            # Every use precedes the definition, so it reads what the PREVIOUS iteration
            # wrote: ``f(loop_var - stride)``. That holds for iterations 2..N; iteration 1
            # instead reads the value ``sym`` carried INTO the loop, so the lagged form is
            # correct iff that entry value is exactly ``f(start - stride)``.
            start = loop_analysis.get_init_assignment(loop)
            stride = loop_analysis.get_loop_stride(loop)
            if start is None or stride is None:
                continue
            # The lagged form also assumes the definition runs on EVERY iteration; a
            # condition on any body top-level edge could skip it and stale the value.
            if any(oe.data.condition.as_string not in ('1', 'True', '(1)') for oe in loop.edges()):
                continue
            entry = _preloop_symbol_value(parent, loop, sym)
            if entry is None:
                continue  # entry value not provable -> cannot show the first iteration agrees
            loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
            if symbolic.simplify(entry - rhs_expr.subs(loop_var_sym, start - stride)) != 0:
                continue  # first iteration would read a value the lagged closed form mispredicts
            body_expr = symbolic.simplify(rhs_expr.subs(loop_var_sym, loop_var_sym - stride))
        else:
            continue  # uses straddle the definition -> per-block offsets needed (unsupported)

        e.data.assignments = {}
        loop.replace_dict({sym: symbolic.symstr(body_expr)})
        # Materialise the post-loop value (the last iteration's) for any reader
        # after the loop; harmless (dead) when ``sym`` is loop-local.
        end = loop_analysis.get_loop_end(loop)
        if end is not None:
            import dace
            post_val = symbolic.symstr(rhs_expr.subs(symbolic.pystr_to_symbolic(loop_var), end))
            dsym_post = parent.add_state(loop.label + '_dsym_post')
            for oe in list(parent.out_edges(loop)):
                parent.add_edge(dsym_post, oe.dst, oe.data)
                parent.remove_edge(oe)
            parent.add_edge(loop, dsym_post, dace.InterstateEdge(assignments={sym: post_val}))
        return True
    return False


def _try_substitute_iedge_iv(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG,
                             sdfg_free_symbols: Set[str]) -> bool:
    """Substitute an interstate-edge induction variable (``sym := sym + literal``)
    in the loop body with its closed form.

    Unlike :func:`_try_substitute` which eliminates the whole loop for a
    pure ``acc = acc OP const`` body, this preserves the loop and only
    removes the loop-carried dependency on the IV symbol. After the
    substitution the surviving loop body is no longer cross-iteration
    coupled through ``sym`` -- the canonical TSVC ``s122 / s125 / s126``
    shape::

        k = 1                               # pre-loop init (unchanged)
        for i in range(N):
            ...
            k = k + 1                       # iedge ``k := k + 1`` -- removed
            flat[k - 1] = ...               # ``k`` substituted to closed form

    After the rewrite the inner body references ``k + (loop_var - start + 1)``
    instead of ``k`` (where ``k`` evaluates to its pre-loop value), so
    ``flat[k - 1] = ...`` becomes ``flat[k + (loop_var - start) ...] = ...``
    -- a per-element write the downstream ``LoopToMap`` can lift. The
    symbol's post-loop value is materialised on the loop's exit edge so
    later readers see ``k + trip_count * step`` (matching the un-rewritten
    sequential semantics).

    Scope today:

    * stride ``== 1`` (matches the existing :func:`_try_substitute` scope);
    * exactly ONE iedge in the body carries an IV assignment ``sym := sym + step``
      (or ``sym := sym - step``), where ``step`` is a numeric literal OR a
      loop-invariant symbolic expression (e.g. a stride argument ``inc`` after
      scalar-to-symbol promotion);
    * the IV iedge has no other assignments and no condition;
    * the IV iedge is at the TOP (sourced from the empty ``loop.start_block`` --
      body is post-increment), at the BOTTOM (its destination is the body's
      unique, empty sink reached via a single in-edge -- body is pre-increment),
      or BETWEEN two non-empty content blocks, which is substitutable with a
      single offset iff every use of ``sym`` sits consistently on one side of the
      increment (see :func:`_consistent_use_side` and the ``side`` branch below;
      TSVC ``s128``);
    * no other iedge in the body writes ``sym`` (the IV is unique);
    * ``sym`` is an SDFG symbol / free symbol (not a data container).

    Neighbouring shapes, closed by the other halves of the pass' fixed point
    (see :meth:`InductionVariableSubstitution.apply_pass`):

    * derived IVs -- a symbol defined by a pure loop-variable expression
      (``j := a*i + b``, no self-reference) -- are folded by
      :func:`_try_substitute_derived_symbol`;
    * an increment that EVERY branch of a body conditional performs identically
      is first hoisted to a plain between-blocks iedge (which this function then
      closes) by :func:`_hoist_branch_uniform_iv` -- TSVC ``s124``.

    :param parent: CFG containing ``loop``.
    :param loop: Candidate ``LoopRegion``.
    :param sdfg: Owning SDFG.
    :returns: ``True`` if the substitution was applied; ``False`` if any
        pre-condition failed (no mutation in that case).
    """
    if not loop.loop_variable:
        return False
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return False

    # 1. Find the IV iedge: exactly one iedge in the body whose ONLY assignment
    #    is ``sym := sym + literal`` (or symmetric). Reject any iedge with a
    #    non-trivial condition.
    iv_candidate = None  # (edge, sym_name, step_sympy)
    for e in loop.edges():
        if e.data.condition.as_string not in ('1', 'True', '(1)'):
            return False
        if not e.data.assignments:
            continue
        if len(e.data.assignments) != 1:
            # An IV iedge here carries only the IV; other assignments would
            # need separate handling.
            continue
        ((lhs, rhs), ) = e.data.assignments.items()
        try:
            rhs_expr = symbolic.pystr_to_symbolic(rhs)
            lhs_sym = symbolic.pystr_to_symbolic(lhs)
            diff = symbolic.simplify(rhs_expr - lhs_sym)
        except Exception:
            # ``rhs`` may be a comparison (``StrictGreaterThan`` etc.) or
            # other non-arithmetic expression on which ``-`` raises
            # ``TypeError`` -- the assignment is not an arithmetic IV.
            continue
        # Step must be loop-invariant: a numeric literal, or a symbolic
        # expression whose free symbols are all loop-invariant (e.g. a stride
        # argument ``inc`` promoted to a symbol). A varying step has no closed form.
        if not getattr(diff, 'is_number', False):
            if not diff.free_symbols or not all(
                    _is_loop_invariant_symbol(str(s), loop, sdfg, sdfg_free_symbols) for s in diff.free_symbols):
                continue
        # ``lhs`` must be an SDFG symbol -- not a data container, not a loop var.
        if lhs == loop.loop_variable:
            continue
        if lhs not in sdfg.symbols and lhs not in sdfg_free_symbols:
            continue
        # No other body iedge may also write ``lhs``.
        other_writers = [oe for oe in loop.edges() if oe is not e and lhs in (oe.data.assignments or {})]
        if other_writers:
            continue
        if iv_candidate is not None:
            return False  # >1 IV pattern; defer to a future multi-IV extension
        iv_candidate = (e, lhs, diff)
    if iv_candidate is None:
        return False
    iv_edge, sym_name, step = iv_candidate

    # The IV symbol must be a counter PRIVATE to this loop. If it is also updated
    # in another (nested or enclosing) loop it is a shared counter, and a per-loop
    # closed form double-counts (TSVC s126 increments ``k`` in BOTH the inner and
    # outer loop). Refuse so such shared counters stay sequential.
    if _symbol_updated_in_other_loop(sdfg, loop, sym_name):
        return False

    # 2. Shape constraint: the IV iedge is at the TOP or the BOTTOM of the body.
    #    The closed form a body block sees depends on how many times this
    #    iteration's increment ran before it:
    #
    #    * TOP -- the iedge sources from the empty loop start block. Every other
    #      body block is reached AFTER the increment, so it sees this iter's
    #      increment too: ``sym = sym_init + (i - start + 1) * step``.
    #    * BOTTOM -- the iedge's destination is the body's unique, empty sink
    #      reached via a single in-edge (the increment is the last thing each
    #      iteration does, after all reads). Every body block is reached BEFORE
    #      the increment: ``sym = sym_init + (i - start) * step``.
    #
    #    (The frontend lowers ``for i: v = a[k]; ...; k += inc`` to the BOTTOM
    #    shape -- the gather reads the pre-increment ``k``; TSVC s318.)
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    sym_sym = symbolic.pystr_to_symbolic(sym_name)
    norm_iter = symbolic.simplify(loop_var_sym - start)

    src_is_empty_start = (iv_edge.src is loop.start_block and isinstance(iv_edge.src, SDFGState)
                          and not iv_edge.src.nodes())
    sinks = [b for b in loop.nodes() if loop.out_degree(b) == 0]
    dst_is_unique_empty_sink = (isinstance(iv_edge.dst, SDFGState) and not iv_edge.dst.nodes() and len(sinks) == 1
                                and sinks[0] is iv_edge.dst and loop.in_degree(iv_edge.dst) == 1)

    if src_is_empty_start:
        body_offset = norm_iter + 1  # update-at-top: body is post-increment
    elif dst_is_unique_empty_sink:
        body_offset = norm_iter  # update-at-bottom: body is pre-increment
    else:
        # The increment sits BETWEEN content blocks. It is still substitutable
        # with a single offset iff every use of ``sym`` is consistently on one
        # side of it -- TSVC s128, where the only ``j`` use (the ``k := j + 1``
        # iedge) precedes ``j := j + 2``. Substituting ``j`` then rewrites that
        # iedge to ``k := 2 * i``, which a fixed-point re-run (see the pass'
        # apply loop) / symbol propagation folds into the ``b[k]`` gathers.
        side = _consistent_use_side(loop, iv_edge, sym_name)
        if side == 'before':
            body_offset = norm_iter
        elif side == 'after':
            body_offset = norm_iter + 1
        elif side == 'unused':
            # Nothing in the body reads ``sym``, so every offset agrees on the (empty) set
            # of body reads -- the substitution is a body no-op and all this does is strip
            # the loop-carried increment. The post-loop value below is what the surviving
            # readers see, and it is independent of the offset chosen here.
            body_offset = norm_iter
        else:
            return False

    # 3. Build the closed form. The SDFG symbol ``sym`` evaluates to its current
    #    value (which IS ``sym_init`` once we strip the iedge increment), so the
    #    body substitution writes ``sym`` -> ``sym + body_offset * step``.
    post_iedge_expr = symbolic.simplify(sym_sym + body_offset * step)

    # 4. Substitute. The loop-level ``replace_dict`` walks every state +
    #    iedge in the body. We protect the IV iedge by clearing its
    #    assignment first (so the substitution doesn't try to rewrite the
    #    IV expression onto itself).
    iv_edge.data.assignments = {}

    # Substitute ``sym`` -> closed form throughout the loop body (every state
    # + every other iedge inside the loop). Memlet subsets, tasklet code,
    # iedge conditions and assignment RHSes all get rewritten.
    loop.replace_dict({sym_name: symbolic.symstr(post_iedge_expr)})

    # 5. Materialise the post-loop value so later readers (including the
    #    next iteration of an enclosing loop, when this one is nested) see
    #    ``sym + trip_count * step`` -- the value the un-rewritten
    #    sequential loop would leave behind. We always splice an empty
    #    "iv-post" state into ``parent`` immediately after ``loop`` and
    #    carry the ``sym := ...`` assignment on the iedge to it. This
    #    handles both shapes uniformly:
    #
    #    * ``loop`` has outgoing iedges -- they get rerouted to start from
    #      ``iv_post`` so any pre-existing exit assignments are preserved.
    #    * ``loop`` has no outgoing iedges (it is the only / last block of
    #      a containing loop body) -- the new ``iv_post`` becomes the
    #      next-block-after-loop inside the parent, ensuring the IV update
    #      runs once per containing-loop iteration before the body restarts.
    import dace
    trip_count = symbolic.simplify((end - start) // stride + 1)
    post_loop_value = symbolic.symstr(symbolic.simplify(sym_sym + trip_count * step))

    iv_post = parent.add_state(loop.label + '_iv_post')
    existing_out = list(parent.out_edges(loop))
    for oe in existing_out:
        parent.add_edge(iv_post, oe.dst, oe.data)
        parent.remove_edge(oe)
    iv_edge_out = dace.InterstateEdge(assignments={sym_name: post_loop_value})
    parent.add_edge(loop, iv_post, iv_edge_out)

    return True
