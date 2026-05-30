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
from typing import Optional, Tuple

from dace import SDFG, dtypes, nodes, properties, symbolic
from dace.sdfg import SDFGState
from dace.sdfg.state import ControlFlowRegion, LoopRegion
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
                and node.func.value.id == 'dace' and node.func.attr in self._TYPECAST_NAMES
                and len(node.args) == 1 and not node.keywords):
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
        count = 0
        for node, parent in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, LoopRegion):
                continue
            if _try_substitute(parent, node, sdfg):
                count += 1
                continue
            # Multi-statement bodies where one of the statements is a scalar IV
            # recurrence on an interstate edge (``sym := sym + literal``) -- the
            # canonical TSVC counter-IV shape (s122/s125/s126: ``k = k + 1``
            # alongside ``flat[k] = ...``). Substitute the body's reads of
            # ``sym`` with the closed form so the cross-iter dependency is gone
            # and the surviving loop is parallelisable. Unlike ``_try_substitute``
            # (which eliminates the loop), this preserves it.
            if _try_substitute_iedge_iv(parent, node, sdfg):
                count += 1
        return count or None


def _try_substitute(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG) -> bool:
    """Return True if the loop matched and was replaced; False otherwise."""
    info = _extract_iv(loop, sdfg)
    if info is None:
        return False
    accum, accum_subset, op_type, const_val, trip_count = info

    # The closed-form RHS reads the seed via the tasklet's ``__in`` connector,
    # NOT via a bare ``accum[subset]`` expression -- the SDFG dataflow is what
    # actually wires the read.
    closed = _CLOSED_FORM[op_type]("__in", const_val, symbolic.symstr(trip_count))

    _replace_loop_with_closed_form(parent, loop, accum, accum_subset, closed, sdfg)
    return True


def _is_loop_invariant_symbol(name: str, loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``name`` refers to an SDFG symbol/constant that the loop does not
    redefine in its body (so its value is stable across iterations).

    Accepts: SDFG symbols and constants, with the loop variable explicitly excluded,
    and with the symbol not appearing as the LHS of any interstate-edge assignment
    inside the loop's body.
    """
    if name == loop.loop_variable:
        return False
    if name not in sdfg.symbols and name not in sdfg.constants and name not in sdfg.free_symbols:
        return False
    # The loop must not assign to ``name`` on any of its body interstate edges --
    # otherwise it would not be loop-invariant.
    for e in loop.edges():
        if e.data.assignments and name in e.data.assignments:
            return False
    return True


def _extract_iv(loop: LoopRegion, sdfg: SDFG) -> Optional[Tuple[str, str, type, object, object]]:
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
                if (sub.id not in sdfg.symbols and sub.id not in sdfg.constants
                        and sub.id not in sdfg.free_symbols and sub.id != 'dace'
                        and not hasattr(__import__('builtins'), sub.id)):
                    return None
                if (sub.id in sdfg.symbols or sub.id in sdfg.constants
                        or sub.id in sdfg.free_symbols) and not _is_loop_invariant_symbol(sub.id, loop, sdfg):
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
    import dace
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


def _try_substitute_iedge_iv(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG) -> bool:
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

    Scope today (kept narrow for v1):

    * stride ``== 1`` (matches the existing :func:`_try_substitute` scope);
    * exactly ONE iedge in the body carries an IV assignment of the form
      ``sym := sym + literal`` (or ``sym := sym - literal``);
    * the IV iedge has no other assignments and no condition;
    * the IV iedge is sourced from ``loop.start_block``, and the source
      block is empty (an iedge-only state) so every other body block is
      reachable AFTER the iedge -- substituting every body block with the
      single ``sym + (loop_var - start + 1) * step`` closed form is sound;
    * no other iedge in the body writes ``sym`` (the IV is unique);
    * ``sym`` is an SDFG symbol / free symbol (not a data container).

    Out of scope (potential follow-ups):

    * IV iedge between two non-empty content blocks (would need pre-iedge
      vs post-iedge block partitioning + two different closed-form
      substitutions);
    * derived IVs (multiple symbols related by ``j := a*i + b``);
    * multi-step IV (``sym := sym + sym2`` where ``sym2`` is a loop-
      invariant non-literal -- the closed form is the same up to
      symbolic-rendering).

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
        # Step must be a number constant (literal). Symbolic step is a follow-up.
        if not getattr(diff, 'is_number', False):
            continue
        # ``lhs`` must be an SDFG symbol -- not a data container, not a loop var.
        if lhs == loop.loop_variable:
            continue
        if lhs not in sdfg.symbols and lhs not in sdfg.free_symbols:
            continue
        # No other body iedge may also write ``lhs``.
        other_writers = [oe for oe in loop.edges()
                         if oe is not e and lhs in (oe.data.assignments or {})]
        if other_writers:
            continue
        if iv_candidate is not None:
            return False  # >1 IV pattern; defer to a future multi-IV extension
        iv_candidate = (e, lhs, diff)
    if iv_candidate is None:
        return False
    iv_edge, sym_name, step = iv_candidate

    # 2. v1 shape constraint: the IV iedge sources from loop.start_block, and
    #    the start block is empty (an iedge-only state). Every other body
    #    block is then post-iedge and substituting the whole body once with
    #    the post-iedge closed form is sound.
    if iv_edge.src is not loop.start_block:
        return False
    if not isinstance(iv_edge.src, SDFGState):
        return False
    if iv_edge.src.nodes():
        # start block has content -- pre-iedge reads of ``sym`` would need a
        # different (pre-step) closed form; defer.
        return False

    # 3. Build the closed form. At ``loop_var = i`` (with ``stride == 1``):
    #    - At iv_edge.src (the empty start block): sym = sym_init + (i - start) * step.
    #    - At iv_edge.dst and post-iedge blocks in the same iter:
    #      sym = sym_init + (i - start + 1) * step.
    #    Since the SDFG symbol ``sym`` evaluates to its current value
    #    (which IS ``sym_init`` once we strip the iedge increment), the
    #    post-iedge substitution writes ``sym`` -> ``sym + ((i - start) + 1) * step``.
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    sym_sym = symbolic.pystr_to_symbolic(sym_name)
    norm_iter = symbolic.simplify(loop_var_sym - start)
    post_iedge_expr = symbolic.simplify(sym_sym + (norm_iter + 1) * step)

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
