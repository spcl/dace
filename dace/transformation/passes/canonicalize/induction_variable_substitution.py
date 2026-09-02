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
* any non-zero stride, symbolic included -- iteration ``i`` is trip
  ``t = int_floor(i - start, stride)`` and the trip count is ``t(end) + 1``.

A multi-statement body whose IV is READ by the other statements cannot be split
off (the reader needs the IV's per-iteration value, so it is not an independent
component ``HoistInductionVariableUpdates`` could fission). It is instead handled
by USE-SITE substitution -- LLVM's SCEV expansion / strength reduction: expand the
IV's closed form *at every read* as a function of the loop variable, then delete
the recurrence. TSVC ``s453`` (``s = s + 2.0; a[i] = s * b[i]``) becomes the pure
per-element map ``a[i] = (s_entry + 2.0*(i+1)) * b[i]``. See
:func:`try_substitute_use_site_iv`.

Out of scope for this implementation (potential follow-ups):

* derived IVs (``j = a*i + b`` from a basic IV ``i``) -- needs the IV graph
  from Ch. 9.6, then strength-reduction or closed-form substitution;
* loop-invariant non-literal operand (``s = s * k[0]`` with ``k[0]`` invariant)
  -- straightforward extension to the constant-detection branch;
* use-site substitution across a MULTI-STATE body -- the "which side of the
  update does this read sit on" question is answered by dataflow inside one
  state; across states it needs the reachability split
  :func:`_consistent_use_side` does for symbol IVs;
* use-site substitution when the accumulator is read through a STAGING transient
  (``acc -> acc_index -> tasklet``, what the frontend emits for a non-transient
  ``a[0]`` accumulator and what the earlier pipeline stages strip for a local
  one) -- looking through the staging is easy, but the staged copy would then
  hold the entry value instead of the updated one, so it also needs a proof that
  nothing outside this state reads it;
* non-commutative WCR ops -- our substitution assumes commutativity, which
  holds for ``+`` and ``*`` but would not for general WCR.

Eliminating the loop turns an O(N) recurrence into O(1) straight-line code,
and runs BEFORE LoopToReduce / LoopToMap so the IV-eligible loop never gets
mis-classified as a fold or a parallel map. The TSVC kernel ``s317``
(``q[0] *= 0.99`` for ``LEN_1D//2`` iters) is the canonical hit.
"""
import ast
import copy
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from dace import SDFG, dtypes, nodes, properties, subsets, symbolic
from dace.frontend.python import astutils
from dace.sdfg import SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion

#: Builtin names the closed-form expression may mention; it is spliced verbatim into a tasklet
#: body. Probing ``builtins`` instead would admit ``open``, ``id``, ``sum``, ... as valid operands.
SPLICEABLE_BUILTINS = dict.fromkeys(['True', 'False', 'None', 'abs', 'min', 'max', 'int', 'float'])
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
    _TYPECAST_NAMES = dict.fromkeys(_dtypes.TYPECLASS_STRINGS)

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
                # (2) a DATA accumulator that the rest of the body READS -> expand its
                #     closed form at every use site and drop the recurrence (s453);
                # (3) an interstate-edge recurrence ``sym := sym + step`` -> closed
                #     form in the body (keeps the loop; the counter-IV shape);
                # (4) a symbol defined purely by a loop-var expression -> inline it
                #     (the derived symbol a primary-IV substitution just freed);
                # (5) an IV incremented identically in every branch of a body
                #     conditional -> hoist it out so (3) can then close it (s124).
                if (_try_substitute(parent, node, sdfg, sdfg_free_symbols)
                        or try_substitute_use_site_iv(parent, node, sdfg, sdfg_free_symbols)
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


def step_is_loop_invariant(step_names, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: Set[str]) -> bool:
    """Whether a candidate IV STEP, built from ``step_names``, is loop-invariant.

    THIS is the test that separates an induction variable from a REDUCTION, and it is the only
    thing that does: ``k = k + 1`` and ``sum = sum + a[i]`` are the same shape to a structural
    matcher. A step that is a literal, or an expression over enclosing symbols alone, has the
    closed form ``k0 + step * trip``. A DATA-DEPENDENT step (``a[i]``, ``a[i] * b[ip[i]]``) reads
    memory that changes every iteration, so no closed form exists; that shape is a reduction and
    belongs to ``LoopToReduce`` / ``LoopToScan``. Substituting a closed form for one would be a
    silent miscompile, not a missed optimization -- so this predicate must stay a WHITELIST.

    Data dependence is excluded structurally rather than sniffed for: such a step reaches a
    tasklet through a second input CONNECTOR, and a connector name is never an SDFG symbol, so it
    fails :func:`_is_loop_invariant_symbol` on the membership test. TSVC ``s3112`` / ``s4115``
    are the guarded cases.

    A step that varies with the LOOP VARIABLE is refused here too, and deliberately stays refused
    (user ruling 2026-09-01). TSVC ``s141`` (``k = k + j + 1``, ``j`` the inner loop variable) does
    have a closed form -- sum an arithmetic series -- but it is QUADRATIC in ``j``, and the point
    of closing an IV in canonicalization is to hand downstream an AFFINE subscript for semantic
    lifting and the SMT contract checks. A quadratic subscript buys none of that, and ``s141``
    already parallelizes fully with ``k`` surviving only as a symbol, so the closed-form engine is
    not extended to reach it.

    :param step_names: the names occurring in the step expression.
    """
    return all(_is_loop_invariant_symbol(str(s), loop, sdfg, sdfg_free_symbols) for s in step_names)


class TaskletIV(NamedTuple):
    """One tasklet's ``__out = __in OP const`` induction-variable shape.

    ``const_val`` may be a Python ``int`` / ``float`` (numeric literal) OR a string
    naming a loop-invariant SDFG symbol -- in the symbolic case the closed form
    ``init + c*N`` / ``init * c**N`` keeps ``c`` as a name and is materialised in
    the post-loop tasklet by the codegen's symbol-binding path.
    """
    accum: str  #: container the update eventually writes (after any staging transients)
    subset: str  #: the single-element, loop-invariant accumulator slot
    op_type: type  #: ``ast.Add`` / ``ast.Mult``
    const_val: Any  #: numeric literal, or a source string over loop-invariant symbols
    in_edge: Any  #: the carried read edge into the tasklet (``in_edge.dst`` is the tasklet itself)
    write_edge: Any  #: the tasklet's update write edge
    reads_accum: bool  #: the carried read traces back to ``accum[subset]`` (a true recurrence)


def _extract_iv(loop: LoopRegion, sdfg: SDFG,
                sdfg_free_symbols: Set[str]) -> Optional[Tuple[str, str, type, object, object]]:
    """Pattern-match a SINGLE-tasklet loop body. Returns
    ``(accum_name, accum_subset_str, ast.BinOp_type, const_val, trip_count)`` or ``None``.
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
    if tasklet is None:
        return None
    iv = extract_tasklet_iv(tasklet, state, loop, sdfg, sdfg_free_symbols)
    if iv is None or not iv.reads_accum:
        return None

    # Trip count = (end - start) // stride + 1 (loop_analysis.get_loop_end is inclusive).
    trip_count = symbolic.simplify(symbolic.int_floor(end - start, stride) + 1)
    return iv.accum, iv.subset, iv.op_type, iv.const_val, trip_count


def extract_tasklet_iv(tasklet: nodes.Tasklet, state: SDFGState, loop: LoopRegion, sdfg: SDFG,
                       sdfg_free_symbols: Set[str]) -> Optional[TaskletIV]:
    """Match ONE tasklet against the ``__out = __in OP const`` IV shape, ignoring its siblings.

    Split out of :func:`_extract_iv` so a MULTI-tasklet body (where the IV is one statement
    among several) can be matched statement by statement -- see :func:`try_substitute_use_site_iv`.
    """
    if tasklet.code.language != dtypes.Language.Python:
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
    in_connector_names = dict.fromkeys(tasklet.in_connectors.keys())

    def _is_carrier(node):
        return isinstance(node, ast.Name) and node.id in in_connector_names

    # EXACTLY one side may be a connector. Two connectors is ``__out = __in1 + __in2``, where the
    # second one carries a per-iteration VALUE -- the reduction shape (``sum = sum + a[i]``,
    # TSVC s3112 / s4115). It has no closed form, so it is refused here rather than approximated;
    # reduction lifting claims it later. This is the dataflow half of ``step_is_loop_invariant``.
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
            if not isinstance(sub, ast.Name) or sub.id == 'dace' or sub.id in SPLICEABLE_BUILTINS:
                continue  # a dtype-cast root, not an operand
            # Every remaining name must be a loop-invariant symbol. A CONNECTOR name reaches here
            # and fails, which is what refuses the reduction shape ``sum = sum + a[i]``.
            if not step_is_loop_invariant([sub.id], loop, sdfg, sdfg_free_symbols):
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
    # A WCR store is ``acc = wcr(acc, rhs)``, not ``acc = rhs``, so it is not the recurrence the
    # closed form below solves.
    if in_edge.data.wcr is not None or write_edge.data.wcr is not None:
        return None
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

    # Trace the input back to the same accumulator slot: only then is the tasklet a
    # RECURRENCE (``acc = acc OP c``) rather than an unrelated read that happens to add.
    src = in_edge.src
    if not isinstance(src, nodes.AccessNode):
        return None
    desc = sdfg.arrays.get(src.data)
    if desc is None:
        return None
    src_name, src_subset = src.data, in_edge.data.subset
    # ``is_empty()`` FIRST: an empty in-edge is an ORDERING edge (the frontend hangs one off the
    # accumulator to sequence a WAR), never the staging copy that would redirect the trace.
    staged = [e for e in state.in_edges(src) if e.data is not None and not e.data.is_empty()]
    if desc.transient and len(staged) == 1:
        pred = staged[0]
        if not isinstance(pred.src, nodes.AccessNode) or pred.data.subset is None:
            return None
        src_name, src_subset = pred.src.data, pred.data.subset
    reads_accum = src_name == final_accum and str(src_subset) == str(final_subset)

    return TaskletIV(final_accum, str(final_subset), type(rhs.op), const_val, in_edge, write_edge, reads_accum)


def closed_form_state(parent: ControlFlowRegion,
                      label: str,
                      accum_name: str,
                      accum_subset: str,
                      closed_form: str,
                      is_start_block: bool = False) -> SDFGState:
    """A fresh state whose single tasklet writes ``closed_form`` back to ``accum_name[accum_subset]``.

    The closed-form RHS reads the seed via the tasklet's ``__in`` connector, NOT via a bare
    ``accum[subset]`` expression -- the SDFG dataflow is what actually wires the read.
    """
    from dace import memlet as mm

    new_state = parent.add_state(label, is_start_block=is_start_block)
    accum_r = new_state.add_read(accum_name)
    accum_w = new_state.add_write(accum_name)
    # add_tasklet turns a connector set into the in/out-connector dict verbatim -- a plain set
    # literal's hash order would become the emitted connector declaration order. Single-element
    # here so it can't actually reorder, but dict.fromkeys matches the pattern used elsewhere.
    tasklet = new_state.add_tasklet(label + "_tlt", dict.fromkeys(['__in']), dict.fromkeys(['__out']),
                                    f"__out = {closed_form}")
    new_state.add_edge(accum_r, None, tasklet, "__in", mm.Memlet(data=accum_name, subset=accum_subset))
    new_state.add_edge(tasklet, "__out", accum_w, None, mm.Memlet(data=accum_name, subset=accum_subset))
    return new_state


def _replace_loop_with_closed_form(parent: ControlFlowRegion, loop: LoopRegion, accum_name: str, accum_subset: str,
                                   closed_form: str, sdfg: SDFG) -> None:
    """Swap ``loop`` for a state whose tasklet writes the closed form back to ``accum_name[accum_subset]``."""
    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))

    new_state = closed_form_state(parent, loop.label + "_iv_closed", accum_name, accum_subset, closed_form, was_start)

    for e in in_edges:
        parent.add_edge(e.src, new_state, e.data)
    for e in out_edges:
        parent.add_edge(new_state, e.dst, e.data)
    parent.remove_node(loop)


# -----------------------------------------------------------------------------
# Use-site closed-form substitution (SCEV expansion) for a DATA accumulator
# -----------------------------------------------------------------------------


def trip_index(loop: LoopRegion, start, stride):
    """Trip index ``t`` of the iteration running with ``loop_variable``: ``loop_var == start + t*stride``.

    Every visited ``loop_var`` makes ``loop_var - start`` an EXACT multiple of ``stride``, so
    ``int_floor`` here never actually rounds -- it is the integer-typed spelling of the division
    (DaCe index arithmetic uses ``symbolic.int_floor``, never ``/`` or ``//``).
    """
    return symbolic.simplify(symbolic.int_floor(symbolic.pystr_to_symbolic(loop.loop_variable) - start, stride))


class ReplaceConnectorWithClosedForm(ast.NodeTransformer):
    """Splice ``closed_form`` in for every READ of input connector ``conn`` in a tasklet body.

    ``closed_form`` mentions ``conn`` itself (it is ``(conn) + c*t`` / ``(conn) * c**t``), but
    ``NodeTransformer`` does not re-visit what ``visit_Name`` returns, so there is no recursion.
    """

    def __init__(self, conn: str, closed_form: str) -> None:
        self.conn = conn
        self.closed_form = closed_form

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.conn or not isinstance(node.ctx, ast.Load):
            return node
        return ast.parse(self.closed_form, mode='eval').body


def splice_closed_form(tasklet: nodes.Tasklet, conn: str, closed_form: str) -> None:
    """Rewrite ``tasklet`` so every read of ``conn`` evaluates the IV's closed form instead."""
    tree = ReplaceConnectorWithClosedForm(conn, closed_form).visit(ast.parse((tasklet.code.as_string or "").strip()))
    ast.fix_missing_locations(tree)
    tasklet.code = properties.CodeBlock(astutils.unparse(tree), dtypes.Language.Python)


class UseSitePlan(NamedTuple):
    """The validated rewrite for one data-accumulator IV inside a multi-statement body."""
    pre_node: nodes.AccessNode  #: accumulator version holding the value on entry to the iteration
    post_node: nodes.AccessNode  #: accumulator version the IV update writes
    chain: List[nodes.AccessNode]  #: staging transients between the IV tasklet and ``post_node``
    pre_reads: List[Any]  #: edges whose consumer reads the PRE-update value (``t`` updates applied)
    post_reads: List[Any]  #: edges whose consumer reads the POST-update value (``t + 1`` applied)


def plan_use_site_substitution(state: SDFGState, sdfg: SDFG, tasklet: nodes.Tasklet,
                               iv: TaskletIV) -> Optional[UseSitePlan]:
    """Validate that ``iv``'s recurrence can be expanded at its use sites, and enumerate them.

    Refuses -- WITHOUT touching the state -- unless the accumulator slot is written EXACTLY ONCE
    in this state, by ``iv``'s update. Every other version of the slot is then the value on entry
    to the iteration, and "which value does this read see" is answerable by dataflow alone --
    which is what makes the ``t`` vs ``t + 1`` offset provable rather than guessed.
    """
    pre_node = iv.in_edge.src
    if not isinstance(pre_node, nodes.AccessNode) or pre_node.data != iv.accum:
        return None  # the carried read is staged through a transient -> entry value not directly readable
    if iv.in_edge.data.data != iv.accum or str(iv.in_edge.data.subset) != iv.subset:
        return None

    # Walk the update's write forward through staging transients to the accumulator version it lands on.
    chain: List[nodes.AccessNode] = []
    cur = iv.write_edge.dst
    while isinstance(cur, nodes.AccessNode) and cur.data != iv.accum:
        desc = sdfg.arrays.get(cur.data)
        if desc is None or not desc.transient or state.in_degree(cur) != 1 or state.out_degree(cur) != 1:
            return None
        chain.append(cur)
        cur = state.out_edges(cur)[0].dst
    if not isinstance(cur, nodes.AccessNode) or cur.data != iv.accum or cur is pre_node:
        return None
    post_node = cur

    # An empty memlet is an ORDERING edge, not a write, so split on ``is_empty()`` before reading
    # ``.data`` / ``.subset``. The frontend hangs ordering edges around the update for the
    # read-before-update body (``a[i] = s*b[i]; s += 2``), to sequence the WAR.
    def written_by(node: nodes.AccessNode) -> List[Any]:
        return [e for e in state.in_edges(node) if e.data is not None and not e.data.is_empty()]

    versions = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == iv.accum]
    if [n for n in versions if written_by(n)] != [post_node]:
        return None  # a second write to the slot -> the value a read sees is no longer this recurrence
    (write_in, ) = written_by(post_node)
    if write_in.data.wcr is not None:  # an accumulation into the slot, not this recurrence's store
        return None
    written = write_in.data.subset if write_in.data.data == iv.accum else write_in.data.dst_subset
    if written is None or str(written) != iv.subset:
        return None
    # Every unwritten version holds the ENTRY value -- unless the update can reach it, in which case
    # its reads observe the post-update value and calling them "pre" would be off by one step.
    entry_versions = [n for n in versions if n is not post_node]
    reachable_from_update = dict.fromkeys(sdutil.dfs_conditional(state, sources=[post_node]))
    if any(n in reachable_from_update for n in entry_versions):
        return None

    pre_reads: List[Any] = []
    post_reads: List[Any] = []
    for node in entry_versions + [post_node]:
        bucket = post_reads if node is post_node else pre_reads
        for e in state.out_edges(node):
            if e is iv.in_edge:
                continue  # the IV update's own carried read, which the rewrite deletes
            # An ordering edge out of a version we are about to delete would lose the order it enforces.
            if e.data is None or e.data.is_empty():
                return None
            if not isinstance(e.dst, nodes.Tasklet) or e.dst.code.language != dtypes.Language.Python:
                return None
            if e.dst_conn is None or e.data.data != iv.accum or str(e.data.subset) != iv.subset:
                return None
            try:
                ast.parse((e.dst.code.as_string or "").strip())
            except SyntaxError:
                return None  # unparseable consumer -> cannot splice the closed form into it
            bucket.append(e)
    # Ordering edges INTO the update are vacuous once the write is gone -- but only while nothing
    # reads the post-update version, else deleting the node also drops "predecessor before reader".
    if post_reads and len(state.in_edges(post_node)) != 1:
        return None
    return UseSitePlan(pre_node, post_node, chain, pre_reads, post_reads)


def try_substitute_use_site_iv(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG,
                               sdfg_free_symbols: Set[str]) -> bool:
    """Expand a DATA accumulator's closed form at every use site, then delete the recurrence.

    :func:`_try_substitute` collapses a loop whose ONLY statement is ``acc = acc OP c``. When the
    body also *reads* ``acc`` -- TSVC ``s453``::

        s = 0.0
        for i in range(N):
            s = s + 2.0
            a[i] = s * b[i]

    -- the update is neither eliminable (the reads need its per-iteration value) nor fissionable
    (``HoistInductionVariableUpdates`` needs an ISOLATED component, and here ``s`` is read by the
    other statement). What IS available is the closed form *at iteration i*: after ``t + 1``
    updates ``s == s_entry + 2.0*(t + 1)``, so the body becomes the per-element
    ``a[i] = (s_entry + 2.0*(i + 1)) * b[i]`` -- LLVM's SCEV expansion / strength reduction.
    With the recurrence gone the loop is a pure map.

    Which offset a read gets is decided by DATAFLOW, not by source order, and getting it wrong
    silently shifts every value by one step:

    * a read of the version the update WROTE sees ``t + 1`` updates (``s453``: the multiply
      consumes the updated ``s``);
    * a read of the entry version (no writer) sees ``t`` updates -- the mirrored
      ``a[i] = s * b[i]; s = s + 2.0`` body, whose first iteration must still read ``s_entry``.

    After the rewrite the loop no longer writes the accumulator, so the entry version holds
    ``s_entry`` for the whole loop and the spliced closed forms read it directly. The value the
    sequential loop would have left behind is materialised in a post-loop state.

    Scope: single-``SDFGState`` body (dataflow answers the offset question exactly; across states
    it would need the reachability split :func:`_consistent_use_side` does), no Map / NestedSDFG
    scope in it, and the slot written exactly once. Everything else refuses without mutating.
    """
    if not loop.loop_variable:
        return False
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride == 0:
        return False

    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], SDFGState):
        return False
    state = blocks[0]
    tasklets = []
    for n in state.nodes():
        if isinstance(n, nodes.Tasklet):
            tasklets.append(n)
        elif not isinstance(n, nodes.AccessNode):
            return False  # a Map / NestedSDFG scope hides the per-iteration execution order
    if len(tasklets) < 2:
        return False  # a single-statement body is _try_substitute's whole-loop collapse instead

    for tasklet in tasklets:
        iv = extract_tasklet_iv(tasklet, state, loop, sdfg, sdfg_free_symbols)
        if iv is None or not iv.reads_accum:
            continue
        plan = plan_use_site_substitution(state, sdfg, tasklet, iv)
        if plan is None:
            continue
        apply_use_site_substitution(parent, loop, state, iv, plan, start, end, stride)
        return True
    return False


def apply_use_site_substitution(parent: ControlFlowRegion, loop: LoopRegion, state: SDFGState, iv: TaskletIV,
                                plan: UseSitePlan, start, end, stride) -> None:
    """Commit the rewrite :func:`plan_use_site_substitution` validated."""
    import dace
    from dace import memlet as mm

    t = trip_index(loop, start, stride)
    # Splice first (the edges still name their consumer + connector), then re-point.
    for reads, applied in ((plan.pre_reads, t), (plan.post_reads, symbolic.simplify(t + 1))):
        count = symbolic.symstr(applied)
        for e in reads:
            splice_closed_form(e.dst, e.dst_conn, _CLOSED_FORM[iv.op_type](e.dst_conn, iv.const_val, count))
    # Post-update readers now compute from the entry value, so they read the entry version. Remove
    # before adding so the consumer's input connector is never momentarily fed by two edges.
    for e in plan.post_reads:
        state.remove_edge(e)
        state.add_edge(plan.pre_node, e.src_conn, e.dst, e.dst_conn, mm.Memlet(data=iv.accum, subset=iv.subset))

    state.remove_node(iv.in_edge.dst)  # the IV tasklet
    for n in plan.chain:
        state.remove_node(n)
    state.remove_node(plan.post_node)
    # Drop every version of the slot nothing reads any more, along with the ordering edges that
    # only sequenced it against the write we just deleted.
    for n in [v for v in state.nodes() if isinstance(v, nodes.AccessNode) and v.data == iv.accum]:
        if not [e for e in state.out_edges(n) if e.data is not None and not e.data.is_empty()]:
            state.remove_node(n)

    # The loop no longer updates the accumulator; materialise the value it used to leave behind.
    trip_count = symbolic.simplify(symbolic.int_floor(end - start, stride) + 1)
    closed = _CLOSED_FORM[iv.op_type]("__in", iv.const_val, symbolic.symstr(trip_count))
    iv_post = closed_form_state(parent, loop.label + '_iv_use_post', iv.accum, iv.subset, closed)
    for oe in list(parent.out_edges(loop)):
        parent.add_edge(iv_post, oe.dst, oe.data)
        parent.remove_edge(oe)
    parent.add_edge(loop, iv_post, dace.InterstateEdge())


# -----------------------------------------------------------------------------
# Iedge-based IV substitution (multi-statement bodies)
# -----------------------------------------------------------------------------


def _symbol_updated_in_other_loop(sdfg: SDFG, loop: LoopRegion, sym_name: str) -> bool:
    """Whether ``sym_name`` is also stepped by a loop NESTED INSIDE ``loop``.

    Such a counter has no per-loop closed form here: ``loop``'s own step is not the only thing that
    happens to it per iteration, so ``sym + trip * step`` under-counts by whatever the inner loop
    added.

    A step in an ENCLOSING or a SIBLING loop is a different matter and is allowed. The substitution
    reads ``sym`` as the value live at THIS loop's entry and materialises ``sym + trips * step`` on
    the way out, so the outer picture is preserved whatever else steps the symbol elsewhere. That is
    what unwinds a two-level counter (TSVC ``s126``: ``k += 1`` in the inner loop and once more per
    outer iteration): closing the inner loop turns its whole contribution into one exit assignment,
    which leaves the outer loop with a single step per iteration for the next round of the pass'
    fixed point to close in turn.

    :param sdfg: The SDFG owning ``loop``.
    :param loop: The loop whose counter is being closed.
    :param sym_name: The counter symbol.
    """
    inner = [r for r in loop.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r is not loop]
    return any(sym_name in (e.data.assignments or {}) for region in inner for e in region.edges())


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
                    if delta.is_number:
                        incs.setdefault(lhs, []).append((e, delta))
            return incs

        per = [branch_increments(br) for br in branches]
        common = set.intersection(*[set(p) for p in per]) if per else set()
        # sorted(): this loop RETURNS on the first symbol it hoists, and the pass is a first-match/restart
        # fixpoint -- so with two co-incrementing IVs (s124/s126/s128) the pick decides whether the OTHER one
        # still passes its guards afterwards, i.e. whether the loop reaches closed form and becomes a Map.
        # Iterating the raw set made that a PYTHONHASHSEED coin-flip (str hashing is per-process randomized).
        # ``set.intersection`` has no meaningful insertion order to preserve, so a stable sort -- not an
        # ordered set -- is the canonical, branch-independent choice.
        for sym in sorted(common):
            if sym == loop.loop_variable or (sym not in sdfg.symbols and sym not in sdfg_free_symbols):
                continue
            if any(len(p[sym]) != 1 for p in per):
                continue  # a branch increments sym more than once
            steps = dict.fromkeys(p[sym][0][1] for p in per)
            if len(steps) != 1:
                continue  # branches disagree on the step
            step = next(iter(steps))
            branch_edges = dict.fromkeys(id(p[sym][0][0]) for p in per)
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
            sides = dict.fromkeys(_consistent_use_side(br, p[sym][0][0], sym) for br, p in zip(branches, per))
            sides.pop('unused', None)
            if not sides:
                side = 'after'
            elif len(sides) == 1 and next(iter(sides)) in ('after', 'before'):
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
            # Ask BEFORE adding the hoist state: the fresh state is a second, isolated source
            # node, which makes ``start_block`` ambiguous until it is wired in below.
            was_start = loop.start_block is cb
            hoist = loop.add_state(cb.label + '_iv_hoist')
            new_rhs = symbolic.symstr(symbolic.pystr_to_symbolic(sym) + step)
            if side == 'after':
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
    before = dict.fromkeys(sdutil.dfs_conditional(loop, sources=[iv_edge.src], reverse=True))
    before[iv_edge.src] = None
    after = dict.fromkeys(sdutil.dfs_conditional(loop, sources=[iv_edge.dst]))
    after[iv_edge.dst] = None

    saw_before = saw_after = False
    # Block uses: any body block that reads ``sym`` -- a state (memlet / tasklet)
    # OR a nested region (a ConditionalBlock whose branches read ``sym``, e.g. the
    # ``a[j]`` writes in s124). ``free_symbols`` is defined for every block type.
    for b in loop.nodes():
        if sym_name in (str(s) for s in b.free_symbols):
            if b in before:
                saw_before = True
            elif b in after:
                saw_after = True
            else:
                return None  # neither strictly before nor after the increment
    # Interstate-edge uses (RHS assignments / condition), excluding the IV edge.
    for e in loop.edges():
        if e is iv_edge or sym_name not in (str(s) for s in e.data.free_symbols):
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
    vals: dict = {}
    for e in in_edges:
        rhs = (e.data.assignments or {}).get(sym_name)
        if rhs is None:
            return None
        try:
            vals[symbolic.pystr_to_symbolic(rhs)] = None
        except Exception:
            return None
    if len(vals) != 1:
        return None
    (val, ) = vals
    if loop.loop_variable in (str(s) for s in val.free_symbols):
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
        # An array-dependent value is a data gather, NOT an induction variable: its
        # per-iteration value is read from memory, not a closed form of the loop var.
        # Inlining it would bake the array read into every memlet subset that uses
        # ``sym`` -- a nested ``Subscript`` codegen can't lower (it emits
        # ``arr[std::make_tuple(...)]``). The frontend already keeps such indirection
        # as its own symbol (cloudsc ``LLINDEX1(JL,IORDER(JL,JM))`` -> the interstate
        # ``iorder_at = IORDER(JL,JM)`` load); leave it a symbol, don't dissolve it.
        if symbolic.arrays(rhs):
            continue
        free = dict.fromkeys(str(s) for s in rhs_expr.free_symbols)
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


def staged_iedge_rhs(rhs: str, src_state, sdfg: SDFG):
    """The arithmetic an interstate assignment's RHS stands for when it is staged through a
    transient, or ``None`` when ``rhs`` is not such a staging and should be read as written.

    The frontend does not always leave a bare ``k := k + inc`` on the edge. When the step is a
    promoted scalar argument it emits the arithmetic as a TASKLET writing a transient scalar and
    binds the symbol to that slot -- ``k := k_plus_inc`` with ``k_plus_inc = k + dace.int64(inc)``
    (TSVC ``s318``) -- so the value the edge carries sits one dataflow hop away and the edge itself
    names nothing but a buffer. Resolve that hop when ``rhs`` is exactly such a slot: a transient
    scalar written once, in the edge's own source state, by a single-statement tasklet with no data
    inputs. No inputs means every operand is already a symbol, so the tasklet body IS the
    expression; a data input would make it a runtime value with no symbolic form.
    """
    desc = sdfg.arrays.get(rhs)
    if desc is None or not desc.transient or desc.total_size != 1 or not isinstance(src_state, SDFGState):
        return None
    writes = [n for n in src_state.data_nodes() if n.data == rhs and src_state.in_degree(n) > 0]
    if len(writes) != 1 or src_state.in_degree(writes[0]) != 1:
        return None
    in_edge = src_state.in_edges(writes[0])[0]
    tasklet = in_edge.src
    if (not isinstance(tasklet, nodes.Tasklet) or src_state.in_degree(tasklet) > 0 or src_state.out_degree(tasklet) != 1
            or tasklet.has_side_effects(sdfg)):
        return None
    code = tasklet.code
    if code.language is not dtypes.Language.Python or len(code.code) != 1:
        return None
    stmt = code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if astutils.rname(stmt.targets[0]) != in_edge.src_conn:
        return None
    value = _UnwrapTypecasts().visit(copy.deepcopy(stmt.value))
    try:
        return symbolic.pystr_to_symbolic(astutils.unparse(value))
    except Exception:
        # A body that is not an arithmetic expression (a call, a comparison) carries no value
        # this can stand in for.
        return None


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

    * any non-zero stride, symbolic included: iteration ``i`` is trip
      ``t = int_floor(i - start, stride)`` (TSVC ``s122``, ``for i in range(n1-1, N, n3)``
      with BOTH bounds symbolic). ``i - start`` is an exact multiple of ``stride`` on every
      visited ``i``, so the floor never rounds;
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
    if start is None or end is None or stride is None or stride == 0:
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
            staged = staged_iedge_rhs(rhs, e.src, sdfg)
            rhs_expr = symbolic.pystr_to_symbolic(rhs) if staged is None else staged
            lhs_sym = symbolic.pystr_to_symbolic(lhs)
            diff = symbolic.simplify(rhs_expr - lhs_sym)
        except Exception:
            # ``rhs`` may be a comparison (``StrictGreaterThan`` etc.) or
            # other non-arithmetic expression on which ``-`` raises
            # ``TypeError`` -- the assignment is not an arithmetic IV.
            continue
        # A numeric literal is always admissible; anything else must pass the IV-vs-reduction
        # discriminator (see ``step_is_loop_invariant``). A varying step has no closed form.
        if not diff.is_number:
            if not diff.free_symbols or not step_is_loop_invariant(diff.free_symbols, loop, sdfg, sdfg_free_symbols):
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
    #    iteration's increment ran before it, counted in TRIPS (``norm_iter``),
    #    not in loop-variable units -- a strided loop advances one trip per
    #    ``stride`` steps of the loop variable:
    #
    #    * TOP -- the iedge sources from the empty loop start block. Every other
    #      body block is reached AFTER the increment, so it sees this iter's
    #      increment too: ``sym = sym_init + (norm_iter + 1) * step``.
    #    * BOTTOM -- the iedge's destination is the body's unique, empty sink
    #      reached via a single in-edge (the increment is the last thing each
    #      iteration does, after all reads). Every body block is reached BEFORE
    #      the increment: ``sym = sym_init + norm_iter * step``.
    #
    #    (The frontend lowers ``for i: v = a[k]; ...; k += inc`` to the BOTTOM
    #    shape -- the gather reads the pre-increment ``k``; TSVC s318.)
    sym_sym = symbolic.pystr_to_symbolic(sym_name)
    norm_iter = trip_index(loop, start, stride)

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
    trip_count = symbolic.simplify(symbolic.int_floor(end - start, stride) + 1)
    post_loop_expr = symbolic.simplify(sym_sym + trip_count * step)
    post_loop_value = symbolic.symstr(post_loop_expr)

    # When the enclosing loop's own step already sits on ``loop``'s exit edge (a two-level counter
    # -- TSVC ``s126``, ``k`` stepped per inner iteration AND once per outer one), splicing
    # ``iv_post`` in front of it would leave that loop with TWO iedges writing ``sym``, which step
    # 1's uniqueness gate refuses: the pass would defeat itself one level up, and
    # ``_symbol_updated_in_other_loop``'s promise that the enclosing loop is left "with a single
    # step per iteration for the next round" would be false. Compose instead --
    # the spliced assignment runs first and iedge assignments are emitted in order
    # (``codegen/control_flow.py``), so this is that sequence written once. Only for a lone
    # unconditional exit edge assigning nothing but ``sym``, where no ordering question arises.
    exit_edges = list(parent.out_edges(loop))
    if (len(exit_edges) == 1 and exit_edges[0].data.is_unconditional() and list(
        (exit_edges[0].data.assignments or {}).keys()) == [sym_name]):
        exit_data = exit_edges[0].data
        composed = symbolic.pystr_to_symbolic(exit_data.assignments[sym_name]).subs(sym_sym, post_loop_expr)
        exit_data.assignments[sym_name] = symbolic.symstr(symbolic.simplify(composed))
        return True

    iv_post = parent.add_state(loop.label + '_iv_post')
    existing_out = list(parent.out_edges(loop))
    for oe in existing_out:
        parent.add_edge(iv_post, oe.dst, oe.data)
        parent.remove_edge(oe)
    iv_edge_out = dace.InterstateEdge(assignments={sym_name: post_loop_value})
    parent.add_edge(loop, iv_post, iv_edge_out)

    return True


# -----------------------------------------------------------------------------
# Loop-carried ROTATION substitution (a delay line -> a shifted array read)
# -----------------------------------------------------------------------------
#
# A carried scalar that is OVERWRITTEN every iteration with a loop-varying array element -- TSVC
# ``s254``::
#
#     x = b[N-1]
#     for i in range(N):
#         a[i] = (b[i] + x) * 0.5   # x holds b[i-1] here
#         x = b[i]                  # seeds the next iteration
#
# -- is a one-element delay line, not an accumulator: at iteration ``i`` it EQUALS
# ``b[i - stride]``. ``LoopToMap`` refuses the loop (``loop_to_map.py:671``: a bare scalar has no
# ``a*i + b`` write subset, so the write is not uniquely indexed by the iteration variable), and
# that guard is right -- the fix is to remove the carry, not to weaken the guard::
#
#     a[0] = (b[0] + b[N-1]) * 0.5                        # peeled first iteration
#     for i in range(1, N): a[i] = (b[i] + b[i-1]) * 0.5  # DOALL
#
# Same idea as ``try_substitute_use_site_iv`` -- expand the carried value's closed form at every
# read, then delete the recurrence -- but the closed form is an ARRAY READ rather than arithmetic,
# so it cannot be spliced as text into an existing connector: each read is REWIRED to a new data
# edge reading ``src[i - stride]``. And unlike an affine closed form it does not hold on the first
# iteration (which reads whatever the loop was entered with), so one iteration is peeled off the
# front first -- reusing ``LoopPeeling``, whose ``peel_limit`` budget also carries the "this loop
# runs more times than we peel it" assumption.
#
# ROTATION AND REDUCTION ARE SURFACE-IDENTICAL: one scalar, written every iteration, blocking
# ``LoopToMap``. Substituting a REDUCTION as if it were a rotation is a SILENT miscompile -- the
# loop parallelizes and every element is wrong, with no error raised. The discriminators, all of
# which must hold (see :func:`plan_rotation`):
#
# * the scalar is written EXACTLY ONCE per iteration, so "which value does this read see" has one
#   answer (a conditional update trails an unknown distance, and is refused with the body shape);
# * the stored value is a pure array element whose index moves with the iteration variable, and
#   does NOT come from the scalar itself (``x = x + b[i]`` is an accumulation -- refused, because
#   chasing the update back lands on the scalar rather than on an independent array);
# * every read of the scalar happens STRICTLY BEFORE that write, decided by DATAFLOW (intra-state
#   reachability) and block order (inter-state), never by source order. A read of the version the
#   update wrote sees ``b[i]``, not ``b[i - stride]``, and is refused;
# * the source array is not WRITTEN in the loop, so ``src[i - stride]`` still holds what iteration
#   ``i - stride`` read (an in-place ``x = a[i]`` delay line would otherwise read the version that
#   iteration WROTE);
# * the scalar is dead after the loop -- deleting its update must not lose a value someone reads.
#
# A multi-stage delay line (TSVC ``s255``: ``y = x; x = b[i]``, so ``x == b[i-1]`` and
# ``y == b[i-2]``) resolves innermost-first through the pass' fixed point rather than through
# dedicated chain logic: ``y``'s stored value is ``x``, which is written LATER in the body, so the
# chase refuses it on the first round. Substituting ``x`` rewrites ``y``'s update to
# ``y = b[i-1]`` -- itself now a rotation -- which the second round closes to ``b[i-2]``, peeling
# one more iteration. Each round peels exactly one iteration, so the cumulative peel is the delay
# depth.
#
# REMATERIALIZATION (TSVC ``s252``). The carried value need not be an array element: it can be the
# previous iteration's value of a COMPUTED transient::
#
#     t = 0.0
#     for i in range(N):
#         s = b[i] * c[i]
#         a[i] = s + t     # t holds b[i-1] * c[i-1] here
#         t = s
#
# The closed form is the producer re-evaluated one iteration back -- ``b[i-1] * c[i-1]`` -- so the
# read is replaced by a CLONE of the producer tasklet whose own reads are shifted, instead of by a
# single shifted read. Everything above still applies; recomputation adds two failure modes the
# shifted read does not have, and :func:`rematerializable_producer` refuses both outright:
#
# * a second evaluation DUPLICATES whatever the producer does besides writing its output. So the
#   accepted producer language is one ``__out = <arithmetic over the input connectors>`` assignment
#   -- no calls at all (a call may be a dace callback), no side effects, one output connector, and
#   no reference to the iteration variable in the code (which the clone would have to rewrite);
# * the clone reads its inputs LATE, so a container the body itself writes cannot simply be re-read
#   after the overwrite -- ``b[i] = ...; t = b[i] * c[i]`` does not rematerialize to
#   ``b[i-1] * c[i-1]``. A body-written input is instead RECOMPUTED in turn when it is a transient
#   whose own single writer is one more pure tasklet, so the clone becomes a chain: the emitter form
#   of s252 stages the product through ``t = s + 0.0``, and only the leaves of that chain are read
#   from memory. Every level re-proves purity, single-writer and ordering, and the leaves are all
#   containers the body never writes -- which is what still refuses a producer fed by the carry
#   itself (``t = t + s`` is a REDUCTION: the read of ``t`` sees a write that comes LATER in the
#   body) or by another carried scalar.
#
# Every level of that chain executes in the SAME iteration as the update -- that is exactly what the
# ordering gate proves -- so one shift by ``stride`` at the leaves closes the whole tree.

#: How far the update's stored value may be chased back through staging transients before giving up.
#: Bounds both the shifted-read chase and the rematerialization chain; the corpus shapes are one or
#: two stages deep, and an unbounded walk on a cyclic body would not terminate.
_ROTATION_CHASE_LIMIT = 4


class RematInput(NamedTuple):
    """One input connector of a cloned producer: a shifted read, or another clone."""
    conn: str  #: connector on the producer this input feeds
    container: str | None  #: array to read, when the value comes straight from memory
    subset: subsets.Subset | None  #: the element to read, ALREADY shifted back by one stride
    source: 'RematSource | None'  #: producer to clone instead, when the body writes the container
    moved: bool  #: whether the shift actually changed the subset


class RematSource(NamedTuple):
    """A pure producer tasklet to re-evaluate at ``i - stride`` in place of the carried read."""
    producer: nodes.Tasklet  #: tasklet that computed the value the update stored
    out_conn: str  #: its single output connector
    inputs: list[RematInput]  #: one entry per input connector, in producer edge order
    dtype: dtypes.typeclass  #: element type the clone's result is stored as
    stage: nodes.AccessNode | None  #: transient version the update read from, when it dies with the update


class RotationPlan(NamedTuple):
    """The validated rewrite for one loop-carried rotation (delay-line) scalar."""
    accum: str  #: the carried scalar container
    src_data: str | None  #: array the rotated value comes from (``None`` when rematerializing)
    src_subset: subsets.Subset | None  #: the element to read, ALREADY shifted back by one stride
    post_node: nodes.AccessNode  #: the accumulator version the update writes
    write_state: SDFGState  #: body state holding that write
    chase: List[Tuple[SDFGState, nodes.AccessNode]]  #: nodes only the update keeps alive
    reads: List[Tuple[SDFGState, nodes.AccessNode]]  #: entry versions whose reads shift back
    touched: List[str]  #: containers the rewrite may leave dangling
    remat: RematSource | None = None  #: producer to clone, when the carried value is computed


def _data_edges(edges) -> List[Any]:
    """Only the edges that MOVE DATA. An empty memlet is an ORDERING edge, so it must never be
    counted as a write (or a read) of the container it hangs off."""
    return [e for e in edges if e.data is not None and not e.data.is_empty()]


def _subset_at(edge, node) -> Optional[subsets.Subset]:
    """The subset ``edge``'s memlet addresses in ``node``'s container (``node`` is one endpoint).

    A memlet names ONE of its two containers in ``data``; the other side lives in ``other_subset``.
    Reading the wrong side turns ``b[i] -> x[0]`` into a read of ``x[i]``, so the side is picked by
    the node's own container name rather than by ``src_subset`` / ``dst_subset`` (which additionally
    depend on ``_is_data_src`` having been initialized).
    """
    memlet = edge.data
    if memlet is None:
        return None
    if not isinstance(node, nodes.AccessNode) or memlet.other_subset is None:
        return memlet.subset
    return memlet.subset if memlet.data == node.data else memlet.other_subset


def rotation_body_chain(loop: LoopRegion) -> Optional[List[SDFGState]]:
    """The loop body as a straight-line list of states, or ``None`` if it is not one.

    A rotation rewrite has to answer "does this read run before that write" for every read. A chain
    of plain states answers it by position; anything else (a conditional, a nested loop, a fork, a
    guarded edge) does not -- and a CONDITIONAL update, whose carried value trails an unknown
    distance, is exactly one of the shapes that must be refused. So the chain requirement is part
    of the safety gate, not a convenience.

    Map / NestedSDFG scopes are rejected for the reason ``try_substitute_use_site_iv`` rejects them:
    they hide the per-iteration execution order inside a scope this pass does not model.
    """
    blocks = loop.nodes()
    if not blocks or any(not isinstance(b, SDFGState) for b in blocks):
        return None
    if len(loop.edges()) != len(blocks) - 1:
        return None
    for e in loop.edges():
        if e.data.assignments or e.data.condition.as_string not in ('1', 'True', '(1)'):
            return None
    if any(loop.in_degree(b) > 1 or loop.out_degree(b) > 1 for b in blocks):
        return None
    chain: List[SDFGState] = []
    seen: Dict[SDFGState, None] = {}
    cur = loop.start_block
    while cur is not None:
        if cur in seen:
            return None
        seen[cur] = None
        chain.append(cur)
        out = loop.out_edges(cur)
        cur = out[0].dst if out else None
    if len(chain) != len(blocks):
        return None
    for st in chain:
        if any(not isinstance(n, (nodes.Tasklet, nodes.AccessNode)) for n in st.nodes()):
            return None
    return chain


def _read_after_loop(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, container: str) -> bool:
    """Whether ``container`` can be READ on any path leaving ``loop``.

    The rewrite DELETES the update, so the value the sequential loop would have left behind stops
    existing -- fine only when nobody looks at it. Reads BEFORE the loop are unaffected (the peeled
    prologue is one of them, which is what lets a multi-stage delay line peel a second time), so
    only forward reachability matters.

    Answers conservatively (``True``) for a loop nested inside another region: an enclosing loop
    re-runs its siblings, so a "later" read is also an "earlier" one and plain forward reachability
    would no longer be the whole answer.
    """
    if parent is not sdfg:
        return True
    # An interstate edge can read a container in a condition or an assignment RHS, where "before"
    # and "after" the loop are not distinguishable by block reachability alone. Rare for a carried
    # scalar, so any such read anywhere refuses rather than being classified.
    for e in sdfg.all_interstate_edges():
        if any(m.data == container for m in e.data.get_read_memlets(sdfg.arrays)):
            return True
    for blk in sdutil.dfs_conditional(sdfg, sources=[loop]):
        if blk is loop:
            continue
        for st in ([blk] if isinstance(blk, SDFGState) else list(blk.all_states())):
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == container and _data_edges(st.out_edges(n)):
                    return True
    return False


def _body_nodes(chain: List[SDFGState], container: str) -> List[Tuple[int, SDFGState, nodes.AccessNode]]:
    """Every access node for ``container`` in the body, as ``(chain index, state, node)``."""
    return [(si, st, n) for si, st in enumerate(chain) for n in st.nodes()
            if isinstance(n, nodes.AccessNode) and n.data == container]


def _body_writes(chain: List[SDFGState], container: str) -> List[Tuple[int, SDFGState, nodes.AccessNode, Any]]:
    """Every DATA write to ``container`` in the body, as ``(chain index, state, node, edge)``."""
    return [(si, st, n, e) for si, st, n in _body_nodes(chain, container) for e in _data_edges(st.in_edges(n))]


def reaches(state: SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    """Whether ``dst`` is downstream of ``src`` in ``state`` -- so ``src`` provably ran first."""
    return dst in dict.fromkeys(sdutil.dfs_conditional(state, sources=[src]))


def pure_producer(sdfg: SDFG, tasklet: nodes.Tasklet, out_conn: str | None, loop_var) -> bool:
    """Whether ``tasklet`` may be EVALUATED A SECOND TIME, on another iteration's inputs.

    A clone runs the code again, so anything the original does beyond writing its output connector is
    duplicated. Rather than classify what is duplicable, the accepted language is one
    ``__out = <arithmetic over the input connectors>`` assignment -- which is what the delay-line
    producers in the corpus are. A CALL is refused outright even when it looks pure: a name that
    resolves to a dace callback is a side effect, and telling the two apart is exactly the judgement
    this refusal avoids.
    """
    if len(tasklet.out_connectors) != 1 or out_conn is None or out_conn not in tasklet.out_connectors:
        return False
    if tasklet.code.language != dtypes.Language.Python:
        return False
    if str(loop_var) in tasklet.free_symbols:
        return False  # the code itself moves with the iteration; the clone would need it rewritten
    if tasklet.has_side_effects(sdfg):
        return False
    try:
        tree = ast.parse((tasklet.code.as_string or '').strip())
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    assign = tree.body[0]
    target = assign.targets[0] if len(assign.targets) == 1 else None
    if not isinstance(target, ast.Name) or target.id != out_conn:
        return False
    for node in ast.walk(assign.value):
        if isinstance(node, ast.Call):
            return False
        if isinstance(node, ast.Name) and node.id not in tasklet.in_connectors and node.id not in sdfg.symbols:
            return False  # a free name that is neither an input nor a symbol is unmodelled state
    return True


def staged_write(sdfg: SDFG, chain: List[SDFGState], reader_si: int, reader_node: nodes.AccessNode,
                 reader_sub: subsets.Subset):
    """The body write whose value the read of ``reader_node[reader_sub]`` provably sees.

    Returns ``(chain index, state, node, edge)`` for that write, or ``None`` when the read cannot be
    tied to a write of THIS iteration -- which is the whole ordering argument of rematerialization,
    applied identically at every level of the clone chain. A read that sees an earlier iteration's
    write is a CARRY, and recomputing it from this iteration's inputs would be a silent miscompile.
    """
    desc = sdfg.arrays.get(reader_node.data)
    if desc is None or not desc.transient:
        return None  # a non-transient may also be written from outside the body's dataflow
    staged = _body_writes(chain, reader_node.data)
    if len(staged) != 1:
        return None
    ssi, sstate, snode, sedge = staged[0]
    if ssi > reader_si:
        return None  # produced LATER in the body -> the read took a CARRIED value, not this one
    if ssi == reader_si and snode is not reader_node and not reaches(sstate, snode, reader_node):
        return None  # two versions in one state, and no dataflow path proves the producer's write ran
    write_sub = _subset_at(sedge, snode)
    if write_sub is None or _one_elem(write_sub) != 1 or str(write_sub) != str(reader_sub):
        return None  # a different element -> that write says nothing about the value read here
    if not isinstance(sedge.src, nodes.Tasklet):
        return None
    return ssi, sstate, snode, sedge


def remat_source(sdfg: SDFG, chain: List[SDFGState], loop_var, stride, reader_si: int, reader_node: nodes.AccessNode,
                 reader_sub: subsets.Subset, depth: int) -> RematSource | None:
    """The clone chain that recomputes, at ``i - stride``, the value read at ``reader_node``.

    Recursive over the producer's own inputs: one the body writes is recomputed in turn rather than
    re-read, because the clone runs LATE and would otherwise see the overwritten version. Descending
    re-applies :func:`staged_write` and :func:`pure_producer` at every level, so nothing is inherited
    from the level above. ``depth`` is bounded by ``_ROTATION_CHASE_LIMIT``: refusing a long chain
    costs a parallelization, guessing at one costs correctness. Mutates nothing.
    """
    if depth >= _ROTATION_CHASE_LIMIT:
        return None
    found = staged_write(sdfg, chain, reader_si, reader_node, reader_sub)
    if found is None:
        return None
    ssi, sstate, snode, sedge = found
    producer = sedge.src
    if not pure_producer(sdfg, producer, sedge.src_conn, loop_var):
        return None

    inputs: List[RematInput] = []
    for e in sstate.in_edges(producer):
        if e.data is None or e.data.is_empty():
            return None  # an ordering edge into the producer has no home on the clone
        src = e.src
        if not isinstance(src, nodes.AccessNode) or e.dst_conn is None:
            return None
        sub = _subset_at(e, src)
        if sub is None or _one_elem(sub) != 1:
            return None
        if _body_writes(chain, src.data):
            # Re-reading this container late would read the overwrite, so recompute it instead. The
            # recursion refuses unless the value is itself this iteration's, produced purely -- which
            # is how the carry itself, and any other carried scalar, still dead-ends here.
            nested = remat_source(sdfg, chain, loop_var, stride, ssi, src, sub, depth + 1)
            if nested is None:
                return None
            inputs.append(RematInput(e.dst_conn, None, None, nested, False))
            continue
        shifted: subsets.Subset = copy.deepcopy(sub)
        shifted.replace({loop_var: loop_var - stride})
        inputs.append(RematInput(e.dst_conn, src.data, shifted, None, str(shifted) != str(sub)))
    if sorted(inp.conn for inp in inputs) != sorted(producer.in_connectors):
        return None  # a connector reads nothing, so the clone would evaluate an undefined name
    return RematSource(producer, sedge.src_conn, inputs, sdfg.arrays[reader_node.data].dtype, None)


def remat_shifts(source: RematSource) -> int:
    """Leaf reads in the clone chain that actually move with the iteration variable."""
    return sum(int(inp.moved) + (remat_shifts(inp.source) if inp.source is not None else 0) for inp in source.inputs)


def rematerializable_producer(sdfg: SDFG, chain: list[SDFGState], accum: str, loop_var, stride, wsi: int,
                              write_edge) -> RematSource | None:
    """The producer whose re-evaluation at ``i - stride`` equals the value ``write_edge`` stores.

    Disjoint from the shifted-read chase in :func:`plan_rotation` by construction: this requires the
    update's source to be a transient at a subset that does NOT move with the iteration variable and
    whose single writer is a TASKLET -- the shape on which that chase dead-ends. Mutates nothing.
    """
    # Cheap structural refusals first; the dataflow walk and the symbolic shift come last.
    stage = write_edge.src
    if not isinstance(stage, nodes.AccessNode):
        return None
    sdesc = sdfg.arrays.get(stage.data)
    stage_sub = _subset_at(write_edge, stage)
    if sdesc is None or stage_sub is None or _one_elem(stage_sub) != 1:
        return None
    if not sdesc.transient or _uses(stage_sub, loop_var):
        return None  # a subset moving with i is the DIRECT shifted read, which the chase handles
    if sdesc.dtype != sdfg.arrays[accum].dtype:
        return None  # the copy into the carry CONVERTS; a clone feeding the reads would skip that
    source = remat_source(sdfg, chain, loop_var, stride, wsi, stage, stage_sub, 0)
    if source is None or not remat_shifts(source):
        return None  # nothing in the chain moves with i, so there is no delay to remove

    # The staging version dies with the update only when the update is its sole consumer and it holds
    # no value of its own; otherwise it stays and the update's write edge alone goes.
    write_state = chain[wsi]
    dies = write_state.out_degree(stage) == 1 and not _data_edges(write_state.in_edges(stage))
    return source._replace(stage=stage if dies else None)


def plan_rotation(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, chain: List[SDFGState], accum: str,
                  stride) -> Optional[RotationPlan]:
    """Validate that ``accum`` is a delay line and enumerate the rewrite. Mutates nothing.

    Every ``return None`` below is one of the discriminators listed at the top of this section.
    They are the whole correctness argument, because from the outside a rotation and a reduction
    are the same loop.
    """
    desc = sdfg.arrays.get(accum)
    if desc is None or not desc.transient or _one_elem(subsets.Range.from_array(desc)) != 1:
        return None

    # (1) written EXACTLY ONCE per iteration -- else a read's value is not this one update's.
    writes = _body_writes(chain, accum)
    if len(writes) != 1:
        return None
    wsi, write_state, post_node, write_edge = writes[0]
    if _one_elem(_subset_at(write_edge, post_node)) != 1:
        return None

    loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
    chase: List[Tuple[SDFGState, nodes.AccessNode]] = [(write_state, post_node)]
    touched: Dict[str, None] = {accum: None}
    src_data = shifted = None

    # (2r) the stored value may be a COMPUTED transient instead of an array element, in which case
    #      its closed form is the producer re-evaluated one iteration back. Tried first only because
    #      it is disjoint from the chase below -- it demands the tasklet writer that chase refuses.
    remat = rematerializable_producer(sdfg, chain, accum, loop_var, stride, wsi, write_edge)
    if remat is not None:
        if remat.stage is not None:
            chase.append((write_state, remat.stage))
    else:
        # (2) chase the stored value back to an array element indexed by the iteration variable. Only
        #     a staging transient whose own single write ALREADY RAN may be looked through, so a
        #     self-referencing update (an accumulation) and a value written later in the body both
        #     dead-end here rather than being mistaken for a delay.
        cur_si, cur_node, cur_sub = wsi, write_edge.src, _subset_at(write_edge, write_edge.src)
        src_subset = None
        for _ in range(_ROTATION_CHASE_LIMIT):
            if not isinstance(cur_node, nodes.AccessNode) or cur_sub is None:
                return None
            cdesc = sdfg.arrays.get(cur_node.data)
            if cdesc is None:
                return None
            if _uses(cur_sub, loop_var):
                src_data, src_subset = cur_node.data, cur_sub
                break
            if not cdesc.transient or _one_elem(cur_sub) != 1:
                return None
            staged = _body_writes(chain, cur_node.data)
            if len(staged) != 1:
                return None
            ssi, _sstate, snode, sedge = staged[0]
            if ssi > cur_si or (ssi == cur_si and snode is not cur_node):
                return None  # written LATER in the body -> a carried value, not this iteration's
            if _one_elem(_subset_at(sedge, snode)) != 1:
                return None
            # The whole staging container dies with the update; (6) below proves nothing else reads it.
            chase.extend((st, n) for _si, st, n in _body_nodes(chain, cur_node.data))
            touched[cur_node.data] = None
            cur_si, cur_node, cur_sub = ssi, sedge.src, _subset_at(sedge, sedge.src)
        if src_data is None:
            return None
        if sdfg.arrays[src_data].dtype != desc.dtype:
            return None  # the copy into the scalar CONVERTS; reading the source direct would skip that
        touched[src_data] = None

        # (3) the source must still hold, at ``i - stride``, what iteration ``i - stride`` read. A
        #     write to it inside the loop breaks that -- an in-place delay line reads the PRE-write
        #     version.
        if _body_writes(chain, src_data):
            return None
        shifted = copy.deepcopy(src_subset)
        shifted.replace({loop_var: loop_var - stride})
        if str(shifted) == str(src_subset):
            return None  # not actually shifted -> a same-iteration copy, no delay to remove

    # (4) every read must see the ENTRY value: nothing may read the version the update wrote, and
    #     no read may be ordered after it. Decided by dataflow + block order, never source order.
    reads: List[Tuple[SDFGState, nodes.AccessNode]] = []
    reachable_from_write = dict.fromkeys(sdutil.dfs_conditional(write_state, sources=[post_node]))
    written_slot = str(_subset_at(write_edge, post_node))
    for si, st, n in _body_nodes(chain, accum):
        if n is post_node:
            continue
        outs = st.out_edges(n)
        if not outs:
            continue
        if any(e.data is None or e.data.is_empty() for e in outs):
            return None  # an ordering edge OUT of a version the rewrite deletes loses its order
        if si > wsi or (si == wsi and n in reachable_from_write):
            return None  # this read observes the POST-update value: b[i], not b[i - stride]
        if any(str(_subset_at(e, n)) != written_slot for e in outs):
            return None  # reads a different slot than the update writes
        reads.append((st, n))
    if write_state.out_edges(post_node):
        return None  # read AFTER the write (value is b[i]), or an ordering edge we would drop
    if remat is not None and any(st.in_edges(n) for st, n in reads):
        return None  # an ordering edge INTO the read has no home on the cloned producer

    # (5) nothing after the loop may read the value the deleted update used to leave behind.
    if _read_after_loop(parent, loop, sdfg, accum):
        return None

    # (6) the staging transients the update alone keeps alive must have no other consumer.
    for container in touched:
        if container in (accum, src_data):
            continue
        if sum(len(_data_edges(st.out_edges(n))) for _si, st, n in _body_nodes(chain, container)) != 1:
            return None
        if _read_after_loop(parent, loop, sdfg, container):
            return None
    return RotationPlan(accum, src_data, shifted, post_node, write_state, chase, reads, list(touched), remat)


def delete_rotation_update(chain: List[SDFGState], plan: RotationPlan) -> None:
    """Drop the delay line's update -- and the staging nodes only it kept alive -- from the body.

    Runs BEFORE the peel, which is not an implementation detail: the peeled iteration is a CLONE of
    the body, and once every read is closed the update is dead in the CLONE TOO (gate (5) proved
    nothing past the loop reads the carry). Peeling first would leave that dead write
    behind, and a dead write to a carried scalar is not inert -- ``ScalarFission`` reads the peeled
    iteration's entry read and its own trailing write as one version, and the read then comes from
    an unwritten container.
    """
    for st, n in plan.chase:
        if n in dict.fromkeys(st.nodes()):
            st.remove_node(n)
    # What fed the update is now an access node holding only ordering edges to a write that no
    # longer exists -- vacuous, so it goes too. Restricted to the containers this rewrite touched,
    # so the pass never quietly prunes dataflow it did not create.
    for st in chain:
        for n in list(st.nodes()):
            if not isinstance(n, nodes.AccessNode) or n.data not in plan.touched:
                continue
            if st.out_degree(n) == 0 and not _data_edges(st.in_edges(n)):
                st.remove_node(n)


def shift_rotation_reads(plan: RotationPlan) -> None:
    """Rewire every read of the carried scalar to the source element one stride back.

    The read's ORDERING edges move with it: they sequence the CONSUMER, not the scalar, so dropping
    them would lose the order they enforce (in a two-stage delay line they are the WAR guard on the
    stage this very read feeds).
    """
    from dace import memlet as mm

    src_data, src_subset = plan.src_data, plan.src_subset
    assert src_data is not None and src_subset is not None  # the caller dispatches on plan.remat
    for st, n in plan.reads:
        new_src = st.add_access(src_data)
        for e in list(st.out_edges(n)):
            if isinstance(e.dst, nodes.AccessNode):
                memlet = mm.Memlet(data=src_data,
                                   subset=copy.deepcopy(src_subset),
                                   other_subset=copy.deepcopy(_subset_at(e, e.dst)))
            else:
                memlet = mm.Memlet(data=src_data, subset=copy.deepcopy(src_subset))
            st.remove_edge(e)
            st.add_edge(new_src, None, e.dst, e.dst_conn, memlet)
        for e in list(st.in_edges(n)):
            st.remove_edge(e)
            st.add_edge(e.src, e.src_conn, new_src, None, copy.deepcopy(e.data))
        st.remove_node(n)


def emit_remat_clone(sdfg: SDFG, st: SDFGState, source: RematSource, hint: str) -> nodes.AccessNode:
    """Build one fresh evaluation of ``source`` in ``st`` and return the access node holding its result.

    Recurses input-first, so a staging chain becomes a chain of clones whose leaves read memory. Each
    clone is BUILT rather than deep-copied so it carries only the code and the connector types -- no
    guid, no debug info, nothing that would make two nodes claim to be one.
    """
    from dace import memlet as mm

    producer = source.producer
    clone = nodes.Tasklet(f'{producer.label}_remat', dict(producer.in_connectors), dict(producer.out_connectors),
                          producer.code.as_string, producer.code.language)
    st.add_node(clone)
    for inp in source.inputs:
        if inp.source is None:
            assert inp.container is not None and inp.subset is not None  # the matcher fills one or the other
            st.add_edge(st.add_access(inp.container), None, clone, inp.conn,
                        mm.Memlet(data=inp.container, subset=copy.deepcopy(inp.subset)))
        else:
            nested = emit_remat_clone(sdfg, st, inp.source, hint)
            st.add_edge(nested, None, clone, inp.conn, mm.Memlet(data=nested.data, subset='0'))
    name, _ = sdfg.add_scalar(f'{hint}_remat', source.dtype, transient=True, find_new_name=True)
    value = st.add_access(name)
    st.add_edge(clone, source.out_conn, value, None, mm.Memlet(data=name, subset='0'))
    return value


def rematerialize_rotation_reads(sdfg: SDFG, plan: RotationPlan) -> None:
    """Replace every read of the carried scalar by a fresh evaluation of its producer at ``i - stride``.

    One clone chain per read, each with its own result scalars: every producer in the chain is pure
    (:func:`pure_producer`) so a second evaluation yields the same value, and every container the
    chain reads from memory was proven unwritten in the body, so reading it late reads the same bytes.
    """
    from dace import memlet as mm

    remat = plan.remat
    assert remat is not None  # the caller dispatches on plan.remat
    for st, n in plan.reads:
        value = emit_remat_clone(sdfg, st, remat, plan.accum)
        for e in list(st.out_edges(n)):
            if isinstance(e.dst, nodes.AccessNode):
                memlet = mm.Memlet(data=value.data, subset='0', other_subset=copy.deepcopy(_subset_at(e, e.dst)))
            else:
                memlet = mm.Memlet(data=value.data, subset='0')
            st.remove_edge(e)
            st.add_edge(value, None, e.dst, e.dst_conn, memlet)
        st.remove_node(n)


def try_substitute_rotation(parent: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, budget: Dict[Any, int]) -> bool:
    """Peel one iteration off ``loop`` and close one carried delay line.

    The closed form is a shifted read of the array that fed the carry, or -- when the carry held a
    COMPUTED transient -- a clone of its producer evaluated one iteration back.

    ``budget`` maps a loop to the peels it has left; a multi-stage delay line spends one per stage.
    """
    from dace.transformation.interstate.loop_peeling import LoopPeeling
    from dace.transformation.passes.parallelization_prep import _constant_trip_count, _unique_block_label

    if not loop.loop_variable or budget.get(loop, 0) <= 0:
        return False
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride == 0:
        return False
    chain = rotation_body_chain(loop)
    if chain is None:
        return False

    # Deterministic candidate order: body-block order, then node order inside each state. The pass
    # restarts after every rewrite, so which carried scalar is taken first decides which shapes the
    # NEXT round still matches -- iterating a hash-ordered set would make that a PYTHONHASHSEED flip.
    candidates = dict.fromkeys(n.data for st in chain for n in st.nodes() if isinstance(n, nodes.AccessNode))
    for accum in candidates:
        plan = plan_rotation(parent, loop, sdfg, chain, accum, stride)
        if plan is None:
            continue
        # ``LoopPeeling`` does NOT guard the peeled iteration, so peeling also asserts the loop runs
        # at least once -- the assumption ``peel_limit`` already licenses for BestEffortLoopPeeling.
        # Where the trip count is known, check it instead of assuming it.
        trip = _constant_trip_count(loop, sdfg)
        if trip is not None and trip < 1:
            return False
        # The peel is the only step that can still fail, and it runs BETWEEN the two halves of the
        # rewrite -- so probe its one raising step (the new loop start) while the SDFG is still
        # untouched, rather than discovering it half-applied.
        try:
            symbolic.evaluate(start + stride, sdfg.constants)
        except Exception:
            return False
        # ``LoopPeeling`` names the peeled iteration after the loop, so a second peel on the same
        # loop -- which a two-stage delay line needs -- would mint a duplicate block label. Rename
        # the remainder first, exactly as ``BestEffortLoopPeeling`` does for its front/back pair.
        peel_prefix = f'{loop.label}_{loop.loop_variable}'
        if any(b.label.startswith(peel_prefix) for b in loop.sdfg.all_control_flow_blocks()):
            loop.label = _unique_block_label(loop.sdfg, loop.label)
        # Order matters: the update is dead everywhere once the reads are closed, so it goes before
        # the peel CLONES the body; the reads shift only in the loop, which the peel has narrowed
        # to the iterations where the shifted read is in range.
        delete_rotation_update(chain, plan)
        LoopPeeling().apply_to(sdfg=loop.sdfg, loop=loop, verify=False, options={'count': 1, 'begin': True})
        if plan.remat is None:
            shift_rotation_reads(plan)
        else:
            rematerialize_rotation_reads(sdfg, plan)
        budget[loop] -= 1
        return True
    return False


@properties.make_properties
@xf.explicit_cf_compatible
class LoopCarriedRotationSubstitution(ppl.Pass):
    """Replace a loop-carried delay-line scalar by its closed form one iteration back.

    That is a shifted read of the array that fed the carry, or a re-evaluation of the producer that
    computed it. Either way the carry that makes ``LoopToMap`` refuse the loop is gone; see the
    section comment above for the shapes, the rewrites, and the gate that keeps a reduction from
    being rewritten as one.
    """

    CATEGORY: str = 'Optimization Preparation'

    peel_limit = properties.Property(dtype=int,
                                     default=4,
                                     desc='Iterations this pass may peel off one loop, which also bounds the delay '
                                     'depth a rotation may have (0 disables the pass). Peeling is what makes the '
                                     'shifted read valid on the first iteration.')

    def __init__(self, peel_limit: int = 4):
        super().__init__()
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        if self.peel_limit <= 0:
            return None
        # Fixed point: a multi-stage delay line only reveals its outer stage once the inner one is
        # gone (``y = x`` becomes ``y = b[i-1]``), so re-scan until nothing matches. Every rewrite
        # removes one carried scalar and spends one peel, so this terminates.
        budget: Dict[Any, int] = {}
        count = 0
        while True:
            for node, parent in list(sdfg.all_nodes_recursive()):
                if not isinstance(node, LoopRegion) or _loop_has_break_or_continue(node):
                    continue
                budget.setdefault(node, self.peel_limit)
                if try_substitute_rotation(parent, node, sdfg, budget):
                    count += 1
                    break  # SDFG mutated -> restart the scan on a fresh node list
            else:
                break
        return count or None
