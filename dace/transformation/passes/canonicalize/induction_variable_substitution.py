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


def _extract_iv(loop: LoopRegion, sdfg: SDFG) -> Optional[Tuple[str, str, type, object, object]]:
    """Pattern-match the loop body. Returns ``(accum_name, accum_subset_str, ast.BinOp_type, const_val, trip_count)``
    or ``None``.
    """
    if not loop.loop_variable:
        return None
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
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
    rhs = tree.body[0].value
    if not isinstance(rhs, ast.BinOp) or type(rhs.op) not in _CLOSED_FORM:
        return None

    # One side must be a Name (the carried accum), the other a numeric constant.
    if isinstance(rhs.left, ast.Name) and isinstance(rhs.right, ast.Constant):
        var_conn, const_val = rhs.left.id, rhs.right.value
    elif isinstance(rhs.right, ast.Name) and isinstance(rhs.left, ast.Constant):
        var_conn, const_val = rhs.right.id, rhs.left.value
    else:
        return None
    if not isinstance(const_val, (int, float)):
        return None

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
