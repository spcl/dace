# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop-carried in-place array reduction to a write-conflict resolution.

A sequential loop that accumulates into a whole array element-wise -- the
contour_integral shape ``for idx: P[i, j] = P[i, j] + X_idx[i, j]`` (an inner
element map over ``i, j`` nested in the outer ``idx`` loop) -- is a REDUCTION of
``P`` over ``idx``. But because ``P`` is read back and written every iteration,
``LoopToMap`` sees a loop-carried dependency on ``P`` and refuses to parallelize
the ``idx`` loop.

This pass recognises that pattern and drops the read-back, turning the in-place
accumulation into a WCR write ``P[i, j] (wcr: +)= X_idx[i, j]``. With ``P`` no
longer read across ``idx`` iterations, ``LoopToMap`` (a later pipeline stage)
parallelizes ``idx``; the WCR resolves the cross-iteration accumulation (an
OpenMP reduction / atomic at code generation).

``AugAssignToWCR`` performs the same edge surgery but only for a single-output
map whose write is NOT injective over its OWN map params; here the write IS
injective over the element map (``P[i, j]`` -- one element per ``i, j``), so it
refuses. The reduction axis is the ENCLOSING loop, invisible to that per-map
check, and the accumulation map is often a fused multi-output map (``P0`` and
``P1`` together) it cannot match. This pass supplies the missing enclosing-loop
view and the multi-output handling.

Soundness -- the lift is value-preserving (up to reduction reassociation, as for
any parallel reduction) when, for the accumulator ``A`` and the loop ``L``:

* the combine is an associative/commutative reduction (``+``, ``*``, ``min``,
  ``max``) -- WCR-safe;
* ``A`` is read at the accumulated subset ``S`` ONLY as the accumulator operand
  of its own in-place write (a PURE accumulator -- never used to compute the
  increment or anything else in ``L``); otherwise it is a recurrence, not a
  reduction;
* ``S`` is invariant over ``L``'s loop variable (every iteration hits the same
  elements -- the cross-iteration conflict the WCR resolves);
* the read-back subset equals the write subset (the tasklet reads exactly the
  element it writes).

The pre-loop value of ``A`` is the reduction seed in BOTH the sequential and the
WCR form, so ``A`` need not be initialised for the lift to be sound.
"""
import ast
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, nodes, properties
from dace.sdfg import SDFGState
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

#: Python AST op -> WCR operator symbol for the associative/commutative reductions.
_REDUCTION_OPS = {ast.Add: '+', ast.Mult: '*'}
#: min / max reductions arrive as a 2-argument Call.
_REDUCTION_FUNCS = ('min', 'max')


def _reduction_operands(tasklet: nodes.Tasklet) -> Optional[Tuple[str, Tuple[ast.AST, ast.AST]]]:
    """If ``tasklet`` is a single assignment whose RHS is an associative binary
    reduction (``+`` / ``*``) or a two-argument ``min`` / ``max``, return
    ``(op, (operand0, operand1))`` -- the reduction operator symbol and the two
    RHS operand AST nodes. Else ``None``.

    WHICH operand is the accumulator is decided by the CALLER from dataflow (the
    operand whose connector reads the accumulator array at the written element),
    not by position here: picking it by identity alone is ambiguous when both
    operands are bare input Names (``out = out + B`` vs ``out = B + out``). The
    other operand is the increment -- it may be any expression and becomes the
    WCR value verbatim (see :func:`_increment_ast`).
    """
    if tasklet.language.name != 'Python' or len(tasklet.code.code) != 1:
        return None
    stmt = tasklet.code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    rhs = stmt.value
    if isinstance(rhs, ast.BinOp) and type(rhs.op) in _REDUCTION_OPS:
        return _REDUCTION_OPS[type(rhs.op)], (rhs.left, rhs.right)
    if (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id in _REDUCTION_FUNCS
            and len(rhs.args) == 2):
        return rhs.func.id, (rhs.args[0], rhs.args[1])
    return None


def _copy_input_connector(tasklet: nodes.Tasklet) -> Optional[str]:
    """If ``tasklet`` is a pure copy ``__out = __inp`` (a single-``Name`` RHS that
    reads an input connector), return that input connector name; else ``None``.

    The frontend emits ``A[i] = A[i] + delta`` as ``t = A[i] + delta`` into a
    scalar transient followed by a copy ``A[i] = t``; before ``TrivialTasklet
    Elimination`` collapses it, the tasklet feeding the map exit is this copy,
    not the reduction. :func:`_trace_to_reduction` follows the copy to the real
    reduction tasklet so the lift matches whether or not the copy was cleaned
    (the cleaned direct shape is what the in-pipeline contour_integral has).
    """
    if tasklet.language.name != 'Python' or len(tasklet.code.code) != 1:
        return None
    stmt = tasklet.code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if isinstance(stmt.value, ast.Name) and stmt.value.id in tasklet.in_connectors:
        return stmt.value.id
    return None


def _trace_to_reduction(st: SDFGState, mx_in) -> Optional[nodes.Tasklet]:
    """The reduction tasklet writing the accumulator through ``mx_in`` (the edge
    feeding the map-exit input connector). Handles the DIRECT shape -- the
    reduction tasklet writes the map exit itself (the cleaned contour shape) --
    and the frontend COPY shape ``reduction_tasklet -> scalar transient -> copy
    tasklet -> map exit``. Returns the reduction tasklet or ``None``."""
    src = mx_in.src
    if not isinstance(src, nodes.Tasklet):
        return None
    if _reduction_operands(src) is not None:
        return src  # direct: the reduction tasklet writes the map exit
    copy_in = _copy_input_connector(src)
    if copy_in is None:
        return None
    ins = [e for e in st.in_edges(src) if e.dst_conn == copy_in]
    if len(ins) != 1 or not isinstance(ins[0].src, nodes.AccessNode):
        return None
    transient = ins[0].src
    producers = st.in_edges(transient)
    if len(producers) != 1 or not isinstance(producers[0].src, nodes.Tasklet):
        return None
    red = producers[0].src
    return red if _reduction_operands(red) is not None else None


def _increment_ast(tasklet: nodes.Tasklet, acc_conn: str) -> ast.AST:
    """The non-accumulator (increment) operand of the reduction tasklet."""
    rhs = tasklet.code.code[0].value
    operands = (rhs.left, rhs.right) if isinstance(rhs, ast.BinOp) else tuple(rhs.args)
    acc = next(o for o in operands if isinstance(o, ast.Name) and o.id == acc_conn)
    return operands[1] if operands[0] is acc else operands[0]


class _AccumulatorCandidate:
    """A matched in-place reduction into ``array`` inside one map, ready to lift."""

    def __init__(self, state, map_entry, map_exit, tasklet, array, op, acc_conn, read_edge, entry_out_edge,
                 tasklet_out_edge, exit_out_edge):
        self.state = state
        self.map_entry = map_entry
        self.map_exit = map_exit
        self.tasklet = tasklet
        self.array = array
        self.op = op
        self.acc_conn = acc_conn
        self.read_edge = read_edge  # AccessNode(A) -> map_entry
        self.entry_out_edge = entry_out_edge  # map_entry -> tasklet.acc_conn
        self.tasklet_out_edge = tasklet_out_edge  # tasklet.__out -> map_exit
        self.exit_out_edge = exit_out_edge  # map_exit -> AccessNode(A)


@properties.make_properties
@explicit_cf_compatible
class LiftLoopCarriedReduction(ppl.Pass):
    """Lift loop-carried in-place array reductions to WCR writes so the enclosing
    loop can be parallelized by ``LoopToMap``."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        lifted = 0
        for sd in sdfg.all_sdfgs_recursive():
            for loop in [n for n in sd.all_control_flow_regions(recursive=True) if isinstance(n, LoopRegion)]:
                if loop.pinned_sequential or not loop.loop_variable:
                    continue
                lifted += self._lift_loop(loop)
        return lifted or None

    def _lift_loop(self, loop: LoopRegion) -> int:
        itervar = loop.loop_variable
        body_states = list(loop.all_states())
        # Candidate reductions: one per (map, accumulator array) with an invariant subset.
        candidates = self._collect_candidates(body_states, itervar)
        if not candidates:
            return 0
        # Per-array purity gate: the accumulator must be read in the loop body ONLY as the
        # read-back of its own reduction (never to compute an increment or anything else),
        # else it is a recurrence and the WCR would change the value.
        reads_by_array = self._accumulator_reads(body_states)
        lifted = 0
        for cand in candidates:
            allowed = {cand.read_edge}
            if reads_by_array.get(cand.array, set()) - allowed:
                continue  # array read elsewhere in the loop -> not a pure accumulator
            self._apply_lift(cand)
            lifted += 1
        return lifted

    def _collect_candidates(self, body_states: List[SDFGState], itervar: str) -> List[_AccumulatorCandidate]:
        out: List[_AccumulatorCandidate] = []
        for st in body_states:
            for mx in [n for n in st.nodes() if isinstance(n, nodes.MapExit)]:
                me = st.entry_node(mx)
                for exit_out in st.out_edges(mx):
                    if not isinstance(exit_out.dst, nodes.AccessNode):
                        continue
                    array = exit_out.dst.data
                    if exit_out.data.wcr is not None or exit_out.data.data != array:
                        continue
                    if itervar in {str(s) for s in exit_out.data.subset.free_symbols}:
                        continue  # write subset must be invariant over the loop variable
                    cand = self._match_reduction(st, me, mx, exit_out, array)
                    if cand is not None:
                        out.append(cand)
        return out

    def _match_reduction(self, st: SDFGState, me: nodes.MapEntry, mx: nodes.MapExit, exit_out,
                         array: str) -> Optional[_AccumulatorCandidate]:
        # The edge feeding this map-exit input connector carries the per-iteration
        # accumulator write; its subset is the element written each iteration.
        conn = exit_out.src_conn  # OUT_x
        in_conn = 'IN' + conn[3:]
        mx_in = [e for e in st.in_edges(mx) if e.dst_conn == in_conn]
        if len(mx_in) != 1:
            return None
        mx_in = mx_in[0]
        write_subset = mx_in.data.subset
        # The reduction tasklet feeding this write -- directly (cleaned contour shape)
        # or through the frontend ``reduction -> transient -> copy -> map exit`` chain.
        tasklet = _trace_to_reduction(st, mx_in)
        if tasklet is None:
            return None
        op, operands = _reduction_operands(tasklet)
        operand_conns = {o.id for o in operands if isinstance(o, ast.Name) and o.id in tasklet.in_connectors}
        # Accumulator read-back: map_entry -> tasklet, reading A at the write subset, into a
        # bare-Name reduction operand. Identify the accumulator by dataflow (which operand
        # reads exactly the written element), so the increment stays whatever the other
        # operand is -- and refuse ``out = out + out`` (both operands read A) as ambiguous.
        acc_edges = [
            e for e in st.in_edges(tasklet) if e.src is me and e.dst_conn in operand_conns
            and e.data.data == array and e.data.subset == write_subset
        ]
        if len(acc_edges) != 1:
            return None
        entry_out = acc_edges[0]
        acc_conn = entry_out.dst_conn
        # Intra-map purity: the map-entry read feeding the accumulator element must be
        # consumed ONLY by this accumulator operand. If that map-entry-out connector fans
        # out to any other tasklet inside the map, A[write_subset] also forms the increment
        # (``out = out + B * out``) -- a recurrence, not a pure reduction. Refuse.
        if len([e for e in st.out_edges(me) if e.src_conn == entry_out.src_conn]) != 1:
            return None
        # the array read into the map entry feeding that accumulator connector
        me_in_conn = 'IN' + entry_out.src_conn[3:]
        read_edges = [e for e in st.in_edges(me) if e.dst_conn == me_in_conn and isinstance(e.src, nodes.AccessNode)
                      and e.src.data == array]
        if len(read_edges) != 1:
            return None
        return _AccumulatorCandidate(st, me, mx, tasklet, array, op, acc_conn, read_edges[0], entry_out, mx_in,
                                     exit_out)

    def _accumulator_reads(self, body_states: List[SDFGState]) -> Dict[str, Set]:
        """Every ``AccessNode(A) -> *`` read edge of each array A across the loop body."""
        reads: Dict[str, Set] = {}
        for st in body_states:
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode):
                    for e in st.out_edges(n):
                        reads.setdefault(n.data, set()).add(e)
        return reads

    def _apply_lift(self, c: _AccumulatorCandidate) -> None:
        st = c.state
        wcr = f'lambda a, b: {c.op}(a, b)' if c.op in _REDUCTION_FUNCS else f'lambda a, b: a {c.op} b'
        # Rewrite the tasklet to emit only the increment (drop the accumulator operand).
        inc = _increment_ast(c.tasklet, c.acc_conn)
        stmt = c.tasklet.code.code[0]
        c.tasklet.code.code = [ast.copy_location(ast.Assign(targets=stmt.targets, value=inc), stmt)]
        # Set the WCR on the write path (tasklet -> map_exit -> array).
        c.tasklet_out_edge.data.wcr = wcr
        c.exit_out_edge.data.wcr = wcr
        # Drop the accumulator read path: map_entry -> tasklet, array -> map_entry, connectors.
        st.remove_edge(c.entry_out_edge)
        st.remove_edge(c.read_edge)
        if c.acc_conn in c.tasklet.in_connectors:
            c.tasklet.remove_in_connector(c.acc_conn)
        me_out_conn = c.entry_out_edge.src_conn
        me_in_conn = 'IN' + me_out_conn[3:]
        if not any(e.src_conn == me_out_conn for e in st.out_edges(c.map_entry)):
            c.map_entry.remove_out_connector(me_out_conn)
            if not any(e.dst_conn == me_in_conn for e in st.in_edges(c.map_entry)):
                c.map_entry.remove_in_connector(me_in_conn)
        if st.degree(c.read_edge.src) == 0:
            st.remove_node(c.read_edge.src)


__all__ = ['LiftLoopCarriedReduction']
