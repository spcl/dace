# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse a chain of consecutive same-accumulator reductions into one.

A loop body that accumulates into the SAME scalar slot more than once per
iteration -- the TSVC ``s319`` shape::

    sum_val = 0.0
    for i in range(N):
        a[i] = c[i] + d[i]
        sum_val = sum_val + a[i]       # step 1
        b[i] = c[i] + e[i]
        sum_val = sum_val + b[i]       # step 2

-- reaches the reduction stage as a chained dataflow inside one state::

    acc --> (+ incA) --> tmpA --copy--> acc --> (+ incB) --> tmpB --copy--> acc

The intermediate read-back of ``acc`` between the two ``+`` breaks the
single-accumulation shape that ``LoopToReduce`` / the WCR-on-scalar lift
matches, so the loop stays sequential (only the element-wise ``a[i]`` / ``b[i]``
writes get fissioned into a parallel map; the reduction loop survives).

By associativity of the reduction operator, ``((acc OP incA) OP incB)`` equals
``acc OP (incA OP incB)`` -- a SINGLE accumulation whose increment is the
combined ``incA OP incB``. This pass performs that re-association: it collapses
the chain into one accumulation feeding the terminal ``acc`` node, with the
increments combined by a small operator tree. The downstream single-accumulation
lift then parallelizes the loop into a Map + WCR-on-scalar reduction.

Soundness -- the fold is value-preserving (up to the reassociation every
parallel reduction already performs) when, for the accumulator ``acc`` and the
chain of steps in one state:

* every step is ``acc[S] = acc[S] OP inc`` with the SAME associative operator
  ``OP`` (``+`` or ``*``) and the SAME single-element slot ``S`` (a scalar slot,
  loop-invariant);
* the steps form a single linear chain: each step reads the ``acc`` node the
  previous step wrote (no branching / no other writer of ``acc[S]`` in the
  state);
* no increment reads ``acc`` (a pure reduction, never a recurrence).

Only ``+`` and ``*`` are folded: they are associative AND commutative, so the
combined-increment tree is order-independent. ``min`` / ``max`` are associative
too but arrive as a Call shape handled by ``ArgMaxLift``; this pass leaves them
untouched.
"""
import ast
import copy
from typing import List, Optional, Tuple

from dace import SDFG, nodes, properties
from dace.sdfg import SDFGState
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

#: AST binop type -> operator source string. Only associative+commutative ops.
_FOLDABLE_OPS = {ast.Add: '+', ast.Mult: '*'}


def _binop_op(tasklet: nodes.Tasklet) -> Optional[type]:
    """The AST binop type if ``tasklet`` is a single ``__out = a OP b`` with a
    foldable ``OP``; else ``None``."""
    if tasklet.language.name != 'Python' or len(tasklet.code.code) != 1:
        return None
    stmt = tasklet.code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    rhs = stmt.value
    if isinstance(rhs, ast.BinOp) and type(rhs.op) in _FOLDABLE_OPS:
        return type(rhs.op)
    return None


def _is_copy(tasklet: nodes.Tasklet) -> Optional[str]:
    """Input connector if ``tasklet`` is a pure copy ``__out = __in``; else ``None``."""
    if tasklet.language.name != 'Python' or len(tasklet.code.code) != 1:
        return None
    stmt = tasklet.code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if isinstance(stmt.value, ast.Name) and stmt.value.id in tasklet.in_connectors:
        return stmt.value.id
    return None


def _chase_write_to_accum(state: SDFGState, sdfg: SDFG, out_edge):
    """From a binop's output edge, follow the staging chain forward to the
    AccessNode it ultimately writes. The frontend stages an accumulator write as
    ``binop -> tmp -> copy -> acc``; that copy is a copy TASKLET before
    ``TrivialTaskletElimination`` and a direct AccessNode->AccessNode memlet copy
    after it -- both are followed here. Returns ``(final_node,
    [intermediate_nodes], [copy_tasklets])`` or ``None`` if the chain is not a
    simple transient-copy staging into an AccessNode.
    """
    intermediates: List = []
    copies: List = []
    node = out_edge.dst
    while True:
        if not isinstance(node, nodes.AccessNode):
            return None
        desc = sdfg.arrays.get(node.data)
        if desc is None:
            return None
        # A non-transient (or multiply-connected) AccessNode is the final target.
        if not desc.transient:
            return node, intermediates, copies
        outs = state.out_edges(node)
        # Terminal transient accumulator (e.g. a transient scalar written back by
        # the loop): no further consumer inside the state.
        if len(outs) != 1 or len(state.in_edges(node)) != 1:
            return node, intermediates, copies
        nxt = outs[0]
        if isinstance(nxt.dst, nodes.AccessNode):
            # Direct transient->AccessNode memlet copy (post-TrivialTaskletElimination shape).
            intermediates.append(node)
            node = nxt.dst
            continue
        if not isinstance(nxt.dst, nodes.Tasklet):
            return node, intermediates, copies
        copy_conn = _is_copy(nxt.dst)
        if copy_conn is None or nxt.dst_conn != copy_conn:
            return node, intermediates, copies
        copy_tasklet = nxt.dst
        cout = state.out_edges(copy_tasklet)
        if len(cout) != 1:
            return None
        intermediates.append(node)
        copies.append(copy_tasklet)
        node = cout[0].dst


class _Step:
    """One ``acc[S] = acc[S] OP inc`` accumulation in the chain."""

    def __init__(self, binop, acc_read_node, acc_read_edge, inc_edge, write_final, write_intermediates, write_copies):
        self.binop = binop
        self.acc_read_node = acc_read_node  # AccessNode(acc) feeding the accumulator connector
        self.acc_read_edge = acc_read_edge  # acc_read_node -> binop (accumulator operand)
        self.inc_edge = inc_edge  # <src> -> binop (increment operand)
        self.write_final = write_final  # AccessNode(acc) this step ultimately writes
        self.write_intermediates = write_intermediates  # transient staging nodes to delete
        self.write_copies = write_copies  # copy tasklets to delete


@properties.make_properties
@explicit_cf_compatible
class FuseChainedScalarReductions(ppl.Pass):
    """Re-associate a chain of same-accumulator ``+`` / ``*`` reductions into one."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            for loop in [n for n in sd.all_control_flow_regions(recursive=True) if isinstance(n, LoopRegion)]:
                if loop.pinned_sequential or not loop.loop_variable:
                    continue
                for st in loop.all_states():
                    fused += self._fuse_state(sd, st)
        return fused or None

    def _fuse_state(self, sdfg: SDFG, st: SDFGState) -> int:
        chains = self._collect_chains(sdfg, st)
        count = 0
        for op_type, steps in chains:
            self._apply_fusion(sdfg, st, op_type, steps)
            count += 1
        return count

    def _collect_chains(self, sdfg: SDFG, st: SDFGState) -> List[Tuple[type, List[_Step]]]:
        # Build the per-binop accumulation record: which acc node it reads and writes.
        steps_by_read = {}  # acc_read_node -> _Step
        steps_by_write = {}  # write_final_node -> _Step
        all_steps: List[_Step] = []
        for tasklet in [n for n in st.nodes() if isinstance(n, nodes.Tasklet)]:
            op_type = _binop_op(tasklet)
            if op_type is None:
                continue
            in_edges = [e for e in st.in_edges(tasklet) if e.data is not None and not e.data.is_empty()]
            out_edges = [e for e in st.out_edges(tasklet) if e.data is not None and not e.data.is_empty()]
            if len(in_edges) != 2 or len(out_edges) != 1:
                continue
            chased = _chase_write_to_accum(st, sdfg, out_edges[0])
            if chased is None:
                continue
            write_final, intermediates, copies = chased
            acc_name = write_final.data
            write_subset = None
            # The edge into write_final carries the written slot.
            wfin = st.in_edges(write_final)
            if len(wfin) != 1:
                continue
            write_subset = wfin[0].data.subset
            # Identify the accumulator operand: the input edge reading acc_name at write_subset.
            acc_edges = [
                e for e in in_edges if isinstance(e.src, nodes.AccessNode) and e.src.data == acc_name
                and str(e.data.subset) == str(write_subset)
            ]
            if len(acc_edges) != 1:
                continue
            acc_edge = acc_edges[0]
            inc_edges = [e for e in in_edges if e is not acc_edge]
            if len(inc_edges) != 1:
                continue
            inc_edge = inc_edges[0]
            # The increment must not read the accumulator (else it is a recurrence).
            if isinstance(inc_edge.src, nodes.AccessNode) and inc_edge.src.data == acc_name:
                continue
            # Single-element, scalar slot.
            desc = sdfg.arrays.get(acc_name)
            if desc is None:
                continue
            step = _Step(tasklet, acc_edge.src, acc_edge, inc_edge, write_final, intermediates, copies)
            step.acc_name = acc_name
            step.write_subset = str(write_subset)
            all_steps.append(step)
            steps_by_read.setdefault(acc_edge.src, step)
            steps_by_write.setdefault(write_final, step)

        # Group steps that share the same accumulator/slot/op and form a linear chain.
        chains: List[Tuple[type, List[_Step]]] = []
        used = set()
        for step in all_steps:
            if step in used:
                continue
            op_type = _binop_op(step.binop)
            # Walk to the head: a step whose acc-read node is not written by another step in the group.
            head = step
            guard = 0
            while head.acc_read_node in steps_by_write and guard < len(all_steps) + 1:
                prev = steps_by_write[head.acc_read_node]
                if (prev.acc_name != head.acc_name or prev.write_subset != head.write_subset
                        or _binop_op(prev.binop) is not op_type):
                    break
                head = prev
                guard += 1
            # Walk forward from head building the chain.
            chain: List[_Step] = []
            cur = head
            guard = 0
            while cur is not None and guard < len(all_steps) + 1:
                if (cur.acc_name != head.acc_name or cur.write_subset != head.write_subset
                        or _binop_op(cur.binop) is not op_type or cur in used):
                    break
                chain.append(cur)
                used.add(cur)
                nxt = steps_by_read.get(cur.write_final)
                cur = nxt
                guard += 1
            if len(chain) >= 2:
                # Require a clean linear chain: each intermediate acc node has exactly the
                # step-to-step wiring and nothing else touches acc[slot] out of chain order.
                if self._chain_is_clean(st, chain):
                    chains.append((op_type, chain))
        return chains

    def _chain_is_clean(self, st: SDFGState, chain: List[_Step]) -> bool:
        # Every intermediate accumulator node (written by step k, read by step k+1) must
        # have exactly one out edge (into step k+1's binop) and one in edge.
        for k in range(len(chain) - 1):
            mid = chain[k].write_final
            if mid is not chain[k + 1].acc_read_node:
                return False
            if len(st.out_edges(mid)) != 1 or len(st.in_edges(mid)) != 1:
                return False
        return True

    def _apply_fusion(self, sdfg: SDFG, st: SDFGState, op_type: type, chain: List[_Step]) -> None:
        from dace import Memlet

        op_str = _FOLDABLE_OPS[op_type]
        first = chain[0]
        last = chain[-1]
        acc_name = first.acc_name

        # 1. Build a combined-increment operator tree over the per-step increment sources.
        #    inc_0 OP inc_1 OP ... folded left-deep into a fresh transient scalar.
        dtype = sdfg.arrays[acc_name].dtype
        inc_edges = [s.inc_edge for s in chain]

        def _fresh_scalar(name_hint: str) -> str:
            return sdfg.add_scalar(name_hint, dtype, transient=True, find_new_name=True)[0]

        # Detach the increment edges from their binops; we re-source them into the fold tree.
        for e in inc_edges:
            st.remove_edge(e)

        # Fold: acc_inc = inc_0 OP inc_1 OP ... via (len-1) binop tasklets.
        cur_scalar_node = None  # AccessNode holding the running fold
        for idx in range(1, len(inc_edges)):
            left_edge = inc_edges[idx - 1] if cur_scalar_node is None else None
            right_edge = inc_edges[idx]
            # Ordered dicts, not set literals: ``add_tasklet`` turns a set into the connector dict, so the
            # set's hash order becomes the in_connectors order and codegen emits the connector declarations
            # in that order -- byte-different C for the same input on every run. ``dict.fromkeys`` is the
            # declared ``Dict[str, typeclass]`` form and matches what the set branch builds ({k: None}).
            fold_t = st.add_tasklet(f'_fuse_red_{idx}', dict.fromkeys(['__in1', '__in2']), dict.fromkeys(['__out']),
                                    f'__out = (__in1 {op_str} __in2)')
            if cur_scalar_node is None:
                st.add_edge(left_edge.src, left_edge.src_conn, fold_t, '__in1', copy.deepcopy(left_edge.data))
            else:
                run = cur_scalar_node.data
                st.add_edge(cur_scalar_node, None, fold_t, '__in1', Memlet.from_array(run, sdfg.arrays[run]))
            st.add_edge(right_edge.src, right_edge.src_conn, fold_t, '__in2', copy.deepcopy(right_edge.data))
            fold_name = _fresh_scalar('_fused_inc')
            fold_node = st.add_access(fold_name)
            st.add_edge(fold_t, '__out', fold_node, None, Memlet.from_array(fold_name, sdfg.arrays[fold_name]))
            cur_scalar_node = fold_node

        # 2. Re-plug the first step's binop increment operand to the folded increment.
        inc_conn = first.inc_edge.dst_conn
        st.add_edge(cur_scalar_node, None, first.binop, inc_conn,
                    Memlet.from_array(cur_scalar_node.data, sdfg.arrays[cur_scalar_node.data]))

        # 3. Redirect the first step's write path to the terminal accumulator node.
        #    The first step's write_final node is an intermediate; splice it out and
        #    point the copy that fed it at the terminal node instead.
        first_write = first.write_final
        term = last.write_final
        if first_write is not term:
            in_e = st.in_edges(first_write)[0]  # copy_tasklet -> first_write
            new_data = copy.deepcopy(in_e.data)
            new_data.data = term.data
            st.add_edge(in_e.src, in_e.src_conn, term, in_e.dst_conn, new_data)
            st.remove_edge(in_e)

        # 4. Delete every downstream chain node (steps 1..k-1): their binop, staging
        #    copy tasklets + transients, and the intermediate accumulator nodes each
        #    step wrote (which the next step read). ``remove_node`` drops incident edges,
        #    so the surviving first-step accumulation feeds ``term`` alone. ``term`` and
        #    the loop-carried input node (chain[0].acc_read_node) are preserved.
        to_remove = set()
        for s in chain[1:]:
            to_remove.add(s.binop)
            to_remove.update(s.write_copies)
            to_remove.update(s.write_intermediates)
        # Intermediate accumulator nodes: every step's write target except the terminal.
        for s in chain[:-1]:
            to_remove.add(s.write_final)
        to_remove.discard(term)
        to_remove.discard(chain[0].acc_read_node)
        for n in list(to_remove):
            if n in st.nodes():
                st.remove_node(n)
        # Any transient AccessNode left dangling by the increment detach.
        for n in list(st.nodes()):
            if isinstance(n, nodes.AccessNode) and st.degree(n) == 0:
                desc = sdfg.arrays.get(n.data)
                if desc is not None and desc.transient:
                    st.remove_node(n)


__all__ = ['FuseChainedScalarReductions']
