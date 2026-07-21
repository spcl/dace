# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower ``if cond(i): acc = acc OP expr`` (a guarded reduction) to an
UNCONDITIONAL masked reduction, so the standard reduction machinery lifts it to
the fast tree-reduction -- an OpenMP ``reduction(OP:acc)`` clause on CPU / a
block-warp tree-reduce on GPU -- instead of the per-passing-thread guarded
atomic the raw conditional lowers to.

Target patterns (TSVC ``s3111``, conditional ``+=``):

.. code-block:: python

    sum_val = 0.0
    for i in range(N):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]

In DaCe IR the conditional accumulator chain sits *inside* the
``ConditionalBlock``'s true-branch, and codegen lowers the guarded WCR write to
``reduce_atomic`` (one atomic per passing thread) -- the slow path.

The rewrite folds the guard into the accumulated value using the reduction
identity, by splicing a mask tasklet in front of the update tasklet's addend
input and hoisting the (now unconditional) branch body into the loop::

    if cond: acc OP= x   ==>   masked_val = (x if cond else IDENTITY(OP))
                               acc        = acc OP masked_val

The masking is value-exact: a false iteration contributes ``IDENTITY(OP)``,
which ``OP`` leaves the accumulator unchanged -- exactly the sequential
semantics of the original guarded update. The update tasklet is reused as-is and
reads the masked value BARE (``acc OP __masked``), so it is the same "compute
then accumulate" shape as a dot product (``s += a[i]*b[i]``); the downstream
``reduction_to_wcr_map`` canonicalize stage lifts it to a parallel WCR-on-scalar
map whose codegen emits the tree reduction. (A single fused tasklet ``acc OP=
(x if cond else id)`` would instead be mis-lifted by ``LoopToReduce``'s
single-tasklet matcher, which keys off the top-level ``OP`` and would silently
drop the mask -- hence the two-tasklet split.)

The addend ``x`` may be COMPUTED (``s += a[i]*a[i]``), not just a bare array
element: splicing into the branch's own state keeps the addend's producer
subgraph attached, and any array read the guard names that is not the addend's
element (here ``a[i]``, which traces to the product transient) is WIRED as an
extra ``__guardN`` input to the mask tasklet. Only a guard read that cannot be
expressed as a memlet subset -- an indirection such as ``a[b[i]]`` -- is refused.

Identities come from :func:`~dace.dtypes.reduction_identity` keyed on the
reduction type detected for the op (``+``/``-`` -> ``0``, ``*`` -> ``1``), never
a hand-maintained table.

Scope
-----

* Single ``ConditionalBlock`` in the loop body (with optional empty wrapper
  states); single non-else branch; no else with content.
* True-branch contains a single content state holding the accumulator chain
  ``acc_read -> ... -> tasklet(+|-|*) -> ... -> acc_write`` where both
  endpoints are :class:`data.Scalar` or length-1 :class:`data.Array` with
  the SAME data name.
* Update tasklet body is exactly ``__out = (__lhs OP __rhs)`` for a single
  associative binary op (``+``, ``-``, ``*``). ``min`` / ``max`` are left for
  :class:`ArgMaxLift`.
* No other writes (to non-transient arrays) inside the true-branch.

Refusals leave the loop unmodified so downstream stages still see it.
"""
import ast
import copy as _copy_module
from typing import Dict, NamedTuple, Optional


def _copy_ast(node: ast.AST) -> ast.AST:
    """Return a deep copy of an AST subtree, so each substitution lands on a
    fresh node (otherwise multiple references to the same binding share the
    same node and ``fix_missing_locations`` mishandles them)."""
    return _copy_module.deepcopy(node)


import numpy as np

from dace import SDFG, data, dtypes, properties
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import (LoopRegion, SDFGState, ControlFlowRegion, ConditionalBlock)
from dace.frontend import operations
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

#: AST binop class -> associative reduction operator string.
_BINOP_TO_OP: Dict[type, str] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
}

#: Reduction operator string -> WCR lambda. Used ONLY to reuse
#: :func:`~dace.frontend.operations.detect_reduction_type` so the neutral
#: (identity) element comes from :func:`~dace.dtypes.reduction_identity`, not a
#: hand-maintained per-op identity table. ``-`` maps to ``+`` because
#: ``acc - x1 - x2`` masks against the SAME additive identity ``0``
#: (``acc - 0 == acc``).
_OP_TO_WCR: Dict[str, str] = {
    '+': 'lambda a, b: a + b',
    '-': 'lambda a, b: a + b',
    '*': 'lambda a, b: a * b',
}


def _identity_value(op_str: str, dtype: dtypes.typeclass):
    """The neutral element of ``op_str`` in ``dtype`` as a plain Python scalar,
    from :func:`~dace.dtypes.reduction_identity` (``+``/``-`` -> ``0``,
    ``*`` -> ``1``, and by extension min -> dtype-max, max -> dtype-min). A
    masked-out iteration contributes this value, which ``OP`` leaves the
    accumulator unchanged -- exactly the sequential semantics of the original
    guarded update. Returns ``None`` if the op has no known identity."""
    redtype = operations.detect_reduction_type(_OP_TO_WCR[op_str])
    ident = dtypes.reduction_identity(dtype, redtype)
    if ident is None:
        return None
    return ident.item() if isinstance(ident, np.generic) else ident


class _Match(NamedTuple):
    loop: LoopRegion
    cond_block: ConditionalBlock
    true_branch: ControlFlowRegion
    true_state: SDFGState
    cond_codeblock: properties.CodeBlock
    acc_name: str
    upd_tasklet: nodes.Tasklet
    acc_in_conn: str
    addend_in_conn: str
    out_conn: str
    op_str: str
    identity_value: object


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToConditionalReduce(ppl.Pass):
    """Rewrite ``if cond: acc = acc OP expr`` to an unconditional masked
    reduction (mask tasklet + plain accumulate) so the ``reduction_to_wcr_map``
    stage lowers it to a tree-reduction (OMP ``reduction`` clause / GPU block
    reduce) instead of a guarded atomic."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        rewritten = 0
        for sd in sdfg.all_sdfgs_recursive():
            for region in list(sd.all_control_flow_regions()):
                if not (isinstance(region, LoopRegion) and region.loop_variable):
                    continue
                # Stale-snapshot guard.
                if region.parent_graph is None or region not in region.parent_graph.nodes():
                    continue
                m = self._match(region, sd)
                if m is None:
                    continue
                if self._rewrite(m, sd):
                    rewritten += 1
        return rewritten or None

    # --------------------------- match ---------------------------

    def _match(self, loop: LoopRegion, sdfg: SDFG) -> Optional[_Match]:
        # Find exactly one ConditionalBlock; other body blocks must be empty SDFGStates.
        cond_blocks = []
        for b in loop.nodes():
            if isinstance(b, ConditionalBlock):
                cond_blocks.append(b)
            elif isinstance(b, SDFGState):
                if len(b.nodes()) != 0:
                    return None  # non-empty non-conditional content -> not our pattern
            else:
                return None  # nested LoopRegion or other CFG -> refuse
        if len(cond_blocks) != 1:
            return None
        cb = cond_blocks[0]

        # Exactly one non-else branch; no else with content.
        non_else = [(c, br) for c, br in cb.branches if c is not None]
        else_branches = [(c, br) for c, br in cb.branches if c is None]
        if len(non_else) != 1:
            return None
        cond_codeblock, true_branch = non_else[0]
        for _c, br in else_branches:
            if self._branch_has_content(br):
                return None

        # True-branch must contain exactly one content SDFGState plus optional empties.
        content_states = []
        for n in true_branch.nodes():
            if isinstance(n, SDFGState):
                if len(n.nodes()) > 0:
                    content_states.append(n)
            else:
                return None  # nested control flow -> refuse
        if len(content_states) != 1:
            return None
        true_state = content_states[0]

        # Locate the unique accumulator: an AN that's a pure source AND another
        # AN with the same data name that's a pure sink.
        sources = {
            n.data: n
            for n in true_state.data_nodes() if true_state.in_degree(n) == 0 and true_state.out_degree(n) > 0
        }
        sinks = {
            n.data: n
            for n in true_state.data_nodes() if true_state.in_degree(n) > 0 and true_state.out_degree(n) == 0
        }
        acc_candidates = []
        for name in sources:
            if name not in sinks:
                continue
            desc = sdfg.arrays.get(name)
            if desc is None:
                continue
            if isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1, )):
                acc_candidates.append(name)
        if len(acc_candidates) != 1:
            return None
        acc_name = acc_candidates[0]

        # The true-branch may have only ONE terminal AccessNode (the accumulator).
        # Multiple sinks => extra writes the rewrite would drop.
        sink_ans = [n for n in true_state.data_nodes() if true_state.in_degree(n) > 0 and true_state.out_degree(n) == 0]
        if len(sink_ans) != 1 or sink_ans[0].data != acc_name:
            return None
        # And no non-transient writes other than the accumulator.
        for n in sink_ans:
            desc = sdfg.arrays.get(n.data)
            if desc is not None and not desc.transient and n.data != acc_name:
                return None

        # Walk back from the sink to find the update tasklet.
        upd_tasklet = self._walk_back_to_update_tasklet(true_state, sinks[acc_name])
        if upd_tasklet is None:
            return None

        # Parse the update tasklet's body: ``__out = (__lhs OP __rhs)``.
        if upd_tasklet.code.language != dtypes.Language.Python:
            return None
        try:
            tree = ast.parse((upd_tasklet.code.as_string or '').strip())
        except SyntaxError:
            return None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return None
        assign = tree.body[0]
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
            return None
        out_conn = assign.targets[0].id
        rhs = assign.value
        if not isinstance(rhs, ast.BinOp):
            return None
        op_str = _BINOP_TO_OP.get(type(rhs.op))
        if op_str is None:
            return None
        identity_value = _identity_value(op_str, sdfg.arrays[acc_name].dtype)
        if identity_value is None:
            return None
        if not (isinstance(rhs.left, ast.Name) and isinstance(rhs.right, ast.Name)):
            return None
        lhs_name, rhs_name = rhs.left.id, rhs.right.id

        # Identify which tasklet input is the accumulator vs the addend by
        # walking back from each input edge to its source AN.
        in_edges = [e for e in true_state.in_edges(upd_tasklet) if e.data is not None and not e.data.is_empty()]
        if len(in_edges) != 2:
            return None
        acc_in_conn = None
        addend_in_conn = None
        for e in in_edges:
            src_an = self._trace_back_to_source_an(true_state, e.src)
            if src_an is not None and src_an.data == acc_name:
                acc_in_conn = e.dst_conn
            else:
                addend_in_conn = e.dst_conn
        if acc_in_conn is None or addend_in_conn is None or acc_in_conn == addend_in_conn:
            return None
        # The connector names in the tasklet body must match.
        if {acc_in_conn, addend_in_conn} != {lhs_name, rhs_name}:
            return None

        # ``-``: the accumulator must be on the LEFT for associativity
        # (``acc - x1 - x2`` is order-independent; ``x1 - acc - x2`` is not).
        if op_str == '-' and acc_in_conn != lhs_name:
            return None

        return _Match(
            loop=loop,
            cond_block=cb,
            true_branch=true_branch,
            true_state=true_state,
            cond_codeblock=cond_codeblock,
            acc_name=acc_name,
            upd_tasklet=upd_tasklet,
            acc_in_conn=acc_in_conn,
            addend_in_conn=addend_in_conn,
            out_conn=out_conn,
            op_str=op_str,
            identity_value=identity_value,
        )

    # ------------------------- match helpers -------------------------

    def _branch_has_content(self, branch) -> bool:
        if not isinstance(branch, ControlFlowRegion):
            return False
        for n in branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return True
            if not isinstance(n, SDFGState):
                return True
        return False

    def _walk_back_to_update_tasklet(self, state: SDFGState, sink_an: nodes.AccessNode) -> Optional[nodes.Tasklet]:
        """Walk back from ``sink_an`` through intermediate transient
        AccessNodes and identity tasklets (``__out = __inp``) until the
        update tasklet (whose body is an associative binop on two inputs).
        """
        cur = sink_an
        while True:
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            upstream = ins[0].src
            if isinstance(upstream, nodes.AccessNode):
                cur = upstream
                continue
            if not isinstance(upstream, nodes.Tasklet):
                return None
            # Is this an identity passthrough? Walk further back.
            try:
                tree = ast.parse((upstream.code.as_string or '').strip())
            except SyntaxError:
                return upstream  # treat as the update; downstream will refuse
            if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
                    and isinstance(tree.body[0].value, ast.Name)):
                # Identity tasklet -> step over.
                t_ins = list(state.in_edges(upstream))
                if len(t_ins) != 1:
                    return None
                cur = t_ins[0].src
                if not isinstance(cur, nodes.AccessNode):
                    return None
                continue
            return upstream

    def _trace_back_to_source_an(self, state: SDFGState, start) -> Optional[nodes.AccessNode]:
        """Walk back from ``start`` through transient intermediate AccessNodes
        to the source AccessNode (the AN with ``in_degree == 0``). Returns
        ``None`` on ambiguity (multi-in)."""
        cur = start
        while True:
            if not isinstance(cur, nodes.AccessNode):
                return None
            if state.in_degree(cur) == 0:
                return cur
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            cur = ins[0].src

    def _addend_gather(self, m: _Match):
        """Return ``(array_name, subset)`` for the addend read, or ``None`` if
        the shape is unrecognized. Handles BOTH the raw transient-hop form
        (``arr -> arr_index(transient) -> tasklet``, as the frontend emits) AND
        the folded direct-read form (``arr -> tasklet``) the scalar-slice fold
        passes produce mid-pipeline -- in the folded form the addend source
        AccessNode is a pure array read (``in_degree == 0``) and the gather
        subset lives on the tasklet input edge itself."""
        addend_edge = next((e for e in m.true_state.in_edges(m.upd_tasklet) if e.dst_conn == m.addend_in_conn), None)
        if addend_edge is None or not isinstance(addend_edge.src, nodes.AccessNode):
            return None
        src = addend_edge.src
        if m.true_state.in_degree(src) == 1:
            pred = m.true_state.in_edges(src)[0]
            if isinstance(pred.src, nodes.AccessNode) and pred.data is not None:
                return pred.src.data, pred.data.subset
        return src.data, addend_edge.data.subset

    # ---------------------------- rewrite ----------------------------

    def _rewrite(self, m: _Match, sdfg: SDFG):
        """Turn the guarded update into an UNCONDITIONAL masked reduction by
        SPLICING a mask tasklet in front of the update tasklet's addend input
        and hoisting the (now unconditional) true-branch state into the loop::

            masked_val = (__addend if (cond) else IDENTITY)   # spliced-in mask
            __out      = __acc OP __masked                    # the ORIGINAL update

        Folding the guard into the masked value via the reduction identity is
        value-exact -- a false iteration contributes the neutral element, so
        ``OP`` leaves the accumulator unchanged, which is exactly the sequential
        semantics of the original guarded update. The update tasklet keeps its
        PLAIN (non-WCR) write and the loop-carried scalar, so the loop stays
        sequential through ``parallelize``; the downstream ``reduction_to_wcr_map``
        stage then lifts the "compute then accumulate" shape to a parallel
        WCR-on-scalar map whose codegen emits an OpenMP ``reduction(OP:acc)``
        clause (CPU) / a block/warp tree-reduce (GPU) -- the fast tree reduction
        -- instead of the guarded atomic. See the module docstring for why the
        mask must be a SEPARATE tasklet.

        Reusing the true-branch state (rather than synthesising a fresh one) is
        what makes a COMPUTED addend work: the addend's producer subgraph
        (``a -> a_index -> _Mult_ -> product``, for ``s += a[i]*a[i]``) lives in
        that state, so hoisting the state carries the producer with it and the
        mask reads a genuinely-defined value. The matcher already guarantees the
        state's only non-transient write is the accumulator, so hoisting it out
        of the guard moves no stores -- only the addend's loads/arithmetic, whose
        masked-out results are discarded by the identity.

        The cond expression is resolved against iedge symbol bindings; the
        addend's array gather is rewritten to the mask's ``__addend`` connector
        and every array read the guard still names is WIRED as an additional
        mask input. Returns ``True`` on a successful rewrite, ``False`` if a
        guard read cannot be expressed as a mask input (loop left untouched).
        """
        loop = m.loop
        true_state = m.true_state
        # The mask tasklet's addend input connector is ``__addend``; the cond
        # resolution rewrites the addend gather to that connector name, and every
        # OTHER array read the guard names becomes a wired ``__guardN`` input.
        guard_inputs: Dict[str, tuple] = {}
        cond_expr_resolved = self._resolve_cond(m, sdfg, addend_conn_name='__addend', guard_inputs=guard_inputs)
        if cond_expr_resolved is None:
            return False  # a guard read is not expressible as a mask input -- leave the loop untouched

        # 1. The mask tasklet: ``__addend`` plus one connector per wired guard read.
        acc_desc = sdfg.arrays[m.acc_name]
        masked_val, _ = sdfg.add_scalar(f'{m.acc_name}_masked_val', acc_desc.dtype, transient=True, find_new_name=True)
        mask_body = self._build_mask_body(cond_expr_resolved, m.identity_value)
        # dict.fromkeys, not a set: ``guard_inputs`` is already in AST-walk order and add_tasklet turns
        # the argument into the connector dict -- a set would randomize the emitted ``const T __guardN``
        # declaration order per process.
        mask_tasklet = true_state.add_tasklet(name=f'{m.acc_name}_mask',
                                              inputs=dict.fromkeys(['__addend', *guard_inputs]),
                                              outputs=dict.fromkeys(['__out']),
                                              code=mask_body,
                                              language=dtypes.Language.Python)

        # 2. Splice the mask between the addend's producer and the update tasklet:
        #    ``producer -> update.addend`` becomes ``producer -> mask -> masked_val
        #    -> update.addend``. The update tasklet is left untouched -- the matcher
        #    already proved its body is ``__out = __acc OP __addend`` with BARE
        #    operands, i.e. exactly the "compute then accumulate" shape (as in a
        #    dot product ``s += a[i]*b[i]``) that ``reduction_to_wcr_map`` lifts
        #    while preserving the mask. A single fused tasklet ``__out = __acc OP
        #    (__addend if cond else id)`` would instead be mis-lifted by
        #    ``LoopToReduce``'s single-tasklet ``_extract`` matcher, which keys off
        #    the top-level ``OP`` and would silently drop the mask.
        addend_edge = next(e for e in true_state.in_edges(m.upd_tasklet) if e.dst_conn == m.addend_in_conn)
        acc_subset = '0'  # scalar / length-1 carrier; matcher enforces this
        masked_an = true_state.add_access(masked_val)
        true_state.add_edge(addend_edge.src, addend_edge.src_conn, mask_tasklet, '__addend',
                            _copy_module.deepcopy(addend_edge.data))
        true_state.remove_edge(addend_edge)
        true_state.add_edge(mask_tasklet, '__out', masked_an, None, mm.Memlet(data=masked_val, subset=acc_subset))
        true_state.add_edge(masked_an, None, m.upd_tasklet, m.addend_in_conn,
                            mm.Memlet(data=masked_val, subset=acc_subset))

        # 3. Wire the guard's own array reads as real mask inputs, so the folded
        #    cond evaluates against genuine data instead of an unbound name (the
        #    ``if a[i] > 0: s += a[i]*a[i]`` case, where the addend traces to the
        #    product transient and the guard's ``a[i]`` cannot map onto it).
        for conn, (arr_name, idx_str) in guard_inputs.items():
            true_state.add_edge(true_state.add_read(arr_name), None, mask_tasklet, conn,
                                mm.Memlet(data=arr_name, subset=idx_str))

        # 4. Hoist the true-branch state into the loop in the ConditionalBlock's
        #    place, stripping dead iedge assignments (the resolved cond no longer
        #    references the gather symbols).
        was_start = (m.cond_block is loop.start_block)
        m.true_branch.remove_node(true_state)
        loop.add_node(true_state, is_start_block=was_start, ensure_unique_name=True)
        for ie in list(loop.in_edges(m.cond_block)):
            stripped = InterstateEdge(condition=ie.data.condition, assignments={})
            loop.add_edge(ie.src, true_state, stripped)
            loop.remove_edge(ie)
        for oe in list(loop.out_edges(m.cond_block)):
            loop.add_edge(true_state, oe.dst, oe.data)
            loop.remove_edge(oe)

        # 5. Drop the now-disconnected ConditionalBlock.
        loop.remove_node(m.cond_block)
        # 6. Collapse empty wrapper states; if ``true_state`` ends up being
        #    the unique non-empty block, the loop body is a clean single-
        #    state accumulator that ``reduction_to_wcr_map`` lifts.
        self._collapse_empty_wrappers(loop)
        sdfg.reset_cfg_list()
        return True

    def _build_mask_body(self, cond_expr: str, identity_value) -> str:
        """Return Python source for the elementwise mask tasklet
        ``__out = (__addend if (cond) else IDENTITY)``, built as an AST so the
        resolved cond sub-expression is spliced structurally (no string surgery)."""
        cond_ast = ast.parse(cond_expr, mode='eval').body
        masked = ast.IfExp(test=cond_ast,
                           body=ast.Name(id='__addend', ctx=ast.Load()),
                           orelse=ast.Constant(value=identity_value))
        assign = ast.Assign(targets=[ast.Name(id='__out', ctx=ast.Store())], value=masked)
        module = ast.Module(body=[assign], type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)

    def _resolve_cond(self,
                      m: _Match,
                      sdfg: SDFG,
                      addend_conn_name: str = '__addend',
                      guard_inputs: Optional[Dict[str, tuple]] = None) -> Optional[str]:
        """Resolve the cond expression to use only tasklet input connectors.

        Steps:

        1. Walk iedges in the loop body and collect symbol bindings
           ``sym := <expr_str>``.
        2. AST-rewrite the cond expression: substitute each ``Name(sym)``
           whose ``sym`` is in the bindings with the parsed RHS, and
           substitute any ``Subscript(Name(arr), idx)`` that matches an
           existing tasklet input edge's memlet with the corresponding
           connector ``Name(__inN)``.
        3. Any subscript that does NOT match the addend gather (the guard reads
           an element other than the addend, or the addend is a computed
           expression the guard's read cannot map onto) is allocated its own
           ``__guardN`` connector and recorded in ``guard_inputs`` as
           ``conn -> (array_name, index_str)``, for the caller to wire as a real
           mask input. Returns ``None`` if such a read is not expressible as a
           mask input -- see :meth:`_wireable_guard_read`.
        4. Unparse back to a Python expression string.

        We use AST-level substitution rather than ``dace.symbolic.subs`` to
        preserve Python subscript syntax (``a[i]``) -- ``pystr_to_symbolic``
        would convert that to sympy's ``Subscript(a, i)`` representation
        which is not valid Python.
        """
        cond_text = m.cond_codeblock.as_string.strip()
        if guard_inputs is None:
            guard_inputs = {}

        # Step 1: collect iedge bindings (parsed as ASTs once).
        binding_asts: Dict[str, ast.AST] = {}
        for e in m.loop.all_interstate_edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                try:
                    binding_asts[lhs] = ast.parse(str(rhs), mode='eval').body
                except SyntaxError:
                    continue

        # Step 2a: discover the addend's gather (arr + subset) -- handling both
        # the transient-hop and folded direct-read forms via ``_addend_gather``
        # -- and map the matching ``(arr, idx)`` to ``addend_conn_name`` (the new
        # accumulator tasklet's input connector, so a cond that reads the SAME
        # element as the addend, e.g. ``a[i] > K``, resolves to ``__addend``).
        connector_for_access: Dict[tuple, str] = {}
        gather = self._addend_gather(m)
        if gather is not None:
            arr_name, sub = gather
            if sub is not None:
                try:
                    key = tuple(str(lo) for lo, _hi, _st in sub.ranges)
                    connector_for_access[(arr_name, key)] = addend_conn_name
                except Exception:
                    pass

        # Step 2b: AST-rewrite the cond. Reads that resolve to neither the addend
        # gather nor a wireable ``__guardN`` input land in ``unwireable``, which
        # turns the whole rewrite into a refusal below.
        unwireable: list = []
        wireable_guard_read = self._wireable_guard_read

        class _Subst(ast.NodeTransformer):

            def visit_Name(self, node: ast.Name):
                # Inline iedge-bound gather symbol (``a_index`` -> ``a[i]`` AST).
                # Recurse into the substituted AST so any Subscript inside it
                # also gets connector-replaced in the same pass.
                if node.id in binding_asts:
                    sub = ast.copy_location(_copy_ast(binding_asts[node.id]), node)
                    return self.visit(sub)
                return node

            def visit_Subscript(self, node: ast.Subscript):
                self.generic_visit(node)  # recurse into the subscript value/slice
                if not isinstance(node.value, ast.Name):
                    unwireable.append(node)  # subscript of a non-name (e.g. a call result)
                    return node
                arr_name = node.value.id
                idx = node.slice
                if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                    idx = idx.value
                try:
                    idx_str = ast.unparse(idx)
                except Exception:
                    unwireable.append(node)
                    return node
                key = (arr_name, (idx_str, ))
                conn = connector_for_access.get(key)
                if conn is None:
                    # Not the addend's element: wire this read as its own mask input.
                    if not wireable_guard_read(m, sdfg, arr_name, idx):
                        unwireable.append(node)
                        return node
                    conn = f'__guard{len(guard_inputs)}'
                    guard_inputs[conn] = (arr_name, idx_str)
                    connector_for_access[key] = conn  # reuse one connector per distinct element
                return ast.copy_location(ast.Name(id=conn, ctx=ast.Load()), node)

        try:
            cond_ast = ast.parse(cond_text, mode='eval').body
        except SyntaxError:
            return cond_text
        new_ast = _Subst().visit(cond_ast)
        if unwireable:
            return None
        ast.fix_missing_locations(new_ast)
        try:
            return ast.unparse(new_ast)
        except Exception:
            return cond_text

    def _wireable_guard_read(self, m: _Match, sdfg: SDFG, arr_name: str, idx: ast.AST) -> bool:
        """Can ``arr_name[idx]`` be handed to the mask tasklet as a plain input
        edge? Requires that

        * ``arr_name`` is a real data descriptor (not a symbol or a builtin);
        * the index is a pure symbolic expression -- every ``Name`` in it is a
          symbol available at the loop body (see :meth:`_available_symbols`) and
          it contains no nested subscript/call. An index that is itself read from
          memory (``a[b[i]]``) is an INDIRECTION: a memlet subset is a static
          symbolic range, so the element the mask must read is not knowable
          without materialising the indirection, which this pass does not do --
          refuse and leave it to the gather passes;
        * the true-branch does not WRITE ``arr_name``. The mask's read node has
          no ordering edge against such a write, so wiring it would race with /
          reorder against the branch's own update.
        """
        desc = sdfg.arrays.get(arr_name)
        if desc is None:
            return False
        available = self._available_symbols(m, sdfg)
        for sub in ast.walk(idx):
            if isinstance(sub, (ast.Subscript, ast.Call)):
                return False  # data-dependent (indirect) index -- not a static memlet subset
            if isinstance(sub, ast.Name) and sub.id not in available:
                return False  # not a symbol we can put in a subset
        written = {n.data for n in m.true_state.data_nodes() if m.true_state.in_degree(n) > 0}
        return arr_name not in written

    def _available_symbols(self, m: _Match, sdfg: SDFG) -> set:
        """Names usable in a memlet subset at the update tasklet, taken from the
        DEFINED-symbol API rather than ``sdfg.symbols`` membership. ``sdfg.symbols``
        holds only the SDFG's GLOBAL symbols, so it sees none of the three binders a
        guard index is actually written in: a region-scoped loop iterator (mid-pipeline
        the frontend's ``i`` is renamed to a ``_loop_it_0`` bound by the LoopRegion), a
        map parameter, and an interstate-edge assignment. Gating on it silently refuses
        every such guard read, so the guarded atomic survives to codegen.

        :meth:`~dace.sdfg.state.SDFGState.symbols_defined_at` contributes what is bound
        at the tasklet's POSITION -- the iterators of all enclosing ``LoopRegion``s and
        the parameters of the dataflow scopes it sits in;
        :meth:`~dace.sdfg.state.SDFGState.defined_symbols` contributes the symbols bound
        by interstate-edge assignments, which the former does not walk. Constants are
        usable in a subset too but are not symbols, so they are added explicitly."""
        return (set(m.true_state.symbols_defined_at(m.upd_tasklet)) | set(m.true_state.defined_symbols())
                | set(sdfg.constants))

    def _collapse_empty_wrappers(self, loop: LoopRegion):
        """Eliminate empty SDFGState blocks in the loop body whose only role
        is to host an iedge -- after we stripped the dead iedge assignments
        in step 4 above, the wrapper is structurally empty AND its outgoing
        iedge carries nothing. Splice the wrapper out by reconnecting its
        predecessors directly to its successor.
        """
        changed = True
        while changed:
            changed = False
            for blk in list(loop.nodes()):
                if not isinstance(blk, SDFGState) or len(blk.nodes()) > 0:
                    continue
                in_es = list(loop.in_edges(blk))
                out_es = list(loop.out_edges(blk))
                # Only collapse states with a single successor + no iedge
                # assignments on the outgoing edge. Multi-successor empties
                # may be deliberate branch joins.
                if len(out_es) != 1:
                    continue
                oe = out_es[0]
                if oe.data.assignments:
                    continue
                is_start = (blk is loop.start_block)
                for ie in in_es:
                    loop.add_edge(ie.src, oe.dst, ie.data)
                    loop.remove_edge(ie)
                loop.remove_edge(oe)
                loop.remove_node(blk)
                if is_start:
                    loop.start_block = loop.node_id(oe.dst)
                changed = True
                break


__all__ = ['LoopToConditionalReduce']
