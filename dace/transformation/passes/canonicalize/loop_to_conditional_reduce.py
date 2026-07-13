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
identity and splits the computation into a mask tasklet and a plain accumulate::

    if cond: acc OP= x   ==>   masked_val = (x if cond else IDENTITY(OP))
                               acc        = acc OP masked_val

The masking is value-exact: a false iteration contributes ``IDENTITY(OP)``,
which ``OP`` leaves the accumulator unchanged -- exactly the sequential
semantics of the original guarded update. The accumulate tasklet reads the
masked value BARE (``acc OP __masked``), so it is the same "compute then
accumulate" shape as a dot product (``s += a[i]*b[i]``); the downstream
``reduction_to_wcr_map`` canonicalize stage lifts it to a parallel WCR-on-scalar
map whose codegen emits the tree reduction. (A single fused tasklet ``acc OP=
(x if cond else id)`` would instead be mis-lifted by ``LoopToReduce``'s
single-tasklet matcher, which keys off the top-level ``OP`` and would silently
drop the mask -- hence the two-tasklet split.)

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

#: Reduction operator string -> AST binary-operator node class, for building the
#: unconditional accumulator tasklet body ``__out = __acc OP (...)`` as an AST.
_OP_TO_AST: Dict[str, type] = {
    '+': ast.Add,
    '-': ast.Sub,
    '*': ast.Mult,
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
            if desc is not None and not getattr(desc, 'transient', False) and n.data != acc_name:
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
        if not hasattr(branch, 'nodes'):
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
        """Replace the ``ConditionalBlock`` with a fresh body state holding an
        UNCONDITIONAL masked reduction -- a mask tasklet writing a fresh
        per-iteration transient, then a plain accumulate reading it BARE::

            masked_val = (__addend if (cond) else IDENTITY)
            __out      = __acc OP __masked

        Folding the guard into the masked value via the reduction identity is
        value-exact -- a false iteration contributes the neutral element, so
        ``OP`` leaves the accumulator unchanged, which is exactly the sequential
        semantics of the original guarded update. The accumulate writes back
        with a PLAIN (non-WCR) memlet and keeps the loop-carried scalar, so the
        loop stays sequential through ``parallelize``; the downstream
        ``reduction_to_wcr_map`` stage then lifts the "compute then accumulate"
        shape to a parallel WCR-on-scalar map whose codegen emits an OpenMP
        ``reduction(OP:acc)`` clause (CPU) / a block/warp tree-reduce (GPU) --
        the fast tree reduction -- instead of the guarded atomic. See the module
        docstring for why the mask must be a SEPARATE tasklet.

        The cond expression is resolved against iedge symbol bindings; the
        addend's array gather is rewritten to the mask tasklet's ``__addend``
        connector and remaining references resolve to loop symbols. Returns
        ``True`` on a successful rewrite, ``False`` if the addend shape is
        unrecognized (loop left untouched).
        """
        loop = m.loop
        # The new tasklet's addend input connector is ``__addend``; the cond
        # resolution rewrites the addend gather to that connector name.
        cond_expr_resolved = self._resolve_cond(m, sdfg, addend_conn_name='__addend')

        # 1. Trace back to the addend's source array + subset.
        gather = self._addend_gather(m)
        if gather is None:
            return False  # unexpected shape -- leave the loop untouched
        addend_src_name, addend_subset = gather

        # 2. Two-tasklet split (mask THEN accumulate). The masked value is
        #    computed in its own elementwise tasklet writing a fresh per-iteration
        #    transient, and the accumulation reads that transient BARE:
        #
        #        masked_val = (__addend if (cond) else IDENTITY)   # mask tasklet
        #        __out      = __acc OP __masked                    # accumulate tasklet
        #
        #    This is the SAME "compute then accumulate" shape as a dot-product
        #    (``s += a[i]*b[i]``). A single fused tasklet ``__out = __acc OP
        #    (__addend if cond else id)`` would instead be mis-lifted by
        #    ``LoopToReduce``'s single-tasklet ``_extract`` matcher, which keys off
        #    the top-level ``OP`` and treats the non-accumulator operand as a bare
        #    array read -- silently dropping the mask. Splitting the mask out gives
        #    the accumulate tasklet a bare transient operand (correctly lifted) and
        #    routes the whole loop through the multi-tasklet compute-then-accumulate
        #    path, which preserves the mask.
        acc_desc = sdfg.arrays[m.acc_name]
        masked_val, _ = sdfg.add_scalar(f'{m.acc_name}_masked_val', acc_desc.dtype, transient=True, find_new_name=True)

        mask_body = self._build_mask_body(cond_expr_resolved, m.identity_value)
        acc_body = self._build_acc_body(m.op_str)

        # 3. Fresh body state with the mask + accumulate tasklets.
        new_state = loop.add_state(loop.label + '_masked_body')
        mask_tasklet = new_state.add_tasklet(name=f'{m.acc_name}_mask',
                                             inputs={'__addend'},
                                             outputs={'__out'},
                                             code=mask_body,
                                             language=dtypes.Language.Python)
        acc_tasklet = new_state.add_tasklet(name=f'{m.acc_name}_masked_acc',
                                            inputs={'__acc', '__masked'},
                                            outputs={'__out'},
                                            code=acc_body,
                                            language=dtypes.Language.Python)

        # 4. Wire: addend_src -> mask -> masked_val -> accumulate; acc_read ->
        #    accumulate -> acc_write (PLAIN, no WCR). The loop keeps its scalar RMW
        #    carry so it stays sequential until ``reduction_to_wcr_map`` lifts it.
        acc_subset = '0'  # scalar / length-1 carrier; matcher enforces this
        src_read = new_state.add_read(addend_src_name)
        masked_an = new_state.add_access(masked_val)
        acc_read = new_state.add_read(m.acc_name)
        acc_write = new_state.add_write(m.acc_name)
        new_state.add_edge(src_read, None, mask_tasklet, '__addend',
                           mm.Memlet(data=addend_src_name, subset=addend_subset))
        new_state.add_edge(mask_tasklet, '__out', masked_an, None, mm.Memlet(data=masked_val, subset=acc_subset))
        new_state.add_edge(masked_an, None, acc_tasklet, '__masked', mm.Memlet(data=masked_val, subset=acc_subset))
        new_state.add_edge(acc_read, None, acc_tasklet, '__acc', mm.Memlet(data=m.acc_name, subset=acc_subset))
        new_state.add_edge(acc_tasklet, '__out', acc_write, None, mm.Memlet(data=m.acc_name, subset=acc_subset))

        # 5. Reroute the loop body: replace ``... -> cond_block`` and
        #    ``cond_block -> ...`` with ``... -> new_state`` /
        #    ``new_state -> ...``, stripping dead iedge assignments (the
        #    resolved cond no longer references the gather symbols).
        for ie in list(loop.in_edges(m.cond_block)):
            stripped = InterstateEdge(condition=ie.data.condition, assignments={})
            loop.add_edge(ie.src, new_state, stripped)
            loop.remove_edge(ie)
        for oe in list(loop.out_edges(m.cond_block)):
            loop.add_edge(new_state, oe.dst, oe.data)
            loop.remove_edge(oe)

        # 6. Drop the now-disconnected ConditionalBlock and the original
        #    true-branch state.
        loop.remove_node(m.cond_block)
        # 7. Collapse empty wrapper states; if ``new_state`` ends up being
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

    def _build_acc_body(self, op_str: str) -> str:
        """Return Python source for the accumulate tasklet
        ``__out = __acc OP __masked`` (both operands bare connectors), built as an
        AST. The bare masked operand is what lets ``LoopToReduce`` lift the
        accumulation without dropping the mask."""
        rhs = ast.BinOp(left=ast.Name(id='__acc', ctx=ast.Load()),
                        op=_OP_TO_AST[op_str](),
                        right=ast.Name(id='__masked', ctx=ast.Load()))
        assign = ast.Assign(targets=[ast.Name(id='__out', ctx=ast.Store())], value=rhs)
        module = ast.Module(body=[assign], type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)

    def _resolve_cond(self, m: _Match, sdfg: SDFG, addend_conn_name: str = '__addend') -> str:
        """Resolve the cond expression to use only tasklet input connectors.

        Steps:

        1. Walk iedges in the loop body and collect symbol bindings
           ``sym := <expr_str>``.
        2. AST-rewrite the cond expression: substitute each ``Name(sym)``
           whose ``sym`` is in the bindings with the parsed RHS, and
           substitute any ``Subscript(Name(arr), idx)`` that matches an
           existing tasklet input edge's memlet with the corresponding
           connector ``Name(__inN)``.
        3. Unparse back to a Python expression string.

        We use AST-level substitution rather than ``dace.symbolic.subs`` to
        preserve Python subscript syntax (``a[i]``) -- ``pystr_to_symbolic``
        would convert that to sympy's ``Subscript(a, i)`` representation
        which is not valid Python.
        """
        cond_text = m.cond_codeblock.as_string.strip()

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

        # Step 2b: AST-rewrite the cond.
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
                    return node
                arr_name = node.value.id
                idx = node.slice
                if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                    idx = idx.value
                try:
                    idx_str = ast.unparse(idx)
                except Exception:
                    return node
                key = (arr_name, (idx_str, ))
                conn = connector_for_access.get(key)
                if conn is None:
                    return node
                return ast.copy_location(ast.Name(id=conn, ctx=ast.Load()), node)

        try:
            cond_ast = ast.parse(cond_text, mode='eval').body
        except SyntaxError:
            return cond_text
        new_ast = _Subst().visit(cond_ast)
        ast.fix_missing_locations(new_ast)
        try:
            return ast.unparse(new_ast)
        except Exception:
            return cond_text

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
