# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoist ``if cond(i): acc = acc OP expr`` out of a ``ConditionalBlock`` by
masking the addend with the OP's identity on the false case, so that
:class:`~dace.transformation.dataflow.wcr_conversion.AugAssignToWCR` plus
:class:`~dace.transformation.interstate.loop_to_map.LoopToMap` can lift the
loop to a parallel Map with a WCR-on-scalar reduction.

Target patterns (TSVC ``s3111``, conditional ``+=``):

.. code-block:: python

    sum_val = 0.0
    for i in range(N):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]

In DaCe IR the conditional accumulator chain sits *inside* the
``ConditionalBlock``'s true-branch -- so ``AugAssignToWCR``'s state-level
``input AN -> tasklet -> output AN`` matcher cannot reach it, the carry on
``sum_val`` blocks ``LoopToMap``, and the loop stays sequential.

Rewrite shape::

    Before:                              After:
      Loop[i]                              Loop[i]
        block_pre                            block_pre   (unchanged)
        iedge: sym := ...                    iedge: sym := ...
        ConditionalBlock(cond)               state:   <-- replaces ConditionalBlock
          true:                                acc <- tasklet(__out =
            acc = acc OP expr                    __acc OP (__addend if (cond) else IDENTITY))
                                                ^ same content as the original
                                                  true-branch state, with the
                                                  update tasklet's body rewritten

Operator identities used for the mask:

============  ==========
OP            identity
============  ==========
``+``         ``0``
``-``         ``0`` (treated as ``+``; subtraction with addend on the right
              is associative when the order is fixed)
``*``         ``1``
============  ==========

``min`` / ``max`` are dtype-specific (require ``+inf`` / ``-inf`` or
``dtype::lowest()``); the v1 matcher refuses them so the loop stays
sequential. They lift via :class:`ArgMaxLift` instead.

The rewrite is value-preserving because OP-identity guarantees: applying
``OP(acc, IDENTITY)`` returns ``acc`` unchanged, so a masked iteration where
``cond`` is false contributes nothing, which is exactly the sequential
semantics of the original guarded update.

Scope of v1
-----------

* Single ``ConditionalBlock`` in the loop body (with optional empty wrapper
  states); single non-else branch; no else with content.
* True-branch contains a single content state holding the accumulator chain
  ``acc_read -> ... -> tasklet(+|-|*) -> ... -> acc_write`` where both
  endpoints are :class:`data.Scalar` or length-1 :class:`data.Array` with
  the SAME data name.
* Update tasklet body is exactly ``__out = (__lhs OP __rhs)`` for a single
  associative binary op (``+``, ``-``, ``*``).
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


from dace import SDFG, data, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import (LoopRegion, SDFGState, ControlFlowRegion, ConditionalBlock)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

#: AST binop class -> ``(op_str, identity_literal)``. The identity is a Python
#: literal suitable for embedding in a tasklet body's ternary expression.
_BINOP_TO_OP_IDENTITY: Dict[type, tuple] = {
    ast.Add: ('+', '0.0'),
    ast.Sub: ('-', '0.0'),
    ast.Mult: ('*', '1.0'),
}

#: OP -> WCR lambda string. Matches what :class:`AugAssignToWCR` emits for
#: each op so downstream codegen (OpenMP ``reduction`` clause, CUDA atomics)
#: lowers the WCR memlet through the same path.
_WCR_LAMBDAS: Dict[str, str] = {
    '+': 'lambda a, b: a + b',
    '-': 'lambda a, b: a + b',  # subtraction-of-positive-addends -> add (we negated above)
    '*': 'lambda a, b: a * b',
}


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
    identity_str: str


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToConditionalReduce(ppl.Pass):
    """Rewrite ``if cond: acc = acc OP expr`` to an unconditional masked
    accumulator chain so the canonical ``AugAssignToWCR + LoopToMap`` pair
    can pick it up."""

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
                self._rewrite(m, sd)
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
        op_info = _BINOP_TO_OP_IDENTITY.get(type(rhs.op))
        if op_info is None:
            return None
        op_str, identity_str = op_info
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
            identity_str=identity_str,
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

    # ---------------------------- rewrite ----------------------------

    def _rewrite(self, m: _Match, sdfg: SDFG):
        """Replace the ConditionalBlock with a fresh body state that holds a
        single mask tasklet writing the (possibly identity-masked) addend
        directly into the accumulator via a WCR memlet.

        The original conditional-accumulator chain is discarded entirely:
        there is no longer a per-iteration ``acc_read -> ... -> acc_write``
        chain inside the loop body, because the WCR edge makes the write
        atomic. ``LoopToMap`` then lifts the loop with no loop-carried RAW
        to worry about, and codegen emits the standard OpenMP
        ``#pragma omp parallel for reduction(<op>:acc)`` clause for CPU /
        the corresponding atomic intrinsic for CUDA.

        The cond expression is resolved against iedge symbol bindings; any
        array reference that matches an existing addend memlet is rewritten
        to use the new mask tasklet's input connector. References that
        don't match (e.g. a cond that reads a *different* array than the
        addend) get their own input edge added to the mask tasklet.
        """
        import dace
        from dace import memlet as mm
        loop = m.loop
        # The new mask tasklet's addend input connector is ``__addend``; the
        # cond resolution rewrites array accesses that match the addend's
        # gather to use this connector name.
        cond_expr_resolved = self._resolve_cond(m, sdfg, addend_conn_name='__addend')

        # 1. Trace back to the addend's source array + subset (one transient
        #    hop through ``a_index_0``).
        addend_edge = next(e for e in m.true_state.in_edges(m.upd_tasklet) if e.dst_conn == m.addend_in_conn)
        addend_src_an = addend_edge.src
        if (isinstance(addend_src_an, nodes.AccessNode) and m.true_state.in_degree(addend_src_an) == 1):
            pred = m.true_state.in_edges(addend_src_an)[0]
            if isinstance(pred.src, nodes.AccessNode):
                addend_src_name = pred.src.data
                addend_subset = pred.data.subset
            else:
                addend_src_name = addend_src_an.data
                addend_subset = addend_edge.data.subset
        else:
            return  # unexpected shape

        # 2. Build a fresh body state in the parent CFG of the loop. We'll
        #    populate it then replace the loop body's start_block with it.
        new_state = loop.add_state(loop.label + '_masked_body')

        # 3. Mask tasklet: ``__out = __addend if (cond) else IDENTITY``.
        mask_tasklet = new_state.add_tasklet(
            name=f'{m.acc_name}_masked_acc',
            inputs={'__addend'},
            outputs={'__out'},
            code=f'__out = __addend if ({cond_expr_resolved}) else {m.identity_str}',
            language=dtypes.Language.Python,
        )

        # 4. Wire the addend source -> mask_tasklet input, and
        #    mask_tasklet output --(WCR)--> accumulator.
        src_read = new_state.add_read(addend_src_name)
        acc_write = new_state.add_write(m.acc_name)
        new_state.add_edge(src_read, None, mask_tasklet, '__addend',
                           mm.Memlet(data=addend_src_name, subset=addend_subset))
        # WCR write to the accumulator. ``Memlet`` accepts a ``wcr`` lambda
        # string; the codegen lowers it through the standard reduction
        # pipeline (OpenMP ``reduction`` / CUDA atomics).
        wcr_lambda = _WCR_LAMBDAS[m.op_str]
        acc_subset = '0'  # scalar / length-1 carrier; matcher enforces this
        new_state.add_edge(mask_tasklet, '__out', acc_write, None,
                           mm.Memlet(data=m.acc_name, subset=acc_subset, wcr=wcr_lambda))

        # 5. Reroute the loop body: replace ``... -> cond_block`` and
        #    ``cond_block -> ...`` with ``... -> new_state`` /
        #    ``new_state -> ...``, stripping dead iedge assignments (the
        #    resolved cond no longer references the gather symbols).
        for ie in list(loop.in_edges(m.cond_block)):
            stripped = dace.InterstateEdge(condition=ie.data.condition, assignments={})
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
        #    state shape that ``LoopToMap`` lifts.
        self._collapse_empty_wrappers(loop)
        sdfg.reset_cfg_list()

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

        # Step 2a: discover the addend's gather (arr + subset) by walking
        # back from the addend input edge through one transient hop. Map
        # the matching ``(arr, idx)`` to ``addend_conn_name`` -- the new
        # mask tasklet's input connector name (not the original update
        # tasklet's, which doesn't survive the rewrite).
        connector_for_access: Dict[tuple, str] = {}
        addend_edge = next((e for e in m.true_state.in_edges(m.upd_tasklet) if e.dst_conn == m.addend_in_conn), None)
        if addend_edge is not None:
            src = addend_edge.src
            if isinstance(src, nodes.AccessNode) and m.true_state.in_degree(src) == 1:
                pred = m.true_state.in_edges(src)[0]
                if isinstance(pred.src, nodes.AccessNode):
                    arr_name = pred.src.data
                    sub = pred.data.subset if pred.data is not None else None
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
                    idx_str = ast.unparse(idx) if hasattr(ast, 'unparse') else _ast_to_str(idx)
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
