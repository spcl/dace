# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite same-write-set ``if/else`` → compute-then/compute-else/apply-ITE CFGs.

Arms → sequential states producing ``_then_<arr>`` / ``_else_<arr>`` temps; final
state folds them with symbolic ``ITE`` (see :mod:`dace.runtime.include.dace.ITE`),
lowered to SIMD blend. Only two-branch ``if/else``, single-state arms, shared writes
= matching element subsets, bodies = tasklets/access nodes; else raises
:class:`NotImplementedError`.
"""
import ast
import copy
import re
from typing import Dict, Optional, Tuple

import sympy

import dace
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.construction_utils import (
    assert_connector_role_matches_edges,
    copy_state_contents,
)
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl


def _wcr_apply_code(wcr_str: str, base_conn: str, acc_conn: str) -> str:
    """Render WCR reduction lambda as Python expr over two connectors.

    WCR memlet stores reduction as ``lambda a, b: <expr>``: ``a`` = dest, ``b`` =
    contribution (``c[i] += s`` -> ``lambda a,b: a+b``). Predicating WCR escape into
    ITE, ``_then`` = ``dest <op> contribution`` (base ``dest`` the WCR would have read);
    ``identity <op> contribution`` silently drops base.

    Emitted Python (``a`` -> ``base_conn``, ``b`` -> ``acc_conn``), consumed by Python
    tasklet so unparser keeps Python operator semantics — ``%`` -> ``dace::math::py_mod``
    not C truncated ``%`` (differ in sign for negatives). Never hand-write operator in C.

    :param wcr_str: memlet ``wcr`` string (2-arg lambda).
    :param base_conn: connector for the destination operand.
    :param acc_conn: connector for the contribution operand.
    :returns: Python expression string, e.g. ``"(_base + _acc)"``.
    """
    expr = ast.parse(wcr_str, mode="eval").body
    if not isinstance(expr, ast.Lambda) or len(expr.args.args) != 2:
        raise NotImplementedError(f"SameWriteSetIfElseToITECFG: unsupported WCR (expected a 2-arg lambda): {wcr_str!r}")
    a0, a1 = expr.args.args[0].arg, expr.args.args[1].arg
    sub = {a0: base_conn, a1: acc_conn}

    class _Rename(ast.NodeTransformer):

        def visit_Name(self, node):  # noqa: N802
            if node.id in sub:
                return ast.copy_location(ast.Name(id=sub[node.id], ctx=node.ctx), node)
            return node

    return ast.unparse(ast.fix_missing_locations(_Rename().visit(expr.body)))


def _rhs_is_predicate(rhs: str) -> bool:
    """Whether RHS is a boolean predicate — comparison (``a > b``) or boolean
    combination (``a or b``, ``a and b``, ``not a``).

    Decides lifted-cond transient dtype: predicate RHS -> ``bool`` element (downstream
    boolean ops + ITE mask); bare-value RHS (``b[i]`` feeding ``b[i] > 0``) keeps operand
    dtype. Upstream simplification may rewrite Python operators to C (``||`` / ``&&`` /
    ``!``) -> normalise back before parsing.

    :param rhs: RHS expression text from the interstate edge.
    :returns: ``True`` if top-level expr is Compare / BoolOp / ``not`` UnaryOp;
        ``False`` otherwise (incl. unparseable).
    """
    import ast as _ast
    text = re.sub(r"\|\|", " or ", str(rhs))
    text = re.sub(r"&&", " and ", text)
    text = re.sub(r"!\s*\(", "not (", text)
    try:
        node = _ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return False
    if isinstance(node, (_ast.Compare, _ast.BoolOp)):
        return True
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.Not):
        return True
    return False


def _symbol_has_external_consumer(sdfg: dace.SDFG, sym_name: str, defining_edge, skip_cb=None) -> bool:
    """Whether ``sym_name`` is consumed outside its own defining edge.

    Decides if lift can delete the assignment + drop the symbol. Walks interstate
    assignment RHSs/conditions, ConditionalBlock branch conditions, LoopRegion
    init/cond/update, Tasklet code, parent NestedSDFG ``symbol_mapping``.

    :param sdfg: SDFG to scan.
    :param sym_name: symbol being lifted.
    :param defining_edge: interstate edge assigning ``sym_name`` (own lhs use excluded).
    :param skip_cb: ``ConditionalBlock`` being rewritten; its branch conditions =
        consumer being rewired away -> not external.
    :returns: ``True`` if an external consumer remains.
    """
    from dace.sdfg.state import LoopRegion as _LoopRegion
    from dace.sdfg.state import ConditionalBlock as _ConditionalBlock

    only = {sym_name}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                # Skip the symbol's OWN definitions (a definition is not a consumption, and every
                # def is being deleted). Covers ALL defining edges, not just one -- a symbol
                # assigned on several edges would otherwise flag its own siblings as consumers.
                if lhs == sym_name:
                    continue
                if symbolic.symbols_in_code(str(rhs), potential_symbols=only):
                    return True
            if e.data.condition is not None:
                cond_text = e.data.condition.as_string
                if symbolic.symbols_in_code(cond_text, potential_symbols=only):
                    return True

    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, _ConditionalBlock):
            if block is skip_cb:
                continue
            for c, _ in block.branches:
                if c is None:
                    continue
                text = c.as_string if isinstance(c, CodeBlock) else str(c)
                if symbolic.symbols_in_code(text, potential_symbols=only):
                    return True

    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, _LoopRegion):
            for code in (region.loop_condition, region.update_statement, region.init_statement):
                if code is None:
                    continue
                text = code.as_string if isinstance(code, CodeBlock) else str(code)
                if symbolic.symbols_in_code(text, potential_symbols=only):
                    return True

    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet):
                code = n.code.as_string if isinstance(n.code, CodeBlock) else str(n.code)
                if symbolic.symbols_in_code(code, potential_symbols=only):
                    return True
            elif isinstance(n, dace.nodes.NestedSDFG):
                # A child nested SDFG consumes the symbol iff it binds an inner symbol to an
                # expression over it (``symbol_mapping`` VALUES live in this SDFG's scope).
                for _k, v in n.symbol_mapping.items():
                    if symbolic.symbols_in_code(str(v), potential_symbols=only):
                        return True

    if sdfg.parent_nsdfg_node is not None:
        for k, v in sdfg.parent_nsdfg_node.symbol_mapping.items():
            if k == sym_name:
                continue
            if symbolic.symbols_in_code(str(v), potential_symbols=only):
                return True

    return False


@properties.make_properties
class SameWriteSetIfElseToITECFG(ppl.Pass):
    """Rewrite same-write-set ``if/else`` blocks into 3-CFG ITE form.

    See module docstring for the pass contract.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rewrite every matching same-write-set block.

        :param sdfg: SDFG to transform in place.
        :returns: number of blocks rewritten, or ``None`` if none.
        """
        # Python frontend often emits empty entry state in an arm whose only effect is
        # an interstate symbol binding (``__sym_<x> = <x>``). That extra state bumps arm
        # node count past the single-state guard below -> broken sequential-single-arm
        # path in ``BranchNormalization``. Hoist those bindings out of every arm first so
        # they're invisible to the match check.
        from dace.transformation.passes.vectorization.branch_normalization import (  # avoid import cycle
            BranchNormalization, )
        _bn = BranchNormalization()
        for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
            for block in list(cfg.nodes()):
                if isinstance(block, ConditionalBlock):
                    _bn._hoist_branch_invariant_assignments(block)
        rewritten = 0
        for cfg in list(sdfg.all_control_flow_regions(recursive=True)):
            for block in list(cfg.nodes()):
                if not isinstance(block, ConditionalBlock):
                    continue
                if not self._matches(block):
                    continue
                self._rewrite(sdfg, block)
                rewritten += 1
        return rewritten or None

    def _matches(self, cb: ConditionalBlock) -> bool:
        """Whether ``cb`` is a rewritable conditional block.

        - **Two-arm** ``if/else``: >=1 shared element write across both single-state arms.
        - **Single-arm** ``if`` (no ``else``): lone arm IS the shared-write set (absent
          else reads pre-cb target via ITE ``else_op = arr``) -> any element-write arm
          matches.

        :param cb: candidate conditional block.
        :returns: ``True`` if ``cb`` matches a variant above.
        """
        if len(cb.branches) == 2:
            (cond0, body0), (cond1, body1) = cb.branches
            if cond0 is None or cond1 is not None:
                return False
            if not (isinstance(body0, ControlFlowRegion) and isinstance(body1, ControlFlowRegion)):
                return False
            if len(body0.nodes()) != 1 or len(body1.nodes()) != 1:
                return False
            s0, s1 = body0.nodes()[0], body1.nodes()[0]
            if not (isinstance(s0, dace.SDFGState) and isinstance(s1, dace.SDFGState)):
                return False
            for s in (s0, s1):
                for n in s.nodes():
                    if isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                        continue
                    return False
            # Arm writing same array at multiple distinct subsets (in-place chain) can't
            # use single-temp clone-redirect below -> defer to BranchNormalization
            # per-write ITE rewrite.
            if (self._arm_writes_array_at_multiple_subsets(s0) or self._arm_writes_array_at_multiple_subsets(s1)):
                return False
            shared = self._shared_writes(s0, s1)
            return bool(shared)
        if len(cb.branches) == 1:
            (cond0, body0), = cb.branches
            if cond0 is None:
                return False
            if not isinstance(body0, ControlFlowRegion):
                return False
            if len(body0.nodes()) != 1:
                return False
            s0 = body0.nodes()[0]
            if not isinstance(s0, dace.SDFGState):
                return False
            for n in s0.nodes():
                if isinstance(n, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
                    continue
                return False
            # Multi-subset in-place chain -> defer to BranchNormalization (single-temp
            # clone-redirect below would merge the writes).
            if self._arm_writes_array_at_multiple_subsets(s0):
                return False
            try:
                w0 = self._collect_write_subsets(s0)
            except NotImplementedError:
                return False
            return bool(w0)
        return False

    def _collect_write_subsets(self, state: dace.SDFGState):
        """Element-wise writes of a state.

        :param state: arm state to inspect.
        :returns: ``{arr_name: subset}`` for every element-wise write.
        :raises NotImplementedError: if any write is not element-wise (ITE rewrite can't
            produce per-element tasklets; ``_shared_writes`` swallows -> ``{}``).
        """
        from dace.transformation.passes.vectorization.utils.queries import collect_element_write_subsets
        out = collect_element_write_subsets(state)
        if out is None:
            raise NotImplementedError(
                f"SameWriteSetIfElseToITECFG: non-element write subset found in state {state.label}")
        return out

    @staticmethod
    def _arm_writes_array_at_multiple_subsets(state: dace.SDFGState) -> bool:
        """Whether some array is element-written at >1 distinct subset in ``state``.

        Single-temp clone-redirect allocates ONE ``_then_<arr>`` / ``_else_<arr>``
        (1,)-shaped scratch per array *name*, redirects every write to it
        (:meth:`_clone_with_redirect`). Collapses an in-place chain writing same array
        at different subsets (cloudsc ``zsolqa[0,3,i] += ...`` then ``zsolqa[3,0,i] -=
        ...``) into one cell, dropping all but one write -> miscompiled ITE. Such an arm
        -> :class:`BranchNormalization` (``_rewrite_writes_to_ite`` gates each write node
        -> each subset keeps own ITE). Detect so :meth:`_matches` can defer.

        :param state: an arm body state.
        :returns: ``True`` if any array has >=2 distinct write subsets.
        """
        subsets_per_array: Dict[str, set] = {}
        for n in state.nodes():
            if not isinstance(n, dace.nodes.AccessNode):
                continue
            for e in state.in_edges(n):
                if e.data is None or e.data.data is None or e.data.subset is None:
                    continue
                subsets_per_array.setdefault(n.data, set()).add(str(e.data.subset))
        return any(len(subs) > 1 for subs in subsets_per_array.values())

    def _shared_writes(self, s0: dace.SDFGState, s1: dace.SDFGState) -> dict:
        """Shared element writes of both arms.

        :param s0: the ``if`` arm state.
        :param s1: the ``else`` arm state.
        :returns: ``{arr_name: subset}`` for arrays written in both arms with identical
            subsets; empty if either arm has a non-element-wise write (caller treats as
            non-match).
        """
        try:
            w0 = self._collect_write_subsets(s0)
            w1 = self._collect_write_subsets(s1)
        except NotImplementedError:
            return {}
        shared = {}
        for k, v in w0.items():
            if k in w1 and str(v) == str(w1[k]):
                shared[k] = v
        return shared

    def _rewrite(self, sdfg: dace.SDFG, cb: ConditionalBlock):
        """Replace ``cb`` with compute-then / compute-else / apply-ITE states.

        Clones each arm (redirecting escaping writes to ``_then_<arr>`` / ``_else_<arr>``
        transients), emits one ``ITE`` tasklet per escaping target; arm-local writes stay
        inline.

        :param sdfg: SDFG used for name resolution.
        :param cb: conditional block to rewrite (removed in place).
        """
        parent = cb.parent_graph
        single_arm = (len(cb.branches) == 1)
        if single_arm:
            (cond_block, then_body), = cb.branches
            else_body = None
        else:
            (cond_block, then_body), (_, else_body) = cb.branches
        then_state: dace.SDFGState = then_body.nodes()[0]
        else_state: Optional[dace.SDFGState] = else_body.nodes()[0] if else_body is not None else None
        # ``cb`` may live in a NestedSDFG; arms' arrays live on the immediate enclosing
        # SDFG, not outermost ``sdfg``.
        local_sdfg: dace.SDFG = cb.sdfg

        # Per-arm escape set drives temp allocation + clone-redirect. Arm-local
        # intermediate nothing outside reads stays inline.
        from dace.transformation.passes.vectorization.branch_normalization import (  # avoid import cycle
            compute_arm_escape_writes, )
        escape_plan = compute_arm_escape_writes(local_sdfg, cb)
        all_escapes = escape_plan.get(0, set()) | escape_plan.get(1, set())
        if not all_escapes:
            return

        # Writing arm's subset sizes the ITE memlet; only allocate ``_<arm>_<arr>`` for
        # arms that write ``arr`` (other arm's ITE operand = pre-cb value of ``arr``).
        then_writes = self._collect_write_subsets(then_state)
        else_writes = self._collect_write_subsets(else_state) if else_state is not None else {}

        def _alloc(prefix: str, arr_name: str) -> str:
            # Element-wise writes (each escaping arm-write subset size 1, per
            # ``_collect_write_subsets``) need only a 1-element scratch, not full base
            # shape. Full-shape forced a heap VLA for symbol-shaped bases (cloudsc
            # ``zliqfrac[kfdia, klev]``) the K-dim tile path can't register-promote,
            # leaving outer scope with untouched ``new[]``. Shape (1,) keeps temp
            # Register-allocable everywhere.
            base = local_sdfg.arrays[arr_name]
            name, _ = local_sdfg.add_array(name=f"{prefix}_{arr_name}",
                                           shape=(1, ),
                                           dtype=base.dtype,
                                           storage=dace.dtypes.StorageType.Register,
                                           transient=True,
                                           find_new_name=True)
            return name

        temp_then = {arr: _alloc("_then", arr) for arr in sorted(all_escapes) if arr in then_writes}
        temp_else = {arr: _alloc("_else", arr) for arr in sorted(all_escapes) if arr in else_writes}

        # Subset per target: arms must agree when both write it (element-write
        # convention M3.1b enforces upstream).
        write_subsets = {}
        for arr in all_escapes:
            t, e = then_writes.get(arr), else_writes.get(arr)
            if t is not None and e is not None and str(t) != str(e):
                raise NotImplementedError(
                    f"SameWriteSetIfElseToITECFG: arms write {arr!r} with different subsets ({t} vs {e})")
            write_subsets[arr] = t if t is not None else e

        # New 3-CFG states in parent graph. compute-else = empty pass-through for
        # single-arm conditionals (no else to clone); apply-merge reads pre-cb value of
        # each target via ``else_op = arr`` fallback.
        ct_state = parent.add_state(f"compute_then_{cb.label}")
        ce_state = parent.add_state(f"compute_else_{cb.label}")
        am_state = parent.add_state(f"apply_ITE_{cb.label}")

        # Clone bodies redirecting only the per-arm escape writes.
        self._clone_with_redirect(then_state, ct_state, temp_then)
        if else_state is not None:
            self._clone_with_redirect(else_state, ce_state, temp_else)

        # Stitch ConditionalBlock in/out edges onto ct_state -> ce_state -> am_state and
        # drop the original block *before* resolving the cond. Resolving on a still
        # disconnected graph leaves the freshly-added states as a separate CFG component,
        # so a staged-read symbol defined on an edge ahead of ``cb`` (``a_index_0 =
        # a[_loop_it_1]`` guarding TSVC s279's nested ``if b[i] > a[i]``) no longer appears
        # to dominate ``am_state`` and gets spuriously reported as a free symbol of the
        # (nested) SDFG. ``_protected_symbols`` then refuses to inline/drop it, leaving an
        # orphaned free symbol -> "Missing symbols on nested SDFG". Wiring the states in
        # first keeps the definition dominating the merge state so the lift stages + prunes
        # it correctly.
        in_edges = list(parent.in_edges(cb))
        out_edges = list(parent.out_edges(cb))
        was_start = (parent.start_block is cb)
        for e in in_edges + out_edges:
            parent.remove_edge(e)
        parent.remove_node(cb)
        for e in in_edges:
            parent.add_edge(e.src, ct_state, e.data)
        parent.add_edge(ct_state, ce_state, dace.InterstateEdge())
        parent.add_edge(ce_state, am_state, dace.InterstateEdge())
        for e in out_edges:
            parent.add_edge(am_state, e.dst, e.data)
        if was_start:
            parent.start_block = parent.node_id(ct_state)

        # ITE tasklets. Non-writing arm contributes pre-cb value (reads original ``arr``,
        # intact because writing arm targets its private temp). Resolve cond once so the
        # symbol-lift side effect (deleting upstream assignment) fires once even when
        # writes share the cond.
        cond_text = cond_block.as_string if isinstance(cond_block, CodeBlock) else str(cond_block)
        any_subset_str = str(next(iter(write_subsets.values())))
        resolved = self._resolve_cond_to_array(local_sdfg, am_state, cond_text, any_subset_str, skip_cb=cb)
        if resolved is None:
            cond_array_name, cond_producer = None, None
        else:
            cond_array_name, cond_producer = resolved
        for arr, subset in write_subsets.items():
            then_op = temp_then.get(arr, arr)
            else_op = temp_else.get(arr, arr)
            self._emit_ite_tasklet(local_sdfg,
                                   am_state,
                                   arr,
                                   subset,
                                   then_op,
                                   else_op,
                                   cond_text,
                                   cond_array_name=cond_array_name,
                                   cond_producer=cond_producer)

        # End-of-pass invariant: every emitted state has well-formed connectors.
        for s in (ct_state, ce_state, am_state):
            assert_connector_role_matches_edges(s)

    def _clone_with_redirect(self, src: dace.SDFGState, dst: dace.SDFGState, rename: dict):
        """Deep-copy ``src`` into ``dst``; rename writes safely.

        Two write-rename rules (in order) stop the clone producing a multi-state write to
        the same array name:

        1. **Escape writes** (in ``rename``): caller pre-allocated a per-arm temp per
           escaping array needing ITE merge. Redirect write to temp; rebind memlets to
           ``[0]`` (temps length-1).
        2. **Internal transient writes** not in ``rename`` but written in the clone ->
           fresh unique name (per arm, per array). Else ``src`` and ``dst`` clone both
           write same internal transient -> downstream single-writer-scalar passes raise.

        Read-only access nodes keep original name — load-bearing for read-modify-write
        arms (``a = a + b*d``): RHS read of ``a`` must reference original array; only LHS
        write redirected.
        """
        node_map = copy_state_contents(src, dst)
        sdfg = dst.sdfg
        # Pass 1: escape-write redirects (caller-supplied).
        redirected_nodes: set = set()
        # Predicated WCR escape (``c[i] += s`` under ``if``) carries reduction on write
        # memlet; redirecting to ``_then`` temp while keeping WCR makes temp accumulate
        # onto its identity (``0`` for ``+``), dropping base ``c[i]``. Capture such edges
        # here (before subset rebound to ``[0]``), rebuild as explicit base-read
        # accumulate in Pass 3.
        wcr_escapes: list = []  # (edge, base_name, base_subset)
        for _old, new in node_map.items():
            if not isinstance(new, dace.nodes.AccessNode):
                continue
            if new.data in rename and dst.in_degree(new) > 0:
                base_name = new.data
                for ie in dst.in_edges(new):
                    if ie.data is not None and ie.data.wcr is not None:
                        wcr_escapes.append((ie, base_name, copy.deepcopy(ie.data.subset)))
                new.data = rename[base_name]
                redirected_nodes.add(new)
        for e in dst.edges():
            if (e.src in redirected_nodes or e.dst in redirected_nodes) and e.data.data in rename:
                e.data.data = rename[e.data.data]
                e.data.subset = dace.subsets.Range([(0, 0, 1)])
        # Pass 2: per-clone-unique renames for INTERNAL transient writes (no multi-state
        # writes). One fresh name per source array; every clone-side AccessNode +
        # incident memlet rewritten to it.
        internal_renames: dict = {}
        for _old, new in node_map.items():
            if not isinstance(new, dace.nodes.AccessNode):
                continue
            if new in redirected_nodes:
                continue
            arr_name = new.data
            desc = sdfg.arrays.get(arr_name)
            if desc is None or not desc.transient:
                continue
            if dst.in_degree(new) == 0:
                continue  # read-only in this clone -- keep the original name
            if arr_name not in internal_renames:
                internal_renames[arr_name] = sdfg.add_datadesc(arr_name, copy.deepcopy(desc), find_new_name=True)
            new.data = internal_renames[arr_name]
        if internal_renames:
            for e in dst.edges():
                if e.data is not None and e.data.data in internal_renames:
                    e.data.data = internal_renames[e.data.data]
        # Pass 3: rebuild predicated WCR escapes as ``_then = base <op> acc``.
        for edge, base_name, base_subset in wcr_escapes:
            self._seed_wcr_then(dst, edge, base_name, base_subset)

    def _seed_wcr_then(self, state: dace.SDFGState, edge, base_name: str, base_subset):
        """Replace a redirected WCR escape write with an explicit base accumulate.

        ``edge`` = already-redirected write ``src -[wcr]-> _then_arr[0]``. Its reduction
        needs base ``base_name[base_subset]`` the original WCR memlet would have read.
        Rebuild as Python tasklet ``_then = op(base, acc)`` (see :func:`_wcr_apply_code`
        for why operator stays Python — correct ``%``) so ITE then operand carries
        ``c[i] + s`` not ``0 + s``.

        :param state: clone (compute-then / compute-else) state.
        :param edge: redirected WCR write edge (``_then_arr`` = dst).
        :param base_name: destination array name (``c``).
        :param base_subset: original element subset of the WCR write (``i``).
        """
        write_node = edge.dst  # the _then_<arr> access node
        src = edge.src
        sdfg = state.sdfg
        if not isinstance(src, dace.nodes.AccessNode) or sdfg.arrays[src.data].total_size != 1:
            raise NotImplementedError(
                "SameWriteSetIfElseToITECFG: predicated WCR write must come from a "
                f"single-element access node (got {type(src).__name__} {getattr(src, 'data', '')!r})")
        op_code = _wcr_apply_code(edge.data.wcr, "_base", "_acc")
        then_name = write_node.data
        src_name = src.data
        src_conn = edge.src_conn
        state.remove_edge(edge)
        acc_t = state.add_tasklet(
            name=f"wcr_acc_{base_name}",
            inputs={"_base", "_acc"},
            outputs={"_o"},
            code=f"_o = {op_code}",
        )
        base_read = state.add_access(base_name)
        state.add_edge(base_read, None, acc_t, "_base", dace.Memlet(expr=f"{base_name}[{base_subset}]"))
        state.add_edge(src, src_conn, acc_t, "_acc", dace.Memlet(expr=f"{src_name}[0]"))
        state.add_edge(acc_t, "_o", write_node, None, dace.Memlet(expr=f"{then_name}[0]"))

    def _emit_ite_tasklet(self,
                          sdfg: dace.SDFG,
                          state: dace.SDFGState,
                          arr_name: str,
                          subset,
                          then_name: str,
                          else_name: str,
                          cond_text: str,
                          *,
                          cond_array_name: Optional[str] = None,
                          cond_producer: Optional[dace.nodes.AccessNode] = None):
        """Emit ``arr[subset] = ITE(_c, _t, _e)``, 3 operands wired as in-connectors.

        ``cond_array_name`` = bool transient already lifted for this cond (``None`` ->
        cond stays free-symbol text in tasklet body). ``cond_producer`` = access node the
        lift/combine tasklet wrote transient through; reuse keeps ITE on same connected
        dataflow path (else codegen may emit ITE before cond). ``None`` -> fresh read node
        (recipe-1 array, produced elsewhere)."""
        access_then = state.add_access(then_name)
        access_else = state.add_access(else_name)
        access_out = state.add_access(arr_name)
        subset_str = str(subset)

        if cond_array_name is not None:
            cond_access = cond_producer if cond_producer is not None else state.add_access(cond_array_name)
            t = state.add_tasklet(
                name=f"ITE_{arr_name}",
                inputs={"_c", "_t", "_e"},
                outputs={"_o"},
                code="_o = ITE(_c, _t, _e)",
            )
            cond_subset = "0" if sdfg.arrays[cond_array_name].total_size == 1 else subset_str
            state.add_edge(cond_access, None, t, "_c", dace.Memlet(expr=f"{cond_array_name}[{cond_subset}]"))
        else:
            t = state.add_tasklet(
                name=f"ITE_{arr_name}",
                inputs={"_t", "_e"},
                outputs={"_o"},
                code=f"_o = ITE({cond_text}, _t, _e)",
            )
        # (1,)-shaped per-arm temp (from ``_alloc``) reads at ``[0]`` regardless of write
        # subset: overwritten each iteration with the per-element value, so position 0 =
        # just-written value. Non-temp operand (absent-else single-arm fallback) names
        # original array, reads its subset.
        then_arr = sdfg.arrays.get(then_name)
        else_arr = sdfg.arrays.get(else_name)
        then_subset = "0" if then_arr is not None and tuple(then_arr.shape) == (1, ) else subset_str
        else_subset = "0" if else_arr is not None and tuple(else_arr.shape) == (1, ) else subset_str
        state.add_edge(access_then, None, t, "_t", dace.Memlet(expr=f"{then_name}[{then_subset}]"))
        state.add_edge(access_else, None, t, "_e", dace.Memlet(expr=f"{else_name}[{else_subset}]"))
        state.add_edge(t, "_o", access_out, None, dace.Memlet(expr=f"{arr_name}[{subset_str}]"))

    def _resolve_cond_to_array(self,
                               sdfg: dace.SDFG,
                               state: dace.SDFGState,
                               cond_text: str,
                               subset_str: str,
                               *,
                               skip_cb=None) -> Optional[Tuple[str, Optional[dace.nodes.AccessNode]]]:
        """Resolve ``cond_text`` to a per-lane bool transient for the ITE ``_c`` source.

        :returns: ``None`` when no transient can be produced (caller keeps cond as
            free-symbol text), or ``(array_name, producer_access)`` where
            ``producer_access`` = access node the emitted lift/combine tasklet writes
            transient through in *this* state — consumers must read from that exact node
            so dataflow stays one connected path. ``producer_access`` is ``None`` only for
            recipe 1 (name already an array produced elsewhere; caller adds own read node).

        Recipes, in order:

        1. ``cond_text`` already names an SDFG array -> return name.
        2. single symbol set by interstate edge -> lift assignment into comparison tasklet.
        3. compound expr (``(c1 or c2)``) over several such symbols -> lift each name
           recursively + emit one combine tasklet over the transients.
        """
        cond_text = cond_text.strip()
        if cond_text in sdfg.arrays:
            return cond_text, None
        direct = self._lift_interstate_cond_to_tasklet(sdfg, state, cond_text, subset_str, skip_cb=skip_cb)
        if direct is not None:
            return direct
        # Recipe 2.5: cond DIRECTLY reads an array subscript (data-dependent guard
        # ``A[i] > K`` never routed through an interstate symbol). Stage each element
        # through an in-connector -> per-lane tile; free-symbol text would inline array
        # head as scalar -> tile-op converter emits ``(int)(A)`` pointer cast. Runs before
        # compound recipe (boolean combination of interstate-defined symbols, not array
        # reads).
        array_pred = self._lift_array_predicate_cond(sdfg, state, cond_text, subset_str, skip_cb=skip_cb)
        if array_pred is not None:
            return array_pred
        return self._lift_compound_cond_to_tasklet(sdfg, state, cond_text, subset_str, skip_cb=skip_cb)

    def _lift_compound_cond_to_tasklet(self,
                                       sdfg: dace.SDFG,
                                       state: dace.SDFGState,
                                       cond_text: str,
                                       subset_str: str,
                                       *,
                                       skip_cb=None):
        """``cond_text`` = Python boolean expr over multiple symbols each set by an
        interstate-edge assignment (``(__tmp0 or __tmp1)``). Recursively lift each name,
        emit one combine tasklet whose body = ``cond_text`` with each name swapped for its
        in-connector."""
        import ast as _ast
        # Upstream simplification may rewrite Python boolean operators to C++ (``||``,
        # ``&&``, ``!``). Normalise back for the AST parser; substituted form = emitted
        # tasklet body.
        py_text = re.sub(r"\|\|", " or ", cond_text)
        py_text = re.sub(r"&&", " and ", py_text)
        py_text = re.sub(r"!\s*\(", "not (", py_text)
        try:
            tree = _ast.parse(py_text, mode="eval").body
        except SyntaxError:
            return None
        names = sorted({n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)})
        # Direct lift handles the bare-name case (``cond_text`` == one symbol). Zero names
        # (pure constant) can't be lifted here -> caller bakes inline.
        if not names or cond_text in names:
            return None

        # Recurse; every name must resolve to an array (no partial lifts).
        # ``_resolve_cond_to_array`` -> ``(array_name, producer_access)``; producer_access
        # = node a lift/combine tasklet in *this* state writes through (``None`` when name
        # was already a pre-existing array).
        lifted = {}
        for name in names:
            resolved = self._resolve_cond_to_array(sdfg, state, name, subset_str, skip_cb=skip_cb)
            if resolved is None:
                return None
            lifted[name] = resolved

        # Shape from any lifted transient; all describe the same per-lane bool result.
        any_arr = next(iter(lifted.values()))[0]
        shape = sdfg.arrays[any_arr].shape

        cond_name, _ = sdfg.add_array(name="_cond_compound",
                                      shape=shape,
                                      dtype=dace.bool_,
                                      storage=dace.dtypes.StorageType.Register,
                                      transient=True,
                                      find_new_name=True)

        # Substitute each lifted name in the normalised expr with its in-connector.
        # Word-boundary regex avoids touching names that are substrings of others.
        in_conn_names = {}
        cleaned = py_text
        for i, name in enumerate(names):
            conn = f"_c_{i}"
            in_conn_names[name] = conn
            cleaned = re.sub(rf"\b{re.escape(name)}\b", conn, cleaned)

        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name=f"combine_cond_{cond_name}",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned})")
        for name, conn in in_conn_names.items():
            lifted_name, lifted_producer = lifted[name]
            # Reuse producing access node so ``lift_cond -> transient -> combine_cond`` =
            # one connected path. Fresh ``state.add_access`` leaves producer disconnected
            # -> codegen may emit combine before lift (TSVC s271: ``_cond_compound =
            # _cond_b_index>0`` ahead of ``_cond_b_index = b``). Only already-an-array case
            # (producer ``None``) needs a fresh read node.
            lifted_access = lifted_producer if lifted_producer is not None else state.add_access(lifted_name)
            lifted_total = sdfg.arrays[lifted_name].total_size
            lifted_subset = "0" if lifted_total == 1 else subset_str
            state.add_edge(lifted_access, None, t, conn, dace.Memlet(expr=f"{lifted_name}[{lifted_subset}]"))

        cond_access = state.add_access(cond_name)
        # Lifted transient flat 1-D ``(N,)``: ``[0]`` for single-element conds, ``[0:N]``
        # for vector. ``subset_str`` (may be multi-dim ``"j, i"``) was for the legacy
        # full-source-shape transient only.
        if shape == (1, ):
            cond_subset = "0"
        elif len(shape) == 1:
            cond_subset = f"0:{shape[0]}"
        else:
            cond_subset = subset_str
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access

    def _promote_gather_indices(self, sdfg: dace.SDFG, def_edge, rhs: str) -> str:
        """Rewrite gather reads ``w[idx[i], k]`` in ``rhs`` to ``w[_gidx, k]`` by promoting each
        NESTED array-read index ``idx[i]`` to a fresh interstate integer symbol assigned on
        ``def_edge`` -- the pre-existing interstate edge that defined the cond value (``w_index =
        w[idx[i], k]``), which dominates the ITE states the lift is building. The staged read
        then carries a symbolic-indexed subset -- the representation a body gather uses and the
        tile machinery vectorizes -- instead of a nested subscript no plain memlet can express.

        Returns ``rhs`` unchanged when it contains no nested (indirect) array subscript.
        """
        if def_edge is None:
            return rhs
        arrays = set(sdfg.arrays.keys())
        try:
            expr = symbolic.SymExpr(rhs)
            base = expr.expr if isinstance(expr, symbolic.SymExpr) else expr
        except Exception:
            return rhs
        printer = symbolic.DaceSympyPrinter(arrays)
        # A NESTED array read -- an ``arr[...]`` subscript sitting inside another array
        # subscript's index (``w[idx[i], k]`` -> the ``idx[i]`` under ``w``) -- is a gather
        # index. Collect the unique ones structurally (sympy ``Subscript`` nodes), dedup by
        # their printed form so a repeated ``idx[i]`` promotes to a single symbol.
        nested: Dict[str, sympy.Basic] = {}
        for node in sympy.preorder_traversal(base):
            if isinstance(node, symbolic.Subscript) and str(node.args[0]) in arrays:
                for index_arg in node.args[1:]:
                    for sub in sympy.preorder_traversal(index_arg):
                        if isinstance(sub, symbolic.Subscript) and str(sub.args[0]) in arrays:
                            nested.setdefault(printer.doprint(sub), sub)
        if not nested:
            return rhs
        replace: Dict[sympy.Basic, sympy.Symbol] = {}
        for index_text, sub in nested.items():
            index_array = str(sub.args[0])
            gidx = 0
            while f'_gidx_{gidx}' in sdfg.symbols or f'_gidx_{gidx}' in sdfg.arrays:
                gidx += 1
            gsym = f'_gidx_{gidx}'
            sdfg.add_symbol(gsym, sdfg.arrays[index_array].dtype)
            def_edge.data.assignments[gsym] = index_text
            replace[sub] = symbolic.pystr_to_symbolic(gsym)
        return printer.doprint(base.xreplace(replace))

    def _inline_interstate_scalar_symbols(self, sdfg: dace.SDFG, rhs_text: str, exclude: set) -> Tuple[str, dict]:
        """Substitute interstate-defined scalar symbols in ``rhs_text`` with their
        definitions, recursively, so a lifted cond tasklet references only SDFG arrays
        (staged through connectors) and mapped symbols (loop iterators / kernel params).

        A cond RHS such as ``(b_index_0 > a_index_0)`` may mix an array transient
        (``b_index_0``, materialised because the arm rewrote ``b[i]``) with a bare
        interstate symbol (``a_index_0 = a[_loop_it_1]``, a staged element read that was
        only loaded). Left as free-symbol text, ``a_index_0`` becomes a free symbol of the
        (nested) SDFG whose interstate definition no longer dominates the relocated
        apply-ITE state -> validation error ("Missing symbols on nested SDFG"); it is also
        a per-lane value that must vectorise. Inlining it into ``a[_loop_it_1]`` lets the
        downstream array-read staging turn it into a connector like any other operand.

        Only names that (a) are assigned on some interstate edge, (b) are NOT SDFG arrays
        (arrays stage through connectors, never inline), and (c) are not in ``exclude`` (the
        symbol currently being lifted) are substituted.

        :param sdfg: SDFG whose interstate edges hold the scalar-symbol definitions.
        :param rhs_text: cond RHS expression to expand.
        :param exclude: symbol names to leave untouched (the lifted cond symbol itself).
        :returns: ``(expanded_text, inlined)`` where ``inlined`` maps each substituted symbol name
            to the LIST of interstate edges that assign it (caller prunes all of them together).
        """
        # Collect EVERY interstate definition of each symbol. A symbol is safe to inline only if
        # it has exactly ONE distinct RHS across all its assigning edges (a single, path-independent
        # reaching value) and is NOT self-referential (``k = k + 1`` is a loop-carried recurrence,
        # not a staged read -- inlining it would re-fire the fixpoint and grow the text). Protected
        # symbols (free / parent-bound / loop variables) are never inlined or pruned.
        all_defs: Dict[str, list] = {}
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for e in cfg.edges():
                for lhs, rhs in (e.data.assignments or {}).items():
                    all_defs.setdefault(lhs, []).append((str(rhs), e))
        protected = self._protected_symbols(sdfg)
        defs: Dict[str, Tuple[str, list]] = {}
        for lhs, deflist in all_defs.items():
            if lhs in protected:
                continue
            if len({r for r, _ in deflist}) != 1:
                continue  # path-dependent reaching value -- inlining an arbitrary one is unsound
            rhs_i = deflist[0][0]
            try:
                if lhs in set(symbolic.free_symbols_and_functions(rhs_i)):
                    continue  # self-referential recurrence, not a staged read
            except Exception:
                pass
            defs[lhs] = (rhs_i, [e for _, e in deflist])
        arrays = set(sdfg.arrays.keys())
        inlined: dict = {}  # sym -> [edges assigning it]; all pruned together by the caller
        text = rhs_text
        # Fixpoint with an iteration cap guarding against pathological (mutually-recursive) cycles;
        # single-def non-self-referential staged reads are acyclic so this converges in one/two passes.
        for _ in range(64):
            try:
                names = set(symbolic.free_symbols_and_functions(text))
            except Exception:
                break
            targets = sorted(n for n in names if n in defs and n not in arrays and n not in exclude)
            if not targets:
                break
            for name in targets:
                def_rhs, def_edges = defs[name]
                text = re.sub(rf"\b{re.escape(name)}\b", f"({def_rhs})", text)
                inlined[name] = def_edges
        return text, inlined

    def _protected_symbols(self, sdfg: dace.SDFG) -> set:
        """Symbols that must never be inlined or dropped: externally-required free symbols,
        parameters bound by the parent nested-SDFG mapping, and loop variables (whose updates live
        in loop metadata, not interstate edges). Mirrors ``SymbolDedup._protected_symbols``."""
        from dace.sdfg.state import LoopRegion
        protected = set(map(str, sdfg.free_symbols))
        if sdfg.parent_nsdfg_node is not None:
            protected |= set(sdfg.parent_nsdfg_node.symbol_mapping.keys())
        for cfg in sdfg.all_control_flow_regions(recursive=False):
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                protected.add(str(cfg.loop_variable))
        return protected

    def _drop_interstate_symbol(self, sdfg: dace.SDFG, sym: str, edges, *, skip_cb=None) -> None:
        """Delete ``sym``'s assignment from EVERY edge in ``edges``, then drop it from the symbol
        registry + parent mapping ONLY once no interstate edge still assigns it. Deleting from every
        assigning edge (not just one) avoids leaving a dangling assignment to a removed symbol; the
        protected-symbol and external-consumer guards keep a still-needed symbol in place."""
        if sym in self._protected_symbols(sdfg):
            return
        if _symbol_has_external_consumer(sdfg, sym, None, skip_cb=skip_cb):
            return
        for e in edges:
            if e is not None and sym in (e.data.assignments or {}):
                del e.data.assignments[sym]
        still_assigned = any(sym in (e.data.assignments or {})
                             for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges())
        if still_assigned:
            return
        if sym in sdfg.symbols:
            sdfg.remove_symbol(sym)
        if sdfg.parent_nsdfg_node is not None and sym in sdfg.parent_nsdfg_node.symbol_mapping:
            del sdfg.parent_nsdfg_node.symbol_mapping[sym]

    def _lift_interstate_cond_to_tasklet(self,
                                         sdfg: dace.SDFG,
                                         state: dace.SDFGState,
                                         cond_sym: str,
                                         subset_str: str,
                                         *,
                                         skip_cb=None):
        """Walk the CFG for an interstate-edge assignment to ``cond_sym``. If found, emit
        a tasklet in ``state`` computing RHS via array-read in-connectors, writing a fresh
        transient. Returns transient name, or ``None`` if no assignment found."""
        rhs = None
        def_edge = None
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for e in cfg.edges():
                assigns = e.data.assignments
                if cond_sym in assigns:
                    rhs = assigns[cond_sym]
                    def_edge = e
                    break
            if rhs is not None:
                break
        if rhs is None:
            return None
        # Cond RHS may reference OTHER interstate-defined scalar symbols (staged element
        # reads like ``a_index_0 = a[_loop_it_1]``). Inline them into their array-read
        # definitions so the lifted tasklet stages them through connectors instead of
        # leaving orphaned free symbols on the (nested) SDFG.
        rhs, inlined_syms = self._inline_interstate_scalar_symbols(sdfg, str(rhs), exclude={cond_sym})
        # Delete the cond symbol's assignment(s) + drop it only when no other consumer. With
        # consumers, the kept assignment serves them while the per-lane lift tasklet supplies the
        # vector form for the ITE. Collect ALL edges assigning cond_sym (not just ``defining_edge``)
        # so a multi-edge symbol is not left dangling by removing it globally after deleting one.
        cond_edges = [e for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges()
                      if cond_sym in (e.data.assignments or {})]
        self._drop_interstate_symbol(sdfg, cond_sym, cond_edges, skip_cb=skip_cb)
        # Prune each inlined scalar symbol's definitions once it has no remaining consumer (its only
        # use was the now-inlined cond RHS). A still-consumed symbol keeps its def -- the inlined
        # tasklet form is an equivalent per-lane duplicate.
        for sym, def_edges in inlined_syms.items():
            self._drop_interstate_symbol(sdfg, sym, def_edges, skip_cb=skip_cb)

        # A GATHER read in the cond value (``w[idx[i], k]``) is an un-representable nested
        # subscript for a plain memlet. Promote each nested index read ``idx[i]`` to a fresh
        # interstate INTEGER symbol (``_gidx = idx[i]`` on the edge feeding this state), so the
        # staged read becomes a symbolic-indexed memlet ``w[_gidx, k]`` -- the same shape a body
        # gather carries, which the tile machinery vectorizes. No-op when the RHS is not indirect.
        rhs = self._promote_gather_indices(sdfg, def_edge, rhs)

        # Collect arrays the RHS reads. Subscript ``c[i]``'s ``free_symbols`` = indices
        # only, so ``free_symbols_and_functions`` misses the array head —
        # :func:`symbolic.arrays` gets it. Union both to catch a bare ref
        # (``cond_sym = c``) and a bracketed one (``cond_sym = c[i]``).
        try:
            free_vars = set(symbolic.arrays(rhs)) | set(symbolic.free_symbols_and_functions(rhs))
        except Exception:
            return None
        arr_reads = sorted(v for v in free_vars if v in sdfg.arrays)

        # Size lifted transient to cond range. ``cond_sym``'s RHS is often a *value*
        # operand a downstream comparison consumes (``cond_sym = b[i]`` feeding
        # ``b[i] > 0.0``), not a boolean; typing ``bool`` truncates float to ``b != 0`` ->
        # wrong comparison (TSVC s271 -> all negative-b lanes wrongly taken). Bare-value
        # RHS carries RHS's own dtype (first array read's).
        #
        # RHS *itself* a predicate — comparison (``c1 > c0``) or boolean combination
        # (``a or b``, ``not a``) — -> lifted element is a bool downstream boolean ops
        # (``_cond_compound`` combine / ITE mask) consume. Typing it after operand dtype
        # (``int64`` from ``c0``) collides with ``bool`` combine output -> tile-op
        # converter rejects mixed-dtype binop. Type predicate RHSs ``bool`` -> single-dtype
        # boolean chain.
        if arr_reads and not _rhs_is_predicate(rhs):
            template = sdfg.arrays[arr_reads[0]]
            cond_dtype = template.dtype
            # Size to cond range's TOTAL element count, not full source-array shape. TSVC
            # s343 (``if bb[j, i] > 0.0``) reads one element; ``bb``'s full
            # ``(LEN_2D, LEN_2D)`` shape leaves downstream 1-D ``[k]`` merge memlet
            # dim-mismatched with 2-D transient -> ``StateFusionExtended`` refuses
            # ("expected 2, got 1"). Flat 1-D extent matches what cond holds; codegen
            # treats length-1 transient as scalar automatically.
            try:
                subset_obj = dace.subsets.Range.from_string(subset_str)
                total = subset_obj.num_elements_exact()
                shape = (int(total), ) if int(total) > 0 else (1, )
            except Exception:
                shape = template.shape
        elif arr_reads:
            # Predicate RHS with array reads: bool result, range-sized.
            cond_dtype = dace.bool_
            try:
                subset_obj = dace.subsets.Range.from_string(subset_str)
                total = subset_obj.num_elements_exact()
                shape = (int(total), ) if int(total) > 0 else (1, )
            except Exception:
                shape = (1, )
        else:
            shape = (1, )
            cond_dtype = dace.bool_
        cond_name, _ = sdfg.add_array(name=f"_cond_{cond_sym}",
                                      shape=shape,
                                      dtype=cond_dtype,
                                      storage=dace.dtypes.StorageType.Register,
                                      transient=True,
                                      find_new_name=True)

        # Substitute each array reference with its in-connector. RHS may write
        # ``arr[idx]`` or bare ``arr``; ``replace_array_accesses_with_connectors`` parses
        # via :class:`SymExpr`, rewrites both structurally so identifier overlaps (``arr``
        # vs ``arr10``) can't corrupt the result.
        in_conn_names = {arr: f"_in_{arr}_{i}" for i, arr in enumerate(arr_reads)}
        cleaned_rhs, extracted_subsets = symbolic.replace_array_accesses_with_connectors(
            rhs, in_conn_names, set(sdfg.arrays.keys()))

        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name=f"lift_cond_{cond_sym}",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned_rhs})")
        for arr, conn in in_conn_names.items():
            an = state.add_access(arr)
            # length-1 / Scalar operand = loop-invariant value (non-transient scalar
            # source like kernel arg ``c`` in ``a[i, j] > c``): must stay a 1-element
            # ``[0]`` read so the vectorizer broadcasts it (array-op-scalar); captured /
            # W-wide subset would OOB-read the 1-element source, can't be reshaped
            # (parent-fed connector). Else prefer the subset RHS wrote (``arr[i, j]`` ->
            # ``[i, j]``), else ``subset_str``.
            captured = extracted_subsets.get(arr)
            if sdfg.arrays[arr].total_size == 1:
                arr_subset = "0"
            elif captured:
                arr_subset = captured.strip("[]")
            else:
                arr_subset = subset_str
            state.add_edge(an, None, t, conn, dace.Memlet(expr=f"{arr}[{arr_subset}]"))
        cond_access = state.add_access(cond_name)
        # Flat 1-D transient indexing (as in the compound recipe): ``[0]`` single-element,
        # ``[0:N]`` vector; ``subset_str`` only for the legacy full-source-shape transient.
        if shape == (1, ):
            cond_subset = "0"
        elif len(shape) == 1:
            cond_subset = f"0:{shape[0]}"
        else:
            cond_subset = subset_str
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access

    def _lift_array_predicate_cond(self,
                                   sdfg: dace.SDFG,
                                   state: dace.SDFGState,
                                   cond_text: str,
                                   subset_str: str,
                                   *,
                                   skip_cb=None) -> Optional[Tuple[str, dace.nodes.AccessNode]]:
        """Stage a guard that DIRECTLY reads array subscripts (``A[i] > K``) into a
        per-lane bool transient, each element read through an in-connector.

        Interstate recipe (:meth:`_lift_interstate_cond_to_tasklet`) fires only when the
        whole cond is a single interstate-assigned symbol; a guard naming an array
        subscript directly falls through. Left as free-symbol text, ITE tasklet inlines
        array head as scalar -> tile-op converter emits ``(int)(A)`` pointer cast. Here
        each read is a connector (captured subset, or ``[0]`` for length-1 / scalar) ->
        operand becomes a tile; genuine symbols (kernel args like ``K``) stay free-symbol
        text.

        A guard can MIX a directly-read array transient (``b_index_0``, materialised
        because an arm rewrote ``b[i]``) with an interstate-defined scalar symbol
        (``a_index_0 = a[_loop_it_1]``, a staged element read that was only loaded) --
        TSVC s279's nested ``if b[i] > a[i]``. Left as free-symbol text, ``a_index_0``
        becomes a free symbol of the (nested) SDFG whose interstate definition no longer
        dominates the relocated apply-ITE state -> "Missing symbols on nested SDFG". Inline
        such symbols into their array-read definitions first (mirrors the interstate
        recipe) so every operand stages through a connector; genuine free symbols (kernel
        args like ``K``, loop iterators) are protected and stay.

        :returns: ``(cond_name, cond_access)`` for the lifted bool transient, or ``None``
            when cond reads no SDFG array or can't be parsed (caller keeps free-symbol
            text).
        """
        # Expand interstate staged-read symbols (no mutation yet -- the prune below only
        # fires once the recipe has committed to producing a lift, so a fall-through return
        # leaves the SDFG pristine for the caller's next recipe).
        expanded, inlined_syms = self._inline_interstate_scalar_symbols(sdfg, cond_text, exclude=set())
        # Union both accessors: ``symbolic.arrays`` catches a bracketed read (``A[i]``)
        # array-head ``free_symbols`` misses; ``free_symbols_and_functions`` catches a
        # BARE array ref (``A`` as current-lane element, the shape the ConditionalBlock
        # cond carries here -- ``threshold_data > K``). Bare read has no captured subset ->
        # wired at ``subset_str`` (write's per-lane subset).
        try:
            free_vars = set(symbolic.arrays(expanded)) | set(symbolic.free_symbols_and_functions(expanded))
        except Exception:  # noqa: BLE001 -- unparsable expr: let the caller fall back
            return None
        arr_reads = sorted(v for v in free_vars if v in sdfg.arrays)
        if not arr_reads:
            return None
        # Recipe commits: the inlined symbols now live inside the lifted tasklet body, so
        # prune each interstate definition whose only remaining consumer was this guard
        # (``_drop_interstate_symbol`` keeps a still-consumed / protected symbol in place).
        cond_text = expanded
        for sym, def_edges in inlined_syms.items():
            self._drop_interstate_symbol(sdfg, sym, def_edges, skip_cb=skip_cb)
        # Guard is a predicate -> one bool per lane; size transient to cond range (flat
        # 1-D extent, as interstate recipe).
        try:
            total = int(dace.subsets.Range.from_string(subset_str).num_elements_exact())
            shape = (total, ) if total > 0 else (1, )
        except Exception:  # noqa: BLE001
            shape = (1, )
        cond_name, _ = sdfg.add_array(name="_cond_expr",
                                      shape=shape,
                                      dtype=dace.bool_,
                                      storage=dace.dtypes.StorageType.Register,
                                      transient=True,
                                      find_new_name=True)
        in_conn_names = {arr: f"_in_{arr}_{i}" for i, arr in enumerate(arr_reads)}
        cleaned_rhs, extracted_subsets = symbolic.replace_array_accesses_with_connectors(
            cond_text, in_conn_names, set(sdfg.arrays.keys()))
        out_conn = f"_out_{cond_name}"
        t = state.add_tasklet(name="lift_cond_expr",
                              inputs=set(in_conn_names.values()),
                              outputs={out_conn},
                              code=f"{out_conn} = ({cleaned_rhs})")
        for arr, conn in in_conn_names.items():
            an = state.add_access(arr)
            captured = extracted_subsets.get(arr)
            if sdfg.arrays[arr].total_size == 1:
                arr_subset = "0"
            elif captured:
                arr_subset = captured.strip("[]")
            else:
                arr_subset = subset_str
            state.add_edge(an, None, t, conn, dace.Memlet(expr=f"{arr}[{arr_subset}]"))
        cond_access = state.add_access(cond_name)
        cond_subset = "0" if shape == (1, ) else f"0:{shape[0]}"
        state.add_edge(t, out_conn, cond_access, None, dace.Memlet(expr=f"{cond_name}[{cond_subset}]"))
        return cond_name, cond_access
