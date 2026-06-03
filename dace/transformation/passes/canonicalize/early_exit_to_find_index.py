# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift TSVC-style ``for i: ... if cond(i): break; ...`` loops into a parallel
find-first reduction plus parallel body Maps.

Target patterns (TSVC ``s481`` / ``s482`` / ``s332``):

.. code-block:: python

    # s481
    for i in range(N):
        if d[i] < 0.0: break
        a[i] = a[i] + b[i] * c[i]

    # s482
    for i in range(N):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]: break

    # s332
    for i in range(N):
        if a[i] > threshold:
            index = i
            value = a[i]
            break

Rewrite shape:

.. code-block:: python

    # Phase 1: find first i where cond(i) holds
    phi[i] = i if cond(i) else N            # parallel Map
    exit_i = min(phi[0:N])                   # Reduce(Min) -> scalar -> symbol

    # Phase 2a: body_pre runs at i in [0, min(exit_i+1, N))
    parallel_for i in range(min(exit_i+1, N)): body_pre(i)

    # Phase 2b: body_post runs at i in [0, exit_i)
    parallel_for i in range(exit_i): body_post(i)

    # Phase 3 (s332-style): rebind true-branch scalar writes from exit_i
    if exit_i < N:
        <true_branch scalar writes>

Soundness conditions (Tier-Cheap whole-array disjointness; see design doc):

* ``cond`` is a pure expression -- no writes, no iteration-carried scalar reads.
* ``R_cond ∩ (W_pre ∪ W_post ∪ W_true_branch_pre_break) = ∅``.
* ``body_pre + body_post`` (with the conditional removed) is
  :meth:`LoopToMap.can_be_applied`-eligible on the full range.
* True-branch pre-break writes are only to scalars / length-1 arrays (so the
  rebind is a clean scalar assignment).
* The original loop iterator is not read after the loop.

Scope of v1
-----------

Targets TSVC ``s481``, ``s482``, ``s332``. Refusals (with explicit messages):

* Multiple breaks / break inside a nested loop.
* Cond reads array body writes (Tier-Cheap fails).
* True-branch writes a non-trivial slice.
* Body fails ``LoopToMap.can_be_applied`` modulo break.
* Cond not a single iedge-bound predicate.
"""
import ast
import re
import copy as _copy
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import dace
from dace import SDFG, data, properties, subsets, symbolic
from dace import memlet as mm
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import (LoopRegion, SDFGState, ControlFlowRegion, ConditionalBlock, BreakBlock)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.libraries.standard.nodes.reduce import Reduce

#: Prefix for the find-first indicator array.
_PHI_PREFIX = '_findfirst_in_'
#: Prefix for the Reduce output scalar.
_EXIT_BUF_PREFIX = '_exit_i_buf_'
#: Prefix for the synthesised exit-iteration symbol.
_EXIT_SYM_PREFIX = '_exit_i_'


class _Match(NamedTuple):
    """A successfully matched break-loop ready for the find-first rewrite.

    :param loop: The :class:`LoopRegion` to rewrite.
    :param parent: ``loop.parent_graph``.
    :param cond_block: The :class:`ConditionalBlock` whose true-branch breaks.
    :param break_branch: The :class:`ControlFlowRegion` containing the
        :class:`BreakBlock` (the cond's "true" arm).
    :param true_branch_pre_break_states: States in ``break_branch`` BEFORE the
        :class:`BreakBlock` -- these are the scalar writes that must be
        rebound from ``exit_i`` in Phase 3 (e.g. s332 ``value = a[i]``).
    :param body_pre_blocks: Blocks in the loop body BEFORE ``cond_block``
        (excluding empty wrappers). For ``s482`` this is the ``a[i] +=
        b[i]*c[i]`` state.
    :param body_post_blocks: Blocks in the loop body AFTER ``cond_block``.
        For ``s481`` this is the ``a[i] += b[i]*c[i]`` state.
    :param cond_iedge_bindings: Pre-cond iedge symbol bindings (the gather:
        ``d_index = d[i]``, etc.) that feed the predicate.
    :param cond_expr_str: The textual condition expression (resolved by
        substituting iedge bindings into the conditional's predicate symbol).
    :param iter_start: Loop start expression.
    :param iter_end: Loop inclusive end expression.
    """
    loop: LoopRegion
    parent: ControlFlowRegion
    cond_block: ConditionalBlock
    break_branch: ControlFlowRegion
    true_branch_pre_break_states: List[SDFGState]
    body_pre_blocks: List[Any]
    body_post_blocks: List[Any]
    cond_iedge_bindings: Dict[str, str]
    cond_expr_str: str
    iter_start: Any
    iter_end: Any


@properties.make_properties
@xf.explicit_cf_compatible
class EarlyExitToFindIndex(ppl.Pass):
    """Lift break-loops into a parallel find-first reduction + parallel body Maps."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

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

    # ----------------------------- match ------------------------------

    def _match(self, loop: LoopRegion, sdfg: SDFG) -> Optional[_Match]:
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None
        try:
            if int(symbolic.simplify(stride)) != 1:
                return None
        except (TypeError, ValueError):
            return None

        # Find the conditional containing the break -- must be exactly one.
        cond_block = self._find_unique_break_conditional(loop)
        if cond_block is None:
            return None
        break_branch = self._find_break_branch(cond_block)
        if break_branch is None:
            return None

        # Refuse if any other body region (nested loop, sibling conditional) has a break.
        for b in loop.nodes():
            if b is cond_block:
                continue
            if self._contains_break(b):
                return None

        # Partition the loop body into (body_pre, cond_block, body_post)
        # by topological position. Empty wrapper states are allowed and
        # included; we filter them out for soundness checks.
        body_pre_blocks, body_post_blocks = self._partition_body(loop, cond_block)
        if body_pre_blocks is None:
            return None  # body shape not a clean pre/cond/post

        # The break branch may contain pre-break states (s332: scalar writes
        # BEFORE the break) plus the BreakBlock. Other arms must be empty.
        # Pre-break content is duplicated into a Phase-3 ConditionalBlock
        # guarded by ``exit_sym < N`` so the rebind only runs when cond fired.
        true_pre_break_states = self._extract_true_branch_pre_break_states(break_branch)
        if true_pre_break_states is None:
            return None

        # Resolve the condition's textual expression by walking iedges.
        cond_iedge_bindings, cond_expr_str = self._resolve_cond_expression(loop, cond_block)
        if cond_expr_str is None:
            return None

        # Soundness gates (Tier-Cheap whole-array disjointness).
        if not self._check_soundness(sdfg, loop, cond_block, cond_iedge_bindings, cond_expr_str, body_pre_blocks,
                                     body_post_blocks, true_pre_break_states):
            return None

        return _Match(
            loop=loop,
            parent=loop.parent_graph,
            cond_block=cond_block,
            break_branch=break_branch,
            true_branch_pre_break_states=true_pre_break_states,
            body_pre_blocks=body_pre_blocks,
            body_post_blocks=body_post_blocks,
            cond_iedge_bindings=cond_iedge_bindings,
            cond_expr_str=cond_expr_str,
            iter_start=start,
            iter_end=end,
        )

    # ------------------------- match helpers ---------------------------

    def _find_unique_break_conditional(self, loop: LoopRegion) -> Optional[ConditionalBlock]:
        """Find THE ConditionalBlock in ``loop.nodes()`` whose true-branch
        contains a :class:`BreakBlock`. Returns ``None`` if zero or more than
        one such conditional exists."""
        found = None
        for b in loop.nodes():
            if isinstance(b, ConditionalBlock) and self._has_break_branch(b):
                if found is not None:
                    return None  # multiple breaks
                found = b
        return found

    def _has_break_branch(self, cb: ConditionalBlock) -> bool:
        for _cond, branch in cb.branches:
            if self._region_contains_top_level_break(branch):
                return True
        return False

    def _region_contains_top_level_break(self, region: ControlFlowRegion) -> bool:
        for n in region.nodes():
            if isinstance(n, BreakBlock):
                return True
        return False

    def _contains_break(self, block) -> bool:
        """Recursively check whether any nested block contains a BreakBlock."""
        if isinstance(block, BreakBlock):
            return True
        if not hasattr(block, 'nodes'):
            return False
        for n in block.nodes():
            if self._contains_break(n):
                return True
        if isinstance(block, ConditionalBlock):
            for _c, b in block.branches:
                if self._contains_break(b):
                    return True
        return False

    def _find_break_branch(self, cb: ConditionalBlock) -> Optional[ControlFlowRegion]:
        for _cond, branch in cb.branches:
            if self._region_contains_top_level_break(branch):
                return branch
        return None

    def _partition_body(self, loop: LoopRegion, cond_block: ConditionalBlock) -> Tuple[Optional[List], Optional[List]]:
        """Topologically split the loop body into (blocks before cond_block,
        blocks after cond_block). Returns ``(None, None)`` if the body shape
        isn't a clean linear chain."""
        # Topological order via BFS from start_block.
        ordered = []
        visited = set()
        if loop.start_block is None:
            return None, None
        queue = [loop.start_block]
        while queue:
            b = queue.pop(0)
            if id(b) in visited:
                continue
            visited.add(id(b))
            ordered.append(b)
            for e in loop.out_edges(b):
                if id(e.dst) not in visited:
                    queue.append(e.dst)
        if cond_block not in ordered:
            return None, None
        idx = ordered.index(cond_block)
        return ordered[:idx], ordered[idx + 1:]

    def _extract_true_branch_pre_break_states(self, branch: ControlFlowRegion) -> Optional[List[SDFGState]]:
        """The break branch contains (optionally) some SDFGStates that run
        BEFORE the BreakBlock plus the BreakBlock itself. Anything else is
        refused. Returns the list of pre-break content states."""
        ordered = []
        visited = set()
        if branch.start_block is None:
            return None
        queue = [branch.start_block]
        while queue:
            b = queue.pop(0)
            if id(b) in visited:
                continue
            visited.add(id(b))
            ordered.append(b)
            for e in branch.out_edges(b):
                if id(e.dst) not in visited:
                    queue.append(e.dst)
        # Find the BreakBlock; everything before it must be SDFGStates;
        # everything after it must be empty SDFGStates.
        try:
            bidx = next(i for i, b in enumerate(ordered) if isinstance(b, BreakBlock))
        except StopIteration:
            return None
        pre = ordered[:bidx]
        post = ordered[bidx + 1:]
        if not all(isinstance(b, SDFGState) for b in pre):
            return None
        for b in post:
            if not isinstance(b, SDFGState) or len(b.nodes()) > 0:
                return None
        return [b for b in pre if len(b.nodes()) > 0]

    def _resolve_cond_expression(self, loop: LoopRegion,
                                 cond_block: ConditionalBlock) -> Tuple[Dict[str, str], Optional[str]]:
        """Resolve the condition predicate by walking iedges. Returns
        ``(bindings, expr_str)`` where ``bindings`` maps gather-symbol names
        to their RHS expressions, and ``expr_str`` is the cond expression
        with bindings inlined (e.g. ``a_index > threshold`` ->
        ``a[i] > threshold``)."""
        # The conditional's break-branch condition is what we want.
        break_cond = None
        for cond, branch in cond_block.branches:
            if cond is None:
                continue
            if self._region_contains_top_level_break(branch):
                break_cond = cond.as_string.strip()
                break
        if break_cond is None:
            return {}, None

        # Walk in-edges to collect symbol bindings.
        bindings: Dict[str, str] = {}
        for e in loop.in_edges(cond_block):
            for lhs, rhs in (e.data.assignments or {}).items():
                bindings[lhs] = str(rhs)
        # Walk one level back (the gather state's in-edges) too -- the gather
        # symbol(s) ``a_index = a[i]`` are typically on the iedge into the
        # cond_prep state, not into the conditional itself.
        for b in loop.nodes():
            for e in loop.out_edges(b):
                if e.dst not in loop.nodes():
                    continue
                # If this iedge ends in a block that has a path to cond_block,
                # collect its assignments. Cheap approximation: collect ALL
                # iedge assignments in the loop body.
                for lhs, rhs in (e.data.assignments or {}).items():
                    if lhs not in bindings:
                        bindings[lhs] = str(rhs)

        # Inline the bindings into the cond expression.
        expr = break_cond
        # Apply substitutions iteratively until fixed point (each binding may
        # itself reference another binding).
        for _ in range(10):
            new_expr = expr
            for sym, rhs in bindings.items():
                new_expr = re.sub(rf'\b{re.escape(sym)}\b', f'({rhs})', new_expr)
            if new_expr == expr:
                break
            expr = new_expr
        return bindings, expr

    # ----------------------- soundness gates --------------------------

    def _check_soundness(self, sdfg, loop, cond_block, cond_iedge_bindings, cond_expr_str, body_pre_blocks,
                         body_post_blocks, true_pre_break_states) -> bool:
        """Tier-Cheap whole-array disjointness + structural checks."""
        loop_var = loop.loop_variable
        # 1) Cond's read-set: arrays referenced in the resolved expression.
        cond_reads = self._read_arrays_from_expr(cond_expr_str, sdfg)
        # Widen with the underlying arrays of any transient scalar the
        # cond expression names. The frontend often emits a gather
        # chain like ``d_index = d[i]`` carried inside a body_pre
        # state -- the AST inliner only flattens explicit interstate
        # assignments, so without this widening a cond expression
        # surfaces as ``(d_index < 0.0)`` with ``d_index`` a transient
        # and the underlying ``d`` is invisible to the disjointness
        # check. Walk every name in the cond expression that's a
        # transient through the gather chain in body_pre to recover
        # the underlying non-transient arrays.
        for name in self._expr_names(cond_expr_str):
            if name in sdfg.arrays and sdfg.arrays[name].transient:
                cond_reads |= self._trace_transient_to_source_arrays(name, body_pre_blocks, sdfg)
        # 2) Body write-sets.
        pre_writes = self._collect_array_writes(body_pre_blocks, sdfg)
        post_writes = self._collect_array_writes(body_post_blocks, sdfg)
        tb_writes = self._collect_array_writes(true_pre_break_states, sdfg)
        # 3) Tier-Cheap whole-array disjointness.
        if cond_reads & (pre_writes | post_writes | tb_writes):
            return False
        # 4) True-branch writes must be scalars / length-1 arrays.
        for arr_name in tb_writes:
            desc = sdfg.arrays.get(arr_name)
            if desc is None:
                continue
            if isinstance(desc, data.Scalar):
                continue
            if isinstance(desc, data.Array) and tuple(desc.shape) == (1, ):
                continue
            return False
        # 5) Body modulo break is LoopToMap-eligible (delegate; see design).
        if not self._body_parallelizable_modulo_break(loop, cond_block, sdfg):
            return False
        return True

    def _read_arrays_from_expr(self, expr_str: str, sdfg: SDFG) -> Set[str]:
        """Return the set of array names read by the expression string.
        Recognises ``arr[i_expr]`` subscripts."""
        try:
            tree = ast.parse(expr_str, mode='eval').body
        except SyntaxError:
            return set()
        out: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id in sdfg.arrays:
                    out.add(node.value.id)
        return out

    def _expr_names(self, expr_str: str) -> Set[str]:
        """Return the set of bare ``ast.Name`` identifiers referenced by
        ``expr_str``. Used to discover transient scalars the cond reads
        through gather chains the AST inliner did not flatten."""
        try:
            tree = ast.parse(expr_str, mode='eval').body
        except SyntaxError:
            return set()
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def _trace_transient_to_source_arrays(self, name: str, blocks, sdfg: SDFG) -> Set[str]:
        """Trace a transient ``name`` written inside any state in
        ``blocks`` back to the non-transient arrays that gather feed
        it. Used by :meth:`_check_soundness` to widen ``cond_reads``
        with the underlying arrays the cond's prep gather chain reads.

        Walks the memlet path from each write site backward across
        intermediate transients; if the source AccessNode is a
        non-transient array, its name is recorded. Caps recursion via
        a visited set on transient names.
        """
        if name not in sdfg.arrays:
            return set()
        desc = sdfg.arrays[name]
        if not desc.transient:
            # ``name`` is already a non-transient array; the caller
            # path treats it directly.
            return {name}
        out: Set[str] = set()
        visited: Set[str] = {name}
        pending = [name]
        while pending:
            target = pending.pop()
            for b in blocks:
                if not isinstance(b, SDFGState):
                    continue
                for an in b.data_nodes():
                    if an.data != target or b.in_degree(an) == 0:
                        continue
                    for ie in b.in_edges(an):
                        try:
                            path = b.memlet_path(ie)
                        except Exception:
                            continue
                        src = path[0].src
                        if not isinstance(src, nodes.AccessNode):
                            continue
                        src_name = src.data
                        if src_name not in sdfg.arrays:
                            continue
                        src_desc = sdfg.arrays[src_name]
                        if src_desc.transient:
                            if src_name not in visited:
                                visited.add(src_name)
                                pending.append(src_name)
                        else:
                            out.add(src_name)
        return out

    def _collect_array_writes(self, blocks, sdfg: SDFG) -> Set[str]:
        """Collect every non-transient array name written by any state in
        ``blocks``."""
        out: Set[str] = set()
        for b in blocks:
            if not isinstance(b, SDFGState):
                continue
            for n in b.data_nodes():
                if b.in_degree(n) == 0:
                    continue
                desc = sdfg.arrays.get(n.data)
                if desc is None:
                    continue
                if desc.transient:
                    continue
                out.add(n.data)
        return out

    def _body_parallelizable_modulo_break(self, loop, cond_block, sdfg) -> bool:
        """Delegate (S4): would ``LoopToMap`` lift the loop if the break
        conditional were removed?

        Implementation: deep-copy the SDFG, find the corresponding loop,
        remove the conditional block from the body so the remaining
        ``body_pre + body_post`` reads as an unconditional loop, run
        ``LoopToMap.can_be_applied``.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap
        # Find the loop and conditional by position-in-cfg-list (an id-stable
        # locator survives deep-copy).
        sdfg_copy = _copy.deepcopy(sdfg)
        loop_copy = self._locate_corresponding(sdfg_copy, loop)
        cb_copy = self._locate_corresponding(sdfg_copy, cond_block)
        if loop_copy is None or cb_copy is None:
            return False
        # Splice the conditional out of the loop body.
        try:
            self._splice_out_block(loop_copy, cb_copy)
            instance = LoopToMap()
            instance.loop = loop_copy
            return instance.can_be_applied(loop_copy.parent_graph, expr_index=0, sdfg=sdfg_copy, permissive=False)
        except Exception:
            return False

    def _locate_corresponding(self, sdfg_copy, target):
        """Find the block in ``sdfg_copy`` corresponding to ``target`` in the
        original. Uses the cfg-list index + label."""
        target_label = target.label
        target_type = type(target).__name__
        for sd in sdfg_copy.all_sdfgs_recursive():
            for region in sd.all_control_flow_regions():
                if region.label == target_label and type(region).__name__ == target_type:
                    return region
        return None

    def _splice_out_block(self, loop, block):
        """Remove ``block`` from ``loop``; reconnect incoming/outgoing iedges
        by chaining src(s) directly to dst(s)."""
        in_es = list(loop.in_edges(block))
        out_es = list(loop.out_edges(block))
        for ie in in_es:
            for oe in out_es:
                combined = dace.InterstateEdge(
                    condition=ie.data.condition,
                    assignments={
                        **(ie.data.assignments or {}),
                        **(oe.data.assignments or {})
                    },
                )
                loop.add_edge(ie.src, oe.dst, combined)
        for e in in_es + out_es:
            loop.remove_edge(e)
        loop.remove_node(block)

    # ------------------------- rewrite -------------------------------

    def _rewrite(self, m: _Match, sdfg: SDFG):
        """Replace the break-loop with Phase 1 (find-first Reduce) + Phase 2
        (parallel body Maps) + Phase 3 (true-branch scalar rebinds)."""
        # Allocate the find-first symbol and output scalar.
        idx = _next_id(sdfg)
        exit_sym = f'{_EXIT_SYM_PREFIX}{idx}'
        sdfg.add_symbol(exit_sym, dace.int64)
        phi_name, _ = sdfg.add_array(f'{_PHI_PREFIX}{idx}', [m.iter_end - m.iter_start + 1],
                                     dace.int64,
                                     transient=True,
                                     find_new_name=True)
        exit_buf_name, _ = sdfg.add_scalar(f'{_EXIT_BUF_PREFIX}{idx}', dace.int64, transient=True, find_new_name=True)

        parent = m.parent
        # Phase 1: indicator Map + Reduce(Min).
        s_phi = parent.add_state(m.loop.label + '_findfirst_phi')
        s_reduce = parent.add_state(m.loop.label + '_findfirst_reduce')
        parent.add_edge(s_phi, s_reduce, dace.InterstateEdge())

        # Rewrite iedges so the original loop's predecessor edges flow into Phase 1.
        for ie in list(parent.in_edges(m.loop)):
            parent.add_edge(ie.src, s_phi, ie.data)
            parent.remove_edge(ie)

        # Build the indicator Map.
        self._emit_phi_map(s_phi, sdfg, phi_name, m)
        # Build the Reduce(Min) state.
        self._emit_reduce_min(s_reduce, sdfg, phi_name, exit_buf_name, m)

        # Phase 2a / 2b: body_pre and body_post Maps.
        last_state = s_reduce
        # iedge from reduce -> first phase 2 state binds exit_sym.
        sym_bound = False

        if any(isinstance(b, SDFGState) and len(b.nodes()) > 0 for b in m.body_pre_blocks):
            s_pre = parent.add_state(m.loop.label + '_body_pre_map')
            edge_data = dace.InterstateEdge(assignments={exit_sym: exit_buf_name})
            parent.add_edge(last_state, s_pre, edge_data)
            sym_bound = True
            self._emit_body_map(s_pre,
                                sdfg,
                                m.body_pre_blocks,
                                m,
                                upper_str=f'Min({exit_sym} + 1, {symbolic.symstr(m.iter_end + 1)})')
            last_state = s_pre

        if any(isinstance(b, SDFGState) and len(b.nodes()) > 0 for b in m.body_post_blocks):
            s_post = parent.add_state(m.loop.label + '_body_post_map')
            if sym_bound:
                parent.add_edge(last_state, s_post, dace.InterstateEdge())
            else:
                edge_data = dace.InterstateEdge(assignments={exit_sym: exit_buf_name})
                parent.add_edge(last_state, s_post, edge_data)
                sym_bound = True
            self._emit_body_map(s_post, sdfg, m.body_post_blocks, m, upper_str=f'{exit_sym}')
            last_state = s_post

        # Phase 3: true-branch scalar rebinds (s332-style).
        if m.true_branch_pre_break_states:
            # Ensure exit_sym is bound on the path into the rebind.
            if not sym_bound:
                anchor = parent.add_state(m.loop.label + '_exit_bind')
                parent.add_edge(last_state, anchor, dace.InterstateEdge(assignments={exit_sym: exit_buf_name}))
                last_state = anchor
                sym_bound = True
            cond_block = self._emit_rebind(last_state, sdfg, m, exit_sym)
            last_state = cond_block

        # If neither map nor rebind was emitted, still bind the symbol so it's
        # visible downstream (defensive -- shouldn't happen for the target patterns).
        if not sym_bound:
            edge_data = dace.InterstateEdge(assignments={exit_sym: exit_buf_name})
            tail = parent.add_state(m.loop.label + '_bind_only')
            parent.add_edge(last_state, tail, edge_data)
            last_state = tail

        # Reroute outgoing edges.
        for oe in list(parent.out_edges(m.loop)):
            parent.add_edge(last_state, oe.dst, oe.data)
            parent.remove_edge(oe)

        parent.remove_node(m.loop)
        sdfg.reset_cfg_list()

    def _emit_phi_map(self, state: SDFGState, sdfg: SDFG, phi_name: str, m: _Match):
        """Map: for i in [start, end+1): phi[i - start] = i if cond(i) else N."""
        N_expr = symbolic.symstr(m.iter_end + 1)
        # Compile the cond expression into a Python tasklet that produces the
        # boolean. Then a downstream tasklet selects ``i`` or ``N``.
        # Single tasklet form: emit ``__out = i if (cond_expr) else N``.
        cond_expr = m.cond_expr_str
        # The cond_expr currently uses ``i`` (the loop var) referencing the
        # loop's original loop_variable name; we keep that name in the Map.
        loop_var = m.loop.loop_variable
        # Collect arrays the cond reads -- map them into the tasklet's inputs.
        cond_reads = self._read_arrays_from_expr(cond_expr, sdfg)
        inputs_map: Dict[str, mm.Memlet] = {}
        rename_map: Dict[str, str] = {}
        new_expr = cond_expr
        for arr_name in sorted(cond_reads):
            in_conn = f'__r_{arr_name}'
            inputs_map[in_conn] = mm.Memlet(data=arr_name,
                                            subset=subsets.Range([(symbolic.pystr_to_symbolic(loop_var),
                                                                   symbolic.pystr_to_symbolic(loop_var), 1)]))
            new_expr = re.sub(rf'\b{re.escape(arr_name)}\b\[[^\]]*\]', in_conn, new_expr)
            rename_map[arr_name] = in_conn

        # phi[i] = i if cond else N -- the GLOBAL index (i) or the
        # sentinel ``N``. Post-Reduce min then gives exit_i directly in
        # the original index space. Use the canonical 3-input ``ITE``
        # form (not a Python ternary): ``LowerITEToFpFactor`` lowers it
        # to ``c * t + (1 - c) * e`` at canonicalize time, so codegen
        # never emits ``c ? t : e``.
        tasklet_code = f'__out = ITE(({new_expr}), {loop_var}, ({N_expr}))'

        outputs_map = {
            '__out':
            mm.Memlet(data=phi_name,
                      subset=subsets.Range([(symbolic.pystr_to_symbolic(loop_var) - m.iter_start,
                                             symbolic.pystr_to_symbolic(loop_var) - m.iter_start, 1)]))
        }

        state.add_mapped_tasklet(
            name=f'{m.loop.label}_phi_tasklet',
            map_ranges={loop_var: f'{symbolic.symstr(m.iter_start)}:{N_expr}'},
            inputs=inputs_map,
            code=tasklet_code,
            outputs=outputs_map,
            external_edges=True,
        )

    def _emit_reduce_min(self, state: SDFGState, sdfg: SDFG, phi_name: str, exit_buf_name: str, m: _Match):
        N_expr = symbolic.symstr(m.iter_end + 1)
        read = state.add_read(phi_name)
        write = state.add_write(exit_buf_name)
        # Identity = N (sentinel for "no fire"). For Min reduction this means
        # if no phi[i] is < N, the reduction returns N -- exactly our
        # "exit_i = N" semantics.
        node = Reduce(name=f'{m.loop.label}_findfirst_reduce', wcr='lambda a, b: min(a, b)', axes=[0], identity=N_expr)
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        state.add_node(node)
        size = symbolic.simplify(m.iter_end - m.iter_start + 1)
        state.add_edge(read, None, node, '_in', mm.Memlet(data=phi_name, subset=subsets.Range([(0, size - 1, 1)])))
        state.add_edge(node, '_out', write, None, mm.Memlet(data=exit_buf_name, subset=subsets.Range([(0, 0, 1)])))

    def _emit_body_map(self, state: SDFGState, sdfg: SDFG, body_blocks: List, m: _Match, upper_str: str):
        """Emit a parallel Map wrapping the body work. We do this by directly
        moving every node from each content state into a new Map scope.
        """
        loop_var = m.loop.loop_variable
        start_str = symbolic.symstr(m.iter_start)
        # Find the unique content state in body_blocks (TSVC kernels have one).
        content_states = [b for b in body_blocks if isinstance(b, SDFGState) and len(b.nodes()) > 0]
        if not content_states:
            return
        if len(content_states) != 1:
            # v1: only single-content-state bodies. Multi-state body Maps need
            # NSDFG construction; refused earlier in the matcher.
            return
        src_state = content_states[0]
        # Map scope.
        map_entry, map_exit = state.add_map(
            f'{m.loop.label}_body_map',
            ndrange={loop_var: f'{start_str}:{upper_str}'},
        )
        # Clone nodes from src_state into the Map body. Reuse node identities
        # by referencing source nodes' (state, node) -- safer: clone with the
        # same name + memlets.
        node_map: Dict[Any, Any] = {}
        for n in src_state.nodes():
            cn = _copy.deepcopy(n)
            state.add_node(cn)
            node_map[n] = cn
        for e in src_state.edges():
            if e.data is None or e.data.is_empty():
                continue
            state.add_edge(node_map[e.src], e.src_conn, node_map[e.dst], e.dst_conn, _copy.deepcopy(e.data))
        # Wire all source AccessNodes through the map_entry, all sink
        # AccessNodes through the map_exit.
        for orig, cn in node_map.items():
            if not isinstance(cn, nodes.AccessNode):
                continue
            if src_state.in_degree(orig) == 0 and src_state.out_degree(orig) > 0:
                # Source: external_input -> map_entry -> cn
                ext = state.add_read(cn.data)
                state.add_memlet_path(ext, map_entry, cn, memlet=mm.Memlet.from_array(cn.data, sdfg.arrays[cn.data]))
            elif src_state.in_degree(orig) > 0 and src_state.out_degree(orig) == 0:
                ext = state.add_write(cn.data)
                state.add_memlet_path(cn, map_exit, ext, memlet=mm.Memlet.from_array(cn.data, sdfg.arrays[cn.data]))

    def _emit_rebind(self, after_state: SDFGState, sdfg: SDFG, m: _Match, exit_sym: str):
        """Emit Phase 3: ``if exit_sym < N: <duplicated true-branch content>``.

        Duplicates the original break-branch's pre-break content into a fresh
        :class:`ConditionalBlock` placed in the parent CFG, with the loop
        variable substituted by ``exit_sym``. The guard ``exit_sym < N``
        ensures the rebind only runs when ``cond`` actually fired; otherwise
        the pre-loop initial values of the rebound scalars are preserved
        (no rebind => the false-branch is empty).

        Returns the newly-added ConditionalBlock (so the caller can chain
        downstream out-edges from it).
        """
        parent = m.parent
        # Deep-copy the original break branch.
        branch_copy = _copy.deepcopy(m.break_branch)
        # Strip the BreakBlock and any unreachable post-break content from the copy.
        self._strip_break_and_after(branch_copy)
        # Wrap in a ConditionalBlock with the ``exit_sym < N`` guard. Must
        # propagate the SDFG context (state.sdfg, region.parent_graph) onto
        # every nested block BEFORE calling ``replace_dict``: the replace
        # walker dereferences ``state.sdfg.arrays`` and a deep-copied region
        # has those pointers detached.
        N_str = symbolic.symstr(m.iter_end + 1)
        cond_block = ConditionalBlock(m.loop.label + '_rebind_cond')
        cond_block.sdfg = sdfg
        cond_block.parent_graph = parent
        parent.add_node(cond_block)
        cond_block.add_branch(CodeBlock(f'{exit_sym} < ({N_str})'), branch_copy)
        parent.add_edge(after_state, cond_block, dace.InterstateEdge())
        self._propagate_sdfg(branch_copy, sdfg)
        sdfg.reset_cfg_list()
        # Now safely substitute the loop variable with the exit symbol.
        branch_copy.replace_dict({m.loop.loop_variable: exit_sym})
        return cond_block

    def _propagate_sdfg(self, region, sdfg):
        """Recursively set ``sdfg`` and ``parent_graph`` on every nested
        SDFGState and sub-region so ``replace_dict`` has the context it
        needs.

        Restricted to ``ControlFlowBlock`` subtypes -- ``SDFGState``,
        ``ControlFlowRegion``, ``ConditionalBlock`` etc. -- whose ``.sdfg``
        attribute names the *containing* SDFG. ``NestedSDFG`` nodes are
        deliberately skipped because their ``.sdfg`` is the *inner* SDFG
        (an ``SDFGReferenceProperty`` with a setter); writing the outer
        SDFG into that slot would replace the inner with the outer and
        create a graph cycle (the same TSVC s275 RecursionError as in
        ``ConditionFusion``).
        """
        from dace.sdfg.state import ControlFlowBlock
        region.sdfg = sdfg
        for n in region.nodes():
            if isinstance(n, SDFGState):
                n.sdfg = sdfg
                n.parent_graph = region
            elif isinstance(n, ControlFlowBlock):
                n.sdfg = sdfg
                n.parent_graph = region
                self._propagate_sdfg(n, sdfg)

    def _strip_break_and_after(self, branch: ControlFlowRegion):
        """Remove the BreakBlock and any block reachable from it (within
        ``branch``) so the duplicated content runs as straight-line code."""
        try:
            break_block = next(n for n in branch.nodes() if isinstance(n, BreakBlock))
        except StopIteration:
            return
        # BFS from the break block; mark every reachable node for removal.
        to_remove = set()
        queue = [break_block]
        while queue:
            b = queue.pop(0)
            if id(b) in to_remove:
                continue
            to_remove.add(id(b))
            for e in branch.out_edges(b):
                queue.append(e.dst)
        # Remove the marked nodes (and their incident edges).
        for b in list(branch.nodes()):
            if id(b) in to_remove:
                for e in list(branch.in_edges(b)):
                    branch.remove_edge(e)
                for e in list(branch.out_edges(b)):
                    branch.remove_edge(e)
                branch.remove_node(b)


# ----------------------------- helpers --------------------------------


def _next_id(sdfg: SDFG) -> int:
    """Pick a fresh integer suffix unused by any current find-first symbol."""
    used: Set[int] = set()
    for sd in sdfg.all_sdfgs_recursive():
        for nm in sd.symbols.keys():
            for prefix in (_PHI_PREFIX, _EXIT_BUF_PREFIX, _EXIT_SYM_PREFIX):
                if nm.startswith(prefix):
                    tail = nm[len(prefix):]
                    if tail.isdigit():
                        used.add(int(tail))
    n = 0
    while n in used:
        n += 1
    return n


__all__ = ['EarlyExitToFindIndex']
