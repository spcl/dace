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
* Predicate reads an array at anything other than exactly ``name[loop_var]`` --
  an offset ``a[i-1]`` / ``a[i+1]``, a coefficient ``a[2*i]``, a gather
  ``a[idx[i]]``, or a multi-dim ``a[i, j]``. The phi Map can only reproduce the
  per-iteration point read ``name[loop_var]``; any other subscript would be
  silently collapsed to it (miscompile) or emit a dangling read (crash).
* A body half whose faithful re-emit would drop a live nested-region effect (a
  non-transient write, an interstate symbol binding, or a transient a kept state
  reads).
"""
import ast
import re
import copy as _copy
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import dace
from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.frontend.python import astutils
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

        # ``_emit_body_loop`` reproduces each body half by cloning its top-level
        # SDFGStates and the interstate edges among them; it DROPS every nested
        # region (a ``cond_prep`` predicate-gather CFR) together with the region's
        # interstate assignments and transients. That is sound only when the
        # dropped content is dead relative to what is kept -- otherwise the emit
        # would silently lose a body write / symbol binding. Refuse rather than
        # miscompile -- the loop stays sequential.
        if not (self._body_emittable(body_pre_blocks, loop, sdfg)
                and self._body_emittable(body_post_blocks, loop, sdfg)):
            return None

        # Resolve the condition's textual expression by walking iedges and
        # inlining body-local transient scalar definitions.
        cond_iedge_bindings, cond_expr_str = self._resolve_cond_expression(loop, cond_block,
                                                                           body_pre_blocks + body_post_blocks, sdfg)
        if cond_expr_str is None:
            return None

        # Refuse-lift guard (correctness): if the predicate still names an
        # unresolved body-local transient (a data container the frontend
        # produced for ``__tmp0 = a[i] > K`` that the resolver could not inline
        # down to array subscripts + symbols + constants), the phi Map would
        # emit a dangling read with no producer. Keep the loop sequential.
        if not self._cond_is_fully_resolved(cond_expr_str, sdfg):
            return None

        # Refuse-lift guard (correctness): the phi Map (``_wire_expr_reads``)
        # reproduces every predicate array read at exactly ``name[loop_var]`` --
        # the per-iteration point read. If the predicate reads an array at any
        # other subscript (an offset ``a[i-1]`` / ``a[i+1]``, a coefficient
        # ``a[2*i]``, a gather ``a[idx[i]]``, or a multi-dim ``a[i, j]``) the
        # wiring would silently collapse the two reads to one ``name[loop_var]``
        # (a miscompiled exit index), emit a dangling ``__r_a]`` (a SyntaxError),
        # or attach a 1-D memlet to an N-D array (a validate error). Refuse so the
        # loop stays a correct sequential LoopRegion.
        if not self._predicate_reads_are_point_at_loop_var(cond_expr_str, sdfg, loop.loop_variable):
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
        if not isinstance(block, (ControlFlowRegion, SDFGState)):
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

    def _resolve_cond_expression(self, loop: LoopRegion, cond_block: ConditionalBlock, body_blocks: List,
                                 sdfg: SDFG) -> Tuple[Dict[str, str], Optional[str]]:
        """Resolve the condition predicate. Returns ``(bindings, expr_str)``
        where ``bindings`` maps gather-symbol names to their RHS expressions,
        and ``expr_str`` is the cond expression with those bindings AND any
        body-local transient scalar definitions inlined (e.g. ``a_index >
        threshold`` -> ``a[i] > threshold``, or -- under ``simplify=False`` --
        a bare ``__tmp0`` -> ``a[i] > threshold`` by inlining the body tasklet
        ``__tmp0 = a_index > threshold`` and the gather copy ``a_index =
        a[i]``)."""
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

        # Body-local transient scalar definitions (the ``simplify=False`` path):
        # ``__tmp0 = a[i] > K`` is a body tasklet, not an iedge assignment, so
        # inline it (and any gather copy chain it reads) into the predicate.
        # Interstate bindings win where a name is defined both ways.
        substitutions: Dict[str, str] = dict(self._collect_scalar_definitions(body_blocks, loop.loop_variable, sdfg))
        substitutions.update(bindings)

        # Inline the substitutions into the cond expression to a fixed point
        # (each RHS may itself reference another definition).
        expr = break_cond
        for _ in range(20):
            new_expr = expr
            for sym, rhs in substitutions.items():
                new_expr = re.sub(rf'\b{re.escape(sym)}\b', f'({rhs})', new_expr)
            if new_expr == expr:
                break
            expr = new_expr
        return bindings, expr

    def _collect_scalar_definitions(self, body_blocks: List, loop_var: str, sdfg: SDFG) -> Dict[str, str]:
        """Map each uniquely-defined body-local transient scalar to the
        expression that produces it, by inlining single-assignment tasklets and
        access-to-access copies found in the straight-line body (recursively,
        through nested regions like the ``simplify=False`` ``cond_prep`` CFR).

        Only side-effect-free single-assignment producers whose inputs are
        point-at-loop-var array subscripts, other scalars, or constants are
        recorded. A scalar written by more than one producer (or one that can
        not be characterised) is treated as ambiguous and dropped, so the
        predicate stays unresolved and the refuse-lift guard fires."""
        defs: Dict[str, str] = {}
        ambiguous: Set[str] = set()
        for state in self._all_states_in_blocks(body_blocks):
            for dn in state.data_nodes():
                if state.in_degree(dn) == 0:
                    continue
                desc = sdfg.arrays.get(dn.data)
                if not isinstance(desc, data.Scalar) or not desc.transient:
                    continue
                rhs = self._scalar_definition_expr(state, dn, loop_var, sdfg)
                if rhs is None:
                    ambiguous.add(dn.data)
                elif dn.data in defs and defs[dn.data] != rhs:
                    ambiguous.add(dn.data)
                else:
                    defs[dn.data] = rhs
        for name in ambiguous:
            defs.pop(name, None)
        return defs

    def _scalar_definition_expr(self, state: SDFGState, dn: nodes.AccessNode, loop_var: str,
                                sdfg: SDFG) -> Optional[str]:
        """RHS expression that writes transient scalar ``dn`` in ``state`` via a
        single producer edge (a copy or a single-assignment tasklet), or
        ``None`` if the writer is not a simple inlinable producer."""
        if state.in_degree(dn) != 1:
            return None
        ie = next(iter(state.in_edges(dn)))
        src = ie.src
        if isinstance(src, nodes.AccessNode):
            return self._render_access_read(src, ie.data, loop_var, sdfg)
        if isinstance(src, nodes.Tasklet):
            return self._render_tasklet_expr(state, src, ie.src_conn, loop_var, sdfg)
        return None

    def _render_access_read(self, access: nodes.AccessNode, memlet: mm.Memlet, loop_var: str,
                            sdfg: SDFG) -> Optional[str]:
        """Render the value read from ``access``. A scalar reads as its name
        (resolved recursively); a NON-transient array reads as the
        ``name[loop_var]`` subscript the phi Map can reproduce -- any other
        subset, or a transient (loop-body-local) array the parent phi Map cannot
        see, is refused (``None``)."""
        name = access.data
        desc = sdfg.arrays.get(name)
        if desc is None:
            return None
        if isinstance(desc, data.Scalar):
            return name
        if isinstance(desc, data.Array):
            if desc.transient or memlet.data != name:
                return None
            sub = memlet.subset
            if not isinstance(sub, subsets.Range) or len(sub) != 1:
                return None
            begin, end, _step = sub.ranges[0]
            if begin != end or symbolic.symstr(begin) != loop_var:
                return None
            return f'{name}[{loop_var}]'
        return None

    def _render_tasklet_expr(self, state: SDFGState, tasklet: nodes.Tasklet, out_conn: str, loop_var: str,
                             sdfg: SDFG) -> Optional[str]:
        """Render a single-assignment Python tasklet ``out_conn = <expr>`` as an
        expression string with every input connector replaced by its source
        read. Refuses anything that is not exactly one pure assignment to
        ``out_conn`` whose inputs are all inlinable access reads."""
        code = tasklet.code
        if code.language != dtypes.Language.Python:
            return None
        try:
            tree = ast.parse(code.as_string)
        except SyntaxError:
            return None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return None
        assign = tree.body[0]
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
            return None
        if assign.targets[0].id != out_conn:
            return None
        conn_exprs: Dict[str, str] = {}
        for ie in state.in_edges(tasklet):
            if ie.dst_conn is None:
                continue
            if not isinstance(ie.src, nodes.AccessNode):
                return None
            rendered = self._render_access_read(ie.src, ie.data, loop_var, sdfg)
            if rendered is None:
                return None
            conn_exprs[ie.dst_conn] = rendered
        rhs = astutils.unparse(assign.value)
        for conn, rendered in conn_exprs.items():
            rhs = re.sub(rf'\b{re.escape(conn)}\b', f'({rendered})', rhs)
        return f'({rhs})'

    def _cond_is_fully_resolved(self, cond_expr_str: str, sdfg: SDFG) -> bool:
        """Whether every data reference in the resolved predicate is wireable as
        an explicit phi-Map in-connector (see :meth:`_wire_expr_reads`): an array
        under a subscript, or a bare loop-invariant (non-transient) scalar. A
        bare transient (an unresolved ``__tmp0``) or a bare array has no wireable
        per-iteration producer, so the lift would dangle -- refuse instead."""
        try:
            tree = ast.parse(cond_expr_str, mode='eval').body
        except SyntaxError:
            return False
        subscript_bases = {
            id(n.value)
            for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or id(node) in subscript_bases:
                continue
            desc = sdfg.arrays.get(node.id)
            if desc is None:
                continue  # symbol / constant -- fine
            if isinstance(desc, data.Scalar) and not desc.transient:
                continue  # loop-invariant scalar param -- wired via a [0] connector
            return False  # bare transient, or bare array -- not wireable
        return True

    def _predicate_reads_are_point_at_loop_var(self, cond_expr_str: str, sdfg: SDFG, loop_var: str) -> bool:
        """Whether every array the resolved predicate reads is subscripted by
        exactly ``name[loop_var]`` -- a single dimension whose index is the loop
        variable. This is the ONLY read shape the phi Map can reproduce (see
        :meth:`_wire_expr_reads`, which wires each array at the per-iteration
        point ``[loop_var]``).

        Returns ``False`` (refuse the lift) for any other subscript, because the
        wiring would then be wrong or crash:

        * an offset ``a[i-1]`` / ``a[i+1]`` or coefficient ``a[2*i]`` -- collapsed
          to a single ``a[loop_var]`` read (a silent miscompile of the exit index);
        * a gather / nested subscript ``a[idx[i]]`` -- the read rewrite emits a
          dangling ``__r_a]`` (a SyntaxError at ``add_mapped_tasklet``);
        * a multi-dim ``a[i, j]`` -- a 1-D point memlet on an N-D array (a
          dimensionality / validate error).

        The index test mirrors :meth:`_render_access_read`'s point check
        (``symbolic.symstr(begin) != loop_var``), applied to the raw predicate
        subscripts."""
        try:
            tree = ast.parse(cond_expr_str, mode='eval').body
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not (isinstance(node.value, ast.Name) and node.value.id in sdfg.arrays):
                continue
            index = node.slice
            # Multi-dim ``a[i, j]`` -- a Tuple index the phi Map cannot point-read.
            if isinstance(index, ast.Tuple):
                return False
            # Gather / nested subscript ``a[idx[i]]`` -- a Subscript within the index.
            if any(isinstance(inner, ast.Subscript) for inner in ast.walk(index)):
                return False
            # Any offset / coefficient: the index must resolve to exactly the loop
            # variable (reusing ``_render_access_read``'s ``symstr == loop_var`` test).
            try:
                index_sym = symbolic.pystr_to_symbolic(astutils.unparse(index))
            except Exception:
                return False
            if symbolic.symstr(index_sym) != loop_var:
                return False
        return True

    def _all_states_in_blocks(self, blocks: List) -> List[SDFGState]:
        """Every SDFGState reachable within ``blocks`` (recursively)."""
        out: List[SDFGState] = []
        for b in blocks:
            out.extend(self._all_states(b))
        return out

    def _all_states(self, region) -> List[SDFGState]:
        """Every SDFGState within ``region``, recursing through nested control
        flow regions and conditional-block branches."""
        if isinstance(region, SDFGState):
            return [region]
        out: List[SDFGState] = []
        if isinstance(region, ConditionalBlock):
            for _cond, branch in region.branches:
                out.extend(self._all_states(branch))
        elif isinstance(region, ControlFlowRegion):
            for n in region.nodes():
                out.extend(self._all_states(n))
        return out

    def _region_interstate_edges(self, region) -> List:
        """Every interstate edge nested within ``region`` (recursively through
        sub-regions and conditional-block branches). Empty for an SDFGState."""
        out: List = []
        if isinstance(region, SDFGState):
            return out
        if isinstance(region, ConditionalBlock):
            for _cond, branch in region.branches:
                out.extend(self._region_interstate_edges(branch))
            return out
        if isinstance(region, ControlFlowRegion):
            out.extend(region.edges())
            for n in region.nodes():
                out.extend(self._region_interstate_edges(n))
        return out

    def _body_emittable(self, body_blocks: List, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Whether :meth:`_emit_body_loop` can faithfully reproduce this body
        half. It clones every top-level body SDFGState and the interstate edges
        *between two such top-level states*, so multi-state bodies are fine; but
        it DROPS every nested region (a ``cond_prep`` predicate-gather CFR), the
        interstate assignments carried inside (or on the loop-level edges within
        this half that touch) that region, and any transient the region produces.

        Refuse (so the loop stays a correct sequential LoopRegion) if a dropped
        block has a live effect the clone would silently lose:

        * a dropped state writes a live (non-transient) array;
        * a dropped interstate edge binds a symbol (``assignments``) -- the clone
          reproduces only edges between two kept top-level states;
        * a dropped nested region writes a transient a kept top-level state reads
          (the clone keeps the consumer but drops the producer -> a dangling read).
        """
        top_level = {id(b) for b in body_blocks if isinstance(b, SDFGState)}
        block_ids = {id(b) for b in body_blocks}
        dropped_blocks = [b for b in body_blocks if id(b) not in top_level]

        # A dropped state must not write a live (non-transient) array.
        for state in self._all_states_in_blocks(dropped_blocks):
            if self._collect_array_writes([state], sdfg):
                return False

        # A dropped interstate edge must not carry a symbol assignment: edges
        # nested inside a dropped region, plus loop-level edges *within this half*
        # that touch a nested region, are not replicated by the clone. Edges
        # leaving the half (into the cond block) carry the predicate gather and
        # are reproduced by the phi Map, so they are deliberately ignored here.
        dropped_edges = []
        for b in dropped_blocks:
            dropped_edges.extend(self._region_interstate_edges(b))
        for b in body_blocks:
            for e in loop.out_edges(b):
                if id(e.dst) not in block_ids:
                    continue  # edge leaves this body half -- not body work
                if id(e.src) in top_level and id(e.dst) in top_level:
                    continue  # replicated faithfully by the clone
                dropped_edges.append(e)
        for e in dropped_edges:
            if e.data is not None and e.data.assignments:
                return False

        # A dropped nested region must not produce a transient a kept top-level
        # state consumes.
        dropped_transients: Set[str] = set()
        for state in self._all_states_in_blocks(dropped_blocks):
            for n in state.data_nodes():
                if state.in_degree(n) == 0:
                    continue
                desc = sdfg.arrays.get(n.data)
                if desc is not None and desc.transient:
                    dropped_transients.add(n.data)
        if dropped_transients:
            kept_reads: Set[str] = set()
            for b in body_blocks:
                if not isinstance(b, SDFGState):
                    continue
                for n in b.data_nodes():
                    if b.out_degree(n) > 0:
                        kept_reads.add(n.data)
            if dropped_transients & kept_reads:
                return False
        return True

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
            desc = sdfg.arrays.get(name)
            if desc is None:
                continue
            if desc.transient:
                cond_reads |= self._trace_transient_to_source_arrays(name, body_pre_blocks, sdfg)
            elif isinstance(desc, data.Scalar):
                # A bare loop-invariant scalar the cond reads (wired as a phi-Map
                # ``[0]`` connector): count it as a read so a body write to the
                # same scalar trips the disjointness gate below.
                cond_reads.add(name)
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

        body_pre = self._emit_body_loop(parent,
                                        sdfg,
                                        m.body_pre_blocks,
                                        m,
                                        upper_str=f'Min({exit_sym} + 1, {symbolic.symstr(m.iter_end + 1)})')
        if body_pre is not None:
            parent.add_edge(last_state, body_pre, dace.InterstateEdge(assignments={exit_sym: exit_buf_name}))
            sym_bound = True
            last_state = body_pre

        body_post = self._emit_body_loop(parent, sdfg, m.body_post_blocks, m, upper_str=f'{exit_sym}')
        if body_post is not None:
            if sym_bound:
                parent.add_edge(last_state, body_post, dace.InterstateEdge())
            else:
                parent.add_edge(last_state, body_post, dace.InterstateEdge(assignments={exit_sym: exit_buf_name}))
                sym_bound = True
            last_state = body_post

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
        """Map: for i in [start, end+1): phi[i - start] = ITE(cond(i), i, N).

        Every data container the predicate reads is wired as an explicit
        in-connector with a memlet edge -- arrays through their ``[i]`` subscript
        and loop-invariant scalars (e.g. ``threshold``) through a ``[0]`` memlet.
        No sdfg.arrays name is left bare in the tasklet body; the
        :meth:`_assert_explicit_dataflow` invariant enforces this (a bare,
        unconnected ``__tmp0`` / ``threshold`` was the original miscompile)."""
        N_expr = symbolic.symstr(m.iter_end + 1)
        # The cond_expr uses ``i`` (the loop var); we keep that name in the Map.
        loop_var = m.loop.loop_variable
        inputs_map, new_expr = self._wire_expr_reads(m.cond_expr_str, sdfg, loop_var)

        # phi[i] = i if cond else N -- the GLOBAL index (i) or the
        # sentinel ``N``. Post-Reduce min then gives exit_i directly in
        # the original index space. Use the canonical 3-input ``ITE``
        # form (not a Python ternary): ``LowerITEToFpFactor`` lowers it
        # to ``c * t + (1 - c) * e`` at canonicalize time, so codegen
        # never emits ``c ? t : e``.
        tasklet_code = f'__out = ITE(({new_expr}), {loop_var}, ({N_expr}))'
        self._assert_explicit_dataflow(tasklet_code, set(inputs_map), sdfg)

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

    def _wire_expr_reads(self, expr_str: str, sdfg: SDFG, loop_var: str) -> Tuple[Dict[str, mm.Memlet], str]:
        """Rewrite ``expr_str`` so every data container it reads is referenced by
        an ``__r_<name>`` in-connector, returning ``(inputs, rewritten_expr)``.
        Arrays are read at ``[loop_var]`` (their per-iteration subscript);
        scalars are read at ``[0]``. Symbols and constants are left untouched."""
        # Fail loud, never silent: the matcher's
        # ``_predicate_reads_are_point_at_loop_var`` guard proves every array read
        # is exactly ``name[loop_var]`` before emit. If a non-point read reached
        # here the per-array rewrite below would collapse an offset/gather to a
        # single ``name[loop_var]`` read -- exactly the miscompile the guard
        # forbids -- so refuse loudly rather than emit wrong dataflow.
        if not self._predicate_reads_are_point_at_loop_var(expr_str, sdfg, loop_var):
            raise RuntimeError('EarlyExitToFindIndex: predicate reads an array at a subset other than '
                               f'``[{loop_var}]`` -- refuse-lift guard bypassed: {expr_str!r}')
        inputs: Dict[str, mm.Memlet] = {}
        new_expr = expr_str
        point = subsets.Range([(symbolic.pystr_to_symbolic(loop_var), symbolic.pystr_to_symbolic(loop_var), 1)])
        for arr_name in sorted(self._read_arrays_from_expr(expr_str, sdfg)):
            in_conn = f'__r_{arr_name}'
            inputs[in_conn] = mm.Memlet(data=arr_name, subset=_copy.deepcopy(point))
            new_expr = re.sub(rf'\b{re.escape(arr_name)}\b\[[^\]]*\]', in_conn, new_expr)
        for name in sorted(self._expr_names(new_expr)):
            desc = sdfg.arrays.get(name)
            if not isinstance(desc, data.Scalar):
                continue
            in_conn = f'__r_{name}'
            if in_conn in inputs:
                continue
            inputs[in_conn] = mm.Memlet(data=name, subset=subsets.Range([(0, 0, 1)]))
            new_expr = re.sub(rf'\b{re.escape(name)}\b', in_conn, new_expr)
        return inputs, new_expr

    def _assert_explicit_dataflow(self, code_str: str, in_connectors: Set[str], sdfg: SDFG):
        """HARD INVARIANT: a tasklet may only read a data container through an
        in-connector. Raise if any bare name in ``code_str`` names an
        sdfg.arrays container that is not one of ``in_connectors`` -- exactly the
        original dangling-read miscompile."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return
        subscript_bases = {
            id(n.value)
            for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
        }
        offenders = sorted({
            node.id
            for node in ast.walk(tree) if isinstance(node, ast.Name) and id(node) not in subscript_bases
            and node.id not in in_connectors and node.id in sdfg.arrays
        })
        if offenders:
            raise RuntimeError(f'EarlyExitToFindIndex: tasklet reads data container(s) {offenders} by bare name '
                               f'without an in-connector -- explicit-dataflow invariant violated: {code_str!r}')

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

    def _emit_body_loop(self, parent: ControlFlowRegion, sdfg: SDFG, body_blocks: List, m: _Match,
                        upper_str: str) -> Optional[LoopRegion]:
        """Emit the body work as a clipped ``for loop_var in [start, upper)``
        LoopRegion reusing the body's per-iteration dataflow; return it (or
        ``None`` if the body half has no live content state to wrap).

        We deliberately emit a LoopRegion, NOT a hand-built Map: the canonicalize
        pipeline lowers every Map to a LoopRegion immediately after this pass, and a
        hand-built Map with whole-array boundary memlets does not survive the
        round-trip -- ``LoopToMap``'s per-iteration write-uniqueness rejects an
        ``a[0:N]`` boundary and the find-first body stays sequential. A loop body
        state instead carries the raw ``a[i]`` accesses -- exactly the shape the
        ``parallelize`` stage's ``LoopToMap`` lifts -- so the body parallelizes on
        its own with no special boundary handling.

        Multi-state bodies (the ``simplify=False`` frontend shape, where ``a[i] =
        a[i] + b[i] * c[i]`` spans several slice/binop states) are reproduced in
        full: every top-level body SDFGState -- and the interstate edges among
        them -- is cloned into the body loop, then the loop's states are fused so
        the downstream ``LoopToMap`` sees the single-state per-iteration shape.
        Nested regions (a ``cond_prep`` predicate-gather CFR) carry only dead
        transient writes after the lift and are dropped (the matcher's
        :meth:`_body_emittable` guard already proved they write no live array)."""
        loop_var = m.loop.loop_variable
        start_str = symbolic.symstr(m.iter_start)
        # All top-level body states, in topological (partition) order. Empty
        # wrapper states are kept so their interstate assignments/topology survive.
        state_blocks = [b for b in body_blocks if isinstance(b, SDFGState)]
        if not any(len(b.nodes()) > 0 for b in state_blocks):
            return None  # no live body work to emit

        body_loop = LoopRegion(label=f'{m.loop.label}_body',
                               condition_expr=f'{loop_var} < ({upper_str})',
                               loop_var=loop_var,
                               initialize_expr=f'{loop_var} = {start_str}',
                               update_expr=f'{loop_var} = {loop_var} + 1')
        # Uniquify: two early-exit loops in the same parent CFG (or a pre-existing
        # ``*_body`` block) would otherwise collide on this derived label and trip the
        # "multiple blocks with the same name" validator. All wiring below is by object
        # reference, so a rename is safe.
        parent.add_node(body_loop, ensure_unique_name=True)

        # Clone each body state (nodes + intra-state edges) and record the mapping
        # so the interstate edges among them can be replicated with the same topology.
        state_map: Dict[SDFGState, SDFGState] = {}
        for i, src_state in enumerate(state_blocks):
            cloned = body_loop.add_state(src_state.label, is_start_block=(i == 0))
            node_map: Dict[Any, Any] = {}
            for n in src_state.nodes():
                cn = _copy.deepcopy(n)
                cloned.add_node(cn)
                node_map[n] = cn
            for e in src_state.edges():
                if e.data is None or e.data.is_empty():
                    continue
                cloned.add_edge(node_map[e.src], e.src_conn, node_map[e.dst], e.dst_conn, _copy.deepcopy(e.data))
            state_map[src_state] = cloned
        # Replicate the interstate edges that connect two cloned body states.
        for src_state in state_blocks:
            for e in m.loop.out_edges(src_state):
                if e.dst in state_map:
                    body_loop.add_edge(state_map[src_state], state_map[e.dst], _copy.deepcopy(e.data))

        # Fuse the cloned states so the per-iteration dataflow is a single state
        # (the shape ``LoopToMap`` lifts); a no-op when there was only one.
        self._fuse_region_states(body_loop, sdfg)
        return body_loop

    def _fuse_region_states(self, region: ControlFlowRegion, sdfg: SDFG):
        """Repeatedly fuse adjacent SDFGStates within ``region`` (only) via
        :class:`StateFusionExtended`, so a multi-state body collapses to the
        single-state per-iteration shape the parallelizer expects. Scoped to
        ``region`` -- the phi/reduce states in the parent CFG are untouched."""
        from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
        changed = True
        while changed:
            changed = False
            for e in list(region.edges()):
                if not (isinstance(e.src, SDFGState) and isinstance(e.dst, SDFGState)):
                    continue
                xform = StateFusionExtended()
                xform.first_state = e.src
                xform.second_state = e.dst
                if not xform.can_be_applied(region, expr_index=0, sdfg=sdfg, permissive=False):
                    continue
                xform.apply(region, sdfg)
                changed = True
                break

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
        # Uniquify: a second rebind in the same parent CFG would collide on this derived
        # label. The branch/edge wiring below is by object reference, so a rename is safe.
        parent.add_node(cond_block, ensure_unique_name=True)
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
