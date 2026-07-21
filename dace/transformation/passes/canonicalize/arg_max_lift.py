# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift TSVC-style ``if a[i] OP bestv: bestv = a[i]`` argmax/argmin loops to a
:class:`~dace.libraries.standard.nodes.reduce.Reduce` libnode.

Pattern recognised (target: TSVC ``s314`` / ``s316`` value-only argmax/argmin):

.. code-block:: python

    x = a[0]
    for i in range(start, end):
        if a[i] > x:           # or `<`, `>=`, `<=`
            x = a[i]

The Python frontend lowers this to a multi-block loop body shaped like::

    [start_state (empty)]
        | iedge: a_index_sym = a[i]
        v
    [cond_prep_state (empty)]
        | iedge: tmp_sym = (a_index_sym OP carrier)
        v
    [ConditionalBlock(condition = tmp_sym)]
        true-branch -> [write_state: a -> a_index_AN -> assign-tasklet -> carrier_AN]

The pass walks this chain backward from the ConditionalBlock, extracts ``a``,
the carrier (``x``), and the comparison operator, then replaces the loop with a
:class:`Reduce` libnode (``Max`` for ``>`` / ``>=``, ``Min`` for ``<`` / ``<=``)
over ``a[start:end]`` writing to the carrier.

Scope
-----

Two carrier storages are matched, with different capabilities:

* **Data carrier** -- ``data.Scalar`` or length-1 array (``shape == (1,)``). The
  in-loop write is an AccessNode chain in the true-branch's single state. Value
  only: the plain unit ``a[i]`` gather, no transform, and no iedge assignment
  anywhere in the true-branch (any such assignment is an extra write, e.g. a
  sibling ``index = i``). The base TSVC ``s314`` / ``s316`` shape.

* **Symbol carrier** -- the in-loop write is a true-branch iedge assignment
  ``carrier := [f](a[b + c*i])``. This path additionally supports:

  - an **index carrier** ``index := i`` tracking the argmax/argmin position
    (TSVC ``s315``), lifted alongside the value; it must itself be a symbol;
  - a **unary gather transform** ``f``, e.g. ``maxv = abs(a[i])`` (TSVC
    ``s3113``), which must match the one the comparison used;
  - an **affine gather** ``a[b + c*i]``. A strided / non-zero-base gather (TSVC
    ``s318``) is lowered ONLY on the combined transform+index path, which
    materialises ``buf[j] = f(a[b + c*(start-1+j)])`` and then arg-reduces;
    every other symbol-carrier shape assumes the unit ``a[i]`` gather.

  The true-branch states must be empty -- tasklet / AccessNode work there would
  be a side effect the rewrite cannot preserve.

* **2-D contiguous nests** (:class:`_Match2D`) -- ``for i: for j:`` over a
  contiguous ``aa[i, j]`` carrying a value symbol plus TWO index symbols
  (``xindex := i`` / ``yindex := j``), TSVC ``s3110`` / ``s13110``. Lifted to one
  flat arg-reduce; the flat index decomposes back to the two indices.

* **No else branch.** Only the canonical ``if cond: write; else: nothing``
  shape; an else branch with side effects would need separate handling.

Soundness
---------

The rewrite is value-preserving because the original loop's sequential
semantics is exactly the running reduction along ``i``: at each iteration the
carrier holds the running ``max``/``min`` over ``a[start:i+1]``. After the
loop, the carrier equals ``max``/``min`` over ``a[start:end]``, which is what
the :class:`Reduce` libnode computes.

The pre-loop init (``x = a[start]``) is preserved by routing it as a direct
copy *before* the libnode -- the libnode itself runs with ``identity=None``
so its first read seeds itself from ``a[start]``.

Tie-breaking
------------

The guard's strictness decides which of several equal extremes the tracked
index refers to:

* **strict** (``>`` / ``<``) -- the carrier is never updated on a tie, so the
  sequential loop keeps the FIRST occurrence. ``ArgReduce`` scans with the same
  strict comparison, so a plain forward arg-reduce matches it directly.
* **non-strict** (``>=`` / ``<=``) -- the carrier IS updated on a tie, so the
  sequential loop keeps the LAST occurrence. This is lifted by reversing the
  scanned order before the (still strict, first-wins) arg-reduce: the first
  occurrence of the extreme in the reversed sequence is exactly the last one in
  the forward sequence. The rewrite materialises the reversed gather with a
  parallel map and maps the returned position back (``i = end - idx``), so the
  lifted shape stays fully parallel and bit-exact with the sequential loop.

The extreme VALUE is identical under either guard, so the value-only reduction
(no index carrier) needs no reversal.

The choice is exposed as the :attr:`ArgMaxLift.tie_break` knob -- ``'infer'``
(the default: derive it from the guard's strictness exactly as above),
``'first'``, or ``'last'``. The explicit settings let a caller whose source
semantics are known state them directly instead of round-tripping through the
comparison operator; ``'infer'`` reproduces the inference verbatim.

Break / early-exit loops
------------------------

A loop whose body can ``break`` is NOT a reduction: it is a find-FIRST search
that stops at the first hit, so its carrier holds the value at the exit
iteration, not the extreme over the whole range. Neither tie rule can express
that -- a forward arg-reduce scans every element -- so any loop containing a
:class:`~dace.sdfg.state.BreakBlock` is refused outright (see
:meth:`ArgMaxLift._contains_break`), regardless of ``tie_break``. The break
shape's parallel lift is
:class:`~dace.transformation.passes.canonicalize.early_exit_to_find_index.EarlyExitToFindIndex`,
which runs earlier in the canonicalize pipeline and rewrites the loop to a
find-first Reduce(Min). A break loop still standing when ArgMaxLift runs is one
that pass already refused, and no Reduce/ArgReduce this pass emits could lift it
correctly either -- so the refusal costs no parallelism ArgMaxLift could have
delivered.
"""
import ast
import re
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np

import dace
from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState, ControlFlowRegion, ConditionalBlock, BreakBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.libraries.standard.nodes.reduce import Reduce

#: Map AST comparison op class -> DaCe reduction type.
_CMP_AST_TO_RTYPE = {
    ast.Gt: dtypes.ReductionType.Max,
    ast.GtE: dtypes.ReductionType.Max,
    ast.Lt: dtypes.ReductionType.Min,
    ast.LtE: dtypes.ReductionType.Min,
}


class _Match(NamedTuple):
    """A successfully matched argmax/argmin loop.

    :param op: ``Max`` or ``Min``.
    :param loop: The :class:`LoopRegion` to rewrite.
    :param parent: ``loop.parent_graph`` (cached).
    :param carrier_name: The carrier scalar's data name (``x`` in s314).
    :param carrier_kind: ``'scalar'`` or ``'length_one_array'``.
    :param carrier_subset: The carrier's single-point subset (``[0]``).
    :param input_array: The reduced-over array's data name (``a`` in s314).
    :param iter_start: Loop start expression.
    :param iter_end: Loop inclusive end expression.
    :param idx_carrier_name: The index carrier symbol (``index`` in s315), when
        the true-branch ALSO tracks the argmax/argmin position; ``None`` for the
        value-only shape. Only the symbol-carrier path supports it (the index is
        bound via an iedge, like the value carrier).
    :param gather_base: Constant term ``b`` of the affine gather index
        ``a[b + c*i]`` (``0`` for the plain ``a[i]`` gather).
    :param gather_coeff: Loop-variable coefficient ``c`` of the affine gather
        index (``1`` for the plain ``a[i]`` gather; a non-unit / symbolic ``c``
        is the strided gather of TSVC s318, where ``c = inc``).
    :param last_wins: True iff an index is tracked AND the resolved tie rule is
        last-occurrence -- under the default ``tie_break='infer'`` that is
        exactly "the guard is NON-STRICT (``>=`` / ``<=``)", i.e. the sequential
        loop keeps the LAST occurrence of the extreme. The rewrite then
        arg-reduces over the REVERSED gather (see the module docstring's
        tie-breaking note). ``False`` -> first-occurrence, the plain forward
        arg-reduce. Resolved by :meth:`ArgMaxLift._resolve_last_wins`.
    """
    op: dtypes.ReductionType
    loop: LoopRegion
    parent: ControlFlowRegion
    carrier_name: str
    carrier_kind: str
    carrier_subset: subsets.Range
    input_array: str
    iter_start: Any
    iter_end: Any
    idx_carrier_name: Optional[str] = None
    transform: Optional[str] = None  # unary gather transform ('abs') or None
    gather_base: Any = 0
    gather_coeff: Any = 1
    last_wins: bool = False


class _Match2D(NamedTuple):
    """A matched 2-D contiguous argmax/argmin over a nested ``for i: for j:`` loop.

    The TSVC ``s3110`` / ``s13110`` shape: a value carrier ``maxv`` plus two index
    carriers ``xindex := i`` (outer) / ``yindex := j`` (inner), all symbols,
    updated together inside ``if aa[i, j] OP maxv``. When the full ``aa[i, j]``
    access is a contiguous subset, the nested reduction is a single flat
    arg-reduce over ``aa`` viewed as 1-D; the flat index ``m`` decomposes back to
    ``xindex = m // ncols`` / ``yindex = m % ncols`` (``ncols`` = the contiguous /
    inner dimension size).

    :param op: ``Max`` or ``Min``.
    :param outer_loop: The outer (``i``) LoopRegion -- the rewrite removes it
        (and the inner loop it contains).
    :param inner_loop: The inner (``j``) LoopRegion.
    :param parent: ``outer_loop.parent_graph`` (cached).
    :param carrier_name: The value carrier symbol (``maxv``).
    :param x_idx_name: The outer index carrier symbol (``xindex := i``).
    :param y_idx_name: The inner index carrier symbol (``yindex := j``).
    :param input_array: The reduced-over 2-D array (``aa``).
    :param ncols: The contiguous (inner) dimension size used to decompose the
        flat index -- ``aa.shape[1]`` for a C-contiguous array.
    :param last_wins: True iff the resolved tie rule is last-occurrence -- under
        the default ``tie_break='infer'`` that is exactly "the guard is
        NON-STRICT (``>=`` / ``<=``)", i.e. the sequential nest keeps the LAST
        (row-major) occurrence of the extreme; the rewrite then arg-reduces over
        the reversed flat order. The 2-D nest always tracks both indices, so the
        knob always applies here.
    """
    op: dtypes.ReductionType
    outer_loop: LoopRegion
    inner_loop: LoopRegion
    parent: ControlFlowRegion
    carrier_name: str
    x_idx_name: str
    y_idx_name: str
    input_array: str
    ncols: Any
    last_wins: bool = False


@properties.make_properties
@xf.explicit_cf_compatible
class ArgMaxLift(ppl.Pass):
    """Lift TSVC-style argmax/argmin loops to :class:`Reduce` libnodes."""

    CATEGORY: str = 'Optimization Preparation'

    tie_break = properties.Property(
        dtype=str,
        default='infer',
        choices=['infer', 'first', 'last'],
        desc="Which of several equal extremes the tracked index refers to. 'infer' (default) derives it from the "
        "guard's strictness -- strict (> / <) keeps the FIRST occurrence, non-strict (>= / <=) the LAST -- "
        "reproducing the sequential loop. 'first' / 'last' pin the rule explicitly. Only meaningful when an "
        "index is tracked: the extreme VALUE is the same either way.")

    def __init__(self, tie_break: str = 'infer'):
        super().__init__()
        self.tie_break = tie_break

    def _resolve_last_wins(self, op_ast, has_index: bool) -> bool:
        """The tie rule for this match: True -> keep the LAST occurrence of the
        extreme (arg-reduce over the reversed gather), False -> the FIRST (the
        plain forward arg-reduce).

        Without an index carrier the rule is unobservable (both tie choices yield
        the same extreme value), so it stays False and the rewrite skips the
        reversal. Otherwise ``tie_break`` decides: ``'infer'`` reads it off the
        guard's strictness -- the sequential loop updates the carrier on a tie
        under ``>=`` / ``<=`` (keeping the last) but not under ``>`` / ``<``
        (keeping the first) -- while ``'first'`` / ``'last'`` pin it.
        """
        if not has_index:
            return False
        if self.tie_break == 'first':
            return False
        if self.tie_break == 'last':
            return True
        return op_ast in (ast.GtE, ast.LtE)

    def _contains_break(self, block) -> bool:
        """Whether ``block`` contains a :class:`BreakBlock` anywhere beneath it.

        A break makes the loop a find-FIRST search rather than a reduction (see
        the module docstring), so :meth:`_match` / :meth:`_match_2d` refuse it --
        NO tie rule can express an early exit, since an arg-reduce scans the whole
        range. Recursing through nested regions is deliberately conservative: the
        rewrites drop everything but the carrier chain, so a break anywhere in the
        matched body is a shape this pass must not touch.
        """
        if isinstance(block, BreakBlock):
            return True
        if isinstance(block, ConditionalBlock):
            return any(self._contains_break(br) for _c, br in block.branches)
        if isinstance(block, ControlFlowRegion):
            return any(self._contains_break(n) for n in block.nodes())
        return False

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

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
                # Stale-snapshot guard: a previous rewrite may have removed this
                # LoopRegion from its parent.
                if region.parent_graph is None or region not in region.parent_graph.nodes():
                    continue
                m = self._match(region, sd)
                if m is not None:
                    self._rewrite(m, sd)
                    rewritten += 1
                    continue
                # 2-D contiguous nested argmax (TSVC s3110 / s13110).
                m2 = self._match_2d(region, sd)
                if m2 is not None:
                    self._rewrite_2d(m2, sd)
                    rewritten += 1
        return rewritten or None

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

        # A break makes this a find-FIRST search, not a reduction: the carrier
        # holds the value at the EXIT iteration, while any arg-reduce this pass
        # emits scans the whole range (e.g. ``x = a[0]; for i: if a[i] > x: x =
        # a[i]; break`` lifts to ``max(a)`` -- a value miscompile, not merely a
        # tie mismatch). The data-carrier path's true-branch check counts only
        # non-empty SDFGStates, so it would otherwise let the BreakBlock through.
        # ``EarlyExitToFindIndex`` is the pass that parallelises this shape.
        if self._contains_break(loop):
            return None

        # Body must hold exactly one ConditionalBlock (with optional empty wrapper states).
        cond_block = None
        for b in loop.nodes():
            if isinstance(b, ConditionalBlock):
                if cond_block is not None:
                    return None
                cond_block = b
            elif isinstance(b, SDFGState):
                if len(b.nodes()) > 0:
                    return None  # any non-empty plain state in the body is unsupported in v1
            else:
                return None
        if cond_block is None:
            return None

        # The conditional must have exactly one (non-else) branch; no else / empty else.
        non_else = [(c, br) for c, br in cond_block.branches if c is not None]
        else_branches = [(c, br) for c, br in cond_block.branches if c is None]
        if len(non_else) != 1:
            return None
        cond_codeblock, true_branch = non_else[0]
        if any(self._branch_has_content(br) for _, br in else_branches):
            return None

        cond_expr_str = cond_codeblock.as_string.strip()
        # The comparison ``gather OP carrier`` reaches the ConditionalBlock in
        # one of two shapes, depending on how Simplify folded the body:
        #  (a) indirected -- the condition is a single symbol ``tmp`` bound by an
        #      upstream iedge ``tmp := (g OP c)``; or
        #  (b) inlined -- the comparison sits directly in the condition
        #      ``(g OP c)`` (current canonicalize output for TSVC s314/s316).
        inline_gather = None
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', cond_expr_str):
            op_ast, gather_sym_name, carrier_name, transform = self._resolve_tmp_iedge(loop, cond_block, cond_expr_str)
        else:
            op_ast, gather_sym_name, carrier_name, transform = self._parse_comparison(cond_expr_str)
            if op_ast is None:
                # The gather may be an INLINE array subscript in the condition
                # (``array[b + c*i] OP carrier``) rather than a bound name -- the
                # shape canonicalize leaves for the value+index argmax (TSVC s315).
                parsed = self._parse_inline_subscript_comparison(cond_expr_str, loop.loop_variable, loop)
                if parsed is not None:
                    op_ast, inline_gather, carrier_name, transform = parsed
                    gather_sym_name = None
        if op_ast is None:
            return None
        op = _CMP_AST_TO_RTYPE[op_ast]

        # Resolve the gather ``arr[b + c*i]``: either the inline subscript parsed
        # above, or (the tmp-symbol / bound-name case) an iedge one level back.
        if inline_gather is not None:
            input_array, gather_base, gather_coeff = inline_gather
            if input_array not in sdfg.arrays:
                return None
        else:
            gather = self._resolve_gather_iedge(loop, cond_block, gather_sym_name, loop.loop_variable, sdfg)
            if gather is None:
                return None
            input_array, gather_base, gather_coeff = gather
        # A non-unit / non-zero-base gather (``arr[inc*i]``, TSVC s318) is only
        # supported on the symbol-carrier transform+index path below; every
        # other shape assumes the plain unit ``arr[i]`` gather.
        is_unit_gather = bool(symbolic.simplify(gather_base) == 0 and symbolic.simplify(gather_coeff) == 1)

        # Classify the carrier's storage first; the body-write check differs
        # by case (scalar / length-1 array use state writes; symbol uses an
        # iedge inside the true-branch).
        carrier_kind, carrier_subset = self._classify_carrier(carrier_name, sdfg)
        if carrier_kind is None:
            return None

        idx_carrier_name = None
        if carrier_kind in ('scalar', 'length_one_array'):
            # Data-carrier path: the in-loop write lives on AccessNode chains;
            # the true-branch's iedges must NOT carry assignments (any iedge
            # assignment is an extra write -- e.g. TSVC s315's ``index = i``).
            # A gather transform (``abs``) is only handled on the symbol path.
            if transform is not None or not is_unit_gather:
                return None
            for e in true_branch.edges():
                if e.data.assignments:
                    return None
            true_state = self._extract_singleton_state(true_branch)
            if true_state is None:
                return None
            if not self._true_state_writes_carrier_from_array(true_state, carrier_name, input_array, loop.loop_variable,
                                                              sdfg):
                return None
        else:
            # Symbol-carrier path: the in-loop write is an iedge assignment
            # ``carrier := [f](arr[b+c*i])`` (or ``carrier := [f](gather_sym)``
            # where ``gather_sym`` was bound to ``arr[b+c*i]``) inside the
            # true-branch, OPTIONALLY plus an index carrier ``idx := loop_var``
            # (argmax position, s315). The transform ``f`` (e.g. ``abs``, s3113)
            # must match the one the comparison used. The true-branch states must
            # be empty -- any tasklet / AccessNode work would be a separate side
            # effect the rewrite cannot preserve.
            ok, idx_carrier_name = self._symbol_true_branch_writes_carrier(true_branch,
                                                                           carrier_name,
                                                                           input_array,
                                                                           gather_sym_name,
                                                                           loop.loop_variable,
                                                                           transform=transform)
            if not ok:
                return None
            # An index carrier must itself be a symbol (bound back via iedge).
            if idx_carrier_name is not None and idx_carrier_name not in sdfg.symbols:
                return None
            # A strided gather (s318) is ONLY lowered on the combined
            # transform+index path (``_rewrite_with_transform_and_index``), which
            # materialises ``buf[j] = f(a[b + c*(start-1+j)])`` then ArgReduces.
            # Every other symbol-carrier shape (value-only / index-only /
            # transform-only) assumes the unit ``arr[i]`` gather.
            has_transform_and_index = (transform is not None and idx_carrier_name is not None)
            if not is_unit_gather and not has_transform_and_index:
                return None
            # The strided combined path makes a load-bearing seed assumption
            # (``buf[0]`` stands in for the pre-loop seed at ``base+coeff*(start-1)``
            # and the index init is ``start-1``). Verify it so a mismatched seed
            # is refused rather than mis-lifted.
            if has_transform_and_index and not self._verify_affine_seed(loop, sdfg, carrier_name, idx_carrier_name,
                                                                        input_array, gather_base, gather_coeff, start,
                                                                        transform):
                return None

        # Tie-break semantics (``tie_break``; 'infer' reads it off the guard's
        # strictness): a non-strict guard (``>=`` / ``<=``) updates the carrier on
        # a TIE, so the sequential loop keeps the LAST occurrence's index, while a
        # strict guard keeps the FIRST. The ArgReduce scan is always strict
        # (first-wins), so the last-wins shape is lifted by arg-reducing over the
        # REVERSED gather (first-of-reversed IS last-of-forward); the rewrite maps
        # the position back. The extreme VALUE is identical under either rule, so
        # the value-only reduction (no index carrier) needs no reversal.
        last_wins = self._resolve_last_wins(op_ast, idx_carrier_name is not None)

        return _Match(
            op=op,
            loop=loop,
            parent=loop.parent_graph,
            carrier_name=carrier_name,
            carrier_kind=carrier_kind,
            carrier_subset=carrier_subset,
            input_array=input_array,
            iter_start=start,
            iter_end=end,
            idx_carrier_name=idx_carrier_name,
            transform=transform,
            gather_base=gather_base,
            gather_coeff=gather_coeff,
            last_wins=last_wins,
        )

    # ------------------------- match helpers -------------------------

    # ------------------------------------------------------------------
    # 2-D contiguous nested argmax (TSVC s3110 / s13110).
    # ------------------------------------------------------------------

    def _single_child_region(self, region, want_type):
        """Return the unique child block of ``region`` of type ``want_type``,
        requiring every other child to be an empty ``SDFGState``; else ``None``."""
        found = None
        for b in region.nodes():
            if isinstance(b, want_type):
                if found is not None:
                    return None
                found = b
            elif isinstance(b, SDFGState):
                if len(b.nodes()) > 0:
                    return None
            else:
                return None
        return found

    def _unit_loop_from_zero(self, loop: LoopRegion):
        """``(start, end)`` for a unit-stride loop starting at 0, else ``None``."""
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
        if symbolic.simplify(start) != 0:
            return None
        return start, end

    def _match_2d(self, outer_loop: LoopRegion, sdfg: SDFG) -> Optional[_Match2D]:
        """Match a nested ``for i: for j: if aa[i, j] OP maxv: maxv = aa[i, j];
        xindex = i; yindex = j`` over the full (contiguous) array. Matched on the
        OUTER loop; the inner loop and the ConditionalBlock are validated below.
        """
        if not outer_loop.loop_variable:
            return None
        # A break anywhere in the nest makes it an early-exit search, not a
        # reduction over the whole (contiguous) array -- refuse (see :meth:`_match`).
        if self._contains_break(outer_loop):
            return None
        o_range = self._unit_loop_from_zero(outer_loop)
        if o_range is None:
            return None
        inner_loop = self._single_child_region(outer_loop, LoopRegion)
        if inner_loop is None or not inner_loop.loop_variable:
            return None
        i_range = self._unit_loop_from_zero(inner_loop)
        if i_range is None:
            return None
        cond_block = self._single_child_region(inner_loop, ConditionalBlock)
        if cond_block is None:
            return None

        non_else = [(c, br) for c, br in cond_block.branches if c is not None]
        else_branches = [(c, br) for c, br in cond_block.branches if c is None]
        if len(non_else) != 1 or any(self._branch_has_content(br) for _, br in else_branches):
            return None
        cond_codeblock, true_branch = non_else[0]
        cond_expr_str = cond_codeblock.as_string.strip()
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', cond_expr_str):
            op_ast, gather_sym, carrier_name, transform = self._resolve_tmp_iedge(inner_loop, cond_block, cond_expr_str)
        else:
            op_ast, gather_sym, carrier_name, transform = self._parse_comparison(cond_expr_str)
        if op_ast is None or transform is not None:
            return None  # transforms (abs) are out of scope for the 2-D path
        # The 2-D nest always tracks two indices, so the tie rule is always
        # observable and ``tie_break`` always applies: under the default 'infer'
        # the guard's strictness picks the occurrence -- a non-strict guard
        # (``>=`` / ``<=``) keeps the LAST one in row-major order, which the
        # rewrite lifts by arg-reducing over the reversed flat order (the flat
        # ArgReduce itself is strict / first-wins).
        last_wins = self._resolve_last_wins(op_ast, True)
        op = _CMP_AST_TO_RTYPE[op_ast]

        outer_var, inner_var = outer_loop.loop_variable, inner_loop.loop_variable
        array = self._resolve_gather_2d(inner_loop, gather_sym, outer_var, inner_var, sdfg)
        if array is None:
            return None
        # Value carrier must be a symbol; the true-branch must write exactly the
        # value (from the same gather) plus the two index carriers (i / j).
        if carrier_name not in sdfg.symbols:
            return None
        idx = self._match_2d_true_branch(true_branch, carrier_name, gather_sym, array, outer_var, inner_var, sdfg)
        if idx is None:
            return None
        x_idx_name, y_idx_name = idx

        # Contiguity: the nested iteration must visit EXACTLY the whole array, in
        # C-contiguous memory order, so a flat 1-D arg-reduce is equivalent.
        desc = sdfg.arrays[array]
        if len(desc.shape) != 2 or not desc.is_packed_c_strides():
            return None
        if symbolic.simplify(o_range[1] - (desc.shape[0] - 1)) != 0:
            return None
        if symbolic.simplify(i_range[1] - (desc.shape[1] - 1)) != 0:
            return None
        full = subsets.Range([(0, desc.shape[0] - 1, 1), (0, desc.shape[1] - 1, 1)])
        if not full.is_contiguous_subset(desc):
            return None
        return _Match2D(op=op,
                        outer_loop=outer_loop,
                        inner_loop=inner_loop,
                        parent=outer_loop.parent_graph,
                        carrier_name=carrier_name,
                        x_idx_name=x_idx_name,
                        y_idx_name=y_idx_name,
                        input_array=array,
                        ncols=desc.shape[1],
                        last_wins=last_wins)

    def _parse_2d_gather(self, rhs_str: str, outer_var: str, inner_var: str) -> Optional[str]:
        """Return the array name iff ``rhs_str`` is exactly ``arr[outer_var,
        inner_var]`` (a 2-D point access, outer index in dim 0, inner in dim 1)."""
        try:
            tree = ast.parse(str(rhs_str), mode='eval').body
        except SyntaxError:
            return None
        if not isinstance(tree, ast.Subscript) or not isinstance(tree.value, ast.Name):
            return None
        idx = tree.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        if not isinstance(idx, ast.Tuple) or len(idx.elts) != 2:
            return None
        d0, d1 = idx.elts
        if not (isinstance(d0, ast.Name) and d0.id == outer_var):
            return None
        if not (isinstance(d1, ast.Name) and d1.id == inner_var):
            return None
        return tree.value.id

    def _resolve_gather_2d(self, inner_loop: LoopRegion, gather_sym: str, outer_var: str, inner_var: str,
                           sdfg: SDFG) -> Optional[str]:
        """Find an iedge binding ``gather_sym := arr[outer_var, inner_var]`` in the
        inner loop and return ``arr`` (validated against ``sdfg.arrays``)."""
        for e in inner_loop.all_interstate_edges():
            rhs = (e.data.assignments or {}).get(gather_sym)
            if rhs is None:
                continue
            arr = self._parse_2d_gather(rhs, outer_var, inner_var)
            if arr is not None and arr in sdfg.arrays:
                return arr
        return None

    def _match_2d_true_branch(self, true_branch, carrier: str, gather_sym: str, array: str, outer_var: str,
                              inner_var: str, sdfg: SDFG):
        """Verify the true-branch binds exactly ``carrier := arr[i, j]`` (or
        ``:= gather_sym``), ``x := outer_var`` and ``y := inner_var`` via iedges,
        with empty states and no other writes. Returns ``(x_name, y_name)`` or
        ``None``."""
        if not isinstance(true_branch, ControlFlowRegion):
            return None
        carrier_seen = False
        x_idx = y_idx = None
        for e in true_branch.edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                rhs_str = str(rhs).strip()
                if lhs == carrier:
                    if rhs_str != gather_sym and self._parse_2d_gather(rhs_str, outer_var, inner_var) != array:
                        return None
                    carrier_seen = True
                elif rhs_str == outer_var and x_idx is None:
                    x_idx = lhs
                elif rhs_str == inner_var and y_idx is None:
                    y_idx = lhs
                else:
                    return None
        for n in true_branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return None
            if not isinstance(n, SDFGState):
                return None  # nested control flow unsupported
        if not (carrier_seen and x_idx is not None and y_idx is not None):
            return None
        if x_idx not in sdfg.symbols or y_idx not in sdfg.symbols:
            return None
        return x_idx, y_idx

    def _rewrite_2d(self, m: _Match2D, sdfg: SDFG):
        """Replace the nested 2-D argmax with a flat :class:`ArgReduce` over the
        whole (contiguous) array, decomposing the flat index back into the two
        index carriers (``xindex = mflat // ncols``, ``yindex = mflat % ncols``).

        Under a non-strict guard (``m.last_wins``) the sequential nest keeps the
        LAST row-major occurrence while the ArgReduce scan keeps the first, so the
        arg-reduce runs over a REVERSED copy of the array (``rev[k] =
        aa_flat[total-1-k]``, materialised by a parallel map) and the returned
        position is mapped back with ``mflat = total - 1 - idx``.
        """
        from dace.codegen.targets.cpp import sym2cpp
        from dace.libraries.standard.nodes import ArgReduce
        desc = sdfg.arrays[m.input_array]
        val_buf, _ = sdfg.add_scalar(f'_argmax2d_val_{m.outer_loop.label}',
                                     desc.dtype,
                                     transient=True,
                                     find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argmax2d_idx_{m.outer_loop.label}',
                                     dtypes.int64,
                                     transient=True,
                                     find_new_name=True)

        nrows, ncols_sym = desc.shape[0], desc.shape[1]
        total = symbolic.simplify(nrows * ncols_sym)
        argmax_state = m.parent.add_state(m.outer_loop.label + '_argreduce2d')
        entry_state = argmax_state
        rev_buf = None
        if m.last_wins:
            # Reversed flat copy: rev[_i*ncols + _j] = aa[nrows-1-_i, ncols-1-_j],
            # whose forward flat position is exactly ``total-1 - (_i*ncols+_j)``.
            # The map is a pure gather (a bijection on the flat index), so the
            # materialisation stays fully parallel.
            rev_buf, _ = sdfg.add_array(f'_argmax2d_rev_{m.outer_loop.label}', [total],
                                        desc.dtype,
                                        transient=True,
                                        find_new_name=True)
            mat_state = m.parent.add_state(m.outer_loop.label + '_argreduce2d_rev')
            mat_state.add_mapped_tasklet(
                name='reverse_gather2d',
                map_ranges={
                    '_i': f'0:{sym2cpp(nrows)}',
                    '_j': f'0:{sym2cpp(ncols_sym)}'
                },
                inputs={
                    '__in':
                    mm.Memlet(data=m.input_array,
                              subset=f'({sym2cpp(symbolic.simplify(nrows - 1))}) - _i, '
                              f'({sym2cpp(symbolic.simplify(ncols_sym - 1))}) - _j')
                },
                code='__out = __in',
                outputs={'__out': mm.Memlet(data=rev_buf, subset=f'_i * ({sym2cpp(ncols_sym)}) + _j')},
                external_edges=True,
            )
            entry_state = mat_state

        # Re-route inbound edges, dropping pre-loop seeds of all three carriers.
        for ie in list(m.parent.in_edges(m.outer_loop)):
            new_assigns = dict(ie.data.assignments or {})
            for k in (m.carrier_name, m.x_idx_name, m.y_idx_name):
                new_assigns.pop(k, None)
            m.parent.add_edge(ie.src, entry_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        if m.last_wins:
            m.parent.add_edge(entry_state, argmax_state, dace.InterstateEdge())

        ncols = symbolic.symstr(m.ncols)
        # Forward flat position of the winner: the reduce's own index when the scan
        # ran forward, mirrored through the array when it ran over the reversed copy.
        flat = idx_buf if not m.last_wins else f'(({symbolic.symstr(symbolic.simplify(total - 1))}) - {idx_buf})'
        bind_state = m.parent.add_state(m.outer_loop.label + '_argreduce2d_bind')
        m.parent.add_edge(
            argmax_state, bind_state,
            dace.InterstateEdge(assignments={
                m.carrier_name: val_buf,
                m.x_idx_name: f'({flat} // ({ncols}))',
                m.y_idx_name: f'({flat} % ({ncols}))',
            }))
        for oe in list(m.parent.out_edges(m.outer_loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.outer_loop)

        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = 'max' if m.op == dtypes.ReductionType.Max else 'min'
        node = ArgReduce(name=f'{m.outer_loop.label}_argreduce2d', op=op)
        argmax_state.add_node(node)
        if m.last_wins:
            read = argmax_state.add_read(rev_buf)
            in_memlet = mm.Memlet(data=rev_buf, subset=subsets.Range([(0, symbolic.simplify(total - 1), 1)]))
        else:
            read = argmax_state.add_read(m.input_array)
            in_memlet = mm.Memlet(data=m.input_array,
                                  subset=subsets.Range([(0, desc.shape[0] - 1, 1), (0, desc.shape[1] - 1, 1)]))
        argmax_state.add_edge(read, None, node, '_in', in_memlet)
        argmax_state.add_edge(node, '_out_val', wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, '_out_idx', wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()

    def _branch_has_content(self, branch) -> bool:
        if not isinstance(branch, ControlFlowRegion):
            return False
        for n in branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return True
            if isinstance(n, (LoopRegion, ConditionalBlock)):
                return True
        return False

    def _parse_compare_node(self, tree):
        """Extract ``(op_cls, gather_name, carrier_name, transform)`` from a
        :class:`ast.Compare` node ``gather OP carrier`` (or ``f(gather) OP
        carrier``). The gather side may carry a recognised unary transform
        (``abs`` -> ``transform='abs'``); the carrier side must be a bare name.
        Returns ``(None, None, None, None)`` on any mismatch.
        """
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
            return None, None, None, None
        op_cls = type(tree.ops[0])
        if op_cls not in _CMP_AST_TO_RTYPE:
            return None, None, None, None
        transform, lhs_name = self._extract_transform(tree.left)
        rhs_name = self._extract_name(tree.comparators[0])
        if lhs_name is None or rhs_name is None:
            return None, None, None, None
        return op_cls, lhs_name, rhs_name, transform

    def _parse_inline_subscript_comparison(self, expr_str: str, loop_var: str, loop: LoopRegion):
        """Parse a comparison whose gather is an INLINE array subscript:
        ``[f](array[b + c*i]) OP carrier``.

        Unlike :meth:`_parse_compare_node` (which needs the gather bound to a bare
        name), this handles the shape canonicalize leaves for TSVC ``s315`` -- the
        array access sits directly in the condition. Returns
        ``(op_cls, (array, base, coeff), carrier_name, transform)`` or ``None``.
        """
        try:
            tree = ast.parse(expr_str, mode='eval').body
        except SyntaxError:
            return None
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
            return None
        op_cls = type(tree.ops[0])
        if op_cls not in _CMP_AST_TO_RTYPE:
            return None
        carrier = self._extract_name(tree.comparators[0])
        if carrier is None:
            return None
        left, transform = tree.left, None
        if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id in self._SUPPORTED_TRANSFORMS and len(left.args) == 1 and not left.keywords):
            transform, left = left.func.id, left.args[0]
        if not (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name)):
            return None
        array = left.value.id
        idx = left.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        if isinstance(idx, (ast.Tuple, ast.Slice, ast.List)):
            return None
        try:
            idx_str = ast.unparse(idx)
        except Exception:  # pragma: no cover -- defensive
            return None
        aff = self._affine_index_in_loop_var(idx_str, loop_var, loop)
        if aff is None:
            return None
        base, coeff = aff
        return op_cls, (array, base, coeff), carrier, transform

    def _resolve_tmp_iedge(self, loop: LoopRegion, cond_block: ConditionalBlock, tmp_sym: str):
        """Walk in-edges of ``cond_block`` looking for one whose assignment binds
        ``tmp_sym`` to a comparison ``[f](g) OP c``. Returns ``(ast_op_cls,
        g_name, c_name, transform)`` or ``(None, None, None, None)``."""
        for ie in loop.in_edges(cond_block):
            assigns = ie.data.assignments or {}
            rhs = assigns.get(tmp_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode='eval').body
            except SyntaxError:
                continue
            res = self._parse_compare_node(tree)
            if res[0] is not None:
                return res
        return None, None, None, None

    def _parse_comparison(self, expr_str: str):
        """Parse a comparison ``[f](g) OP c`` inlined directly in the condition.

        Mirrors :meth:`_resolve_tmp_iedge` but on the condition string itself
        (no ``tmp`` iedge indirection). Returns ``(ast_op_cls, g_name, c_name,
        transform)`` or ``(None, None, None, None)``.
        """
        try:
            tree = ast.parse(expr_str, mode='eval').body
        except SyntaxError:
            return None, None, None, None
        return self._parse_compare_node(tree)

    def _resolve_gather_iedge(self, loop: LoopRegion, cond_block: ConditionalBlock, gather_sym: str, loop_var: str,
                              sdfg: SDFG) -> Optional[Tuple[str, Any, Any]]:
        """Walk back two levels to find an iedge binding ``gather_sym = arr[idx]``
        where ``idx`` is an AFFINE function of the loop variable ``b + c*i``.

        Returns ``(arr, base, coeff)`` -- ``base`` (``b``) and ``coeff`` (``c``)
        are sympy expressions, both loop-invariant. The plain ``arr[i]`` gather
        yields ``(arr, 0, 1)``; the strided gather ``arr[inc*i]`` (TSVC s318,
        after ``InductionVariableSubstitution`` closes the secondary IV ``k``)
        yields ``(arr, 0, inc)``. Returns ``None`` if no such iedge exists.
        """
        # Collect every iedge in the loop and try to find one binding ``gather_sym``.
        for e in loop.all_interstate_edges():
            assigns = e.data.assignments or {}
            rhs = assigns.get(gather_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode='eval').body
            except SyntaxError:
                continue
            if not isinstance(tree, ast.Subscript):
                continue
            if not isinstance(tree.value, ast.Name):
                continue
            arr = tree.value.id
            if arr not in sdfg.arrays:
                continue
            idx = tree.slice
            if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                idx = idx.value
            # Only a single 1-D affine index is handled; a multi-dim subscript
            # ``a[i, j]`` (ast.Tuple / Slice) is refused (the 2-D argmax of TSVC
            # s3110 / s13110 is out of scope).
            if isinstance(idx, (ast.Tuple, ast.Slice, ast.List)):
                continue
            try:
                idx_str = ast.unparse(idx)
            except Exception:  # pragma: no cover -- unparse failure is defensive
                continue
            aff = self._affine_index_in_loop_var(idx_str, loop_var, loop)
            if aff is None:
                continue
            base, coeff = aff
            return arr, base, coeff
        return None

    def _affine_index_in_loop_var(self, idx_str: str, loop_var: str, loop: LoopRegion):
        """Decompose a gather index ``idx_str`` as ``base + coeff*loop_var``.

        Returns ``(base, coeff)`` (sympy exprs) iff ``idx_str`` is affine and
        linear in ``loop_var`` AND both ``base`` and ``coeff`` are loop-invariant
        (free of the loop variable and of any symbol reassigned on a body iedge).
        The loop variable MUST appear (a constant index is not a per-iteration
        gather). Returns ``None`` on any mismatch -- the conservative default,
        so a non-affine / loop-carried index is never mis-lifted.
        """
        try:
            idx = symbolic.pystr_to_symbolic(idx_str)
            lv = symbolic.pystr_to_symbolic(loop_var)
        except Exception:  # pragma: no cover -- defensive parse guard
            return None
        if lv not in idx.free_symbols:
            return None
        # Linear-in-``lv`` decomposition: idx == base + coeff*lv with neither
        # term referencing ``lv`` (so no ``lv**2`` etc.). Expand first so
        # ``.coeff`` sees the loop var inside products like ``inc*(i-1)`` (the
        # closed form ``InductionVariableSubstitution`` leaves -- otherwise
        # ``.coeff`` returns 0 for the unexpanded form and the affine match fails).
        idx = idx.expand()
        coeff = idx.coeff(lv, 1)
        base = idx.coeff(lv, 0)
        if symbolic.simplify(idx - (base + coeff * lv)) != 0:
            return None
        if lv in coeff.free_symbols or lv in base.free_symbols:
            return None
        # Every symbol feeding base / coeff must be loop-invariant: not assigned
        # on any body interstate edge (a varying stride/base breaks the closed form).
        body_assigned = set()
        for e in loop.all_interstate_edges():
            body_assigned.update((e.data.assignments or {}).keys())
        for s in set(coeff.free_symbols) | set(base.free_symbols):
            if str(s) in body_assigned:
                return None
        return base, coeff

    def _extract_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        return None

    #: Recognised unary gather transforms ``f(g)`` -> the Python builtin name.
    _SUPPORTED_TRANSFORMS = {'abs'}

    def _extract_transform(self, node) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(transform, name)`` for a possibly-transformed operand.

        A bare ``Name`` ``g`` -> ``(None, 'g')``; a recognised unary call
        ``abs(g)`` -> ``('abs', 'g')``; anything else -> ``(None, None)``.
        """
        if isinstance(node, ast.Name):
            return None, node.id
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in self._SUPPORTED_TRANSFORMS and len(node.args) == 1 and not node.keywords):
            inner = self._extract_name(node.args[0])
            if inner is not None:
                return node.func.id, inner
        return None, None

    def _extract_singleton_state(self, branch) -> Optional[SDFGState]:
        if not isinstance(branch, ControlFlowRegion):
            return None
        content_states = [n for n in branch.nodes() if isinstance(n, SDFGState) and len(n.nodes()) > 0]
        if len(content_states) != 1:
            return None
        return content_states[0]

    def _true_state_writes_carrier_from_array(self, state: SDFGState, carrier: str, array: str, loop_var: str,
                                              sdfg: SDFG) -> bool:
        """Check the true-branch state has the shape ``arr -> arr_index_AN ->
        assign_tasklet -> carrier_AN`` writing ``carrier = arr[loop_var]``."""
        # Single write AccessNode for the carrier.
        carrier_writes = [n for n in state.data_nodes() if n.data == carrier and state.in_degree(n) > 0]
        if len(carrier_writes) != 1:
            return False
        carrier_an = carrier_writes[0]
        # **Strict refusal**: the true-branch may only have ONE terminal
        # AccessNode (the carrier). A second terminal AN means the conditional
        # updates an extra independent value (TSVC s315 ALSO writes ``index =
        # i``), and the v1 rewrite would silently drop it. Intermediate
        # transients in the carrier chain have ``out_degree > 0`` and are not
        # counted here.
        terminal_outs = [n for n in state.data_nodes() if state.in_degree(n) > 0 and state.out_degree(n) == 0]
        if len(terminal_outs) != 1 or terminal_outs[0] is not carrier_an:
            return False
        # Walk back through the chain (AN <- [Tasklet?] <- AN <- ... <- AN(array)).
        # The Python frontend emits an identity ``__out = __inp`` assign tasklet;
        # ``TrivialTaskletElimination`` strips it. Both shapes are accepted.
        source_an = self._walk_back_to_source(state, carrier_an)
        if source_an is None or source_an.data != array:
            return False
        # Verify the memlet from the source array references ``[loop_var]``.
        # Walk forward one edge from source to find the gather memlet.
        out_edges = list(state.out_edges(source_an))
        if not out_edges:
            return False
        loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
        for oe in out_edges:
            if oe.data is None or oe.data.subset is None:
                continue
            if any(loop_var_sym in symbolic.pystr_to_symbolic(str(lo)).free_symbols
                   for lo, _, _ in oe.data.subset.ranges):
                return True
        return False

    def _walk_back_to_source(self, state: SDFGState, carrier_an: nodes.AccessNode) -> Optional[nodes.AccessNode]:
        """Walk back from ``carrier_an`` through a chain of transients and
        identity Tasklets to the source AccessNode (the array we're reading
        from). Returns the source AN, or ``None`` if the chain doesn't form
        a single-source linear path.
        """
        cur = carrier_an
        while True:
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            upstream = ins[0].src
            if isinstance(upstream, nodes.Tasklet):
                t_ins = list(state.in_edges(upstream))
                if len(t_ins) != 1:
                    return None
                upstream = t_ins[0].src
                if not isinstance(upstream, nodes.AccessNode):
                    return None
            elif not isinstance(upstream, nodes.AccessNode):
                return None
            # ``upstream`` is now an AccessNode. If it has no in-edges it's the source.
            if state.in_degree(upstream) == 0:
                return upstream
            cur = upstream

    def _classify_carrier(self, name: str, sdfg: SDFG) -> Tuple[Optional[str], Optional[subsets.Range]]:
        desc = sdfg.arrays.get(name)
        if desc is not None:
            if isinstance(desc, data.Scalar):
                return 'scalar', subsets.Range([(0, 0, 1)])
            if isinstance(desc, data.Array) and tuple(desc.shape) == (1, ):
                return 'length_one_array', subsets.Range([(0, 0, 1)])
            return None, None
        # Symbol carrier: present in ``sdfg.symbols`` but not in ``sdfg.arrays``.
        if name in sdfg.symbols:
            return 'symbol', None
        return None, None

    def _rhs_is_value_write(self, rhs_str: str, gather_sym: str, array: str, loop_var: str,
                            transform: Optional[str]) -> bool:
        """True iff ``rhs_str`` is the value-carrier write under ``transform``:
        ``[f](gather_sym)`` or ``[f](array[loop_var])``, where ``f`` is the
        recognised transform (``None`` -> no wrapping call allowed)."""
        try:
            tree = ast.parse(rhs_str, mode='eval').body
        except SyntaxError:
            return False
        if transform is not None:
            if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == transform
                    and len(tree.args) == 1 and not tree.keywords):
                return False
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return False  # an unexpected transform when none was matched
        if isinstance(tree, ast.Name):
            return tree.id == gather_sym
        if isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array:
            idx = tree.slice
            if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                idx = idx.value
            return isinstance(idx, ast.Name) and idx.id == loop_var
        return False

    def _symbol_true_branch_writes_carrier(self,
                                           true_branch,
                                           carrier: str,
                                           array: str,
                                           gather_sym: str,
                                           loop_var: str,
                                           transform: Optional[str] = None):
        """For the symbol-carrier case, verify the true-branch binds the value
        carrier (``carrier := [f](array[loop_var])`` or ``carrier :=
        [f](gather_sym)``, with the same gather transform ``f`` the comparison
        used) and, optionally, ONE index carrier (``idx := loop_var`` -- the
        argmax position, TSVC s315). The true-branch states must all be empty
        (no tasklet / AccessNode work). Any other iedge assignment is refused.

        :returns: ``(ok, idx_carrier)`` -- ``ok`` is True iff the value carrier
            write was found and every assignment was recognised; ``idx_carrier``
            is the index-carrier symbol name when an ``idx := loop_var`` write is
            present, else ``None``.
        """
        if not isinstance(true_branch, ControlFlowRegion):
            return False, None
        carrier_write_seen = False
        idx_carrier = None
        for e in true_branch.edges():
            assigns = e.data.assignments or {}
            for lhs, rhs in assigns.items():
                rhs_str = str(rhs).strip()
                if lhs == carrier:
                    if not self._rhs_is_value_write(rhs_str, gather_sym, array, loop_var, transform):
                        return False, None
                    carrier_write_seen = True
                elif rhs_str == loop_var and idx_carrier is None:
                    # Index carrier := the loop variable (argmax position).
                    idx_carrier = lhs
                else:
                    return False, None  # unrecognised extra write -> refuse
        # All true-branch states must be empty -- any contained tasklet /
        # AccessNode work is a separate side effect.
        for n in true_branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return False, None
            if not isinstance(n, SDFGState):
                return False, None  # nested control flow not supported in v1
        return carrier_write_seen, idx_carrier

    def _collect_preloop_assignments(self, loop: LoopRegion) -> dict:
        """Walk back the linear pre-loop chain (each block reached by a single
        in-edge) and accumulate interstate-edge assignments. The binding closest
        to the loop wins for each name (it is the value live at loop entry).

        The seed of a TSVC argmax is spread over this chain -- e.g.
        ``block: a_index := a[0]; index := 0`` then ``-> loop: maxv := abs(a_index)``.
        """
        preloop: dict = {}
        parent = loop.parent_graph
        cur = loop
        seen = set()
        while True:
            ins = parent.in_edges(cur)
            if len(ins) != 1:
                break
            e = ins[0]
            for lhs, rhs in (e.data.assignments or {}).items():
                if lhs not in preloop:  # closest-to-loop wins
                    preloop[lhs] = str(rhs)
            if e.src in seen or not isinstance(e.src, (SDFGState, ControlFlowRegion)):
                break
            seen.add(e.src)
            cur = e.src
        return preloop

    def _verify_affine_seed(self, loop: LoopRegion, sdfg: SDFG, value_carrier: str, idx_carrier: str, array: str,
                            base: Any, coeff: Any, start: Any, transform: Optional[str]) -> bool:
        """For the strided transform+index path, verify the pre-loop seed is
        consistent with the buffer the rewrite builds: the value carrier must be
        seeded ``value_carrier := [f](array[Q])`` with ``Q`` equal to the gather's
        seed-iteration position ``base + coeff*(start-1)``, and the index carrier
        ``idx_carrier := (start-1)``.

        This is the load-bearing assumption of
        :meth:`_rewrite_with_transform_and_index` -- the buffer's first element
        ``buf[0] = f(a[base + coeff*(start-1)])`` stands in for the seed, and the
        index bind ``idx_carrier := (start-1) + idx_buf`` yields the seed's init
        index when the seed wins. A loop whose seed sits elsewhere (or whose
        index init != start-1) would be mis-lifted, so refuse it.

        Handles the real frontend shape, where the seed is spread over a pre-loop
        chain with indirection: ``base`` / ``coeff`` carry the secondary-IV symbol
        ``k`` (bound pre-loop to ``inc``), and the value seed is ``maxv :=
        abs(a_index)`` with ``a_index := a[0]`` on an earlier edge. Both the
        position comparison and the gather lookup substitute the chain's bindings.
        """
        preloop = self._collect_preloop_assignments(loop)
        if value_carrier not in preloop or idx_carrier not in preloop:
            return False

        # Pure-symbol pre-loop bindings (skip carriers + array-read bindings like
        # ``a_index := a[0]``) -- these resolve the closed form's IV symbol ``k``.
        subs = {}
        for lhs, rhs in preloop.items():
            if lhs in (value_carrier, idx_carrier):
                continue
            try:
                expr = symbolic.pystr_to_symbolic(rhs)
            except Exception:  # pragma: no cover -- defensive
                continue
            if any(str(s) in sdfg.arrays for s in expr.free_symbols):
                continue  # an array-read binding, not a scalar symbol value
            subs[symbolic.pystr_to_symbolic(lhs)] = expr

        def _resolve(expr):
            try:
                return symbolic.simplify(symbolic.pystr_to_symbolic(str(expr)).subs(subs))
            except Exception:  # pragma: no cover -- defensive
                return None

        seed_pos = _resolve(base + coeff * (start - 1))
        seed_idx = _resolve(start - 1)
        if seed_pos is None or seed_idx is None:
            return False

        # Value seed: ``[f](array[Q])``, possibly indirected through a gather
        # symbol bound on an earlier pre-loop edge (``maxv := abs(a_index)``).
        q = self._seed_gather_position(preloop[value_carrier], array, transform, preloop)
        if q is None:
            return False
        q_resolved = _resolve(q)
        if q_resolved is None or symbolic.simplify(q_resolved - seed_pos) != 0:
            return False
        # Index init must equal start-1.
        idx_resolved = _resolve(preloop[idx_carrier])
        if idx_resolved is None or symbolic.simplify(idx_resolved - seed_idx) != 0:
            return False
        return True

    def _seed_gather_position(self, rhs_str: str, array: str, transform: Optional[str], preloop: dict):
        """If ``rhs_str`` is ``[f](array[P])`` -- or ``[f](g)`` where ``g`` is a
        pre-loop symbol bound to ``array[P]`` -- return the index string ``P``;
        else ``None``."""
        try:
            tree = ast.parse(rhs_str, mode='eval').body
        except SyntaxError:
            return None
        if transform is not None:
            if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == transform
                    and len(tree.args) == 1 and not tree.keywords):
                return None
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return None
        # Follow one level of gather-symbol indirection (``g`` bound to ``a[P]``).
        if isinstance(tree, ast.Name) and tree.id in preloop:
            try:
                tree = ast.parse(preloop[tree.id], mode='eval').body
            except SyntaxError:
                return None
        if not (isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array):
            return None
        idx = tree.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        try:
            return ast.unparse(idx)
        except Exception:  # pragma: no cover -- defensive
            return None

    # ------------------------- rewrite -------------------------

    def _rewrite(self, m: _Match, sdfg: SDFG):
        """Replace the loop with a :class:`Reduce` (value-only) or
        :class:`ArgReduce` (value + index) libnode."""
        if m.transform is not None and m.idx_carrier_name is not None:
            return self._rewrite_with_transform_and_index(m, sdfg)
        if m.transform is not None:
            return self._rewrite_with_transform(m, sdfg)
        if m.idx_carrier_name is not None:
            return self._rewrite_with_index(m, sdfg)
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)

        # Allocate the output container per carrier kind.
        if m.carrier_kind == 'symbol':
            # Fresh transient scalar -> Reduce output -> iedge bind to the symbol.
            output_dtype = sdfg.symbols[m.carrier_name]
            out_name, _ = sdfg.add_scalar(f'_arg_max_buf_{m.loop.label}',
                                          output_dtype,
                                          transient=True,
                                          find_new_name=True)
            output_subset = subsets.Range([(0, 0, 1)])
        else:
            out_name = m.carrier_name
            output_subset = m.carrier_subset

        # Reduce-state replaces the loop.
        reduce_state = m.parent.add_state(m.loop.label + '_argmax')
        # Re-route inbound edges. For symbol carriers, drop any pre-loop iedge
        # assignment that binds the carrier symbol -- the reduce subsumes it.
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            if m.carrier_kind == 'symbol' and m.carrier_name in new_assigns:
                del new_assigns[m.carrier_name]
            new_iedge = dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            m.parent.add_edge(ie.src, reduce_state, new_iedge)
            m.parent.remove_edge(ie)

        if m.carrier_kind == 'symbol':
            # Insert a bind state AFTER the reduce so the carrier symbol gets
            # re-materialised from the transient scalar before any downstream
            # state references it. Scalars are bare names (no subscript) in
            # iedge assignment RHS expressions.
            bind_state = m.parent.add_state(m.loop.label + '_argmax_bind')
            m.parent.add_edge(reduce_state, bind_state, dace.InterstateEdge(assignments={m.carrier_name: out_name}))
            for oe in list(m.parent.out_edges(m.loop)):
                m.parent.add_edge(bind_state, oe.dst, oe.data)
                m.parent.remove_edge(oe)
        else:
            for oe in list(m.parent.out_edges(m.loop)):
                m.parent.add_edge(reduce_state, oe.dst, oe.data)
                m.parent.remove_edge(oe)

        m.parent.remove_node(m.loop)

        # Inputs / outputs for the libnode.
        read = reduce_state.add_read(m.input_array)
        write = reduce_state.add_write(out_name)

        # For scalar / length-1 array carriers the pre-loop init ``x = a[0]``
        # is already in a prior state writing to the same AccessNode; the
        # libnode's ``identity=None`` semantics fold that pre-existing value
        # into the running reduction (WCR-Max). For symbol carriers we have
        # to include the seed position explicitly in the input slice because
        # the dropped pre-loop iedge no longer materialises the seed.
        wcr_str = 'lambda a, b: max(a, b)' if m.op == dtypes.ReductionType.Max else 'lambda a, b: min(a, b)'
        # Identity (accumulator seed) for the reduction.
        #  * symbol carrier -> the Reduce writes a FRESH transient with no
        #    pre-seeded value (and the input slice already covers the original
        #    seed ``a[start-1]``). ``identity=None`` makes the pure expansion
        #    default the accumulator to 0, which is correct only for a Max over
        #    non-negative data -- it silently corrupts a Min (``min(0, positives)
        #    == 0``) and a Max over all-negative data. Use the proper neutral
        #    element: the dtype's most-negative value for Max, most-positive for
        #    Min (finite extremes, so codegen stays a plain numeric literal).
        #  * scalar / length-1 carrier -> keep ``identity=None``: it WCR-folds
        #    into the pre-loop ``x = a[start]`` seed already in the carrier AN.
        if m.carrier_kind == 'symbol':
            _nt = sdfg.arrays[m.input_array].dtype.type
            if np.issubdtype(_nt, np.floating):
                _info = np.finfo(_nt)
            else:
                _info = np.iinfo(_nt)
            identity = (_info.min if m.op == dtypes.ReductionType.Max else _info.max).item()
        else:
            identity = None
        node = Reduce(name=f'{m.loop.label}_argmax_reduce', wcr=wcr_str, axes=[0], identity=identity)
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        reduce_state.add_node(node)
        if m.carrier_kind == 'symbol':
            # Extend the slice down to ``start - 1`` so a[start - 1] (the seed)
            # is included in the reduction. (TSVC s314 init reads ``a[0]`` for
            # ``start = 1``; same shape generalised.)
            slice_lo = symbolic.simplify(start - 1)
            if slice_lo < 0:
                slice_lo = symbolic.simplify(0)
        else:
            slice_lo = start
        input_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(slice_lo, end, 1)]))
        reduce_state.add_edge(read, None, node, '_in', input_memlet)
        output_memlet = mm.Memlet(data=out_name, subset=output_subset)
        reduce_state.add_edge(node, '_out', write, None, output_memlet)
        sdfg.reset_cfg_list()

    def _rewrite_with_index(self, m: _Match, sdfg: SDFG):
        """Replace an argmax/argmin-with-index loop (TSVC s315) with an
        :class:`~dace.libraries.standard.nodes.ArgReduce` libnode.

        The lift mirrors the symbol-carrier value-only path but uses the
        two-output ``ArgReduce`` (value + index). Both outputs are fresh
        transient SCALARS -- ``val_buf`` (the array's dtype) and ``idx_buf``
        (``int64``) -- bound back to the carrier symbols after the reduce:
        ``carrier := val_buf`` and ``idx_carrier := slice_lo + idx_buf`` (the
        ``ArgReduce`` index is slice-local; ``slice_lo`` recovers the
        original-array position). The pre-loop seed iedges binding either
        carrier are dropped -- the reduce subsumes them (the seed position is
        kept by extending the input slice down to ``start - 1``).

        Under a non-strict guard (``m.last_wins``) the sequential loop keeps the
        LAST occurrence of the extreme while the ArgReduce scan keeps the first.
        The scan is then run over a REVERSED copy of the slice (``rev[j] =
        a[end-j]``, materialised by a parallel map), whose first extreme IS the
        forward slice's last one; the position maps back as ``idx_carrier :=
        end - idx_buf``.
        """
        from dace.codegen.targets.cpp import sym2cpp
        from dace.libraries.standard.nodes import ArgReduce
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        val_buf, _ = sdfg.add_scalar(f'_argmax_val_{m.loop.label}', arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argmax_idx_{m.loop.label}', dtypes.int64, transient=True, find_new_name=True)

        # Include the seed ``a[start-1]`` in the slice (clamped at 0).
        slice_lo = symbolic.simplify(start - 1)
        try:
            if slice_lo < 0:
                slice_lo = symbolic.simplify(0)
        except TypeError:  # symbolic start; assume the seed sits at >= 0
            pass
        lo_is_zero = bool(symbolic.simplify(slice_lo) == 0)

        argmax_state = m.parent.add_state(m.loop.label + '_argreduce')
        entry_state = argmax_state
        rev_buf = None
        if m.last_wins:
            # Reversed copy of the scanned slice: rev[_j] = a[end - _j], _j in
            # 0:n. This path only ever sees the unit gather (a[i]), so the array
            # position and the iteration index coincide.
            n_elems = symbolic.simplify(end + 1 - slice_lo)
            rev_buf, _ = sdfg.add_array(f'_argmax_rev_{m.loop.label}', [n_elems],
                                        arr_dtype,
                                        transient=True,
                                        find_new_name=True)
            mat_state = m.parent.add_state(m.loop.label + '_argreduce_rev')
            mat_state.add_mapped_tasklet(
                name='reverse_gather',
                map_ranges={'_j': f'0:{sym2cpp(n_elems)}'},
                inputs={'__in': mm.Memlet(data=m.input_array, subset=f'({sym2cpp(end)}) - _j')},
                code='__out = __in',
                outputs={'__out': mm.Memlet(data=rev_buf, subset='_j')},
                external_edges=True,
            )
            entry_state = mat_state

        # Re-route inbound edges, dropping pre-loop binds of BOTH carriers.
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(m.idx_carrier_name, None)
            m.parent.add_edge(ie.src, entry_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        if m.last_wins:
            m.parent.add_edge(entry_state, argmax_state, dace.InterstateEdge())

        # Bind state: re-materialise both carrier symbols from the scalars (bare
        # names). The index recovers the original-array position from the
        # slice-local one: forward -> add the slice base; reversed -> mirror it
        # through the slice's top end.
        if m.last_wins:
            idx_rhs = f'(({symbolic.symstr(end)}) - {idx_buf})'
        else:
            idx_rhs = idx_buf if lo_is_zero else f'({symbolic.symstr(slice_lo)} + {idx_buf})'
        bind_state = m.parent.add_state(m.loop.label + '_argreduce_bind')
        m.parent.add_edge(argmax_state, bind_state,
                          dace.InterstateEdge(assignments={
                              m.carrier_name: val_buf,
                              m.idx_carrier_name: idx_rhs
                          }))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = 'max' if m.op == dtypes.ReductionType.Max else 'min'
        node = ArgReduce(name=f'{m.loop.label}_argreduce', op=op)
        argmax_state.add_node(node)
        if m.last_wins:
            read = argmax_state.add_read(rev_buf)
            in_memlet = mm.Memlet(data=rev_buf, subset=subsets.Range([(0, symbolic.simplify(end - slice_lo), 1)]))
        else:
            read = argmax_state.add_read(m.input_array)
            in_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(slice_lo, end, 1)]))
        argmax_state.add_edge(read, None, node, '_in', in_memlet)
        argmax_state.add_edge(node, '_out_val', wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, '_out_idx', wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()

    def _rewrite_with_transform(self, m: _Match, sdfg: SDFG):
        """Replace a transformed value-only reduction (TSVC s3113,
        ``maxv = max(|a[i]|)``) with: a map materialising ``buf[j] = f(a[lo+j])``
        into a fresh contiguous transient, then a :class:`Reduce` over ``buf``.

        Only the value-only symbol-carrier shape is handled here (the unit
        ``a[i]`` gather); transform + index (TSVC s318, a strided gather) is
        lowered by :meth:`_rewrite_with_transform_and_index`. The transform
        ``f`` is applied per element in the materialisation map, so the
        reduction itself stays a plain Max/Min fold over a unit-stride buffer.
        """
        from dace.codegen.targets.cpp import sym2cpp
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        # Include the seed ``a[start-1]`` (clamped at 0); buf spans the slice.
        slice_lo = symbolic.simplify(start - 1)
        try:
            if slice_lo < 0:
                slice_lo = symbolic.simplify(0)
        except TypeError:  # symbolic start; assume the seed sits at >= 0
            pass
        n_elems = symbolic.simplify(end + 1 - slice_lo)

        buf, _ = sdfg.add_array(f'_argf_buf_{m.loop.label}', [n_elems], arr_dtype, transient=True, find_new_name=True)

        # Materialisation state: buf[_j] = f(input_array[slice_lo + _j]).
        mat_state = m.parent.add_state(m.loop.label + '_argf')
        mat_state.add_mapped_tasklet(
            name='transform_gather',
            map_ranges={'_j': f'0:{sym2cpp(n_elems)}'},
            inputs={'__in': mm.Memlet(data=m.input_array, subset=f'{sym2cpp(slice_lo)} + _j')},
            code=f'__out = {m.transform}(__in)',
            outputs={'__out': mm.Memlet(data=buf, subset='_j')},
            external_edges=True,
        )

        # Reduce over the (contiguous) buffer.
        out_dtype = sdfg.symbols[m.carrier_name]
        out_name, _ = sdfg.add_scalar(f'_argf_val_{m.loop.label}', out_dtype, transient=True, find_new_name=True)
        reduce_state = m.parent.add_state(m.loop.label + '_argf_reduce')

        # Re-route inbound edges to the materialise state, dropping the pre-loop
        # carrier seed (the reduce over buf subsumes it).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            m.parent.add_edge(ie.src, mat_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        m.parent.add_edge(mat_state, reduce_state, dace.InterstateEdge())

        # Bind state: carrier := reduced value.
        bind_state = m.parent.add_state(m.loop.label + '_argf_bind')
        m.parent.add_edge(reduce_state, bind_state, dace.InterstateEdge(assignments={m.carrier_name: out_name}))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        read = reduce_state.add_read(buf)
        write = reduce_state.add_write(out_name)
        wcr_str = 'lambda a, b: max(a, b)' if m.op == dtypes.ReductionType.Max else 'lambda a, b: min(a, b)'
        _info = np.finfo(arr_dtype.type) if np.issubdtype(arr_dtype.type, np.floating) else np.iinfo(arr_dtype.type)
        identity = (_info.min if m.op == dtypes.ReductionType.Max else _info.max).item()
        node = Reduce(name=f'{m.loop.label}_argf_reduce', wcr=wcr_str, axes=[0], identity=identity)
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        reduce_state.add_node(node)
        reduce_state.add_edge(read, None, node, '_in',
                              mm.Memlet(data=buf, subset=subsets.Range([(0, symbolic.simplify(n_elems - 1), 1)])))
        reduce_state.add_edge(node, '_out', write, None, mm.Memlet(data=out_name, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()

    def _rewrite_with_transform_and_index(self, m: _Match, sdfg: SDFG):
        """Replace a transformed argmax/argmin-WITH-INDEX over a (possibly
        strided) gather -- TSVC s318, ``maxv = max(|a[k]|)`` with ``k = inc*i``
        and ``index`` tracking the iteration of the max -- with: a map
        materialising ``buf[j] = f(a[pos_lo + coeff*j])`` into a fresh contiguous
        transient, then an :class:`~dace.libraries.standard.nodes.ArgReduce`
        over ``buf`` yielding the value + the slice-local index.

        Combines the abs-transform materialisation (:meth:`_rewrite_with_transform`)
        with the index-tracking ArgReduce (:meth:`_rewrite_with_index`). The
        gather is affine ``a[gather_base + gather_coeff*i]``; element ``j`` of the
        contiguous buffer maps to iteration ``i = (start-1) + j`` (the seed sits
        at ``i = start-1``, where ``index`` holds its init value). Hence the
        materialised position for ``buf[j]`` is ``pos_lo + coeff*j`` with
        ``pos_lo = base + coeff*(start-1)``, and the recovered iteration index is
        ``index := (start-1) + idx_buf`` (``idx_buf`` is slice-local). Both
        carrier seeds are dropped from the inbound iedges -- the buffer's first
        element subsumes them.

        Under a non-strict guard (``m.last_wins``) the sequential loop keeps the
        LAST extreme while the ArgReduce scan keeps the first, so the buffer is
        materialised in REVERSE iteration order (``buf[j]`` <-> iteration
        ``i = end - j``, i.e. array position ``base + coeff*end - coeff*j``) and
        the index maps back as ``index := end - idx_buf``. Reversal costs nothing
        extra here -- it only flips the direction of a map that already runs.
        """
        from dace.codegen.targets.cpp import sym2cpp
        from dace.libraries.standard.nodes import ArgReduce
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype
        base, coeff = m.gather_base, m.gather_coeff

        # Iterations covered, INCLUDING the seed at ``i = start-1`` (where the
        # index carrier still holds its init value). ``j`` ranges 0..n-1 over
        # ``i = iter_lo + j``.
        iter_lo = symbolic.simplify(start - 1)
        try:
            if iter_lo < 0:
                iter_lo = symbolic.simplify(0)
        except TypeError:  # symbolic start; assume the seed sits at >= 0
            pass
        n_elems = symbolic.simplify(end - iter_lo + 1)
        # Array position of ``buf[j]``: pos(i) = base + coeff*i, i = iter_lo + j.
        pos_lo = symbolic.simplify(base + coeff * iter_lo)

        buf, _ = sdfg.add_array(f'_argfi_buf_{m.loop.label}', [n_elems], arr_dtype, transient=True, find_new_name=True)
        val_buf, _ = sdfg.add_scalar(f'_argfi_val_{m.loop.label}', arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argfi_idx_{m.loop.label}', dtypes.int64, transient=True, find_new_name=True)

        # Materialisation: buf[_j] = f(input_array[pos_lo + coeff*_j]), or the
        # reversed order (buf[_j] <-> iteration ``end - _j``, i.e. position
        # ``pos_hi - coeff*_j``) when the guard is non-strict / last-wins. Both
        # cover exactly the same set of positions, in opposite order.
        mat_state = m.parent.add_state(m.loop.label + '_argfi')
        if m.last_wins:
            # Iteration ``iter_lo + n_elems - 1`` == ``end``, so buf[0] holds the
            # LAST scanned iteration and buf[n-1] the seed.
            pos_hi = symbolic.simplify(base + coeff * end)
            in_subset = f'({sym2cpp(pos_hi)}) - ({sym2cpp(coeff)}) * _j'
        else:
            in_subset = f'({sym2cpp(pos_lo)}) + ({sym2cpp(coeff)}) * _j'
        mat_state.add_mapped_tasklet(
            name='transform_gather_idx',
            map_ranges={'_j': f'0:{sym2cpp(n_elems)}'},
            inputs={'__in': mm.Memlet(data=m.input_array, subset=in_subset)},
            code=f'__out = {m.transform}(__in)',
            outputs={'__out': mm.Memlet(data=buf, subset='_j')},
            external_edges=True,
        )

        argmax_state = m.parent.add_state(m.loop.label + '_argfi_reduce')
        # Re-route inbound edges to the materialise state, dropping the pre-loop
        # binds of BOTH carriers (the ArgReduce over buf subsumes them).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(m.idx_carrier_name, None)
            m.parent.add_edge(ie.src, mat_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        m.parent.add_edge(mat_state, argmax_state, dace.InterstateEdge())

        # Bind state: value carrier := val_buf; index carrier := iter_lo + idx_buf
        # (idx_buf is the slice-local position; iter_lo recovers the iteration).
        # The reversed buffer mirrors it instead: iteration ``end - idx_buf``.
        lo_is_zero = bool(symbolic.simplify(iter_lo) == 0)
        if m.last_wins:
            idx_rhs = f'(({symbolic.symstr(end)}) - {idx_buf})'
        else:
            idx_rhs = idx_buf if lo_is_zero else f'({symbolic.symstr(iter_lo)} + {idx_buf})'
        bind_state = m.parent.add_state(m.loop.label + '_argfi_bind')
        m.parent.add_edge(argmax_state, bind_state,
                          dace.InterstateEdge(assignments={
                              m.carrier_name: val_buf,
                              m.idx_carrier_name: idx_rhs
                          }))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        read = argmax_state.add_read(buf)
        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = 'max' if m.op == dtypes.ReductionType.Max else 'min'
        node = ArgReduce(name=f'{m.loop.label}_argfi_argreduce', op=op)
        argmax_state.add_node(node)
        argmax_state.add_edge(read, None, node, '_in',
                              mm.Memlet(data=buf, subset=subsets.Range([(0, symbolic.simplify(n_elems - 1), 1)])))
        argmax_state.add_edge(node, '_out_val', wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, '_out_idx', wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()


__all__ = ['ArgMaxLift']
