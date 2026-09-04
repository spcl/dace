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
  only: a unit-stride gather ``a[b + i]``, no transform, and no iedge assignment
  anywhere in the true-branch (any such assignment is an extra write, e.g. a
  sibling ``index = i``). The base TSVC ``s314`` / ``s316`` shape.

* **Symbol carrier** -- the in-loop write is a true-branch iedge assignment
  ``carrier := [f](a[b + c*i])``. This path additionally supports:

  - an **index carrier** ``index := i`` tracking the argmax/argmin position
    (TSVC ``s315``), lifted alongside the value; it must itself be a symbol;
  - a **unary gather transform** ``f``, e.g. ``maxv = abs(a[i])`` (TSVC
    ``s3113``), which must match the one the comparison used;
  - an **affine gather** ``a[b + c*i]``. A STRIDED gather (``c != 1``, TSVC
    ``s318``) is lowered ONLY on the combined transform+index path, whose
    ``ArgReduce`` reads the strided, transformed gather in one pass; every
    other symbol-carrier shape needs the unit stride. A SHIFTED gather
    (``b != 0``, e.g. ``a[i + 1]`` over a 0-based loop -- what rebasing the
    loop origin leaves behind) reduces over the same elements as its unshifted
    form, so the plain value-only lift folds ``b`` into the emitted slice; the
    index-only / transform-only rewrites still equate position with iteration
    and keep refusing it.

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

Predicate-index loops (no value carrier)
----------------------------------------

TSVC ``s331`` tracks a position with NO value carrier at all -- the guard is a
predicate against a loop-invariant threshold, and the only in-loop write is the
index itself::

    j = -1
    for i in range(N):
        if a[i] < 0.0:
            j = i

Because iterations run in increasing order the last writer wins, so the loop
computes ``j = max{i : a[i] < 0.0}``, falling back to the pre-loop seed when the
set is empty. That is a plain MAX over the masked iteration index::

    j = max(seed, max over i of (i if a[i] < 0.0 else seed))

Both halves of the equivalence need ``seed <= start``: with an empty predicate
set every iteration contributes ``seed`` so the fold returns it, and with a
non-empty set the largest matching ``i`` (which is ``>= start``) must beat the
seed. :meth:`ArgMaxLift.match_predicate_index` proves that inequality and
refuses when it is undecidable -- a seed above the iteration range would make
the fold return the seed where the sequential loop returned a real position.

The lift emits ``init(seed) -> Map(masked index, WCR max onto a private scalar)
-> bind``, i.e. exactly the WCR-on-scalar shape
:class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce` produces in
``wcr-scalar`` mode, which codegen lowers to ``#pragma omp parallel for``. No
:class:`Reduce` libnode is used: the reduced quantity is the ITERATION INDEX
masked by a predicate, not a slice of any array, so no array-slice fold can
express it.

The guard may name loop-invariant symbols, scalar containers and affine gathers
``arr[b + c*i]``, each wired as a tasklet input; it may NOT name the tracked
index (``if a[i] > 0 and j < 0`` is a find-FIRST search, not a max) nor read
through an indirection (``a[b[i]]``).

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
chunked parallel find-first search. A break loop still standing when ArgMaxLift runs is one
that pass already refused, and no Reduce/ArgReduce this pass emits could lift it
correctly either -- so the refusal costs no parallelism ArgMaxLift could have
delivered.
"""
import ast
import copy
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

import dace
from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.frontend.python import astutils
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState, ControlFlowRegion, ConditionalBlock, BreakBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.induction_variable_substitution import staged_iedge_rhs
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


class MatchPredIndex(NamedTuple):
    """A matched "last index at which a predicate holds" loop (TSVC ``s331``).

    Distinct from :class:`_Match`: there is no value carrier and no comparison
    against one. The reduced quantity is the ITERATION INDEX, masked by a
    predicate over loop-invariant data, so the lift is a WCR-max map rather than
    a :class:`Reduce` / ``ArgReduce`` over an array slice.

    :param loop: The :class:`LoopRegion` to replace.
    :param parent: ``loop.parent_graph`` (cached).
    :param idx_carrier: The symbol tracking the position (``j``).
    :param seed: ``idx_carrier``'s pre-loop value, proven ``<= iter_start``. It
        is both the fold's identity and the mask's false-value, which is what
        makes the empty-predicate case return it unchanged.
    :param guard_code: The branch guard with every array read replaced by a
        tasklet input connector name.
    :param guard_inputs: ``(connector, memlet)`` per wired read, in wiring order.
    :param iter_start: Loop start expression.
    :param iter_end: Loop INCLUSIVE end expression.
    """
    loop: LoopRegion
    parent: ControlFlowRegion
    idx_carrier: str
    seed: Any
    guard_code: str
    guard_inputs: List[Tuple[str, mm.Memlet]]
    iter_start: Any
    iter_end: Any


class GuardReadWiring(ast.NodeTransformer):
    """Replace every data read in a predicate guard by a tasklet input connector.

    Two read shapes are wired: a 1-D subscript at a position that is either affine
    in the loop variable (``arr[b + c*i]``, one element per iteration) or wholly
    loop-invariant (``thr[0]``, broadcast into the map), and a bare
    :class:`~dace.data.Scalar` / length-1 array name. Identical reads share one
    connector. Anything else that touches a data container -- a multi-dimensional
    subscript, an indirection ``a[b[i]]``, a bare name that is a full array -- sets
    :attr:`refused`, because no single memlet expresses it.
    """

    def __init__(self, sdfg: SDFG, loop: LoopRegion):
        self.sdfg = sdfg
        self.loop = loop
        self.reads: Dict[str, Tuple[str, mm.Memlet]] = {}
        self.refused = False

    def connector(self, key: str, array: str, subset: subsets.Range) -> ast.Name:
        entry = self.reads.get(key)
        if entry is None:
            entry = (f'__guard{len(self.reads)}', mm.Memlet(data=array, subset=subset))
            self.reads[key] = entry
        return ast.Name(id=entry[0], ctx=ast.Load())

    def visit_Subscript(self, node: ast.Subscript):
        if not (isinstance(node.value, ast.Name) and node.value.id in self.sdfg.arrays):
            self.refused = True
            return node
        array = node.value.id
        # One index expression only makes a 1-D memlet; a rank mismatch would emit a
        # subset of the wrong dimensionality.
        if len(self.sdfg.arrays[array].shape) != 1:
            self.refused = True
            return node
        idx = node.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        if isinstance(idx, (ast.Tuple, ast.Slice, ast.List)):
            self.refused = True
            return node
        try:
            idx_str = ast.unparse(idx)
        except Exception:  # pragma: no cover -- defensive
            self.refused = True
            return node
        pos = self.position(idx_str)
        if pos is None:
            self.refused = True
            return node
        return ast.copy_location(self.connector(f'{array}[{pos}]', array, subsets.Range([(pos, pos, 1)])), node)

    def position(self, idx_str: str) -> Optional[Any]:
        """The single array position a guard read touches -- affine in the loop
        variable, or wholly loop-invariant. ``None`` when it is neither, which is
        exactly the indirection / loop-carried-index case no memlet can express.
        """
        try:
            idx = symbolic.pystr_to_symbolic(idx_str)
        except Exception:  # pragma: no cover -- defensive
            return None
        loop_var = symbolic.pystr_to_symbolic(self.loop.loop_variable)
        if loop_var in idx.free_symbols:
            # Reuses the affine gather parser, so an indirection ``a[b[i]]`` is
            # refused exactly where the value paths refuse it.
            aff = ArgMaxLift._affine_index_in_loop_var(idx_str, self.loop.loop_variable, self.loop)
            if aff is None:
                return None
            base, coeff = aff
            return symbolic.simplify(base + coeff * loop_var)
        assigned: Dict[str, None] = {}
        for e in self.loop.all_interstate_edges():
            assigned.update(dict.fromkeys((e.data.assignments or {}).keys()))
        if any(str(s) in assigned for s in idx.free_symbols):
            return None  # a per-iteration symbol in the index is not loop-invariant
        return symbolic.simplify(idx)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.sdfg.arrays:
            return node
        desc = self.sdfg.arrays[node.id]
        if not (isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1, ))):
            self.refused = True
            return node
        return ast.copy_location(self.connector(node.id, node.id, subsets.Range([(0, 0, 1)])), node)


#: Right-hand operand values that make a binary op the identity on its left operand.
_IDENTITY_RHS = ((ast.Add, 0), (ast.Sub, 0), (ast.Mult, 1), (ast.Div, 1))
#: ... and the mirrored form, for the commutative ops only.
_IDENTITY_LHS = ((ast.Add, 0), (ast.Mult, 1))


def strip_identity(expr: ast.AST) -> ast.AST:
    """``expr`` with outer identity arithmetic (``+ 0``, ``- 0``, ``* 1``, ``/ 1``) peeled off.

    A scalar assignment does not always reach us spelled as one. ``x = y`` on a rank-0 value makes
    ``x`` a second NAME for ``y``'s container rather than a copy, so a producer that needs a copy
    writes ``x = y + 0`` on purpose -- which is what the numpy-to-DaCe emitter does for every
    scalar alias it lowers. Matching the surface syntax then refuses the very write the identity
    was added to express: TSVC ``s318`` binds ``maxv := abs(a[k]) + 0.0``, the arg-reduce match
    asks for a bare ``abs(...)`` call, and a textbook argmax stays a sequential loop.

    Peeled, not evaluated: only an operand that is literally the neutral constant is removed, so
    nothing here changes a value or depends on floating-point reasoning.
    """
    while isinstance(expr, ast.BinOp):
        right, left = expr.right, expr.left
        if (isinstance(right, ast.Constant) and isinstance(right.value,
                                                           (int, float)) and not isinstance(right.value, bool)
                and any(isinstance(expr.op, op) and right.value == v for op, v in _IDENTITY_RHS)):
            expr = left
            continue
        if (isinstance(left, ast.Constant) and isinstance(left.value,
                                                          (int, float)) and not isinstance(left.value, bool)
                and any(isinstance(expr.op, op) and left.value == v for op, v in _IDENTITY_LHS)):
            expr = right
            continue
        return expr
    return expr


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
        return {}

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
                    continue
                # Predicate-index, no value carrier (TSVC s331).
                mp = self.match_predicate_index(region, sd)
                if mp is not None:
                    self.rewrite_predicate_index(mp, sd)
                    rewritten += 1
        return rewritten or None

    def guarded_loop_skeleton(self, loop: LoopRegion):
        """The gate every payload analysis in this pass shares.

        Both the value-carrier argmax (:meth:`_match`, TSVC s314 / s315 / s318) and
        the predicate index (:meth:`match_predicate_index`, TSVC s331) require the
        SAME loop skeleton -- a unit-stride, break-free loop whose body is exactly
        one :class:`ConditionalBlock` (plus empty wrapper states) with a single
        non-else branch and no else content. Only what the branch WRITES tells the
        two apart, so the skeleton is decided once here and each analysis then reads
        the guard and the branch its own way.

        A break makes the loop a find-FIRST search, not a reduction: the carrier
        holds the value at the EXIT iteration, while any arg-reduce this pass emits
        scans the whole range (``x = a[0]; for i: if a[i] > x: x = a[i]; break``
        would lift to ``max(a)`` -- a value miscompile, not merely a tie mismatch).
        ``EarlyExitToFindIndex`` is the pass that parallelises that shape. The
        data-carrier path's true-branch check counts only non-empty SDFGStates, so
        it would otherwise let a BreakBlock through.

        A non-empty plain state in the body is refused: it is per-iteration work the
        rewrites drop. TSVC ``s318`` lands here -- its secondary induction variable
        ``k += inc`` survives as a dataflow tasklet because ``inc`` is a scalar
        CONTAINER, so ``InductionVariableSubstitution`` cannot close ``k`` into
        ``a[inc*i]``. The argmax analysis below would otherwise handle s318 (the
        closed-form gather lifts today); the gap is IV closure, not arg-reduction.

        :returns: ``(iter_start, iter_end, cond_block, guard_codeblock, true_branch)``
            or ``None``.
        """
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
        if self._contains_break(loop):
            return None

        cond_block = None
        for b in loop.nodes():
            if isinstance(b, ConditionalBlock):
                if cond_block is not None:
                    return None
                cond_block = b
            elif isinstance(b, SDFGState):
                if len(b.nodes()) > 0:
                    return None
            else:
                return None
        if cond_block is None:
            return None

        non_else = [(c, br) for c, br in cond_block.branches if c is not None]
        if len(non_else) != 1:
            return None
        if any(self._branch_has_content(br) for c, br in cond_block.branches if c is None):
            return None
        guard, true_branch = non_else[0]
        return start, end, cond_block, guard, true_branch

    def _match(self, loop: LoopRegion, sdfg: SDFG) -> Optional[_Match]:
        skeleton = self.guarded_loop_skeleton(loop)
        if skeleton is None:
            return None
        start, end, cond_block, cond_codeblock, true_branch = skeleton

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
        # The two halves of the affine gather constrain the rewrites separately:
        #  * a non-unit COEFF (``arr[inc*i]``, TSVC s318) gathers a strided set,
        #    which only the transform+index path handles (its ArgReduce takes the
        #    stride and the transform as operand properties);
        #  * a non-zero BASE merely SHIFTS the gathered slice. It is folded into
        #    the emitted range by :meth:`_gather_range`, so the plain value-only
        #    lift handles it -- notably ``a[i + 1]`` over a 0-based loop, which
        #    is what rebasing ``a[i]`` over ``1:N`` to a 0-based origin leaves.
        unit_coeff = bool(symbolic.simplify(gather_coeff) == 1)
        zero_base = bool(symbolic.simplify(gather_base) == 0)

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
            if transform is not None or not unit_coeff:
                return None
            for e in true_branch.edges():
                if e.data.assignments:
                    return None
            true_state = self._extract_singleton_state(true_branch)
            if true_state is None:
                return None
            if not self._true_state_writes_carrier_from_array(true_state, loop, carrier_name, input_array, gather_base,
                                                              gather_coeff):
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
                                                                           loop,
                                                                           carrier_name,
                                                                           input_array,
                                                                           gather_sym_name,
                                                                           gather_base,
                                                                           gather_coeff,
                                                                           transform=transform)
            if not ok:
                return None
            # An index carrier must itself be a symbol (bound back via iedge).
            if idx_carrier_name is not None and idx_carrier_name not in sdfg.symbols:
                return None
            # A strided gather (s318) is ONLY lowered on the combined
            # transform+index path (``_rewrite_with_transform_and_index``), whose
            # ArgReduce reads the strided slice directly. Every other
            # symbol-carrier shape (value-only / index-only / transform-only)
            # assumes the unit stride ``arr[b + i]``.
            has_transform_and_index = (transform is not None and idx_carrier_name is not None)
            if not unit_coeff and not has_transform_and_index:
                return None
            # A SHIFTED gather is folded into the emitted slice by the plain
            # value-only rewrite and by the transform+index buffer. The
            # index-only / transform-only rewrites still equate the gathered
            # position with the iteration -- the tracked index is recovered
            # straight from the slice-local ArgReduce position, and the
            # transform buffer indexes off the iteration -- so a non-zero base
            # stays refused there.
            value_only = transform is None and idx_carrier_name is None
            folds_base = has_transform_and_index or value_only
            if not zero_base and not folds_base:
                return None
            # Both base-folding rewrites DROP the pre-loop bind and reconstruct the
            # seed positionally at ``base + coeff*(start-1)`` -- the combined path
            # by starting its slice there (plus an index init of ``start-1``), the
            # value-only path by extending the emitted slice down to it. Neither is implied by
            # the match, so verify it; a seed that reads elsewhere would reduce over
            # a set missing the real seed and holding an element never gathered.
            if folds_base and not self._verify_affine_seed(loop, sdfg, carrier_name, idx_carrier_name, input_array,
                                                           gather_base, gather_coeff, start, transform):
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
        # The bounds are reparsed from loop CodeBlocks while the shape carries the declared
        # assumptions, so the same name arrives as two sympy instances that never cancel.
        o_end, i_end, dim0, dim1 = symbolic.equalize_symbols_across(o_range[1], i_range[1], desc.shape[0],
                                                                    desc.shape[1])
        if symbolic.simplify(o_end - (dim0 - 1)) != 0:
            return None
        if symbolic.simplify(i_end - (dim1 - 1)) != 0:
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
            # Symbolic bounds and subsets, never a rendered string: ``sym2cpp`` spells a symbolic
            # extent as C++ (``dace::math::ipow(R, K)``), and the range parser splits on ':', so
            # the qualified name comes back as bogus tokens.
            _i, _j = symbolic.symbol('_i'), symbolic.symbol('_j')
            rev_i = symbolic.simplify(nrows - 1) - _i
            rev_j = symbolic.simplify(ncols_sym - 1) - _j
            flat = _i * ncols_sym + _j
            mat_state.add_mapped_tasklet(
                name='reverse_gather2d',
                map_ranges={
                    '_i': (0, nrows - 1, 1),
                    '_j': (0, ncols_sym - 1, 1)
                },
                inputs={
                    '__in': mm.Memlet(data=m.input_array, subset=subsets.Range([(rev_i, rev_i, 1), (rev_j, rev_j, 1)]))
                },
                code='__out = __in',
                outputs={'__out': mm.Memlet(data=rev_buf, subset=subsets.Range([(flat, flat, 1)]))},
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
            dace.InterstateEdge(
                assignments={
                    m.carrier_name: val_buf,
                    m.x_idx_name: f'int_floor({flat}, {ncols})',
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

    @staticmethod
    def _affine_index_in_loop_var(idx_str: str, loop_var: str, loop: LoopRegion):
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
        # membership and .coeff both go through identity; equalize before either
        idx, lv = symbolic.equalize_symbols_across(idx, lv)
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
        body_assigned: dict = {}
        for e in loop.all_interstate_edges():
            body_assigned.update(dict.fromkeys((e.data.assignments or {}).keys()))
        # membership-only scan (early-exit), iteration order does not affect the result
        for s in dict.fromkeys([*coeff.free_symbols, *base.free_symbols]):
            if str(s) in body_assigned:
                return None
        return base, coeff

    def _extract_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        return None

    #: Recognised unary gather transforms ``f(g)`` -> the Python builtin name.
    #: Adding one here is not enough for the transform+index shape: that rewrite hands the name to
    #: ``ArgReduce.transform``, whose own set is what decides how it is spelled in C++.
    _SUPPORTED_TRANSFORMS = dict.fromkeys(['abs'])

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

    def _true_state_writes_carrier_from_array(self, state: SDFGState, loop: LoopRegion, carrier: str, array: str,
                                              gather_base: Any, gather_coeff: Any) -> bool:
        """Check the true-branch state has the shape ``arr -> arr_index_AN ->
        assign_tasklet -> carrier_AN`` writing ``carrier = arr[gather_base +
        gather_coeff*loop_var]`` -- the SAME element the comparison gathered."""
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
        # Verify the memlet from the source array gathers the single element the
        # comparison gathered. Accepting any subset that merely MENTIONS the loop
        # variable would let ``if a[i] > x: x = a[i + 1]`` through -- it stores a
        # different element than it compared, so it is no reduction at all.
        out_edges = list(state.out_edges(source_an))
        if not out_edges:
            return False
        loop_var = loop.loop_variable
        for oe in out_edges:
            if oe.data is None or oe.data.subset is None:
                continue
            ranges = oe.data.subset.ranges
            if len(ranges) != 1:
                continue
            lo, hi, step = ranges[0]
            if symbolic.simplify(hi - lo) != 0 or symbolic.simplify(step) != 1:
                continue  # not a single point
            aff = self._affine_index_in_loop_var(str(lo), loop_var, loop)
            if aff is None:
                continue
            base, coeff = aff
            if symbolic.simplify(base - gather_base) == 0 and symbolic.simplify(coeff - gather_coeff) == 0:
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

    def _rhs_is_value_write(self, rhs_str: str, gather_sym: str, array: str, loop_var: str, loop: LoopRegion,
                            gather_base: Any, gather_coeff: Any, transform: Optional[str]) -> bool:
        """True iff ``rhs_str`` is the value-carrier write under ``transform``:
        ``[f](gather_sym)`` or ``[f](array[idx])``, where ``f`` is the recognised
        transform (``None`` -> no wrapping call allowed) and ``idx`` decomposes
        to the SAME affine index ``gather_base + gather_coeff*loop_var`` the
        comparison gathered.

        Comparing the write's index by its affine decomposition rather than
        against the bare loop variable keeps a shifted gather recognised
        (``if a[i + 1] > x: x = a[i + 1]`` reduces the same elements as its
        unshifted form) and rejects a write that reads a DIFFERENT element than
        the one compared.
        """
        try:
            tree = ast.parse(rhs_str, mode='eval').body
        except SyntaxError:
            return False
        tree = strip_identity(tree)
        if transform is not None:
            if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == transform
                    and len(tree.args) == 1 and not tree.keywords):
                return False
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return False  # an unexpected transform when none was matched
        if isinstance(tree, ast.Name):
            return tree.id == gather_sym
        if not (isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array):
            return False
        idx = tree.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        if isinstance(idx, (ast.Tuple, ast.Slice, ast.List)):
            return False
        try:
            idx_str = ast.unparse(idx)
        except Exception:  # pragma: no cover -- defensive
            return False
        aff = self._affine_index_in_loop_var(idx_str, loop_var, loop)
        if aff is None:
            return False
        base, coeff = aff
        return bool(symbolic.simplify(base - gather_base) == 0 and symbolic.simplify(coeff - gather_coeff) == 0)

    def _symbol_true_branch_writes_carrier(self,
                                           true_branch,
                                           loop: LoopRegion,
                                           carrier: str,
                                           array: str,
                                           gather_sym: str,
                                           gather_base: Any,
                                           gather_coeff: Any,
                                           transform: Optional[str] = None):
        """For the symbol-carrier case, verify the true-branch binds the value
        carrier (``carrier := [f](array[gather_base + gather_coeff*loop_var])``
        or ``carrier := [f](gather_sym)``, with the same gather transform ``f``
        the comparison used) and, optionally, ONE index carrier (``idx :=
        loop_var`` -- the argmax position, TSVC s315). The true-branch states
        must all be empty (no tasklet / AccessNode work). Any other iedge
        assignment is refused.

        :returns: ``(ok, idx_carrier)`` -- ``ok`` is True iff the value carrier
            write was found and every assignment was recognised; ``idx_carrier``
            is the index-carrier symbol name when an ``idx := loop_var`` write is
            present, else ``None``.
        """
        if not isinstance(true_branch, ControlFlowRegion):
            return False, None
        loop_var = loop.loop_variable
        carrier_write_seen = False
        idx_carrier = None
        for e in true_branch.edges():
            assigns = e.data.assignments or {}
            for lhs, rhs in assigns.items():
                rhs_str = str(rhs).strip()
                if lhs == carrier:
                    if not self._rhs_is_value_write(rhs_str, gather_sym, array, loop_var, loop, gather_base,
                                                    gather_coeff, transform):
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

    def _collect_preloop_assignments(self, loop: LoopRegion, sdfg: SDFG) -> dict:
        """Walk back the linear pre-loop chain (each block reached by a single
        in-edge) and accumulate interstate-edge assignments. The binding closest
        to the loop wins for each name (it is the value live at loop entry).

        The seed of a TSVC argmax is spread over this chain -- e.g.
        ``block: a_index := a[0]; index := 0`` then ``-> loop: maxv := abs(a_index)``.
        A binding staged through a transient scalar (``k := k_plus_inc`` with the
        arithmetic in a tasklet) is resolved to the arithmetic it stands for, so the
        seed-position check below sees a symbol value rather than an array name it
        must discard.
        """
        preloop: dict = {}
        parent = loop.parent_graph
        cur = loop
        seen: dict = {}
        while True:
            ins = parent.in_edges(cur)
            if len(ins) != 1:
                break
            e = ins[0]
            for lhs, rhs in (e.data.assignments or {}).items():
                if lhs not in preloop:  # closest-to-loop wins
                    staged = staged_iedge_rhs(str(rhs), e.src, sdfg)
                    preloop[lhs] = str(rhs) if staged is None else symbolic.symstr(staged)
            if e.src in seen or not isinstance(e.src, (SDFGState, ControlFlowRegion)):
                break
            seen[e.src] = None
            cur = e.src
        return preloop

    @staticmethod
    def _seed_position_negative(position: Any) -> Optional[bool]:
        """Is the seed's gathered array position negative? ``None`` when the sign is
        undecidable (a symbolic base such as ``a[K + i]``, where ``K == 0`` and
        ``K > 0`` want different slice lower bounds).

        Callers must pass a position with the pre-loop chain already substituted --
        a secondary-IV symbol left unresolved reads as undecidable when it is not.
        """
        try:
            return bool(symbolic.simplify(position) < 0)
        except TypeError:
            return None

    def _verify_affine_seed(self, loop: LoopRegion, sdfg: SDFG, value_carrier: str, idx_carrier: Optional[str],
                            array: str, base: Any, coeff: Any, start: Any, transform: Optional[str]) -> bool:
        """Verify the pre-loop seed sits where the rewrite assumes: the value
        carrier must be seeded ``value_carrier := [f](array[Q])`` with ``Q`` equal
        to the gather's seed-iteration position ``base + coeff*(start-1)``, and --
        when an index carrier is present -- ``idx_carrier := (start-1)``.

        This is the load-bearing assumption of every rewrite that DROPS the
        pre-loop bind and reconstructs the seed positionally: the transform+index
        buffer, whose ``buf[0] = f(a[base + coeff*(start-1)])`` stands in for the
        seed and whose index bind ``idx_carrier := (start-1) + idx_buf`` yields the
        seed's init index, and the plain symbol value-only reduction, whose emitted
        slice extends down to that same seed position. Nothing in the match forces
        the seed to read there -- a loop seeded from anywhere else reduces over a
        set that both omits the real seed and includes an element the loop never
        gathers -- so refuse when it cannot be proven.

        ``idx_carrier`` is ``None`` on the value-only path, where there is no index
        to check and the position comparison alone is the requirement.

        Handles the real frontend shape, where the seed is spread over a pre-loop
        chain with indirection: ``base`` / ``coeff`` carry the secondary-IV symbol
        ``k`` (bound pre-loop to ``inc``), and the value seed is ``maxv :=
        abs(a_index)`` with ``a_index := a[0]`` on an earlier edge. Both the
        position comparison and the gather lookup substitute the chain's bindings.
        """
        preloop = self._collect_preloop_assignments(loop, sdfg)
        if value_carrier not in preloop:
            return False
        if idx_carrier is not None and idx_carrier not in preloop:
            return False

        # Pure-symbol pre-loop bindings (skip carriers + array-read bindings like
        # ``a_index := a[0]``) -- these resolve the closed form's IV symbol ``k``.
        subs = {}
        for lhs, rhs in preloop.items():
            if lhs == value_carrier or lhs == idx_carrier:
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
        # The rewrite picks its slice lower bound off the sign of this position, so
        # an undecidable sign has no single correct lowering -- refuse rather than
        # assume in range.
        if self._seed_position_negative(seed_pos) is None:
            return False

        # Value seed: ``[f](array[Q])``, possibly indirected through a gather
        # symbol bound on an earlier pre-loop edge (``maxv := abs(a_index)``).
        q = self._seed_gather_position(preloop[value_carrier], array, transform, preloop)
        if q is None:
            return False
        q_resolved = _resolve(q)
        if q_resolved is None or symbolic.simplify(q_resolved - seed_pos) != 0:
            return False
        if idx_carrier is None:
            return True
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

    def _seed_iteration(self, m: _Match, start: Any) -> Any:
        """The iteration the pre-loop seed stands at.

        Normally ``start - 1``, backed off to ``start`` when that iteration
        would gather from a NEGATIVE array position -- i.e. the loop already
        starts at the array's first element, so the seed IS its first gathered
        element. The decision is on the gathered POSITION ``gather_base +
        gather_coeff*(start - 1)``, not on the iteration itself, so a SHIFTED
        gather keeps its in-bounds seed: ``a[i + 1]`` over ``0:N-1`` seeds at
        position 0 from iteration ``-1``.

        The sign is always decidable here -- :meth:`_verify_affine_seed` makes it a
        precondition of every base-folding match -- so an undecidable position never
        reaches the rewrite to be silently assumed in range (which would emit the
        negative lower bound ``a[K - 1 : N]``, reading ``a[-1]`` at ``K == 0``).
        """
        iter_lo = symbolic.simplify(start - 1)
        if self._seed_position_negative(m.gather_base + m.gather_coeff * iter_lo):
            return symbolic.simplify(start)
        return iter_lo

    def _gather_range(self, m: _Match, iter_lo: Any, iter_hi: Any) -> Tuple[Any, Any]:
        """Inclusive ARRAY-POSITION bounds of the gather ``arr[gather_base +
        gather_coeff*i]`` over the iterations ``iter_lo .. iter_hi``."""
        return (symbolic.simplify(m.gather_base + m.gather_coeff * iter_lo),
                symbolic.simplify(m.gather_base + m.gather_coeff * iter_hi))

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
        # The slice is in ARRAY-POSITION space, so a SHIFTED gather (``a[i + 1]``)
        # reduces over exactly the elements its unshifted form would. A symbol
        # carrier extends down to the seed iteration -- its dropped pre-loop bind
        # no longer materialises the seed (TSVC s314 seeds from ``a[0]`` at
        # ``start = 1``); a scalar / length-1 carrier keeps the seed in its
        # AccessNode, so it starts at ``start``. Unit stride is gated in ``_match``.
        iter_lo = self._seed_iteration(m, start) if m.carrier_kind == 'symbol' else start
        pos_lo, pos_hi = self._gather_range(m, iter_lo, end)
        input_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(pos_lo, pos_hi, 1)]))
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
        from dace.libraries.standard.nodes import ArgReduce
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        val_buf, _ = sdfg.add_scalar(f'_argmax_val_{m.loop.label}', arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argmax_idx_{m.loop.label}', dtypes.int64, transient=True, find_new_name=True)

        # Include the seed ``a[start-1]`` in the slice. This path is gated to the
        # unit ``a[i]`` gather, so the position and the iteration coincide.
        slice_lo = self._seed_iteration(m, start)
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
            rev = end - symbolic.symbol('_j')
            mat_state.add_mapped_tasklet(
                name='reverse_gather',
                map_ranges={'_j': (0, n_elems - 1, 1)},
                inputs={'__in': mm.Memlet(data=m.input_array, subset=subsets.Range([(rev, rev, 1)]))},
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
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        # Include the seed ``a[start-1]``; buf spans the slice. This path is gated
        # to the unit ``a[i]`` gather, so position and iteration coincide.
        slice_lo = self._seed_iteration(m, start)
        n_elems = symbolic.simplify(end + 1 - slice_lo)

        buf, _ = sdfg.add_array(f'_argf_buf_{m.loop.label}', [n_elems], arr_dtype, transient=True, find_new_name=True)

        # Materialisation state: buf[_j] = f(input_array[slice_lo + _j]).
        mat_state = m.parent.add_state(m.loop.label + '_argf')
        gather = slice_lo + symbolic.symbol('_j')
        mat_state.add_mapped_tasklet(
            name='transform_gather',
            map_ranges={'_j': (0, n_elems - 1, 1)},
            inputs={'__in': mm.Memlet(data=m.input_array, subset=subsets.Range([(gather, gather, 1)]))},
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
        and ``index`` tracking the iteration of the max -- with a single
        :class:`~dace.libraries.standard.nodes.ArgReduce` reading the gather
        DIRECTLY, yielding the value + the slice-local index.

        The gather is affine ``a[gather_base + gather_coeff*i]``; iteration
        ``i = iter_lo + j`` reads array position ``pos_lo + coeff*j`` with
        ``pos_lo = base + coeff*(start-1)`` (the seed sits at ``i = start-1``,
        where ``index`` still holds its init value). That is exactly a strided
        slice ``a[pos_lo : pos_hi : coeff]``, which the ArgReduce takes as its
        operand, with ``transform=f`` applied per element as it reads. So there
        is no buffer: the abs and the arg-reduction happen in one streaming pass
        over ``a``. Staging ``buf[j] = f(a[pos_lo + coeff*j])`` first, as this
        used to, wrote and re-read a whole extra copy of the array to hold a
        value the scan computes in a register. The recovered iteration index is
        ``index := iter_lo + idx_buf`` (``idx_buf`` is slice-local). Both carrier
        seeds are dropped from the inbound iedges -- the slice's first element
        subsumes them.

        Under a non-strict guard (``m.last_wins``) the sequential loop keeps the
        LAST extreme while the ArgReduce scan keeps the first, so the scan runs
        over a REVERSED copy (``buf[j]`` <-> iteration ``i = end - j``, i.e.
        array position ``base + coeff*end - coeff*j``) and the index maps back as
        ``index := end - idx_buf``. A memlet range walks upward, so reversing the
        order is the one case that still needs a materialised gather; no
        non-strict guard occurs in TSVC.
        """
        from dace.libraries.standard.nodes import ArgReduce
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype
        coeff = m.gather_coeff

        # Iterations covered, INCLUDING the seed at ``i = start-1`` (where the
        # index carrier still holds its init value). ``j`` ranges 0..n-1 over
        # ``i = iter_lo + j``.
        iter_lo = self._seed_iteration(m, start)
        n_elems = symbolic.simplify(end - iter_lo + 1)
        # Array position of ``buf[j]``: pos(i) = base + coeff*i, i = iter_lo + j.
        pos_lo, pos_hi = self._gather_range(m, iter_lo, end)

        val_buf, _ = sdfg.add_scalar(f'_argfi_val_{m.loop.label}', arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argfi_idx_{m.loop.label}', dtypes.int64, transient=True, find_new_name=True)

        # Forward scan reads the gather in place. Only the reversed order needs a materialised
        # copy: ``buf[_j]`` holds iteration ``end - _j`` (position ``pos_hi - coeff*_j``), which
        # covers exactly the same positions in the opposite direction, and a memlet range cannot
        # be walked downward.
        argmax_state = m.parent.add_state(m.loop.label + '_argfi_reduce')
        entry_state = argmax_state
        buf = None
        if m.last_wins:
            buf, _ = sdfg.add_array(f'_argfi_buf_{m.loop.label}', [n_elems],
                                    arr_dtype,
                                    transient=True,
                                    find_new_name=True)
            mat_state = m.parent.add_state(m.loop.label + '_argfi')
            in_idx = pos_hi - coeff * symbolic.symbol('_j')
            mat_state.add_mapped_tasklet(
                name='transform_gather_idx',
                map_ranges={'_j': (0, n_elems - 1, 1)},
                inputs={'__in': mm.Memlet(data=m.input_array, subset=subsets.Range([(in_idx, in_idx, 1)]))},
                code=f'__out = {m.transform}(__in)',
                outputs={'__out': mm.Memlet(data=buf, subset='_j')},
                external_edges=True,
            )
            entry_state = mat_state

        # Re-route inbound edges to the first state of the replacement, dropping the pre-loop
        # binds of BOTH carriers (the arg-reduce over the slice subsumes them).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(m.idx_carrier_name, None)
            m.parent.add_edge(ie.src, entry_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        if m.last_wins:
            m.parent.add_edge(entry_state, argmax_state, dace.InterstateEdge())

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

        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = 'max' if m.op == dtypes.ReductionType.Max else 'min'
        if m.last_wins:
            read = argmax_state.add_read(buf)
            in_memlet = mm.Memlet(data=buf, subset=subsets.Range([(0, symbolic.simplify(n_elems - 1), 1)]))
            # The transform is already applied while materialising the reversed copy.
            node = ArgReduce(name=f'{m.loop.label}_argfi_argreduce', op=op)
        else:
            read = argmax_state.add_read(m.input_array)
            # ``volume`` is stated rather than left to the subset: a subset counts itself as
            # ``ceiling((hi - lo + 1) / step)``, which a symbolic stride (s318's ``inc``) leaves
            # unresolved. The iteration count is known here exactly.
            in_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(pos_lo, pos_hi, coeff)]), volume=n_elems)
            node = ArgReduce(name=f'{m.loop.label}_argfi_argreduce', op=op, transform=m.transform)
        argmax_state.add_node(node)
        argmax_state.add_edge(read, None, node, '_in', in_memlet)
        argmax_state.add_edge(node, '_out_val', wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, '_out_idx', wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()

    # ------------------- predicate index (TSVC s331) -------------------

    def match_predicate_index(self, loop: LoopRegion, sdfg: SDFG) -> Optional[MatchPredIndex]:
        """Match ``for i: if pred(a[i]): j = i`` -- a position tracked with NO value
        carrier, i.e. ``j = max{i : pred}`` seeded from the pre-loop value of ``j``.

        Runs only after :meth:`_match` / :meth:`_match_2d` have refused, so a loop
        that carries a comparison against a value carrier never reaches here. The
        loop skeleton is the SAME gate the value-carrier path uses
        (:meth:`guarded_loop_skeleton`); only the branch payload differs -- an index
        write with no value carrier, against a guard that names no carrier at all.
        """
        skeleton = self.guarded_loop_skeleton(loop)
        if skeleton is None:
            return None
        start, end, _cond_block, cond_codeblock, true_branch = skeleton

        idx_carrier = self.true_branch_writes_index_only(true_branch, loop.loop_variable)
        if idx_carrier is None or idx_carrier not in sdfg.symbols:
            return None

        # The ConditionalBlock owns no edges of its own, so ``loop.edges()`` is
        # exactly the pre-guard binding chain (``a_index := a[i]``, ``t := a_index < 0``).
        bindings: Dict[str, str] = {}
        for e in loop.edges():
            if not e.data.is_unconditional():
                return None
            for lhs, rhs in (e.data.assignments or {}).items():
                if lhs in bindings or lhs == idx_carrier or lhs == loop.loop_variable:
                    return None  # rebound within one iteration -> order-dependent
                bindings[lhs] = str(rhs)

        wired = self.wire_guard(cond_codeblock.as_string.strip(), bindings, idx_carrier, loop, sdfg)
        if wired is None:
            return None
        guard_code, guard_inputs = wired

        seed = self.predicate_seed(loop, idx_carrier, bindings, start)
        if seed is None:
            return None
        return MatchPredIndex(loop=loop,
                              parent=loop.parent_graph,
                              idx_carrier=idx_carrier,
                              seed=seed,
                              guard_code=guard_code,
                              guard_inputs=guard_inputs,
                              iter_start=start,
                              iter_end=end)

    def true_branch_writes_index_only(self, true_branch, loop_var: str) -> Optional[str]:
        """The one index carrier the true branch writes (``idx := loop_var``), or
        ``None`` if it writes anything else, writes twice, or holds any node work --
        the rewrite drops the branch wholesale, so a second effect would be lost.
        """
        if not isinstance(true_branch, ControlFlowRegion):
            return None
        idx_carrier = None
        for e in true_branch.edges():
            if not e.data.is_unconditional():
                return None
            for lhs, rhs in (e.data.assignments or {}).items():
                if idx_carrier is not None or str(rhs).strip() != loop_var:
                    return None
                idx_carrier = lhs
        for n in true_branch.nodes():
            if not isinstance(n, SDFGState) or len(n.nodes()) > 0:
                return None  # nested control flow / tasklet work is out of scope
        return idx_carrier

    def wire_guard(self, guard_str: str, bindings: Dict[str, str], idx_carrier: str, loop: LoopRegion,
                   sdfg: SDFG) -> Optional[Tuple[str, List[Tuple[str, mm.Memlet]]]]:
        """Turn the branch guard into mask-tasklet code plus the data inputs it reads.

        The pre-guard bindings are substituted to a fixed point, then
        :class:`GuardReadWiring` replaces every data read with an input connector.
        Whatever names survive must be the loop variable or a loop-invariant symbol;
        a call (``abs(a[i]) > t``) leaves its callee as a bare name that is no
        symbol, so it is refused rather than emitted into the tasklet unresolved.
        Naming ``idx_carrier`` is refused too: a guard reading the tracked position
        is a find-FIRST search, not a max over positions.
        """
        try:
            tree = ast.parse(guard_str, mode='eval').body
        except SyntaxError:
            return None
        # One round per binding suffices for an acyclic chain; the bound stops a
        # cyclic one (``x := y`` on one edge, ``y := x`` on another).
        for _round in range(len(bindings) + 1):
            finder = astutils.ASTFindReplace(dict(bindings))
            tree = finder.visit(tree)
            if finder.replace_count == 0:
                break
        else:
            return None

        wiring = GuardReadWiring(sdfg, loop)
        tree = wiring.visit(tree)
        if wiring.refused:
            return None
        connectors = dict.fromkeys(conn for conn, _memlet in wiring.reads.values())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name):
                continue
            if node.id in connectors or node.id == loop.loop_variable:
                continue
            if node.id == idx_carrier or node.id in bindings or node.id not in sdfg.symbols:
                return None
        try:
            code = ast.unparse(ast.fix_missing_locations(tree))
        except Exception:  # pragma: no cover -- defensive
            return None
        return code, list(wiring.reads.values())

    def predicate_seed(self, loop: LoopRegion, idx_carrier: str, bindings: Dict[str, str], start: Any) -> Optional[Any]:
        """``idx_carrier``'s pre-loop value, iff loop-invariant and provably ``<= start``.

        That inequality is the whole soundness argument of the lift: the mask folds
        non-matching iterations to the seed, so the max returns the seed exactly when
        no iteration matched -- but only while every matching position (all ``>=
        start``) still beats it. An undecidable comparison is refused, never assumed.

        The binding is looked up on a chain of PLAIN STATES each reached by exactly one
        in-edge, so nothing can rebind the carrier between the seed and the loop; a
        region on the chain (a prior loop writing the same symbol) leaves the collected
        value stale and is refused rather than trusted.
        """
        parent = loop.parent_graph
        cur, rhs = loop, None
        walked: Dict[Any, None] = {}
        while rhs is None:
            ins = parent.in_edges(cur)
            if len(ins) != 1:
                return None
            edge = ins[0]
            rhs = (edge.data.assignments or {}).get(idx_carrier)
            if rhs is None:
                if not isinstance(edge.src, SDFGState) or edge.src in walked:
                    return None  # a cycle has no single pre-loop value to read
                walked[edge.src] = None
                cur = edge.src
        try:
            seed = symbolic.pystr_to_symbolic(str(rhs))
        except Exception:  # pragma: no cover -- defensive
            return None
        blocked = dict.fromkeys([loop.loop_variable, idx_carrier, *bindings])
        if any(str(s) in blocked for s in seed.free_symbols):
            return None
        try:
            if not bool(symbolic.simplify(seed - start) <= 0):
                return None
        except TypeError:
            return None
        return seed

    def rewrite_predicate_index(self, m: MatchPredIndex, sdfg: SDFG):
        """Replace the loop with ``init(seed) -> Map(masked index, WCR max) -> bind``.

        The WCR-on-scalar map is the shape ``LoopToReduce(prefer='wcr-scalar')``
        produces, which codegen lowers to a parallel reduction; the bind state
        re-materialises the carrier symbol from the private scalar.
        """
        priv, _ = sdfg.add_scalar(f'_pred_index_{m.loop.label}',
                                  sdfg.symbols[m.idx_carrier],
                                  transient=True,
                                  find_new_name=True)
        seed_str = symbolic.symstr(m.seed)
        priv_subset = subsets.Range([(0, 0, 1)])

        init_state = m.parent.add_state(m.loop.label + '_predidx_init')
        seed_tasklet = init_state.add_tasklet('pred_index_seed', {}, dict.fromkeys(['__out']), f'__out = {seed_str}')
        init_state.add_edge(seed_tasklet, '__out', init_state.add_write(priv), None,
                            mm.Memlet(data=priv, subset=copy.deepcopy(priv_subset)))

        # The mask's false value IS the seed, so a non-matching iteration folds to the
        # fold's identity and an all-false range leaves the seeded scalar untouched.
        ivar = m.loop.loop_variable
        map_state = m.parent.add_state(m.loop.label + '_predidx')
        map_state.add_mapped_tasklet(
            name='pred_index',
            map_ranges={ivar: subsets.Range([(symbolic.simplify(m.iter_start), symbolic.simplify(m.iter_end), 1)])},
            inputs={
                conn: copy.deepcopy(memlet)
                for conn, memlet in m.guard_inputs
            },
            code=f'__out = ({ivar} if ({m.guard_code}) else ({seed_str}))',
            outputs={'__out': mm.Memlet(data=priv, subset=copy.deepcopy(priv_subset), wcr='lambda a, b: max(a, b)')},
            external_edges=True)

        bind_state = m.parent.add_state(m.loop.label + '_predidx_bind')
        for ie in list(m.parent.in_edges(m.loop)):
            m.parent.add_edge(ie.src, init_state, ie.data)
            m.parent.remove_edge(ie)
        m.parent.add_edge(init_state, map_state, dace.InterstateEdge())
        m.parent.add_edge(map_state, bind_state, dace.InterstateEdge(assignments={m.idx_carrier: priv}))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)
        sdfg.reset_cfg_list()


__all__ = ['ArgMaxLift']
