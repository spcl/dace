# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift TSVC-style ``if a[i] OP bestv: bestv = a[i]`` argmax/argmin loops to a
:class:`~dace.libraries.standard.nodes.reduce.Reduce` libnode."""

import ast
import copy
import functools
import operator
import re
from typing import Any, NamedTuple, TypeVar

import numpy as np

import dace
from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.frontend.python import astutils
from dace.libraries.standard.nodes.reduce import Reduce
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_expr, as_range, coeff_of, simplified
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    SDFGState,
    is_written,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.induction_variable_substitution import staged_iedge_rhs

#: Map AST comparison op class -> DaCe reduction type.
CMP_AST_TO_RTYPE: dict[type[ast.cmpop], dtypes.ReductionType] = {
    ast.Gt: dtypes.ReductionType.Max,
    ast.GtE: dtypes.ReductionType.Max,
    ast.Lt: dtypes.ReductionType.Min,
    ast.LtE: dtypes.ReductionType.Min,
}

#: ``(start, end, conditional block, guard, true branch)`` of an ascending, break-free guarded loop.
LoopSkeleton = tuple[
    symbolic.SymbolicType, symbolic.SymbolicType, ConditionalBlock, properties.CodeBlock, ControlFlowRegion
]

#: A comparison operator class, as ``ast`` parses ``gather OP carrier``.
CompareOp = type[ast.cmpop]

BlockT = TypeVar("BlockT", bound=ControlFlowBlock)


#: The combine of a NaN-propagating fold ``if x != x or x > acc: acc = x`` (``<`` for a min), verbatim: a NaN
#: element replaces the accumulator, and a NaN accumulator is never replaced by a number (``x > nan`` is false).
NAN_PROPAGATING_WCR: dict[dtypes.ReductionType, str] = {
    dtypes.ReductionType.Max: "lambda a, b: b if (b != b or b > a) else a",
    dtypes.ReductionType.Min: "lambda a, b: b if (b != b or b < a) else a",
}

#: The combine of a strict fold ``if x > acc: acc = x``: ``max(acc, x)`` keeps the accumulator on a tie and on a
#: NaN ``x`` (``acc < nan`` is false), exactly as the guard does; likewise ``min`` for ``<``.
STRICT_WCR: dict[dtypes.ReductionType, str] = {
    dtypes.ReductionType.Max: "lambda a, b: max(a, b)",
    dtypes.ReductionType.Min: "lambda a, b: min(a, b)",
}


#: The comparison that holds with its operands swapped.
MIRRORED_CMP: dict[type, type] = {ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Lt: ast.Gt, ast.LtE: ast.GtE}


class PartialReduction(NamedTuple):
    """A reduction nest of any depth folding into one element ``carrier[K]``
    (:meth:`ArgMaxLift.match_partial_reduction`)."""

    loops: list[LoopRegion]
    parent: ControlFlowRegion
    carrier: str
    carrier_subset: subsets.Range
    source: str
    source_subset: subsets.Range
    wcr: str
    extents: list[tuple[Any, Any]]


class MaskedArgReduction(NamedTuple):
    """A guarded argmin / argmax ``if mask and source[I] OP carrier[K]: index[J] = i; carrier[K] = source[I]``
    (:meth:`ArgMaxLift.match_masked_arg`), over the loop ``reduced``; ``column`` is an independent loop between it and
    the guard, which the rewrite makes the outer one."""

    reduced: LoopRegion
    column: LoopRegion | None
    carrier: str
    carrier_subset: subsets.Range
    index: str
    index_subset: subsets.Range
    guard: str  # wired: every data read a connector of ``inputs``
    gather: ast.Subscript
    wired_gather: str
    wired_seed: str
    inputs: dict[str, mm.Memlet]
    op: dtypes.ReductionType
    last_wins: bool
    start: Any
    end: Any


class _Match(NamedTuple):
    """A matched argmax/argmin loop."""

    op: dtypes.ReductionType
    loop: LoopRegion
    parent: ControlFlowRegion
    carrier_name: str
    carrier_kind: str
    carrier_subset: subsets.Range | None
    input_array: str
    iter_start: Any
    iter_end: Any
    idx_carrier_name: str | None = None
    transform: str | None = None  # unary gather transform ('abs') or None
    gather_base: Any = 0
    gather_coeff: Any = 1
    last_wins: bool = False
    keeps_seed: bool = False  # the fold starts from the carrier's value, which is no element of the gather
    stride: Any = 1  # of the loop: only a value fold takes one, its gather striding along


class _Match2D(NamedTuple):
    """A matched contiguous argmax/argmin over a perfect nest ``for i0: ... for i(n-1):`` of any depth ``n >= 2`` (TSVC
    s3110 / s13110 at ``n = 2``)."""

    op: dtypes.ReductionType
    loops: list[LoopRegion]
    parent: ControlFlowRegion
    carrier_name: str
    idx_names: list[str]
    input_array: str
    shape: list[Any]
    last_wins: bool = False


class MatchPredIndex(NamedTuple):
    """A matched "last index at which a predicate holds" loop (TSVC ``s331``)."""

    loop: LoopRegion
    parent: ControlFlowRegion
    idx_carrier: str
    seed: Any
    guard_code: str
    guard_inputs: list[tuple[str, mm.Memlet]]
    iter_start: Any
    iter_end: Any


class GuardReadWiring(ast.NodeTransformer):
    """Replace every data read in a predicate guard by a tasklet input connector."""

    def __init__(self, sdfg: SDFG, loop: LoopRegion) -> None:
        self.sdfg = sdfg
        self.loop = loop
        self.reads: dict[str, tuple[str, mm.Memlet]] = {}
        self.refused = False

    def connector(self, key: str, array: str, subset: subsets.Range) -> ast.Name:
        entry = self.reads.get(key)
        if entry is None:
            entry = (f"__guard{len(self.reads)}", mm.Memlet(data=array, subset=subset))
            self.reads[key] = entry
        return ast.Name(id=entry[0], ctx=ast.Load())

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not (isinstance(node.value, ast.Name) and node.value.id in self.sdfg.arrays):
            self.refused = True
            return node
        array = node.value.id
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        # One index per dimension: a rank mismatch would emit a subset of the wrong dimensionality.
        if len(indices) != len(self.sdfg.arrays[array].shape) or any(
            isinstance(idx, (ast.Slice, ast.List, ast.Starred)) for idx in indices
        ):
            self.refused = True
            return node
        positions = [self.position(ast.unparse(idx)) for idx in indices]
        if any(pos is None for pos in positions):
            self.refused = True
            return node
        key = f"{array}[{', '.join(str(pos) for pos in positions)}]"
        return ast.copy_location(self.connector(key, array, subsets.Range([(pos, pos, 1) for pos in positions])), node)

    def position(self, idx_str: str) -> Any | None:
        """The single array position a guard read touches -- affine in the loop variable, or wholly loop-invariant."""
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
        assigned: dict[str, None] = {}
        for e in self.loop.all_interstate_edges():
            assigned.update(dict.fromkeys((e.data.assignments or {}).keys()))
        if any(str(s) in assigned for s in idx.free_symbols):
            return None  # a per-iteration symbol in the index is not loop-invariant
        return symbolic.simplify(idx)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self.sdfg.arrays:
            return node
        desc = self.sdfg.arrays[node.id]
        if not (isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1,))):
            self.refused = True
            return node
        return ast.copy_location(self.connector(node.id, node.id, subsets.Range([(0, 0, 1)])), node)


#: Right-hand operand values that make a binary op the identity on its left operand.
IDENTITY_RHS: tuple[tuple[type[ast.operator], int], ...] = ((ast.Add, 0), (ast.Sub, 0), (ast.Mult, 1), (ast.Div, 1))
#: ... and the mirrored form, for the commutative ops only.
IDENTITY_LHS: tuple[tuple[type[ast.operator], int], ...] = ((ast.Add, 0), (ast.Mult, 1))


def strip_identity(expr: ast.AST) -> ast.AST:
    """``expr`` with outer identity arithmetic (``+ 0``, ``- 0``, ``* 1``, ``/ 1``) peeled off."""
    while isinstance(expr, ast.BinOp):
        right, left = expr.right, expr.left
        if (
            isinstance(right, ast.Constant)
            and isinstance(right.value, (int, float))
            and not isinstance(right.value, bool)
            and any(isinstance(expr.op, op) and right.value == v for op, v in IDENTITY_RHS)
        ):
            expr = left
            continue
        if (
            isinstance(left, ast.Constant)
            and isinstance(left.value, (int, float))
            and not isinstance(left.value, bool)
            and any(isinstance(expr.op, op) and left.value == v for op, v in IDENTITY_LHS)
        ):
            expr = right
            continue
        return expr
    return expr


def bound_expression(text: str, bindings: dict[str, str]) -> ast.expr | None:
    """``text`` parsed, with a bare name replaced by its binding until it is no longer a bound name."""
    seen: set[str] = set()
    try:
        tree = ast.parse(text, mode="eval").body
        while isinstance(tree, ast.Name) and tree.id in bindings and tree.id not in seen:
            seen.add(tree.id)
            tree = ast.parse(bindings[tree.id], mode="eval").body
    except SyntaxError:
        return None
    return tree


def neutral_extreme(numeric_type: type, op: dtypes.ReductionType) -> int | float:
    """The neutral element of a Max (most negative value) or Min (most positive value) over ``numeric_type``."""
    limits = np.finfo(numeric_type) if np.issubdtype(numeric_type, np.floating) else np.iinfo(numeric_type)
    return numeric_type(limits.min if op == dtypes.ReductionType.Max else limits.max).item()


@properties.make_properties
@xf.explicit_cf_compatible
class ArgMaxLift(ppl.Pass):
    """Lift TSVC-style argmax/argmin loops to :class:`Reduce` libnodes."""

    CATEGORY: str = "Optimization Preparation"

    tie_break = properties.Property(
        dtype=str,
        default="infer",
        choices=["infer", "first", "last"],
        desc="Which of several equal extremes the tracked index refers to. 'infer' (default) derives it from the "
        "guard's strictness -- strict (> / <) keeps the FIRST occurrence, non-strict (>= / <=) the LAST -- "
        "reproducing the sequential loop. 'first' / 'last' pin the rule explicitly. Only meaningful when an "
        "index is tracked: the extreme VALUE is the same either way.",
    )

    def __init__(self, tie_break: str = "infer", only: list | None = None) -> None:
        super().__init__()
        self.tie_break = tie_break
        # When given, only these loops are candidates (the segments a split left behind).
        self._only = only

    def _resolve_last_wins(self, op_ast: CompareOp, has_index: bool) -> bool:
        """The tie rule for this match: True -> keep the LAST occurrence of the extreme (``tie='last'``), False -> the
        FIRST (``tie='first'``)."""
        if not has_index:
            return False
        if self.tie_break == "first":
            return False
        if self.tie_break == "last":
            return True
        return op_ast in (ast.GtE, ast.LtE)

    def _contains_break(self, block: ControlFlowBlock) -> bool:
        """Whether ``block`` contains a :class:`BreakBlock` anywhere beneath it."""
        if isinstance(block, BreakBlock):
            return True
        if isinstance(block, ConditionalBlock):
            return any(self._contains_break(br) for condition, br in block.branches)
        if isinstance(block, ControlFlowRegion):
            return any(self._contains_break(n) for n in block.nodes())
        return False

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        rewritten = 0
        only = None if self._only is None else {id(loop) for loop in self._only}
        if only is not None and not only:
            return None
        for sd in sdfg.all_sdfgs_recursive():
            replaced: set[int] = set()  # every region under a loop a rewrite removed
            for region in list(sd.all_control_flow_regions()):
                if not (isinstance(region, LoopRegion) and region.loop_variable):
                    continue
                if only is not None and id(region) not in only:
                    continue
                # Stale-snapshot guard: a previous rewrite may have removed this
                # LoopRegion from its parent, or the loop around it.
                if region.parent_graph is None or region not in region.parent_graph.nodes() or id(region) in replaced:
                    continue
                # Both matchers gate on this skeleton, whose break check walks the whole body.
                skeleton = self.guarded_loop_skeleton(region)
                m = self._match(region, sd, skeleton) if skeleton is not None else None
                if m is not None:
                    self._rewrite(m, sd)
                    rewritten += 1
                    continue
                # Contiguous nested argmax of any depth (TSVC s3110 / s13110 in 2-D).
                m2 = self._match_2d(region, sd)
                if m2 is not None:
                    self._rewrite_2d(m2, sd)
                    replaced.update(id(r) for r in region.all_control_flow_regions())
                    rewritten += 1
                    continue
                # Predicate-index, no value carrier (TSVC s331).
                mp = self.match_predicate_index(region, sd, skeleton) if skeleton is not None else None
                if mp is not None:
                    self.rewrite_predicate_index(mp, sd)
                    rewritten += 1
                    continue
                # A max / min / any / all fold into one element, nested to any depth.
                mr = self.match_partial_reduction(region, sd)
                if mr is not None:
                    self.rewrite_partial_reduction(mr, sd)
                    replaced.update(id(r) for r in region.all_control_flow_regions())
                    rewritten += 1
                    continue
                # A masked argmin / argmax into array elements, under an independent column loop or not (CloudSC).
                ma = self.match_masked_arg(region, sd)
                if ma is not None:
                    replaced.update(id(r) for r in region.all_control_flow_regions())
                    self.rewrite_masked_arg(ma, sd)
                    rewritten += 1
        return rewritten or None

    def guarded_loop_skeleton(self, loop: LoopRegion) -> LoopSkeleton | None:
        """The gate every payload analysis in this pass shares."""
        # Body shape first: it is a scan of the loop's direct children, while the bound parses
        # below are sympy round trips and the break check walks the whole body.
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
        if cond_block is None or self.stray_assignment(loop, cond_block):
            return None

        non_else = [(c, br) for c, br in cond_block.branches if c is not None]
        if len(non_else) != 1:
            return None
        if any(self._branch_has_content(br) for c, br in cond_block.branches if c is None):
            return None

        if self._contains_break(loop):
            return None
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None
        try:
            if int(as_expr(symbolic.simplify(stride))) < 1:
                return None
        except (TypeError, ValueError):
            return None

        guard, true_branch = non_else[0]
        return start, end, cond_block, guard, true_branch

    def _match(self, loop: LoopRegion, sdfg: SDFG, skeleton: LoopSkeleton | None = None) -> _Match | None:
        skeleton = skeleton if skeleton is not None else self.guarded_loop_skeleton(loop)
        if skeleton is None:
            return None
        start, end, cond_block, cond_codeblock, true_branch = skeleton
        stride = symbolic.simplify(required(loop_analysis.get_loop_stride(loop)))

        guard = self._guard_comparison(loop, sdfg, cond_block, cond_codeblock, true_branch)
        if guard is None:
            return None
        op_ast, op, gather_sym_name, carrier_name, transform, input_array, gather_base, gather_coeff = guard
        # The two halves of the affine gather constrain the rewrites separately: * a non-unit COEFF (``arr[inc*i]``,
        # TSVC s318) gathers a strided set, which only the transform+index path handles (its ArgReduce takes the stride
        # and the transform as operand properties); * a non-zero BASE merely SHIFTS the gathered slice.
        unit_coeff = bool(symbolic.simplify(gather_coeff) == 1)
        zero_base = bool(symbolic.simplify(gather_base) == 0)

        # Classify the carrier's storage first
        carrier_kind, carrier_subset = self._classify_carrier(carrier_name, sdfg)
        if carrier_kind is None:
            return None

        idx_carrier_name = None
        keeps_seed = False
        if carrier_kind in ("scalar", "length_one_array"):
            # Data-carrier path: the in-loop write lives on AccessNode chains
            if transform is not None or not unit_coeff:
                return None
            for e in true_branch.edges():
                if e.data.assignments:
                    return None
            true_state = self._extract_singleton_state(true_branch)
            if true_state is None:
                return None
            if not self._true_state_writes_carrier_from_array(
                true_state, loop, carrier_name, input_array, gather_base, gather_coeff
            ):
                return None
        else:
            # Symbol-carrier path: the in-loop write is an iedge assignment ``carrier := [f](arr[b+c*i])`` (or ``carrier
            # := [f](gather_sym)`` where ``gather_sym`` was bound to ``arr[b+c*i]``) inside the true-branch, OPTIONALLY
            # plus an index carrier ``idx := loop_var`` (argmax position, s315).
            ok, idx_carrier_name = self._symbol_true_branch_writes_carrier(
                true_branch,
                loop,
                carrier_name,
                input_array,
                gather_sym_name,
                gather_base,
                gather_coeff,
                transform=transform,
            )
            if not ok:
                return None
            # An index carrier must itself be a symbol (bound back via iedge).
            if idx_carrier_name is not None and idx_carrier_name not in sdfg.symbols:
                return None
            # A strided gather (s318) is ONLY lowered on the combined transform+index path
            # (``_rewrite_with_transform_and_index``), whose ArgReduce reads the strided slice directly.
            has_transform_and_index = transform is not None and idx_carrier_name is not None
            if not unit_coeff and not has_transform_and_index:
                return None
            # A SHIFTED gather is folded into the emitted slice by the plain value-only rewrite and by the
            # transform+index buffer.
            value_only = transform is None and idx_carrier_name is None
            folds_base = has_transform_and_index or value_only
            if not zero_base and not folds_base:
                return None
            # Every symbol-carrier rewrite DROPS the pre-loop bind and reconstructs the seed positionally at ``base +
            # coeff*(start-1)`` -- the combined path by starting its slice there (plus an index init of ``start-1``),
            # the others by extending the emitted slice down to it.
            keeps_seed = stride != 1 or not self._verify_affine_seed(
                loop, sdfg, carrier_name, idx_carrier_name, input_array, gather_base, gather_coeff, start, transform
            )
            if keeps_seed and not value_only:
                return None

        # Tie-break semantics (``tie_break``; 'infer' reads it off the guard's strictness): a non-strict guard (``>=`` /
        # ``<=``) updates the carrier on a TIE, so the sequential loop keeps the LAST occurrence's index, while a strict
        # guard keeps the FIRST; ArgReduce's ``tie`` states which.
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
            keeps_seed=keeps_seed,
            stride=stride,
        )

    def _guard_comparison(
        self,
        loop: LoopRegion,
        sdfg: SDFG,
        cond_block: ConditionalBlock,
        cond_codeblock: properties.CodeBlock,
        true_branch: ControlFlowRegion,
    ) -> tuple | None:
        """The guard ``arr[b + c*i] OP carrier``: its comparison operator (AST and reduction type), the gather symbol,
        the carrier, the gather's transform, array, base and coefficient."""
        cond_expr_str = cond_codeblock.as_string.strip()
        # The comparison is a symbol bound upstream (``tmp := (g OP c)``) or inline in the condition (s314/s316).
        inline_gather = None
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cond_expr_str):
            op_ast, gather_sym_name, carrier_name, transform = self._resolve_tmp_iedge(loop, cond_block, cond_expr_str)
        else:
            op_ast, gather_sym_name, carrier_name, transform = self._parse_comparison(cond_expr_str)
            if op_ast is None:
                # An inline array subscript ``array[b + c*i] OP carrier`` (s315).
                parsed = self._parse_inline_subscript_comparison(cond_expr_str, loop.loop_variable, loop)
                if parsed is not None:
                    op_ast, inline_gather, carrier_name, transform = parsed
                    gather_sym_name = None
        if op_ast is None or carrier_name is None:
            return None
        # ``x < a[i]`` is ``a[i] > x``: the carrier is the name the branch writes, the gather the one bound to an element
        if (
            inline_gather is None
            and transform is None
            and (
                gather_sym_name in sdfg.arrays
                or any(gather_sym_name in e.data.assignments for e in true_branch.all_interstate_edges())
            )
            and self._resolve_gather_iedge(loop, cond_block, carrier_name, loop.loop_variable, sdfg) is not None
        ):
            gather_sym_name, carrier_name, op_ast = carrier_name, gather_sym_name, MIRRORED_CMP[op_ast]
        op = CMP_AST_TO_RTYPE[op_ast]

        # The gather ``arr[b + c*i]``: the inline subscript, or an interstate binding one level back.
        if inline_gather is not None:
            input_array, gather_base, gather_coeff = inline_gather
            if input_array not in sdfg.arrays:
                return None
        else:
            if gather_sym_name is None:
                return None
            gather = self._resolve_gather_iedge(loop, cond_block, gather_sym_name, loop.loop_variable, sdfg)
            if gather is None:
                return None
            input_array, gather_base, gather_coeff = gather
        return op_ast, op, gather_sym_name, carrier_name, transform, input_array, gather_base, gather_coeff

    # match helpers

    # 2-D contiguous nested argmax (TSVC s3110 / s13110).

    def _single_child_region(self, region: ControlFlowRegion, want_type: type[BlockT]) -> BlockT | None:
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
        return None if found is None or self.stray_assignment(region, found) else found

    @staticmethod
    def stray_assignment(region: ControlFlowRegion, child: ControlFlowBlock) -> bool:
        """Whether an edge of ``region`` binds a name ``child`` does not read, directly or through another binding:
        per-iteration work the rewrites would drop with the loop."""
        bound = {k: str(v) for e in region.edges() for k, v in e.data.assignments.items()}
        if not bound:
            return False
        used = {str(sym) for sym in child.free_symbols}
        pending = dict(bound)
        while hits := [name for name in pending if name in used]:
            for name in hits:
                used |= {str(sym) for sym in symbolic.pystr_to_symbolic(pending.pop(name)).free_symbols}
        return bool(pending)

    def _unit_loop_from_zero(self, loop: LoopRegion) -> tuple[symbolic.SymbolicType, symbolic.SymbolicType] | None:
        """``(start, end)`` for a unit-stride loop starting at 0, else ``None``."""
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None
        try:
            if int(as_expr(symbolic.simplify(stride))) != 1:
                return None
        except (TypeError, ValueError):
            return None
        if symbolic.simplify(start) != 0:
            return None
        return start, end

    def _match_2d(self, outer_loop: LoopRegion, sdfg: SDFG) -> _Match2D | None:
        """Match a perfect nest ``for i0: ... for i(n-1): if aa[i0, ..., i(n-1)] OP maxv: maxv = aa[...]; idx0 = i0;
        ...; idx(n-1) = i(n-1)`` over the full (contiguous) ``n``-D array, ``n >= 2``."""
        if not outer_loop.loop_variable:
            return None
        # Nest shape first: a scan of each region's direct children, cheaper than the break walk and bound parses.
        nest = self._perfect_nest_to_guard(outer_loop)
        if nest is None:
            return None
        loops, cond_block = nest
        # A break anywhere in the nest makes it an early-exit search, not a reduction over the whole (contiguous) array
        if self._contains_break(outer_loop):
            return None
        ranges = [self._unit_loop_from_zero(loop) for loop in loops]
        if any(r is None for r in ranges):
            return None

        non_else = [(c, br) for c, br in cond_block.branches if c is not None]
        else_branches = [(c, br) for c, br in cond_block.branches if c is None]
        if len(non_else) != 1 or any(self._branch_has_content(br) for _, br in else_branches):
            return None
        cond_codeblock, true_branch = non_else[0]
        cond_expr_str = cond_codeblock.as_string.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cond_expr_str):
            op_ast, gather_sym, carrier_name, transform = self._resolve_tmp_iedge(loops[-1], cond_block, cond_expr_str)
        else:
            op_ast, gather_sym, carrier_name, transform = self._parse_comparison(cond_expr_str)
        if op_ast is None or transform is not None or carrier_name is None or gather_sym is None:
            return None  # transforms (abs) are out of scope for the nested path
        # The nest always tracks one index per loop, so the tie rule is always observable and ``tie_break`` always
        # applies: under the default 'infer' the guard's strictness picks the occurrence -- a non-strict guard (``>=`` /
        # ``<=``) keeps the LAST one in row-major order (``tie='last'``).
        last_wins = self._resolve_last_wins(op_ast, True)
        op = CMP_AST_TO_RTYPE[op_ast]

        loop_vars = [loop.loop_variable for loop in loops]
        if len(set(loop_vars)) != len(loop_vars):
            return None
        array = self._resolve_gather_2d(loops[-1], gather_sym, loop_vars, sdfg)
        if array is None:
            return None
        # Value carrier must be a symbol
        if carrier_name not in sdfg.symbols:
            return None
        idx_names = self._match_2d_true_branch(true_branch, carrier_name, gather_sym, array, loop_vars, sdfg)
        if idx_names is None:
            return None
        if not self._seeded_at_origin(outer_loop, sdfg, carrier_name, idx_names, array):
            return None
        desc = sdfg.arrays[array]
        if not self._scans_whole_array(desc, ranges):
            return None
        return _Match2D(
            op=op,
            loops=loops,
            parent=outer_loop.parent_graph,
            carrier_name=carrier_name,
            idx_names=idx_names,
            input_array=array,
            shape=list(desc.shape),
            last_wins=last_wins,
        )

    def _perfect_nest_to_guard(self, outer_loop: LoopRegion) -> tuple[list[LoopRegion], ConditionalBlock] | None:
        """The loops of a perfect nest of at least two from ``outer_loop``, and the one conditional block it holds."""
        loops = [outer_loop]
        while True:
            inner = self._single_child_region(loops[-1], LoopRegion)
            if inner is None:
                break
            if not inner.loop_variable:
                return None
            loops.append(inner)
        if len(loops) < 2:
            return None
        cond_block = self._single_child_region(loops[-1], ConditionalBlock)
        return None if cond_block is None else (loops, cond_block)

    def _seeded_at_origin(
        self, outer_loop: LoopRegion, sdfg: SDFG, carrier_name: str, idx_names: list[str], array: str
    ) -> bool:
        """Whether the carriers are seeded with the first element and index zero: the rewrite drops the seeds."""
        preloop = self._collect_preloop_assignments(outer_loop, sdfg)
        if carrier_name not in preloop or any(symbolic.pystr_to_symbolic(preloop.get(n, "1")) != 0 for n in idx_names):
            return False
        position = self._seed_gather_position(preloop[carrier_name], array, None, preloop)
        return position is not None and all(symbolic.pystr_to_symbolic(p) == 0 for p in position.strip("()").split(","))

    @staticmethod
    def _scans_whole_array(desc: data.Data, ranges: list) -> bool:
        """Whether loops over ``ranges`` visit exactly the whole C-contiguous array, so a flat arg-reduce matches."""
        if len(desc.shape) != len(ranges) or not desc.is_packed_c_strides():
            return False
        # Loop bounds are reparsed while the shape carries its assumptions: one name, two sympy instances.
        equalized = symbolic.equalize_symbols_across(*(r[1] for r in ranges), *desc.shape)
        ends, dims = equalized[: len(ranges)], equalized[len(ranges) :]
        if any(symbolic.simplify(end - (dim - 1)) != 0 for end, dim in zip(ends, dims, strict=True)):
            return False
        return subsets.Range([(0, dim - 1, 1) for dim in desc.shape]).is_contiguous_subset(desc)

    def _parse_2d_gather(self, rhs_str: str, loop_vars: list[str]) -> str | None:
        """Return the array name iff ``rhs_str`` is exactly ``arr[v0, ..., v(n-1)]`` for ``loop_vars`` (a point
        access, the outermost loop's variable in dimension 0, and so on)."""
        try:
            tree = ast.parse(str(rhs_str), mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(tree, ast.Subscript) or not isinstance(tree.value, ast.Name):
            return None
        idx = tree.slice
        if not isinstance(idx, ast.Tuple) or len(idx.elts) != len(loop_vars):
            return None
        if not all(isinstance(d, ast.Name) and d.id == var for d, var in zip(idx.elts, loop_vars, strict=True)):
            return None
        return tree.value.id

    def _resolve_gather_2d(
        self, inner_loop: LoopRegion, gather_sym: str, loop_vars: list[str], sdfg: SDFG
    ) -> str | None:
        """Find an iedge binding ``gather_sym := arr[v0, ..., v(n-1)]`` in the
        innermost loop and return ``arr`` (validated against ``sdfg.arrays``)."""
        for e in inner_loop.all_interstate_edges():
            rhs = (e.data.assignments or {}).get(gather_sym)
            if rhs is None:
                continue
            arr = self._parse_2d_gather(rhs, loop_vars)
            if arr is not None and arr in sdfg.arrays:
                return arr
        return None

    def _match_2d_true_branch(
        self,
        true_branch: ControlFlowBlock,
        carrier: str,
        gather_sym: str,
        array: str,
        loop_vars: list[str],
        sdfg: SDFG,
    ) -> list[str] | None:
        """Verify the true-branch binds exactly ``carrier := arr[v0, ...]`` (or ``:= gather_sym``) and one ``idx_t :=
        v_t`` per loop via iedges, with empty states and no other writes."""
        if not isinstance(true_branch, ControlFlowRegion):
            return None
        carrier_seen = False
        idx: dict[str, str] = {}
        for e in true_branch.edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                rhs_str = str(rhs).strip()
                if lhs == carrier:
                    if rhs_str != gather_sym and self._parse_2d_gather(rhs_str, loop_vars) != array:
                        return None
                    carrier_seen = True
                elif rhs_str in loop_vars and rhs_str not in idx:
                    idx[rhs_str] = lhs
                else:
                    return None
        for n in true_branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return None
            if not isinstance(n, SDFGState):
                return None  # nested control flow unsupported
        if not carrier_seen or len(idx) != len(loop_vars):
            return None
        names = [idx[var] for var in loop_vars]
        if len(set(names)) != len(names) or any(name not in sdfg.symbols for name in names):
            return None
        return names

    def _rewrite_2d(self, m: _Match2D, sdfg: SDFG) -> None:
        """Replace the nested argmax with a flat :class:`ArgReduce` over the whole (contiguous) array, decomposing the
        flat index back into the index carriers (``idx_t = (mflat // stride_t) % shape_t``, ``stride_t`` the product of
        the extents inside ``t``; in 2-D ``mflat // ncols`` and ``mflat % ncols``)."""
        from dace.libraries.standard.nodes import ArgReduce

        outer_loop = m.loops[0]
        desc = sdfg.arrays[m.input_array]
        val_buf, _ = sdfg.add_scalar(
            f"_argmax2d_val_{outer_loop.label}", desc.dtype, transient=True, find_new_name=True
        )
        idx_buf, _ = sdfg.add_scalar(
            f"_argmax2d_idx_{outer_loop.label}", dtypes.int64, transient=True, find_new_name=True
        )

        argmax_state = m.parent.add_state(outer_loop.label + "_argreduce2d")

        # Re-route inbound edges, dropping pre-loop seeds of every carrier.
        for ie in list(m.parent.in_edges(outer_loop)):
            new_assigns = dict(ie.data.assignments or {})
            for k in (m.carrier_name, *m.idx_names):
                new_assigns.pop(k, None)
            m.parent.add_edge(
                ie.src, argmax_state, dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            )
            m.parent.remove_edge(ie)

        assignments = {m.carrier_name: val_buf}
        for t, (name, dim) in enumerate(zip(m.idx_names, m.shape, strict=True)):
            inside = m.shape[t + 1 :]
            stride = symbolic.symstr(symbolic.simplify(functools.reduce(operator.mul, inside))) if inside else None
            position = f"int_floor({idx_buf}, {stride})" if stride is not None else idx_buf
            assignments[name] = position if t == 0 else f"py_mod({position}, {symbolic.symstr(dim)})"
        bind_state = m.parent.add_state(outer_loop.label + "_argreduce2d_bind")
        m.parent.add_edge(argmax_state, bind_state, dace.InterstateEdge(assignments=assignments))
        for oe in list(m.parent.out_edges(outer_loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(outer_loop)

        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = "max" if m.op == dtypes.ReductionType.Max else "min"
        node = ArgReduce(name=f"{outer_loop.label}_argreduce2d", op=op, tie="last" if m.last_wins else "first")
        argmax_state.add_node(node)
        read = argmax_state.add_read(m.input_array)
        in_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(0, dim - 1, 1) for dim in desc.shape]))
        argmax_state.add_edge(read, None, node, "_in", in_memlet)
        argmax_state.add_edge(node, "_out_val", wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, "_out_idx", wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))

    def _branch_has_content(self, branch: ControlFlowBlock) -> bool:
        if not isinstance(branch, ControlFlowRegion):
            return False
        if any(e.data.assignments for e in branch.edges()):
            return True
        return any(not isinstance(n, SDFGState) or len(n.nodes()) > 0 for n in branch.nodes())

    def _parse_compare_node(self, tree: ast.expr) -> tuple[CompareOp | None, str | None, str | None, str | None]:
        """Extract ``(op_cls, gather_name, carrier_name, transform)`` from a :class:`ast.Compare` node ``gather OP
        carrier`` (or ``f(gather) OP carrier``)."""
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
            return None, None, None, None
        op_cls = type(tree.ops[0])
        if op_cls not in CMP_AST_TO_RTYPE:
            return None, None, None, None
        transform, lhs_name = self._extract_transform(tree.left)
        rhs_name = self._extract_name(tree.comparators[0])
        if lhs_name is None or rhs_name is None:
            return None, None, None, None
        return op_cls, lhs_name, rhs_name, transform

    def _parse_inline_subscript_comparison(
        self, expr_str: str, loop_var: str, loop: LoopRegion
    ) -> tuple[CompareOp, tuple[str, symbolic.SymbolicType, symbolic.SymbolicType], str, str | None] | None:
        """Parse a comparison whose gather is an INLINE array subscript: ``[f](array[b + c*i]) OP carrier``."""
        try:
            tree = ast.parse(expr_str, mode="eval").body
        except SyntaxError:
            return None
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
            return None
        op_cls = type(tree.ops[0])
        if op_cls not in CMP_AST_TO_RTYPE:
            return None
        carrier = self._extract_name(tree.comparators[0])
        if carrier is None:
            return None
        left, transform = tree.left, None
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id in self.SUPPORTED_TRANSFORMS
            and len(left.args) == 1
            and not left.keywords
        ):
            transform, left = left.func.id, left.args[0]
        if not (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name)):
            return None
        array = left.value.id
        idx = left.slice
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

    def _resolve_tmp_iedge(
        self, loop: LoopRegion, cond_block: ConditionalBlock, tmp_sym: str
    ) -> tuple[CompareOp | None, str | None, str | None, str | None]:
        """Walk in-edges of ``cond_block`` looking for one whose assignment binds ``tmp_sym`` to a comparison ``[f](g)
        OP c``."""
        for ie in loop.in_edges(cond_block):
            assigns = ie.data.assignments or {}
            rhs = assigns.get(tmp_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode="eval").body
            except SyntaxError:
                continue
            res = self._parse_compare_node(tree)
            if res[0] is not None:
                return res
        return None, None, None, None

    def _parse_comparison(self, expr_str: str) -> tuple[CompareOp | None, str | None, str | None, str | None]:
        """Parse a comparison ``[f](g) OP c`` inlined directly in the condition."""
        try:
            tree = ast.parse(expr_str, mode="eval").body
        except SyntaxError:
            return None, None, None, None
        return self._parse_compare_node(tree)

    def _resolve_gather_iedge(
        self, loop: LoopRegion, cond_block: ConditionalBlock, gather_sym: str, loop_var: str, sdfg: SDFG
    ) -> tuple[str, Any, Any] | None:
        """Walk back two levels to find an iedge binding ``gather_sym = arr[idx]`` where ``idx`` is an AFFINE function
        of the loop variable ``b + c*i``."""
        # Collect every iedge in the loop and try to find one binding ``gather_sym``.
        for e in loop.all_interstate_edges():
            assigns = e.data.assignments or {}
            rhs = assigns.get(gather_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode="eval").body
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
            # Only a single 1-D affine index is handled
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
    def _affine_index_in_loop_var(
        idx_str: str, loop_var: str, loop: LoopRegion
    ) -> tuple[symbolic.SymbolicType, symbolic.SymbolicType] | None:
        """Decompose a gather index ``idx_str`` as ``base + coeff*loop_var``."""
        try:
            idx = symbolic.pystr_to_symbolic(idx_str)
            lv = symbolic.pystr_to_symbolic(loop_var)
        except Exception:  # pragma: no cover -- defensive parse guard
            return None
        # membership and .coeff both go through identity; equalize before either
        idx, lv = symbolic.equalize_symbols_across(idx, lv)
        if lv not in idx.free_symbols:
            return None
        # Linear-in-``lv`` decomposition: idx == base + coeff*lv with neither term referencing ``lv`` (so no ``lv**2``
        # etc.).
        idx = as_expr(idx).expand()
        coeff = coeff_of(idx, as_expr(lv), 1)
        base = coeff_of(idx, as_expr(lv), 0)
        if symbolic.simplify(idx - (base + coeff * lv)) != 0:
            return None
        if lv in coeff.free_symbols or lv in base.free_symbols:
            return None
        # Every symbol feeding base / coeff must be loop-invariant: not assigned
        # on any body interstate edge (a varying stride/base breaks the closed form).
        body_assigned: dict[str, None] = {}
        for e in loop.all_interstate_edges():
            body_assigned.update(dict.fromkeys((e.data.assignments or {}).keys()))
        # membership-only scan (early-exit), iteration order does not affect the result
        for s in dict.fromkeys([*coeff.free_symbols, *base.free_symbols]):
            if str(s) in body_assigned:
                return None
        return base, coeff

    def _extract_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        return None

    #: Recognised unary gather transforms ``f(g)`` -> the Python builtin name.
    SUPPORTED_TRANSFORMS: dict[str, None] = dict.fromkeys(["abs"])

    def _extract_transform(self, node: ast.AST) -> tuple[str | None, str | None]:
        """Return ``(transform, name)`` for a possibly-transformed operand."""
        if isinstance(node, ast.Name):
            return None, node.id
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.SUPPORTED_TRANSFORMS
            and len(node.args) == 1
            and not node.keywords
        ):
            inner = self._extract_name(node.args[0])
            if inner is not None:
                return node.func.id, inner
        return None, None

    def _extract_singleton_state(self, branch: ControlFlowBlock) -> SDFGState | None:
        if not isinstance(branch, ControlFlowRegion) or not all(isinstance(n, SDFGState) for n in branch.nodes()):
            return None
        content_states = [n for n in branch.nodes() if n.number_of_nodes() > 0]
        if len(content_states) != 1:
            return None
        return content_states[0]

    def _true_state_writes_carrier_from_array(
        self, state: SDFGState, loop: LoopRegion, carrier: str, array: str, gather_base: Any, gather_coeff: Any
    ) -> bool:
        """Check the true-branch state has the shape ``arr -> arr_index_AN ->
        assign_tasklet -> carrier_AN`` writing ``carrier = arr[gather_base +
        gather_coeff*loop_var]`` -- the SAME element the comparison gathered."""
        # Single write AccessNode for the carrier.
        carrier_writes = [n for n in state.data_nodes() if n.data == carrier and is_written(state, n)]
        if len(carrier_writes) != 1:
            return False
        carrier_an = carrier_writes[0]
        # **Strict refusal**: the true-branch may only have ONE terminal AccessNode (the carrier).
        terminal_outs = [n for n in state.data_nodes() if state.in_degree(n) > 0 and state.out_degree(n) == 0]
        if len(terminal_outs) != 1 or terminal_outs[0] is not carrier_an:
            return False
        # Walk back through the chain (AN <- [Tasklet?] <- AN <- ...
        source_an = self._walk_back_to_source(state, carrier_an)
        if source_an is None or source_an.data != array:
            return False
        # Verify the memlet from the source array gathers the single element the comparison gathered.
        out_edges = list(state.out_edges(source_an))
        if not out_edges:
            return False
        loop_var = loop.loop_variable
        for oe in out_edges:
            if oe.data is None or oe.data.subset is None:
                continue
            ranges = as_range(oe.data.subset).ranges
            if len(ranges) != 1:
                continue
            lo, hi, step = ranges[0]
            if symbolic.simplify(as_expr(hi) - as_expr(lo)) != 0 or symbolic.simplify(step) != 1:
                continue  # not a single point
            aff = self._affine_index_in_loop_var(str(lo), loop_var, loop)
            if aff is None:
                continue
            base, coeff = aff
            if symbolic.simplify(base - gather_base) == 0 and symbolic.simplify(coeff - gather_coeff) == 0:
                return True
        return False

    def _walk_back_to_source(self, state: SDFGState, carrier_an: nodes.AccessNode) -> nodes.AccessNode | None:
        """Walk back from ``carrier_an`` through a chain of transients and identity Tasklets to the source AccessNode
        (the array we're reading from)."""
        cur = carrier_an
        while True:
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            upstream = ins[0].src
            if isinstance(upstream, nodes.Tasklet):
                t_ins = list(state.in_edges(upstream))
                if len(t_ins) != 1 or not AugAssignToWCR.is_copy(upstream, ins[0].src_conn, t_ins[0].dst_conn):
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

    def _classify_carrier(self, name: str, sdfg: SDFG) -> tuple[str | None, subsets.Range | None]:
        desc = sdfg.arrays.get(name)
        if desc is not None:
            if isinstance(desc, data.Scalar):
                return "scalar", subsets.Range([(0, 0, 1)])
            if isinstance(desc, data.Array) and tuple(desc.shape) == (1,):
                return "length_one_array", subsets.Range([(0, 0, 1)])
            return None, None
        # Symbol carrier: present in ``sdfg.symbols`` but not in ``sdfg.arrays``.
        if name in sdfg.symbols:
            return "symbol", None
        return None, None

    def _rhs_is_value_write(
        self,
        rhs_str: str,
        gather_sym: str | None,
        array: str,
        loop_var: str,
        loop: LoopRegion,
        gather_base: symbolic.SymbolicType,
        gather_coeff: symbolic.SymbolicType,
        transform: str | None,
    ) -> bool:
        """True iff ``rhs_str`` is the value-carrier write under ``transform``: ``[f](gather_sym)`` or
        ``[f](array[idx])``, where ``f`` is the recognised transform (``None`` -> no wrapping call allowed) and ``idx``
        decomposes to the SAME affine index ``gather_base + gather_coeff*loop_var`` the comparison gathered."""
        try:
            tree: ast.AST = ast.parse(rhs_str, mode="eval").body
        except SyntaxError:
            return False
        tree = strip_identity(tree)
        if transform is not None:
            if not (
                isinstance(tree, ast.Call)
                and isinstance(tree.func, ast.Name)
                and tree.func.id == transform
                and len(tree.args) == 1
                and not tree.keywords
            ):
                return False
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return False  # an unexpected transform when none was matched
        if isinstance(tree, ast.Name):
            return tree.id == gather_sym
        if not (isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array):
            return False
        idx = tree.slice
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

    def _symbol_true_branch_writes_carrier(
        self,
        true_branch: ControlFlowBlock,
        loop: LoopRegion,
        carrier: str,
        array: str,
        gather_sym: str | None,
        gather_base: symbolic.SymbolicType,
        gather_coeff: symbolic.SymbolicType,
        transform: str | None = None,
    ) -> tuple[bool, str | None]:
        """For the symbol-carrier case, verify the true-branch binds the value carrier (``carrier :=
        [f](array[gather_base + gather_coeff*loop_var])`` or ``carrier := [f](gather_sym)``, with the same gather
        transform ``f`` the comparison used) and, optionally, ONE index carrier (``idx := loop_var`` -- the argmax
        position, TSVC s315)."""
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
                    if not self._rhs_is_value_write(
                        rhs_str, gather_sym, array, loop_var, loop, gather_base, gather_coeff, transform
                    ):
                        return False, None
                    carrier_write_seen = True
                elif rhs_str == loop_var and idx_carrier is None:
                    # Index carrier := the loop variable (argmax position).
                    idx_carrier = lhs
                else:
                    return False, None  # unrecognised extra write -> refuse
        # All true-branch states must be empty -- any contained tasklet / AccessNode work is a separate side effect.
        for n in true_branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return False, None
            if not isinstance(n, SDFGState):
                return False, None  # nested control flow not supported in v1
        return carrier_write_seen, idx_carrier

    def _collect_preloop_assignments(self, loop: LoopRegion, sdfg: SDFG) -> dict[str, str]:
        """Walk back the linear pre-loop chain (each block reached by a single in-edge) and accumulate interstate-edge
        assignments."""
        preloop: dict[str, str] = {}
        parent = loop.parent_graph
        cur: ControlFlowBlock = loop
        seen: dict = {}
        overwritten: set[str] = set()  # by a block between the binding and the loop: the binding is no longer live
        while True:
            ins = parent.in_edges(cur)
            if len(ins) != 1:
                break
            e = ins[0]
            for lhs, rhs in (e.data.assignments or {}).items():
                if (
                    lhs not in preloop
                    and lhs not in overwritten
                    and overwritten.isdisjoint(re.findall(r"\w+", str(rhs)))
                ):
                    staged = staged_iedge_rhs(str(rhs), e.src, sdfg)
                    preloop[lhs] = str(rhs) if staged is None else symbolic.symstr(staged)
            overwritten.update(e.data.assignments or {})  # a farther binding of it is shadowed
            if e.src in seen or not isinstance(e.src, (SDFGState, ControlFlowRegion)):
                break
            seen[e.src] = None
            overwritten.update(e.src.read_and_write_sets()[1])
            if isinstance(e.src, ControlFlowRegion):
                overwritten.update(k for edge in e.src.all_interstate_edges() for k in edge.data.assignments)
            cur = e.src
        return preloop

    @staticmethod
    def _seed_position_negative(position: Any) -> bool | None:
        """Is the seed's gathered array position negative?"""
        try:
            return bool(symbolic.simplify(position) < 0)
        except TypeError:
            return None

    def _verify_affine_seed(
        self,
        loop: LoopRegion,
        sdfg: SDFG,
        value_carrier: str,
        idx_carrier: str | None,
        array: str,
        base: Any,
        coeff: Any,
        start: Any,
        transform: str | None,
    ) -> bool:
        """Check the pre-loop seed is ``value_carrier := [f](array[base + coeff*(start-1)])`` and, with an index
        carrier, ``idx_carrier := start-1`` -- ``start`` for both when ``start-1`` gathers a negative position (see
        :meth:`_seed_iteration`)."""
        preloop = self._collect_preloop_assignments(loop, sdfg)
        if value_carrier not in preloop:
            return False
        if idx_carrier is not None and idx_carrier not in preloop:
            return False

        # Pure-symbol pre-loop bindings (skip carriers + array-read bindings like ``a_index := a[0]``)
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

        def _resolve(expr: object) -> symbolic.SymbolicType | None:
            try:
                return symbolic.simplify(symbolic.pystr_to_symbolic(str(expr)).subs(subs))
            except Exception:  # pragma: no cover -- defensive
                return None

        seed_pos = _resolve(base + coeff * (start - 1))
        seed_idx = _resolve(start - 1)
        if seed_pos is None or seed_idx is None:
            return False
        # The rewrite picks its slice lower bound off the sign of this position, so an undecidable sign has no single
        # correct lowering -- refuse rather than assume in range.
        negative = self._seed_position_negative(seed_pos)
        if negative is None:
            return False
        # A loop that already starts at the array's first element seeds from that element
        # (``x = a[0]; for i in range(N)``): the seed stands at ``start``, as in :meth:`_seed_iteration`.
        if negative:
            seed_pos, seed_idx = _resolve(base + coeff * start), _resolve(start)
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
        if idx_carrier is None:
            return True
        # Index init must equal start-1.
        idx_resolved = _resolve(preloop[idx_carrier])
        if idx_resolved is None or symbolic.simplify(idx_resolved - seed_idx) != 0:
            return False
        return True

    def _seed_gather_position(
        self, rhs_str: str, array: str, transform: str | None, preloop: dict[str, str]
    ) -> str | None:
        """If ``rhs_str`` is ``[f](array[P])`` -- or ``[f](g)`` where ``g`` is a
        pre-loop symbol bound to ``array[P]`` -- return the index string ``P``;
        else ``None``."""
        # A renamed copy (``maxv := maxv_0``, ``maxv_0 := abs(a[0])``) binds the seed one name away.
        tree = bound_expression(rhs_str, preloop)
        if transform is not None:
            if not (
                isinstance(tree, ast.Call)
                and isinstance(tree.func, ast.Name)
                and tree.func.id == transform
                and len(tree.args) == 1
                and not tree.keywords
            ):
                return None
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return None
        # Gather-symbol indirection (``g`` bound to ``a[P]``).
        if isinstance(tree, ast.Name):
            tree = bound_expression(tree.id, preloop)
        if not (isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array):
            return None
        idx = tree.slice
        try:
            return ast.unparse(idx)
        except Exception:  # pragma: no cover -- defensive
            return None

    # rewrite

    def _seed_iteration(self, m: _Match, start: Any) -> Any:
        """The iteration the pre-loop seed stands at."""
        iter_lo = symbolic.simplify(start - 1)
        if self._seed_position_negative(m.gather_base + m.gather_coeff * iter_lo):
            return symbolic.simplify(start)
        return iter_lo

    def _gather_range(self, m: _Match, iter_lo: Any, iter_hi: Any) -> tuple[Any, Any]:
        """Inclusive ARRAY-POSITION bounds of the gather ``arr[gather_base +
        gather_coeff*i]`` over the iterations ``iter_lo .. iter_hi``."""
        return (
            symbolic.simplify(m.gather_base + m.gather_coeff * iter_lo),
            symbolic.simplify(m.gather_base + m.gather_coeff * iter_hi),
        )

    def _rewrite(self, m: _Match, sdfg: SDFG) -> None:
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
        if m.carrier_kind == "symbol":
            # Fresh transient scalar -> Reduce output -> iedge bind to the symbol.
            output_dtype = sdfg.symbols[m.carrier_name]
            out_name, _ = sdfg.add_scalar(
                f"_arg_max_buf_{m.loop.label}", output_dtype, transient=True, find_new_name=True
            )
            output_subset = subsets.Range([(0, 0, 1)])
        else:
            out_name = m.carrier_name
            output_subset = required(m.carrier_subset)

        # Reduce-state replaces the loop.
        reduce_state = m.parent.add_state(m.loop.label + "_argmax")
        entry = reduce_state
        if m.keeps_seed:
            entry = m.parent.add_state(m.loop.label + "_argmax_seed")
            m.parent.add_edge(entry, reduce_state, dace.InterstateEdge())
            seed = entry.add_tasklet("seed", {}, {"_out": None}, f"_out = {m.carrier_name}")
            entry.add_edge(
                seed,
                "_out",
                entry.add_write(out_name),
                None,
                mm.Memlet(data=out_name, subset=copy.deepcopy(output_subset)),
            )
        # Re-route inbound edges.
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            if m.carrier_kind == "symbol" and m.carrier_name in new_assigns and not m.keeps_seed:
                del new_assigns[m.carrier_name]
            new_iedge = dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            m.parent.add_edge(ie.src, entry, new_iedge)
            m.parent.remove_edge(ie)

        if m.carrier_kind == "symbol":
            # Insert a bind state AFTER the reduce so the carrier symbol gets re-materialised from the transient scalar
            # before any downstream state references it.
            bind_state = m.parent.add_state(m.loop.label + "_argmax_bind")
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

        # For scalar / length-1 array carriers the pre-loop init ``x = a[0]`` is already in a prior state writing to the
        # same AccessNode; the libnode's ``identity=None`` semantics fold that pre-existing value into the running
        # reduction (WCR-Max).
        wcr_str = "lambda a, b: max(a, b)" if m.op == dtypes.ReductionType.Max else "lambda a, b: min(a, b)"
        # Identity (accumulator seed) for the reduction.
        if m.carrier_kind == "symbol" and not m.keeps_seed:
            numeric_type = sdfg.arrays[m.input_array].dtype.type
            identity = neutral_extreme(numeric_type, m.op)
        else:
            identity = None
        node = Reduce(name=f"{m.loop.label}_argmax_reduce", wcr=wcr_str, axes=[0], identity=identity)
        node.add_in_connector("_in")
        node.add_out_connector("_out")
        reduce_state.add_node(node)
        # The slice is in ARRAY-POSITION space, so a SHIFTED gather (``a[i + 1]``) reduces over exactly the elements its
        # unshifted form would.
        iter_lo = self._seed_iteration(m, start) if m.carrier_kind == "symbol" and not m.keeps_seed else start
        pos_lo, pos_hi = self._gather_range(m, iter_lo, end)
        input_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(pos_lo, pos_hi, m.stride)]))
        reduce_state.add_edge(read, None, node, "_in", input_memlet)
        output_memlet = mm.Memlet(data=out_name, subset=output_subset)
        reduce_state.add_edge(node, "_out", write, None, output_memlet)

    def _rewrite_with_index(self, m: _Match, sdfg: SDFG) -> None:
        """Replace an argmax/argmin-with-index loop (TSVC s315) with a two-output ``ArgReduce``."""
        from dace.libraries.standard.nodes import ArgReduce

        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        val_buf, _ = sdfg.add_scalar(f"_argmax_val_{m.loop.label}", arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f"_argmax_idx_{m.loop.label}", dtypes.int64, transient=True, find_new_name=True)

        # Include the seed ``a[start-1]`` in the slice.
        slice_lo = self._seed_iteration(m, start)
        lo_is_zero = bool(symbolic.simplify(slice_lo) == 0)

        argmax_state = m.parent.add_state(m.loop.label + "_argreduce")

        # Re-route inbound edges, dropping pre-loop binds of BOTH carriers.
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(required(m.idx_carrier_name), None)
            m.parent.add_edge(
                ie.src, argmax_state, dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            )
            m.parent.remove_edge(ie)

        # Bind state: re-materialise both carrier symbols from the scalars (bare names).
        idx_rhs = idx_buf if lo_is_zero else f"({symbolic.symstr(slice_lo)} + {idx_buf})"
        bind_state = m.parent.add_state(m.loop.label + "_argreduce_bind")
        m.parent.add_edge(
            argmax_state,
            bind_state,
            dace.InterstateEdge(assignments={m.carrier_name: val_buf, required(m.idx_carrier_name): idx_rhs}),
        )
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = "max" if m.op == dtypes.ReductionType.Max else "min"
        node = ArgReduce(name=f"{m.loop.label}_argreduce", op=op, tie="last" if m.last_wins else "first")
        argmax_state.add_node(node)
        read = argmax_state.add_read(m.input_array)
        in_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(slice_lo, end, 1)]))
        argmax_state.add_edge(read, None, node, "_in", in_memlet)
        argmax_state.add_edge(node, "_out_val", wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, "_out_idx", wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))

    def _rewrite_with_transform(self, m: _Match, sdfg: SDFG) -> None:
        """Replace a transformed value-only reduction (TSVC s3113, ``maxv = max(|a[i]|)``) with: a map materialising
        ``buf[j] = f(a[lo+j])`` into a fresh contiguous transient, then a :class:`Reduce` over ``buf``."""
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        # Include the seed ``a[start-1]``; buf spans the slice.
        slice_lo = self._seed_iteration(m, start)
        n_elems = symbolic.simplify(end + 1 - slice_lo)

        buf, _ = sdfg.add_array(f"_argf_buf_{m.loop.label}", [n_elems], arr_dtype, transient=True, find_new_name=True)

        # Materialisation state: buf[_j] = f(input_array[slice_lo + _j]).
        mat_state = m.parent.add_state(m.loop.label + "_argf")
        gather = slice_lo + symbolic.symbol("_j")
        mat_state.add_mapped_tasklet(
            name="transform_gather",
            map_ranges={"_j": (0, n_elems - 1, 1)},
            inputs={"__in": mm.Memlet(data=m.input_array, subset=subsets.Range([(gather, gather, 1)]))},
            code=f"__out = {m.transform}(__in)",
            outputs={"__out": mm.Memlet(data=buf, subset="_j")},
            external_edges=True,
        )

        # Reduce over the (contiguous) buffer.
        out_dtype = sdfg.symbols[m.carrier_name]
        out_name, _ = sdfg.add_scalar(f"_argf_val_{m.loop.label}", out_dtype, transient=True, find_new_name=True)
        reduce_state = m.parent.add_state(m.loop.label + "_argf_reduce")

        # Re-route inbound edges to the materialise state, dropping the pre-loop
        # carrier seed (the reduce over buf subsumes it).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            m.parent.add_edge(
                ie.src, mat_state, dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            )
            m.parent.remove_edge(ie)
        m.parent.add_edge(mat_state, reduce_state, dace.InterstateEdge())

        # Bind state: carrier := reduced value.
        bind_state = m.parent.add_state(m.loop.label + "_argf_bind")
        m.parent.add_edge(reduce_state, bind_state, dace.InterstateEdge(assignments={m.carrier_name: out_name}))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        read = reduce_state.add_read(buf)
        write = reduce_state.add_write(out_name)
        wcr_str = "lambda a, b: max(a, b)" if m.op == dtypes.ReductionType.Max else "lambda a, b: min(a, b)"
        identity = neutral_extreme(arr_dtype.type, m.op)
        node = Reduce(name=f"{m.loop.label}_argf_reduce", wcr=wcr_str, axes=[0], identity=identity)
        node.add_in_connector("_in")
        node.add_out_connector("_out")
        reduce_state.add_node(node)
        reduce_state.add_edge(
            read, None, node, "_in", mm.Memlet(data=buf, subset=subsets.Range([(0, symbolic.simplify(n_elems - 1), 1)]))
        )
        reduce_state.add_edge(node, "_out", write, None, mm.Memlet(data=out_name, subset=subsets.Range([(0, 0, 1)])))

    def _rewrite_with_transform_and_index(self, m: _Match, sdfg: SDFG) -> None:
        """Replace a transformed argmax/argmin with index over an affine gather (TSVC s318, ``max(|a[inc*i]|)``) by one
        ``ArgReduce`` reading the strided slice ``a[pos_lo : pos_hi : coeff]`` directly, ``transform`` applied per
        element; ``index := iter_lo + idx_buf``."""
        from dace.libraries.standard.nodes import ArgReduce

        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype
        coeff = m.gather_coeff

        # Iterations covered, INCLUDING the seed at ``i = start-1`` (where the index carrier still holds its init
        # value).
        iter_lo = self._seed_iteration(m, start)
        n_elems = symbolic.simplify(end - iter_lo + 1)
        # Array position of ``buf[j]``: pos(i) = base + coeff*i, i = iter_lo + j.
        pos_lo, pos_hi = self._gather_range(m, iter_lo, end)

        val_buf, _ = sdfg.add_scalar(f"_argfi_val_{m.loop.label}", arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f"_argfi_idx_{m.loop.label}", dtypes.int64, transient=True, find_new_name=True)

        argmax_state = m.parent.add_state(m.loop.label + "_argfi_reduce")

        # Re-route inbound edges, dropping the pre-loop binds of BOTH carriers (the arg-reduce over
        # the slice subsumes them).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(required(m.idx_carrier_name), None)
            m.parent.add_edge(
                ie.src, argmax_state, dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            )
            m.parent.remove_edge(ie)

        # Bind state: value carrier := val_buf
        lo_is_zero = bool(symbolic.simplify(iter_lo) == 0)
        idx_rhs = idx_buf if lo_is_zero else f"({symbolic.symstr(iter_lo)} + {idx_buf})"
        bind_state = m.parent.add_state(m.loop.label + "_argfi_bind")
        m.parent.add_edge(
            argmax_state,
            bind_state,
            dace.InterstateEdge(assignments={m.carrier_name: val_buf, required(m.idx_carrier_name): idx_rhs}),
        )
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = "max" if m.op == dtypes.ReductionType.Max else "min"
        read = argmax_state.add_read(m.input_array)
        # ``volume`` is stated rather than left to the subset: a subset counts itself as ``ceiling((hi - lo + 1) /
        # step)``, which a symbolic stride (s318's ``inc``) leaves unresolved.
        in_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(pos_lo, pos_hi, coeff)]), volume=n_elems)
        node = ArgReduce(
            name=f"{m.loop.label}_argfi_argreduce",
            op=op,
            transform=m.transform,
            tie="last" if m.last_wins else "first",
        )
        argmax_state.add_node(node)
        argmax_state.add_edge(read, None, node, "_in", in_memlet)
        argmax_state.add_edge(node, "_out_val", wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, "_out_idx", wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))

    # predicate index (TSVC s331)

    def match_predicate_index(
        self, loop: LoopRegion, sdfg: SDFG, skeleton: LoopSkeleton | None = None
    ) -> MatchPredIndex | None:
        """Match ``for i: if pred(a[i]): j = i`` -- a position tracked with NO value carrier, i.e. ``j = max{i : pred}``
        seeded from the pre-loop value of ``j``."""
        skeleton = skeleton if skeleton is not None else self.guarded_loop_skeleton(loop)
        if skeleton is None or symbolic.simplify(required(loop_analysis.get_loop_stride(loop))) != 1:
            return None
        start, end, cond_block, cond_codeblock, true_branch = skeleton

        idx_carrier = self.true_branch_writes_index_only(true_branch, loop.loop_variable)
        if idx_carrier is None or idx_carrier not in sdfg.symbols:
            return None

        # The ConditionalBlock owns no edges of its own, so ``loop.edges()`` is
        # exactly the pre-guard binding chain (``a_index := a[i]``, ``t := a_index < 0``).
        bindings: dict[str, str] = {}
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
        return MatchPredIndex(
            loop=loop,
            parent=loop.parent_graph,
            idx_carrier=idx_carrier,
            seed=seed,
            guard_code=guard_code,
            guard_inputs=guard_inputs,
            iter_start=start,
            iter_end=end,
        )

    def true_branch_writes_index_only(self, true_branch: ControlFlowBlock, loop_var: str) -> str | None:
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

    def wire_guard(
        self, guard_str: str, bindings: dict[str, str], idx_carrier: str, loop: LoopRegion, sdfg: SDFG
    ) -> tuple[str, list[tuple[str, mm.Memlet]]] | None:
        """Turn the branch guard into mask-tasklet code plus the data inputs it reads."""
        try:
            tree = ast.parse(guard_str, mode="eval").body
        except SyntaxError:
            return None
        # One round per binding suffices for an acyclic chain
        for round_index in range(len(bindings) + 1):
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
        connectors = dict.fromkeys(conn for conn, memlet in wiring.reads.values())
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

    def predicate_seed(self, loop: LoopRegion, idx_carrier: str, bindings: dict[str, str], start: Any) -> Any | None:
        """``idx_carrier``'s pre-loop value, iff loop-invariant and provably ``<= start``."""
        parent = loop.parent_graph
        cur: ControlFlowBlock = loop
        rhs: str | None = None
        walked: dict[Any, None] = {}
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
            if not bool(simplified(as_expr(seed) - as_expr(start)) <= 0):
                return None
        except TypeError:
            return None
        return seed

    def rewrite_predicate_index(self, m: MatchPredIndex, sdfg: SDFG) -> None:
        """Replace the loop with ``init(seed) -> Map(masked index, WCR max) -> bind``."""
        priv, _ = sdfg.add_scalar(
            f"_pred_index_{m.loop.label}", sdfg.symbols[m.idx_carrier], transient=True, find_new_name=True
        )
        seed_str = symbolic.symstr(m.seed)
        priv_subset = subsets.Range([(0, 0, 1)])

        init_state = m.parent.add_state(m.loop.label + "_predidx_init")
        seed_tasklet = init_state.add_tasklet("pred_index_seed", {}, dict.fromkeys(["__out"]), f"__out = {seed_str}")
        init_state.add_edge(
            seed_tasklet,
            "__out",
            init_state.add_write(priv),
            None,
            mm.Memlet(data=priv, subset=copy.deepcopy(priv_subset)),
        )

        # The mask's false value IS the seed, so a non-matching iteration folds to the
        # fold's identity and an all-false range leaves the seeded scalar untouched.
        ivar = m.loop.loop_variable
        map_state = m.parent.add_state(m.loop.label + "_predidx")
        map_state.add_mapped_tasklet(
            name="pred_index",
            map_ranges={ivar: subsets.Range([(symbolic.simplify(m.iter_start), symbolic.simplify(m.iter_end), 1)])},
            inputs={conn: copy.deepcopy(memlet) for conn, memlet in m.guard_inputs},
            code=f"__out = ({ivar} if ({m.guard_code}) else ({seed_str}))",
            outputs={"__out": mm.Memlet(data=priv, subset=copy.deepcopy(priv_subset), wcr="lambda a, b: max(a, b)")},
            external_edges=True,
        )

        bind_state = m.parent.add_state(m.loop.label + "_predidx_bind")
        for ie in list(m.parent.in_edges(m.loop)):
            m.parent.add_edge(ie.src, init_state, ie.data)
            m.parent.remove_edge(ie)
        m.parent.add_edge(init_state, map_state, dace.InterstateEdge())
        m.parent.add_edge(map_state, bind_state, dace.InterstateEdge(assignments={m.idx_carrier: priv}))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

    # Partial reductions: a fold into one element, nested to any depth.

    def _unguarded_nest(self, loop: LoopRegion) -> list[LoopRegion] | None:
        """The perfect nest from ``loop``, each level reached by unconditional edges without assignments."""
        nest = [loop]
        while True:
            inner = self._single_child_region(nest[-1], LoopRegion)
            if inner is None:
                return nest
            if any(e.data.assignments or not e.data.is_unconditional() for e in nest[-1].edges()):
                return None
            nest.append(inner)

    @staticmethod
    def _guard_bindings(innermost: LoopRegion, names: list[str]) -> dict[str, str] | None:
        """The unconditional bindings of the innermost body, each name bound once and no loop variable among them."""
        bindings: dict[str, str] = {}
        for e in innermost.edges():
            if not e.data.is_unconditional():
                return None
            for lhs, rhs in (e.data.assignments or {}).items():
                if lhs in bindings or lhs in names:
                    return None
                bindings[lhs] = str(rhs)
        return bindings

    def _box_extents(self, nest: list[LoopRegion], varying: set[str]) -> list[tuple[Any, Any]] | None:
        """The ``(start, end)`` of each unit-stride loop of the nest, when no bound moves with the nest."""
        extents: list[tuple[Any, Any]] = []
        for reduced in nest:
            if reduced.inverted:
                return None  # a do-while runs its body once even over an empty range
            start = loop_analysis.get_init_assignment(reduced)
            end = loop_analysis.get_loop_end(reduced)
            stride = loop_analysis.get_loop_stride(reduced)
            if start is None or end is None or stride is None or symbolic.simplify(stride) != 1:
                return None
            if self._names_in(start, end) & varying:
                return None
            extents.append((start, end))
        return extents

    def match_partial_reduction(self, loop: LoopRegion, sdfg: SDFG) -> PartialReduction | None:
        """``K`` names no reduced loop and every reduced loop moves exactly one dimension of ``I`` by itself, so the
        nest visits a box."""
        nest = self._unguarded_nest(loop)
        if nest is None:
            return None
        names = [n.loop_variable for n in nest]
        if not all(names) or len(set(names)) != len(names):
            return None
        skeleton = self.guarded_loop_skeleton(nest[-1])
        if skeleton is None or symbolic.simplify(required(loop_analysis.get_loop_stride(nest[-1]))) != 1:
            return None
        _, _, _, guard, true_branch = skeleton
        # The binding chain in front of the guard (``x_index := source[...]``, ``acc_index := carrier[K]``).
        bindings = self._guard_bindings(nest[-1], names)
        if bindings is None:
            return None
        varying = set(names) | set(bindings)
        extents = self._box_extents(nest, varying)
        if extents is None:
            return None

        written = self.partial_write(true_branch, sdfg)
        if written is None:
            return None
        carrier, carrier_subset, value, scratch = written
        if any(begin != end_ for begin, end_, _ in carrier_subset.ranges):
            return None
        if self._names_in(*(begin for begin, _, _ in carrier_subset.ranges)) & varying:
            return None

        tree = self.substituted_guard(guard.as_string.strip(), bindings)
        if tree is None:
            return None
        fold = self.partial_fold(tree, value, carrier, carrier_subset, sdfg)
        if fold is None:
            return None
        wcr, gather = fold
        source = gather.value.id  # type: ignore[attr-defined]  -- partial_fold returns a Name-valued subscript
        if source == carrier or source not in sdfg.arrays:
            return None
        if sdfg.arrays[source].dtype != sdfg.arrays[carrier].dtype:
            return None  # the guard compares before the store converts; the reduction would convert first
        if value is not None and not isinstance(value, bool) and not self.same_element(value, gather):
            return None  # the branch stores another element than the one it compared
        source_subset = self.visited_box(gather, names, extents, varying)
        if source_subset is None:
            return None
        if loop_analysis.read_outside(sdfg, nest[0], set(bindings), scratch):
            return None
        return PartialReduction(
            loops=nest,
            parent=loop.parent_graph,
            carrier=carrier,
            carrier_subset=carrier_subset,
            source=source,
            source_subset=source_subset,
            wcr=wcr,
            extents=extents,
        )

    @staticmethod
    def same_element(left: ast.expr, right: ast.expr) -> bool:
        """Whether two subscripts name the same element: one array, and every index equal (``symbolic.equal``)."""
        if not (isinstance(left, ast.Subscript) and isinstance(right, ast.Subscript)):
            return False
        if (
            not (isinstance(left.value, ast.Name) and isinstance(right.value, ast.Name))
            or left.value.id != right.value.id
        ):
            return False
        lidx = left.slice.elts if isinstance(left.slice, ast.Tuple) else [left.slice]
        ridx = right.slice.elts if isinstance(right.slice, ast.Tuple) else [right.slice]
        if len(lidx) != len(ridx):
            return False
        try:
            return all(
                symbolic.equal(
                    symbolic.pystr_to_symbolic(ast.unparse(a)),
                    symbolic.pystr_to_symbolic(ast.unparse(b)),
                    is_length=False,
                )
                is True
                for a, b in zip(lidx, ridx, strict=True)
            )
        except (TypeError, ValueError, SyntaxError):
            return False

    @staticmethod
    def _names_in(*expressions: Any) -> set[str]:
        """The free symbol names of ``expressions``."""
        out: set[str] = set()
        for expr in expressions:
            out |= {str(name) for name in symbolic.free_symbols_and_functions(expr)}
        return out

    def partial_write(
        self, true_branch: ControlFlowBlock, sdfg: SDFG
    ) -> tuple[str, subsets.Range, ast.expr | bool | None, set[str]] | None:
        """``(carrier, element, value, scratch)`` of the one write the true branch makes, or ``None``."""
        if not isinstance(true_branch, ControlFlowRegion):
            return None
        if any(not isinstance(n, SDFGState) for n in true_branch.nodes()):
            return None
        if any(e.data.assignments or not e.data.is_unconditional() for e in true_branch.edges()):
            return None
        state = self._extract_singleton_state(true_branch)
        if state is None:
            return None
        sinks = [n for n in state.sink_nodes() if isinstance(n, nodes.AccessNode)]
        if len(sinks) != 1 or len(state.sink_nodes()) != 1:
            return None
        sink = sinks[0]
        into = state.in_edges(sink)
        if len(into) != 1 or into[0].data.wcr is not None or into[0].data.is_empty():
            return None
        element = into[0].data.get_dst_subset(into[0], state)
        if not isinstance(element, subsets.Range):
            return None
        walked = self._copy_chain(state, sink, into[0].src, sdfg)
        if walked is None:
            return None
        chain, value, scratch = walked
        if len(chain) != len(state.nodes()):
            return None  # something beside the chain: a second effect the rewrite would drop
        return sink.data, element, value, scratch

    @staticmethod
    def _copy_chain(
        state: SDFGState, sink: nodes.AccessNode, node: nodes.Node, sdfg: SDFG
    ) -> tuple[list[nodes.Node], ast.expr | bool, set[str]] | None:
        """Walk back from ``sink`` through identity tasklets and single-element scratch to a constant tasklet
        ``out = True`` / ``False`` (right before the sink) or to a source element; the chain, the value (the constant
        or ``source[I]``) and the scratch containers."""
        chain: list[nodes.Node] = [sink]
        scratch: set[str] = set()
        while True:
            chain.append(node)
            ins = state.in_edges(node)
            if isinstance(node, nodes.Tasklet):
                code = node.code.as_string.strip()
                if node.side_effects or node.code.language != dtypes.Language.Python or len(node.out_connectors) != 1:
                    return None
                out_conn = next(iter(node.out_connectors))
                if not ins:
                    constant = re.fullmatch(rf"{re.escape(out_conn)}\s*=\s*(True|False)", code)
                    if constant is None or len(chain) != 2:
                        return None
                    value: ast.expr | bool | None = constant.group(1) == "True"
                    break
                if len(ins) != 1 or code != f"{out_conn} = {ins[0].dst_conn}":
                    return None
            elif isinstance(node, nodes.AccessNode):
                if not ins:  # the source
                    out = state.out_edges(node)
                    if len(out) != 1:
                        return None
                    indices = out[0].data.get_src_subset(out[0], state)
                    if not isinstance(indices, subsets.Range) or any(b != e for b, e, _ in indices.ranges):
                        return None
                    index = ", ".join(symbolic.symstr(b) for b, _, _ in indices.ranges)
                    value = ast.parse(f"{node.data}[{index}]", mode="eval").body
                    break
                desc = node.desc(sdfg)
                if not desc.transient or desc.total_size != 1 or len(ins) != 1 or len(state.out_edges(node)) != 1:
                    return None
                scratch.add(node.data)
            else:
                return None
            if len(ins) != 1 or ins[0].data.wcr is not None:
                return None
            node = ins[0].src
        return chain, value, scratch

    def substituted_guard(self, guard: str, bindings: dict[str, str]) -> ast.expr | None:
        """``guard`` parsed, every name the binding chain binds replaced by what it binds (one level: a binding
        that names another bound name is refused)."""

        class Substitute(ast.NodeTransformer):
            def __init__(self) -> None:
                self.ok = True

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id not in bindings:
                    return node
                try:
                    bound = ast.parse(bindings[node.id], mode="eval").body
                except SyntaxError:
                    self.ok = False
                    return node
                if any(isinstance(n, ast.Name) and n.id in bindings for n in ast.walk(bound)):
                    self.ok = False
                return bound

        try:
            tree = ast.parse(guard, mode="eval").body
        except SyntaxError:
            return None
        substitute = Substitute()
        tree = substitute.visit(tree)
        return tree if substitute.ok else None

    @staticmethod
    def is_carrier_element(node: ast.expr, carrier: str, carrier_subset: subsets.Range, sdfg: SDFG) -> bool:
        """Whether ``node`` reads ``carrier[K]``, ``K`` the one element of ``carrier_subset``."""
        if isinstance(node, ast.Name):
            return node.id == carrier and sdfg.arrays[carrier].total_size == 1
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == carrier):
            return False
        idx = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(idx) != len(carrier_subset.ranges):
            return False
        try:
            return all(
                symbolic.equal(symbolic.pystr_to_symbolic(ast.unparse(i)), bounds[0], is_length=False) is True
                for i, bounds in zip(idx, carrier_subset.ranges, strict=True)
            )
        except (TypeError, ValueError, SyntaxError):
            return False

    def carrier_compare(
        self, node: ast.expr, carrier: str, carrier_subset: subsets.Range, sdfg: SDFG
    ) -> tuple[CompareOp, ast.Subscript] | None:
        """``(OP, gather)`` of a comparison read as ``gather OP carrier[K]`` (``carrier[K] < gather`` gives ``>``)."""
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1):
            return None
        op, left, right = type(node.ops[0]), node.left, node.comparators[0]
        if op not in CMP_AST_TO_RTYPE:
            return None
        for gather, other, read_op in ((left, right, op), (right, left, MIRRORED_CMP[op])):
            if (
                isinstance(gather, ast.Subscript)
                and isinstance(gather.value, ast.Name)
                and not self.is_carrier_element(gather, carrier, carrier_subset, sdfg)
                and self.is_carrier_element(other, carrier, carrier_subset, sdfg)
            ):
                return read_op, gather
        return None

    def partial_fold(
        self,
        tree: ast.expr,
        value: ast.expr | bool | None,
        carrier: str,
        carrier_subset: subsets.Range,
        sdfg: SDFG,
    ) -> tuple[str, ast.Subscript] | None:
        """``(combine, gather)`` for a guard (with its bindings substituted) and the value the branch writes, or
        ``None``; see :meth:`match_partial_reduction` for the forms."""

        def is_gather(node: ast.expr) -> bool:
            return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and not is_carrier(node)

        def is_carrier(node: ast.expr) -> bool:
            return self.is_carrier_element(node, carrier, carrier_subset, sdfg)

        def compare(node: ast.expr) -> tuple[dtypes.ReductionType, bool, ast.Subscript] | None:
            """``(op, strict, gather)`` of ``gather > carrier`` / ``carrier < gather`` / ... ."""
            parsed = self.carrier_compare(node, carrier, carrier_subset, sdfg)
            if parsed is None:
                return None
            op, gather = parsed
            return CMP_AST_TO_RTYPE[op], op in (ast.Gt, ast.Lt), gather

        if isinstance(value, bool):
            # ``any`` / ``all`` over boolean data: the guard is the element, or its negation.
            negated = isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not)
            gather = tree.operand if negated else tree  # type: ignore[attr-defined]
            if not is_gather(gather) or negated == value:
                return None
            if sdfg.arrays[gather.value.id].dtype != dtypes.bool_ or sdfg.arrays[carrier].dtype != dtypes.bool_:
                return None
            return ("lambda a, b: a and b" if negated else "lambda a, b: a or b"), gather
        if value is None:
            return None
        integral = sdfg.arrays[carrier].dtype.type is not None and np.issubdtype(
            sdfg.arrays[carrier].dtype.type, np.integer
        )
        if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.Or) and len(tree.values) == 2:
            # ``x != x or x > acc``, in either order.
            for nan_test, comparison in (tree.values, tree.values[::-1]):
                parsed = compare(comparison)
                if not (
                    parsed is not None
                    and isinstance(nan_test, ast.Compare)
                    and len(nan_test.ops) == 1
                    and isinstance(nan_test.ops[0], ast.NotEq)
                    and self.same_element(nan_test.left, parsed[2])
                    and self.same_element(nan_test.comparators[0], parsed[2])
                ):
                    continue
                reduction, strict, gather = parsed
                if not strict and not integral:
                    return None
                return NAN_PROPAGATING_WCR[reduction], gather
            return None
        parsed = compare(tree)
        if parsed is None:
            return None
        reduction, strict, gather = parsed
        if not strict and not integral:
            return None
        return STRICT_WCR[reduction], gather

    def visited_box(
        self, gather: ast.Subscript, names: list[str], extents: list[tuple[Any, Any]], varying: set[str]
    ) -> subsets.Range | None:
        """The box of the source ``gather`` visits over the nest, or ``None`` unless every reduced loop moves
        exactly one dimension by itself (``j + c``, ``c`` invariant) and every other dimension stays put."""
        idx = gather.slice.elts if isinstance(gather.slice, ast.Tuple) else [gather.slice]
        bounds = dict(zip(names, extents, strict=True))
        ranges = []
        moved: set[str] = set()
        for index in idx:
            if isinstance(index, ast.Slice):
                return None
            text = ast.unparse(index)
            try:
                expr = symbolic.pystr_to_symbolic(text)
            except (TypeError, ValueError, SyntaxError):
                return None
            used = self._names_in(expr) & varying
            if not used:
                ranges.append((expr, expr, 1))
                continue
            if len(used) != 1 or not used <= set(names) or used & moved:
                return None
            var = used.pop()
            offset = symbolic.pystr_to_symbolic(f"({text}) - ({var})")
            if self._names_in(offset) & varying:
                return None
            start, end = bounds[var]
            ranges.append(
                (
                    symbolic.pystr_to_symbolic(f"({symbolic.symstr(start)}) + ({symbolic.symstr(offset)})"),
                    symbolic.pystr_to_symbolic(f"({symbolic.symstr(end)}) + ({symbolic.symstr(offset)})"),
                    1,
                )
            )
            moved.add(var)
        if moved != set(names):
            return None
        return subsets.Range(ranges)

    def rewrite_partial_reduction(self, m: PartialReduction, sdfg: SDFG) -> None:
        """Replace the nest with one :class:`Reduce` of ``source[box]`` into ``carrier[K]``, ``identity=None``: it folds
        into what ``carrier[K]`` already holds, the seed the nest started from."""
        loop = m.loops[0]
        state = SDFGState(loop.label + "_reduce", sdfg=sdfg)
        nonempty = [
            f"(({symbolic.symstr(end)}) >= ({symbolic.symstr(start)}))"
            for start, end in m.extents
            if not (symbolic.issymbolic(end - start) is False and end - start >= 0)
        ]
        if nonempty:
            branch = ControlFlowRegion(loop.label + "_reduce_body", sdfg=sdfg)
            branch.add_node(state, is_start_block=True)
            block: ControlFlowBlock = ConditionalBlock(loop.label + "_nonempty", sdfg=sdfg)
            block.add_branch(properties.CodeBlock(" and ".join(nonempty)), branch)
        else:
            block = state
        m.parent.add_node(block, is_start_block=m.parent.start_block is loop)
        for edge in list(m.parent.in_edges(loop)):
            m.parent.add_edge(edge.src, block, edge.data)
            m.parent.remove_edge(edge)
        for edge in list(m.parent.out_edges(loop)):
            m.parent.add_edge(block, edge.dst, edge.data)
            m.parent.remove_edge(edge)
        m.parent.remove_node(loop)

        node = Reduce(name=f"{loop.label}_reduce", wcr=m.wcr, axes=None, identity=None)
        node.add_in_connector("_in")
        node.add_out_connector("_out")
        state.add_node(node)
        state.add_edge(state.add_read(m.source), None, node, "_in", mm.Memlet(data=m.source, subset=m.source_subset))
        state.add_edge(
            node, "_out", state.add_write(m.carrier), None, mm.Memlet(data=m.carrier, subset=m.carrier_subset)
        )

    # Masked argmin / argmax into array elements (CloudSC's species sort).

    def match_masked_arg(self, loop: LoopRegion, sdfg: SDFG) -> MaskedArgReduction | None:
        """``for i: [for k:] if mask and source[I] OP carrier[K]: index[J] = i; carrier[K] = source[I]``. ``K`` and
        ``J`` are invariant in ``i``; under a column loop ``k`` each moves with ``k`` in one dimension, so the columns
        are independent and the column loop goes outside."""
        column = self._single_child_region(loop, LoopRegion)
        if column is not None and (
            not column.loop_variable or any(e.data.assignments or not e.data.is_unconditional() for e in loop.edges())
        ):
            return None
        innermost = column if column is not None else loop
        skeleton = self.guarded_loop_skeleton(innermost)
        if skeleton is None:
            return None
        guard, true_branch = skeleton[3], skeleton[4]
        nest = [loop] if column is None else [loop, column]
        names = [n.loop_variable for n in nest]
        if len(set(names)) != len(names):
            return None
        bindings = self._guard_bindings(innermost, names)
        if bindings is None:
            return None
        extents = self._box_extents(nest, set(names) | set(bindings))
        if extents is None:
            return None
        writes = self.arg_writes(true_branch, loop.loop_variable, sdfg)
        if writes is None:
            return None
        carrier, carrier_subset, value, scratch, index, index_subset = writes
        varying = {loop.loop_variable, *bindings}
        for element in (carrier_subset, index_subset):
            if any(rng[0] != rng[1] for rng in element.ranges) or self._names_in(*element.min_element()) & varying:
                return None
            if column is not None and not self.moves_with(element, column.loop_variable, varying):
                return None
        tree = self.substituted_guard(guard.as_string.strip(), bindings)
        parsed = None if tree is None or carrier == index else self.masked_compare(tree, writes, sdfg)
        if parsed is None:
            return None
        op_ast, gather, seed = parsed
        source = gather.value.id  # type: ignore[attr-defined]  -- carrier_compare returns a Name-valued subscript
        if source not in sdfg.arrays or sdfg.arrays[source].dtype != sdfg.arrays[carrier].dtype:
            return None
        if not self.same_element(value, gather):
            return None  # the branch stores another element than the one it compared
        if loop_analysis.read_outside(sdfg, loop, set(bindings), scratch):
            return None
        wiring = GuardReadWiring(sdfg, loop)
        wired_guard = ast.unparse(wiring.visit(copy.deepcopy(tree)))
        wired_gather = ast.unparse(wiring.visit(copy.deepcopy(gather)))
        wired_seed = ast.unparse(wiring.visit(copy.deepcopy(seed)))
        if wiring.refused:
            return None
        return MaskedArgReduction(
            reduced=loop,
            column=column,
            carrier=carrier,
            carrier_subset=carrier_subset,
            index=index,
            index_subset=index_subset,
            guard=wired_guard,
            gather=gather,
            wired_gather=wired_gather,
            wired_seed=wired_seed,
            inputs=dict(wiring.reads.values()),
            op=CMP_AST_TO_RTYPE[op_ast],
            last_wins=self._resolve_last_wins(op_ast, True),
            start=extents[0][0],
            end=extents[0][1],
        )

    def masked_compare(
        self, tree: ast.expr, writes: tuple, sdfg: SDFG
    ) -> tuple[CompareOp, ast.Subscript, ast.expr] | None:
        """``(OP, gather, carrier read)`` of a guard ``mask and ... and gather OP carrier[K]``: the carriers are read
        only by the comparison, and the masks read nothing the loop writes."""
        carrier, carrier_subset, scratch, index = writes[0], writes[1], writes[3], writes[4]
        conjuncts = tree.values if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And) else [tree]
        compares = [c for c in conjuncts if self.carrier_compare(c, carrier, carrier_subset, sdfg) is not None]
        if len(compares) != 1:
            return None
        compare = compares[0]
        op_ast, gather = required(self.carrier_compare(compare, carrier, carrier_subset, sdfg))
        read = {n.id for c in conjuncts if c is not compare for n in ast.walk(c) if isinstance(n, ast.Name)}
        read |= {n.id for n in ast.walk(gather) if isinstance(n, ast.Name)}
        if read & {carrier, index, *scratch}:
            return None
        return op_ast, gather, compare.comparators[0] if compare.left is gather else compare.left  # type: ignore[attr-defined]

    def moves_with(self, element: subsets.Range, var: str, blocked: set[str]) -> bool:
        """Whether one dimension of ``element`` is ``var + c``, ``c`` free of ``var`` and ``blocked``."""
        for begin in element.min_element():
            offset = symbolic.pystr_to_symbolic(f"({symbolic.symstr(begin)}) - ({var})")
            if var in self._names_in(begin) and not self._names_in(offset) & (blocked | {var}):
                return True
        return False

    def arg_writes(
        self, true_branch: ControlFlowBlock, var: str, sdfg: SDFG
    ) -> tuple[str, subsets.Range, ast.expr, set[str], str, subsets.Range] | None:
        """``(carrier, K, source[I], scratch, index, J)`` of a true branch that does exactly ``index[J] = var`` and
        ``carrier[K] = source[I]``, or ``None``."""
        if any(e.data.assignments or not e.data.is_unconditional() for e in true_branch.all_interstate_edges()):
            return None
        state = self._extract_singleton_state(true_branch)
        if state is None or len(state.sink_nodes()) != 2:
            return None
        value_write, index_write, covered = None, None, 0
        for sink in state.sink_nodes():
            into = state.in_edges(sink)
            if not isinstance(sink, nodes.AccessNode) or len(into) != 1 or into[0].data.wcr is not None:
                return None
            element = into[0].data.get_dst_subset(into[0], state)
            src = into[0].src
            if not isinstance(element, subsets.Range):
                return None
            if (
                isinstance(src, nodes.Tasklet)
                and src.code.language == dtypes.Language.Python
                and not src.side_effects
                and not state.in_edges(src)
                and state.out_degree(src) == 1
                and re.fullmatch(rf"{re.escape(into[0].src_conn)}\s*=\s*{re.escape(var)}", src.code.as_string.strip())
            ):
                index_write = (sink.data, element)
                covered += 2
                continue
            walked = self._copy_chain(state, sink, src, sdfg)
            if walked is None or isinstance(walked[1], bool) or value_write is not None:
                return None
            chain, value, scratch = walked
            value_write = (sink.data, element, value, scratch)
            covered += len(chain)
        if value_write is None or index_write is None or covered != len(state.nodes()):
            return None
        return *value_write, *index_write

    def rewrite_masked_arg(self, m: MaskedArgReduction, sdfg: SDFG) -> None:
        """Replace the reduced loop with ``best = carrier[K]``; a map folding every passing ``source[I]`` into ``best``;
        a map folding into ``pick`` the first (last) passing ``i`` whose element equals ``best``; and, when one
        passed, ``index[J] = pick; carrier[K] = source[I(pick)]``: the very element the loop would have kept."""
        loop, var = m.reduced, m.reduced.loop_variable
        parent = loop.parent_graph
        best = sdfg.add_scalar(
            f"_argbest_{loop.label}", sdfg.arrays[m.carrier].dtype, transient=True, find_new_name=True
        )[0]
        pick = sdfg.add_scalar(
            f"_argpick_{loop.label}", sdfg.arrays[m.index].dtype, transient=True, find_new_name=True
        )[0]
        sentinel = symbolic.symstr(m.start - 1 if m.last_wins else m.end + 1)
        point = subsets.Range([(0, 0, 1)])

        host = parent if m.column is None else m.column
        for block in list(host.nodes()) if m.column is not None else []:
            host.remove_node(block)
        init = host.add_state(
            f"{loop.label}_arg_init", is_start_block=m.column is not None or parent.start_block is loop
        )
        init.add_edge(
            init.add_read(m.carrier),
            None,
            init.add_write(best),
            None,
            mm.Memlet(data=m.carrier, subset=copy.deepcopy(m.carrier_subset), other_subset=copy.deepcopy(point)),
        )
        seed = init.add_tasklet("arg_pick_seed", {}, dict.fromkeys(["__out"]), f"__out = {sentinel}")
        init.add_edge(seed, "__out", init.add_write(pick), None, mm.Memlet(data=pick, subset=copy.deepcopy(point)))

        # Each map evaluates the loop's own guard, short-circuit included: the reads are dynamic, taken on use.
        guard, gather = m.guard, m.wired_gather
        inputs = {conn: copy.deepcopy(memlet) for conn, memlet in m.inputs.items()}
        for memlet in inputs.values():
            memlet.dynamic = True
        ranges = {var: subsets.Range([(m.start, m.end, 1)])}
        fold = "min" if m.op == dtypes.ReductionType.Min else "max"
        value_state = host.add_state(f"{loop.label}_arg_value")
        value_state.add_mapped_tasklet(
            "arg_value",
            ranges,
            copy.deepcopy(inputs),
            f"__out = ({gather} if ({guard}) else {m.wired_seed})",
            {"__out": mm.Memlet(data=best, subset=copy.deepcopy(point), wcr=STRICT_WCR[m.op])},
            external_edges=True,
        )
        index_state = host.add_state(f"{loop.label}_arg_index")
        index_state.add_mapped_tasklet(
            "arg_index",
            ranges,
            {**copy.deepcopy(inputs), "__best": mm.Memlet(data=best, subset=copy.deepcopy(point))},
            f"__out = ({var} if (({guard}) and {gather} == __best) else {sentinel})",
            {
                "__out": mm.Memlet(
                    data=pick, subset=copy.deepcopy(point), wcr=f"lambda a, b: {'max' if m.last_wins else 'min'}(a, b)"
                )
            },
            external_edges=True,
        )

        source = m.gather.value.id  # type: ignore[attr-defined]
        kept = astutils.ASTFindReplace({var: "__pick"}).visit(copy.deepcopy(m.gather.slice))
        bind_state = host.add_state(f"{loop.label}_arg_bind")
        bind = bind_state.add_tasklet(
            "arg_bind",
            dict.fromkeys(["__pick", "__src"]),
            dict.fromkeys(["__index", "__value"]),
            f"if __pick != {sentinel}:\n    __index = __pick\n    __value = __src[{ast.unparse(kept)}]",
        )
        bind_state.add_edge(
            bind_state.add_read(pick), None, bind, "__pick", mm.Memlet(data=pick, subset=copy.deepcopy(point))
        )
        src_memlet = mm.Memlet.from_array(source, sdfg.arrays[source])
        src_memlet.dynamic = True
        bind_state.add_edge(bind_state.add_read(source), None, bind, "__src", src_memlet)
        for conn, name, element in (("__index", m.index, m.index_subset), ("__value", m.carrier, m.carrier_subset)):
            bind_state.add_edge(
                bind,
                conn,
                bind_state.add_write(name),
                None,
                mm.Memlet(data=name, subset=copy.deepcopy(element), dynamic=True),
            )
        host.add_edge(init, value_state, dace.InterstateEdge())
        host.add_edge(value_state, index_state, dace.InterstateEdge())
        host.add_edge(index_state, bind_state, dace.InterstateEdge())

        # The column loop, its body now the reduction, takes the reduced loop's place.
        first, last = (init, bind_state) if m.column is None else (m.column, m.column)
        if m.column is not None:
            parent.add_node(m.column, is_start_block=parent.start_block is loop)
        for edge in list(parent.in_edges(loop)):
            parent.add_edge(edge.src, first, edge.data)
            parent.remove_edge(edge)
        for edge in list(parent.out_edges(loop)):
            parent.add_edge(last, edge.dst, edge.data)
            parent.remove_edge(edge)
        if m.column is not None:
            loop.remove_node(m.column)
        parent.remove_node(loop)


__all__ = ["ArgMaxLift"]
