# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Detect scalar-accumulator loops -> ``Reduce`` nodes.

Emitted ``Reduce`` uses ``identity=None`` so the pre-loop accumulator seeds the fold.

Three shapes:

- **Tasklet**: single state, one two-input tasklet writing the accumulator.
- **Interstate edge**: 2 empty states + one iedge assigning ``{sym: sym <op> arr[<f(i)>]}``.
- **Conditional interstate edge**: single ``ConditionalBlock``, one branch guarded by
  ``sym <cmp> arr[<f(i)>]`` (``cmp`` in ``>``/``>=``/``<``/``<=``); branch = the 2-empty-states
  shape assigning ``{sym: arr[<f(i)>]}``. ``>``/``>=`` -> ``max``, ``<``/``<=`` -> ``min``.

Accumulator: ``Scalar``, length-1 ``Array``, or one loop-invariant slice of an ``Array`` (``C[k]``).
"""
import ast
import copy
from typing import Dict, NamedTuple, Optional

import sympy

from dace import SDFG, SDFGState, data, dtypes, memlet as mm, nodes, properties, subsets, symbolic
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.symbolic import AND, OR, bitwise_and, bitwise_or, Subscript
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

# Ops here are commutative by construction -> skip is_op_commutative (returns None
# for max/min: Python builtins choke on symbolic args).
_BINOP_TO_WCR: Dict[type, str] = {
    ast.Add: "lambda a, b: a + b",
    ast.Mult: "lambda a, b: a * b",
    ast.BitAnd: "lambda a, b: a & b",
    ast.BitOr: "lambda a, b: a | b",
    ast.BitXor: "lambda a, b: a ^ b",
}
_BOOLOP_TO_WCR: Dict[type, str] = {
    ast.Or: "lambda a, b: a | b",
    ast.And: "lambda a, b: a & b",
}
_CALL_TO_WCR: Dict[str, str] = {
    "max": "lambda a, b: max(a, b)",
    "min": "lambda a, b: min(a, b)",
}
# Guard `lhs <cmp> rhs` with body `sym = arr[i]`: reduction = max iff guard fires when arr > sym.
_CMP_GT = (ast.Gt, ast.GtE)
_CMP_LT = (ast.Lt, ast.LtE)


class _Reduction(NamedTuple):
    wcr: str
    accum: str  # data-descriptor name, or DaCe symbol
    accum_subset: subsets.Subset
    array: str
    array_subset: subsets.Subset


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToReduce(ppl.Pass):
    """Lift scalar-accumulator loops to Reduction library nodes."""

    permissive = properties.Property(
        dtype=bool,
        default=False,
        desc="Enable extractors that make semantic assumptions about input "
        "data (e.g. the ``any``/``all`` conditional-const-assign pattern "
        "which assumes the guard array is 0/1-valued).",
    )

    prefer = properties.Property(
        dtype=str,
        default='reduce-libnode',
        choices=('reduce-libnode', 'wcr-scalar'),
        desc="Emission strategy. ``reduce-libnode`` (default) emits a single "
        "``Reduce`` library node; ``wcr-scalar`` keeps a LoopRegion that "
        "accumulates into a transient scalar with WCR, plus init + writeback "
        "states. The wcr-scalar form lets downstream ``LoopToMap`` parallelize "
        "the loop into a Map+WCR-on-scalar shape that the WCR codegen lowers "
        "to ``#pragma omp parallel for reduction(op:scalar)`` -- the right "
        "lowering for accumulator ops the ``Reduce`` libnode cannot express.",
    )

    def __init__(self, permissive: bool = False, prefer: str = 'reduce-libnode'):
        super().__init__()
        self.permissive = permissive
        self.prefer = prefer

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        # WCR edges -> in-body augassign so the matcher sees a uniform
        # ``acc <op>= arr[f(i)]`` tasklet. No-op if already augassign.
        from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})

        # D4 (CleanAccessNode + CleanTasklet) deliberately NOT applied: matcher already
        # handles scalar-slice intermediates, WCRToAugAssign above covers normalization.
        # Redundant + risks the ordering pitfalls seen in LoopToScan.

        count = 0
        for node, parent in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, LoopRegion):
                continue
            info = _extract(node, sdfg, permissive=self.permissive)
            if info is None:
                continue
            if self.prefer == 'wcr-scalar':
                _lift_wcr_scalar(parent, node, info)
            else:
                _lift(parent, node, info)
            count += 1

        # Multi-tasklet ``compute then accumulate`` shapes (TSVC s313/vdotr
        # ``dot[0] += a[i]*b[i]``, s4115 gather+sum) don't match ``_extract``: the
        # single-tasklet matcher refuses multi-tasklet bodies by design (relaxing it
        # would silently lift GEMM contractions to a ``Reduce`` libnode, bypassing
        # ``LiftEinsum`` BLAS lowering). wcr-scalar mode only: run ``TTE + AugAssignToWCR``
        # to collapse the in-body copy chain to a clean ``WCR-on-accum[c]`` write the
        # retarget lift can privatise. The ``Reduce`` libnode can't express the in-body
        # compute chain -> gated on ``prefer == 'wcr-scalar'``.
        if self.prefer == 'wcr-scalar':
            from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
            from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
            # TTE collapses the frontend's trivial ``out = in`` passthrough tasklets
            # around the accumulator load/store so ``AugAssignToWCR`` (matches the 5-node
            # ``arr -> copy_in -> tasklet -> copy_out -> arr`` shape) sees a clean pattern.
            PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
            # ``permissive=False`` required: permissive mode matches scan-shape bodies
            # (TSVC recurrence_down ``b[i] = b[i+1] + a[i]`` after ``LoopToScan``) as
            # reductions and rewrites them to WCR writes later parallelised -> carried
            # dependence lost, off-by-one. Pinned by the descending-recurrence value-
            # preservation test.
            sdfg.apply_transformations_repeated(AugAssignToWCR, validate=False, validate_all=False, permissive=False)
            for node, parent in list(sdfg.all_nodes_recursive()):
                if not isinstance(node, LoopRegion):
                    continue
                wcr_info = _extract_wcr_body(node, sdfg)
                if wcr_info is None:
                    continue
                _lift_wcr_scalar_retarget(parent, node, *wcr_info)
                count += 1

            # Dedicated matcher for the gather+sum shape (TSVC s4115
            # ``s += a[i] * b[ip[i]]``): transient scalar accumulator, body split into
            # pre-load + compute states joined by a gather-index iedge, final write
            # through an extra transient AccessNode -> past every ``AugAssignToWCR`` shape.
            # In place: drop the tasklet's carry input, add WCR to the final write, wrap
            # with init + writeback. Same gating as ``_extract_wcr_body`` (wcr-scalar only).
            for node, parent in list(sdfg.all_nodes_recursive()):
                if not isinstance(node, LoopRegion):
                    continue
                chain_info = _extract_multi_state_chain(node, sdfg)
                if chain_info is None:
                    continue
                _lift_multi_state_chain(parent, node, chain_info)
                count += 1

        if count > 0:
            # Narrow memlets on the new ``Reduce`` libnode + read/write edges: lifting
            # uses array-extent memlets, propagation collapses to the reduction-axis
            # range so codegen / DCE see the tight subset.
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
        return count or None


def _one_elem(subset) -> Optional[int]:
    """Integer number of elements in ``subset``, or ``None`` if non-constant."""
    if subset is None:
        return None
    try:
        s = symbolic.simplify(subset.num_elements())
    except Exception:
        return None
    return int(s) if s.is_Integer else None


def _uses(subset: subsets.Subset, sym: sympy.Symbol) -> bool:
    return subset is not None and any(symbolic.pystr_to_symbolic(str(e)) == sym for e in subset.free_symbols)


def _scalar_equiv(sdfg: SDFG, a: str, b: str) -> bool:
    """Same descriptor, or two distinct dtype-compatible scalar-equivalents."""
    if a == b:
        return True
    da, db = sdfg.arrays.get(a), sdfg.arrays.get(b)
    if da is None or db is None or da.dtype != db.dtype:
        return False

    def scalar_like(d) -> bool:
        return isinstance(d, data.Scalar) or (isinstance(d, data.Array) and all(s == 1 for s in d.shape))

    return scalar_like(da) and scalar_like(db)


def _chase_forward_to_accum(state, sdfg: SDFG, start_node, start_subset):
    """Walk a copy chain forward from ``start_node`` to its non-transient destination.

    Hop only across a transient AccessNode with exactly 1 in-edge + 1 out-edge whose
    memlet preserves the single-element write scope. Halts at non-transient, branch, or
    scope break. Returns ``(name, subset)`` of the final node.

    Lets ``_extract`` see the frontend's ``compute -> tmp -> assign-copy -> accumulator``
    staging: without it ``_scalar_equiv`` rejects an accumulator like ``sum_out: float64[N]``
    written at ``[0]`` (descriptor not scalar-like though the access is).
    """
    cur, cur_sub = start_node, start_subset
    visited = set()
    while True:
        if id(cur) in visited:
            return cur.data, cur_sub
        visited.add(id(cur))
        desc = sdfg.arrays.get(cur.data)
        if desc is None or not desc.transient:
            return cur.data, cur_sub
        out_edges = state.out_edges(cur)
        in_edges = state.in_edges(cur)
        if len(out_edges) != 1 or len(in_edges) != 1:
            return cur.data, cur_sub
        oe = out_edges[0]
        if (not isinstance(oe.dst, nodes.AccessNode) or oe.data is None or oe.data.subset is None
                or _one_elem(oe.data.subset) != _one_elem(cur_sub)):
            return cur.data, cur_sub
        cur, cur_sub = oe.dst, oe.data.subset


def _expand_over_loop(subset: subsets.Subset,
                      loop_var: sympy.Symbol,
                      start,
                      end,
                      loop_stride=1) -> Optional[subsets.Range]:
    """Widen ``subset`` dims that use ``loop_var`` linearly over ``[start, end]`` (inclusive)
    stepped by ``loop_stride``. Dims not involving ``loop_var`` (e.g. outer ``jl`` in a
    per-row inner reduction ``arr[jl, jm]`` over ``jm``) stay as-is; only the reduction axis widens.

    Read index ``coeff*loop_var + off`` over ``[start, end : loop_stride]`` -> arithmetic
    sequence first ``coeff*start + off``, last ``coeff*end + off``, step ``coeff*loop_stride``.
    So a strided read (``a[2*k]``) and a strided loop (``range(0, N, 2)``) both fold into one
    strided reduce subset. Only ascending integer read coeff; symbolic/zero/negative refused
    (left to LoopToMap)."""
    if not isinstance(subset, subsets.Range):
        return None
    ranges = []
    for rb, re_, rs in subset.ndrange():
        if rb != re_ or rs != 1:
            return None
        rb_sym = symbolic.pystr_to_symbolic(str(rb))
        if loop_var not in rb_sym.free_symbols:
            ranges.append((rb, re_, 1))  # axis independent of reduction
            continue
        # read index = coeff*loop_var + off (affine)
        coeff = symbolic.simplify(sympy.diff(rb_sym, loop_var))
        off = symbolic.simplify(rb_sym - coeff * loop_var)
        if coeff.has(loop_var) or off.has(loop_var):
            return None  # non-linear
        if not (coeff.is_integer and coeff.is_positive):
            return None  # only ascending integer read stride
        step = symbolic.simplify(coeff * loop_stride)
        ranges.append((symbolic.simplify(coeff * start + off), symbolic.simplify(coeff * end + off), step))
    return subsets.Range(ranges)


def _cmp_to_wcr(cond, target: str, array: str) -> Optional[str]:
    """Map a ``sym <cmp> arr[...]`` (or reversed) guard to a max/min WCR."""
    try:
        tree = ast.parse(cond.as_string, mode="eval").body
    except (SyntaxError, TypeError, ValueError):
        return None
    if not isinstance(tree, ast.Compare) or len(tree.ops) != 1:
        return None
    op_type = type(tree.ops[0])
    if op_type not in _CMP_GT and op_type not in _CMP_LT:
        return None

    def _is_target(n):
        return isinstance(n, ast.Name) and n.id == target

    def _is_array(n):
        return (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == array)

    left, right = tree.left, tree.comparators[0]
    if _is_target(left) and _is_array(right):
        array_on_left = False
    elif _is_array(left) and _is_target(right):
        array_on_left = True
    else:
        return None
    is_gt = op_type in _CMP_GT
    arr_is_larger = array_on_left == is_gt
    return "lambda a, b: max(a, b)" if arr_is_larger else "lambda a, b: min(a, b)"


def _extract_any_pattern(cond, const_rhs: int, target: str, sdfg: SDFG, loop_var_sym, start,
                         end) -> Optional["_Reduction"]:
    """Match ``{sym: const}`` conditional-iedge "any"/"all".

    Body = ``ConditionalBlock``, one branch, guard ``arr[<subs>] <cmp> C`` (C int),
    branch = 2 empty states + iedge ``{sym: <const_rhs>}``, ``const_rhs`` in {0, 1}.

    Guard array assumed 0/1-valued -> ``any(arr[...] == 1)`` = bitwise-OR of ``arr[...]``,
    so plain ``Reduce(|)`` suffices. ``const_rhs == 1`` -> OR; ``const_rhs == 0`` -> AND.
    """
    if const_rhs == 1:
        wcr = "lambda a, b: a | b"
    elif const_rhs == 0:
        wcr = "lambda a, b: a & b"
    else:
        return None

    try:
        tree = ast.parse(cond.as_string, mode="eval").body
    except (SyntaxError, TypeError, ValueError):
        return None
    if not isinstance(tree, ast.Compare) or len(tree.ops) != 1:
        return None
    left, right = tree.left, tree.comparators[0]

    def _is_subscript_on_array(n):
        return (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in sdfg.arrays)

    def _is_int_const(n):
        return isinstance(n, ast.Constant) and isinstance(n.value, int)

    if _is_subscript_on_array(left) and _is_int_const(right):
        sub = left
    elif _is_subscript_on_array(right) and _is_int_const(left):
        sub = right
    else:
        return None

    array = sub.value.id
    slice_node = sub.slice
    args_ast = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]

    try:
        sym_args = [symbolic.pystr_to_symbolic(ast.unparse(a)) for a in args_ast]
    except Exception:
        return None

    if len(sym_args) != len(sdfg.arrays[array].shape):
        return None

    # Exactly one axis must depend on the loop variable (linearly, offset ∉ sym).
    axis_for_iter = None
    offset = None
    for i, a in enumerate(sym_args):
        if a.has(loop_var_sym):
            if axis_for_iter is not None:
                return None
            axis_for_iter = i
            try:
                off = symbolic.simplify(a - loop_var_sym)
            except Exception:
                return None
            if off.has(loop_var_sym):
                return None
            offset = off
    if axis_for_iter is None:
        return None

    ranges = []
    for i, a in enumerate(sym_args):
        if i == axis_for_iter:
            ranges.append((symbolic.simplify(start + offset), symbolic.simplify(end + offset), 1))
        else:
            ranges.append((a, a, 1))
    return _Reduction(
        wcr=wcr,
        accum=target,
        accum_subset=subsets.Range([(0, 0, 1)]),
        array=array,
        array_subset=subsets.Range(ranges),
    )


def _extract(loop: LoopRegion, sdfg: SDFG, permissive: bool = False) -> Optional[_Reduction]:
    if not loop.loop_variable:
        return None
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    # A strided loop (``range(0, N, 2)``) folds its stride into the reduce subset
    # (see ``_expand_over_loop``); only an ascending integer loop stride is handled.
    if stride != 1:
        try:
            if int(stride) <= 0:
                return None
        except (TypeError, ValueError):
            return None

    blocks = loop.nodes()
    loop_var = loop.loop_variable
    loop_var_sym = symbolic.pystr_to_symbolic(loop_var)

    # Tasklet pattern: single state with exactly one tasklet.
    #
    # The single-tasklet + EXACTLY-2-data-inputs constraints also guarantee this pass
    # does NOT lift GEMM/tensor contractions: ``for k: c[i,j] += a[i,k] * b[k,j]`` gives
    # either 2 tasklets (Mul then Add) or 1 tasklet with 3 data inputs (acc + 2 gathers),
    # both refused -> contraction goes to ``LoopToMap`` + WCR, the canonical input for
    # :class:`~dace.transformation.dataflow.lift_einsum.LiftEinsum` (``ik,kj->ij``).
    # Relaxing the 2-input constraint without contraction-detection would silently turn
    # matmuls into ``Reduce`` libnodes, bypassing that BLAS path.
    if len(blocks) == 1 and isinstance(blocks[0], SDFGState):
        state = blocks[0]
        tasklet = None
        for n in state.nodes():
            if isinstance(n, nodes.Tasklet):
                if tasklet is not None:
                    return None
                tasklet = n
            elif not isinstance(n, nodes.AccessNode):
                return None
        if tasklet is None:
            return None
        if tasklet.code.language != dtypes.Language.Python:
            return None

        # Classify the tasklet's single-assignment body.
        try:
            tree = ast.parse((tasklet.code.as_string or "").strip())
        except SyntaxError:
            return None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return None
        rhs = tree.body[0].value
        if isinstance(rhs, ast.BinOp):
            wcr = _BINOP_TO_WCR.get(type(rhs.op))
        elif isinstance(rhs, ast.BoolOp) and len(rhs.values) == 2:
            wcr = _BOOLOP_TO_WCR.get(type(rhs.op))
        elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and len(rhs.args) == 2):
            wcr = _CALL_TO_WCR.get(rhs.func.id)
        else:
            wcr = None
        if wcr is None:
            return None

        # Tasklet must have exactly 2 data inputs and 1 data output.
        def _has_data(e):
            return e.data is not None and not e.data.is_empty()

        in_edges = [e for e in state.in_edges(tasklet) if _has_data(e)]
        out_edges = [e for e in state.out_edges(tasklet) if _has_data(e)]
        if len(in_edges) != 2 or len(out_edges) != 1:
            return None

        write_edge = out_edges[0]
        if not isinstance(write_edge.dst, nodes.AccessNode):
            return None
        accum = write_edge.dst.data
        if accum not in sdfg.arrays:
            return None
        write_subset = write_edge.data.subset
        if _one_elem(write_subset) != 1 or _uses(write_subset, loop_var_sym):
            return None
        # Refuse when the write subset uses a symbol REASSIGNED on a loop iedge
        # (e.g. ``k = k + j + 1``): value changes each iteration -> N independent
        # writes, not a single-slot reduction. Pinned by TSVC s141
        # (``flat_2d_array[k] = flat_2d_array[k] + bb[j, i]``, ``k`` incremented per iter).
        loop_iedge_assignees = set()
        for e in loop.edges():
            loop_iedge_assignees.update(e.data.assignments.keys())
        for free_sym in write_subset.free_symbols:
            if str(free_sym) in loop_iedge_assignees:
                return None
        # Chase forward through the copy chain (``compute -> tmp -> assign-copy -> accumulator``)
        # so the carry-input check can match an accumulator whose descriptor is not
        # scalar-like (``sum_out: float64[N]`` written at ``[0]``) but whose access is.
        final_accum, final_subset = _chase_forward_to_accum(state, sdfg, write_edge.dst, write_subset)

        # Resolve each tasklet input.
        resolved = []
        for e in in_edges:
            src = e.src
            if not isinstance(src, nodes.AccessNode):
                return None
            desc = sdfg.arrays.get(src.data)
            if desc is None:
                return None
            if desc.transient and len(state.in_edges(src)) == 1 and len(state.out_edges(src)) == 1:
                pred = state.in_edges(src)[0]
                if (not isinstance(pred.src, nodes.AccessNode) or pred.data is None or pred.data.subset is None
                        or _one_elem(e.data.subset) != _one_elem(pred.data.subset)):
                    return None
                resolved.append((pred.src.data, copy.deepcopy(pred.data.subset)))
            else:
                resolved.append((src.data, e.data.subset))

        # Accumulator input must be loop-carried: resolved data is written inside the loop,
        # so its value flows from a prior iteration. This makes ``acc = acc op x`` a reduction
        # and distinguishes a genuine carried accumulator (maybe via a scalar-equivalent staging
        # copy) from a loop-invariant scalar that merely happens to be scalar-equivalent to the
        # write target -- e.g. scaled scatter ``out_slice = arr[jl, jk] * zq`` (``zq = 1/ptsphy``
        # hoisted out), NOT a reduction.
        carried = {an.data for st in loop.all_states() for an in st.data_nodes() if st.in_degree(an) > 0}
        accum_ok = False
        array, arr_subset = None, None
        carried_accum, carried_sub = None, None
        for name, sub in resolved:
            if _uses(sub, loop_var_sym):
                if array is not None:
                    return None
                array, arr_subset = name, sub
            elif (_one_elem(sub) == 1 and name in carried
                  and ((name == accum and sub == write_subset) or (name == final_accum and sub == final_subset) or
                       (name != accum and _scalar_equiv(sdfg, name, accum)))):
                accum_ok = True
                carried_accum, carried_sub = name, sub
        if not accum_ok or array is None or array == accum:
            return None

        # The carried accumulator (survives across iterations, read downstream) may differ
        # from the tasklet's write target when the frontend stages via a temp
        # (``tmp = acc + a[i]; acc = tmp``). Reduce into the carried accumulator, not the temp:
        # it holds the pre-loop seed so the seeded ``Reduce`` (identity=None) folds correctly,
        # and dropping the temp is safe (it doesn't outlive the body).
        if carried_accum is not None and carried_accum != accum:
            accum, write_subset = carried_accum, carried_sub

        # A pure reduction writes ONLY the accumulator (+ loop-local staging temps). If the
        # body also writes a non-transient container -- e.g. per-iteration scan output
        # ``b[i] = sum`` in ``sum = sum + a[i]; b[i] = sum`` -- the running accumulator is
        # observed every iteration; collapsing to a single ``Reduce`` drops it. Refuse.
        written = {an.data for st in loop.all_states() for an in st.data_nodes() if st.in_degree(an) > 0}
        allowed = {accum, array} | ({carried_accum} if carried_accum is not None else set())
        if any(w not in allowed and not sdfg.arrays[w].transient for w in written):
            return None

        expanded = _expand_over_loop(arr_subset, loop_var_sym, start, end, stride)
        if expanded is None:
            return None
        return _Reduction(wcr, accum, write_subset, array, expanded)

    # Interstate-edge pattern: 2 empty states + 1 edge with 1 assignment, at loop level
    # or inside a single-branch ConditionalBlock guarded by a >/>=/</<= accumulator-vs-array cmp.
    cond = None
    body: ControlFlowRegion = loop
    if len(blocks) == 1 and isinstance(blocks[0], ConditionalBlock):
        cb = blocks[0]
        if len(cb.branches) != 1:
            return None
        cond, body = cb.branches[0]
        if cond is None:
            return None
        blocks = body.nodes()

    if len(blocks) == 2 and all(isinstance(b, SDFGState) for b in blocks):
        s1, s2 = blocks
        if s1.nodes() or s2.nodes():
            return None
        edges = body.edges()
        if len(edges) != 1:
            return None
        (edge, ) = edges
        if {edge.src, edge.dst} != {s1, s2}:
            return None
        assignments = edge.data.assignments or {}
        if len(assignments) != 1:
            return None
        ((target, expr_str), ) = assignments.items()
        # Interstate-edge assignment targets are always DaCe symbols.
        if target not in sdfg.symbols:
            return None

        try:
            expr = symbolic.pystr_to_symbolic(expr_str)
        except Exception:
            return None

        # ``pystr_to_symbolic`` renders ``B[i]`` as ``Subscript(B, i)`` (head, indices).
        if cond is None:
            # Top-level op must be a 2-arg commutative reduction.
            if isinstance(expr, sympy.Add) and len(expr.args) == 2:
                wcr = "lambda a, b: a + b"
            elif isinstance(expr, sympy.Mul) and len(expr.args) == 2:
                wcr = "lambda a, b: a * b"
            elif isinstance(expr, (OR, bitwise_or)) and len(expr.args) == 2:
                wcr = "lambda a, b: a | b"
            elif isinstance(expr, (AND, bitwise_and)) and len(expr.args) == 2:
                wcr = "lambda a, b: a & b"
            else:
                return None

            target_sym = symbolic.pystr_to_symbolic(target)
            arr_call = None
            other = None
            for arg in expr.args:
                if isinstance(arg, Subscript) and symbolic.arrays(arg) & sdfg.arrays.keys():
                    if arr_call is not None:
                        return None
                    arr_call = arg
                else:
                    other = arg
            if arr_call is None or other != target_sym:
                return None
        else:
            # Conditional-iedge path.
            # "any"/"all": ``{sym: <const>}`` with array-predicate guard -> OR/AND over the
            # (0/1-valued) guard array. Gated on ``permissive``: only sound if the guard
            # array holds only 0/1, which the pass can't verify statically.
            if permissive and isinstance(expr, sympy.Integer) and int(expr) in (0, 1):
                return _extract_any_pattern(cond, int(expr), target, sdfg, loop_var_sym, start, end)
            # Pure copy ``sym = arr[f(i)]`` gated by a max/min comparison.
            if not (isinstance(expr, Subscript) and symbolic.arrays(expr) & sdfg.arrays.keys()):
                return None
            arr_call = expr
            wcr = _cmp_to_wcr(cond, target, str(arr_call.args[0]))
            if wcr is None:
                return None

        array = str(arr_call.args[0])
        # ``Subscript`` carries the head plus indices, so a 1-D access has two args.
        if len(sdfg.arrays[array].shape) != 1 or len(arr_call.args) != 2:
            return None
        offset = symbolic.simplify(arr_call.args[1] - loop_var_sym)
        if offset.has(loop_var_sym):
            return None

        return _Reduction(
            wcr=wcr,
            accum=target,
            accum_subset=subsets.Range([(0, 0, 1)]),
            array=array,
            array_subset=subsets.Range([(symbolic.simplify(start + offset), symbolic.simplify(end + offset), 1)]),
        )

    # Branched min/max pattern (TSVC s314, s316): ``for i: if a[i] > x: x = a[i]``.
    # Frontend lowers to 3 blocks ``[ConditionalBlock, cond_prep_state, post_state]``
    # threading the cmp through a temp symbol:
    #   (block) -> (cond_prep) {arr_sym: arr[i]}
    #   (cond_prep) -> (if_N) {guard_sym: arr_sym <cmp> accum}
    #   if_N TRUE branch body: ``accum = arr[i]`` (passthrough copy)
    # max/min idempotent -> the conditional is redundant at wcr level
    # (``acc = max(acc, arr[i])`` correct whether or not the guard fires). Both lifts
    # consume the resulting ``_Reduction`` as-is.
    info = _extract_branched_minmax(loop, sdfg, loop_var_sym, start, end)
    if info is not None:
        return info

    return None


def _extract_branched_minmax(loop: LoopRegion, sdfg: SDFG, loop_var_sym: sympy.Symbol, start,
                             end) -> Optional[_Reduction]:
    """Match ``for i: if arr[i] <cmp> accum: accum = arr[i]`` where the frontend lowers
    the masked update into:

    - one ``ConditionalBlock`` (single TRUE branch guarded by a temp symbol),
    - one ``cond_prep`` SDFGState hubbing the guard-computing iedge,
    - one trailing empty SDFGState,

    threaded by iedges ``{arr_sym: arr[i]}`` and ``{guard_sym: arr_sym <cmp> accum}``.

    Returns ``_Reduction`` (``wcr`` = ``max`` for ``>``/``>=``, ``min`` for ``<``/``<=``),
    or ``None`` on any structural/semantic failure.
    """
    blocks = loop.nodes()
    cond_blocks = [b for b in blocks if isinstance(b, ConditionalBlock)]
    if len(cond_blocks) != 1:
        return None
    cb = cond_blocks[0]
    if len(cb.branches) != 1:
        return None
    branch_cond, branch_body = cb.branches[0]
    if branch_cond is None:
        return None
    guard_sym = branch_cond.as_string.strip()
    if not guard_sym.isidentifier():
        return None

    # Walk one hop back through iedges to where guard_sym is assigned.
    guard_iedge = next(
        (ie for ie in loop.edges() if ie.dst is cb and ie.data is not None and guard_sym in ie.data.assignments),
        None,
    )
    if guard_iedge is None:
        return None
    guard_rhs = guard_iedge.data.assignments[guard_sym]
    try:
        guard_tree = ast.parse(guard_rhs.strip().lstrip('(').rstrip(')'), mode='eval').body
    except SyntaxError:
        return None
    if not isinstance(guard_tree, ast.Compare) or len(guard_tree.ops) != 1:
        return None
    cmp_op = guard_tree.ops[0]
    if not isinstance(cmp_op, _CMP_GT + _CMP_LT):
        return None
    if not (isinstance(guard_tree.left, ast.Name) and isinstance(guard_tree.comparators[0], ast.Name)):
        return None
    cmp_lhs_sym, cmp_rhs_sym = guard_tree.left.id, guard_tree.comparators[0].id

    # One more hop back: the iedge into ``guard_iedge.src`` assigns the array-temp symbol.
    prep_state = guard_iedge.src
    arr_iedge = next(
        (ie for ie in loop.edges() if ie.dst is prep_state and ie.data is not None and ie.data.assignments),
        None,
    )
    if arr_iedge is None:
        return None
    arr_sym = None
    accum_name = None
    cmp_is_gt = None
    if cmp_lhs_sym in arr_iedge.data.assignments:
        arr_sym, accum_name = cmp_lhs_sym, cmp_rhs_sym
        # cmp is ``arr_sym <op> accum`` -- ``>``/``>=`` => max
        cmp_is_gt = isinstance(cmp_op, _CMP_GT)
    elif cmp_rhs_sym in arr_iedge.data.assignments:
        arr_sym, accum_name = cmp_rhs_sym, cmp_lhs_sym
        # cmp is ``accum <op> arr_sym`` -- ``<``/``<=`` => max (guard fires when accum
        # smaller, i.e. arr_sym larger)
        cmp_is_gt = isinstance(cmp_op, _CMP_LT)
    if arr_sym is None:
        return None

    arr_rhs = arr_iedge.data.assignments[arr_sym]
    try:
        arr_tree = ast.parse(arr_rhs.strip(), mode='eval').body
    except SyntaxError:
        return None
    if not isinstance(arr_tree, ast.Subscript) or not isinstance(arr_tree.value, ast.Name):
        return None
    array_name = arr_tree.value.id
    desc = sdfg.arrays.get(array_name)
    if desc is None:
        return None

    # TRUE branch body must be a single state whose only effect is ``accum_name = arr[i]``
    # (or ``= arr_sym``): find an AccessNode write to ``accum_name`` with one in-edge
    # tracing back through a passthrough chain to a read of ``array_name``.
    branch_states = [s for s in branch_body.nodes() if isinstance(s, SDFGState)]
    if len(branch_states) != 1:
        return None
    body_state = branch_states[0]

    accum_writes = [
        n for n in body_state.nodes()
        if isinstance(n, nodes.AccessNode) and n.data == accum_name and body_state.in_degree(n) > 0
    ]
    array_reads = [
        n for n in body_state.nodes()
        if isinstance(n, nodes.AccessNode) and n.data == array_name and body_state.out_degree(n) > 0
    ]
    if len(accum_writes) != 1 or len(array_reads) != 1:
        return None

    # Per-iteration array subset (the carry-input form ``arr[i]``).
    array_read_edges = [e for e in body_state.out_edges(array_reads[0]) if e.data and not e.data.is_empty()]
    if len(array_read_edges) != 1:
        return None
    array_subset = array_read_edges[0].data.subset
    if not _uses(array_subset, loop_var_sym):
        return None
    if _one_elem(array_subset) != 1:
        return None

    # Accumulator slot from the body's write to accum_name.
    accum_write_edges = [e for e in body_state.in_edges(accum_writes[0]) if e.data and not e.data.is_empty()]
    if len(accum_write_edges) != 1:
        return None
    accum_subset = accum_write_edges[0].data.subset
    if _one_elem(accum_subset) != 1 or _uses(accum_subset, loop_var_sym):
        return None

    wcr = _CALL_TO_WCR["max"] if cmp_is_gt else _CALL_TO_WCR["min"]
    expanded = _expand_over_loop(array_subset, loop_var_sym, start, end)
    if expanded is None:
        return None
    return _Reduction(wcr, accum_name, accum_subset, array_name, expanded)


def _lift(parent: ControlFlowRegion, loop: LoopRegion, info: _Reduction):
    """Replace ``loop`` with a ``Reduce``. Data-descriptor accumulator -> write directly;
    symbol accumulator -> synthesize a transient scalar, seed from the symbol, assign back on exit."""
    import dace
    root = parent
    while not isinstance(root, SDFG):
        root = root.parent_graph

    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    extra_assignments: Dict[str, str] = {}

    if info.accum in root.arrays:
        red_state = parent.add_state(loop.label + "_reduce", is_start_block=was_start)
        entry = red_state
        dest_name, dest_subset = info.accum, info.accum_subset
    else:
        tmp_name, _ = root.add_scalar(f"_red_tmp_{info.accum}",
                                      dtype=root.symbols[info.accum],
                                      transient=True,
                                      find_new_name=True)
        init_state = parent.add_state(loop.label + "_init", is_start_block=was_start)
        red_state = parent.add_state(loop.label + "_reduce")
        parent.add_edge(init_state, red_state, dace.InterstateEdge())
        seed = init_state.add_tasklet("seed", set(), {"_out"}, f"_out = {info.accum}")
        init_state.add_edge(seed, "_out", init_state.add_write(tmp_name), None,
                            mm.Memlet(data=tmp_name, subset=subsets.Range([(0, 0, 1)])))
        entry = init_state
        dest_name = tmp_name
        dest_subset = subsets.Range([(0, 0, 1)])
        extra_assignments[info.accum] = tmp_name

    for e in in_edges:
        parent.add_edge(e.src, entry, e.data)
    for e in out_edges:
        assigns = dict(e.data.assignments or {})
        assigns.update(extra_assignments)
        cond = e.data.condition.as_string if e.data.condition is not None else "1"
        parent.add_edge(red_state, e.dst, dace.InterstateEdge(condition=cond, assignments=assigns))
    parent.remove_node(loop)

    arr = red_state.add_read(info.array)
    dst = red_state.add_write(dest_name)
    red = red_state.add_reduce(info.wcr, axes=list(range(len(info.array_subset))), identity=None)
    red_state.add_edge(arr, None, red, None, mm.Memlet(data=info.array, subset=info.array_subset))
    red_state.add_edge(red, None, dst, None, mm.Memlet(data=dest_name, subset=dest_subset))


def _lift_wcr_scalar(parent: ControlFlowRegion, loop: LoopRegion, info: _Reduction):
    """Replace ``loop`` with ``init -> LoopRegion(WCR-on-scalar body) -> writeback``.

    Mirrors the reduce-libnode lift but keeps iteration explicit so downstream
    :class:`~dace.transformation.interstate.loop_to_map.LoopToMap` turns the LoopRegion
    into a Map with a WCR-on-scalar exit memlet -> WCR codegen lowers it to
    ``#pragma omp parallel for reduction(op:scalar)``.

    Body: ``arr_an --(arr[f(i)], wcr)--> priv_an``. The AN-to-AN WCR memlet tells codegen
    to combine via ``info.wcr`` (no intermediate tasklet). Init seeds the scalar from
    ``info.accum[info.accum_subset]`` (or a symbol assignment); writeback copies the
    post-loop scalar back (or assigns the symbol via an out-edge iedge).
    """
    import dace
    root = parent
    while not isinstance(root, SDFG):
        root = root.parent_graph

    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    extra_assignments: Dict[str, str] = {}

    accum_in_arrays = info.accum in root.arrays
    dtype = root.arrays[info.accum].dtype if accum_in_arrays else root.symbols[info.accum]
    priv_name, _ = root.add_scalar(f"_priv_{info.accum}", dtype=dtype, transient=True, find_new_name=True)

    init_state = parent.add_state(loop.label + "_priv_init", is_start_block=was_start)
    if accum_in_arrays:
        seed_r = init_state.add_read(info.accum)
        seed_w = init_state.add_write(priv_name)
        init_state.add_edge(seed_r, None, seed_w, None,
                            mm.Memlet(data=info.accum, subset=copy.deepcopy(info.accum_subset)))
    else:
        seed = init_state.add_tasklet("seed", set(), {"_out"}, f"_out = {info.accum}")
        init_state.add_edge(seed, "_out", init_state.add_write(priv_name), None,
                            mm.Memlet(data=priv_name, subset=subsets.Range([(0, 0, 1)])))

    new_loop = LoopRegion(
        loop.label + "_priv",
        condition_expr=loop.loop_condition.as_string,
        loop_var=loop.loop_variable,
        initialize_expr=loop.init_statement.as_string,
        update_expr=loop.update_statement.as_string,
    )
    parent.add_node(new_loop, ensure_unique_name=True)  # derived label; wired below by object ref

    body = new_loop.add_state(loop.label + "_body", is_start_block=True)
    arr_an = body.add_read(info.array)
    priv_an = body.add_write(priv_name)
    # ``info.array_subset`` is the union-over-iterations extent. Project each axis whose
    # range spans the loop (``loop_start`` -> ``loop_end``) back onto ``loop_var + per-iter
    # offset``; loop-invariant constant slices stay as-is. Covers full-array (``a[0:N]`` for
    # ``range(N)``) and offset (``a[1:N]`` for ``range(1, N)`` -- TSVC s314).
    iter_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    loop_start = loop_analysis.get_init_assignment(loop)
    loop_end = loop_analysis.get_loop_end(loop)
    loop_stride = loop_analysis.get_loop_stride(loop)
    per_iter_ranges = []
    for axis, (lo, hi, st) in enumerate(info.array_subset.ndrange()):
        # Reduction axis = axis whose element count equals the iteration count.
        # ``_expand_over_loop`` widened ``coeff*i + off`` into
        # ``(coeff*start+off, coeff*end+off, coeff*loop_stride)``; invert via
        # ``coeff = st / loop_stride`` to recover the per-iter index. Unit read over
        # unit loop -> ``coeff == 1``; strided read/loop -> ``coeff >= 1``.
        is_reduction_axis = False
        coeff = None
        if (loop_start is not None and loop_end is not None and loop_stride is not None and loop_stride != 0
                and st != 0):
            coeff = symbolic.simplify(st / loop_stride)
            axis_elems = symbolic.simplify((hi - lo) / st + 1)
            iters = symbolic.simplify((loop_end - loop_start) / loop_stride + 1)
            if coeff.is_integer and coeff.is_positive and not symbolic.inequal_symbols(axis_elems, iters):
                is_reduction_axis = True
        if is_reduction_axis:
            idx = symbolic.simplify(coeff * iter_sym + (lo - coeff * loop_start))
            per_iter_ranges.append((idx, idx, 1))
        else:
            per_iter_ranges.append((lo, hi, st))
    body.add_edge(
        arr_an, None, priv_an, None,
        mm.Memlet(data=info.array,
                  subset=subsets.Range(per_iter_ranges),
                  other_subset=subsets.Range([(0, 0, 1)]),
                  wcr=info.wcr))

    wb_state = parent.add_state(loop.label + "_priv_wb")
    if accum_in_arrays:
        wb_r = wb_state.add_read(priv_name)
        wb_w = wb_state.add_write(info.accum)
        wb_state.add_edge(wb_r, None, wb_w, None, mm.Memlet(data=info.accum, subset=copy.deepcopy(info.accum_subset)))
    else:
        # Symbol accumulator: lift the value via an iedge assignment on the writeback
        # state's OUT-edge, same as the reduce-libnode path.
        extra_assignments[info.accum] = priv_name

    parent.add_edge(init_state, new_loop, dace.InterstateEdge())
    parent.add_edge(new_loop, wb_state, dace.InterstateEdge())
    for e in in_edges:
        parent.add_edge(e.src, init_state, e.data)
    for e in out_edges:
        assigns = dict(e.data.assignments or {})
        assigns.update(extra_assignments)
        cond = e.data.condition.as_string if e.data.condition is not None else "1"
        parent.add_edge(wb_state, e.dst, dace.InterstateEdge(condition=cond, assignments=assigns))
    parent.remove_node(loop)


def _extract_wcr_body(loop: LoopRegion, sdfg: SDFG):
    """Locate a single WCR-bearing write to a constant slot of a non-transient array.

    After ``TTE + AugAssignToWCR``, a multi-tasklet ``compute then accumulate`` shape
    (``dot[0] = dot[0] + a[i]*b[i]``) collapses to a clean WCR write to ``accum[c]``.
    Refuse any body without exactly one such write (multiple independent reductions =
    ambiguous, not privatised as one accumulator).

    :returns: ``(wcr_state, wcr_edge, accum_name, accum_subset)`` or ``None``.
    """
    if not loop.loop_variable:
        return None
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    # Same per-iter-mutated-symbol guard as ``_extract``: write subset must not reference
    # a symbol reassigned on a loop iedge (TSVC s141 ``k = k + j + 1``), else each iteration
    # writes a different slot -> not a single-accumulator reduction.
    loop_iedge_assignees = set()
    for e in loop.edges():
        loop_iedge_assignees.update(e.data.assignments.keys())
    candidates = []
    for state in loop.all_states():
        for e in state.edges():
            if e.data is None or e.data.wcr is None:
                continue
            if not isinstance(e.dst, nodes.AccessNode):
                continue
            desc = sdfg.arrays.get(e.dst.data)
            if desc is None or desc.transient:
                continue
            if e.data.subset is None or _uses(e.data.subset, loop_var_sym):
                continue
            if _one_elem(e.data.subset) != 1:
                continue
            if any(str(s) in loop_iedge_assignees for s in e.data.subset.free_symbols):
                continue
            candidates.append((state, e, e.dst.data, copy.deepcopy(e.data.subset)))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _lift_wcr_scalar_retarget(parent: ControlFlowRegion, loop: LoopRegion, wcr_state: SDFGState, wcr_edge,
                              accum_name: str, accum_subset: subsets.Subset):
    """Wrap ``loop`` with ``init -> loop -> writeback`` and retarget the in-body WCR write
    from ``accum[accum_subset]`` to a fresh transient scalar ``_priv_<accum>``.

    Preserves the body (incl. any compute chain feeding the accumulator); only the WCR
    edge's destination changes. Same Map-able shape as ``PrivatizeReductionAccumulator``
    via ``AugAssignToWCR + LoopToMap + PRA``, but parallelisation happens later (downstream
    ``LoopToMap``) not here.
    """
    import dace
    root = parent
    while not isinstance(root, SDFG):
        root = root.parent_graph

    desc = root.arrays[accum_name]
    priv_name, _ = root.add_scalar(f"_priv_{accum_name}", dtype=desc.dtype, transient=True, find_new_name=True)

    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))

    init_state = parent.add_state(loop.label + "_priv_init", is_start_block=was_start)
    init_state.add_edge(init_state.add_read(accum_name), None, init_state.add_write(priv_name), None,
                        mm.Memlet(data=accum_name, subset=copy.deepcopy(accum_subset)))

    accum_sink = wcr_edge.dst
    wcr_state.remove_edge(wcr_edge)
    priv_an = wcr_state.add_write(priv_name)
    wcr_state.add_edge(wcr_edge.src, wcr_edge.src_conn, priv_an, None,
                       mm.Memlet(data=priv_name, subset=subsets.Range([(0, 0, 1)]), wcr=wcr_edge.data.wcr))
    if wcr_state.degree(accum_sink) == 0:
        wcr_state.remove_node(accum_sink)

    wb_state = parent.add_state(loop.label + "_priv_wb")
    wb_state.add_edge(wb_state.add_read(priv_name), None, wb_state.add_write(accum_name), None,
                      mm.Memlet(data=accum_name, subset=copy.deepcopy(accum_subset)))

    for e in in_edges:
        parent.remove_edge(e)
        parent.add_edge(e.src, init_state, e.data)
    parent.add_edge(init_state, loop, dace.InterstateEdge())
    parent.add_edge(loop, wb_state, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(wb_state, e.dst, e.data)


def _extract_multi_state_chain(loop: LoopRegion, sdfg: SDFG):
    """Find a body state with a ``read-accum -> op-tasklet -> (transient passthrough)* ->
    write-accum`` chain on the SAME constant slot, beyond ``AugAssignToWCR``'s reach (an
    extra transient AccessNode between combining tasklet and sink takes it past the 5-node
    ``arr -> copy_in -> tasklet -> copy_out -> arr`` pattern). Walks transient AccessNodes
    with in_degree == out_degree == 1 either direction.

    Accepts transient (TSVC s4115 ``s``) and non-transient accumulators; either way the
    privatised scalar is seeded/written-back at the boundary.

    :returns: ``(state, final_tasklet, carry_in_edge, value_in_edge, first_write_edge,
              last_write_edge, accum_source_an, accum_sink_an, wcr_lambda, accum_name,
              accum_subset)`` or ``None``.
    """
    if not loop.loop_variable:
        return None
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    for state in loop.all_states():
        by_data: Dict[str, List[nodes.AccessNode]] = {}
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode):
                by_data.setdefault(n.data, []).append(n)
        for data_name, ans in by_data.items():
            sources = [n for n in ans if state.in_degree(n) == 0 and state.out_degree(n) >= 1]
            sinks = [n for n in ans if state.in_degree(n) >= 1 and state.out_degree(n) == 0]
            if len(sources) != 1 or len(sinks) != 1:
                continue
            src_an, sink_an = sources[0], sinks[0]

            # Source's single out-edge must go to a Tasklet (the carry input).
            src_out = list(state.out_edges(src_an))
            if len(src_out) != 1:
                continue
            carry_in_edge = src_out[0]
            if not isinstance(carry_in_edge.dst, nodes.Tasklet):
                continue
            final_tasklet = carry_in_edge.dst

            # Walk back from sink through 1-in/1-out transient passthroughs to a Tasklet;
            # must be the same ``final_tasklet`` reached from the source.
            sink_in = list(state.in_edges(sink_an))
            if len(sink_in) != 1:
                continue
            last_write_edge = sink_in[0]
            cur_edge = last_write_edge
            cur_src = cur_edge.src
            while isinstance(cur_src, nodes.AccessNode):
                desc = sdfg.arrays.get(cur_src.data)
                if (desc is None or not desc.transient or state.in_degree(cur_src) != 1
                        or state.out_degree(cur_src) != 1):
                    break
                prev = list(state.in_edges(cur_src))[0]
                cur_edge = prev
                cur_src = cur_edge.src
            if cur_src is not final_tasklet:
                continue
            first_write_edge = cur_edge

            # Carry-read and final write must use the same constant single-element,
            # loop-invariant slot.
            carry_subset = carry_in_edge.data.subset if carry_in_edge.data is not None else None
            write_subset = last_write_edge.data.subset if last_write_edge.data is not None else None
            if carry_subset is None or write_subset is None:
                continue
            if _one_elem(carry_subset) != 1 or _one_elem(write_subset) != 1:
                continue
            if _uses(carry_subset, loop_var_sym) or _uses(write_subset, loop_var_sym):
                continue
            if str(carry_subset) != str(write_subset):
                continue

            # Validate the final tasklet's body is a known WCR-able op.
            try:
                tree = ast.parse((final_tasklet.code.as_string or "").strip())
            except SyntaxError:
                continue
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                continue
            rhs = tree.body[0].value
            if isinstance(rhs, ast.BinOp):
                wcr = _BINOP_TO_WCR.get(type(rhs.op))
            elif isinstance(rhs, ast.BoolOp) and len(rhs.values) == 2:
                wcr = _BOOLOP_TO_WCR.get(type(rhs.op))
            elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and len(rhs.args) == 2):
                wcr = _CALL_TO_WCR.get(rhs.func.id)
            else:
                wcr = None
            if wcr is None:
                continue

            # Tasklet: exactly 2 data inputs (carry + value), 1 data output.
            data_in = [e for e in state.in_edges(final_tasklet) if e.data is not None and not e.data.is_empty()]
            data_out = [e for e in state.out_edges(final_tasklet) if e.data is not None and not e.data.is_empty()]
            if len(data_in) != 2 or len(data_out) != 1:
                continue
            value_in_edge = next((e for e in data_in if e is not carry_in_edge), None)
            if value_in_edge is None:
                continue

            return (state, final_tasklet, carry_in_edge, value_in_edge, first_write_edge, last_write_edge, src_an,
                    sink_an, wcr, data_name, copy.deepcopy(carry_subset))
    return None


def _lift_multi_state_chain(parent: ControlFlowRegion, loop: LoopRegion, info):
    """Rewrite the chain found by ``_extract_multi_state_chain``:

    - Drop the tasklet's carry input (RHS keeps only the value subexpr; remove carry
      in-edge + connector).
    - Add WCR to the final write edge, retarget the sink AccessNode to ``_priv_<accum>``.
    - Add ``init`` (accum -> ``_priv_X``) and ``wb`` (``_priv_X`` -> accum) states around the loop.
    """
    import dace
    (state, final_tasklet, carry_in_edge, value_in_edge, first_write_edge, last_write_edge, src_an, sink_an, wcr,
     accum_name, accum_subset) = info

    root = parent
    while not isinstance(root, SDFG):
        root = root.parent_graph

    desc = root.arrays[accum_name]
    priv_name, _ = root.add_scalar(f"_priv_{accum_name}", dtype=desc.dtype, transient=True, find_new_name=True)

    # ---- rewrite the tasklet's RHS to drop the carry operand ----
    tree = ast.parse((final_tasklet.code.as_string or "").strip())
    assign_node = tree.body[0]
    rhs = assign_node.value
    carry_conn = carry_in_edge.dst_conn

    new_rhs = None
    if isinstance(rhs, ast.BinOp):
        if isinstance(rhs.left, ast.Name) and rhs.left.id == carry_conn:
            new_rhs = rhs.right
        elif isinstance(rhs.right, ast.Name) and rhs.right.id == carry_conn:
            new_rhs = rhs.left
    elif isinstance(rhs, ast.BoolOp):
        kept = [v for v in rhs.values if not (isinstance(v, ast.Name) and v.id == carry_conn)]
        if len(kept) == 1:
            new_rhs = kept[0]
    elif isinstance(rhs, ast.Call):
        kept_args = [a for a in rhs.args if not (isinstance(a, ast.Name) and a.id == carry_conn)]
        if len(kept_args) == 1:
            new_rhs = kept_args[0]
    if new_rhs is None:
        # Could not simplify -- leave the SDFG untouched.
        return

    final_tasklet.code.code = [ast.copy_location(ast.Assign(targets=assign_node.targets, value=new_rhs), assign_node)]

    state.remove_edge(carry_in_edge)
    if carry_conn in final_tasklet.in_connectors:
        final_tasklet.remove_in_connector(carry_conn)
    if state.degree(src_an) == 0:
        state.remove_node(src_an)

    # ---- retarget the final write to ``_priv_X`` with WCR ----
    last_src = last_write_edge.src
    last_src_conn = last_write_edge.src_conn
    state.remove_edge(last_write_edge)
    priv_sink_an = state.add_write(priv_name)
    state.add_edge(last_src, last_src_conn, priv_sink_an, None,
                   mm.Memlet(data=priv_name, subset=subsets.Range([(0, 0, 1)]), wcr=wcr))
    if state.degree(sink_an) == 0:
        state.remove_node(sink_an)

    # ---- wrap the loop with init + writeback states ----
    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))

    init_state = parent.add_state(loop.label + "_priv_init", is_start_block=was_start)
    init_state.add_edge(init_state.add_read(accum_name), None, init_state.add_write(priv_name), None,
                        mm.Memlet(data=accum_name, subset=copy.deepcopy(accum_subset)))

    wb_state = parent.add_state(loop.label + "_priv_wb")
    wb_state.add_edge(wb_state.add_read(priv_name), None, wb_state.add_write(accum_name), None,
                      mm.Memlet(data=accum_name, subset=copy.deepcopy(accum_subset)))

    for e in in_edges:
        parent.remove_edge(e)
        parent.add_edge(e.src, init_state, e.data)
    parent.add_edge(init_state, loop, dace.InterstateEdge())
    parent.add_edge(loop, wb_state, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(wb_state, e.dst, e.data)
