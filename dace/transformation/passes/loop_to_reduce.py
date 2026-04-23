# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Detect scalar accumulator loops and replace them with ``Reduce`` nodes.

Three loop shapes are recognised (``identity=None`` on the emitted
``Reduce`` so the pre-loop accumulator seeds the fold):

- **Tasklet**: a single-state containing one two-input tasklet that
  writes to the accumulator.
- **Interstate edge**: body = 2 empty states joined by one interstate
  edge with assignment ``{sym: sym <op> arr[<f(i)>]}``.
- **Conditional interstate edge**: body = a single ``ConditionalBlock``
  with one branch guarded by ``sym <cmp> arr[<f(i)>]`` (``cmp`` in
  ``>``/``>=``/``<``/``<=``) whose body is the 2-empty-states + edge
  shape above with assignment ``{sym: arr[<f(i)>]}``. ``>``/``>=`` lift
  to ``max``, ``<``/``<=`` lift to ``min``.

Accumulator forms accepted: a ``Scalar``, a length-1 ``Array``, a single
loop-invariant slice of a multi-element ``Array`` (``C[k]``).
"""
import ast
import copy as _copy
from typing import Dict, NamedTuple, Optional

import sympy

from dace import SDFG, SDFGState, data, dtypes, memlet as mm, nodes, properties, subsets, symbolic
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.symbolic import AND, OR, bitwise_and, bitwise_or
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

# Ops in these tables are commutative by construction, so we skip calling
# ``dace.frontend.operations.is_op_commutative`` (which returns ``None`` for
# ``max`` / ``min`` because Python's builtins choke on symbolic arguments).
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
# For a guard `lhs <cmp> rhs` where the assignment inside writes `sym = arr[i]`,
# the reduction is max iff the condition fires when arr is larger than sym.
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

    def __init__(self, permissive: bool = False):
        super().__init__()
        self.permissive = permissive

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for node, parent in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, LoopRegion):
                continue
            info = _extract(node, sdfg, permissive=self.permissive)
            if info is None:
                continue
            _lift(parent, node, info)
            count += 1
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


def _expand_over_loop(subset: subsets.Subset, loop_var: sympy.Symbol, start, end) -> Optional[subsets.Range]:
    """Widen ``subset`` -- which uses ``loop_var`` linearly -- over the
    iteration range ``[start, end]``."""
    if not isinstance(subset, subsets.Range):
        return None
    ranges = []
    for rb, re_, rs in subset.ndrange():
        if rb != re_ or rs != 1:
            return None
        offset = symbolic.simplify(rb - loop_var)
        if offset.has(loop_var):
            return None
        ranges.append((symbolic.simplify(start + offset), symbolic.simplify(end + offset), 1))
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


def _extract_any_pattern(cond, const_rhs: int, target: str, sdfg: SDFG, loop_var_sym,
                         start, end) -> Optional["_Reduction"]:
    """Match ``{sym: const}`` conditional-interstate-edge "any"/"all".

    Body = ``ConditionalBlock`` with one branch, guard ``arr[<subs>] <cmp> C``
    (C integer), branch = 2 empty states + interstate edge with assignment
    ``{sym: <const_rhs>}`` where ``const_rhs`` is 0 or 1.

    The guard array is assumed to be 0/1-valued, so ``any(arr[...] == 1)``
    over the iteration range is equivalent to the bitwise-OR of ``arr[...]``
    -- no predicate synthesis needed, a plain ``Reduce(|)`` over the array
    slice suffices. ``const_rhs == 1`` lifts to OR; ``const_rhs == 0`` lifts
    to AND.
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
        return (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id in sdfg.arrays)

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
            ranges.append((symbolic.simplify(start + offset),
                           symbolic.simplify(end + offset), 1))
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
    if start is None or end is None or stride is None or stride != 1:
        return None

    blocks = loop.nodes()
    loop_var = loop.loop_variable
    loop_var_sym = symbolic.pystr_to_symbolic(loop_var)

    # Tasklet pattern: single state with exactly one tasklet.
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
                resolved.append((pred.src.data, _copy.deepcopy(pred.data.subset)))
            else:
                resolved.append((src.data, e.data.subset))

        accum_ok = False
        array, arr_subset = None, None
        for name, sub in resolved:
            if _uses(sub, loop_var_sym):
                if array is not None:
                    return None
                array, arr_subset = name, sub
            elif _one_elem(sub) == 1 and ((name == accum and sub == write_subset) or
                                          (name != accum and _scalar_equiv(sdfg, name, accum))):
                accum_ok = True
        if not accum_ok or array is None or array == accum:
            return None

        expanded = _expand_over_loop(arr_subset, loop_var_sym, start, end)
        if expanded is None:
            return None
        return _Reduction(wcr, accum, write_subset, array, expanded)

    # Interstate-edge pattern: 2 empty states + 1 edge with 1 assignment,
    # either at loop level or inside a single-branch ConditionalBlock whose
    # guard is a >/>=/</<= comparison between the accumulator and the array.
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

        # ``pystr_to_symbolic`` renders ``B[i]`` as a sympy ``Function("B")(i)``.
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
                if isinstance(arg, sympy.Function) and str(arg.func) in sdfg.arrays:
                    if arr_call is not None:
                        return None
                    arr_call = arg
                else:
                    other = arg
            if arr_call is None or other != target_sym:
                return None
        else:
            # Conditional-interstate-edge path.
            # "any"/"all" pattern: ``{sym: <const>}`` with an array-predicate
            # guard; lifts to OR / AND over the (0/1-valued) guard array.
            # Gated on ``permissive`` -- the lift is only semantically correct
            # if the guard array happens to hold only 0/1 values, which the
            # pass cannot verify statically.
            if permissive and isinstance(expr, sympy.Integer) and int(expr) in (0, 1):
                return _extract_any_pattern(cond, int(expr), target, sdfg,
                                            loop_var_sym, start, end)
            # Pure copy ``sym = arr[f(i)]`` gated by a max/min comparison.
            if not (isinstance(expr, sympy.Function) and str(expr.func) in sdfg.arrays):
                return None
            arr_call = expr
            wcr = _cmp_to_wcr(cond, target, str(arr_call.func))
            if wcr is None:
                return None

        array = str(arr_call.func)
        if len(sdfg.arrays[array].shape) != 1 or len(arr_call.args) != 1:
            return None
        offset = symbolic.simplify(arr_call.args[0] - loop_var_sym)
        if offset.has(loop_var_sym):
            return None

        return _Reduction(
            wcr=wcr,
            accum=target,
            accum_subset=subsets.Range([(0, 0, 1)]),
            array=array,
            array_subset=subsets.Range([(symbolic.simplify(start + offset), symbolic.simplify(end + offset), 1)]),
        )

    return None


def _lift(parent: ControlFlowRegion, loop: LoopRegion, info: _Reduction):
    """Replace ``loop`` with a ``Reduce``. If the accumulator is a data
    descriptor we write to it directly; if it's a symbol we synthesize a
    transient scalar, seed it from the symbol, and assign back on exit."""
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
