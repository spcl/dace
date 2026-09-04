# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reduction-expression emission helpers for the vectorization pipeline.

- ``IDENTITY`` maps a reduction-op string to its accumulator identity
  element.
- ``emit_chain_reduction`` emits a linear left-associated reduction
  (critical path ``O(W)``).
- ``emit_tree_reduction`` emits a balanced-tree reduction
  (critical path ``O(log W)``); not bit-exact with the chain form for
  non-associative ops.

Both emitters return a single C++/Python expression string that the
caller wraps with ``_out = ...``.
"""
import ast
from dataclasses import dataclass
from typing import List, Optional

import dace

_INFIX_OPS = {"+", "-", "*", "/", "&", "|", "^"}
_FUNCALL_OPS = {"max", "min"}

# Identity element for each supported reduction op. Used to initialise the
# per-lane vector accumulator without depending on a literal-zero-init
# tasklet pattern. Strings are interpreted as numeric literals at emit time
# (so ``"-inf"`` is the C++ ``-INFINITY``-equivalent placeholder).
IDENTITY = {
    "+": "0",
    "*": "1",
    "&": "~0",
    "|": "0",
    "^": "0",
    "max": "-inf",
    "min": "+inf",
}


def _validate(op: str, vector_width: int) -> None:
    """Validate a reduction op and lane count.

    :param op: Reduction operator string.
    :param vector_width: Lane count.
    :raises NotImplementedError: if ``op`` is not a supported reduction op.
    :raises ValueError: if ``vector_width`` is less than 1.
    """
    if op not in _INFIX_OPS and op not in _FUNCALL_OPS:
        raise NotImplementedError(f"emit_*_reduction: unsupported op {op!r}")
    if vector_width < 1:
        raise ValueError(f"emit_*_reduction: vector_width must be >= 1, got {vector_width}")


def _wrap_pair(op: str, left: str, right: str) -> str:
    if op in _INFIX_OPS:
        return f"({left} {op} {right})"
    return f"{op}({left}, {right})"


def emit_chain_reduction(input_var: str, vector_width: int, op: str) -> str:
    """Emit a linear left-associated reduction — critical path ``O(W)``.

    :param input_var: Name of the W-wide input buffer.
    :param vector_width: Lane count.
    :param op: Reduction operator string.
    :returns: ``a op b op c op d`` for infix ops; ``op(op(op(a, b), c), d)``
        for function-call ops.
    """
    _validate(op, vector_width)
    lanes = [f"{input_var}[{i}]" for i in range(vector_width)]
    if vector_width == 1:
        return lanes[0]
    if op in _INFIX_OPS:
        return f" {op} ".join(lanes)
    expr = lanes[0]
    for lane in lanes[1:]:
        expr = _wrap_pair(op, expr, lane)
    return expr


def emit_tree_reduction(input_var: str, vector_width: int, op: str) -> str:
    """Emit a balanced-tree reduction — critical path ``O(log W)``.

    Pairs lanes ``(0,1)``, ``(2,3)``, ... at each level, halving until one
    operand remains. An odd lane trails forward unchanged to the next level.

    :param input_var: Name of the W-wide input buffer.
    :param vector_width: Lane count.
    :param op: Reduction operator string.
    :returns: The balanced-tree reduction expression.
    """
    _validate(op, vector_width)
    operands: List[str] = [f"{input_var}[{i}]" for i in range(vector_width)]
    while len(operands) > 1:
        nxt: List[str] = []
        for i in range(0, len(operands) - 1, 2):
            nxt.append(_wrap_pair(op, operands[i], operands[i + 1]))
        if len(operands) % 2 == 1:
            nxt.append(operands[-1])
        operands = nxt
    return operands[0]


# AST node -> reduction-op token (an ``IDENTITY`` key). Mirrors the
# loop_to_reduce WCR tables but yields this module's short op token
# rather than a ``lambda a, b: ...`` string (a different representation,
# not duplicated behaviour). max / min are function calls, not infix.
_AST_BINOP_TO_OP = {
    ast.Add: "+",
    ast.Mult: "*",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
}
_AST_BOOLOP_TO_OP = {
    ast.Or: "|",
    ast.And: "&",
}
_CALL_REDUCERS = {"max", "min"}


@dataclass(frozen=True)
class ReductionInfo:
    """Result of :func:`recognize_reduction` — data only, no behaviour.

    The only sanctioned class category here (a frozen result record, same
    rule as the name-scheme dataclasses; no OOP abstractions).

    :param op: The reduction operator as an :data:`IDENTITY` key
        (``+``, ``*``, ``&``, ``|``, ``^``, ``max``, ``min``).
    :param accumulator: Data-descriptor name of the read-modify-written
        scalar accumulator.
    :param identity: The op's identity element (``IDENTITY[op]``), used
        to seed the per-lane vector accumulator without relying on a
        literal-init tasklet pattern.
    """

    op: str
    accumulator: str
    identity: str


def _single_assignment(code: str) -> Optional[ast.Assign]:
    """Parse ``code`` and return its body iff it is one bare ``Assign``."""
    try:
        tree = ast.parse((code or "").strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    return tree.body[0]


def _reduction_op_and_operands(rhs: ast.AST):
    """``(op_token, [operand_ast, ...])`` for a reduction RHS, or ``None``.

    Recognises ``a <binop> b``, ``a and/or b``, and ``max(a, b)`` /
    ``min(a, b)``. The operands are returned verbatim so the caller can
    test which one is the accumulator; the *other* operand may be an
    arbitrary expression (a compound product, an indirect gather, …) —
    that is the robustness win over ``_extract_single_op``.
    """
    if isinstance(rhs, ast.BinOp):
        op = _AST_BINOP_TO_OP.get(type(rhs.op))
        return (op, [rhs.left, rhs.right]) if op is not None else None
    if isinstance(rhs, ast.BoolOp) and len(rhs.values) == 2:
        op = _AST_BOOLOP_TO_OP.get(type(rhs.op))
        return (op, list(rhs.values)) if op is not None else None
    if (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id in _CALL_REDUCERS
            and len(rhs.args) == 2):
        return (rhs.func.id, list(rhs.args))
    return None


def _has_data(edge) -> bool:
    return edge.data is not None and not edge.data.is_empty()


def recognize_reduction(state: "dace.SDFGState", tasklet: "dace.nodes.Tasklet") -> Optional[ReductionInfo]:
    """Recognise a read-modify-write scalar reduction tasklet.

    Detects the canonical accumulation shape ``acc = acc <op> <expr>``
    (or ``acc = <op>(acc, <expr>)`` for ``max`` / ``min``) where the
    accumulator connector is *both* read (an in-edge) and written (the
    out-edge) to the **same** data descriptor. ``<expr>`` is left
    unconstrained: a compound product or an indirect gather (the spmv /
    cloudsc shapes) is accepted — this is the robustness gain over the
    fragile single-op ``_extract_single_op`` path, which mis-detects the
    operator on a compound right-hand side.

    Only the structurally-safe associative reduction operators in
    :data:`IDENTITY` are recognised; subtraction / division (no
    identity, non-associative) are rejected.

    :param state: The state containing ``tasklet``.
    :param tasklet: The candidate update tasklet.
    :returns: A :class:`ReductionInfo`, or ``None`` if ``tasklet`` is not
        a recognised scalar reduction.
    """
    if not isinstance(tasklet, dace.nodes.Tasklet):
        return None
    if tasklet.code.language != dace.dtypes.Language.Python:
        return None
    assign = _single_assignment(tasklet.code.as_string)
    if assign is None or len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    parsed = _reduction_op_and_operands(assign.value)
    if parsed is None:
        return None
    op, operands = parsed
    if op not in IDENTITY:
        return None

    out_edges = [e for e in state.out_edges(tasklet) if _has_data(e)]
    in_edges = [e for e in state.in_edges(tasklet) if _has_data(e)]
    if len(out_edges) != 1:
        return None
    write_edge = out_edges[0]
    if not isinstance(write_edge.dst, dace.nodes.AccessNode):
        return None
    accum = write_edge.dst.data

    # The accumulator must also be *read* by this tasklet: some in-edge
    # comes from an AccessNode for the same data, and that in-connector
    # appears as a direct operand of the top-level reduction op.
    operand_names = {o.id for o in operands if isinstance(o, ast.Name)}
    for e in in_edges:
        if not isinstance(e.src, dace.nodes.AccessNode) or e.src.data != accum:
            continue
        if e.dst_conn in operand_names:
            return ReductionInfo(op=op, accumulator=accum, identity=IDENTITY[op])
    return None


@dataclass
class MapReductionInfo:
    """Result of :func:`recognize_map_reduction` — a per-iteration scalar
    reduction carried *across* an innermost map (the spmv "row reduction").

    Unlike :class:`ReductionInfo` (a single flat tasklet), this describes an
    innermost map ``R`` that read-modify-writes a scalar accumulator through
    an opaque body — typically a :class:`~dace.nodes.NestedSDFG` performing a
    gather + product (``acc = acc + data[idx] * x[indices[idx]]``). The body
    is an *indirect-access* NSDFG that cannot be inlined, so the operator is
    recovered by peeking (read-only) at the combining tasklet inside it.

    :param op: Reduction operator (an :data:`IDENTITY` key).
    :param identity: ``IDENTITY[op]`` — the fold seed.
    :param accumulator: The carried scalar's data-descriptor name.
    :param map_entry: The reduction map's entry node.
    :param map_exit: The reduction map's exit node.
    :param body: The single node in the map scope (NSDFG or tasklet).
    :param read_edge: ``map_entry -> body`` edge carrying ``acc`` in.
    :param write_edge: ``body -> map_exit`` edge carrying ``acc`` out.
    """

    op: str
    identity: str
    accumulator: str
    map_entry: "dace.nodes.MapEntry"
    map_exit: "dace.nodes.MapExit"
    body: "dace.nodes.Node"
    read_edge: object
    write_edge: object


def _reduction_op_for_connector(tasklet: "dace.nodes.Tasklet", conn: str) -> Optional[str]:
    """Reduction op of ``tasklet`` iff ``conn`` is an operand of its top-level
    associative binop (``__out = conn <op> other`` / ``op(conn, other)``).

    Reuses :func:`_reduction_op_and_operands` so the recognised op set stays
    identical to the flat :func:`recognize_reduction` path.

    :param tasklet: The candidate combining tasklet.
    :param conn: The accumulator-carrying input connector name.
    :returns: The op token (an :data:`IDENTITY` key) or ``None``.
    """
    if not isinstance(tasklet, dace.nodes.Tasklet) or tasklet.code.language != dace.dtypes.Language.Python:
        return None
    assign = _single_assignment(tasklet.code.as_string)
    if assign is None or len(assign.targets) != 1:
        return None
    parsed = _reduction_op_and_operands(assign.value)
    if parsed is None:
        return None
    op, operands = parsed
    if op not in IDENTITY:
        return None
    if conn in {o.id for o in operands if isinstance(o, ast.Name)}:
        return op
    return None


def _op_through_body(state: "dace.SDFGState", body: "dace.nodes.Node", read_edge, write_edge) -> Optional[str]:
    """Recover the reduction op combining the accumulator inside ``body``.

    ``body`` is either a flat tasklet (delegate to :func:`recognize_reduction`)
    or an opaque NSDFG. For the NSDFG case the accumulator enters on connector
    ``read_edge.dst_conn``; the inner array of that name is consumed by exactly
    the combining tasklet — parse it for the associative op.

    :param state: The state holding ``body``.
    :param body: The map-scope body node.
    :param read_edge: ``map_entry -> body`` edge (accumulator in).
    :param write_edge: ``body -> map_exit`` edge (accumulator out).
    :returns: The op token or ``None`` if not a recognised reduction.
    """
    if isinstance(body, dace.nodes.Tasklet):
        info = recognize_reduction(state, body)
        return info.op if info is not None else None
    if not isinstance(body, dace.nodes.NestedSDFG):
        return None
    cin = read_edge.dst_conn
    if cin is None:
        return None
    inner = body.sdfg
    for st in inner.states():
        for an in st.data_nodes():
            if an.data != cin:
                continue
            for oe in st.out_edges(an):
                op = _reduction_op_for_connector(oe.dst, oe.dst_conn)
                if op is not None:
                    return op
    return None


def recognize_map_reduction(state: "dace.SDFGState", map_entry: "dace.nodes.MapEntry") -> Optional[MapReductionInfo]:
    """Recognise a scalar reduction carried across an innermost map.

    The shape (the spmv ``for idx: tmp = tmp + data[idx]*x[indices[idx]]``
    row reduction): a single-param, unit-step innermost map whose scope is one
    body node that *reads* a scalar ``acc[0]`` at the entry and *writes* the
    same ``acc[0]`` at the exit — a loop-carried read-modify-write. The body is
    an opaque indirect-access NSDFG (the gather cannot be inlined), so the
    operator is recovered by peeking at the combining tasklet inside it.

    :param state: The state containing ``map_entry``.
    :param map_entry: The candidate innermost reduction map.
    :returns: A :class:`MapReductionInfo`, or ``None`` if not recognised.
    """
    if not isinstance(map_entry, dace.nodes.MapEntry):
        return None
    if len(map_entry.map.params) != 1:
        return None
    _, _, step = map_entry.map.range[-1]
    if (step != 1) and (str(step) != "1"):
        return None
    map_exit = state.exit_node(map_entry)
    # Innermost only: no nested map in scope.
    inner = state.all_nodes_between(map_entry, map_exit) or set()
    if any(isinstance(n, dace.nodes.MapEntry) for n in inner):
        return None
    body_nodes = [n for n in inner if n not in (map_entry, map_exit)]
    if len(body_nodes) != 1:
        return None
    body = body_nodes[0]

    def _scalar_slot(e) -> bool:
        return (e.data is not None and e.data.data is not None and e.data.subset is not None
                and e.data.subset.num_elements() == 1)

    reads = {e.data.data: e for e in state.out_edges(map_entry) if e.dst is body and _scalar_slot(e)}
    writes = {e.data.data: e for e in state.in_edges(map_exit) if e.src is body and _scalar_slot(e)}
    for acc in set(reads) & set(writes):
        desc = state.sdfg.arrays.get(acc)
        if desc is None or not isinstance(desc, (dace.data.Scalar, dace.data.Array)):
            continue
        read_edge, write_edge = reads[acc], writes[acc]
        op = _op_through_body(state, body, read_edge, write_edge)
        if op is None or op not in IDENTITY:
            continue
        return MapReductionInfo(op=op,
                                identity=IDENTITY[op],
                                accumulator=acc,
                                map_entry=map_entry,
                                map_exit=map_exit,
                                body=body,
                                read_edge=read_edge,
                                write_edge=write_edge)
    return None
