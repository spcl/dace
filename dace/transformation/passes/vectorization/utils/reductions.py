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
