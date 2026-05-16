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
from typing import List

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
