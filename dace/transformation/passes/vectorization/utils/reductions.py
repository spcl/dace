# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction emission helpers for the vectorization pipeline.

This module is the staged home for the reductions redesign (R-1 .. R-5
in the plan). Today it contains only the read-only building blocks
R-1 introduces; the actual integration into ``move_out_reduction`` /
``reduce_before_use`` is a separate R-2 slice.

What lives here:

- ``IDENTITY`` — table mapping reduction-op string to the identity
  element used to initialise the vector accumulator. The current
  reduction-lift path only recognises literal-zero accumulator init
  (``acc = 0``) which restricts it to additive reductions; the
  identity table is the data half of the fix.
- ``emit_chain_reduction`` — the existing linear-chain emitter,
  factored out so callers can switch between chain and tree forms
  without copying code. Behaviour-equivalent to the inline emitter
  ``reduce_before_use`` currently uses.
- ``emit_tree_reduction`` — log-depth tree fallback. For an associative
  op the chain ``a + b + c + d + e + f + g + h`` (critical path 7)
  becomes ``((a + b) + (c + d)) + ((e + f) + (g + h))`` (critical
  path 3). For non-infix ops (``max``/``min``) the equivalent shape
  is ``max(max(max(a, b), max(c, d)), max(max(e, f), max(g, h)))``.

Both emitters return a single C++/Python expression string that the
caller wraps with ``_out = ...`` and feeds to ``add_tasklet``.

Per the locked policy, this slice is mechanical and purely additive —
no callers switch over yet. R-2 will migrate ``reduce_before_use``
to use these helpers behind a knob.
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
    if op not in _INFIX_OPS and op not in _FUNCALL_OPS:
        raise NotImplementedError(f"emit_*_reduction: unsupported op {op!r}")
    if vector_width < 1:
        raise ValueError(f"emit_*_reduction: vector_width must be >= 1, got {vector_width}")


def _wrap_pair(op: str, left: str, right: str) -> str:
    """Combine two operands using ``op``.

    Infix ops produce ``(left op right)``; the ``max``/``min`` function-call
    ops produce ``op(left, right)``.
    """
    if op in _INFIX_OPS:
        return f"({left} {op} {right})"
    return f"{op}({left}, {right})"


def emit_chain_reduction(input_var: str, vector_width: int, op: str) -> str:
    """Linear left-associated chain — critical path ``O(W)``.

    Matches the form used by ``reduce_before_use`` today:
    ``a op b op c op d`` for infix ops; ``op(op(op(a, b), c), d)`` for
    function-call ops.
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
    """Balanced-tree reduction — critical path ``O(log W)``.

    Pairs lanes ``(0,1)``, ``(2,3)``, ... at each level, halving until one
    operand remains. Odd lane counts trail the lone lane forward unchanged
    until the next level pairs it up.
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
