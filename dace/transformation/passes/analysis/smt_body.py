# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Symbolic execution of a straight-line tasklet body into SMT bitvector terms.

:mod:`~dace.transformation.passes.analysis.smt_dependence` answers questions about the INDEX
domain: which elements a read and a write touch, and whether an iteration order separates them.
Some loop-carried dependences do not live there. A CRC's carried word decides its own next value,
so no subset test can see it; the question is what the body computes, not where it reads.

This module is the encoder for that second domain. It runs a tasklet body over z3 bitvectors
instead of over numbers, so one body yields either a concrete value (substitute constants) or a
formula quantified over the state (substitute variables). The consumers use it both ways -- the
same encoder extracts a candidate transition by evaluating at basis points and then discharges the
proof obligation that the candidate is exact. Extraction by evaluation alone would be inadmissible;
it is the second use that makes it sound.

Only straight-line code plus ``if``/``else`` is accepted. Branches are merged into ``z3.If`` rather
than explored, so the result is one term per assigned name whatever path the body would take. A
loop, a call or a subscript makes the whole body unsupported and the encoder returns ``None``: this
is an oracle, and refusing is always a correct answer.

The width matters. Python integers are unbounded and ``>>`` on a non-negative one is a LOGICAL
shift, so ``z3.LShR`` is the only correct encoding of ``>>`` here -- z3's ``>>`` is arithmetic and
would smear the sign bit through every CRC step. Encode wide enough that the body's own masks, not
the width, do the truncating.
"""

import ast
from typing import Any

from dace.frontend.python import astutils
from dace.optionals import required

try:
    import z3

    HAS_Z3 = True
except Exception:
    z3 = None  # type: ignore
    HAS_Z3 = False

#: Encoding width. Wide enough that a body masking to 8, 16 or 32 bits never overflows into it.
DEFAULT_WIDTH = 64

BINOPS = {
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitXor: lambda a, b: a ^ b,
    ast.LShift: lambda a, b: a << b,
    # Logical, not arithmetic: Python shifts a non-negative int in zeros, z3's ``>>`` shifts in the
    # sign bit. On a CRC the difference is every iteration after the first.
    ast.RShift: lambda a, b: required(z3).LShR(a, b),
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
}

COMPARES = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: required(z3).ULT(a, b),
    ast.LtE: lambda a, b: required(z3).ULE(a, b),
    ast.Gt: lambda a, b: required(z3).UGT(a, b),
    ast.GtE: lambda a, b: required(z3).UGE(a, b),
}


#: C semantics on two's-complement signed integers: ``>>`` copies the sign bit and comparisons are signed. A value of
#: a narrower or unsigned type is encoded extended to ``width`` (see :func:`fit`), so both agree on it.
SIGNED_BINOPS = {**BINOPS, ast.RShift: lambda a, b: a >> b}

SIGNED_COMPARES = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}

#: Integer casts a tasklet may spell, as ``dace.int64(x)`` or ``int64(x)``: name -> (bits, signed).
INTEGER_CASTS = {
    **{f"int{bits}": (bits, True) for bits in (8, 16, 32, 64)},
    **{f"uint{bits}": (bits, False) for bits in (8, 16, 32, 64)},
}


def fit(term: Any, bits: int, signed: bool, width: int) -> Any:
    """``term`` truncated to ``bits`` and extended back to ``width`` the way a ``bits``-wide value converts."""
    if bits >= width:
        return term
    low = required(z3).Extract(bits - 1, 0, term)
    return required(z3).SignExt(width - bits, low) if signed else required(z3).ZeroExt(width - bits, low)


def cast_target(node: ast.Call) -> tuple[int, bool] | None:
    """The ``(bits, signed)`` of an integer cast call, or ``None``."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    return INTEGER_CASTS.get(name) if name is not None and len(node.args) == 1 else None


def bitvec(name: str, width: int = DEFAULT_WIDTH) -> Any:
    """A fresh bitvector variable to stand for one input of the body."""
    return required(z3).BitVec(name, width)


def constant(value: int, width: int = DEFAULT_WIDTH) -> Any:
    """``value`` as a bitvector literal of ``width`` bits."""
    return required(z3).BitVecVal(value, width)


def truth(term: Any) -> Any:
    """Python truthiness of a bitvector: non-zero. A body may branch on ``crc & 1`` directly."""
    if required(z3).is_bool(term):
        return term
    return term != 0


def encode_expr(node: ast.AST, env: dict[str, Any], width: int, signed: bool = False) -> Any | None:
    """Translate one expression to a z3 term under ``env``, or ``None`` if unsupported. ``signed`` selects C
    semantics on signed integers (:data:`SIGNED_BINOPS`) over Python semantics on non-negative ones."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            return None
        return constant(node.value, width)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp):
        op = (SIGNED_BINOPS if signed else BINOPS).get(type(node.op))
        if op is None:
            return None
        left, right = encode_expr(node.left, env, width, signed), encode_expr(node.right, env, width, signed)
        if left is None or right is None:
            return None
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = encode_expr(node.operand, env, width, signed)
        if operand is None:
            return None
        if isinstance(node.op, ast.Invert):
            return ~operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return required(z3).Not(truth(operand))
        return None
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            return None
        op = (SIGNED_COMPARES if signed else COMPARES).get(type(node.ops[0]))
        if op is None:
            return None
        left, right = encode_expr(node.left, env, width, signed), encode_expr(node.comparators[0], env, width, signed)
        if left is None or right is None:
            return None
        return op(left, right)
    if isinstance(node, ast.BoolOp):
        parts = [encode_expr(v, env, width, signed) for v in node.values]
        if any(p is None for p in parts):
            return None
        joiner = required(z3).And if isinstance(node.op, ast.And) else required(z3).Or
        return joiner([truth(p) for p in parts])
    if isinstance(node, ast.Call) and signed and cast_target(node) is not None:
        operand = encode_expr(node.args[0], env, width, signed)
        bits, is_signed = required(cast_target(node))
        return None if operand is None else fit(operand, bits, is_signed, width)
    if isinstance(node, ast.IfExp):
        cond = encode_expr(node.test, env, width, signed)
        body = encode_expr(node.body, env, width, signed)
        orelse = encode_expr(node.orelse, env, width, signed)
        if cond is None or body is None or orelse is None:
            return None
        return required(z3).If(truth(cond), body, orelse)
    return None


def merge_branches(cond: Any, then_env: dict[str, Any], else_env: dict[str, Any]) -> dict[str, Any]:
    """One environment from two, selecting per name on ``cond``.

    A name assigned on only one side keeps the other side's incoming value, which is what the
    branch does; a name assigned on neither is untouched and compares equal, so ``z3.If`` folds.
    """
    merged = dict(else_env)
    for name, then_val in then_env.items():
        else_val = else_env.get(name)
        merged[name] = then_val if else_val is None else required(z3).If(cond, then_val, else_val)
    return merged


def encode_statements(body, env: dict[str, Any], width: int, signed: bool = False) -> dict[str, Any] | None:
    """Run a statement list over ``env``, returning the environment after it, or ``None``."""
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return None
            value = encode_expr(stmt.value, env, width, signed)
            if value is None:
                return None
            env = dict(env)
            env[stmt.targets[0].id] = value
            continue
        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                return None
            op = (SIGNED_BINOPS if signed else BINOPS).get(type(stmt.op))
            current = env.get(stmt.target.id)
            value = encode_expr(stmt.value, env, width, signed)
            if op is None or current is None or value is None:
                return None
            env = dict(env)
            env[stmt.target.id] = op(current, value)
            continue
        if isinstance(stmt, ast.If):
            cond = encode_expr(stmt.test, env, width, signed)
            if cond is None:
                return None
            then_env = encode_statements(stmt.body, env, width, signed)
            else_env = encode_statements(stmt.orelse, env, width, signed) if stmt.orelse else env
            if then_env is None or else_env is None:
                return None
            env = merge_branches(truth(cond), then_env, else_env)
            continue
        return None
    return env


def encode_body(
    code: str, env: dict[str, Any], width: int = DEFAULT_WIDTH, signed: bool = False
) -> dict[str, Any] | None:
    """Symbolically execute ``code`` over ``env``.

    :param code: the tasklet body, as Python source.
    :param env: input name to z3 term. Names the body reads but ``env`` lacks make it unsupported.
    :param width: bitvector width to encode in; see the module docstring.
    :param signed: C semantics on signed integers instead of Python semantics on non-negative ones.
    :returns: the environment after the body, or ``None`` if any construct is unsupported.
    """
    if not HAS_Z3:
        return None
    try:
        module = ast.parse(astutils._remove_outer_indentation(code))
    except SyntaxError:
        return None
    return encode_statements(module.body, env, width, signed)
