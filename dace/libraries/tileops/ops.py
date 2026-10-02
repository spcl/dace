# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The elementwise ops of the tile library, each described once.

An op is rendered two ways: as a per-lane C++ expression by the ``pure`` expansions, and as one op code the ISA headers
(``dace/tile_ops/<backend>.h``) take as the ``Op`` template argument of ``tile_binop`` / ``tile_unop`` /
``tile_reduce``. An op with no ISA code has no vector form and lowers through the pure loop, where the compiler's
vector-math library can still vectorize its ``std::`` call.

The codes are single characters because that is what the headers switch on. The binary and the unary table are
separate namespaces, so ``'l'`` is ``<=`` in one and ``log`` in the other.
"""
from dataclasses import dataclass

import dace


@dataclass(frozen=True, slots=True)
class BinaryOp:
    """Per-lane C++ ``prefix lhs infix rhs suffix`` and the op code of the ISA headers."""
    prefix: str
    infix: str
    suffix: str
    isa_code: str | None = None
    comparison: bool = False

    def cpp(self, lhs: str, rhs: str) -> str:
        return f"{self.prefix}{lhs}{self.infix}{rhs}{self.suffix}"


@dataclass(frozen=True, slots=True)
class UnaryOp:
    """Per-lane C++ ``prefix operand suffix`` and the op code of the ISA headers."""
    prefix: str
    suffix: str
    isa_code: str | None = None

    def cpp(self, operand: str) -> str:
        return f"{self.prefix}{operand}{self.suffix}"


def infix_op(operator: str, isa_code: str | None = None, comparison: bool = False) -> BinaryOp:
    return BinaryOp("(", f" {operator} ", ")", isa_code, comparison)


def call_op(function: str, isa_code: str | None = None) -> BinaryOp:
    return BinaryOp(f"{function}(", ", ", ")", isa_code)


def std_call(function: str, isa_code: str | None = None) -> UnaryOp:
    return UnaryOp(f"std::{function}(", ")", isa_code)


#: ``std::`` spells the elemental functions because the ISA headers call ``std::min`` / ``std::max`` too.
BINARY_OPS = {
    "+": infix_op("+", "+"),
    "-": infix_op("-", "-"),
    "*": infix_op("*", "*"),
    "/": infix_op("/", "/"),
    "%": call_op("py_mod", "p"),  # Python's ``%`` floors
    "py_mod": call_op("py_mod", "p"),
    "c_mod": call_op("c_mod", "%"),  # C's modulo, which ``c_mod`` also takes on floats
    "<": infix_op("<", "<", comparison=True),
    "<=": infix_op("<=", "l", comparison=True),
    ">": infix_op(">", ">", comparison=True),
    ">=": infix_op(">=", "g", comparison=True),
    "==": infix_op("==", "=", comparison=True),
    "!=": infix_op("!=", "!", comparison=True),
    "&&": infix_op("&&", "&"),
    "||": infix_op("||", "|"),
    "&": infix_op("&"),
    "|": infix_op("|"),
    "^": infix_op("^"),
    "min": call_op("std::min", "m"),
    "max": call_op("std::max", "M"),
    # ``PowerOperatorExpansion`` rewrites a literal integer exponent above 1 to multiplies upstream, so a ``**`` that
    # arrives here has an exponent of 0 or 1, a non-integer literal or a runtime value.
    "**": call_op("std::pow"),
    "pow": call_op("std::pow"),
    # ``dace::math::ipow`` is the exact repeated multiply: bit-exact with NumPy integer powers and right for a
    # negative base, where ``std::pow`` is not.
    "ipow": call_op("dace::math::ipow"),
    "atan2": call_op("std::atan2"),
    "hypot": call_op("std::hypot"),
    "fmod": call_op("std::fmod"),
}

#: ``std::`` rather than ``dace::math::``, whose ``abs`` only has overloads for ``typeless_nan`` and unsigned integers.
UNARY_OPS = {
    "neg": UnaryOp("(-", ")", "n"),
    "not": UnaryOp("(!", ")", "!"),
    "abs": std_call("abs", "a"),
    "exp": std_call("exp", "e"),
    "log": std_call("log", "l"),
    "sqrt": std_call("sqrt", "s"),
    "sin": std_call("sin", "S"),
    "cos": std_call("cos", "C"),
    "tan": std_call("tan"),
    "asin": std_call("asin"),
    "acos": std_call("acos"),
    "atan": std_call("atan"),
    "sinh": std_call("sinh"),
    "cosh": std_call("cosh"),
    "floor": std_call("floor", "f"),
    "ceil": std_call("ceil", "c"),
    "tanh": std_call("tanh", "t"),
    # numpy>=2 ``sign``, ``(0 < x) - (x < 0)``; the runtime defines the template at global scope under this name.
    "sign_numpy_2": UnaryOp("sign_numpy_2(", ")"),
}

#: A unary op named after a dtype is the explicit conversion to it, spelled as the ``dace::<dtype>(x)`` cast function
#: the C++ codegen lowers a bare ``float64(x)`` to. It is the one op that may narrow.
CAST_OPS = {dtype_name.split("::")[-1]: dtype_name for dtype_name in dace.dtypes.TYPECLASS_TO_STRING.values()}

#: The reductions the ISA headers implement; their codes are the binary ops' ``+``, ``*``, ``m`` and ``M``.
REDUCE_OPS = ("+", "*", "min", "max")

#: The op codes of the ops that have one.
UNARY_ISA_CODES = {op: spec.isa_code for op, spec in UNARY_OPS.items() if spec.isa_code is not None}
