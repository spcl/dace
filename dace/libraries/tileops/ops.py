# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The elementwise ops of the tile library, each described once.

An op is rendered three ways: as a per-lane C++ expression by the ``pure`` expansions, as a Python expression over whole
tiles by the tile DSL backends (:mod:`dace.codegen.python`), and as one op code the ISA backends
take: the CPU tile functions (:mod:`dace.tile_functions`) name it, the SVE and CUDA headers take it as the ``Op``
template argument of ``tile_binop`` / ``tile_unop`` / ``tile_reduce``. An op with no ISA code has no vector form and lowers through the pure loop, where the compiler's
vector-math library can still vectorize its ``std::`` call.

The codes are single characters because that is what the headers switch on. The binary and the unary table are
separate namespaces, so ``'l'`` is ``<=`` in one and ``log`` in the other.
"""

from dataclasses import dataclass

import dace


@dataclass(frozen=True, slots=True)
class BinaryOp:
    """Per-lane C++ ``prefix lhs infix rhs suffix``, the op code of the ISA backends and the Python template.

    The ``python`` template formats ``{a}``, ``{b}`` and ``{f[name]}``, the function ``name`` of the tile dialect.
    """

    prefix: str
    infix: str
    suffix: str
    isa_code: str | None = None
    comparison: bool = False
    python: str = ""

    def cpp(self, lhs: str, rhs: str) -> str:
        return f"{self.prefix}{lhs}{self.infix}{rhs}{self.suffix}"


@dataclass(frozen=True, slots=True)
class UnaryOp:
    """Per-lane C++ ``prefix operand suffix``, the op code of the ISA backends and the Python template of ``{a}``."""

    prefix: str
    suffix: str
    isa_code: str | None = None
    python: str = ""

    def cpp(self, operand: str) -> str:
        return f"{self.prefix}{operand}{self.suffix}"


def infix_op(
    operator: str, isa_code: str | None = None, comparison: bool = False, python_operator: str | None = None
) -> BinaryOp:
    return BinaryOp("(", f" {operator} ", ")", isa_code, comparison, f"({{a}} {python_operator or operator} {{b}})")


def call_op(function: str, python: str, isa_code: str | None = None) -> BinaryOp:
    return BinaryOp(f"{function}(", ", ", ")", isa_code, python=python)


def std_call(function: str, isa_code: str | None = None) -> UnaryOp:
    return UnaryOp(f"std::{function}(", ")", isa_code, f"{{f[{function}]}}({{a}})")


#: ``%`` floors as Python does, built from the truncating remainder the tile DSLs share.
PYTHON_MOD = "((({a}) % ({b}) + ({b})) % ({b}))"


#: ``std::`` spells the elemental functions, whose semantics the ISA backends reproduce.
BINARY_OPS = {
    "+": infix_op("+", "+"),
    "-": infix_op("-", "-"),
    "*": infix_op("*", "*"),
    "/": infix_op("/", "/"),
    "%": call_op("py_mod", PYTHON_MOD, "p"),  # Python's ``%`` floors
    "py_mod": call_op("py_mod", PYTHON_MOD, "p"),
    "c_mod": call_op("c_mod", "{f[fmod]}({a}, {b})", "%"),  # C's modulo, which ``c_mod`` also takes on floats
    "<": infix_op("<", "<", comparison=True),
    "<=": infix_op("<=", "l", comparison=True),
    ">": infix_op(">", ">", comparison=True),
    ">=": infix_op(">=", "g", comparison=True),
    "==": infix_op("==", "=", comparison=True),
    "!=": infix_op("!=", "!", comparison=True),
    "&&": infix_op("&&", "&", python_operator="&"),
    "||": infix_op("||", "|", python_operator="|"),
    "&": infix_op("&"),
    "|": infix_op("|"),
    "^": infix_op("^"),
    "min": call_op("std::min", "{f[minimum]}({a}, {b})", "m"),
    "max": call_op("std::max", "{f[maximum]}({a}, {b})", "M"),
    # ``PowerOperatorExpansion`` rewrites a literal integer exponent above 1 to multiplies upstream, so a ``**`` that
    # arrives here has an exponent of 0 or 1, a non-integer literal or a runtime value.
    "**": call_op("std::pow", "{f[pow]}({a}, {b})"),
    "pow": call_op("std::pow", "{f[pow]}({a}, {b})"),
    # ``dace::math::ipow`` is the exact repeated multiply: bit-exact with NumPy integer powers and right for a
    # negative base, where ``std::pow`` is not.
    "ipow": call_op("dace::math::ipow", "{f[pow]}({a}, {b})"),
    "atan2": call_op("std::atan2", "{f[atan2]}({a}, {b})"),
    "hypot": call_op("std::hypot", "{f[sqrt]}({a} * {a} + {b} * {b})"),
    "fmod": call_op("std::fmod", "{f[fmod]}({a}, {b})"),
}

#: ``std::`` rather than ``dace::math::``, whose ``abs`` only has overloads for ``typeless_nan`` and unsigned integers.
UNARY_OPS = {
    "neg": UnaryOp("(-", ")", "n", "(-{a})"),
    "not": UnaryOp("(!", ")", "!", "(~{a})"),
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
    "sign_numpy_2": UnaryOp("sign_numpy_2(", ")", python="{f[where]}({a} > 0, 1, {f[where]}({a} < 0, -1, 0))"),
}

#: A unary op named after a dtype is the explicit conversion to it, spelled as the ``dace::<dtype>(x)`` cast function
#: the C++ codegen lowers a bare ``float64(x)`` to. It is the one op that may narrow.
CAST_OPS = {dtype_name.split("::")[-1]: dtype_name for dtype_name in dace.dtypes.TYPECLASS_TO_STRING.values()}

#: The reductions the ISA backends implement; their codes are the binary ops' ``+``, ``*``, ``m`` and ``M``.
REDUCE_OPS = ("+", "*", "min", "max")

#: The op codes of the ops that have one.
UNARY_ISA_CODES = {op: spec.isa_code for op, spec in UNARY_OPS.items() if spec.isa_code is not None}
