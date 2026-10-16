# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU ISA lowering of the tile library nodes: one ``static inline`` function per signature.

A tile node expanded for ``scalar``, ``avx2``, ``avx512`` or ``neon`` calls a function whose name spells its whole
signature (``tile_binop_f64x8_add_avx512``), and :func:`definition` writes that function from the name alone: the
intrinsics directly, a ``#pragma omp simd`` lane loop for ``scalar``. The text is the common subset of C23 and C++,
so normal code generation (:func:`definitions_in`) and both CPF dialects emit the same definitions.

Every backend computes the scalar contract bit for bit: an op with no vector form for an element type runs the
per-lane loop, ``fma`` is single-rounded everywhere, ``min(a, b)`` is ``b < a ? b : a``, a producer zero-fills an
inactive lane and an array writer skips it.
"""

import re
from collections.abc import Callable
from typing import NamedTuple

from dace.optionals import required


class ElementType(NamedTuple):
    """A tile element type: the name code, its C spelling, and how it computes."""

    code: str
    c: str
    floating: bool
    signed: bool

    @property
    def math_suffix(self) -> str:
        return "f" if self.c == "float" else ""


#: The element types the functions take, by the dace ``ctype`` that names them.
ELEMENT_TYPES: dict[str, ElementType] = {
    "bool": ElementType("b8", "bool", False, False),
    "int8_t": ElementType("i8", "int8_t", False, True),
    "int16_t": ElementType("i16", "int16_t", False, True),
    "int": ElementType("i32", "int32_t", False, True),
    "int64_t": ElementType("i64", "int64_t", False, True),
    "uint8_t": ElementType("u8", "uint8_t", False, False),
    "uint16_t": ElementType("u16", "uint16_t", False, False),
    "dace::uint": ElementType("u32", "uint32_t", False, False),
    "uint64_t": ElementType("u64", "uint64_t", False, False),
    "float": ElementType("f32", "float", True, True),
    "double": ElementType("f64", "double", True, True),
}
BY_CODE: dict[str, ElementType] = {t.code: t for t in ELEMENT_TYPES.values()}

#: The CPU backends, by the implementation name of the tile nodes.
ISAS = ("scalar", "avx2", "avx512", "neon")

#: Op code (as :mod:`dace.libraries.tileops.ops` gives it) -> name in a function name.
BINARY_NAMES = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "cmod",
    "p": "pymod",
    "m": "min",
    "M": "max",
    "<": "lt",
    "l": "le",
    ">": "gt",
    "g": "ge",
    "=": "eq",
    "!": "ne",
    "&": "and",
    "|": "or",
}
UNARY_NAMES = {
    "n": "neg",
    "!": "not",
    "a": "abs",
    "e": "exp",
    "l": "log",
    "s": "sqrt",
    "S": "sin",
    "C": "cos",
    "f": "floor",
    "c": "ceil",
    "t": "tanh",
}
UNARY_MATH = {"e": "exp", "l": "log", "s": "sqrt", "S": "sin", "C": "cos", "f": "floor", "c": "ceil", "t": "tanh"}

#: Kind -> its op table and how many value operands carry a broadcast flag.
KINDS: dict[str, tuple[dict[str, str], int]] = {
    "binop": (BINARY_NAMES, 2),
    "unop": (UNARY_NAMES, 1),
    "fma": ({}, 3),
    "ite": ({}, 2),
    "reduce": (BINARY_NAMES, 0),
    "gather": ({}, 0),
    "maskgen": ({}, 0),
    "load": ({}, 0),
    "store": ({}, 0),
}

#: One call of any tile function, as :func:`definitions_in` and the CPF tables find it in emitted code.
CALL_NAMES = rf"tile_(?:{'|'.join(KINDS)})_\w+?_(?:{'|'.join(ISAS)})"
CALL = re.compile(rf"(?<![\w:.])({CALL_NAMES})\s*\(")


class TileFunction(NamedTuple):
    """One signature: ``kind`` over ``vlen`` lanes of ``element`` (a type code) on ``isa``.

    ``broadcast`` holds ``v`` (a tile) or ``b`` (one value read for every lane) per value operand, ``aux`` the type
    code of an ITE condition or a gather index.
    """

    kind: str
    isa: str
    element: str
    vlen: int
    op: str = ""
    broadcast: str = ""
    masked: bool = False
    aux: str = ""

    @property
    def name(self) -> str:
        parts = ["tile", self.kind, f"{self.element}x{self.vlen}"]
        if self.aux:
            parts.append(self.aux)
        if self.op:
            parts.append(KINDS[self.kind][0][self.op])
        if "b" in self.broadcast:
            parts.append(self.broadcast)
        if self.masked:
            parts.append("m")
        return "_".join([*parts, self.isa])

    @classmethod
    def parse(cls, name: str) -> "TileFunction | None":
        tokens = name.split("_")
        if len(tokens) < 4 or tokens[0] != "tile" or tokens[1] not in KINDS or tokens[-1] not in ISAS:
            return None
        lanes = re.fullmatch(r"([a-z]\d+)x(\d+)", tokens[2])
        if lanes is None or lanes.group(1) not in BY_CODE:
            return None
        ops, operands = KINDS[tokens[1]]
        op_codes = {spelled: code for code, spelled in ops.items()}
        fields: dict[str, object] = {"broadcast": "v" * operands}
        for token in tokens[3:-1]:
            if token in BY_CODE:
                fields["aux"] = token
            elif token in op_codes:
                fields["op"] = op_codes[token]
            elif token == "m":
                fields["masked"] = True
            elif re.fullmatch(r"[vb]+", token):
                fields["broadcast"] = token
            else:
                return None
        function = cls(tokens[1], tokens[-1], lanes.group(1), int(lanes.group(2)), **fields)
        return function if function.name == name and function.valid() else None

    def valid(self) -> bool:
        ops, operands = KINDS[self.kind]
        return (
            len(self.broadcast) == operands
            and (self.op in ops) == bool(ops)
            and (self.kind in ("ite", "gather")) == bool(self.aux)
            and self.vlen > 0
        )


def includes_name(isa: str) -> str:
    return f"tile_includes_{isa}"


def pymod_name(element: ElementType) -> str:
    return f"tile_pymod_{element.code}"


def definition(name: str) -> tuple[str, tuple[str, ...]] | None:
    """``(definition, names it calls)`` of a tile function, its ``#include`` block or a per-lane helper; ``None`` for
    any other name."""
    for isa, headers in INCLUDES.items():
        if name == includes_name(isa):
            return "\n".join(f"#include <{header}>" for header in headers), ()
    for element in ELEMENT_TYPES.values():
        if name == pymod_name(element):
            return pymod_definition(element), (includes_name("scalar"),)
    function = TileFunction.parse(name)
    if function is None:
        return None
    dependencies = [includes_name(function.isa)]
    if function.op == "p" and function.kind in ("binop", "reduce"):
        dependencies.append(pymod_name(BY_CODE[function.element]))
    return GENERATORS[function.kind](function), tuple(dependencies)


def definitions_in(code: str) -> str:
    """The definitions every tile function ``code`` calls needs, callees first, each once."""
    # A name shaped like a tile function that does not parse is someone else's (a nested SDFG's function).
    pending = sorted({match.group(1) for match in CALL.finditer(code) if definition(match.group(1)) is not None})
    ordered: list[str] = []
    placed: set[str] = set()

    def place(name: str) -> None:
        if name in placed:
            return
        found = required(definition(name))
        placed.add(name)
        for dependency in found[1]:
            place(dependency)
        ordered.append(found[0])

    for name in pending:
        place(name)
    return "".join(f"{text}\n" for text in ordered)


INCLUDES = {
    "scalar": ("math.h", "stdint.h"),
    "avx2": ("immintrin.h", "math.h", "stdint.h"),
    "avx512": ("immintrin.h", "math.h", "stdint.h"),
    "neon": ("arm_neon.h", "math.h", "stdint.h"),
}

#: What a backend's functions need beyond the flags of the translation unit: CPF builds without ``-m`` flags.
TARGET_ATTRIBUTES = {"avx2": '__attribute__((target("avx2"))) ', "avx512": '__attribute__((target("avx512f"))) '}


def pymod_definition(element: ElementType) -> str:
    """Python's ``%`` on one lane, the remainder half of the runtime's ``py_divmod``."""
    t, name = element.c, pymod_name(element)
    if element.floating:
        s = element.math_suffix
        body = (
            f"    {t} r = fmod{s}(a, b);\n"
            "    if (b == 0) return r;\n"
            f"    if (r == 0) return copysign{s}(({t})0, b);\n"
            "    if ((b < 0) != (r < 0)) r += b;\n"
            "    return r;\n"
        )
    elif element.signed:
        body = (
            f"    if (b == 0 || b == ({t})-1) return 0;\n"
            f"    {t} r = ({t})(a % b);\n"
            f"    if (r != 0 && ((r < 0) != (b < 0))) r = ({t})(r + b);\n"
            "    return r;\n"
        )
    else:
        body = f"    if (b == 0) return 0;\n    return ({t})(a % b);\n"
    return f"static inline {t} {name}({t} a, {t} b) {{\n{body}}}"


def binary(op: str, element: ElementType, a: str, b: str) -> str:
    """One lane of a binary op, as the runtime's ``tile_apply`` computes it."""
    t = element.c
    if op in "+-*/":
        return f"({t})({a} {op} {b})"
    if op == "%":
        return f"({t})fmod{element.math_suffix}({a}, {b})" if element.floating else f"({t})({a} % {b})"
    if op == "p":
        return f"{pymod_name(element)}({a}, {b})"
    if op == "m":
        return f"(({b}) < ({a}) ? ({b}) : ({a}))"
    if op == "M":
        return f"(({a}) < ({b}) ? ({b}) : ({a}))"
    relation = {"<": "<", "l": "<=", ">": ">", "g": ">=", "=": "==", "!": "!=", "&": "&&", "|": "||"}[op]
    return f"(({a}) {relation} ({b}) ? ({t})1 : ({t})0)"


def unary(op: str, element: ElementType, a: str) -> str:
    t = element.c
    if op == "n":
        return f"({t})(-{a})"
    if op == "!":
        return f"({t})(!{a})"
    if op == "a":
        if element.floating:
            return f"fabs{element.math_suffix}({a})"
        return f"({t})({a} < 0 ? -{a} : {a})" if element.signed else a
    function = UNARY_MATH[op]
    return f"{function}{element.math_suffix}({a})" if element.floating else f"({t}){function}((double)({a}))"


def fused(element: ElementType, a: str, b: str, c: str) -> str:
    if element.floating:
        return f"fma{element.math_suffix}({a}, {b}, {c})"
    return f"({element.c})fma((double)({a}), (double)({b}), (double)({c}))"


def lane(name: str, flag: str, index: str = "i") -> str:
    return f"{name}[0]" if flag == "b" else f"{name}[{index}]"


def produced(function: TileFunction, value: str) -> str:
    """A producer's lane: zero where the mask is off."""
    return f"mask[i] ? {value} : ({BY_CODE[function.element].c})0" if function.masked else value


def header(function: TileFunction, result: str, parameters: list[str]) -> str:
    return f"static inline {TARGET_ATTRIBUTES.get(function.isa, '')}{result} {function.name}({', '.join(parameters)})"


def value_parameters(function: TileFunction, names: str) -> list[str]:
    t = BY_CODE[function.element].c
    parameters = [f"{t} *__restrict__ out"] + [f"const {t} *__restrict__ {name}" for name in names]
    return parameters + (["const bool *__restrict__ mask"] if function.masked else [])


def scalar_loop(function: TileFunction, statement: str, tail: bool = False, simd: bool = False) -> str:
    """The lane loop over the whole tile, or over the lanes from ``i`` the register chunks left (``tail``)."""
    pragma = "    #pragma omp simd\n" if simd else ""
    start = "" if tail else "int64_t i = 0"
    return f"{pragma}    for ({start}; i < {function.vlen}; ++i) {{ {statement} }}\n"


class Vector(NamedTuple):
    """One register of an ISA for one element type; templates take ``{p}`` (pointer), ``{v}``, ``{x}``."""

    width: int
    type: str
    load: str
    store: str
    splat: str
    zero_fill: Callable[[str], tuple[str, str]]


def avx512_mask(width: int) -> str:
    mask = f"__mmask{width}"
    return (
        f"        {mask} k = 0;\n        for (int j = 0; j < {width}; ++j) if (mask[i + j]) k |= ({mask})(1u << j);\n"
    )


def avx2_mask(lane_bits: int, width: int) -> str:
    return (
        f"        int{lane_bits}_t mb[{width}];\n"
        f"        for (int j = 0; j < {width}; ++j) mb[j] = mask[i + j] ? -1 : 0;\n"
        "        const __m256i vm = _mm256_loadu_si256((const __m256i *)mb);\n"
    )


def neon_mask(lane_bits: int, width: int) -> str:
    return (
        f"        uint{lane_bits}_t mb[{width}];\n"
        f"        for (int j = 0; j < {width}; ++j) mb[j] = mask[i + j] ? (uint{lane_bits}_t)-1 : 0;\n"
        f"        const uint{lane_bits}x{width}_t vm = vld1q_u{lane_bits}(mb);\n"
    )


def avx512_vector(suffix: str, width: int) -> Vector:
    kind = "__m512" if suffix == "ps" else "__m512d"
    return Vector(
        width,
        kind,
        f"_mm512_loadu_{suffix}({{p}})",
        f"_mm512_storeu_{suffix}({{p}}, {{v}})",
        f"_mm512_set1_{suffix}({{x}})",
        lambda r: (avx512_mask(width), f"_mm512_maskz_mov_{suffix}(k, {r})"),
    )


def avx2_float_vector(suffix: str, width: int) -> Vector:
    lane_bits = 32 if suffix == "ps" else 64
    return Vector(
        width,
        "__m256" if suffix == "ps" else "__m256d",
        f"_mm256_loadu_{suffix}({{p}})",
        f"_mm256_storeu_{suffix}({{p}}, {{v}})",
        f"_mm256_set1_{suffix}({{x}})",
        lambda r: (
            avx2_mask(lane_bits, width),
            f"_mm256_blendv_{suffix}(_mm256_setzero_{suffix}(), {r}, _mm256_castsi256_{suffix}(vm))",
        ),
    )


def avx2_integer_vector(lane_bits: int) -> Vector:
    width = 256 // lane_bits
    splat = "_mm256_set1_epi32({x})" if lane_bits == 32 else "_mm256_set1_epi64x({x})"
    return Vector(
        width,
        "__m256i",
        "_mm256_loadu_si256((const __m256i *)({p}))",
        "_mm256_storeu_si256((__m256i *)({p}), {v})",
        splat,
        lambda r: (avx2_mask(lane_bits, width), f"_mm256_blendv_epi8(_mm256_setzero_si256(), {r}, vm)"),
    )


def neon_vector(suffix: str, lane_bits: int) -> Vector:
    width = 128 // lane_bits
    zero = "0.0f" if suffix == "f32" else "0.0" if suffix == "f64" else "0"
    kind = {"f32": "float32x4_t", "f64": "float64x2_t", "s32": "int32x4_t", "s64": "int64x2_t"}[suffix]
    return Vector(
        width,
        kind,
        f"vld1q_{suffix}({{p}})",
        f"vst1q_{suffix}({{p}}, {{v}})",
        f"vdupq_n_{suffix}({{x}})",
        lambda r: (neon_mask(lane_bits, width), f"vbslq_{suffix}(vm, {r}, vdupq_n_{suffix}({zero}))"),
    )


VECTORS: dict[tuple[str, str], Vector] = {
    ("avx512", "f32"): avx512_vector("ps", 16),
    ("avx512", "f64"): avx512_vector("pd", 8),
    ("avx2", "f32"): avx2_float_vector("ps", 8),
    ("avx2", "f64"): avx2_float_vector("pd", 4),
    ("avx2", "i32"): avx2_integer_vector(32),
    ("avx2", "i64"): avx2_integer_vector(64),
    ("neon", "f32"): neon_vector("f32", 32),
    ("neon", "f64"): neon_vector("f64", 64),
    ("neon", "i32"): neon_vector("s32", 32),
    ("neon", "i64"): neon_vector("s64", 64),
}


def avx2_float_binary(suffix: str) -> dict[str, str]:
    one = f"_mm256_set1_{suffix}(1.0)"
    compare = {"<": "_CMP_LT_OQ", "l": "_CMP_LE_OQ", ">": "_CMP_GT_OQ", "g": "_CMP_GE_OQ", "=": "_CMP_EQ_OQ"}
    compare["!"] = "_CMP_NEQ_UQ"
    ops = {
        "+": f"_mm256_add_{suffix}({{a}}, {{b}})",
        "-": f"_mm256_sub_{suffix}({{a}}, {{b}})",
        "*": f"_mm256_mul_{suffix}({{a}}, {{b}})",
        "/": f"_mm256_div_{suffix}({{a}}, {{b}})",
        # MINPS is (x < y) ? x : y, so the second operand wins on NaN and on signed zero: swapped to match ``min``.
        "m": f"_mm256_min_{suffix}({{b}}, {{a}})",
        "M": f"_mm256_max_{suffix}({{b}}, {{a}})",
    }
    for op, predicate in compare.items():
        ops[op] = f"_mm256_and_{suffix}(_mm256_cmp_{suffix}({{a}}, {{b}}, {predicate}), {one})"
    for op, combine in (("&", "and"), ("|", "or")):
        nonzero = f"_mm256_cmp_{suffix}({{x}}, _mm256_setzero_{suffix}(), _CMP_NEQ_UQ)"
        ops[op] = (
            f"_mm256_and_{suffix}(_mm256_{combine}_{suffix}({nonzero.format(x='{a}')}, {nonzero.format(x='{b}')}), "
            f"{one})"
        )
    return ops


def avx2_int32_binary() -> dict[str, str]:
    one, ones = "_mm256_set1_epi32(1)", "_mm256_set1_epi32(-1)"
    ops = {
        "+": "_mm256_add_epi32({a}, {b})",
        "-": "_mm256_sub_epi32({a}, {b})",
        "*": "_mm256_mullo_epi32({a}, {b})",
        "m": "_mm256_min_epi32({a}, {b})",
        "M": "_mm256_max_epi32({a}, {b})",
        "=": f"_mm256_and_si256(_mm256_cmpeq_epi32({{a}}, {{b}}), {one})",
        ">": f"_mm256_and_si256(_mm256_cmpgt_epi32({{a}}, {{b}}), {one})",
        "<": f"_mm256_and_si256(_mm256_cmpgt_epi32({{b}}, {{a}}), {one})",
        "g": f"_mm256_and_si256(_mm256_andnot_si256(_mm256_cmpgt_epi32({{b}}, {{a}}), {ones}), {one})",
        "l": f"_mm256_and_si256(_mm256_andnot_si256(_mm256_cmpgt_epi32({{a}}, {{b}}), {ones}), {one})",
        "!": f"_mm256_and_si256(_mm256_andnot_si256(_mm256_cmpeq_epi32({{a}}, {{b}}), {ones}), {one})",
    }
    nonzero = f"_mm256_andnot_si256(_mm256_cmpeq_epi32({{x}}, _mm256_setzero_si256()), {ones})"
    for op, combine in (("&", "and"), ("|", "or")):
        ops[op] = (
            f"_mm256_and_si256(_mm256_{combine}_si256({nonzero.format(x='{a}')}, {nonzero.format(x='{b}')}), {one})"
        )
    return ops


def neon_binary(suffix: str, lane_bits: int, ops: str) -> dict[str, str]:
    unsigned = f"u{lane_bits}"
    one_zero = (
        f"vbslq_{suffix}({{m}}, vdupq_n_{suffix}(1.0), vdupq_n_{suffix}(0.0))"
        if suffix.startswith("f")
        else f"vandq_{suffix}(vreinterpretq_{suffix}_{unsigned}({{m}}), vdupq_n_{suffix}(1))"
    )
    not_equal = (
        f"vmvnq_u32(vceqq_{suffix}({{a}}, {{b}}))"
        if lane_bits == 32
        else f"veorq_u64(vceqq_{suffix}({{a}}, {{b}}), vdupq_n_u64(~0ull))"
    )
    table = {
        "+": f"vaddq_{suffix}({{a}}, {{b}})",
        "-": f"vsubq_{suffix}({{a}}, {{b}})",
        "*": f"vmulq_{suffix}({{a}}, {{b}})",
        "/": f"vdivq_{suffix}({{a}}, {{b}})",
        # FMIN propagates NaN and picks -0: only a compare-select reproduces ``min``.
        "m": f"vbslq_{suffix}(vcltq_{suffix}({{b}}, {{a}}), {{b}}, {{a}})",
        "M": f"vbslq_{suffix}(vcltq_{suffix}({{a}}, {{b}}), {{b}}, {{a}})",
        "<": one_zero.format(m=f"vcltq_{suffix}({{a}}, {{b}})"),
        "l": one_zero.format(m=f"vcleq_{suffix}({{a}}, {{b}})"),
        ">": one_zero.format(m=f"vcgtq_{suffix}({{a}}, {{b}})"),
        "g": one_zero.format(m=f"vcgeq_{suffix}({{a}}, {{b}})"),
        "=": one_zero.format(m=f"vceqq_{suffix}({{a}}, {{b}})"),
        "!": one_zero.format(m=not_equal),
    }
    if suffix.startswith("s"):
        table["m"] = f"vminq_{suffix}({{a}}, {{b}})"
        table["M"] = f"vmaxq_{suffix}({{a}}, {{b}})"
    return {op: table[op] for op in ops}


#: ``(isa, element) -> op code -> vector expression over {a}, {b}``; a missing op runs per lane.
BINARY_VECTOR: dict[tuple[str, str], dict[str, str]] = {
    ("avx512", "f32"): {
        "+": "_mm512_add_ps({a}, {b})",
        "-": "_mm512_sub_ps({a}, {b})",
        "*": "_mm512_mul_ps({a}, {b})",
        "/": "_mm512_div_ps({a}, {b})",
        "m": "_mm512_min_ps({b}, {a})",
        "M": "_mm512_max_ps({b}, {a})",
    },
    ("avx512", "f64"): {
        "+": "_mm512_add_pd({a}, {b})",
        "-": "_mm512_sub_pd({a}, {b})",
        "*": "_mm512_mul_pd({a}, {b})",
        "/": "_mm512_div_pd({a}, {b})",
        "m": "_mm512_min_pd({b}, {a})",
        "M": "_mm512_max_pd({b}, {a})",
    },
    ("avx2", "f32"): avx2_float_binary("ps"),
    ("avx2", "f64"): avx2_float_binary("pd"),
    ("avx2", "i32"): avx2_int32_binary(),
    ("neon", "f32"): neon_binary("f32", 32, "+-*/mM<l>g=!"),
    ("neon", "f64"): neon_binary("f64", 64, "+-*/mM<l>g=!"),
    ("neon", "i32"): neon_binary("s32", 32, "+-*mM<l>g=!"),
    ("neon", "i64"): neon_binary("s64", 64, "+-<l>g=!"),
}

FMA_VECTOR = {
    ("avx512", "f32"): "_mm512_fmadd_ps({a}, {b}, {c})",
    ("avx512", "f64"): "_mm512_fmadd_pd({a}, {b}, {c})",
    ("avx2", "f32"): "_mm256_fmadd_ps({a}, {b}, {c})",
    ("avx2", "f64"): "_mm256_fmadd_pd({a}, {b}, {c})",
    ("neon", "f32"): "vfmaq_f32({c}, {a}, {b})",
    ("neon", "f64"): "vfmaq_f64({c}, {a}, {b})",
}


def vector_operand(vector: Vector, name: str, flag: str) -> str:
    return vector.splat.format(x=f"{name}[0]") if flag == "b" else vector.load.format(p=f"{name} + i")


def chunk_loop(function: TileFunction, vector: Vector, result: str, prelude: str = "") -> str:
    """The register-wide steps over the whole chunks of the tile, writing ``result`` (zero-filled when masked)."""
    w = vector.width
    if function.masked:
        mask, result = vector.zero_fill(result)
        prelude += mask
    return (
        f"    for (; i + {w} <= {function.vlen}; i += {w}) {{\n{prelude}"
        f"        {vector.store.format(p='out + i', v=result)};\n    }}\n"
    )


def elementwise(
    function: TileFunction, names: str, value: str, vector_result: str | None, guard: str | None = None
) -> str:
    """A producer over ``names``: register chunks where ``vector_result`` exists, the lane loop for the rest."""
    element = BY_CODE[function.element]
    statement = f"out[i] = {produced(function, value)};"
    head = header(function, "void", value_parameters(function, names))
    if function.isa == "scalar":
        return f"{head} {{\n{scalar_loop(function, statement, simd=True)}}}"
    vector = VECTORS.get((function.isa, element.code))
    body = "    int64_t i = 0;\n"
    if vector is not None and vector_result is not None:
        operands = {
            name: vector_operand(vector, name, flag) for name, flag in zip(names, function.broadcast, strict=True)
        }
        chunks = chunk_loop(function, vector, vector_result.format(**operands))
        body += f"#if {guard}\n{chunks}#endif\n" if guard else chunks
    return f"{head} {{\n{body}{scalar_loop(function, statement, tail=True)}}}"


def binop_function(function: TileFunction) -> str:
    element = BY_CODE[function.element]
    a, b = (lane(name, flag) for name, flag in zip("ab", function.broadcast, strict=True))
    vector = BINARY_VECTOR.get((function.isa, element.code), {}).get(function.op)
    return elementwise(function, "ab", binary(function.op, element, a, b), vector)


def unop_function(function: TileFunction) -> str:
    element = BY_CODE[function.element]
    value = unary(function.op, element, lane("a", function.broadcast))
    statement = f"out[i] = {produced(function, value)};"
    return f"{header(function, 'void', value_parameters(function, 'a'))} {{\n{scalar_loop(function, statement, simd=True)}}}"


def fma_function(function: TileFunction) -> str:
    element = BY_CODE[function.element]
    a, b, c = (lane(name, flag) for name, flag in zip("abc", function.broadcast, strict=True))
    vector = FMA_VECTOR.get((function.isa, element.code))
    # The AVX2 FMA needs ``-mfma``; without it the lane loop's ``fma`` computes the same bits.
    guard = "defined(__FMA__)" if function.isa == "avx2" else None
    return elementwise(function, "abc", fused(element, a, b, c), vector, guard)


def ite_function(function: TileFunction) -> str:
    element, condition = BY_CODE[function.element], BY_CODE[function.aux]
    then_flag, else_flag = function.broadcast
    value = f"(cond[i] ? {lane('t', then_flag)} : {lane('e', else_flag)})"
    statement = f"out[i] = {produced(function, value)};"
    parameters = value_parameters(function, "te")
    parameters.insert(1, f"const {condition.c} *__restrict__ cond")
    head = header(function, "void", parameters)
    if function.isa == "scalar":
        return f"{head} {{\n{scalar_loop(function, statement, simd=True)}}}"
    vector = VECTORS.get((function.isa, element.code))
    body = "    int64_t i = 0;\n"
    if vector is not None and function.isa == "avx2":
        lane_bits = 32 if element.code in ("f32", "i32") else 64
        w = vector.width
        prelude = (
            f"        int{lane_bits}_t cb[{w}];\n"
            f"        for (int j = 0; j < {w}; ++j) cb[j] = cond[i + j] ? -1 : 0;\n"
            "        const __m256i cm = _mm256_loadu_si256((const __m256i *)cb);\n"
        )
        vt, ve = vector_operand(vector, "t", then_flag), vector_operand(vector, "e", else_flag)
        if element.floating:
            s = "ps" if element.code == "f32" else "pd"
            result = f"_mm256_blendv_{s}({ve}, {vt}, _mm256_castsi256_{s}(cm))"
        else:
            result = f"_mm256_blendv_epi8({ve}, {vt}, cm)"
        body += chunk_loop(function, vector, result, prelude)
    elif vector is not None and function.isa == "neon" and condition == element and function.broadcast == "vv":
        suffix = vector.load.split("_")[1].split("(")[0]
        zero = {"f32": "0.0f", "f64": "0.0"}.get(suffix, "0")
        equal = f"vceqq_{suffix}({vector.load.format(p='cond + i')}, vdupq_n_{suffix}({zero}))"
        nonzero = f"vmvnq_u32({equal})" if vector.width == 4 else f"veorq_u64({equal}, vdupq_n_u64(~0ull))"
        result = f"vbslq_{suffix}({nonzero}, {vector.load.format(p='t + i')}, {vector.load.format(p='e + i')})"
        body += chunk_loop(function, vector, result)
    return f"{head} {{\n{body}{scalar_loop(function, statement, tail=True)}}}"


def reduce_function(function: TileFunction) -> str:
    element = BY_CODE[function.element]
    t, v = element.c, function.vlen
    combine = binary(function.op, element, "buf[2 * i]", "buf[2 * i + 1]")
    # The balanced pairwise tree the vectorized ``Reduce`` node folds with, on every backend.
    return (
        f"{header(function, t, [f'const {t} *__restrict__ src'])} {{\n"
        f"    {t} buf[{v}];\n"
        f"    for (int64_t i = 0; i < {v}; ++i) buf[i] = src[i];\n"
        f"    int64_t n = {v};\n"
        "    while (n > 1) {\n"
        "        const int64_t half = n / 2;\n"
        f"        for (int64_t i = 0; i < half; ++i) buf[i] = {combine};\n"
        "        if (n & 1) buf[half] = buf[n - 1];\n"
        "        n = half + (n & 1);\n"
        "    }\n"
        "    return buf[0];\n"
        "}"
    )


def copy_parameters(function: TileFunction) -> list[str]:
    t = BY_CODE[function.element].c
    parameters = [f"{t} *__restrict__ dst", f"const {t} *__restrict__ src"]
    return parameters + (["const bool *__restrict__ mask"] if function.masked else []) + ["int64_t stride"]


def load_function(function: TileFunction) -> str:
    element = BY_CODE[function.element]
    t = element.c
    read = f"mask[i] ? src[i * stride] : ({t})0" if function.masked else "src[i * stride]"
    statement = f"dst[i] = {read};"
    head = header(function, "void", copy_parameters(function))
    if function.isa == "scalar":
        return f"{head} {{\n{scalar_loop(function, statement, simd=True)}}}"
    vector = VECTORS.get((function.isa, element.code))
    unit = ""
    w = 0 if vector is None else vector.width
    if function.isa == "avx512" and element.code in ("f32", "f64"):
        s = "ps" if element.code == "f32" else "pd"
        assert vector is not None
        if function.masked:
            step = f"{avx512_mask(w)}        _mm512_storeu_{s}(dst + i, _mm512_maskz_loadu_{s}(k, src + i));\n"
        else:
            step = f"        _mm512_storeu_{s}(dst + i, _mm512_loadu_{s}(src + i));\n"
        unit = f"    for (; i + {w} <= {function.vlen}; i += {w}) {{\n{step}    }}\n"
        if element.code == "f64":
            lanes = ", ".join(f"(long long)((i + {j}) * stride)" for j in reversed(range(8)))
            gather = (
                f"{avx512_mask(8)}        _mm512_storeu_pd(dst + i, "
                "_mm512_mask_i64gather_pd(_mm512_setzero_pd(), k, vindex, src, 8));\n"
                if function.masked
                else "        _mm512_storeu_pd(dst + i, _mm512_i64gather_pd(vindex, src, 8));\n"
            )
            strided = (
                f"    for (; i + 8 <= {function.vlen}; i += 8) {{\n"
                f"        const __m512i vindex = _mm512_set_epi64({lanes});\n{gather}    }}\n"
            )
            return (
                f"{head} {{\n    int64_t i = 0;\n    if (stride == 1) {{\n{indent(unit)}    }} else {{\n"
                f"{indent(strided)}    }}\n{scalar_loop(function, statement, tail=True)}}}"
            )
    elif function.isa == "avx2" and vector is not None:
        lane_bits = 32 if element.code in ("f32", "i32") else 64
        if function.masked:
            masked_load = {
                "f32": "_mm256_maskload_ps(src + i, vm)",
                "f64": "_mm256_maskload_pd(src + i, vm)",
                "i32": "_mm256_maskload_epi32((const int *)(src + i), vm)",
                "i64": "_mm256_maskload_epi64((const long long *)(src + i), vm)",
            }[element.code]
            step = f"{avx2_mask(lane_bits, w)}        {vector.store.format(p='dst + i', v=masked_load)};\n"
        else:
            step = f"        {vector.store.format(p='dst + i', v=vector.load.format(p='src + i'))};\n"
        unit = f"    for (; i + {w} <= {function.vlen}; i += {w}) {{\n{step}    }}\n"
    elif function.isa == "neon" and vector is not None and not function.masked:
        step = f"        {vector.store.format(p='dst + i', v=vector.load.format(p='src + i'))};\n"
        unit = f"    for (; i + {w} <= {function.vlen}; i += {w}) {{\n{step}    }}\n"
    if not unit:
        return f"{head} {{\n{scalar_loop(function, statement)}}}"
    return (
        f"{head} {{\n    int64_t i = 0;\n    if (stride == 1) {{\n{indent(unit)}    }}\n"
        f"{scalar_loop(function, statement, tail=True)}}}"
    )


def store_function(function: TileFunction) -> str:
    element = BY_CODE[function.element]
    statement = "if (mask[i]) dst[i * stride] = src[i];" if function.masked else "dst[i * stride] = src[i];"
    head = header(function, "void", copy_parameters(function))
    if function.isa == "scalar":
        return f"{head} {{\n{scalar_loop(function, statement, simd=True)}}}"
    vector = VECTORS.get((function.isa, element.code))
    step = ""
    if function.isa == "avx2" and vector is not None:
        lane_bits = 32 if element.code in ("f32", "i32") else 64
        value = vector.load.format(p="src + i")
        if function.masked:
            masked_store = {
                "f32": "_mm256_maskstore_ps(dst + i, vm, {v})",
                "f64": "_mm256_maskstore_pd(dst + i, vm, {v})",
                "i32": "_mm256_maskstore_epi32((int *)(dst + i), vm, {v})",
                "i64": "_mm256_maskstore_epi64((long long *)(dst + i), vm, {v})",
            }[element.code]
            step = f"{avx2_mask(lane_bits, vector.width)}        {masked_store.format(v=value)};\n"
        else:
            step = f"        {vector.store.format(p='dst + i', v=value)};\n"
    elif function.isa == "neon" and vector is not None and not function.masked:
        step = f"        {vector.store.format(p='dst + i', v=vector.load.format(p='src + i'))};\n"
    if not step:
        return f"{head} {{\n{scalar_loop(function, statement)}}}"
    assert vector is not None
    w = vector.width
    unit = f"    for (; i + {w} <= {function.vlen}; i += {w}) {{\n{step}    }}\n"
    return (
        f"{head} {{\n    int64_t i = 0;\n    if (stride == 1) {{\n{indent(unit)}    }}\n"
        f"{scalar_loop(function, statement, tail=True)}}}"
    )


def gather_function(function: TileFunction) -> str:
    element, index = BY_CODE[function.element], BY_CODE[function.aux]
    t = element.c
    read = f"mask[i] ? src[idx[i]] : ({t})0" if function.masked else "src[idx[i]]"
    statement = f"dst[i] = {read};"
    parameters = [f"{t} *__restrict__ dst", f"const {t} *__restrict__ src", f"const {index.c} *__restrict__ idx"]
    head = header(function, "void", parameters + (["const bool *__restrict__ mask"] if function.masked else []))
    step = ""
    w = 0
    if function.isa == "avx512" and element.code == "f64" and index.code in ("i64", "u64"):
        w = 8
        load = "        const __m512i vindex = _mm512_loadu_si512((const void *)(idx + i));\n"
        if function.masked:
            step = (
                f"{load}{avx512_mask(8)}        _mm512_storeu_pd(dst + i, "
                "_mm512_mask_i64gather_pd(_mm512_setzero_pd(), k, vindex, src, 8));\n"
            )
        else:
            step = f"{load}        _mm512_storeu_pd(dst + i, _mm512_i64gather_pd(vindex, src, 8));\n"
    elif function.isa == "avx2" and element.code in ("f32", "f64", "i32", "i64"):
        vector = VECTORS[("avx2", element.code)]
        w = vector.width
        bits = 64 if element.code == "i64" else 32
        load = (
            "_mm_loadu_si128((const __m128i *)ib)"
            if element.code == "f64"
            else "_mm256_loadu_si256((const __m256i *)ib)"
        )
        index_type = "__m128i" if element.code == "f64" else "__m256i"
        prelude = (
            f"        int{bits}_t ib[{w}];\n"
            f"        for (int j = 0; j < {w}; ++j) ib[j] = (int{bits}_t)idx[i + j];\n"
            f"        const {index_type} vindex = {load};\n"
        )
        gathered, masked_gather, mask_vector = {
            "f32": (
                "_mm256_i32gather_ps(src, vindex, 4)",
                "_mm256_mask_i32gather_ps(_mm256_setzero_ps(), src, vindex, {m}, 4)",
                "_mm256_castsi256_ps(vm)",
            ),
            "f64": (
                "_mm256_i32gather_pd(src, vindex, 8)",
                "_mm256_mask_i32gather_pd(_mm256_setzero_pd(), src, vindex, {m}, 8)",
                "_mm256_castsi256_pd(vm)",
            ),
            "i32": (
                "_mm256_i32gather_epi32((const int *)src, vindex, 4)",
                "_mm256_mask_i32gather_epi32(_mm256_setzero_si256(), (const int *)src, vindex, {m}, 4)",
                "vm",
            ),
            "i64": (
                "_mm256_i64gather_epi64((const long long *)src, vindex, 8)",
                "_mm256_mask_i64gather_epi64(_mm256_setzero_si256(), (const long long *)src, vindex, {m}, 8)",
                "vm",
            ),
        }[element.code]
        if function.masked:
            prelude += avx2_mask(32 if element.code in ("f32", "i32") else 64, w)
            gathered = masked_gather.format(m=mask_vector)
        step = f"{prelude}        {vector.store.format(p='dst + i', v=gathered)};\n"
    if not step:
        return f"{head} {{\n{scalar_loop(function, statement)}}}"
    return (
        f"{head} {{\n    int64_t i = 0;\n    for (; i + {w} <= {function.vlen}; i += {w}) {{\n{step}    }}\n"
        f"{scalar_loop(function, statement, tail=True)}}}"
    )


def maskgen_function(function: TileFunction) -> str:
    head = header(function, "void", ["bool *__restrict__ out", "int64_t base", "int64_t ub"])
    statement = "out[i] = (base + i) < ub;"
    v = function.vlen
    if function.isa == "avx512":
        step = (
            "        const __m512i lanes = _mm512_add_epi64(_mm512_set1_epi64((long long)(base + i)), "
            "_mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0));\n"
            "        const __mmask8 k = _mm512_cmplt_epi64_mask(lanes, _mm512_set1_epi64((long long)ub));\n"
            "        for (int j = 0; j < 8; ++j) out[i + j] = (k >> j) & 1;\n"
        )
        w = 8
    elif function.isa == "avx2":
        step = (
            "        const __m256i lanes = _mm256_add_epi64(_mm256_set1_epi64x((long long)(base + i)), "
            "_mm256_set_epi64x(3, 2, 1, 0));\n"
            "        int64_t active[4];\n"
            "        _mm256_storeu_si256((__m256i *)active, _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)ub), lanes));\n"
            "        for (int j = 0; j < 4; ++j) out[i + j] = active[j] != 0;\n"
        )
        w = 4
    elif function.isa == "neon":
        step = (
            "        const int64_t seed[2] = {base + i, base + i + 1};\n"
            "        const uint64x2_t active = vcltq_s64(vld1q_s64(seed), vdupq_n_s64(ub));\n"
            "        out[i] = vgetq_lane_u64(active, 0) != 0;\n"
            "        out[i + 1] = vgetq_lane_u64(active, 1) != 0;\n"
        )
        w = 2
    else:
        return f"{head} {{\n{scalar_loop(function, statement)}}}"
    return (
        f"{head} {{\n    int64_t i = 0;\n    for (; i + {w} <= {v}; i += {w}) {{\n{step}    }}\n"
        f"{scalar_loop(function, statement, tail=True)}}}"
    )


def indent(text: str) -> str:
    return "".join(f"    {line}\n" if line else "\n" for line in text.splitlines())


GENERATORS: dict[str, Callable[[TileFunction], str]] = {
    "binop": binop_function,
    "unop": unop_function,
    "fma": fma_function,
    "ite": ite_function,
    "reduce": reduce_function,
    "gather": gather_function,
    "maskgen": maskgen_function,
    "load": load_function,
    "store": store_function,
}
