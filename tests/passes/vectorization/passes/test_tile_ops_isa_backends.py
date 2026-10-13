# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compile and numeric gates for the CPU tile functions (:mod:`dace.tile_functions`) and the SVE header.

Only the host's own ISA is ever *run*, so a backend can rot -- a mistyped intrinsic, a missing include -- without a
numeric test noticing. The first gate compiles every tile function of every CPU backend over every op, element type,
broadcast and mask flag and a spread of widths, as C23 and as C++, with the backend's compiler and no ``-m`` flag
(the functions carry their own target attribute, as a CPF unit needs). NEON needs an AArch64 compiler; where none is
installed its text is checked for the NEON intrinsics instead. The SVE header keeps its templates and its
``-fsyntax-only`` gate over :file:`tile_ops_all_ops_driver.cpp`.

The second gate BUILDS AND RUNS one generated driver per backend the host executes and demands a bit-for-bit match
against the scalar backend over random operands, mask / broadcast / stride / index-sign edge cases and the
adversarial values (NaN, signed zero, infinities, integer extremes). That is what caught the AVX min/max operand
order and the AVX-512 one-shot horizontal reduce.
"""

import os
import platform
import re
import shutil
import subprocess
import tempfile

import pytest

import dace
from dace import tile_functions
from dace.tile_functions import TileFunction

HERE = os.path.dirname(os.path.abspath(__file__))
SVE_DRIVER = os.path.join(HERE, "tile_ops_all_ops_driver.cpp")
DACE_INCLUDE = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), "runtime", "include")
CPP_STANDARD = dace.Config.get("compiler", "cpp_standard")
HOST_IS_X86 = platform.machine().lower() in ("x86_64", "amd64", "x64")
HOST_IS_ARM64 = platform.machine().lower() in ("aarch64", "arm64")
HOST_CC = os.environ.get("CC", "gcc")
HOST_CXX = os.environ.get("CXX", "g++")
ARM64_CXX = (HOST_CXX if HOST_IS_ARM64 else None) or shutil.which("aarch64-linux-gnu-g++")
ARM64_CC = (HOST_CC if HOST_IS_ARM64 else None) or shutil.which("aarch64-linux-gnu-gcc")

#: (element type code, widths) every gate drives.
SHAPES = [("f64", (8, 3, 17)), ("f32", (16, 5, 9)), ("i32", (8, 11)), ("i64", (4, 7))]
#: Every binop flag combination the numeric gate runs: (broadcast, masked).
BINOP_FLAGS = [("vv", False), ("vv", True), ("bv", True), ("vb", True), ("bb", False)]
UNOP_FLAGS = [("v", False), ("v", True), ("b", True)]
FMA_FLAGS = [("vvv", False), ("vvv", True), ("bvv", True), ("vbv", True), ("vvb", True), ("bbb", False)]
ITE_FLAGS = ["vv", "bv", "vb"]
INTEGER_UNOPS = "n!afc"
#: Adversarial operands per element type; ``special(k)`` cycles through them.
SPECIALS = {
    "f64": [
        "0.0",
        "-0.0",
        "1.0",
        "-1.0",
        "INFINITY",
        "-INFINITY",
        "NAN",
        "DBL_TRUE_MIN",
        "2.5",
        "-2.5",
        "DBL_MAX",
        "-DBL_MAX",
    ],
    "f32": [
        "0.0f",
        "-0.0f",
        "1.0f",
        "-1.0f",
        "INFINITY",
        "-INFINITY",
        "NAN",
        "FLT_TRUE_MIN",
        "2.5f",
        "-2.5f",
        "FLT_MAX",
        "-FLT_MAX",
    ],
    "i32": ["0", "1", "-1", "2", "-2", "7", "INT32_MAX", "INT32_MIN"],
    "i64": ["0", "1", "-1", "2", "-2", "7", "INT64_MAX", "INT64_MIN"],
}


def every_function(isa: str) -> list[TileFunction]:
    """Every signature of ``isa`` the gates drive, including the element types no backend vectorizes."""
    functions = []
    for code in ("f64", "f32", "i32", "i64", "i8", "u32", "b8"):
        for vlen in (1, 17):
            for op in tile_functions.BINARY_NAMES:
                functions += [
                    TileFunction("binop", isa, code, vlen, op, b, m) for b in ("vv", "bv", "vb") for m in (False, True)
                ]
            functions += [TileFunction("unop", isa, code, vlen, op, "v", True) for op in tile_functions.UNARY_NAMES]
            functions += [TileFunction("fma", isa, code, vlen, "", b, m) for b in ("vvv", "bvb") for m in (False, True)]
            functions += [
                TileFunction("ite", isa, code, vlen, "", b, False, c) for c in ("b8", code) for b in ("vv", "bv")
            ]
            functions += [TileFunction("reduce", isa, code, vlen, op) for op in "+*mM"]
            for masked in (False, True):
                functions += [TileFunction("gather", isa, code, vlen, "", "", masked, i) for i in ("i32", "i64")]
                functions += [TileFunction(kind, isa, code, vlen, masked=masked) for kind in ("load", "store")]
        functions.append(TileFunction("maskgen", isa, "i64", 8))
    return functions


def compile_check(compiler: str, language: str, source: str, *flags: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "functions." + ("c" if language == "c" else "cpp"))
        with open(path, "w") as unit:
            unit.write(source)
        # c2x: the C23 spelling every gcc >= 9 takes (the Ubuntu 24.04 aarch64 cross gcc 13 refuses c23)
        standard = "-std=c2x" if language == "c" else f"-std=c++{CPP_STANDARD}"
        proc = subprocess.run(
            [compiler, standard, "-fsyntax-only", "-Werror=implicit-function-declaration", *flags, path],
            check=False,
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, f"{compiler} {language}:\n{proc.stderr[:4000]}"


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512", "neon"])
def test_every_tile_function_compiles(isa: str, language: str) -> None:
    """Every tile function of ``isa`` compiles as C23 and as C++, with no ISA flag on the command line."""
    functions = every_function(isa)
    assert all(TileFunction.parse(function.name) == function for function in functions)
    source = tile_functions.definitions_in("\n".join(f"{function.name}();" for function in functions))
    if isa == "neon":
        compiler = ARM64_CC if language == "c" else ARM64_CXX
        if compiler is None:
            assert {"vld1q_f64", "vbslq_f32", "vfmaq_f64", "vcltq_s64", "#include <arm_neon.h>"} <= set(
                re.findall(r"vld1q_f64|vbslq_f32|vfmaq_f64|vcltq_s64|#include <arm_neon.h>", source)
            )
            return
        compile_check(compiler, language, source, "-march=armv8-a")
        return
    if isa != "scalar" and not HOST_IS_X86:
        pytest.skip(f"{isa} needs an x86 compiler")
    compile_check(HOST_CC if language == "c" else HOST_CXX, language, source)


def test_sve_header_compiles() -> None:
    """The SVE header, which keeps its templates, instantiates every tile op cleanly."""
    if ARM64_CXX is None:
        pytest.skip("no AArch64 compiler")
    proc = subprocess.run(
        [
            ARM64_CXX,
            f"-std=c++{CPP_STANDARD}",
            "-fsyntax-only",
            "-I",
            DACE_INCLUDE,
            "-march=armv8-a+sve",
            "-DTILE_OPS_BACKEND_HEADER=<dace/tile_ops/arm_sve.h>",
            SVE_DRIVER,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def element_c(code: str) -> str:
    return tile_functions.BY_CODE[code].c


DRIVER_PRELUDE = r"""
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static int64_t raw(uint64_t salt, int lane) { return (int64_t)(mix(salt * 1000003ULL + (uint64_t)lane) % 4001ULL) - 2000; }
static void emit(const char *label, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    printf("%s ", label);
    for (size_t i = 0; i < n; ++i) printf("%02x", b[i]);
    printf("\n");
}
"""


class Driver:
    """The C text of a driver that runs every case and dumps each result as hex under a label that names it."""

    def __init__(self, isa: str):
        self.isa = isa
        self.cases: list[str] = []
        self.calls: list[str] = []
        self.shapes: dict[tuple[str, int], None] = {}

    def call(self, function: TileFunction, arguments: list[str]) -> str:
        function = function._replace(isa=self.isa)
        self.calls.append(function.name)
        return f"{function.name}({', '.join(arguments)})"

    def case(
        self, function: TileFunction, arguments: list[str], salt: int, suffix: str = "", prelude: str = ""
    ) -> None:
        """One call of ``function`` on fresh operands; a store dumps its destination, the rest ``out``."""
        code, vlen = function.element, function.vlen
        t = element_c(code)
        self.shapes[(code, vlen)] = None
        call = self.call(function, arguments)
        if function.kind == "reduce":
            call = f"out[0] = {call}"
        dump = "dst" if function.kind == "store" else "out"
        label = function._replace(isa="").name + suffix
        self.cases.append(f"""{{
    {t} out[{vlen}], a[{vlen}], b[{vlen}], c[{vlen}], cond[{vlen}], src[{4 * vlen}], dst[{4 * vlen}];
    bool mask[{vlen}];
    int32_t idx32[{vlen}];
    int64_t idx64[{vlen}];
    fill_{code}_{vlen}({salt}ULL, out, a, b, c, cond, src, dst, mask, idx32, idx64);
    {prelude}{call};
    emit("{label}", {dump}, sizeof({dump}));
}}""")

    def fill(self, code: str, vlen: int) -> str:
        """The operands of every case of one shape, generated from the case's salt alone."""
        t = element_c(code)
        value = f"({t})raw(s, i) / ({t})8" if code.startswith("f") else f"({t})(raw(s, i) % 251)"
        return f"""static void fill_{code}_{vlen}(uint64_t s0, {t} *out, {t} *a, {t} *b, {t} *c, {t} *cond, {t} *src,
                         {t} *dst, bool *mask, int32_t *idx32, int64_t *idx64) {{
    for (int i = 0; i < {vlen}; ++i) {{
        uint64_t s = s0 + 1; a[i] = {value};
        s = s0 + 2; b[i] = {value}; if (b[i] == 0) b[i] = 3;
        s = s0 + 3; c[i] = {value};
        cond[i] = ({t})(mix(s0 + 4 + (uint64_t)i) % 3ULL);
        mask[i] = (mix(s0 + 5 + (uint64_t)i) % 3ULL) != 0ULL;
        idx32[i] = (int32_t)(mix(s0 + 6 + (uint64_t)i) % {vlen}ULL);
        idx64[i] = (int64_t)(mix(s0 + 7 + (uint64_t)i) % {vlen}ULL);
        out[i] = ({t})-99;
    }}
    for (int i = 0; i < {4 * vlen}; ++i) {{ uint64_t s = s0 + 9; src[i] = {value}; s = s0 + 13; dst[i] = {value}; }}
}}
"""

    def text(self) -> str:
        definitions = tile_functions.definitions_in("\n".join(f"{name}();" for name in self.calls))
        fills = "".join(self.fill(code, vlen) for code, vlen in self.shapes)
        main = "\n".join(self.cases)
        return f"{DRIVER_PRELUDE}{definitions}\n{fills}\nint main(void) {{\n{main}\n    return 0;\n}}\n"


def masked_arguments(arguments: list[str], masked: bool) -> list[str]:
    return [*arguments, "mask"] if masked else arguments


def add_elementwise_cases(driver: Driver, code: str, vlen: int, salt: int) -> None:
    isa = driver.isa
    for index, op in enumerate(tile_functions.BINARY_NAMES):
        for flag, (broadcast, masked) in enumerate(BINOP_FLAGS):
            function = TileFunction("binop", isa, code, vlen, op, broadcast, masked)
            driver.case(function, masked_arguments(["out", "a", "b"], masked), salt + 100 * index + flag)
    for index, op in enumerate(tile_functions.UNARY_NAMES):
        if code.startswith("f") or op in INTEGER_UNOPS:
            for flag, (broadcast, masked) in enumerate(UNOP_FLAGS):
                function = TileFunction("unop", isa, code, vlen, op, broadcast, masked)
                driver.case(function, masked_arguments(["out", "a"], masked), salt + 10000 + 100 * index + flag)
    for flag, (broadcast, masked) in enumerate(FMA_FLAGS):
        function = TileFunction("fma", isa, code, vlen, "", broadcast, masked)
        driver.case(function, masked_arguments(["out", "a", "b", "c"], masked), salt + 20000 + flag)
    for flag, broadcast in enumerate(ITE_FLAGS):
        driver.case(
            TileFunction("ite", isa, code, vlen, "", broadcast, False, code),
            ["out", "cond", "a", "b"],
            salt + 30000 + flag,
        )
    for index, op in enumerate("+*mM"):
        driver.case(TileFunction("reduce", isa, code, vlen, op), ["a"], salt + 60000 + index)


def add_memory_cases(driver: Driver, code: str, vlen: int, salt: int) -> None:
    isa = driver.isa
    for stride in (1, 3):
        for masked in (False, True):
            load = TileFunction("load", isa, code, vlen, masked=masked)
            arguments = [*masked_arguments(["out", "src"], masked), str(stride)]
            driver.case(load, arguments, salt + 40000 + stride + 10 * masked, f"_s{stride}")
            store = TileFunction("store", isa, code, vlen, masked=masked)
            arguments = [*masked_arguments(["dst", "a"], masked), str(stride)]
            driver.case(store, arguments, salt + 41000 + stride + 10 * masked, f"_s{stride}")
    for index in ("i32", "i64"):
        indices = f"idx{index[1:]}"
        for masked in (False, True):
            gather = TileFunction("gather", isa, code, vlen, "", "", masked, index)
            driver.case(gather, masked_arguments(["out", "a", indices], masked), salt + 50000 + 10 * masked)
            # Negative indices through a biased base pointer, and the all-false / all-true masks.
            prelude = (
                f"for (int i = 0; i < {vlen}; ++i) {{ {indices}[i] = (i % 2) ? -(i % {vlen}) : (i % {vlen}); "
                f"mask[i] = {int(masked)}; }}\n    "
            )
            arguments = ["out", f"src + {vlen}", indices, "mask"]
            driver.case(gather._replace(masked=True), arguments, salt + 51000, f"_neg{int(masked)}", prelude)


def add_special_cases(driver: Driver, code: str, vlen: int) -> None:
    """The adversarial operands, where a SIMD instruction and the scalar contract part ways."""
    isa, specials = driver.isa, SPECIALS[code]
    count = len(specials)
    for shift in (1, 3, 6):
        prelude = (
            f"const {element_c(code)} sp[{count}] = {{{', '.join(specials)}}};\n    "
            f"for (int i = 0; i < {vlen}; ++i) {{ a[i] = sp[i % {count}]; b[i] = sp[(i + {shift}) % {count}]; "
            f"cond[i] = sp[(i + 1) % {count}]; mask[i] = (i % 3) != 1; }}\n    "
        )
        suffix = f"_sp{shift}"
        for op in "mM<l>g=!&|+-*":
            for masked in (False, True):
                function = TileFunction("binop", isa, code, vlen, op, "vv", masked)
                driver.case(function, masked_arguments(["out", "a", "b"], masked), 7, suffix, prelude)
        driver.case(
            TileFunction("ite", isa, code, vlen, "", "vv", False, code), ["out", "cond", "a", "b"], 7, suffix, prelude
        )
        for op in "mM+":
            driver.case(TileFunction("reduce", isa, code, vlen, op), ["a"], 7, suffix, prelude)


def build_driver(isa: str) -> Driver:
    driver = Driver(isa)
    salt = 1
    for code, widths in SHAPES:
        for vlen in widths:
            salt += 100000
            add_elementwise_cases(driver, code, vlen, salt)
            add_memory_cases(driver, code, vlen, salt)
            add_special_cases(driver, code, vlen)
    for vlen in (8, 17, 5):
        for base, bound in ((0, vlen), (0, vlen // 2), (3, vlen), (vlen, vlen), (-2, vlen - 1)):
            call = driver.call(TileFunction("maskgen", isa, "i64", vlen), ["lanes", str(base), str(bound)])
            label = f"maskgen_{vlen}_{base}_{bound}"
            driver.cases.append(f'{{ bool lanes[{vlen}]; {call}; emit("{label}", lanes, sizeof(lanes)); }}')
    return driver


def run_driver(isa: str, tmpdir: str) -> dict[str, str]:
    source = os.path.join(tmpdir, f"driver_{isa}.c")
    binary = os.path.join(tmpdir, f"driver_{isa}")
    with open(source, "w") as unit:
        unit.write(build_driver(isa).text())
    build = subprocess.run(
        [HOST_CC, "-std=c23", "-O2", "-ffp-contract=off", source, "-o", binary, "-lm"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, f"{isa} driver failed to build:\n{build.stderr[:4000]}"
    run = subprocess.run([binary], capture_output=True, text=True, check=False)
    if run.returncode == -4:  # SIGILL: the host CPU does not implement this ISA
        pytest.skip(f"host CPU cannot execute {isa}")
    assert run.returncode == 0, f"{isa} driver exited {run.returncode}:\n{run.stderr}"
    return dict(line.split() for line in run.stdout.splitlines())


def host_isas() -> list[str]:
    if HOST_IS_X86:
        return ["avx2", "avx512"]
    return ["neon"] if HOST_IS_ARM64 else []


@pytest.mark.parametrize("isa", ["avx2", "avx512", "neon"])
def test_tile_functions_match_scalar_bit_for_bit(isa: str) -> None:
    """Every case of an ISA's tile functions reproduces the scalar backend exactly: ``fma`` everywhere, one reduce
    association, ``c_mod`` for ``%``, ``py_mod`` for ``p``, ``min``'s operand order on NaN and signed zero."""
    if isa not in host_isas():
        pytest.skip(f"the host cannot run {isa}")
    with tempfile.TemporaryDirectory() as tmpdir:
        reference = run_driver("scalar", tmpdir)
        actual = run_driver(isa, tmpdir)
    assert reference.keys() == actual.keys(), "the dumps cover different cases"
    mismatched = [label for label, value in reference.items() if actual[label] != value]
    detail = "\n".join(f"  {k}\n    scalar {reference[k]}\n    {isa:<6} {actual[k]}" for k in mismatched[:12])
    assert not mismatched, f"{isa} differs from scalar in {len(mismatched)} of {len(reference)} cases:\n{detail}"


if __name__ == "__main__":
    for isa in ("scalar", "avx2", "avx512", "neon"):
        test_every_tile_function_compiles(isa, "c")
        test_every_tile_function_compiles(isa, "c++")
    test_sve_header_compiles()
    test_tile_functions_match_scalar_bit_for_bit("avx2")
    test_tile_functions_match_scalar_bit_for_bit("avx512")
    test_tile_functions_match_scalar_bit_for_bit("neon")
