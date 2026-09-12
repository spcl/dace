# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compile + numeric gates for the per-ISA K=1 tile-op backend headers.

Each ``dace/tile_ops/<isa>.h`` exposes the SAME ``dace::tileops`` signatures but
lowers to that ISA's intrinsics. Only the host's own ISA is ever *run* by the
numeric tests (an AVX-512 dev box never executes the NEON / SVE paths), so a
header can rot -- an undefined helper, a missing include, a mistyped intrinsic --
without any test noticing. This gate CROSS-COMPILES every backend header with the
matching target compiler + ``-march`` at ``-fsyntax-only`` (no run, no link),
driving :file:`tile_ops_all_ops_driver.cpp` which odr-uses every tile op with
every operand, dtype, broadcast/masked flag, op char, and a spread of widths.

A backend is skipped only when its target toolchain is genuinely unavailable on
this host (e.g. no ``aarch64-linux-gnu-g++`` for the ARM headers); where the
toolchain exists the header MUST compile. On the ARM-cross dev box all five
(scalar / avx2 / avx512 / arm_neon / arm_sve) run.

Compiling is not computing, so the second gate here BUILDS AND RUNS
:file:`tile_ops_numeric_driver.cpp` for every backend the host can execute and
demands a bit-for-bit match against the scalar reference. That is what caught
the AVX min/max operand order (NaN and signed zero went the wrong way) and the
AVX-512 one-shot horizontal reduce (a different association than the pairwise
tree every other backend uses).
"""
import os
import platform
import shutil
import subprocess
import tempfile

import pytest

import dace

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "tile_ops_all_ops_driver.cpp")
NUMERIC_DRIVER = os.path.join(HERE, "tile_ops_numeric_driver.cpp")
DACE_INCLUDE = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), "runtime", "include")

HOST_CXX = os.environ.get("CXX", "g++")
#: The standard DaCe actually builds generated code with. Hardcoding one here
#: silently un-gates every backend the day the runtime headers start using a
#: newer feature (``dace/types.h`` moved to ``std::bit_cast``, so a pinned
#: ``c++17`` failed on ``<dace/dace.h>`` before reaching any tile-op header).
CPP_STANDARD = dace.Config.get("compiler", "cpp_standard")
HOST_MACHINE = platform.machine().lower()
HOST_IS_X86 = HOST_MACHINE in ("x86_64", "amd64", "x64")
HOST_IS_ARM64 = HOST_MACHINE in ("aarch64", "arm64")

#: An AArch64-targeting C++ compiler: the native one on an ARM host, else the
#: Debian/Ubuntu cross compiler. ``None`` if neither is installed (ARM headers
#: are then skipped, not failed).
ARM64_CXX = (HOST_CXX if HOST_IS_ARM64 else None) or shutil.which("aarch64-linux-gnu-g++")


def _x86_case(header: str, *isa_flags: str):
    """An x86 backend: buildable only with an x86-targeting host compiler."""
    if not HOST_IS_X86:
        return None
    return (HOST_CXX, [*isa_flags], header)


def _arm_case(header: str, *isa_flags: str):
    """An AArch64 backend: buildable with a native or cross aarch64 compiler."""
    if ARM64_CXX is None:
        return None
    return (ARM64_CXX, [*isa_flags], header)


# (ISA name) -> (compiler, extra flags, backend header) or None when the target
# toolchain is unavailable on this host.
CASES = {
    "scalar": (HOST_CXX, [], "dace/tile_ops/scalar.h"),  # portable: host compiler, no ISA flags
    "avx2": _x86_case("dace/tile_ops/avx2.h", "-mavx2", "-mfma"),
    "avx512": _x86_case("dace/tile_ops/avx512.h", "-mavx512f"),
    "arm_neon": _arm_case("dace/tile_ops/arm_neon.h", "-march=armv8-a"),
    "arm_sve": _arm_case("dace/tile_ops/arm_sve.h", "-march=armv8-a+sve"),
}


@pytest.mark.parametrize("isa", list(CASES.keys()))
def test_tile_ops_backend_header_compiles(isa: str) -> None:
    """The ISA backend header instantiates every tile op (all operands) cleanly."""
    case = CASES[isa]
    if case is None:
        pytest.skip(f"no target compiler for {isa} on {HOST_MACHINE}")
    cxx, isa_flags, header = case
    if shutil.which(cxx) is None:
        pytest.skip(f"compiler {cxx!r} not found")

    cmd = [
        cxx, f"-std=c++{CPP_STANDARD}", "-fsyntax-only", "-I", DACE_INCLUDE, *isa_flags,
        f"-DTILE_OPS_BACKEND_HEADER=<{header}>", DRIVER
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (f"{isa} backend header {header} failed to compile with "
                                  f"{cxx} {' '.join(isa_flags)}:\n{proc.stderr}")


def _build_and_run(tmpdir: str, isa: str) -> str:
    """Compile the numeric driver against ``isa`` and return its stdout dump."""
    case = CASES[isa]
    if case is None:
        pytest.skip(f"no target compiler for {isa} on {HOST_MACHINE}")
    cxx, isa_flags, header = case
    if cxx != HOST_CXX:
        pytest.skip(f"{isa} targets a foreign architecture; cannot execute on {HOST_MACHINE}")
    if shutil.which(cxx) is None:
        pytest.skip(f"compiler {cxx!r} not found")

    binary = os.path.join(tmpdir, f"tile_ops_numeric_{isa}")
    build = subprocess.run([
        cxx, f"-std=c++{CPP_STANDARD}", "-O2", "-I", DACE_INCLUDE, *isa_flags, f"-DTILE_OPS_BACKEND_HEADER=<{header}>",
        NUMERIC_DRIVER, "-o", binary
    ],
                           capture_output=True,
                           text=True)
    assert build.returncode == 0, f"{isa} numeric driver failed to build:\n{build.stderr}"

    run = subprocess.run([binary], capture_output=True, text=True)
    if run.returncode == -4:  # SIGILL: the host CPU does not implement this ISA
        pytest.skip(f"host CPU cannot execute {isa}")
    assert run.returncode == 0, f"{isa} numeric driver exited {run.returncode}:\n{run.stderr}"
    return run.stdout


@pytest.mark.parametrize("isa", [i for i in CASES if i != "scalar"])
def test_tile_ops_backend_matches_scalar_bit_for_bit(isa: str) -> None:
    """An ISA backend must reproduce the scalar reference EXACTLY, every op.

    The sibling syntax gate only proves a header compiles. This one runs it:
    :file:`tile_ops_numeric_driver.cpp` is built once per backend and dumps the
    raw bytes of every tile-op result over random operands, mask / broadcast /
    stride / index-sign edge cases, and the adversarial values (NaN, signed
    zero, infinities, integer extremes) where a SIMD instruction and the scalar
    contract are most likely to part ways. The headers promise bit-for-bit
    agreement -- ``std::fma`` on every backend, one shared reduce association,
    py_mod for ``%`` -- so the comparison is exact, not a tolerance.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reference = _build_and_run(tmpdir, "scalar")
        actual = _build_and_run(tmpdir, isa)

    ref_rows = dict(line.split() for line in reference.splitlines())
    isa_rows = dict(line.split() for line in actual.splitlines())
    assert ref_rows.keys() == isa_rows.keys(), "backend dumps cover different cases"
    mismatched = [k for k, v in ref_rows.items() if isa_rows[k] != v]
    detail = "\n".join(f"  {k}\n    scalar {ref_rows[k]}\n    {isa:<6} {isa_rows[k]}" for k in mismatched[:12])
    assert not mismatched, (f"{isa} differs from the scalar reference in {len(mismatched)} of "
                            f"{len(ref_rows)} cases:\n{detail}")


if __name__ == "__main__":
    for isa in CASES:
        for gate in (test_tile_ops_backend_header_compiles, test_tile_ops_backend_matches_scalar_bit_for_bit):
            if isa == "scalar" and gate is test_tile_ops_backend_matches_scalar_bit_for_bit:
                continue
            try:
                gate(isa)
                print(f"{isa} {gate.__name__}: OK")
            except Exception as exc:  # noqa: BLE001 -- CLI summary only
                print(f"{isa} {gate.__name__}: {exc}")
