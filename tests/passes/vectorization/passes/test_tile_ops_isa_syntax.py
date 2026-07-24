# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Syntax-only compile gate for the per-ISA K=1 tile-op backend headers.

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
"""
import os
import platform
import shutil
import subprocess

import pytest

import dace

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRIVER = os.path.join(_HERE, "tile_ops_all_ops_driver.cpp")
_DACE_INCLUDE = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), "runtime", "include")

_HOST_CXX = os.environ.get("CXX", "g++")
_HOST_MACHINE = platform.machine().lower()
_HOST_IS_X86 = _HOST_MACHINE in ("x86_64", "amd64", "x64")
_HOST_IS_ARM64 = _HOST_MACHINE in ("aarch64", "arm64")

#: An AArch64-targeting C++ compiler: the native one on an ARM host, else the
#: Debian/Ubuntu cross compiler. ``None`` if neither is installed (ARM headers
#: are then skipped, not failed).
_ARM64_CXX = (_HOST_CXX if _HOST_IS_ARM64 else None) or shutil.which("aarch64-linux-gnu-g++")


def _x86_case(header: str, *isa_flags: str):
    """An x86 backend: buildable only with an x86-targeting host compiler."""
    if not _HOST_IS_X86:
        return None
    return (_HOST_CXX, [*isa_flags], header)


def _arm_case(header: str, *isa_flags: str):
    """An AArch64 backend: buildable with a native or cross aarch64 compiler."""
    if _ARM64_CXX is None:
        return None
    return (_ARM64_CXX, [*isa_flags], header)


# (ISA name) -> (compiler, extra flags, backend header) or None when the target
# toolchain is unavailable on this host.
_CASES = {
    "scalar": (_HOST_CXX, [], "dace/tile_ops/scalar.h"),  # portable: host compiler, no ISA flags
    "avx2": _x86_case("dace/tile_ops/avx2.h", "-mavx2", "-mfma"),
    "avx512": _x86_case("dace/tile_ops/avx512.h", "-mavx512f"),
    "arm_neon": _arm_case("dace/tile_ops/arm_neon.h", "-march=armv8-a"),
    "arm_sve": _arm_case("dace/tile_ops/arm_sve.h", "-march=armv8-a+sve"),
}


@pytest.mark.parametrize("isa", list(_CASES.keys()))
def test_tile_ops_backend_header_compiles(isa: str) -> None:
    """The ISA backend header instantiates every tile op (all operands) cleanly."""
    case = _CASES[isa]
    if case is None:
        pytest.skip(f"no target compiler for {isa} on {_HOST_MACHINE}")
    cxx, isa_flags, header = case
    if shutil.which(cxx) is None:
        pytest.skip(f"compiler {cxx!r} not found")

    cmd = [
        cxx, "-std=c++17", "-fsyntax-only", "-I", _DACE_INCLUDE, *isa_flags, f"-DTILE_OPS_BACKEND_HEADER=<{header}>",
        _DRIVER
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (f"{isa} backend header {header} failed to compile with "
                                  f"{cxx} {' '.join(isa_flags)}:\n{proc.stderr}")


if __name__ == "__main__":
    for _isa in _CASES:
        try:
            test_tile_ops_backend_header_compiles(_isa)
            print(f"{_isa}: OK")
        except Exception as exc:  # noqa: BLE001 -- CLI summary only
            print(f"{_isa}: {exc}")
