#!/usr/bin/env python3
"""Reproducer: `np.real` / `np.imag` emit an UNQUALIFIED `real(...)` / `imag(...)`.

The replacement builds a tasklet body out of the bare function name, and `cppunparse` writes an
unrecognised call name through verbatim. `real` / `imag` are absent from its `_renamed_funcs`
table, so the generated C++ names a function nothing declares. A COMPLEX operand compiles by
accident (ADL finds `std::real` through `std::complex`); a REAL one reaches no namespace at all.

Exits 0 while both cases still behave as `../08-unqualified-real-imag.md` documents, 1 when
something moved. Needs a C++ compiler.
"""
import sys

import numpy as np

import dace

FAILED = []
N = dace.symbol('N', dtype=dace.int64)


@dace.program
def take_real(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = np.real(a[i])


@dace.program
def take_imag(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = np.imag(a[i])


@dace.program
def take_real_complex(a: dace.complex128[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = np.real(a[i])


for label, prog, token in (("np.real(float64)", take_real, "real"), ("np.imag(float64)", take_imag, "imag")):
    try:
        prog.to_sdfg(simplify=True).compile()
        print(f"{label}: COMPILES -- changed")
        FAILED.append(label)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        # gcc quotes the name with U+2018/U+2019, clang with ASCII apostrophes -- match neither.
        plain = text.translate({0x2018: None, 0x2019: None, 0x27: None})
        hit = f"{token} was not declared in this scope" in plain
        print(f"{label}: {'REPRODUCES' if hit else 'FAILS DIFFERENTLY'}")
        for line in text.splitlines():
            if "error:" in line:
                print("   ", line.strip()[:160])
        if not hit:
            FAILED.append(label)

# The control: the same replacement on the type it was written for. Compiling here is what makes
# the defect above easy to miss, so a change in EITHER direction is worth reporting.
try:
    csdfg = take_real_complex.to_sdfg(simplify=True).compile()
    a = np.array([1.0 + 2.0j, -3.0 + 4.0j])
    out = np.zeros(2, dtype=np.float64)
    csdfg(a=a, out=out, N=2)
    ok = np.allclose(out, a.real)
    print(f"np.real(complex128): COMPILES and {'agrees' if ok else 'DISAGREES'} with numpy -- as documented")
    if not ok:
        FAILED.append("np.real(complex128) answer")
except Exception as exc:  # noqa: BLE001
    print(f"np.real(complex128): now fails too ({type(exc).__name__}) -- changed")
    FAILED.append("np.real(complex128)")

if FAILED:
    print("\nCHANGED:", FAILED)
    sys.exit(1)
print("\nBoth spellings reproduce as documented; the complex operand still compiles.")
sys.exit(0)
