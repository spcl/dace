#!/usr/bin/env python3
"""Reproducer: `cmplx<T> <op> <integer>` in the runtime headers.

Two defects in `dace/runtime/include/dace/complex.h`, both reachable from ordinary Python:

(a) HARD ERROR   -- `complex / integer-symbol` does not compile: the header supplies mixed
                    `operator*` and nothing else, so `std::complex<double> / int64_t` has no
                    overload. A naive DFT (`np.exp(-1j * 2 * pi * k * n / N)`) hits it.
(b) SILENT WRONG -- the mixed `operator*` that DOES exist takes `const int&`, so an `int64_t`
                    symbol is narrowed to 32 bits with no diagnostic.

Exits 0 while both still behave as `../07-complex-integer-operators.md` documents, 1 when
something moved. Needs a C++ compiler; (b) also runs the compiled program.
"""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

import dace

INC = str(pathlib.Path(dace.__file__).resolve().parent / "runtime" / "include")
FAILED = []

N = dace.symbol('N', dtype=dace.int64)


# (a) the division
@dace.program
def dft(x: dace.complex128[N], y: dace.complex128[N]):
    for k in range(N):
        y[k] = 0j
        for n in range(N):
            y[k] += x[n] * np.exp(-1j * (2.0 * 3.141592653589793 * k * n / N))


try:
    dft.to_sdfg(simplify=True).compile()
    print("(a) complex / int64 symbol: COMPILES -- changed")
    FAILED.append("(a) division")
except Exception as exc:  # noqa: BLE001
    text = str(exc)
    hit = "no match for" in text and "operator/" in text
    print(f"(a) complex / int64 symbol: {'REPRODUCES' if hit else 'FAILS DIFFERENTLY'}")
    for line in text.splitlines():
        if "error:" in line:
            print("   ", line.strip()[:160])
    if not hit:
        FAILED.append("(a) division")

# (b) the narrowing, straight against the header
SRC = """
#include <dace/dace.h>
#include <cstdio>
int main() {
  int64_t k = 3000000000LL;          /* > 2**31 */
  dace::complex128 c(1.0, 0.0);
  auto r = c * k;
  printf("%.1f\\n", r.real());
  return 0;
}
"""
with tempfile.TemporaryDirectory() as td:
    src = pathlib.Path(td) / "narrow.cpp"
    exe = pathlib.Path(td) / "narrow"
    src.write_text(SRC)
    cc = subprocess.run(["c++", f"-I{INC}", "-std=c++20", "-fopenmp", str(src), "-o", str(exe)],
                        capture_output=True,
                        text=True)
    if cc.returncode != 0:
        print("(b) complex * int64: no longer compiles -- changed")
        FAILED.append("(b) narrowing")
    else:
        val = float(subprocess.run([str(exe)], capture_output=True, text=True).stdout.strip())
        print(f"(b) complex * int64: (1+0j) * 3000000000 -> {val:.1f}, expected 3000000000.0 "
              f"-> {'REPRODUCES' if val != 3e9 else 'AGREES'}")
        if val == 3e9:
            FAILED.append("(b) narrowing")

if FAILED:
    print("\nCHANGED:", FAILED)
    sys.exit(1)
print("\nBoth defects reproduce as documented.")
sys.exit(0)
