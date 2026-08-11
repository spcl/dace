#!/usr/bin/env python3
"""Reproducer: `b = a` on a scalar ALIASES the container instead of copying the value.

Three cases, all wrong answers with no diagnostic:

(a) chained init  -- `s0 = s1 = 0.0` desugared to a temp makes every accumulator the
                     SAME container, so an 11-way unrolled reduction sums 11x.
(b) branch clobber -- `pc = rp0` in one branch and `pc = rp1` in the next writes rp1
                     INTO rp0; rp0 is destroyed for every later reader.
(c) plain copy     -- `b = a; b += 1.0` mutates `a` too.

Exits 0 while every case still behaves as `../05-scalar-assign-aliases.md` documents,
1 when something moved.  Needs a C++ compiler (the cases are about RUNTIME values).
"""
import sys

import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64)
FAILED = []


def check(name: str, got, want, should_differ: bool) -> None:
    differs = not np.allclose(got, want)
    print(f"{name}: got {got}  expected {want}  ->  {'REPRODUCES' if differs else 'AGREES'}")
    if differs != should_differ:
        FAILED.append(name)


# (a) chained initialisation: two accumulators become one
@dace.program
def chained(a: dace.float64[N], out: dace.float64[1]):
    __chain0 = 0.0
    s0 = __chain0
    s1 = __chain0
    i = 0
    while i + 2 <= N:
        s0 += a[i + 0]
        s1 += a[i + 1]
        i += 2
    out[0] = s0 + s1


a = np.arange(1.0, 9.0)
out = np.zeros(1)
chained.to_sdfg(simplify=True).compile()(a=a, out=out, N=8)
check("(a) chained accumulators", out[0], a.sum(), should_differ=True)


# (b) a branch dispatch writes through the alias and destroys the source
@dace.program
def branches(out: dace.float64[6]):
    rp0 = 10.0
    rp1 = 20.0
    rp2 = 30.0
    for idir in range(3):
        if idir == 0:
            pc = rp0
        elif idir == 1:
            pc = rp1
        else:
            pc = rp2
        out[idir] = pc
    out[3] = rp0
    out[4] = rp1
    out[5] = rp2


got = np.zeros(6)
branches.to_sdfg(simplify=True).compile()(out=got)
check("(b) branch clobber", got, np.array([10., 20., 30., 10., 20., 30.]), should_differ=True)


# (c) the bare case
@dace.program
def plain(out: dace.float64[2]):
    a = 1.0
    b = a
    b += 1.0
    out[0] = a
    out[1] = b


got = np.zeros(2)
plain.to_sdfg(simplify=True).compile()(out=got)
check("(c) plain copy", got, np.array([1.0, 2.0]), should_differ=True)

# The frontend mapping that causes it, straight out of the parser state.
sdfg = branches.to_sdfg(simplify=False)
print("\ncontainers in the (b) SDFG:", sorted(n for n in sdfg.arrays if not n.startswith('__')))
print("  -> `rp0` is absent: `pc` took its place and every `pc = ...` writes there.")

if FAILED:
    print("\nCHANGED:", FAILED)
    sys.exit(1)
print("\nAll three cases reproduce as documented.")
sys.exit(0)
