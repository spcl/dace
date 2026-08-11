#!/usr/bin/env python3
"""Reproducer: a name first bound to an INT keeps an int64 container forever, and a later
float assignment to it is silently TRUNCATED.

`udiff = 1` then `udiff = <float expr>` is the canonical convergence-loop idiom. In DaCe the
container's dtype is fixed by the FIRST assignment, the float write is narrowed with no
warning, and `while udiff > 1e-3` therefore exits as soon as the true value drops below 1.0.

Exits 0 while the case still behaves as `../06-scalar-reassign-narrows.md` documents, 1 when
something moved. Needs a C++ compiler for the runtime half; the dtype probe is parse-only.
"""
import sys

import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64)
FAILED = []


@dace.program
def converge(a: dace.float64[N], trips: dace.int64[1], last: dace.float64[1]):
    udiff = 1
    n = 0
    while udiff > 0.001:
        a[:] = a[:] * 0.5
        udiff = 0.5 + 0.0 * a[0]  # a float, constant at 0.5, never below the 1e-3 threshold
        n += 1
    trips[0] = n
    last[0] = udiff


# Parse-only probe: which containers exist and with what dtype.
sdfg = converge.to_sdfg(simplify=False)
descs = {n: str(d.dtype) for n, d in sdfg.arrays.items() if n.startswith('udiff')}
print("udiff containers:", descs)
narrowed = any(v.startswith('int') for v in descs.values())
print("  -> the name bound by `udiff = 1` is an INTEGER container:", narrowed)
if not narrowed:
    FAILED.append("dtype probe")

for cfg in sdfg.all_control_flow_regions(recursive=True):
    cond = getattr(cfg, 'loop_condition', None)
    if cond is not None:
        print("  loop condition reads:", cond.as_string)

# Runtime: python does 1 -> 0.5 -> exit is never reached (0.5 > 1e-3 forever). DaCe truncates
# 0.5 to 0 and leaves after the FIRST body.
a = np.ones(8)
trips = np.zeros(1, dtype=np.int64)
last = np.zeros(1)
converge.to_sdfg(simplify=True).compile()(a=a, trips=trips, last=last, N=8)
print(f"trips={int(trips[0])} last={last[0]}  (python: this loop never terminates)")
if int(trips[0]) != 1 or last[0] != 0.0:
    FAILED.append("runtime truncation")
else:
    print("  -> REPRODUCES: one trip, and the float 0.5 was stored as 0.")

if FAILED:
    print("\nCHANGED:", FAILED)
    sys.exit(1)
sys.exit(0)
