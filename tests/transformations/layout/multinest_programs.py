# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Multi-nest fixture programs for the GLOBAL layout assignment (GLOBAL_LAYOUT_DESIGN.md, task D1).

Each program is a flat line of loop nests over shared arrays. Two properties are load-bearing and
deliberately engineered:

  * **The nests must survive maximal fusion.** Canonicalize fuses pointwise producer-consumer maps,
    so every consumer here reads its producer NON-pointwise (transposed, at a stencil offset, or
    reversed) -- the line graph the fixtures promise is genuine, not an artifact of skipping fusion.
  * **The conflict must survive schedule canonicalization.** A lone transposed read is dissolved by
    a loop permutation (the schedule pass makes THAT array contiguous and moves the cost elsewhere),
    so each conflicting nest pins its canonical schedule with a MAJORITY of straight accesses,
    leaving the conflict array as the schedule's victim: its remedy is a LAYOUT, not a schedule.

The fixtures (k17 pattern, greedy-vs-global-vs-oracle):

  * ``conflict2`` -- B is written row-major (nest 1) and read transposed under a pinned schedule
    (nest 2). One boundary: keep B row-major (nest 2 pays), col-major (nest 1 pays), or relayout.
  * ``conflict3`` -- B additionally read transposed in nest 3: carrying the "wrong" layout now costs
    twice, so the relayout-after-nest-1 trajectory gains a nest of benefit over conflict2.
  * ``agree2`` -- stencil consumer, every array row-major-happy in both nests: dominance settles it
    with no trajectory (the control fixture).

Per-nest oracles (``*_NEST_ORACLES``) mirror one nest each for A1 externalization tests; whole
program oracles (``*_oracle``) are the end-to-end reference.
"""
import numpy

import dace

N = dace.symbol("N")


@dace.program
def conflict2(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = B[j, i] + A[i, j]


def conflict2_oracle(A):
    B = 2.0 * A
    return {"B": B, "C": B.T + A}


CONFLICT2_NEST_ORACLES = (
    lambda A, **_: {"B": 2.0 * A},
    lambda A, B, **_: {"C": B.T + A},
)


@dace.program
def conflict3(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], D: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = B[j, i] + A[i, j]
    for i, j in dace.map[0:N, 0:N]:
        D[i, j] = B[j, i] * 0.5 + C[i, N - 1 - j] + A[i, j]


def conflict3_oracle(A):
    B = 2.0 * A
    C = B.T + A
    return {"B": B, "C": C, "D": 0.5 * B.T + C[:, ::-1] + A}


CONFLICT3_NEST_ORACLES = (
    lambda A, **_: {"B": 2.0 * A},
    lambda A, B, **_: {"C": B.T + A},
    lambda A, B, C, **_: {"D": 0.5 * B.T + C[:, ::-1] + A},
)


@dace.program
def agree2(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N - 1]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N - 1]:
        C[i, j] = B[i, j] + B[i, j + 1]


def agree2_oracle(A):
    B = 2.0 * A
    return {"B": B, "C": B[:, :-1] + B[:, 1:]}


AGREE2_NEST_ORACLES = (
    lambda A, **_: {"B": 2.0 * A},
    lambda A, B, **_: {"C": B[:, :-1] + B[:, 1:]},
)

PROGRAMS = {
    "conflict2": (conflict2, conflict2_oracle, CONFLICT2_NEST_ORACLES),
    "conflict3": (conflict3, conflict3_oracle, CONFLICT3_NEST_ORACLES),
    "agree2": (agree2, agree2_oracle, AGREE2_NEST_ORACLES),
}


def make_inputs(n, seed=0):
    """The shared input A plus zero-initialized outputs sized for every fixture."""
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((n, n))}


def output_arrays(program_name, n):
    """Zero-initialized output buffers matching each fixture's descriptor shapes."""
    shapes = {
        "conflict2": {"B": (n, n), "C": (n, n)},
        "conflict3": {"B": (n, n), "C": (n, n), "D": (n, n)},
        "agree2": {"B": (n, n), "C": (n, n - 1)},
    }
    return {name: numpy.zeros(shape) for name, shape in shapes[program_name].items()}
