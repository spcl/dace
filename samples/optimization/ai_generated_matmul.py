# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Matrix multiplication whose innermost microkernel is described rather than written.

The tiling, the parallel map and the data movement are ordinary DaCe: the program says which tile
of ``A`` and which tile of ``B`` reach the kernel, and the frontend turns those slices into views
that the kernel reads through. What is *not* written here is the kernel itself. ``dace.ai`` puts a
node carrying the description below into the map, and the description is the specification: when
the program is compiled, a language model is asked to write the tasklet that goes in its place,
given the description and everything DaCe knows about the slot -- the strides of each view, the
enclosing parallel map, and the compiler and CPU the code will be built for.

The first answer is rarely the last, so the sample also shows the loop around it: it prints the
generated kernel (``dace.libraries.ai.show``), measures it against NumPy, hands that measurement
back to the model as feedback for another round (``dace.libraries.ai.refine``), and lists the
rounds the kernel has been through (``dace.libraries.ai.history``)::

    python samples/optimization/ai_generated_matmul.py 512            # generate, run, measure
    python samples/optimization/ai_generated_matmul.py 512 --refine   # ... and ask for a faster one
    python samples/optimization/ai_generated_matmul.py 512 --refine "Use 4 accumulators per row."

Running it needs a model provider, so either set an API key::

    ANTHROPIC_API_KEY=... python samples/optimization/ai_generated_matmul.py
    # or, for the OpenAI SDK:
    DACE_ai_provider=responses DACE_ai_api_key_envvar=OPENAI_API_KEY DACE_ai_model=gpt-6-astra python samples/optimization/ai_generated_matmul.py

or relay the prompt by hand, without an API key, through a chat interface::

    DACE_ai_provider=manual python samples/optimization/ai_generated_matmul.py

The answer is cached, so a second run compiles without asking again. See :ref:`ai` for the whole
feature: the providers, what the model is told about a slot, and the rest of the iteration API
(``rollback`` to an earlier round, ``pin`` to freeze a kernel that is good enough).
"""

import argparse
import time

import numpy as np

import dace
from dace.libraries import ai

#: Reduction length. A symbol, so that the generated kernel is written once for any matrix size.
N = dace.symbol('N')

#: Edge of the output tile one invocation of the kernel computes. Eight single-precision elements
#: are one AVX2 register, which is what makes a register-blocked kernel natural here.
TILE = 8

#: Name of the node, and therefore of the slot that ``show``, ``refine`` and ``history`` address.
KERNEL = 'gemm_tile'

MICROKERNEL = f"""
Compute one {TILE}x{TILE} output tile of a single-precision matrix multiplication:

    _out[i][j] = sum over k in 0..N of _a[i][k] * _b[k][j],   for i, j in 0..{TILE}

Write it as a register-blocked microkernel using AVX2 intrinsics (immintrin.h): hold the
{TILE}x{TILE} accumulator tile in YMM registers across the whole reduction, accumulate with FMA
instructions, and store the tile once at the end. A scalar triple loop is not an acceptable
answer.

The three connectors are views into larger matrices rather than dense buffers, so step through
them with the strides given for each of them, and do not assume a row of a view is followed by the
next row. The enclosing map is already parallel: do not open a parallel region of your own.
""".strip()

#: Sent to the model when ``--refine`` is given without a message of its own. The point of a
#: refinement is that the model already knows this slot -- it sees the kernel it wrote and this
#: note, rather than the original question again -- so the feedback is a measurement, not a respec.
FEEDBACK = """
The kernel you wrote is correct, but slow: multiplying two {n}x{n} single-precision matrices with
it takes {ours:.2f} ms, while NumPy's BLAS does the same in {theirs:.2f} ms ({factor:.1f}x faster)
on this machine. Write a faster version.

Consider what the timing implies about the memory traffic: every invocation streams a whole
{tile}-row band of A and a whole {tile}-column band of B, so the reduction loop is where the time
goes. Unroll it, use several independent accumulator registers to hide the FMA latency, and keep
the loads aligned and sequential where the strides allow it.
"""


@dace.program
def matmul(A: dace.float32[N, N], B: dace.float32[N, N], C: dace.float32[N, N]):
    for ti, tj in dace.map[0:N:TILE, 0:N:TILE]:
        C[ti:ti + TILE, tj:tj + TILE] = dace.ai(MICROKERNEL,
                                                a=A[ti:ti + TILE, 0:N],
                                                b=B[0:N, tj:tj + TILE],
                                                shape=(TILE, TILE),
                                                dtype=dace.float32,
                                                name=KERNEL)


def best_of(call, repeats: int = 3) -> float:
    """
    Times a callable, ignoring the outliers a first run and a busy machine produce.

    :param call: What to time. Called once before the measurement to warm up (which is also what
                 compiles the program).
    :param repeats: How many times to measure.
    :return: The shortest wall-clock time observed, in milliseconds.
    """
    call()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1e3)
    return min(times)


def measure(sdfg: dace.SDFG, size: int):
    """
    Runs the generated program, checks it against NumPy, and times both.

    The program is compiled once and called through the compiled handle: calling the SDFG itself
    re-checks (and here, because a library of that name is already loaded, re-runs) the build on
    every call, which would dominate the measurement of a kernel this small.

    :param sdfg: The expanded SDFG.
    :param size: Edge length of the matrices.
    :return: A tuple of (milliseconds for the generated kernel, milliseconds for NumPy).
    :raises ValueError: If the generated kernel does not compute a matrix multiplication.
    """
    rng = np.random.default_rng(0)
    a = rng.random((size, size), dtype=np.float32)
    b = rng.random((size, size), dtype=np.float32)
    c = np.zeros((size, size), dtype=np.float32)
    expected = np.zeros((size, size), dtype=np.float32)

    compiled = sdfg.compile()
    ours = best_of(lambda: compiled(A=a, B=b, C=c, N=size))
    compiled.finalize()

    # Into a buffer, like the generated program, so that the two timings measure the same work
    theirs = best_of(lambda: np.matmul(a, b, out=expected))
    error = np.linalg.norm(c - expected) / np.linalg.norm(expected)
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(f'The generated kernel does not compute A @ B (relative error {error:.2e}). Read it with '
                         f'dace.libraries.ai.show(sdfg, "{KERNEL}"), then say so with '
                         f'dace.libraries.ai.refine(sdfg, "{KERNEL}", "...").')

    print(f'Relative error: {error:.2e}   generated: {ours:.2f} ms   NumPy: {theirs:.2f} ms')
    return ours, theirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, nargs='?', default=256)
    parser.add_argument('--refine',
                        nargs='?',
                        const='',
                        default=None,
                        metavar='FEEDBACK',
                        help='Ask the model for another round on the kernel. Without a message of your own, the '
                        'measured runtime is handed back as the feedback.')
    args = parser.parse_args()
    if args.N % TILE != 0:
        parser.error(f'N must be a multiple of the tile size ({TILE})')

    # Expanding explicitly, rather than letting compilation do it, so that the kernel can be read
    # before it runs. Everything from here on is an ordinary SDFG: it can be saved, transformed and
    # run without the model being involved again.
    sdfg = matmul.to_sdfg()
    sdfg.expand_library_nodes()

    print('=== Generated kernel ===')
    ai.show(sdfg, KERNEL)
    ours, theirs = measure(sdfg, args.N)

    if args.refine is not None:
        # A refinement is the next round of the same conversation, and it is atomic: if the new
        # kernel fails to generate or to compile, the one measured above is left exactly as it was.
        message = args.refine or FEEDBACK.format(n=args.N, tile=TILE, ours=ours, theirs=theirs,
                                                 factor=ours / theirs).strip()
        print('\n=== Feedback ===')
        print(message)
        ai.refine(sdfg, KERNEL, message)

        print('\n=== Revised kernel ===')
        ai.show(sdfg, KERNEL)
        revised, _ = measure(sdfg, args.N)
        change = 100 * (ours - revised) / ours
        print(f'The revised kernel is {abs(change):.1f}% {"faster" if change > 0 else "slower"} than the first one')

    # Every round is on disk, in the session for this slot, and can be restored without asking the
    # model anything: dace.libraries.ai.rollback(sdfg, KERNEL, round=1)
    rounds = ai.history(sdfg, KERNEL)
    print('\n=== History ===')
    for entry in rounds:
        print(f'  {entry}')
    if not rounds:
        print('  (no session on disk -- sessions are off, or this answer came from the cache of another run)')
