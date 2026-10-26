# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tiled matrix multiplication whose microkernel is written by a language model (``dace.ai``).

The sample prints the generated kernel, compares it with NumPy, and with ``--refine`` asks for a
faster version with the measurement as feedback::

    python samples/optimization/ai_generated_matmul.py 512
    python samples/optimization/ai_generated_matmul.py 512 --refine
    python samples/optimization/ai_generated_matmul.py 512 --refine "Use 4 accumulators per row."

It needs a model provider (see the "AI-Generated Implementations" documentation)::

    ANTHROPIC_API_KEY=... python samples/optimization/ai_generated_matmul.py
    DACE_ai_provider=openai DACE_ai_model=<model> OPENAI_API_KEY=... python samples/optimization/ai_generated_matmul.py
    DACE_ai_provider=manual python samples/optimization/ai_generated_matmul.py
"""

import argparse
import time

import numpy as np

import dace
from dace.libraries import ai

#: Matrix size. A symbol, so that the kernel is generated once for every size.
N = dace.symbol("N")

#: Edge of the output tile of one kernel invocation; eight floats fill one AVX2 register.
TILE = 8

#: Name of the node, as used by ``show``, ``refine`` and ``history``.
KERNEL = "gemm_tile"

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

#: The feedback of ``--refine`` without a message.
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
        C[ti : ti + TILE, tj : tj + TILE] = dace.ai(
            MICROKERNEL,
            a=A[ti : ti + TILE, 0:N],
            b=B[0:N, tj : tj + TILE],
            shape=(TILE, TILE),
            dtype=dace.float32,
            name=KERNEL,
        )


def best_of(call, repeats: int = 3) -> float:
    """
    Times a callable after one warm-up call.

    :param call: What to time.
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
    Runs the compiled program, checks it against NumPy, and times both.

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

    theirs = best_of(lambda: np.matmul(a, b, out=expected))
    error = np.linalg.norm(c - expected) / np.linalg.norm(expected)
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(
            f"The generated kernel does not compute A @ B (relative error {error:.2e}). Read it with "
            f'dace.libraries.ai.show(sdfg, "{KERNEL}"), then say so with '
            f'dace.libraries.ai.refine(sdfg, "{KERNEL}", "...").'
        )

    print(f"Relative error: {error:.2e}   generated: {ours:.2f} ms   NumPy: {theirs:.2f} ms")
    return ours, theirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=256)
    parser.add_argument(
        "--refine",
        nargs="?",
        const="",
        default=None,
        metavar="FEEDBACK",
        help="Ask the model for another round on the kernel. Without a message of your own, the "
        "measured runtime is handed back as the feedback.",
    )
    args = parser.parse_args()
    if args.N % TILE != 0:
        parser.error(f"N must be a multiple of the tile size ({TILE})")

    # Expanded before compilation, to print the kernel
    sdfg = matmul.to_sdfg()
    sdfg.expand_library_nodes()

    print("=== Generated kernel ===")
    ai.show(sdfg, KERNEL)
    ours, theirs = measure(sdfg, args.N)

    if args.refine is not None:
        # If the new round fails, the kernel measured above is kept
        message = (
            args.refine or FEEDBACK.format(n=args.N, tile=TILE, ours=ours, theirs=theirs, factor=ours / theirs).strip()
        )
        print("\n=== Feedback ===")
        print(message)
        ai.refine(sdfg, KERNEL, message)

        print("\n=== Revised kernel ===")
        ai.show(sdfg, KERNEL)
        revised, _ = measure(sdfg, args.N)
        change = 100 * (ours - revised) / ours
        print(f"The revised kernel is {abs(change):.1f}% {'faster' if change > 0 else 'slower'} than the first one")

    # Earlier rounds can be restored with dace.libraries.ai.rollback(sdfg, KERNEL, round=1)
    rounds = ai.history(sdfg, KERNEL)
    print("\n=== History ===")
    for entry in rounds:
        print(f"  {entry}")
    if not rounds:
        print("  (no session on disk: sessions are off, or the answer came from the cache)")
