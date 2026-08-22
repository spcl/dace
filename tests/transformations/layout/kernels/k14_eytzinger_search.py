# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k14 eytzinger search -- the Shuffle (value-permutation) layout witness.

Eytzinger / BFS-heap layout of a sorted key set for cache-friendly, branch-free
predecessor search: the sorted keys are stored in heap order (root at the front,
then its two children, ...), so the hot upper tree levels sit contiguously and a
descent touches far fewer cache lines than a classic sorted binary search.

Source / attribution:
  * P.-V. Khuong, P. Morin, "Array Layouts for Comparison-Based Searching," ACM
    J. Exp. Algorithmics 22, 2017 (arXiv:1509.05053) -- the Eytzinger/BFS layout,
    branch-free + prefetch descent as the fastest general search layout.
  * S. Slotin, "Eytzinger Binary Search" (algorithmica.org/en/eytzinger).

Primitive: Shuffle. The layout decision is *which value-permutation sigma* to
store the key array ``A`` in. A shuffle is TRANSPARENT (the physical reorder
``A'[j] = A[sigma(j)]`` plus the inverse-composed body access
``A[e] -> A'[sigma^{-1}(e)]`` preserve the result), so -- exactly like a
dimension permutation -- every registered shuffle candidate must reproduce the
oracle. The sweep picks the layout; the shuffle algebra guarantees correctness.

SIMPLIFICATION (noted honestly): the true Eytzinger BFS index is defined by an
in-order tree traversal and is NOT a closed-form arithmetic function of the
element index over a RUNTIME symbol ``N`` -- it needs ``N`` (and the tree depth)
concretely to expand. ``register_shuffle`` requires a closed-form
forward/inverse pair over the reserved index ``i`` (+ SDFG symbols), so we model
the Eytzinger decision with a closed-form *affine stand-in*: a median-to-front
cyclic renumbering ``sigma(i) = (i + N//2) % N`` that, like Eytzinger, pulls the
median (the BST root) to the front of the array. Alongside it we register two
more closed-form reorderings (a unit cyclic shift and a reversal) so the sweep
explores a real family of element layouts. Every one is a bijection on ``[0, N)``
for any ``N``, hence transparent.

Also a transparent-kernel simplification: the real Eytzinger win is a
data-dependent branch-free descent over a query batch, whose control flow depends
on the layout and is therefore NOT layout-transparent. The transparent witness of
the *layout* is a plain elementwise pass over the key array; we keep that -- a
per-key touch of ``A`` -- and drop the descent (it is a compute detail, not a
layout decision).
"""
import numpy
import dace

from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.brute_force import shuffle_candidates

N = dace.symbol("N")

# The Eytzinger stand-in (median-to-front) plus two more closed-form transparent
# reorderings. Registered at import so ``candidates()`` (whose apply closures call
# ``ShuffleElements`` -> ``get_shuffle``) works standalone. All are bijections on
# ``[0, N)`` for any ``N`` (floored ``%`` via ``shuffle_pymod``, floored ``//``).
SHUFFLES = ("eytzinger", "cyc", "rev")
register_shuffle("eytzinger", "(i + N // 2) % N", "(i + N - N // 2) % N")  # median-to-front cyclic stand-in
register_shuffle("cyc", "(i + 1) % N", "(i + N - 1) % N")  # unit cyclic shift
register_shuffle("rev", "N - 1 - i", "N - 1 - i")  # reversal (an involution)


@dace.program
def eytzinger_search(A: dace.float64[N], out: dace.float64[N]):
    # A per-key elementwise pass: read every key point-wise and write a monotone
    # transform of it. The point access on ``A`` is what the shuffle composes
    # ``sigma^{-1}`` into; the multiply keeps a genuine per-element tasklet so the
    # read stays point-wise (a bare copy could fold to a full-extent bulk memlet).
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        out[i] = A[i] * A[i]


def oracle(A):
    """Pure-numpy reference. Layout-independent: the shuffle is transparent, so the
    reordered key array reproduces the same per-key result."""
    return {"out": A * A}


def make_inputs(n, seed=0):
    """A sorted key set of length ``n`` (Eytzinger operates on sorted keys)."""
    rng = numpy.random.default_rng(seed)
    return {"A": numpy.sort(rng.random(n))}


def candidates():
    """The global layout candidates for this kernel: the unshuffled key array plus
    each registered element reordering of ``A`` (Eytzinger stand-in, cyclic, reversal)."""
    return dict(shuffle_candidates("A", 0, SHUFFLES))


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep. The shuffle preserves ``A``'s
    shape (it clones a same-shape transient and reorders internally), so the caller
    passes the logical key array unchanged -- no descriptor reshape is needed."""

    def run(sdfg):
        out = numpy.zeros(n)
        sdfg(A=inputs["A"].copy(), out=out, N=n)
        return {"out": out}

    return run
