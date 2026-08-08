# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k06 gather-accumulate-scatter -- the Shuffle witness (QUANTUM ESPRESSO ``addusxx_g``).

The reference kernel is the US-augmentation charge update in G-space::

    rhoc[gvec_pos[i]] += aux[i]        (a data-dependent scatter, zaxpy-flavored, complex128)

With an *injective* index the scatter is a value-permutation, and the SC26 layout paper (SS IV-C)
recasts the permutation as a LAYOUT decision rather than a runtime index array: the augmentation
charge ``x`` is physically renumbered by a closed-form bijection ``sigma`` (a Shuffle), the running
charge ``y`` is accumulated elementwise, and because a Shuffle is transparent (the gather ``x o
sigma`` plus the inverse-composed consumer preserve the result) every candidate reproduces the
oracle. The sweep chooses the element layout of ``x``; the algebra guarantees correctness -- the
value-permutation analog of the permutation witness k04.

Layout decisions: shuffle ``x``'s single dimension by each registered bijection (XOR swizzle,
cyclic shift) vs the unshuffled base. ``sigma`` is closed form, so no indirection subgraph is needed.

Source: P. Giannozzi et al., "QUANTUM ESPRESSO," J. Phys.: Condens. Matter 21 (2009), ``addusxx_g``
(exact-exchange/HSE US-augmentation); SC26 layout-algebra paper SS IV-C (Gather-Accumulate-Scatter
isolation). Elementwise accumulate form so the layout is honest -- a BLAS ``zaxpy`` would hide it.
"""
import numpy
import dace

from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.brute_force import shuffle_candidates

N = dace.symbol("N")


@dace.program
def gather_accumulate_scatter(x: dace.complex128[N], y: dace.complex128[N], out: dace.complex128[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        out[i] = y[i] + x[i]


def oracle(x, y):
    """The running charge plus the (gathered) augmentation charge -- the value-permuted accumulate."""
    return {"out": y + x}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.random(n) + 1j * rng.random(n)
    y = rng.random(n) + 1j * rng.random(n)
    return {"x": x.astype(numpy.complex128), "y": y.astype(numpy.complex128)}


def candidates():
    """The layout candidates: renumber ``x`` by each registered bijection ``sigma`` (plus the
    unshuffled base). Both are self-consistent bijections on ``[0, N)`` -- the XOR swizzle needs
    ``N`` a multiple of 4 (it permutes within each block of 4); the cyclic shift holds for any ``N``.
    Every Shuffle is transparent, so all candidates reproduce the oracle."""
    register_shuffle("gas_xor", "i ^ 3", "i ^ 3")
    register_shuffle("gas_cyc", "(i + 1) % N", "(i + N - 1) % N")
    return dict(shuffle_candidates("x", 0, ["gas_xor", "gas_cyc"]))


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds a fresh zeroed output each call."""

    def run(sdfg):
        out = numpy.zeros(n, dtype=numpy.complex128)
        sdfg(x=inputs["x"].copy(), y=inputs["y"].copy(), out=out, N=n)
        return {"out": out}

    return run
