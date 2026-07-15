# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k11 batched tridiagonal (Thomas) -- interleaved vs system-major coefficient layout.

Solve NB independent K-point tridiagonal systems (a,b,c | d) -- the ICON implicit vertical-diffusion
/ ADI analog. The k-loop is sequential (forward elimination + back substitution); the batch (NB) is
the vector dimension. The layout decision is the coefficient orientation:

    interleaved  (K, NB): level k of all systems contiguous -> each k-step touches contiguous rows.
    system-major (NB, K): each system contiguous            -> each k-step gathers a stride-K column.

expressed as a Permute of the four coefficient arrays (transparent under add_permute_maps). The
temporaries stay (K, NB) in both, so only the coefficient reads differ.

Source: Laszlo, Giles, Appleyard, ACM TOMS 42(4) 2016 (cuThomasBatch convention); Zaengl et al.,
QJRMS'15 (ICON vertical solvers).
"""
import numpy
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions

K, NB = dace.symbol("K"), dace.symbol("NB")


@dace.program
def thomas(a: dace.float64[K, NB], b: dace.float64[K, NB], c: dace.float64[K, NB], d: dace.float64[K, NB],
           x: dace.float64[K, NB]):
    cp = numpy.empty((K, NB), dace.float64)
    dp = numpy.empty((K, NB), dace.float64)
    for j in dace.map[0:NB]:
        cp[0, j] = c[0, j] / b[0, j]
        dp[0, j] = d[0, j] / b[0, j]
    for k in range(1, K):
        for j in dace.map[0:NB]:
            m = b[k, j] - a[k, j] * cp[k - 1, j]
            cp[k, j] = c[k, j] / m
            dp[k, j] = (d[k, j] - a[k, j] * dp[k - 1, j]) / m
    for j in dace.map[0:NB]:
        x[K - 1, j] = dp[K - 1, j]
    for k in range(K - 2, -1, -1):
        for j in dace.map[0:NB]:
            x[k, j] = dp[k, j] - cp[k, j] * x[k + 1, j]


def oracle(a, b, c, d):
    k_, nb_ = a.shape
    cp = numpy.empty((k_, nb_))
    dp = numpy.empty((k_, nb_))
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for k in range(1, k_):
        m = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / m
        dp[k] = (d[k] - a[k] * dp[k - 1]) / m
    x = numpy.empty((k_, nb_))
    x[k_ - 1] = dp[k_ - 1]
    for k in range(k_ - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return {"x": x}


def make_inputs(k, nb, seed=0):
    rng = numpy.random.default_rng(seed)
    a = rng.random((k, nb)) + 0.1
    c = rng.random((k, nb)) + 0.1
    b = a + c + rng.random((k, nb)) + 1.0  # diagonally dominant
    d = rng.random((k, nb)) + 0.1
    return {"a": a, "b": b, "c": c, "d": d}


def candidates():
    """interleaved (K,NB) vs system-major (NB,K): permute all four coefficient arrays."""

    def system_major(sdfg):
        PermuteDimensions(permute_map={n: [1, 0] for n in "abcd"}, add_permute_maps=True).apply_pass(sdfg, {})

    return {"interleaved": (lambda sdfg: None), "system_major": system_major}


def run_closure(inputs, k, nb):

    def run(sdfg):
        x = numpy.zeros((k, nb))
        sdfg(a=inputs["a"].copy(), b=inputs["b"].copy(), c=inputs["c"].copy(), d=inputs["d"].copy(), x=x, K=k, NB=nb)
        return {"x": x}

    return run
