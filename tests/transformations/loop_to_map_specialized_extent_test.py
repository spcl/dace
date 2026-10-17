# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop lifted into a map over a nested SDFG keeps every symbol its moved descriptors use mapped, also a
compile-time constant (``SDFG.specialize``): the nested SDFG inherits the constant, so ``free_symbols`` leaves it
out, but a connector's descriptor is still compared in the parent's symbols (scattering_self_energies' ``Norb``)."""

import numpy as np

import dace
from dace.transformation.passes.parallelize_loops import ParallelizeLoops

N, K = dace.symbol("N"), dace.symbol("K")


@dace.program
def guarded_products(a: dace.float64[N, K, K], b: dace.float64[N, K, K]):
    for i in range(N):
        if a[i, 0, 0] > 0.5:
            b[i] = a[i] @ a[i]


def test_a_lifted_loop_maps_a_specialized_extent_of_its_containers():
    sdfg = guarded_products.to_sdfg(simplify=True)
    sdfg.specialize(dict(K=3))
    assert ParallelizeLoops().apply_pass(sdfg, {})
    sdfg.validate()

    a = np.random.default_rng(20261015).random((5, 3, 3))
    b = np.zeros_like(a)
    sdfg(a=a, b=b, N=5)
    assert np.allclose(b, np.array([x @ x if x[0, 0] > 0.5 else np.zeros((3, 3)) for x in a]))


if __name__ == "__main__":
    test_a_lifted_loop_maps_a_specialized_extent_of_its_containers()
