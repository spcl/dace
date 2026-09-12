# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.moveaxis``, which lowers as the axis permutation it is."""
import numpy as np
import dace
from common import compare_numpy_output

from dace.sdfg.nodes import LibraryNode

N = dace.symbol('N', dtype=dace.int64)


@compare_numpy_output()
def test_moveaxis_single(A: dace.float32[10, 5, 3, 2]):
    return np.moveaxis(A, 0, 2)


@compare_numpy_output()
def test_moveaxis_negative(A: dace.float32[10, 5, 3, 2]):
    return np.moveaxis(A, -1, 0)


@compare_numpy_output()
def test_moveaxis_multiple(A: dace.float32[10, 5, 3, 2]):
    return np.moveaxis(A, [0, 3], [3, 0])


@compare_numpy_output()
def test_moveaxis_identity(A: dace.float32[10, 5, 3, 2]):
    return np.moveaxis(A, [0, 1, 2, 3], [0, 1, 2, 3])


@dace.program
def hpsi(x: dace.float64[N, N, N], vloc: dace.float64[N, N, N], half_inv_h2: dace.float64, lap: dace.float64[N, N]):
    """Separable Laplacian: one contraction per spatial axis, each moved back where it came from."""
    t0 = np.tensordot(lap, x, axes=([1], [0]))
    t1 = np.moveaxis(np.tensordot(lap, x, axes=([1], [1])), 0, 1)
    t2 = np.moveaxis(np.tensordot(lap, x, axes=([1], [2])), 0, 2)
    return -half_inv_h2 * (t0 + t1 + t2) + vloc * x


def library_nodes(sdfg: dace.SDFG) -> dict:
    counts = {}
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, LibraryNode):
            counts[type(node).__name__] = counts.get(type(node).__name__, 0) + 1
    return counts


def test_moveaxis_lowers_to_a_library_node():
    """The permutation must reach ``TensorTranspose``.

    Without a replacement the frontend does not refuse ``moveaxis`` -- it silently degrades to a
    pyobject callback into the interpreter and then dies one statement later on the untyped
    result. Counting the nodes is what separates "lowered" from "called back", which no numeric
    comparison can see.
    """
    sdfg = hpsi.to_sdfg(simplify=False)
    sdfg.validate()
    assert library_nodes(sdfg) == {'TensorDot': 3, 'TensorTranspose': 2}


def test_moveaxis_separable_laplacian():
    rng = np.random.default_rng(0)
    n = 7
    x = rng.random((n, n, n))
    vloc = rng.random((n, n, n))
    lap = rng.random((n, n))
    half_inv_h2 = 0.5

    t0 = np.tensordot(lap, x, axes=([1], [0]))
    t1 = np.moveaxis(np.tensordot(lap, x, axes=([1], [1])), 0, 1)
    t2 = np.moveaxis(np.tensordot(lap, x, axes=([1], [2])), 0, 2)
    expected = -half_inv_h2 * (t0 + t1 + t2) + vloc * x

    result = hpsi(x=x, vloc=vloc, half_inv_h2=half_inv_h2, lap=lap, N=n)
    assert np.allclose(result, expected)


if __name__ == '__main__':
    test_moveaxis_single()
    test_moveaxis_negative()
    test_moveaxis_multiple()
    test_moveaxis_identity()
    test_moveaxis_lowers_to_a_library_node()
    test_moveaxis_separable_laplacian()
