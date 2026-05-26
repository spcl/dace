# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Python frontend must give each memlet its own ``Subset`` object.

Repeated reads of the same array slice hit the access cache and shared one
``Range`` object (``Memlet.simple`` stores the subset by reference), so an
in-place subset rewrite on one edge corrupted the other.
"""
import numpy as np

import dace

N = dace.symbol('N')


@dace.program
def two_sibling_slice_reads(out: dace.float64[N, N], arr: dace.float64[N, N]):
    """Reads ``arr[i, k]`` in two sibling ``k`` loops; the second hits the cache."""
    for i in dace.map[0:N]:
        for k in range(N):
            out[i, k] = arr[i, k]
        for k in range(N):
            out[i, k] += arr[i, k]


def test_frontend_memlets_do_not_share_subset_objects():
    """No two memlets may share a subset (or other_subset) object."""
    sdfg = two_sibling_slice_reads.to_sdfg(simplify=False)
    seen = {}
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for e in state.edges():
                if e.data is None:
                    continue
                for sub in (e.data.subset, e.data.other_subset):
                    if sub is None:
                        continue
                    here = (state.label, e.data.data, str(sub))
                    assert id(sub) not in seen, \
                        f'subset object shared by two memlets: {seen[id(sub)]} and {here}'
                    seen[id(sub)] = here


def test_sibling_slice_reads_value_preserving():
    """End-to-end value check for the shape that triggered the aliasing."""
    n = 8
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, n)).astype(np.float64)
    out = np.zeros((n, n))
    two_sibling_slice_reads(out=out, arr=arr, N=n)
    assert np.allclose(out, 2.0 * arr)


if __name__ == '__main__':
    test_frontend_memlets_do_not_share_subset_objects()
    test_sibling_slice_reads_value_preserving()
